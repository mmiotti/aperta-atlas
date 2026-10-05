"""
Estimate per-edge AADT (annual average daily traffic) on the car network.

Pipeline: initial duration priors on edges → tiered OD pairs → shortest-
path costs → sample origins (pop-weighted from 03a's calibrated
`node_trip_weights_car`) + destinations (employment-weighted with
bin-adjusted cost weights from 03a's `flow_cost_bins_car`) → edge
betweenness → scale to AADT.

Simplifications: single deterministic shortest path per OD pair, no
capacity-based slowdown. Net effect: over-predicts highways / main
roads, under-predicts parallel local routes. Acceptable for a relative
"traffic pressure" feature; the fitted BPR multiplier absorbs the
scale downstream.

Inputs:
    coefs/calibrated/{node_trip_weights_car,flow_cost_bins_car}.csv  # from 03a
    properties/cells_*.csv + properties/nodes_car_extended.csv

Outputs (under <scenario>/, PUBLIC):
    properties/edges_car_flows.csv       # flow_estimate + vc + vc_beta_*
    properties/nodes_car_flows_avg.csv   # per-node r{250,500} neighborhood avgs
                                         # of traffic_flow + vc_beta_2.0
"""

import logging

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import KDTree

from aperta import network_processing, od_pairs, routing, traffic_flows
from aperta_atlas.context import init_context, Storage, _edge_id
from aperta_atlas.osm import OSM_HIGHWAY_RANKS
from aperta_atlas.utils import step

from scenarios import get_scenario


# Aggregation radii (meters) for the per-node traffic_flow feature.
# Same 250 / 500 pattern as 02b's bike_infra_score + speed_limit.
_RADII_NEIGHBORHOOD = (250, 500)

# Per-pair routing cutoff (seconds). Tier radii come from
# `scenario.mode_configs['car'].radii`.
_CAR_TIME_CUTOFF_S = 5400

# Initial edge-duration prior coefficients
_BASELINE_DURATION = 1.2
_INITIAL_MULT = {'density_r1000_norm': 0.15}
_INITIAL_ADD = {'is_4way': 15.0, 'is_traffic_signal': 15.0}

# Trip-generation weights + cost-bin edges are calibrated in 03a and
# loaded via `context.get_coefs(...)`.

# Sampling. TODO: make configurable / fraction of population count.
_N_ORIG = 50_000
_N_DEST = 200
_RNG_SEED = 42

# MTMC average. TODO: derive from data in 03a.
_TRIPS_PER_PERSON_PER_DAY = 1.8

# BPR-style congestion intensity per edge. `flow_estimate` is in vehicles
# /day (AADT), so capacities are expressed in veh/day/lane too — they're
# the textbook hourly capacities (2000 / 1000 / 600 veh/h/lane for
# highway / main / local) scaled by a ~15 h daily-equivalent factor that
# absorbs the peak-hour fraction of AADT. Lets `vc = flow / capacity`
# stay dimensionless without the calibration coefficient having to
# absorb the day-vs-hour mismatch downstream.
_CAP_PER_LANE_HIGHWAY =  30_000.0   # motorway, trunk
_CAP_PER_LANE_MAIN    =  15_000.0   # primary, secondary, tertiary
_CAP_PER_LANE_LOCAL   =  10_000.0   # residential, unclassified, ...

# Same tier thresholds as flows_vs_counters.py.
_TIER_HIGHWAY_MIN_RANK = 6
_TIER_MAIN_MIN_RANK    = 3

# BPR exponent. Textbook β=4 (US Highway Capacity Manual). β=2 was found
# to fit Swiss data better in the older extended-notebook calibration.
_VC_BETA = [2.0, 4.0]


def _cap_per_lane(highway) -> float:
    """Capacity per lane in veh/day/lane, by OSM highway-tier rank."""
    if isinstance(highway, list):
        highway = highway[0] if highway else None
    rank = OSM_HIGHWAY_RANKS.get(str(highway) if highway is not None else '', -1)
    if rank >= _TIER_HIGHWAY_MIN_RANK:
        return _CAP_PER_LANE_HIGHWAY
    if rank >= _TIER_MAIN_MIN_RANK:
        return _CAP_PER_LANE_MAIN
    return _CAP_PER_LANE_LOCAL

_KMH_TO_MS = 1.0 / 3.6



def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    radii = scenario.mode_configs['car'].radii

    # Car graph: skeleton + 'core' (OSM tags + decorated overlay) +
    # 'from_nodes' (per-edge density / is_4way / is_traffic_signal /
    # elevation, etc. — the prior-coefficient features below).
    graph = context.get_nw(
        data_name='car',
        add_node_properties='core',
        add_edge_properties=['core', 'from_nodes'],
    )

    cells = context.get_properties('cells' , ['population', 'employment', 'snap'], add_shapes=True)
    cells['node_id'] = cells['node_id_car']
    zones = context.get_properties('zones', ['population', 'snap'], add_shapes=True)
    zones['node_id'] = zones['node_id_car']
    n_before = len(cells)
    # Broad filter: include EVERY cell that can plausibly generate flow.
    # Cross-border + non-AOI cells matter — a German commuter driving to
    # Basel is a real trip on Swiss roads. Restricting to `is_active` or
    # `is_in_ch` here would systematically under-predict inflows. The
    # only requirements: snap to car, positive exposure, and zoned for
    # tier construction.
    cells = cells[
        cells['node_id'].notna()
        & (cells['combined_total'] > 0)
        & cells['zone_id'].notna()
        & cells['zone_id'].isin(zones.index)
    ].copy()
    logging.info(f"  → cells: {len(cells):,} of {n_before:,} kept "
                 f"(car-snapped + non-zero exposure + zoned)")

    with step('per-cell node weights from `node_trip_weights_car` (03a)'):
        # Coef rows: 'const' + one row per feature in `_NODE_TRIP_FEATURES`.
        # Features are all derived from raw cell columns + raw node
        # features pre-joined via the snap node — mirror of 03a's
        # `_RAW_NODE_FEATURES` + `_derive_features`. Keep the two in sync.
        # Apply Poisson predictor: `node_weight = exp(const + X · β)`.
        _RAW_NODE_FEATURES = ('density_r500_norm',)
        ntw = context.get_coefs('node_trip_weights_car')['car']
        feature_names = [n for n in ntw.index if n != 'const']
        node_props = context.get_properties('nodes', 'car_extended')
        cells = cells.join(node_props[list(_RAW_NODE_FEATURES)], on='node_id')
        pop = cells['population_total'].astype(float)
        emp = cells['employment_total'].astype(float)
        density = cells['density_r500_norm'].astype(float)
        cells['log1p_population'] = np.log1p(pop)
        cells['log1p_employment'] = np.log1p(emp)
        cells['log1p_pop_x_density'] = cells['log1p_population'] * density
        cells['log1p_emp_x_density'] = cells['log1p_employment'] * density
        missing = [f for f in feature_names if cells[f].isna().any()]
        if missing:
            raise ValueError(
                f"node_trip_weights_car references features {missing} that are "
                f"NaN on some cells. Either rebuild 02b with those features "
                f"present, or re-calibrate 03a with a feature set this "
                f"scenario covers.")
        linpred = ntw['const'] + cells[feature_names].astype(float) @ ntw[feature_names]
        cells['node_weight'] = np.exp(linpred)
        # Sum cell-level node_weights into the parent zone so c2z + z2z
        # destinations are weighted consistently. `lookup_dest_column_node`'s
        # conservation invariant requires additive cells → zones aggregation.
        zones['node_weight'] = (
            cells.groupby('zone_id')['node_weight'].sum()
                 .reindex(zones.index).fillna(0.0)
        )
        logging.info(
            f"  → node_weight: median={cells['node_weight'].median():.3f}, "
            f"P5={np.percentile(cells['node_weight'], 5):.3f}, "
            f"P95={np.percentile(cells['node_weight'], 95):.3f}")

    # ---------- Initial per-edge duration ---------------------------------
    with step('compute initial per-edge duration (priors)'):
        n_set = 0
        for u, v, k, d in graph.edges(keys=True, data=True):
            length = float(d['length'])
            speed_kph = float(d['speed_kph'])
            base = length / (speed_kph * _KMH_TO_MS) * _BASELINE_DURATION
            mult_term = base * sum(
                c * float(d.get(f, 0.0)) for f, c in _INITIAL_MULT.items()
            )
            add_term = sum(
                c * float(d.get(f, 0.0)) for f, c in _INITIAL_ADD.items()
            )
            d['duration_initial'] = max(base + mult_term + add_term, base * 0.2)
            n_set += 1
        logging.info(
            f"  → set duration_initial on {n_set:,} edges; sample stats: "
            f"min={min(d['duration_initial'] for *_, d in graph.edges(keys=True, data=True)):.1f} s, "
            f"max={max(d['duration_initial'] for *_, d in graph.edges(keys=True, data=True)):.0f} s")

    # Pre-sample origins BEFORE routing so `tiered_path_costs` can be
    # restricted (via `orig_cells=`) to just the origins we'll actually
    # use. ~6-8× fewer Dijkstra calls for typical N_orig / cell counts.
    with step(f'pre-sample {_N_ORIG:,} origins (with replacement, node_weight-weighted)'):
        rng = np.random.RandomState(_RNG_SEED)
        all_origin_node_ids = cells['node_id'].to_numpy()
        # Use the calibrated per-cell trip weight (`node_weight` from
        # `node_trip_weights_car` above) — NOT raw `combined_total`.
        orig_weights = cells['node_weight'].to_numpy(dtype=float)
        orig_weights = orig_weights / orig_weights.sum()
        chosen = rng.choice(all_origin_node_ids, _N_ORIG, replace=True, p=orig_weights)
        unique_origin_set = set(chosen.tolist())
        orig_mask = cells['node_id'].isin(unique_origin_set)
        logging.info(
            f"  → {len(unique_origin_set):,} unique origins among {_N_ORIG:,} picks "
            f"({100 * len(unique_origin_set) / _N_ORIG:.1f} %; rest are duplicates)")

    # ---------- Tiered OD pairs + routing costs (restricted to sampled) ---
    with step(
        f'build tiered OD pairs from sampled origins only '
        f'(r_cells={radii.r_cells:.0f}, r_medium={radii.r_medium:.0f}, '
        f'r_zones={radii.r_zones:.0f})'
    ):
        pairs = od_pairs.get_pairs(
            cells, r_cells=radii.r_cells, node_column='node_id',
            zones=zones, r_zones=radii.r_zones, r_medium=radii.r_medium,
            orig_cells=orig_mask,    # ← restrict to pre-sampled origins
        )
        logging.info(
            f"  → tiers: "
            f"c2c={len(pairs.cells_to_cells or {}):,}, "
            f"c2z={len(pairs.cells_to_zones or {}):,}, "
            f"z2z={len(pairs.zones_to_zones or {}):,}")

    with step(f'compute path costs (cutoff = {_CAR_TIME_CUTOFF_S} s)'):
        costs = routing.tiered_path_costs(graph, pairs,
            weight='duration_initial',
            cutoff=_CAR_TIME_CUTOFF_S,
        )
        # Note: we're currently ignoring snap distance. A snap distance overhead
        # could be added for more accuracy, but the penalty is generally small.

    with step('build sampling weights'):
        # Destinations weighted by the same calibrated trip weight as origins.
        # Symmetric: a cell that's a high-rate trip GENERATOR is also a
        # high-rate trip ATTRACTOR under the current single-rate fit.
        dest_weights = od_pairs.lookup_dest_column_node('node_weight', pairs, cells, 'node_id', zones=zones)
        cell_to_zone_node = od_pairs.build_cell_to_zone_node_map(cells, zones, node_column='node_id')

    with step('bin-adjusted dest weights (`flow_cost_bins_car` from 03a)'):
        # Per-bin reweighting so the sampled trip-cost distribution
        # matches the target P(C) baked into the coef.
        bin_edges = context.get_coefs('flow_cost_bins_car')['car'].to_numpy()
        logging.info(f"  → loaded {len(bin_edges)} bin edges (s)")
        adjusted_dest_weights = traffic_flows.bin_adjusted_dest_weights(
            pairs, costs, dest_weights, bin_edges,
        )

    # ---------- Sample destinations + accumulate flows --------------------
    with step(f'sample destinations per origin ({_N_DEST} per origin-pick, {_N_ORIG:,} picks total)'):
        # Pass the pre-sampled `chosen` array so nested_node_sample
        # skips its own origin draw. With bin-adjusted weights,
        # `cost_to_weight` is the identity.
        nested_sample = traffic_flows.nested_node_sample(
            pairs=pairs,
            weights=adjusted_dest_weights,
            costs=costs,
            cell_to_zone_node=cell_to_zone_node,
            orig_weights=None,            # using `chosen` instead
            cost_to_weight=np.ones_like,
            n_orig=_N_ORIG,
            n_dest=_N_DEST,
            random_state=rng,
            chosen=chosen,
        )
        logging.info(
            f"  → sampled to {len(nested_sample):,} unique origin nodes "
            f"(each with `n_picks × _N_DEST` destinations)")

    with step('accumulate edge betweenness from sampled paths'):
        edge_bc = network_processing.get_nested_edge_betweenness(
            graph, nested_sample,
            weight='duration_initial',
            cutoff=od_pairs.max_cost(costs),
        )

    with step('scale to AADT (vehicles/day)'):
        total_pop = float(cells['population_total'].sum())
        aadt_scale = (total_pop * _TRIPS_PER_PERSON_PER_DAY) / (_N_ORIG * _N_DEST)
        flows = edge_bc * aadt_scale
        logging.info(
            f"  → fraction of expected real trip count captured in sample: {1/aadt_scale*100:.1f}%)")
        logging.info(
            f"  → flow_estimate (veh/day): mean {flows.mean():.0f}, max {flows.max():.0f}")

    # BPR-style congestion intensity: `vc = flow / capacity` (both
    # veh/day), then `vc_beta = vc**β`. 04_edge_weights uses `vc_beta`
    # as a multiplier feature in car profiles.
    with step(f'compute BPR vc + vc_beta (β={_VC_BETA})'):
        flows_dict = flows.to_dict()
        caps_per_edge = {}
        for u, v, k, d in graph.edges(keys=True, data=True):
            lanes = d.get('lanes_per_direction', 1) or 1
            caps_per_edge[(u, v, k)] = _cap_per_lane(d.get('highway')) * max(1, lanes)
        logging.info(
            f"  → capacity (veh/day/lane × lanes): "
            f"median {np.median(list(caps_per_edge.values())):.0f}")

    # ---------- Save ------------------------------------------------------
    with step('save edges_car_flows.csv'):
        # Reindex to ALL edges (untraversed → 0) so downstream consumers
        # can rely on a 1:1 mapping with the car-network edges.
        rows = {}
        for keys in graph.edges(keys=True):
            flow = float(flows_dict.get(keys, 0.0))
            cap = float(caps_per_edge[keys])
            vc = flow / cap if cap > 0 else 0.0
            rows[_edge_id(*keys)] = {
                'flow_estimate': flow,
                'capacity':      cap,
                'vc':            vc,
            }
            for beta in _VC_BETA:
                rows[_edge_id(*keys)][f'vc_beta_{beta:.1f}'] = vc ** beta
        flows_df = pd.DataFrame.from_dict(rows, orient='index').round(4)
        flows_df.index.name = 'edge_id'
        logging.info(
            f"  → vc: median {flows_df['vc'].median():.3f}, "
            f"P95 {flows_df['vc'].quantile(0.95):.3f}, "
            f"max {flows_df['vc'].max():.3f}")
        context.create_properties(flows_df, data_name='car_flows')

    # Per-node neighborhood aggregations: mean of car-edge quantities
    # over edges whose midpoint is within r of the node. Consumed by
    # 09a as endpoint features (`traffic_flow_avg_r*` + `vc_beta_*_avg_r*`).
    _NEIGHBORHOOD_FIELDS: list[tuple[str, str]] = [
        ('flow_estimate', 'traffic_flow_avg'),
        ('vc_beta_2.0',   'vc_beta_2.0_avg'),
    ]
    with step(f'per-node car-edge neighborhood avgs (r{_RADII_NEIGHBORHOOD})'):
        car_nodes, car_edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        mid = car_edges.geometry.interpolate(0.5, normalized=True)
        edge_xy = np.column_stack(
            [mid.x.to_numpy(), mid.y.to_numpy()])
        edge_ids = pd.Index(
            [_edge_id(u, v, k) for u, v, k in car_edges.index],
            name='edge_id')
        node_xy = np.column_stack([
            car_nodes.geometry.x.to_numpy(),
            car_nodes.geometry.y.to_numpy(),
        ])
        tree = KDTree(edge_xy)
        # Query once per radius, apply to every field — cheaper than
        # rebuilding the tree / re-querying per field.
        idx_lists_per_r = {
            r: tree.query_ball_point(node_xy, r=r) for r in _RADII_NEIGHBORHOOD
        }
        out = pd.DataFrame(index=car_nodes.index)
        for src_col, out_prefix in _NEIGHBORHOOD_FIELDS:
            values = flows_df.loc[edge_ids, src_col].to_numpy(dtype=float)
            for r, idx_lists in idx_lists_per_r.items():
                avg = np.zeros(len(car_nodes), dtype=float)
                for k, idxs in enumerate(idx_lists):
                    if idxs:
                        avg[k] = float(values[idxs].mean())
                out[f'{out_prefix}_r{r}'] = avg
        out.index = out.index.rename('node_id')
        context.create_properties(
            out, data_name='car_flows_avg', float_format='%.3g')
        logging.info(
            f"  → traffic_flow_avg_r250 mean={out['traffic_flow_avg_r250'].mean():.0f} veh/day; "
            f"vc_beta_2.0_avg_r250 mean={out['vc_beta_2.0_avg_r250'].mean():.3f}")

    context.close()


if __name__ == '__main__':
    main()
