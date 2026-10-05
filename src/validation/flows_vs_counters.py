"""
Validate atlas's per-edge AADT flow estimates (from 03b_traffic_flows.py)
against Swiss ASTRA traffic counters.

Loads the saved `edges_car_flows.csv` + the car graph from atlas,
re-tags each edge with a tier bucket (`highway` / `main` / `local`)
derived from `OSM_HIGHWAY_RANKS`, snaps each counter to the right
edge using `calibration.snap_counters_to_edges` (with per-counter
search radius + bearing tolerance + tier-matched eligibility), then
runs `calibration.evaluate_against_counters` for the overall set and
once per tier — yielding R² + slope + RMSE + n_matched per group.

No plots. For visual diagnostics (scatter plots, calibrated-speed maps,
inner-core polygon overlays, etc.), see the extended showcase notebook
`aperta/examples/extended/traffic_flows.py`. The numbers logged here
are the same numbers the notebook puts in plot titles.

Inputs:
    <scenario>/  (PUBLIC, via context)
        nw/car.graphml + properties/{nodes,edges}_car_core.csv
        properties/edges_car_flows.csv      # from 03b_traffic_flows.py
    PRIVATE/preparation/switzerland/traffic_counters/traffic_counters.gpkg
        # directional point counters with `traffic_cars`, `bearing_deg`,
        # `is_highway` / `is_main` / `is_local` flags.

Outputs:
    Logs only — overall + per-tier R²/slope/RMSE/n.
"""

import logging

import geopandas as gpd
import pandas as pd

from aperta import calibration
from aperta.network_processing import parse_edge_id
from aperta_atlas.context import Storage, init_context
from aperta_atlas.osm import OSM_HIGHWAY_RANKS
from aperta_atlas.utils import step

from preparation.world.areas import AREAS
from preparation.world.common import buffered_place_polygon
from scenarios import get_scenario


# Counter-to-edge snap. Highway counters get a wider radius (sparser
# layout, lower risk of catching a parallel local road); non-highway
# counters get a tight radius. Bearing tolerance prevents
# opposite-direction counters from cross-snapping on two-way roads.
_HIGHWAY_RADIUS_M = 25.0
_NON_HIGHWAY_RADIUS_M = 10.0
_BEARING_TOL_DEG = 10.0

# Tier-classification cutoffs over `OSM_HIGHWAY_RANKS`. Matches the
# extended notebook's bucketing exactly.
_TIER_HIGHWAY_MIN_RANK = 6   # motorway / trunk
_TIER_MAIN_MIN_RANK = 3      # primary / secondary / tertiary
# Anything below tertiary → 'local' (residential / service / unknown).

_COUNTERS_NAMESPACE = 'preparation/switzerland/traffic_counters'
_COUNTERS_FILE = 'traffic_counters.gpkg'

_AOI_BUFFER = -20_000


def _edge_tier(d: dict) -> str:
    """Map an edge attribute dict to a tier bucket via OSM highway rank."""
    hwy = d.get('highway')
    if isinstance(hwy, list):
        hwy = hwy[0] if hwy else None
    rank = OSM_HIGHWAY_RANKS.get(hwy, -1)
    if rank >= _TIER_HIGHWAY_MIN_RANK:
        return 'highway'
    if rank >= _TIER_MAIN_MIN_RANK:
        return 'main'
    return 'local'


def _log_fit(label: str, ev: dict) -> None:
    if ev['n_matched'] == 0:
        logging.info(f"  → {label:8s}: n=0 — no counters in this group")
        return
    logging.info(
        f"  → {label:8s}: R²={ev['r2']:.3f}, slope={ev['slope']:.3f}, "
        f"RMSE={ev['rmse']:,.0f} veh/day, n={ev['n_matched']:,}")


def main():
    context = init_context()
    assert context.scenario is not None, \
        ('atlas requires a scenario (set --scenario or '
         "DEFAULT_SCENARIO in src/scenarios.py)")
    scenario = get_scenario(context.scenario)
    crs_main = scenario.crs_main

    # ---------- Load graph + flow estimates ------------------------------
    with step('load car graph (+ core overlay for highway tags)'):
        graph = context.get_nw(
            data_name='car',
            add_node_properties='core',
            add_edge_properties=['core', 'from_nodes'],
        )
        logging.info(
            f"  → {graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges")

    with step('tag every edge with tier (highway / main / local)'):
        n_per_tier = {'highway': 0, 'main': 0, 'local': 0}
        for _, _, _, d in graph.edges(keys=True, data=True):
            d['_tier'] = _edge_tier(d)
            n_per_tier[d['_tier']] += 1
        logging.info(f"  → tier distribution: {n_per_tier}")

    with step('load edge flow estimates (from 03b_traffic_flows.py)'):
        flows_df = context.get_properties('edges', 'car_flows')
        # Flow CSV is indexed by `edge_id` (string `'u:v:k'`); convert to
        # tuple index so `evaluate_against_counters` can do its (u,v,k)
        # lookup against the graph's edges.
        flows = pd.Series(
            flows_df['flow_estimate'].to_numpy(),
            index=pd.MultiIndex.from_tuples(
                [parse_edge_id(s) for s in flows_df.index],
                names=['u', 'v', 'k'],
            ),
            name='flow_estimate',
        )
        logging.info(
            f"  → {len(flows):,} edges with flow; "
            f"median {flows.median():,.0f}, P95 {flows.quantile(0.95):,.0f} veh/day")

    # ---------- Load + snap counters --------------------------------------
    with step('load counters (PRIVATE Swiss ASTRA dataset)'):
        ctr_ctx = context.source(_COUNTERS_NAMESPACE, storage=Storage.PRIVATE)
        counters_path = ctr_ctx.path_for(ctr_ctx.default_storage, _COUNTERS_FILE)
        counters = gpd.read_file(counters_path).to_crs(crs_main)
        logging.info(
            f"  → {len(counters):,} counters loaded "
            f"(highway: {int(counters['is_highway'].sum()):,}, "
            f"main: {int(counters['is_main'].sum()):,}, "
            f"local: {int(counters['is_local'].sum()):,})")

    with step('filter counters → area polygon (inner core, no buffer)'):
        # The area polygon (e.g. Canton of Bern boundary) — counters
        # outside it see traffic going to/from places the simulation
        # can't model (because they're outside the cell+zone layer),
        # which inflates observed AADT relative to modeled. Filter to
        # in-area-polygon counters for a fair comparison.
        area = AREAS[scenario.area_name]
        # `union_all()` on a 1-row GeoSeries returns that single geometry;
        # cleaner type-wise than `.geometry.iloc[0]` (whose pandas-side
        # return type isn't narrowed to BaseGeometry by geopandas stubs).
        core_polygon = (
            buffered_place_polygon(area.place, _AOI_BUFFER)
            .to_crs(crs_main).geometry.union_all())
        n_before = len(counters)
        in_core = counters.geometry.within(core_polygon)
        counters = counters[in_core].copy()
        logging.info(
            f"  → kept {len(counters):,} of {n_before:,} counters "
            f"({100 * len(counters) / n_before:.1f} %) inside the "
            f"{area.place!r} polygon")

    with step(
        f'snap counters → edges (bearing_tol={_BEARING_TOL_DEG}°, '
        f'radius {_HIGHWAY_RADIUS_M:.0f}/{_NON_HIGHWAY_RADIUS_M:.0f} m for '
        f'highway/other)'
    ):
        # Per-counter radius: wider for highway counters.
        max_distance = counters['is_highway'].map(
            {1: _HIGHWAY_RADIUS_M, 0: _NON_HIGHWAY_RADIUS_M})

        # Per-counter eligibility: highway counters only snap to highway
        # edges, main to main, local to local. Prevents catching the wrong
        # road class on parallel infrastructure (e.g. a highway counter
        # snapping to a frontage road).
        def _eligible_for_counter(counter_row, candidate_edges):
            if counter_row['is_highway']:
                wanted = 'highway'
            elif counter_row['is_main']:
                wanted = 'main'
            else:
                wanted = 'local'
            return candidate_edges[candidate_edges['_tier'] == wanted]

        snapped = calibration.snap_counters_to_edges(
            counters, graph,
            max_distance=max_distance,
            bearing_tol_deg=_BEARING_TOL_DEG,
            eligible_edges=_eligible_for_counter,
        )
        counters = counters.join(snapped)
        n_matched = int(counters['u'].notna().sum())
        logging.info(
            f"  → snapped {n_matched:,} of {len(counters):,} counters "
            f"({100 * n_matched / len(counters):.1f} %)")
        for label, col in [('highway', 'is_highway'), ('main', 'is_main'),
                           ('local', 'is_local')]:
            mask = counters[col] == 1
            n_tot = int(mask.sum())
            n_ma = int((mask & counters['u'].notna()).sum())
            if n_tot:
                logging.info(
                    f"     {label:7s}: {n_ma:,} of {n_tot:,} matched "
                    f"({100 * n_ma / n_tot:.1f} %)")

    # ---------- Evaluate: overall + per-tier ------------------------------
    with step('evaluate modeled vs observed AADT (overall + per-tier)'):
        ev_all = calibration.evaluate_against_counters(flows, counters)
        _log_fit('all', ev_all)

        for label, col in [('highway', 'is_highway'), ('main', 'is_main'),
                           ('local', 'is_local')]:
            subset = counters[counters[col] == 1]
            if subset.empty:
                _log_fit(label, {'r2': float('nan'), 'slope': float('nan'),
                                 'rmse': float('nan'), 'n_matched': 0})
                continue
            ev = calibration.evaluate_against_counters(flows, subset)
            _log_fit(label, ev)

    context.close()


if __name__ == '__main__':
    main()
