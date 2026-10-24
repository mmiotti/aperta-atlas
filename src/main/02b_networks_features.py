"""
Network FEATURES for the Swiss Urban Mobility Atlas: per-node + per-edge.

Stage 2 of the two-stage network pipeline. Reads the static topology
written by `02a_networks_snap.py` (graph + shapes + snap CSVs) and
computes/saves all the per-node and per-edge features. Cheap to iterate:
edit a feature definition or add a new column and re-run this script
without paying the snap cost.

Per mode (walk, bike, car):
  - Load saved graph (already in scenario.crs_main, with cell virtuals + decorated
    attrs preserved). Extract nodes/edges GDFs.
  - Per-edge: bike_infra_score (0 / 0.5 / 1 / 1.5, `edge_bike_infra_score`),
    elevation delta / gain / loss, slope (+ uphill / downhill), FIETS
    uphill index, is_bridge_or_tunnel. Elevation features are scaled
    down on bridge / tunnel edges.
  - Per-node: elevation (DEM raster, smoothed over the graph), per-node
    pop+emp (via cell→node snap from 02a), density at `_RADII_DENSITY`
    (+ sqrt-normalised `_norm`), mean |slope| ("hilliness") at
    `_RADII_NEIGHBORHOOD`, decorated intersection flags (passed through
    for edge aggregation).
  - Per-mode neighborhood aggregations at `_RADII_NEIGHBORHOOD`:
      * bike_infra_score_avg — on every mode's own edges.
      * speed_limit_avg — on every mode's own edges, FILTERED to
        car-shared road classes (walk / bike graphs also include
        pedestrian-only ways whose OSMnx-filled `speed_kph` is a
        meaningless fallback; excluded via `_car_road_mask`).
    `main/utility_config.py` picks the source graph per feature.
  - Map node features onto edges (mean of endpoints).

The bike-infra-score callback is module-level (`edge_bike_infra_score`)
so it can be swapped without touching the rest of the pipeline.

Inputs:
    nw/<mode>.graphml + properties/{nodes,edges}_<mode>_core.csv   # from 02a
    properties/cells_snap_<mode>.csv                              # from 02a
    properties/cells_population.csv                               # from 01 (combined_total)
    preparation/world/elevation/dem_<area>.tif

Outputs (scenario storage, under <scenario>/):
    properties/nodes_<mode>_extended.csv    # per-node features
    properties/edges_<mode>_extended.csv    # bike_infra_score, elevation_*, slope_*, …
    properties/edges_<mode>_from_nodes.csv  # node-derived edge features

Run all variants (walk / bike / car) sequentially (default):
    python -m main.02b_networks_features --scenario <name>
Single mode:
    python -m main.02b_networks_features --scenario <name> --variant car
"""

import logging
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from scipy.spatial import KDTree

from aperta import geo_processing, network_processing
from aperta_atlas.context import init_context, Storage, _edge_id
from aperta_atlas.variant import Variants
from aperta_atlas.utils import step

from scenarios import get_scenario


# Modes to process. Mirror 02a's set.
variants = Variants([('mode', str)])
for _m in ('walk', 'bike', 'car'):
    variants.add(name=_m, mode=_m)

# Density aggregation radii (meters). Density is used as a predictor/proxy in
# several contexts.
_RADII_DENSITY = (250, 500, 1000)
# Neighborhood-metric radii (meters) for the utility-model endpoint
# features (bike_infra_score, speed_limit, mean_abs_slope).
_RADII_NEIGHBORHOOD = (100, 250)

# Gaussian 1-σ (metres) for smoothing node elevations over the graph.
_SMOOTH_ELEV_SCALE = 25.0


# Cycleway values that mean "no bike infrastructure on THIS edge's carriageway". `separate` and
# `use_sidepath` say the bike way is mapped as its own way (scored there, not on the road).
# They are skipped like `no`, so a real lane tagged on the other side is still found.
_NO_INFRA_ON_EDGE = ('no', 'none', 'separate', 'use_sidepath')


def _parse_cycleway(value_list) -> str:
    for value in value_list:
        if isinstance(value, list):
            value = value[0] if value else None
        if (value and value not in _NO_INFRA_ON_EDGE
                and (isinstance(value, str) or ~np.isnan(value))):
            return str(value)
    return ''


# Track cycleway values we've warned about (once each) so the log
# doesn't repeat the same message per edge across an entire run.
_UNKNOWN_CYCLEWAY_WARNED: set[str] = set()


def edge_bike_infra_score(u, v, data) -> float:
    """Per-edge bike-infrastructure score — pure infrastructure signal.

    Scale {0, 0.5, 1, 1.5}:
      - 1.5: dedicated separated bike way (`highway=cycleway`,
        `bicycle=designated`, `cycleway=track|segregated|…`). Capped
        below the natural 2.0 to keep radius-averages inside the
        linear utility fit's data-rich range.
      - 1.0: proper on-road bike lane (`cycleway=lane|lanes|…`,
        `cyclestreet=yes`).
      - 0.5: shared markings (`cycleway=shared_lane|pictogram|…`).
      - 0.0: no infra (or footway/pedestrian/path — informal only). A road tagged
        `cycleway*=separate|use_sidepath` scores 0 here: its bike way is a separate edge.

    Inputs come from post-consolidation edge attrs. PathAggregation-
    compatible signature.
    """
    h = data.get('highway')
    if isinstance(h, list):
        h = h[0] if h else None
    if h == 'cycleway' or data.get('bicycle') == 'designated':
        return 1.5

    cw = [data.get('cycleway'), data.get('cycleway:both'),
          data.get('cycleway:left'), data.get('cycleway:right')]
    cw = _parse_cycleway(cw)

    if cw in ('track', 'designated', 'segregated'):
        return 1.5
    if cw in ('lane', 'lanes', 'share_sidewalk', 'share_busway', 'right',
              'yes', 'y', 'opposite_track', '|lane|', 'line', 'crossing',
              'sidepath', 'mtb', 'sidewalk', 'both'):
        return 1.0
    if cw in ('shared_lane', 'shared', 'soft_lane', 'shoulder',
              'opposite', 'opposite_lane', 'opposite_share_busway',
              'pictogram', 'opposite_bike_lane', 'link', 'road',
              'left'):
        return 0.5
    # Cyclestreets (roads designated as priority for bikes) — 1.
    if data.get('cyclestreet') in ('yes', True, 'true', '1', 1):
        return 1.0
    # Rare unclear tags — treat as no infra to avoid false positives.
    if cw in ('n', 'p'):
        return 0.0
    if cw and cw not in _UNKNOWN_CYCLEWAY_WARNED:
        # Warn once per novel tag, fall through to 0.0 — a strict raise
        # would block wide-scenario runs on the first long-tail tag.
        _UNKNOWN_CYCLEWAY_WARNED.add(cw)
        logging.warning(
            f"  ⚠ cycleway={cw!r}: not in any bike_infra_score bucket; "
            f"treating as no infra (0.0). Add to the mapping in "
            f"`edge_bike_infra_score` if it denotes real infra.")
    return 0.0


def _per_node_edge_attr_avg(
    node_index: pd.Index,
    node_xy: np.ndarray,
    edge_xy: np.ndarray,
    values: np.ndarray,
    radii: tuple[int, ...],
    col_prefix: str,
) -> pd.DataFrame:
    """For each node, average `values[i]` over edges `i` whose midpoint is
    within each radius. KDTree query on the edge midpoint cloud; safe to
    call cross-graph (e.g., walk-graph nodes against bike-graph edges).

    Returns a DataFrame indexed by `node_index`, with one column per
    radius: `{col_prefix}_r{R}`. Nodes with no edges in range get 0.0.
    """
    tree = KDTree(edge_xy)
    values = np.asarray(values, dtype=float)
    out = pd.DataFrame(index=node_index)
    for r in radii:
        idx_lists = tree.query_ball_point(node_xy, r=r)
        avg = np.zeros(len(node_index), dtype=float)
        for k, idxs in enumerate(idx_lists):
            if idxs:
                avg[k] = float(values[idxs].mean())
        out[f'{col_prefix}_r{r}'] = avg
    return out


def _edge_midpoints(edges: gpd.GeoDataFrame) -> np.ndarray:
    mid = edges.geometry.interpolate(0.5, normalized=True)
    return np.column_stack([mid.x.to_numpy(), mid.y.to_numpy()])


# Highway classes where `speed_kph` reflects a real car speed limit.
# Excludes pedestrian-only classes (footway / path / pedestrian /
# cycleway / …) whose `speed_kph` is an OSMnx fallback with no
# meaning for a "local speed-limit" aggregation.
_CAR_ROAD_HIGHWAYS: frozenset[str] = frozenset({
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'unclassified',
    'residential', 'living_street', 'service',
})


def _car_road_mask(edges: gpd.GeoDataFrame) -> np.ndarray:
    """Boolean mask over `edges` where the highway class is a car-shared
    road (i.e., `speed_kph` reflects an actual road speed limit rather
    than an OSMnx per-class fallback filled onto pedestrian-only ways)."""
    def _first(v):
        if isinstance(v, list):
            return v[0] if v else None
        return v
    hw = edges['highway'].map(_first)
    return hw.isin(_CAR_ROAD_HIGHWAYS).to_numpy()


def _delta_elevation_per_edge(
    edges: gpd.GeoDataFrame,
    node_elevation: pd.Series,
) -> pd.Series:
    """Per-edge elevation change = elevation(v) − elevation(u).

    Edges come as MultiIndex (u, v, key). Look up endpoint elevations
    from the node series. Sign convention: positive = climb.
    """
    u = edges.index.get_level_values('u')
    v = edges.index.get_level_values('v')
    elev_u = node_elevation.reindex(u).to_numpy()
    elev_v = node_elevation.reindex(v).to_numpy()
    delta = elev_v - elev_u
    # Some (should be very few) nodes can get NaN assigned - fill with 0 elevation gain/loss
    delta[np.isnan(delta)] = 0
    return pd.Series(delta, index=edges.index, name='delta_elevation')


def main(variant) -> None:
    context = init_context(variant)
    mode = variant.mode
    scenario = get_scenario(context.scenario)
    if mode not in scenario.mode_configs:
        logging.info(f"mode={mode!r} not in scenario mode_configs "
                     f"({list(scenario.mode_configs)}); skipping")
        context.close()
        return

    # ---------- Load cells (with pop+emp totals) -------------------------
    cells = context.get_properties('cells', 'population')
    logging.info(f"  → {len(cells):,} cells loaded")

    # ---------- DEM raster path (per-mode-shared) ------------------------
    elev_ctx = context.source('preparation/world/elevation')
    dem_path = elev_ctx.path_for(Storage.PUBLIC, f'dem_{scenario.area_name}.tif')
    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM raster not found at {dem_path}. Run preparation/world/elevation/.")
    # `sample_raster_at_points` doesn't reproject — read DEM CRS here,
    # reproject nodes per-mode below just for sampling.
    with rasterio.open(dem_path) as _src:
        _dem_crs = _src.crs

    # Per-mode snap from 02a (per-mode CSV, not the merged one — 02c
    # writes that later, after all 02a/02b invocations complete).
    cell_snap = context.get_properties('cells', f'snap_{mode}')

    graph = context.get_nw(
        data_name=mode,
        add_node_properties='core',
        add_edge_properties='core',
    )

    with step(f'mode={mode}: extract nodes + edges GDFs'):
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        # Decorated intersection-type flags
        is_flag_cols = sorted(c for c in nodes.columns 
                              if c.startswith('is_') and c != 'is_virtual')
        if is_flag_cols:
            nodes[is_flag_cols] = nodes[is_flag_cols].fillna(0).astype(float)
        logging.info(
            f"  → {len(nodes):,} nodes, {len(edges):,} edges; "
            f"intersection flags: {len(is_flag_cols)}")

    # ---------- Per-edge extended features ------------------------------
    with step(f'mode={mode}: compute bike_infra_score per edge (0/0.5/1/1.5)'):
        bike_infra_score = pd.Series(
            [edge_bike_infra_score(u, v, data) for u, v, data in graph.edges(data=True)],
            index=edges.index, name='bike_infra_score',
        )
        # Fraction breakdown for quick sanity — most edges should be 0
        # (no bike infra); increasing tiers get rarer.
        counts = bike_infra_score.value_counts(normalize=True).sort_index()
        logging.info(
            "  → bike_infra_score: "
            + ', '.join(f'{v:g}={p*100:.1f}%' for v, p in counts.items()))

    # ---------- Per-node features -------------------------------------
    with step(f'mode={mode}: sample elevation per node'):
        # Sample on a temporary copy reprojected to the DEM's CRS; the
        # returned Series keeps nodes.index, so the assignment aligns.
        nodes['elevation'] = geo_processing.sample_raster_at_points(
            nodes.to_crs(_dem_crs), dem_path, name='elevation')

    with step(
        f'mode={mode}: topology-smooth elevation '
        f'(Gaussian σ={_SMOOTH_ELEV_SCALE:.0f} m, 1 iteration)'
    ):
        # Round-trip: write the raw sampled elevation to the graph, smooth
        # there (Gaussian weight over edge `length`), read back into `nodes`.
        nx.set_node_attributes(graph, nodes['elevation'].to_dict(), 'elevation')
        network_processing.smooth_node_attribute(
            graph, 'elevation', length_scale=_SMOOTH_ELEV_SCALE)
        nodes['elevation'] = pd.Series(
            nx.get_node_attributes(graph, 'elevation'),
            name='elevation',
        ).reindex(nodes.index)
        logging.info(
            f"  → elevation: mean={nodes['elevation'].mean():.0f} m, "
            f"min={nodes['elevation'].min():.0f}, "
            f"max={nodes['elevation'].max():.0f}")

    with step(f'mode={mode}: aggregate cell pop+emp onto nodes (via 02a snap)'):
        # Cell → node mapping for this mode comes from 02a's snap.
        cell_to_node = pd.DataFrame({
            'node_id': cell_snap['node_id'].values,
            'combined_total': cells['combined_total'].reindex(cell_snap.index).values,
        }).dropna(subset=['node_id'])
        lu = (cell_to_node.groupby('node_id').sum().reindex(nodes.index, fill_value=0.0))
        nodes = nodes.join(lu)

    with step(f'mode={mode}: density per node, radii {_RADII_DENSITY} m (circular only)'):
        # `add_filled_densities=True` doesn't work on points (nodes have
        # zero area → divide-by-zero). A "filled" density would have to be
        # computed per cell (cells have area) and mapped via the snap.
        density_frames = []
        for r in _RADII_DENSITY:
            density_frames.append(
                geo_processing.sum_within_radius(
                    nodes, ['combined_total'], r,
                    return_densities=True,
                )
            )
        density_all = pd.concat(density_frames, axis=1).rename(columns={
            f'combined_total_r{r}': f'density_r{r}' for r in _RADII_DENSITY
        })
        for col in density_all.columns:
            density_all[f'{col}_norm'] = np.sqrt(density_all[col] / 10_000)
        nodes = nodes.join(density_all)

    # Per-mode neighborhood aggregations.
    #
    # - speed_limit_avg is aggregated over CAR-SHARED road edges only
    #   (`_CAR_ROAD_HIGHWAYS`). On walk / bike graphs, OSMnx's
    #   `add_edge_speeds` (networks_decorate.py) fills a fallback 50 km/h
    #   onto pedestrian-only ways (footway / path / cycleway / steps …).
    # - bike_infra_score is aggregated on every mode's own edges, so each
    #   mode sees its own edge set (walk graph includes cycle paths, car
    #   graph the on-road bike lanes, etc.).
    cross_cols: list[str] = []
    with step(f'mode={mode}: bike_infra + speed_limit neighborhood avg, '
              f'radii {_RADII_NEIGHBORHOOD} m'):
        edge_xy = _edge_midpoints(edges)
        node_xy = np.column_stack(
            [nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()])

        # bike_infra_score aggregation — over ALL of this mode's edges.
        bike_agg = _per_node_edge_attr_avg(
            nodes.index, node_xy, edge_xy,
            bike_infra_score.to_numpy(dtype=float),
            _RADII_NEIGHBORHOOD, 'bike_infra_score_avg')

        # speed_limit aggregation — over edges on car-shared road classes only.
        road_mask = _car_road_mask(edges)
        n_road = int(road_mask.sum())
        logging.info(
            f"  → speed_limit source: {n_road:,}/{len(edges):,} edges "
            f"({100*n_road/max(len(edges),1):.1f} %) on car-shared road classes")
        if n_road:
            speed_agg = _per_node_edge_attr_avg(
                nodes.index, node_xy,
                edge_xy[road_mask],
                edges['speed_kph'].astype(float).to_numpy()[road_mask],
                _RADII_NEIGHBORHOOD, 'speed_limit_avg')
        else:
            # No road edges in this mode's graph — write zero columns
            # so downstream schema stays stable.
            speed_agg = pd.DataFrame(
                {f'speed_limit_avg_r{r}': 0.0 for r in _RADII_NEIGHBORHOOD},
                index=nodes.index)

        nodes = nodes.join(pd.concat([bike_agg, speed_agg], axis=1))
        cross_cols = list(bike_agg.columns) + list(speed_agg.columns)

    with step(f'mode={mode}: mean |slope| per node ("hilliness", network-weighted), '
              f'radii {_RADII_NEIGHBORHOOD} m'):
        # Length-weighted mean |slope| within radius = Σ|Δh| / Σlength over all
        # edges whose centroid is within `r` of the node: the average grade
        # you'd actually traverse in this area. Computed BEFORE the
        # bridge / tunnel mask below is applied.
        edges_for_slope = edges[['geometry', 'length']].copy()
        delta_elev = _delta_elevation_per_edge(edges, nodes['elevation'])
        edges_for_slope['abs_delta_elevation'] = delta_elev.abs()
        for r in _RADII_NEIGHBORHOOD:
            sum_abs_dh = geo_processing.cross_sum_within_radius(
                nodes, edges_for_slope, r,
                weight_column='abs_delta_elevation',
                name=f'sum_abs_dh_r{r}',
            )
            # Nodes with no in-radius edges → 0 (mean undefined)
            sum_len = geo_processing.cross_sum_within_radius(
                nodes, edges_for_slope, r,
                weight_column='length',
                name=f'sum_length_r{r}',
            ).replace(0, np.nan)
            nodes[f'mean_abs_slope_r{r}'] = (sum_abs_dh / sum_len).fillna(0.0)

    # ---------- Edge: delta_elevation + slope -------------------------
    with step(f'mode={mode}: edge delta_elevation + slope'):
        length = edges['length'].astype(float)
        slope = (delta_elev / length.replace(0, np.nan)).fillna(0.0)
        slope.name = 'slope'
        # FIETS index, uphill only (delta_h ** 2 / dist; equal to slope * delta_h)
        fiets = (np.maximum(0, delta_elev)**2 / length.replace(0, np.nan)).fillna(0.0)

        # Bridge / tunnel mask: scale elevation change / slope down by the
        # edge's bridge + tunnel share. Underestimates elevation change where
        # the bridge or tunnel isn't flat, but the alternative can badly
        # overestimate gain/loss, especially with nodes on the bridge or in
        # the tunnel.
        bridge_frac = edges['bridge'].astype(float).fillna(0.0)
        tunnel_frac = edges['tunnel'].astype(float).fillna(0.0)
        elev_mask = (1.0 - bridge_frac - tunnel_frac).clip(lower=0.0)
        n_masked = int((elev_mask < 1.0).sum())
        if n_masked:
            logging.info(
                f"  → bridge/tunnel mask: scaling elevation features "
                f"on {n_masked:,} edges (mean mask "
                f"{elev_mask[elev_mask < 1.0].mean():.2f})")
        delta_elev = delta_elev * elev_mask
        slope = slope * elev_mask
        fiets = fiets * elev_mask

    # ---------- Map node features → edges -----------------------------
    with step(f'mode={mode}: map node features onto edges (mean of endpoints)'):
        # For intersection flags (is_traffic_signal, etc.), the mean
        # aggregator IS the half-allocation: edge value = (val_u + val_v) / 2.
        # → one signal endpoint  → 0.5
        # → two signal endpoints → 1.0
        # Summing along a path gives 1.0 per node traversal (entry +
        # exit edges each contribute 0.5).
        node_feature_cols = [
            'elevation', 'combined_total',
            *[f'density_r{r}' for r in _RADII_DENSITY],
            *[f'density_r{r}_norm' for r in _RADII_DENSITY],
            *[f'mean_abs_slope_r{r}' for r in _RADII_NEIGHBORHOOD],
            *cross_cols,  # bike_infra_score_avg_r* + speed_limit_avg_r* (every mode)
        ]
        edges_from_nodes = network_processing.aggregate_nodes_to_edges(
            nodes, node_feature_cols + is_flag_cols, graph, aggregator='mean',
        ).reindex(edges.index, fill_value=0.0)

    # ---------- Assemble + save ---------------------------------------
    with step(f'mode={mode}: save nodes + edges (extended + from-nodes)'):
        # `create_properties` requires canonical index names: 'node_id'
        # for nodes, 'edge_id' for edges. osmnx writes index name
        # 'osmid' and a MultiIndex (u, v, key) — convert before saving.
        node_props = nodes[node_feature_cols].copy()
        node_props.index = node_props.index.rename('node_id')

        edge_id_index = pd.Index(
            [_edge_id(u, v, k) for u, v, k in edges.index],
            name='edge_id')
        # Bridge + tunnel share in [0, 1], one column for downstream use.
        # Clipped: a stacked bridge + tunnel (rare) would otherwise exceed 1.
        is_bridge_or_tunnel = (bridge_frac + tunnel_frac).clip(upper=1.0)

        extended_edges = pd.DataFrame({
            'bike_infra_score': bike_infra_score.to_numpy(),
            'elevation_delta': delta_elev.to_numpy(),
            'elevation_gain': np.maximum(delta_elev.to_numpy(), 0.0),
            'elevation_loss': np.abs(np.minimum(delta_elev.to_numpy(), 0.0)),
            'slope': slope.to_numpy(),
            'slope_uphill': np.maximum(slope.to_numpy(), 0.0),
            'slope_downhill': np.abs(np.minimum(slope.to_numpy(), 0.0)),
            'fiets_uphill': fiets.to_numpy(),
            'is_bridge_or_tunnel': is_bridge_or_tunnel.to_numpy(),
        }, index=edge_id_index)
        edges_from_nodes.index = edge_id_index

        context.create_properties(extended_edges, f'{mode}_extended', float_format='%.3g')
        context.create_properties(edges_from_nodes, f'{mode}_from_nodes', float_format='%.3g')
        context.create_properties(node_props, f'{mode}_extended', float_format='%.3g')

    context.close()


if __name__ == '__main__':
    variants.run(main)
