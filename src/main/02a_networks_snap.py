"""
Network TOPOLOGY for the atlas: snap + virtual-node insertion, per mode.

Stage 1 of the two-stage network pipeline (heavy + stable). Per mode
(walk, bike, car): load consolidated + decorated network → insert
virtual nodes at cell centroids → snap cells + zones to nearest
network nodes → persist. The lighter `02b_networks_features.py` can
then iterate on features without re-running snap.

Per invocation writes one per-mode snap CSV. The merged
`cells_snap.csv` / `zones_snap.csv` (with `is_active`) is assembled
later by `02c_active_flags.py`.

Zone centroids:
  - NPVM path (Swiss): population-weighted centroids from BFS;
    foreign zones' placeholders replaced with `transport_centroid`
    on the car network.
  - H3 path (non-Swiss): `transport_centroid` for every zone (car
    network — densest, most representative routable centre).

Outputs (under <scenario>/, PUBLIC):
    nw/<mode>.graphml                  # network (with cell virtuals)
    shapes/nodes_<mode>.gpkg           # post-insertion node geometries
    shapes/edges_<mode>.gpkg           # post-insertion edge geometries
    shapes/zones_centroids.gpkg        # custom-for-CH, transport-for-foreign
    properties/cells_<mode>_snap.csv   # node_id + distance (raw)
    properties/zones_<mode>_snap.csv   # same for zones
"""

import logging

import geopandas as gpd
import numpy as np  # noqa: F401  (used implicitly by aperta helpers)
import osmnx as ox
import pandas as pd

from aperta import network_snap, routing_prep
from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from scenarios import get_scenario


# Modes to process. For the Swiss case these match `case.buffers.keys()`.
variants = Variants([('mode', str)])
for _m in ('walk', 'bike', 'car'):
    variants.add(name=_m, mode=_m)

# Snap radii + insertion thresholds live in `scenario.snap` (SnapConfig
# in scenarios.py) — region-dependent (sparse rural networks need larger
# radii than dense urban).
#
# All atlas graphs are directed (one-way streets + elevation gradients
# need direction). compute_snap_eligibility + compute_snap_eligible_nodes
# are called with this setting → snap eligibility = largest STRONGLY
# connected component (no traps).
_DIRECTEDNESS = 'directed_scc'
# No tag-based cost exclusions on top of what the PBF extraction already
# filters per-mode (walk doesn't get motorways, etc.) — the user's
# constraint is "finite cost + connected to the rest of the network",
# both of which the SCC computation handles directly.
_COST_EXCLUDED_TAGS: frozenset[str] = frozenset()
# Network-type label per mode — accurate for our pipeline (PBF extraction
# already mode-filters the graphs). Reported to aperta's `compute_snap_eligibility`
# so its warning policies have the correct context.
_NETWORK_TYPE_BY_MODE: dict[str, str] = {
    'walk': 'walk', 'bike': 'bike', 'car': 'drive',
}

# Per-node intersection-type flags that mark "priority-for-zone-snap"
# nodes. ANY of these → priority. They're written by networks_decorate.py.
_PRIORITY_NODE_FLAGS = (
    'is_t_junction_anchor', 'is_4way_anchor',
    'is_t_junction_major',  'is_4way_major',
)


def _load_mode_graph(osm_ctx, area_name: str, mode: str, to_crs: str):
    """Load `<area>_<mode>_consolidated` and reproject to `to_crs`.

    Centralised loader so the projection step can't be forgotten at one
    call site and not another. Consolidated graphs ship in WGS84 (per
    `networks_consolidate.py`); everything downstream (insert / snap /
    sum_within_radius / transport_centroid) wants metric coordinates.
    """
    name = f'{area_name}_{mode}_consolidated'
    return ox.project_graph(
        osm_ctx.get_nw(
            name,
            add_node_properties='decorated',
            add_edge_properties=[name, 'decorated'],
            allow_cache=False,
        ),
        to_crs=to_crs,
    )


def main(variant) -> None:
    context = init_context(variant)
    mode = variant.mode
    scenario = get_scenario(context.scenario)
    crs_main = scenario.crs_main

    # ---------- Load cells + zones once (reused per mode) ----------------
    cells = context.get_shapes('cells').to_crs(crs_main)
    logging.info(f"  → {len(cells):,} cells loaded")
    zones = context.get_shapes('zones').to_crs(crs_main)
    logging.info(f"  → {len(zones):,} zones loaded")

    # Pre-compute cell centroids (reused per mode for insertion + snap).
    cell_centroids = gpd.GeoDataFrame(geometry=cells.geometry.centroid)

    osm_ctx = context.source('preparation/world/osm')

    # Zone centroids — see module docstring. Both paths produce
    # `shapes/zones_centroids.gpkg` with schema (geometry, is_in_ch).
    if scenario.zone_h3_resolution is None:
        with step('load prepared zone centroids (custom-for-CH placeholder)'):
            gen_ctx = context.source('preparation/switzerland/general')
            zone_centroids = (
                gen_ctx.get_generic('traffic_zones_centroids.gpkg')
                .set_index('zone_id')
                .to_crs(crs_main)
                .loc[zones.index])  # restrict to atlas-kept zones (01's filter)
            # Carry `is_in_ch` from zones (cells.gpkg) so the swap below
            # + downstream consumers can rely on a single canonical flag.
            zone_centroids['is_in_ch'] = zones['is_in_ch'].astype(bool)

        with step('replace foreign-zone centroids with transport_centroid (car network)'):
            # Pre-loop, no virtuals inserted → car graph is pristine here.
            car_graph = _load_mode_graph(osm_ctx, scenario.area_name, 'car', crs_main)
            # Restrict the centroid basis to the car SCC — isolated islands
            # would otherwise pull the median toward an unreachable node.
            car_snap_eligible, _ = routing_prep.compute_snap_eligibility(
                car_graph, 'car',
                directedness=_DIRECTEDNESS,
                cost_excluded_tags=_COST_EXCLUDED_TAGS,
                network_type='drive',
            )
            foreign_idx = zone_centroids.index[~zone_centroids['is_in_ch']]
            foreign_polys = zones.loc[foreign_idx]
            transport_foreign = network_snap.transport_centroid(
                foreign_polys, car_graph,
                eligible_node_ids=car_snap_eligible,
            )
            zone_centroids.loc[foreign_idx, 'geometry'] = transport_foreign.geometry
            logging.info(f"  → {len(foreign_idx):,} foreign zones updated")
    else:
        with step(f'compute transport_centroid for all H3 zones (car-network node median)'):
            car_graph = _load_mode_graph(osm_ctx, scenario.area_name, 'car', crs_main)
            car_snap_eligible, _ = routing_prep.compute_snap_eligibility(
                car_graph, 'car',
                directedness=_DIRECTEDNESS,
                cost_excluded_tags=_COST_EXCLUDED_TAGS,
                network_type='drive',
            )
            zone_centroids = network_snap.transport_centroid(
                zones, car_graph,
                eligible_node_ids=car_snap_eligible,
            )
            # Carry `is_in_ch` forward so downstream code (and the
            # saved GPKG) matches the NPVM branch's schema.
            zone_centroids['is_in_ch'] = zones['is_in_ch'].astype(bool)
            logging.info(
                f"  → {len(zone_centroids):,} H3 zone centroids "
                f"({int(zone_centroids['is_in_ch'].sum()):,} in-CH / "
                f"{int((~zone_centroids['is_in_ch']).sum()):,} buffer)")

    context.create_shapes(zone_centroids[['is_in_ch', 'geometry']], data_name='centroids')
    graph = _load_mode_graph(osm_ctx, scenario.area_name, mode, crs_main)

    with step(f'mode={mode}: compute snap eligibility (largest SCC)'):
        # Pre-insertion: writes cost_excluded_<mode> per edge; returns
        # the SCC node set used to anchor the virtual-node insertion
        # below (so virtuals can never land in an isolated island).
        snap_eligible_pre, cost_excluded_flag = (
            routing_prep.compute_snap_eligibility(
                graph, mode,
                directedness=_DIRECTEDNESS,
                cost_excluded_tags=_COST_EXCLUDED_TAGS,
                network_type=_NETWORK_TYPE_BY_MODE[mode],
            ))
        logging.info(
            f"  → snap-eligible (SCC): {len(snap_eligible_pre):,} of "
            f"{graph.number_of_nodes():,} nodes "
            f"({100*len(snap_eligible_pre)/graph.number_of_nodes():.1f} %)")

    with step(f'mode={mode}: insert virtual nodes at cell centroids'):
        n_before = graph.number_of_nodes()
        network_snap.insert_projected_nodes(
            cell_centroids, graph,
            max_distance=scenario.snap.insert_max_radius_m,
            node_spacing=scenario.snap.insert_node_spacing_m[mode],
            eligible_node_ids=snap_eligible_pre,
            cost_excluded_flag=cost_excluded_flag,
        )
        logging.info(
            f"  → inserted {graph.number_of_nodes() - n_before:,} virtual nodes")

    with step(f'mode={mode}: recompute SCC eligibility (post-insertion)'):
        # Pure compute, no graph mutation. Picks up the per-edge
        # cost-mask flag that was written in the pre-insertion
        # compute_snap_eligibility above (child edges from insertion
        # inherited the parent's flag value). The post-insertion SCC
        # = pre-insertion SCC plus the inserted virtuals.
        snap_eligible = routing_prep.compute_snap_eligible_nodes(
            graph, directedness=_DIRECTEDNESS,
            cost_excluded_flag=cost_excluded_flag,
        )
        logging.info(
            f"  → snap-eligible post-insertion: {len(snap_eligible):,} of "
            f"{graph.number_of_nodes():,} nodes "
            f"({100*len(snap_eligible)/graph.number_of_nodes():.1f} %)")

    with step(f'mode={mode}: compute priority-node set for zone snap'):
        # Anchor and/or major (4-way or T-junction) intersections —
        # the nodes a router would treat as "real" decision points.
        # Passed via priority_node_ids= below, no graph mutation. The
        # 500 m radius cap is enforced by snap_to_network_nodes via
        # `priority_node_max_distance`, not by pre-filtering here.
        priority_nodes = frozenset(
            n for n, data in graph.nodes(data=True)
            if any(data.get(f, 0) for f in _PRIORITY_NODE_FLAGS)
        ) & snap_eligible  # never snap to a priority node outside the SCC
        logging.info(
            f"  → {len(priority_nodes):,} priority nodes "
            f"({100*len(priority_nodes)/graph.number_of_nodes():.1f} %)")

    with step(f'mode={mode}: snap cells + zones → network nodes'):
        # Cells: every cell has a virtual nearby → tiny snap distance.
        cell_node_ids, cell_dists = network_snap.snap_to_network_nodes(
            cell_centroids, graph,
            max_distance=scenario.snap.max_cell_radius_m,
            eligible_node_ids=snap_eligible,
        )
        # Zones: priority tier targets anchor/major intersections
        # within `zone_priority_radius_m`; fallback tier snaps to any
        # snap-eligible node (real or virtual).
        zone_node_ids, zone_dists = network_snap.snap_to_network_nodes(
            zone_centroids, graph,
            max_distance=scenario.snap.max_zone_radius_m,
            eligible_node_ids=snap_eligible,
            priority_node_ids=priority_nodes,
            priority_node_max_distance=scenario.snap.zone_priority_radius_m,
        )
        # Self-contained per-mode output (one variant invocation = one
        # file pair). The merged shape with per-mode columns + the active
        # flag is built later by `02c_active_flags.py`.
        cell_snap = pd.DataFrame(
            {'node_id': cell_node_ids, 'distance': cell_dists},
            index=cells.index,
        )
        zone_snap = pd.DataFrame(
            {'node_id': zone_node_ids, 'distance': zone_dists},
            index=zones.index,
        )
        context.create_properties(cell_snap, data_name=f'snap_{mode}')
        context.create_properties(zone_snap, data_name=f'snap_{mode}')
        logging.info(
            f"  → cells: median snap dist {cell_dists.median():.1f} m, "
            f"max {cell_dists.max():.1f} m; "
            f"zones: median {zone_dists.median():.1f} m, "
            f"max {zone_dists.max():.1f} m")

    context.create_nw(
        graph, data_name=mode, save_skeleton=True, save_shapes=True,
        save_properties=True, properties_name='core'
    )

    context.close()


if __name__ == '__main__':
    variants.run(main)
