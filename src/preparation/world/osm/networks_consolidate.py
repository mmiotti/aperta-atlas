"""
Consolidate intersections in a per-mode OSM network graph using OSMnx's
`consolidate_intersections` + `aperta_atlas.osm.clean_consolidated_edges`.

Step 3 of the network preparation pipeline (after `clip_pbf.py` and
`networks_from_pbf.py`). Topology-changing: produces a new graphml
with consolidated intersection nodes.

Deliberately compact. Decoration steps (street count, OSM classification
flags, obstacle snap) are SEPARATE in `networks_decorate.py`. Decorations
are pure attribute additions — they save as companion
`properties/edges_*.csv` / `properties/nodes_*.csv` without rewriting
the large consolidated `.graphml`.

`tolerance_m` defaults to 10 m across all modes — matches OSMnx's
convention and standard usage in the literature. Per-mode tuning is
available via `_TOLERANCE_M` below; revisit if empirical results show
one mode wants different aggregation.

Output CRS is WGS84 (`EPSG:4326`), uniform across the
`preparation/world/` namespace. Country-specific reprojection (e.g. to
LV95 for Swiss work) happens at project-stage init, not here.

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_<mode>` variant per (area, mode) combination.

Inputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>.graphml             # from networks_from_pbf.py
    properties/edges_<area_name>_<mode>.csv   # companion edge attrs

Outputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>_consolidated.graphml          # new graph (WGS84, new int IDs)
    properties/edges_<area_name>_<mode>_consolidated.csv
    properties/nodes_<area_name>_<mode>_consolidated.csv

Run after `networks_from_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.networks_consolidate
Single variant:
    python -m preparation.world.osm.networks_consolidate --variant switzerland_car
    python -m preparation.world.osm.networks_consolidate --variant bern_walk
"""

import logging
from typing import cast

import networkx as nx
import osmnx as ox

from aperta.network_processing import attach_edge_properties
from aperta_atlas.context import init_context
from aperta_atlas.graph_simplification import (
    round_coords, simplify_edge_geometries,
)
from aperta_atlas.osm import clean_consolidated_edges
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS


# Edge attributes kept through consolidation — read by downstream
# routing + calibration.
_KEEP_EDGE_ATTRS = [
    'highway',      # edge type — cost-by-class, calibration
    'oneway',       # input to clean_consolidated_edges; can be used in downstream metrics
    'lanes',        # capacity
    'maxspeed',     # speed → time cost
    'bridge',       # float ∈ [0, 1] — length-weighted bridge fraction
    'tunnel',       # float ∈ [0, 1] — length-weighted tunnel fraction
    'access',       # access restrictions
    'service',      # service-road sub-classification
    # Bike-friendliness tags — used to calculate "bike score" later
    # (and for any future bike-routing logic).
    'bicycle', 'cyclestreet', 'cycleway',
    'cycleway:left', 'cycleway:right', 'cycleway:both',
]


# Per-mode script-level tuning. Currently uniform across modes but
# kept as a dict so per-mode overrides are a one-line change.
_TOLERANCE_M = {'walk': 10, 'bike': 10, 'car': 10}


variants = Variants([
    ('area_name', str), ('mode', str), ('tolerance_m', int),
])
for area in AREAS.values():
    for mode in area.buffers:
        variants.add(
            name=f'{area.name}_{mode}',
            area_name=area.name, mode=mode,
            tolerance_m=_TOLERANCE_M[mode],
        )


def main(variant) -> None:
    context = init_context(variant)
    name = f'{variant.area_name}_{variant.mode}'
    out_name = f'{name}_consolidated'

    with step('load network graph + edge properties'):
        g = context.get_nw(data_name=name)
        edges_df = context.get_properties('edges', name)
        edges_df = edges_df[[c for c in _KEEP_EDGE_ATTRS if c in edges_df.columns]]
        attach_edge_properties(g, edges_df)
        logging.info(
            f"  → loaded {g.number_of_nodes():,} nodes, "
            f"{g.number_of_edges():,} edges; "
            f"edge attrs: {sorted(edges_df.columns)}")

    # Consolidation needs a metric CRS for `tolerance_m` to be meaningful
    # (distance-in-meters thresholding). OSMnx's `project_graph` picks an
    # appropriate UTM by default.
    with step('project to metric CRS'):
        g = ox.project_graph(g)

    with step(f'ox.consolidate_intersections (tolerance={variant.tolerance_m} m)'):
        before_n, before_e = g.number_of_nodes(), g.number_of_edges()
        # `rebuild_graph=True` guarantees a MultiDiGraph return; the cast
        # narrows OSMnx's `MultiDiGraph | GeoSeries` union type for the
        # type checker.
        g = cast(
            nx.MultiDiGraph,
            ox.consolidate_intersections(
                g, tolerance=variant.tolerance_m,
                rebuild_graph=True, reconnect_edges=True),
        )
        logging.info(
            f"  → kept {g.number_of_nodes():,} / {before_n:,} nodes, "
            f"{g.number_of_edges():,} / {before_e:,} edges")

    # Douglas-Peucker simplification on each edge's LineString. Tolerance
    # is 0.5 m — drops near-collinear intermediate vertices that
    # contribute essentially nothing to shape. Done while still in the
    # metric CRS so the tolerance value is meters. Must precede
    # `clean_consolidated_edges` so that step's length-recompute uses
    # the simplified geometry.
    with step('simplify_edge_geometries (tolerance=0.5 m)'):
        simplify_edge_geometries(g, tolerance=0.5)

    # Post-consolidation edge cleanup: drop noise attrs, collapse list-
    # valued lanes/maxspeed, recompute length from the (now-simplified)
    # geometry. Derived per-edge attributes like `lanes_per_direction`
    # are written by `networks_decorate.py`, not here.
    with step('clean_consolidated_edges'):
        clean_consolidated_edges(g, drop_edge_attrs=['name', 'osmid'])

    # Project back to WGS84 — uniform across the preparation/world/
    # namespace. Country-specific reprojection happens at project init.
    with step('project back to WGS84'):
        g = ox.project_graph(g, to_crs='EPSG:4326')

    # The project/unproject round-trip leaves coords with 15-17 trailing
    # digits — pure precision noise that costs real bytes in graphml.
    with step('round_coords (7 decimals ≈ 1 cm)'):
        round_coords(g, decimals=7)

    with step('save consolidated graphml + properties'):
        context.create_nw(g, data_name=out_name, save_properties=True)

    context.close()


if __name__ == '__main__':
    variants.run(main)
