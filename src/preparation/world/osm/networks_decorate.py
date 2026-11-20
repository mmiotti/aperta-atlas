"""
Decorate a per-mode consolidated OSM network with per-node / per-edge
attributes for downstream routing + calibration.

Pure attribute additions on the consolidated graph — no topology changes, so this
step is much cheaper than `networks_consolidate.py` and safe to iterate
without re-running the bottleneck.

Sequence, applied in order:

  1. `flag_node_intersection_topology` — `n_streets`, `is_t_junction`, `is_4way` per node.
  2. `flag_node_osm_classification` — highway ranks and major / anchor junction flags per
     node, from each edge's single-string `highway` (collapsed in `networks_consolidate.py`).
  3. `lanes_per_direction` per edge, one-way derived from the graph structure.
  4. Fill missing `lanes` with 1 (after step 3, which handles missing values itself).
  5. `ox.add_edge_speeds` + `ox.add_edge_travel_times` — `speed_kph` from `maxspeed`, else
     the class default (`_DEFAULT_HIGHWAY_SPEEDS_KPH`), and `travel_time`.
  6. Obstacle snap — `is_<kind>` per node for each obstacle kind, within
     `obstacle_buffer_m` (`_OBSTACLE_BUFFER_M`), in a metric CRS; traffic signals on bike /
     car via the road edge they sit on (`_snap_signals_to_road_edges`).

Writes to a separate `_decorated` suffix file rather than overwriting
the `_consolidated` base CSVs. This way decorate is idempotent: every
re-run reads the same canonical `_consolidated` properties (whatever
consolidate produced) and re-emits the decoration overlay from scratch,
no cumulative drift. Downstream readers layer base + decoration via
`get_nw(..., add_edge_properties=[name, f'{name}_decorated'])`.

The graphml skeleton is unchanged and not re-saved.

Inputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>_consolidated.graphml          # networks_consolidate.py output
    properties/edges_<area_name>_<mode>_consolidated.csv
    obstacles_<area_name>.gpkg                          # obstacles_from_pbf.py output

Outputs (PUBLIC, under preparation/world/osm/):
    properties/edges_<area_name>_<mode>_consolidated_decorated.csv
        # highway, lanes (missing → 1), lanes_per_direction, speed_kph, travel_time
    properties/nodes_<area_name>_<mode>_consolidated_decorated.csv
        # topology flags + OSM classification flags + obstacle flags

Run all variants sequentially (default):
    python -m preparation.world.osm.networks_decorate
Single variant:
    python -m preparation.world.osm.networks_decorate --variant switzerland_car
    python -m preparation.world.osm.networks_decorate --variant bern-metro_walk
"""

import logging
import math

import osmnx as ox
import shapely
from shapely.geometry import LineString

from aperta.network_processing import (
    flag_node_intersection_topology,
    snap_features_to_nodes,
)
from aperta_atlas.context import init_context
from aperta_atlas.osm import flag_node_osm_classification, lanes_per_direction
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS


# Plural OSM tag values (as collected by `obstacles_from_pbf.ObstacleHandler`)
# mapped to the singular flag names that downstream cost models expect.
# `snap_features_to_nodes` prepends `is_`, so the on-graph attrs are
# `is_traffic_signal`, `is_stop`, etc.
_OBSTACLE_KIND_TO_FLAG: dict[str, str] = {
    'traffic_signals': 'traffic_signal',
    'stop':            'stop',
    'give_way':        'give_way',
    'crossing':        'crossing',
    'mini_roundabout': 'mini_roundabout',
    'roundabout':      'roundabout',
}


# Per-mode obstacle-snap distance (m). Walk is shorter: its streets with separately mapped
# sidewalks are not in the network, so a signal there snaps to a sidewalk node, and only the
# near ones are at the crossing (bern-metro: 90 % of the signals within 30 m are within 20 m).
_OBSTACLE_BUFFER_M = {'walk': 20.0, 'bike': 30.0, 'car': 30.0}

# Traffic signals on bike / car snap to the road edge they sit on (OSM puts them on their road's
# way) within `_SIGNAL_EDGE_M`, then to that edge's nearer end within `obstacle_buffer_m`
# (`_snap_signals_to_road_edges`). A nearest-node snap also picks side-street, sidewalk and path
# nodes near the junction (bern-metro: 9 % of signals on bike, 4 % on car). Walk keeps the
# nearest-node snap: most of its signals sit on streets the walk network does not contain.
_SIGNAL_EDGE_MODES = ('bike', 'car')
_SIGNAL_EDGE_M = 10.0
# Roads a signal can control: the car classes without `service` (driveways, parking aisles).
_SIGNAL_ROADS: frozenset[str] = frozenset({
    'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified',
    'residential', 'living_street', 'road',
})


# Per-highway-class speed defaults (km/h) for `ox.add_edge_speeds`, where an edge's own
# `maxspeed` is missing or unparseable: typical Swiss / Central-European legal limits, the same
# for all areas. `_FALLBACK_SPEED_KPH` covers classes not listed.
_DEFAULT_HIGHWAY_SPEEDS_KPH: dict[str, int] = {
    'motorway':       120, 'motorway_link':   60,
    'trunk':          100, 'trunk_link':      60,
    'primary':         80, 'primary_link':    50,
    'secondary':       60, 'secondary_link':  40,
    'tertiary':        50, 'tertiary_link':   30,
    'unclassified':    50,
    'residential':     30, 'living_street':   20,
    'service':         30,
}
_FALLBACK_SPEED_KPH = 50


variants = Variants([
    ('area_name', str), ('mode', str), ('obstacle_buffer_m', float),
])
for area in AREAS.values():
    for mode in area.buffers:
        variants.add(
            name=f'{area.name}_{mode}',
            area_name=area.name, mode=mode,
            obstacle_buffer_m=_OBSTACLE_BUFFER_M[mode],
        )


def _write_lanes_per_direction(graph) -> None:
    """Per-edge `lanes_per_direction`, with one-way taken from the graph rather than the OSM
    tag: an edge `(u, v)` is one-way iff its reverse `(v, u)` does not exist, i.e. as
    `networks_from_pbf.py` emitted the directions for the mode."""
    for u, v, _, d in graph.edges(keys=True, data=True):
        d['lanes_per_direction'] = lanes_per_direction({
            **d,
            'oneway': not graph.has_edge(v, u),
        })


def _snap_signals_to_road_edges(
    graph, locations: list[tuple[float, float]], *, max_edge_m: float, max_node_m: float,
) -> None:
    """Write `is_traffic_signal` per node, in place: each signal snaps to the nearest road edge
    (`_SIGNAL_ROADS`) within `max_edge_m` and flags that edge's nearer end, if within
    `max_node_m`. Signals farther from any road edge, or mid-block (no end within `max_node_m`),
    flag nothing. Metric CRS."""
    for _, d in graph.nodes(data=True):
        d['is_traffic_signal'] = 0
    xy = {n: (d['x'], d['y']) for n, d in graph.nodes(data=True)}
    roads = [(u, v, d.get('geometry') or LineString([xy[u], xy[v]]))
             for u, v, d in graph.edges(data=True) if d.get('highway') in _SIGNAL_ROADS]
    if not roads or not locations:
        return
    tree = shapely.STRtree([geom for *_, geom in roads])
    # One edge per signal; ties (both directions of a street, the edges at a junction-centre
    # signal) share the nearer end.
    loc_idx, edge_idx = tree.query_nearest(
        shapely.points(locations), max_distance=max_edge_m, all_matches=False)
    for p, e in zip(loc_idx, edge_idx):
        x, y = locations[p]
        dist, node = min((math.hypot(xy[n][0] - x, xy[n][1] - y), n) for n in roads[e][:2])
        if dist <= max_node_m:
            graph.nodes[node]['is_traffic_signal'] = 1


def main(variant) -> None:
    context = init_context(variant)
    name = f'{variant.area_name}_{variant.mode}_consolidated'

    with step('load consolidated graph + edge properties'):
        g = context.get_nw(data_name=name, add_edge_properties=name)
        logging.info(f"  → loaded {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    with step('flag_node_intersection_topology'):
        flag_node_intersection_topology(g)

    with step('flag_node_osm_classification'):
        flag_node_osm_classification(g)

    with step('lanes_per_direction (structural oneway)'):
        _write_lanes_per_direction(g)

    # After `lanes_per_direction`, which handles missing `lanes` itself.
    with step('fill missing lanes (1)'):
        n_filled = 0
        for _, _, _, d in g.edges(keys=True, data=True):
            lanes = d.get('lanes')
            if lanes is None or (isinstance(lanes, float) and lanes != lanes):
                d['lanes'] = 1
                n_filled += 1
        logging.info(f"  → filled {n_filled:,} edges (of {g.number_of_edges():,}) with lanes=1")

    # `maxspeed` itself is left intact.
    with step('add_edge_speeds (fills speed_kph; class defaults)'):
        ox.add_edge_speeds(
            g,
            hwy_speeds=_DEFAULT_HIGHWAY_SPEEDS_KPH,
            fallback=_FALLBACK_SPEED_KPH,
        )

    # Companion to add_edge_speeds: derives `travel_time` (seconds) from
    # `speed_kph` + edge `length`.
    with step('add_edge_travel_times'):
        ox.add_edge_travel_times(g)

    # Obstacle snap requires a metric CRS so `obstacle_buffer_m` is meters.
    # OSMnx's `project_graph` picks an appropriate UTM by default.
    with step('project to metric CRS'):
        g = ox.project_graph(g)

    with step('load obstacles GDF'):
        obstacles = context.get_generic(f'obstacles_{variant.area_name}.gpkg')
        # Reproject to the metric CRS that the graph is now in so the
        # snap distance is comparable.
        obstacles = obstacles.to_crs(g.graph['crs'])
        logging.info(
            f"  → {len(obstacles):,} obstacles ({dict(obstacles['kind'].value_counts())})")

    with step(f'snap_features_to_nodes (buffer={variant.obstacle_buffer_m} m)'):
        for kind, flag_name in _OBSTACLE_KIND_TO_FLAG.items():
            subset = obstacles[obstacles['kind'] == kind]
            locations = [(p.x, p.y) for p in subset.geometry]
            if kind == 'traffic_signals' and variant.mode in _SIGNAL_EDGE_MODES:
                _snap_signals_to_road_edges(
                    g, locations,
                    max_edge_m=_SIGNAL_EDGE_M, max_node_m=variant.obstacle_buffer_m,
                )
            else:
                snap_features_to_nodes(
                    g, locations,
                    flag_name=flag_name,
                    max_distance=variant.obstacle_buffer_m,
                )
            n_snapped = sum(1 for _, d in g.nodes(data=True) if d.get(f'is_{flag_name}') == 1)
            logging.info(
                f"  → is_{flag_name}: {n_snapped:,} nodes "
                f"(from {len(locations):,} obstacles)")

    # Back to WGS84 for the save — uniform across `preparation/world/`.
    with step('project back to WGS84'):
        g = ox.project_graph(g, to_crs='EPSG:4326')

    decoration_columns = [
        # Edge decorations
        'highway',
        'lanes', 'lanes_per_direction',
        'speed_kph', 'travel_time',
        # Node topology
        'n_streets', 'is_t_junction', 'is_4way',
        # Node OSM classification
        'max_highway_rank', 'min_highway_rank',
        'is_t_junction_major', 'is_4way_major',
        'is_t_junction_anchor', 'is_4way_anchor',
        # Node obstacle flags (one per `_OBSTACLE_KIND_TO_FLAG` value)
        *(f'is_{flag}' for flag in _OBSTACLE_KIND_TO_FLAG.values()),
    ]
    with step('save decoration overlay (edges + nodes, suffix=decorated)'):
        context.create_nw(
            g, data_name=name,
            save_skeleton=False, save_properties=True,
            properties_name='decorated',
            properties_columns=decoration_columns,
        )

    context.close()


if __name__ == '__main__':
    variants.run(main)
