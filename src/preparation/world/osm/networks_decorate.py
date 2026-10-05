"""
Decorate a per-mode consolidated OSM network with per-node / per-edge
attributes for downstream routing + calibration.

Step 4 of the network preparation pipeline (after `clip_pbf.py`,
`networks_from_pbf.py`, `networks_consolidate.py`). Pure attribute
additions on the consolidated graph — no topology changes, so this
step is much cheaper than `networks_consolidate.py` and safe to iterate
without re-running the bottleneck.

Sequence, applied in order:

  1. `flag_node_intersection_topology` — `n_streets`, `is_t_junction`,
     `is_4way` per node. Network-agnostic (uses degree only).

  2. `flag_node_osm_classification` — `max_highway_rank`,
     `min_highway_rank`, `is_t_junction_major`, `is_4way_major`,
     `is_t_junction_anchor`, `is_4way_anchor` per node. OSM-aware
     (reads each edge's `highway`, which `networks_consolidate.py`
     guarantees as a single string via `clean_consolidated_edges`'s
     `highway` aggregator).

  3. `lanes_per_direction` per edge — total OSM `lanes` corrected for
     two-way roads. `oneway` was dropped during consolidation, so we
     derive it structurally: an edge `(u, v)` is one-way iff its
     reverse `(v, u)` doesn't exist in the directed graph. Internally
     returns 1.0 where raw `lanes` is missing or unparseable.

  4. Fill missing `lanes` with 1. Done AFTER step 3 so the
     per-direction calc sees genuine NaN (and applies its own missing
     logic) rather than the filled value.

  5. `ox.add_edge_speeds` + `ox.add_edge_travel_times` — write
     `speed_kph` (km/h) from `maxspeed` with class-based defaults
     (`_DEFAULT_HIGHWAY_SPEEDS_KPH`), then derive `travel_time`
     (seconds) from `speed_kph` + edge `length`. Explicit per-edge
     `maxspeed` tags always win; the dict only fills gaps.

  6. Obstacle snap — for each obstacle kind in `obstacles_<area_name>.gpkg`
     (from `obstacles_from_pbf.py`), snap to the nearest consolidated
     node within `obstacle_buffer_m` and write
     `is_<kind>` (boolean per node). Reprojects to a metric CRS for the
     snap, then projects back to WGS84 (matches the
     `preparation/world/` WGS84-everywhere convention).

Writes to a separate `_decorated` suffix file rather than overwriting
the `_consolidated` base CSVs. This way decorate is idempotent: every
re-run reads the same canonical `_consolidated` properties (whatever
consolidate produced) and re-emits the decoration overlay from scratch,
no cumulative drift. Downstream readers layer base + decoration via
`get_nw(..., add_edge_properties=[name, f'{name}_decorated'])`.

The graphml skeleton is unchanged and not re-saved.

Obstacle kind → flag name translation (singular, what downstream cost
models expect):

    traffic_signals  → is_traffic_signal
    stop             → is_stop
    give_way         → is_give_way
    crossing         → is_crossing
    mini_roundabout  → is_mini_roundabout
    roundabout       → is_roundabout

`obstacle_buffer_m` defaults to 30 m across all modes — comfortably
covers signalised intersections (signal node typically sits 5-15 m off
the intersection centre in OSM).

Inputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>_consolidated.graphml          # consolidate output
    properties/edges_<area_name>_<mode>_consolidated.csv
    properties/nodes_<area_name>_<mode>_consolidated.csv
    obstacles_<area_name>.gpkg                          # obstacles_from_pbf output

Outputs (PUBLIC, under preparation/world/osm/):
    properties/edges_<area_name>_<mode>_consolidated_decorated.csv
        # collapsed highway (overrides base) + lanes_per_direction
    properties/nodes_<area_name>_<mode>_consolidated_decorated.csv
        # topology flags + OSM classification flags + obstacle flags

Run after `networks_consolidate.py` AND `obstacles_from_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.networks_decorate
Single variant:
    python -m preparation.world.osm.networks_decorate --variant switzerland_car
    python -m preparation.world.osm.networks_decorate --variant bern_walk
"""

import logging

import osmnx as ox

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


# Per-mode obstacle-snap distance. Uniform across modes for now.
_OBSTACLE_BUFFER_M = {'walk': 30.0, 'bike': 30.0, 'car': 30.0}


# Per-highway-class speed defaults (km/h) for `ox.add_edge_speeds`.
# Values reflect typical Swiss / Central-European legal limits. OSMnx
# uses these only where the edge's own `maxspeed` is missing or
# unparseable; explicit per-edge `maxspeed` tags always win. The
# `fallback` covers highway classes not listed here.
#
# These are script-level constants (area-agnostic for now). If a future
# non-European area needs different limits, promote to per-area via
# `Case.highway_speeds_kph` or override here per variant.
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
    """Per-edge `lanes_per_direction` using structural oneway probe.

    OSM `oneway` was dropped during `networks_consolidate.py`'s cleanup
    pass (redundant once the graph is directed), so we derive it here
    from graph structure: an edge `(u, v)` is one-way iff its reverse
    `(v, u)` does NOT exist. This matches the post-`emit_directions`
    behavior of `networks_from_pbf.py`, where two-way streets emit edges
    in both directions and one-way streets emit one.
    """
    for u, v, _, d in graph.edges(keys=True, data=True):
        d['lanes_per_direction'] = lanes_per_direction({
            **d,
            'oneway': not graph.has_edge(v, u),
        })


def main(variant) -> None:
    context = init_context(variant)
    name = f'{variant.area_name}_{variant.mode}_consolidated'

    # Load consolidated graphml + the companion edges CSV from consolidate
    # in one shot. Node properties not loaded (consolidate doesn't write
    # any — node attrs are decorate's output).
    with step('load consolidated graph + edge properties'):
        g = context.get_nw(data_name=name, add_edge_properties=name)
        logging.info(f"  → loaded {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    # `highway` list-collapse already done in `networks_consolidate.py`
    # via `clean_consolidated_edges` (alongside `lanes`/`maxspeed`); no
    # need to repeat here. `flag_node_osm_classification` below relies
    # on single-string `highway` per edge — consolidate is responsible.

    with step('flag_node_intersection_topology'):
        flag_node_intersection_topology(g)

    with step('flag_node_osm_classification'):
        flag_node_osm_classification(g)

    with step('lanes_per_direction (structural oneway)'):
        _write_lanes_per_direction(g)

    # Raw `lanes` may still be NaN/missing where OSM didn't tag it.
    # `lanes_per_direction` handles this internally (returns 1.0 on
    # missing), but downstream consumers that read raw `lanes` would
    # see NaN. Fill to 1 AFTER lanes_per_direction so the per-direction
    # calc sees the genuine NaN and applies its own missing logic.
    with step('fill missing lanes (1)'):
        n_filled = 0
        for _, _, _, d in g.edges(keys=True, data=True):
            lanes = d.get('lanes')
            if lanes is None or (isinstance(lanes, float) and lanes != lanes):
                d['lanes'] = 1
                n_filled += 1
        logging.info(f"  → filled {n_filled:,} edges (of {g.number_of_edges():,}) with lanes=1")

    # Fill `speed_kph` from `maxspeed` with per-highway-class defaults.
    # OSMnx parses any explicit maxspeed string first, then falls back
    # to the per-class value from `hwy_speeds`, then to `fallback`.
    # Original `maxspeed` left intact; `speed_kph` is a new attribute.
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
        # Edge decorations (override `highway` for the list-valued area;
        # NaN-fill `lanes`; derive lanes_per_direction, speed_kph, travel_time)
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
