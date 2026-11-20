"""
Build OSM road / cycling / walking networks from a locally clipped PBF using a
pyosmium streaming handler. Streams the PBF and pre-simplifies during the build
(only intersection nodes enter the NetworkX graph; intermediate way-vertices
go into edge LineString geometry), keeping peak RAM bounded by output graph
size — suitable for country-scale extracts.

Pipeline, per variant: per-variant osmium extract → pyosmium stream →
pre-simplified MultiDiGraph build (mode-filtered at way level by
`aperta_atlas.osm.is_usable`; the bike network leaves out mountain-hiking
trails unless they are part of a mountain-bike route) → walk / bike:
drop the ways of streets mapped twice that the mode does not use (a sidewalk
for bike, a road with separately mapped sidewalks for walk; cautiously, see
`aperta_atlas.osm.drop_sidepath_ways`) → OSMnx `largest_component` →
`collapse_degree_2_chains` (our own geometry-preserving simplifier) →
`add_edge_lengths` → save graphml + polygon.

Inputs (PUBLIC, under raw/global/osm/):
    <pbf_name>-latest.osm.pbf       # output of `clip_pbf.py`

Outputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>.graphml          # OSM network skeleton (WGS84)
    <area_name>_<mode>_polygon.gpkg        # buffered AOI polygon (WGS84)

Areas are defined in `preparation/world/areas.py`; one `<area_name>_<mode>`
variant per (area, mode). Requires the `osmium` CLI (conda-forge `osmium-tool`).

Run all variants sequentially (default):
    python -m preparation.world.osm.networks_from_pbf
Single variant:
    python -m preparation.world.osm.networks_from_pbf \\
        --variant switzerland_car
    python -m preparation.world.osm.networks_from_pbf \\
        --variant bern-metro_walk
"""

import logging
from collections import Counter

import networkx as nx
import osmium
import osmnx as ox
from shapely.geometry import LineString

from aperta_atlas.context import init_context
from aperta_atlas.graph_simplification import (
    collapse_degree_2_chains, prune_short_dead_ends,
)
from aperta_atlas.osm import (
    BRIDGE_YES_VALUES,
    NEEDED_TAGS as _OSM_NEEDED_TAGS,
    TUNNEL_YES_VALUES,
    cycleway_in_direction,
    drop_sidepath_ways,
    lanes_in_direction,
    motor_traffic_oneway,
    parse_maxspeed_kmh,
    emit_directions,
    is_usable,
)
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS
from preparation.world.osm.common import clipped_pbf


# `network_type` is the OSMnx vocabulary (`drive` vs the aperta `car` mode
# label). `prune_dead_end_max_m` removes dead-end branches shorter than the
# given length (0 = no pruning); single-pass (not iterative), so the new final
# node is never further than this value from the previous one.
_NETWORK_TYPE = {'walk': 'walk', 'bike': 'bike', 'car': 'drive'}
# TODO: integrate below constant into area definitions in areas.py
_PRUNE_DEAD_END_MAX_M = {'walk': 25, 'bike': 50, 'car': 100}


variants = Variants([
    ('place', str), ('area_name', str), ('pbf_name', str), ('mode', str),
    ('network_type', str), ('buffer', int), ('prune_dead_end_max_m', int),
])
for area in AREAS.values():
    for mode, buffer in area.buffers.items():
        variants.add(
            name=f'{area.name}_{mode}',
            place=area.place, area_name=area.name,
            pbf_name=f'{area.name}_buffered',
            mode=mode, network_type=_NETWORK_TYPE[mode], buffer=buffer,
            prune_dead_end_max_m=_PRUNE_DEAD_END_MAX_M[mode],
        )


# Tags pulled from each highway way into edge attributes — filtering
# (`is_usable`), routing direction (incl. mode-specific contraflow),
# capacity, calibration, navigation context.
_AUXILIARY_WAY_TAGS: tuple[str, ...] = (
    'oneway', 'lanes', 'maxspeed', 'ref',
    'bridge', 'tunnel', 'service', 'junction',
    'oneway:bicycle', 'oneway:foot', 'cyclestreet',
    'cycleway', 'cycleway:left', 'cycleway:right', 'cycleway:both',
    # `lanes_in_direction` (bike traffic stress):
    'lanes:forward', 'lanes:backward', 'lanes:bus', 'lanes:psv',
    'lanes:bus:forward', 'lanes:bus:backward', 'lanes:psv:forward', 'lanes:psv:backward',
    'surface', 'tracktype', 'smoothness',     # → `surface_type` (main/02b)
    'mtb:scale',                              # → `bike_push` (main/02b)
)
_USEFUL_WAY_TAGS: tuple[str, ...] = _OSM_NEEDED_TAGS + _AUXILIARY_WAY_TAGS


class HighwayGraphHandler(osmium.SimpleHandler):
    """Streaming pyosmium handler that collects only highway ways and their
    node geometry. Bounded RAM regardless of input PBF size.

    `locations=True` (passed to `apply_file`) enables pyosmium's
    `NodeLocationsForWays` cache, which resolves way-node geometry on
    the fly in C++ memory. Without it, `n.location` on way nodes is
    unresolved.

    Also collects the ways in `route=mtb` relations. Relations follow the
    ways in a PBF, so `mark_mtb_routes` tags them (`mtb_route=yes`, read by
    `is_usable`) after the parse.
    """

    def __init__(self) -> None:
        super().__init__()
        # (osm_way_id, useful_tags_dict, [(node_id, lat, lon), ...])
        self.ways: list[tuple[int, dict, list[tuple[int, float, float]]]] = []
        self.n_seen = 0
        self.mtb_route_ways: set[int] = set()

    def relation(self, r) -> None:
        if r.tags.get('route') == 'mtb':
            self.mtb_route_ways.update(m.ref for m in r.members if m.type == 'w')

    def mark_mtb_routes(self) -> int:
        """Set `mtb_route=yes` on the collected ways in a `route=mtb` relation; returns their
        number."""
        n = 0
        for way_id, tags, _ in self.ways:
            if way_id in self.mtb_route_ways:
                tags['mtb_route'] = 'yes'
                n += 1
        return n

    def way(self, w) -> None:
        self.n_seen += 1
        if 'highway' not in w.tags:
            return
        tags = {k: w.tags[k] for k in _USEFUL_WAY_TAGS if k in w.tags}
        nodes = [(n.ref, n.location.lat, n.location.lon) for n in w.nodes if n.location.valid()]
        if len(nodes) >= 2:
            self.ways.append((w.id, tags, nodes))


def build_multidigraph(handler_ways, network_type: str,
                       left_hand_traffic: bool = False) -> nx.MultiDiGraph:
    """Build a PRE-SIMPLIFIED MultiDiGraph from streaming handler output,
    pre-filtered to `network_type`-compatible highway values.

    Algorithm — two passes over `handler_ways`:

      Pass 1: detect intersections. A node is an intersection iff it is
        - a way endpoint (start or end of any KEPT way), OR
        - referenced by 2+ KEPT ways.
        Ways that fail `aperta_atlas.osm.is_usable(tags, network_type)`
        are skipped at this stage and never contribute to intersection
        detection or the output graph. (`network_type='all'` skips the
        highway-value filter — only universal access guards remain.)

      Pass 2: emit one edge per inter-intersection segment of each KEPT
        way. The intermediate way-vertices between two intersections
        become the edge's LineString geometry. Only intersection nodes
        ever enter the graph as actual nodes.

    Filtering at the build step (not post-build) means we never
    materialize the ~50-70% of intersection nodes that would have come
    only from non-drivable / non-bikeable ways. For Switzerland + 50 km
    buffer + `network_type='drive'`, this cuts intersections from ~6M
    to ~2M and proportionally reduces edge count + memory.

    Pops `handler_ways` as it processes so the input list shrinks in
    parallel — frees handler memory before the graph is fully built.

    Output shape (OSMnx-compatible):
      - Nodes: `x` (lon), `y` (lat), `osmid` (= node id).
      - Edges: `osmid` (source OSM way id), `geometry` (LineString of
        (lon, lat) tuples in WGS84), plus whichever of `_USEFUL_WAY_TAGS`
        was present on the source way.
      - Edge direction(s) emitted per `aperta_atlas.osm.emit_directions`
        — per-mode, respects bike-specific contraflow tags and
        pedestrian-direction overrides (not just the bare `oneway`).
      - Parallel edges (different OSM ways connecting same intersection
        pair) get distinct NetworkX edge keys.
      - `graph['crs']` = EPSG:4326.

    `ox.distance.add_edge_lengths` is called by `main` after this to
    populate `length` from each edge's `geometry`.
    """
    g = nx.MultiDiGraph()
    g.graph['crs'] = 'EPSG:4326'

    # Pass 1: detect intersections from KEPT ways only.
    node_uses: Counter = Counter()
    endpoints: set[int] = set()
    n_ways_kept = 0
    for _, tags, nodes in handler_ways:
        if not nodes or not is_usable(tags, network_type):
            continue
        n_ways_kept += 1
        for nid, _, _ in nodes:
            node_uses[nid] += 1
        endpoints.add(nodes[0][0])
        endpoints.add(nodes[-1][0])

    intersections: set[int] = {
        nid for nid, count in node_uses.items() if count >= 2
    } | endpoints
    logging.info(
        f"  → pass 1 (network_type={network_type!r}): "
        f"{n_ways_kept:,} of {len(handler_ways):,} ways kept; "
        f"{len(intersections):,} intersections "
        f"(of {len(node_uses):,} unique nodes — "
        f"{len(intersections) * 100 / max(len(node_uses), 1):.1f}% kept)")
    # node_uses + endpoints no longer needed; free before pass 2.
    del node_uses, endpoints

    # Pass 2: emit edges from KEPT ways, segmenting at intersections.
    # Pop ways off the input list so the handler's data frees as we go.
    logging.info("  → pass 2: emitting edges")
    while handler_ways:
        way_id, tags, nodes = handler_ways.pop()
        if len(nodes) < 2 or not is_usable(tags, network_type):
            continue

        emit_forward, emit_backward = emit_directions(tags, network_type)

        seg_start = 0
        n_nodes = len(nodes)
        for i in range(1, n_nodes):
            nid_i = nodes[i][0]
            # Split at intersections AND at way end (last node always
            # closes the final segment).
            if nid_i in intersections or i == n_nodes - 1:
                seg = nodes[seg_start:i + 1]
                start_id, start_lat, start_lon = seg[0]
                end_id, end_lat, end_lon = seg[-1]

                # Add endpoint nodes if not already present.
                if start_id not in g:
                    g.add_node(start_id, y=start_lat, x=start_lon, osmid=start_id)
                if end_id not in g:
                    g.add_node(end_id, y=end_lat, x=end_lon, osmid=end_id)

                # LineString of (lon, lat) — shapely uses (x, y) ordering.
                coords = [(n[2], n[1]) for n in seg]
                edge_tags = {**tags, 'osmid': way_id}

                # `cycleway_dir` / `lanes_dir`: bike infrastructure and motor-traffic lanes in
                # this edge's direction, resolved from side- and direction-specific tags while
                # the way's direction is still known; `oneway_motor`: one-way motor traffic.
                oneway_motor = 1.0 if motor_traffic_oneway(tags) else 0.0
                if emit_forward:
                    g.add_edge(
                        start_id, end_id,
                        key=g.new_edge_key(start_id, end_id),
                        geometry=LineString(coords), **edge_tags,
                        cycleway_dir=cycleway_in_direction(tags, True, left_hand_traffic),
                        lanes_dir=lanes_in_direction(tags, True), oneway_motor=oneway_motor)
                if emit_backward:
                    g.add_edge(
                        end_id, start_id,
                        key=g.new_edge_key(end_id, start_id),
                        geometry=LineString(list(reversed(coords))),
                        **edge_tags,
                        cycleway_dir=cycleway_in_direction(tags, False, left_hand_traffic),
                        lanes_dir=lanes_in_direction(tags, False), oneway_motor=oneway_motor)

                seg_start = i

    return g


def main(variant) -> None:
    context = init_context(variant)
    name = f'{variant.area_name}_{variant.mode}'

    with clipped_pbf(context, variant, name) as (pbf_path, polygon):
        # Stream-parse the clipped PBF. RAM bounded by collected highway data only.
        with step('stream parse (pyosmium handler)'):
            handler = HighwayGraphHandler()
            handler.apply_file(pbf_path, locations=True)
            n_mtb = handler.mark_mtb_routes()
        logging.info(
            f"  → collected {len(handler.ways):,} highway ways "
            f"(of {handler.n_seen:,} total ways seen); {n_mtb:,} in mountain-bike routes")

    # Build the pre-simplified, mode-filtered MultiDiGraph. The
    #    `network_type` filter is applied INSIDE the builder (at the OSM-way
    #    level in pass 1) so non-mode-relevant ways never materialize as
    #    nodes/edges in the graph.
    with step(f'build MultiDiGraph (network_type={variant.network_type!r})'):
        g = build_multidigraph(handler.ways, variant.network_type,
                               left_hand_traffic=AREAS[variant.area_name].left_hand_traffic)
    logging.info(
        f"  → mode-filtered graph: {g.number_of_nodes():,} nodes, "
        f"{g.number_of_edges():,} edges")

    # A street whose sidewalk or cycle track is mapped as its own way is in the walk / bike
    # network twice; drop the way the mode does not use, where that is safe.
    if variant.mode in ('walk', 'bike'):
        with step('drop_sidepath_ways'):
            stats = drop_sidepath_ways(g, variant.mode)
            logging.info(
                f"  → dropped {stats['ways_dropped']:,} ways ({stats['km_dropped']:,.0f} km); "
                f"kept {stats['ways_kept_no_alongside']:,} with no way alongside and put back "
                f"{stats['ways_put_back']:,} for connectivity")

    with step('largest_component'):
        before_n, before_e = g.number_of_nodes(), g.number_of_edges()
        g = ox.truncate.largest_component(g, strongly=False)
        logging.info(
            f"  → kept {g.number_of_nodes():,} / {before_n:,} nodes, "
            f"{g.number_of_edges():,} / {before_e:,} edges")

    # Per-edge lengths must exist before `collapse_degree_2_chains` runs
    # — the chain merge does length-weighted attribute aggregation.
    with step('add_edge_lengths'):
        g = ox.distance.add_edge_lengths(g)

    # Convert `bridge` and `tunnel` from OSM strings (`'yes'`, `'viaduct'`,
    # `'no'`, missing, ...) to numeric 0/1 BEFORE the chain collapse.
    # `collapse_degree_2_chains` then treats them as numeric and produces
    # a length-weighted MEAN (= true bridge/tunnel fraction ∈ [0, 1] for
    # the merged chain) rather than the length-weighted MODE it applies
    # to categoricals — which would drop the bridge information from a
    # 100m-bridge + 200m-non-bridge chain (mode would be "no").
    # `steps` likewise: `highway` becomes the chain's most common value, so a short
    # flight of stairs inside a footway chain would otherwise vanish (walk_prm
    # excludes edges with any stairs, 04_edge_weights).
    with step('numerize bridge / tunnel / steps / maxspeed tags (for length-weighted collapse)'):
        for *_, d in g.edges(keys=True, data=True):
            d['bridge'] = 1.0 if d.get('bridge') in BRIDGE_YES_VALUES else 0.0
            d['tunnel'] = 1.0 if d.get('tunnel') in TUNNEL_YES_VALUES else 0.0
            d['steps'] = 1.0 if d.get('highway') == 'steps' else 0.0
            # Speed limits as km/h numbers (`CH:rural`, `30 mph`, `50;80` …), so the chain
            # collapse averages them instead of keeping the most common text.
            d['maxspeed'] = parse_maxspeed_kmh(d.get('maxspeed'))

    # Custom topological simplification — collapses degree-2 chains
    # (false intersections left over from filter-in-build) while
    # CONCATENATING per-edge LineString geometries AND length-weighting
    # other attributes (e.g. `maxspeed` becomes a numeric mean weighted
    # by segment length; `highway` becomes the most common value by
    # length, NOT a list). OSMnx's simplify_graph would do the topology
    # right but rebuild merged-edge geometry from node x/y only AND
    # collect differing attribute values into lists — both undesirable
    # for our pipeline.
    with step('collapse_degree_2_chains'):
        before_n, before_e = g.number_of_nodes(), g.number_of_edges()
        g = collapse_degree_2_chains(g)
        logging.info(
            f"  → kept {g.number_of_nodes():,} / {before_n:,} nodes, "
            f"{g.number_of_edges():,} / {before_e:,} edges")

    # Per-mode dead-end pruning (`_PRUNE_DEAD_END_MAX_M`; 0 = none).
    if variant.prune_dead_end_max_m > 0:
        with step(f'prune_short_dead_ends (<{variant.prune_dead_end_max_m} m)'):
            before_n, before_e = g.number_of_nodes(), g.number_of_edges()
            g = prune_short_dead_ends(g, variant.prune_dead_end_max_m)
            logging.info(
                f"  → kept {g.number_of_nodes():,} / {before_n:,} nodes, "
                f"{g.number_of_edges():,} / {before_e:,} edges")

    # Mark as simplified so downstream OSMnx checks pass.
    g.graph['simplified'] = True

    with step('save graphml + properties'):
        context.create_nw(g, data_name=name, save_properties=True)
        context.create_generic(polygon, f'{name}_polygon.gpkg')

    context.close()


if __name__ == '__main__':
    variants.run(main)
