"""
Build OSM road / cycling / walking networks from a locally clipped PBF using a
pyosmium streaming handler. Streams the PBF and pre-simplifies during the build
(only intersection nodes enter the NetworkX graph; intermediate way-vertices
go into edge LineString geometry), keeping peak RAM bounded by output graph
size — suitable for country-scale extracts.

Pipeline, per variant: per-variant osmium extract → pyosmium stream →
pre-simplified MultiDiGraph build (mode-filtered at way level) → OSMnx
`largest_component` → `collapse_degree_2_chains` (our own geometry-preserving
simplifier) → `add_edge_lengths` → save graphml + polygon.

Inputs (PUBLIC, under raw/global/osm/):
    <pbf_name>-latest.osm.pbf       # output of `clip_pbf.py`

Outputs (PUBLIC, under preparation/world/osm/):
    nw/<area_name>_<mode>.graphml          # OSM network skeleton (WGS84)
    <area_name>_<mode>_polygon.gpkg        # buffered AOI polygon (WGS84)

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_<mode>` variant per (area, mode) combination —
typically 3 modes × N cases.

Requires `osmium-tool` (CLI) AND `pyosmium` (Python):
    conda install -c conda-forge osmium-tool pyosmium

Run after `clip_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.networks_from_pbf
Single variant:
    python -m preparation.world.osm.networks_from_pbf \\
        --variant switzerland_car
    python -m preparation.world.osm.networks_from_pbf \\
        --variant bern_walk
"""

import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import networkx as nx
import osmium
import osmnx as ox
from shapely.geometry import LineString

from aperta_atlas.context import init_context, Storage
from aperta_atlas.graph_simplification import (
    collapse_degree_2_chains, prune_short_dead_ends,
)
from aperta_atlas.osm import (
    BRIDGE_YES_VALUES,
    NEEDED_TAGS as _OSM_NEEDED_TAGS,
    TUNNEL_YES_VALUES,
    emit_directions,
    is_usable,
)
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS
from preparation.world.common import buffered_place_polygon, write_poly_file


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
)
_USEFUL_WAY_TAGS: tuple[str, ...] = _OSM_NEEDED_TAGS + _AUXILIARY_WAY_TAGS


class HighwayGraphHandler(osmium.SimpleHandler):
    """Streaming pyosmium handler that collects only highway ways and their
    node geometry. Bounded RAM regardless of input PBF size.

    `locations=True` (passed to `apply_file`) enables pyosmium's
    `NodeLocationsForWays` cache, which resolves way-node geometry on
    the fly in C++ memory. Without it, `n.location` on way nodes is
    unresolved.

    Usage:
        handler = HighwayGraphHandler()
        handler.apply_file(pbf_path, locations=True)
        g = build_multidigraph(handler.ways, network_type='drive')
    """

    def __init__(self) -> None:
        super().__init__()
        # (osm_way_id, useful_tags_dict, [(node_id, lat, lon), ...])
        self.ways: list[tuple[int, dict, list[tuple[int, float, float]]]] = []
        self.n_seen = 0

    def way(self, w) -> None:
        self.n_seen += 1
        if 'highway' not in w.tags:
            return
        tags = {k: w.tags[k] for k in _USEFUL_WAY_TAGS if k in w.tags}
        nodes: list[tuple[int, float, float]] = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            nodes.append((n.ref, n.location.lat, n.location.lon))
        if len(nodes) >= 2:
            self.ways.append((w.id, tags, nodes))


def build_multidigraph(handler_ways, network_type: str) -> nx.MultiDiGraph:
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

    # Per-mode usability filter (highway-value rules + universal access
    # + mode-specific access). Single source of truth in `aperta_atlas.osm`.
    def _way_kept(tags: dict) -> bool:
        return is_usable(tags, network_type)

    # Pass 1: detect intersections from KEPT ways only.
    node_uses: Counter = Counter()
    endpoints: set[int] = set()
    n_ways_kept = 0
    for _, tags, nodes in handler_ways:
        if not nodes or not _way_kept(tags):
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
        if len(nodes) < 2 or not _way_kept(tags):
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

                if emit_forward:
                    g.add_edge(
                        start_id, end_id,
                        key=g.new_edge_key(start_id, end_id),
                        geometry=LineString(coords), **edge_tags)
                if emit_backward:
                    g.add_edge(
                        end_id, start_id,
                        key=g.new_edge_key(end_id, start_id),
                        geometry=LineString(list(reversed(coords))),
                        **edge_tags)

                seg_start = i

    return g


def main(variant) -> None:
    if shutil.which('osmium') is None:
        raise RuntimeError(
            "osmium-tool not found on PATH. Install via "
            "`conda install -c conda-forge osmium-tool`.")

    context = init_context(variant)
    name = f'{variant.area_name}_{variant.mode}'

    source_pbf = context.raw_path(Storage.PUBLIC, f'global/osm/{variant.pbf_name}-latest.osm.pbf')
    if not os.path.exists(source_pbf):
        raise RuntimeError(
            f"Source PBF not found: {source_pbf}\n"
            f"Run `python -m preparation.world.osm.clip_pbf "
            f"--variant {variant.area_name}_buffered` first.")

    with step('buffered_place_polygon'):
        polygon = buffered_place_polygon(variant.place, variant.buffer)

    with tempfile.TemporaryDirectory() as tmpdir:
        poly_path = os.path.join(tmpdir, f'{name}.poly')
        clipped_pbf = os.path.join(tmpdir, f'{name}_clipped.osm.pbf')

        write_poly_file(polygon, poly_path)

        # 1. Per-variant spatial clip.
        with step('osmium extract (per-variant clip)'):
            subprocess.run(
                ['osmium', 'extract', '--polygon', poly_path,
                 source_pbf, '-o', clipped_pbf, '--overwrite'],
                check=True,
            )
            clip_mb = os.path.getsize(clipped_pbf) / (1024 * 1024)
            logging.info(f"  → clipped PBF: {clip_mb:,.1f} MB")

        # 2. Stream-parse the clipped PBF. RAM bounded by collected highway
        #    data only — no XML expansion.
        with step('stream parse (pyosmium handler)'):
            handler = HighwayGraphHandler()
            handler.apply_file(clipped_pbf, locations=True)
        logging.info(
            f"  → collected {len(handler.ways):,} highway ways "
            f"(of {handler.n_seen:,} total ways seen)")

    # 3. Build the pre-simplified, mode-filtered MultiDiGraph. The
    #    `network_type` filter is applied INSIDE the builder (at the OSM-way
    #    level in pass 1) so non-mode-relevant ways never materialize as
    #    nodes/edges in the graph.
    with step(f'build MultiDiGraph (network_type={variant.network_type!r})'):
        g = build_multidigraph(handler.ways, variant.network_type)
    logging.info(
        f"  → mode-filtered graph: {g.number_of_nodes():,} nodes, "
        f"{g.number_of_edges():,} edges")

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
    with step('numerize bridge / tunnel tags (for length-weighted collapse)'):
        for *_, d in g.edges(keys=True, data=True):
            d['bridge'] = 1.0 if d.get('bridge') in BRIDGE_YES_VALUES else 0.0
            d['tunnel'] = 1.0 if d.get('tunnel') in TUNNEL_YES_VALUES else 0.0

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

    # Optional per-mode dead-end pruning. For bike: typically 100 m
    # (cyclists park near street, snap error within that range is fine).
    # For walk/drive: 0 means skip (door-accurate snap matters).
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
