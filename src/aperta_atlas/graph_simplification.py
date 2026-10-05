"""
Graph topology simplification helpers — chain collapse and (future)
intersection consolidation.

Two related but distinct operations both reduce graph complexity:

  1. **Chain collapse** (`collapse_degree_2_chains`): removes degree-2
     interior nodes by merging the chain through them into a single
     edge. Concatenates per-edge LineString geometry instead of
     rebuilding it from node x/y (which is what
     `ox.simplification.simplify_graph` does — and what destroys shape
     detail when intersection-only pre-simplified graphs are processed).

  2. **Intersection consolidation** (planned wrapper around
     `ox.consolidate_intersections`): merges spatially-close degree-3+
     nodes that conceptually form one intersection (split-by-OSM-tag
     junctions, roundabouts). To be added when the
     `networks_consolidation.py` orchestration script is built.

Follows OSMnx's `osmnx.simplification` module precedent — both ops live
under one umbrella.

Module-level invariants:
  - Operates on NetworkX `MultiDiGraph` / `MultiGraph`.
  - Edge `geometry` attribute (shapely `LineString`) is preserved and
    extended, never overwritten by straight-line reconstruction.
  - `length` attribute is summed across merged segments; per-edge
    numeric attributes (e.g. `maxspeed`, `lanes`) get length-weighted
    means; categorical attrs get length-weighted modes. `osmid` stays
    as a list of source OSM way IDs (OSMnx convention, preserves
    provenance).
"""

from collections import Counter

import networkx as nx
from shapely.geometry import LineString

from aperta_atlas.utils import round_to_significant_figures


def _is_chain_interior(g: nx.MultiDiGraph, n) -> bool:
    """A node is interior (collapsible into a chain) iff it has exactly 2
    distinct undirected neighbors, no self-loops, and a clean
    through-pattern: either total degree 2 (one-way) or total degree 4
    (two-way). Any other configuration (branches, dead-ends, parallel
    edges, oneway↔twoway transitions) makes it a true endpoint.

    Matches `ox.simplification._is_endpoint(strict=True)` semantics.
    """
    neighbors = set(g.predecessors(n)) | set(g.successors(n))
    if n in neighbors:
        return False
    if len(neighbors) != 2:
        return False
    return g.degree(n) in (2, 4)


def _walk_chain(
    g: nx.MultiDiGraph, start, first_interior, interior: set,
) -> list:
    """Walk from `start` through `first_interior` and onward through
    interior nodes until reaching an endpoint. Returns the full node path
    [start, ..., endpoint].
    """
    path = [start, first_interior]
    prev, cur = start, first_interior
    while cur in interior:
        # Interior node has exactly 2 distinct neighbors; one is prev.
        nbrs = (set(g.successors(cur)) | set(g.predecessors(cur))) - {prev}
        if len(nbrs) != 1:
            break
        nxt = next(iter(nbrs))
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _aggregate_attr(values_lengths: list[tuple]):
    """Length-weighted aggregation of one attribute across a chain of edges.

    `values_lengths` is a list of `(value, length)` pairs, one per chain
    edge. Values that are `None` are dropped.

    Two reductions, both length-weighted, chosen automatically by the
    type of the input values:

      - **Numeric input** → length-weighted MEAN, rounded to 2
        significant figures. Produces a continuous summary value.
        Catches `maxspeed`, `lanes`, and similar numeric-but-stringified
        OSM tags, plus `bridge` and `tunnel` (which `networks_from_pbf`
        deliberately numerizes from 0/1 strings BEFORE the chain
        collapse, so the mean produces a true length-weighted fraction
        in [0, 1] rather than the lossy mode).
      - **Non-numeric (categorical) input** → length-weighted MODE
        (most common value by total cumulative length; ties broken
        arbitrarily by `Counter`). Picks a representative dominant
        value. Catches `highway`, `oneway`, `cycleway`,
        `cycleway:left/right/both`, `cyclestreet`, `service`,
        `junction`, `access`, `ref`, etc.

    Design rule of thumb: if you'd want a continuous summary of a
    categorical attribute (e.g. "what fraction of this merged edge is
    a bridge?"), numerize it to 0/1 upstream — the numeric mean does
    the right thing for you. Mode is for "what flavour does this
    merged edge predominantly have?" — used when the categories carry
    distinct downstream semantics that can't be averaged (e.g. you'd
    rather know the edge is mostly a `cycleway=track` than the mean
    of {`track`, `lane`}).

    Deviation from OSMnx's `simplify_graph` convention (which collects
    differing values into a list per edge) — we collapse to a single
    representative value here, which is what downstream analysis
    actually wants and removes the need for downstream after-the-fact
    list cleanup.
    """
    pairs = [(v, w) for v, w in values_lengths if v is not None]
    if not pairs:
        return None

    # Try numeric (length-weighted mean). Round to 2 significant figures
    # for cleaner downstream display.
    try:
        numeric = [(float(v), w) for v, w in pairs]
        total_w = sum(w for _, w in numeric)
        mean = sum(v * w for v, w in numeric) / total_w if total_w > 0 else numeric[0][0]
        return round_to_significant_figures(mean, 2)
    except (ValueError, TypeError):
        pass

    # Categorical (length-weighted mode).
    counter: Counter = Counter()
    for v, w in pairs:
        counter[v] += w
    return counter.most_common(1)[0][0]


def _merge_chain(g: nx.MultiDiGraph, path: list) -> dict | None:
    """Build merged edge attrs for the directed chain
    `path[0] → path[1] → ... → path[-1]`. Returns None if any forward
    edge is missing (e.g. this direction not traversable for a one-way
    chain we're walking the wrong way).

    Geometry: concatenates per-edge LineStrings (dropping duplicate
    join-points). Length: sum of per-edge lengths. Osmid: list of source
    way IDs (OSMnx convention — preserves provenance for downstream).
    All other attributes: length-weighted via `_aggregate_attr` (numeric
    mean for `maxspeed`/`lanes`/etc; mode for `highway`/categorical).

    Requires per-edge `length` to be populated before this runs (call
    `ox.distance.add_edge_lengths` first).
    """
    edge_records: list[tuple[dict, float]] = []
    for u, v in zip(path[:-1], path[1:]):
        if not g.has_edge(u, v):
            return None
        # Interior chain edges have a single key per (u, v); first key wins.
        keys = list(g[u][v].keys())
        edge_attrs = g[u][v][keys[0]]
        edge_length = float(edge_attrs.get('length', 0.0))
        edge_records.append((edge_attrs, edge_length))

    if not edge_records:
        return None

    # Geometry: concatenate LineStrings.
    coords: list = []
    for attrs, _ in edge_records:
        geom = attrs.get('geometry')
        if geom is not None:
            if not coords:
                coords.extend(geom.coords)
            else:
                coords.extend(list(geom.coords)[1:])

    # Osmid: flatten into a list of source way ids (single value if one).
    osmids: list = []
    for attrs, _ in edge_records:
        oid = attrs.get('osmid')
        if isinstance(oid, list):
            osmids.extend(oid)
        elif oid is not None:
            osmids.append(oid)

    # Length: sum source lengths.
    total_length = sum(length for _, length in edge_records)

    # All other attribute keys: length-weighted aggregation.
    merged: dict = {}
    all_keys = set()
    for attrs, _ in edge_records:
        all_keys.update(attrs.keys())
    all_keys -= {'geometry', 'osmid', 'length'}
    for key in all_keys:
        result = _aggregate_attr([(attrs.get(key), length) for attrs, length in edge_records])
        if result is not None:
            merged[key] = result

    if coords:
        merged['geometry'] = LineString(coords)
    if osmids:
        merged['osmid'] = osmids if len(osmids) > 1 else osmids[0]
    if total_length > 0:
        merged['length'] = total_length

    return merged


def simplify_edge_geometries(
    g: nx.MultiDiGraph, tolerance: float,
) -> None:
    """Apply Douglas-Peucker simplification to each edge's LineString
    `geometry`. Mutates `g` in place.

    `tolerance` is in the graph CRS's units — meters for a projected
    graph, degrees for WGS84. Call this while the graph is in a metric
    CRS so `tolerance` is meaningful as a sub-meter threshold (e.g.
    0.5 m drops near-collinear vertices that contribute essentially
    nothing to the shape).

    `preserve_topology=True` ensures the simplified line stays within
    `tolerance` of the original AND doesn't self-intersect. Endpoint
    coordinates are kept exactly (DP guarantees the first/last
    vertices), so simplified edges still meet their node positions.

    Length-recomputation should follow this (e.g. via
    `clean_consolidated_edges` in aperta) since the simplified
    geometry is slightly shorter than the original.
    """
    for _, _, _, d in g.edges(keys=True, data=True):
        geom = d.get('geometry')
        if geom is not None:
            d['geometry'] = geom.simplify(tolerance, preserve_topology=True)


def round_coords(g: nx.MultiDiGraph, decimals: int) -> None:
    """Round node `x` / `y` and edge `geometry` coords to `decimals`
    decimal places. Mutates `g` in place.

    Intended use: remove precision noise from `project_graph` round-
    trips, which can leave coords with 15-17 trailing digits. Text
    formats like graphml store those digits as bytes, so trimming
    pure-precision-noise saves real disk space.

    For WGS84 (degrees):
      - 7 decimals ≈ 1 cm precision (10^-7 deg × ~111 km/deg ≈ 1 cm)
      - 8 decimals ≈ 1 mm precision

    For metric CRSs (meters), `decimals` just rounds to fractional
    meters (e.g. `decimals=2` = cm precision).

    Endpoint coordinates of edge geometries are rounded the same way
    as their incident nodes, so endpoint-node alignment is preserved
    after rounding (as long as both go through this function — which
    they do here).
    """
    for n in g.nodes:
        g.nodes[n]['x'] = round(g.nodes[n]['x'], decimals)
        g.nodes[n]['y'] = round(g.nodes[n]['y'], decimals)
    for _, _, _, d in g.edges(keys=True, data=True):
        geom = d.get('geometry')
        if geom is not None:
            coords = [(round(x, decimals), round(y, decimals))
                      for x, y in geom.coords]
            d['geometry'] = LineString(coords)


def prune_short_dead_ends(
    g: nx.MultiDiGraph, max_length_m: float,
) -> nx.MultiDiGraph:
    """Single-pass: drop dead-end branches whose connecting edge is
    shorter than `max_length_m`, then run `collapse_degree_2_chains`
    once to merge any junctions that lost an arm.

    A "dead-end" is a node with exactly one distinct undirected
    neighbor. The branch length comes from the `length` attribute on
    the connecting edge(s) (for two-way streets the forward and
    reverse edges have equal length; for one-way it's the only one).

    **Single-pass is intentional** (do NOT iterate). Iterating creates
    a cascade through small neighborhoods: a junction at the entry to
    a small cul-de-sac neighborhood with three short driveway-stubs
    would, after pass 1's stub-pruning, itself become a degree-1 node
    with a short entry road. Pass 2 would then prune the neighborhood
    entry node, then pass 3 the next stub root, etc. — recursively
    amputating entire small areas. Single-pass gives a hard bound:
    after pruning, no destination is ever more than `max_length_m`
    farther from its (now-pruned) "true end" than it would have been
    in the input. Iterative pruning has no such bound.

    Use case: routing networks inherit dense residential / service
    driveway micro-stubs that add nodes without contributing to
    through-routing. Single-pass pruning reduces graph size by
    ~15–25% (compared to ~40% for unbounded iteration).

    **The actual snap-distance penalty is even smaller than
    `max_length_m`** because of how aperta's snap pipeline handles
    geometry: a pruned-leaf's parent junction collapses into the
    through-edge, preserving its coordinates as a LineString vertex,
    and `insert_projected_nodes` (aperta) projects each destination
    onto the through-edge at snap time — landing essentially at the
    original parent-junction position. Snap error is bounded by
    LineString vertex spacing, not branch length.

    Suitable thresholds:

      - Bike: ~100 m. Cyclists park near street.
      - Car: 50–100 m. Safe IF an upstream filter has already
        removed driveways/parking-aisle service roads (see
        `aperta_atlas.osm._DRIVE_ALLOWED_SERVICE_SUBTYPES`).
      - Walk: 20–30 m (catches micro-stubs without affecting
        door-accurate snap), or 0 to skip.

    Pass `max_length_m=0` to skip pruning entirely.

    Requires per-edge `length` to be populated before this runs.
    """
    if max_length_m <= 0:
        return g

    # Identify all dead-ends in the CURRENT graph (before any pruning).
    # We compute the full list up front, then drop in one shot — never
    # re-evaluate dead-end status after pruning (see docstring on why
    # iteration is intentionally avoided).
    dead_ends: list = []
    for n in g.nodes:
        neighbors = set(g.predecessors(n)) | set(g.successors(n))
        if len(neighbors) != 1 or n in neighbors:
            continue
        nbr = next(iter(neighbors))
        lengths: list[float] = []
        for u, v in ((n, nbr), (nbr, n)):
            if g.has_edge(u, v):
                for k in g[u][v]:
                    lengths.append(float(g[u][v][k].get('length', float('inf'))))
        if lengths and min(lengths) < max_length_m:
            dead_ends.append(n)

    g.remove_nodes_from(dead_ends)

    # Drop any isolated nodes (degree-0). Can happen if a junction's
    # entire spider of short spokes was in the prune set above.
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    g.remove_nodes_from(isolated)

    # Collapse degree-2 chains formed by junctions that lost an arm
    # (e.g. a 3-way junction with one pruned driveway becomes a degree-2
    # through-point and should be merged into the surviving edge).
    g = collapse_degree_2_chains(g)

    return g


def collapse_degree_2_chains(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Collapse degree-2 chains in `g`, concatenating per-edge LineString
    geometries (rather than rebuilding from node x/y coords as
    `ox.simplification.simplify_graph` does).

    Needed because building a pre-simplified graph (only intersections
    as nodes, shape vertices baked into edge LineStrings) plus a
    per-mode filter leaves some "false intersection" nodes: OSM way
    boundaries (e.g. `maxspeed` changes mid-road, or 3-way junctions
    where 2 of 3 ways aren't mode-relevant) that look like intersections
    at build time but resolve to degree-2 through-nodes after filtering.
    This pass cleans them up.

    Compared to `ox.simplification.simplify_graph`: same topology
    result, but two cleanness wins:
      - **Geometry**: concatenates the per-edge multi-vertex LineStrings
        baked in upstream (OSMnx instead rebuilds merged geometry from
        chain node x/y, replacing shape detail with a straight line
        between chain endpoints).
      - **Attributes**: length-weighted aggregation via `_aggregate_attr`
        — numeric attrs (`maxspeed`, `lanes`, …) become a length-weighted
        mean across the chain; categorical attrs (`highway`, `oneway`,
        `bridge`, …) become the length-weighted mode. OSMnx instead
        collects differing values into a list per attribute, leaving
        downstream code to do after-the-fact list cleanup.

    Requires per-edge `length` to be populated before this runs (call
    `ox.distance.add_edge_lengths` first).
    """
    interior = {n for n in g.nodes if _is_chain_interior(g, n)}

    # Walk every chain exactly once. Use undirected (start, end) tuples
    # to dedupe; chains starting in each direction visit the same nodes.
    visited_pairs: set[tuple] = set()
    nodes_to_remove: set = set()

    for start in (set(g.nodes) - interior):
        first_steps = (set(g.successors(start)) | set(g.predecessors(start)))
        for first_step in list(first_steps):
            if first_step not in interior:
                continue
            path = _walk_chain(g, start, first_step, interior)
            end = path[-1]
            pair = tuple(sorted((start, end))) if start != end else (start,)
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)

            forward = _merge_chain(g, path)
            reverse = _merge_chain(g, list(reversed(path)))

            if forward is not None:
                g.add_edge(start, end, key=g.new_edge_key(start, end), **forward)
            if reverse is not None:
                g.add_edge(end, start, key=g.new_edge_key(end, start), **reverse)

            nodes_to_remove.update(path[1:-1])

    g.remove_nodes_from(nodes_to_remove)
    return g
