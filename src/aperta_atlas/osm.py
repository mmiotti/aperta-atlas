"""
OSM way classification rules per network type (walk / bike / drive / all),
plus OSM-specific post-consolidation helpers.

Single source of truth for "is this OSM way usable for routing in this
mode" — used at two points in the pipeline:

  1. Network EXTRACTION (build time): filter the incoming OSM way stream
     to only mode-usable ways. Consumed by
     `preparation/world/osm/networks_from_pbf.py` via `is_usable` (or
     the `filter_ways` generator).

  2. Network POST-PROCESSING (cost assignment): mark unusable edges
     with `cost_excluded_<network_type>=True`, leaving them in the
     graph for visualization / multi-modal analysis but ensuring
     downstream cost-assignment can skip them. Use case: unifying
     API-derived and PBF-derived graphs to the same usability semantics
     (the API path uses OSMnx's filter, which differs from ours — e.g.
     OSMnx's walk filter drops cycleways even when `foot=designated`).

Plus the OSM-specific helpers that clean up a graph after
`osmnx.consolidate_intersections`: `clean_consolidated_edges` and its
component `lanes_per_direction`. These read OSM tag conventions
(`lanes` is total across both directions, `oneway` toggles directional
attribution) so they live in aperta-atlas alongside the rest of the
OSM-aware code rather than in the algorithm library.

**Philosophy: lean permissive.** Exclude only when ≥99.99% certain the
mode is NOT allowed. Let OSM's explicit access tags (`foot=no`,
`bicycle=no`, `access=private`, `motor_vehicle=no`, etc.) catch edge
cases. False exclusions cause topology cutoffs that are very hard to
recover from downstream (the Cambridge-MA cycleway-as-walk-connector
issue is the canonical example: OSMnx's walk filter drops ALL cycleways,
breaking the pedestrian network where bike paths bridge two walkable
areas).

Four rule layers (ALL must pass for a way to be usable):

  1. **Universal access**: ways tagged `access=private`, `access=no`, or
     `area=yes` excluded for any mode. First two are explicit "no public
     access"; third is a polygon (plaza, parking lot) that isn't a
     linear route.

  2. **Universal highway-value exclusion**: highway values that aren't
     routes at all for any mode (`abandoned`, `construction`, `proposed`,
     etc.) — see `_UNIVERSAL_HIGHWAY_EXCLUDE`.

  3. **Mode-specific access exclusions**: explicit "this mode excluded":
       - walk: `foot=no`; or `motorroad=yes` (motorway-equivalent)
       - bike: `bicycle=no`; or `motorroad=yes`
       - drive: `motor_vehicle=no` OR `motorcar=no`

  4. **Mode-specific highway-value rule** (`_HIGHWAY_RULES`):
     per-mode whitelist (`keep`) or blacklist (`exclude`) of
     `highway=...` values. Walk uses a narrow blacklist (motor-only +
     bus-only); bike a slightly wider blacklist (+ steps, escalator);
     drive a positive whitelist of car-drivable categories incl.
     `service`.
"""


# 2. Universal highway-value exclusions — apply to every network_type.
# Match OSMnx's convention in `_overpass.py`. These represent things
# that aren't usable routes for any mode (planned/abandoned/demolished
# roads, transit platforms, motorway service areas, race tracks, etc.).
_UNIVERSAL_HIGHWAY_EXCLUDE: frozenset[str] = frozenset({
    'abandoned',     # no longer exists
    'construction',  # under construction, not currently usable
    'demolished',    # OSM tag variant of razed
    'no',            # explicit "not a highway" (rare; sometimes used)
    'planned',       # planned but not built
    'platform',      # transit platform — a feature, not a route
    'proposed',      # proposed but not built
    'raceway',       # closed-course racing, not public routing
    'razed',         # demolished
    'rest_area',     # motorway rest area — feature, not route
    'services',      # motorway services area — feature, not route
})


# 4. Mode-specific highway-value rules. Each rule either
# `{'keep': set}` (whitelist) or `{'exclude': set}` (blacklist). The
# universal exclusion (layer 2) applies BEFORE this rule.
_HIGHWAY_RULES: dict[str, dict | None] = {
    # 'all': no value filter (only layers 1-2 apply). For when you want
    # every routable highway regardless of mode.
    'all': None,

    # WALK: motor-only excluded, plus bus-only. Notably KEPT:
    #   - trunk / trunk_link: region-dependent walkability (CH/UK/DE
    #     trunk roads typically allow pedestrians; US "trunk" sometimes
    #     bans them via `motorroad=yes` which layer 3 catches).
    #   - cycleway: often shared with pedestrians (Cambridge-MA case).
    #     `foot=no` catches truly bike-only segments.
    #   - footway / path / steps / pedestrian / service / corridor /
    #     bridleway: walkable.
    'walk': {'exclude': {
        'motorway', 'motorway_link',
        'bus_guideway', 'busway',
    }},

    # BIKE: motor-only + bus-only excluded, plus categories where
    # cycling is physically impossible (steps, escalator). Notably KEPT:
    #   - footway / pedestrian / corridor: bikes often allowed
    #     (`bicycle=designated`, `bicycle=yes` common). `bicycle=no`
    #     catches true bans.
    #   - path / track / bridleway: typically bikeable.
    'bike': {'exclude': {
        'motorway', 'motorway_link', 'trunk', 'trunk_link',
        'bus_guideway', 'busway',
        'steps', 'escalator',
    }},

    # DRIVE: positive whitelist. INCLUDES `service` (alleys, parking
    # aisles, driveways) — cars universally drive these;
    # `access=private` catches the truly-private ones.
    'drive': {'keep': {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'unclassified', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
        'tertiary_link',
        'living_street', 'service',
    }},
}


# Drive-mode sub-type filter for `highway=service`. The full `service`
# category includes many sub-types — only the ones cars can publicly
# route on are kept. Excluded:
#   - `driveway`        — private property access
#   - `parking_aisle`   — internal parking-lot routing
#   - `emergency_access`— restricted to emergency vehicles
#   - `bus`             — bus-only service road
#   - untagged          — no `service=*` sub-tag; often residential
#                         driveways without explicit tagging; excluded
#                         conservatively (revisit if rural routing
#                         needs them)
# Walk and bike include ALL service sub-types (pedestrians and cyclists
# do use driveways, alleys, parking aisles — handled by the universal
# keep behavior for non-drive modes).
_DRIVE_ALLOWED_SERVICE_SUBTYPES: frozenset[str] = frozenset({
    'alley',          # narrow road between buildings, public access
    'drive-through',  # restaurant / business drive-through access
})


# OSM tag keys that `is_usable` reads. Callers building OSM way / edge
# dicts (e.g. `HighwayGraphHandler` in `networks_from_pbf.py`) must
# include AT LEAST these keys when present on the source way; without
# them the rules can't apply. Auxiliary tags (oneway, lanes, maxspeed,
# etc.) are the caller's separate concern.
NEEDED_TAGS: tuple[str, ...] = (
    'highway',
    'service',
    'access',
    'area',
    'foot',
    'bicycle',
    'motor_vehicle',
    'motorcar',
    'motorroad',
)


def is_usable(tags: dict, network_type: str) -> bool:
    """True iff a way (or edge) with these OSM tags is usable for
    routing in `network_type`. Applies all four rule layers — see
    module docstring.

    Tag values are compared with string `==` / `in`, matching OSM's
    convention that tag values are strings. The merged-edge case
    (where `highway` may be list-valued after our pre-simplification)
    is handled for `highway` only — other tags shouldn't be
    list-valued by construction.

    Raises `ValueError` for unknown `network_type`.
    """
    if network_type not in _HIGHWAY_RULES:
        raise ValueError(
            f"Unknown network_type {network_type!r}. Known: "
            f"{sorted(_HIGHWAY_RULES)}.")

    # Layer 1: universal access exclusions.
    if tags.get('access') in ('private', 'no'):
        return False
    if tags.get('area') == 'yes':
        return False

    # Layer 3: mode-specific access exclusions.
    if network_type == 'walk':
        if tags.get('foot') == 'no':
            return False
        # motorroad=yes implies motorway-equivalent (limited access,
        # motor vehicles only) — common on US trunks tagged as motorroad.
        if tags.get('motorroad') == 'yes':
            return False
    elif network_type == 'bike':
        if tags.get('bicycle') == 'no':
            return False
        if tags.get('motorroad') == 'yes':
            return False
    elif network_type == 'drive':
        if tags.get('motor_vehicle') == 'no' or tags.get('motorcar') == 'no':
            return False

    # Layers 2 + 4: highway-value rule (universal first, then mode-specific).
    hw = tags.get('highway')
    if hw is None:
        return False
    rule = _HIGHWAY_RULES[network_type]

    # Merged-edge case: highway may be list-valued after
    # collapse_degree_2_chains' length-weighted aggregation. Universal
    # exclude filters per-element; way is usable if ANY of its remaining
    # highway values passes the mode rule.
    if isinstance(hw, list):
        hw = [h for h in hw if h not in _UNIVERSAL_HIGHWAY_EXCLUDE]
        if not hw:
            return False
        if rule is None:
            return True
        if 'keep' in rule:
            return any(h in rule['keep'] for h in hw)
        return any(h not in rule['exclude'] for h in hw)

    if hw in _UNIVERSAL_HIGHWAY_EXCLUDE:
        return False
    # Drive-mode sub-type refinement for `highway=service`: keep only
    # publicly-routable sub-types (alley, drive-through), drop private
    # / internal / restricted ones — see `_DRIVE_ALLOWED_SERVICE_SUBTYPES`.
    if (network_type == 'drive' and hw == 'service'
            and tags.get('service') not in _DRIVE_ALLOWED_SERVICE_SUBTYPES):
        return False
    if rule is None:
        return True
    if 'keep' in rule:
        return hw in rule['keep']
    return hw not in rule['exclude']


def emit_directions(tags: dict, network_type: str) -> tuple[bool, bool]:
    """Decide whether an OSM way should produce a forward edge, a
    backward edge, or both, for the given `network_type`. Returns
    `(emit_forward, emit_backward)`.

    Per-mode logic:

      walk:  pedestrians ignore car one-ways (sidewalks on both sides;
             contraflow walking is universal). Only respect an explicit
             `oneway:foot` override.

      bike:  explicit overrides first (`oneway:bicycle=no` or
             `cycleway*=opposite*` → contraflow allowed). Otherwise
             follow the regular `oneway` tag — bike defaults to
             matching car directionality (Swiss federal convention:
             contraflow cycling requires signage; tagger should set
             `oneway:bicycle=no` where it's allowed).

      drive / all: follow OSM `oneway` directly.

    Asymmetric per-direction costs (uphill/downhill penalties, traffic
    flow direction, etc.) require edges in BOTH directions to be
    distinguishable — that's why undirected routing is wrong for walk
    and bike. Hence this function emits both directions for walk
    almost-always, and for bike whenever contraflow tags allow it.
    """
    if network_type == 'walk':
        foot_ow = tags.get('oneway:foot')
        if foot_ow == 'yes':
            return (True, False)
        if foot_ow == '-1':
            return (False, True)
        return (True, True)

    if network_type == 'bike':
        # Explicit bike-direction override wins.
        bike_ow = tags.get('oneway:bicycle')
        if bike_ow == 'no':
            return (True, True)
        if bike_ow == 'yes':
            return (True, False)
        if bike_ow == '-1':
            return (False, True)
        # cycleway*=opposite* historically means contraflow cycling
        # is allowed on this car-one-way street.
        for key in ('cycleway', 'cycleway:left',
                    'cycleway:right', 'cycleway:both'):
            val = tags.get(key, '')
            if isinstance(val, str) and val.startswith('opposite'):
                return (True, True)
        # Fall through to general oneway logic.

    # drive, all, bike (without overrides): follow OSM `oneway` directly.
    ow = tags.get('oneway')
    if ow == '-1':
        return (False, True)
    if ow in ('yes', 'true', '1'):
        return (True, False)
    return (True, True)


def filter_ways(ways_iter, network_type: str):
    """Generator yielding only `(way_id, tags, nodes)` tuples whose
    `tags` pass `is_usable(tags, network_type)`.
    """
    for way_id, tags, nodes in ways_iter:
        if is_usable(tags, network_type):
            yield (way_id, tags, nodes)


def mark_unusable_edges(
    graph, network_type: str, *, attr_key: str | None = None,
) -> int:
    """Set a boolean `cost_excluded_<network_type>` attribute (or
    `attr_key` if given) on every edge in `graph`: True for edges that
    fail `is_usable` in this network_type, False otherwise. Returns
    the count of edges marked True.

    Used by downstream cost-assignment code: edges marked True should
    receive infinite cost so routing won't traverse them, but the edge
    stays in the graph for visualization / multi-modal analysis.

    Also useful for post-processing an API-derived graph (built with
    OSMnx's filter conventions) to match the PBF-extraction
    semantics — call once per mode to flag any mismatches.
    """
    key = attr_key if attr_key is not None else f'cost_excluded_{network_type}'
    n_unusable = 0
    for _, _, _, d in graph.edges(keys=True, data=True):
        unusable = not is_usable(d, network_type)
        d[key] = unusable
        if unusable:
            n_unusable += 1
    return n_unusable


# -----------------------------------------------------------------------
# Post-consolidation edge cleanup (OSM-specific). Moved from
# `aperta.network_processing` 2026-06-05: the `lanes`/`oneway` semantics
# these helpers depend on are OSM conventions, so they belong in the
# OSM-aware aperta-atlas layer rather than the algorithm library.
# -----------------------------------------------------------------------


def _mean_numeric(values: list):
    """Mean over values coercible to float; first value as fallback if none coerce.

    OSM `lanes` / `maxspeed` come through as strings (sometimes numeric like
    `'50'`, sometimes with units / labels like `'50 mph'` or `'RU:urban'`).
    Coercible values are averaged; non-coercible are skipped. If nothing
    parses, returns the first raw value (preserves a sensible default rather
    than producing `NaN`).
    """
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if nums:
        return sum(nums) / len(nums)
    return values[0] if values else None


def _highest_rank_highway(values: list) -> str:
    """Pick the highest-rank `highway` tag value from a list.

    After `osmnx.consolidate_intersections`, edges built from multiple
    source ways have `highway` as a *list* of strings. We collapse to
    the most *major* value via `OSM_HIGHWAY_RANKS` (motorway > trunk >
    primary > … > unclassified) rather than silently picking the first
    element — the latter is what e.g. OSMnx's `add_edge_speeds` does
    internally, and is not principled when the merged edges differ in
    road class.
    """
    ranks = [OSM_HIGHWAY_RANKS.get(v, -1) for v in values]
    return values[ranks.index(max(ranks))]


# OSM tag values that map to "bridge present" (likewise for tunnels).
# `'no'`, `None`, missing, anything else → 0. Public because
# `networks_from_pbf.py` uses them to convert string OSM tags to
# numeric 0/1 BEFORE `collapse_degree_2_chains` — that way the chain
# merger's length-weighted numeric mean produces a true length-weighted
# bridge fraction (instead of the length-weighted MODE that lossy
# categoricals get, which would drop the bridge information from a
# 100m-bridge + 200m-non-bridge chain).
BRIDGE_YES_VALUES = {
    'yes', 'viaduct', 'aqueduct', 'cantilever', 'suspension',
    'covered', 'truss', 'movable',
}
TUNNEL_YES_VALUES = {
    'yes', 'building_passage', 'culvert', 'avalanche_protector',
}


# Default edge-attribute aggregators applied to LIST-VALUED edge attrs
# post-consolidation. `lanes` and `maxspeed` get numeric-mean so merged
# edges expose single values; `highway` collapses to the highest-rank
# string. `bridge` and `tunnel` arrive as floats from
# `collapse_degree_2_chains` (set numeric by `networks_from_pbf.py`); the
# rare list case osmnx produces for parallel-edge merges is collapsed by
# `_mean_numeric` — which is length-weighted in the typical case
# (consolidate tolerance ~10 m → parallel sub-edges have similar
# lengths). Non-list-valued attrs pass through untouched (the per-edge
# loop in `clean_consolidated_edges` only applies the aggregator when
# it sees `isinstance(d.get(attr), list)`).
#
# `length` deliberately not here: OSMnx 2.x sums it across merged edges,
# but the merged edge has a single geometry whose actual length is
# *smaller* than that sum (parallel paths collapse to one). We recompute
# `length` from `geometry.length` post-consolidation in metric units.
_DEFAULT_EDGE_ATTR_AGGS = {
    "lanes": _mean_numeric,
    "maxspeed": _mean_numeric,
    "highway": _highest_rank_highway,
    "bridge": _mean_numeric,
    "tunnel": _mean_numeric,
}


def _parse_lanes(raw) -> float | None:
    """OSM `lanes` is messy — string, list, missing, NaN. Returns float or None.

    NaN is treated as missing (returns None) — pandas/numpy give us
    `float('nan')` for unset CSV cells, which `float()` accepts without
    raising. Returning the NaN through would propagate as `nan / 2.0`
    in `lanes_per_direction` and produce NaN output instead of the
    intended 1.0 default.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN check — NaN is the only float that's not equal to itself
        return None
    return v


def lanes_per_direction(edge_data: dict) -> float:
    """Per-direction lane count for a directed edge. OSM-specific: reads the
    OSM `lanes` and `oneway` tag conventions.

    OSM's `lanes` tag is the **total** lane count across both directions on
    two-way roads, and OSMnx inherits the same value on both directional
    edges. Any per-direction quantity (directional AADT, per-lane capacity)
    is therefore off by ~2x on two-way segments without correction — and
    biased *unequally* between mostly-one-way road classes (motorways) and
    mostly-two-way ones (primary / secondary), which a single coefficient
    can't absorb.

    Rules:
      - `oneway=True`: all lanes are in this direction → return lanes.
      - `lanes` missing: OSM implicit default (1 per direction) → return 1.
      - `lanes <= 1`: can't split a single lane → return 1.
      - otherwise: return lanes / 2.

    Pure function over `edge_data` — caller decides whether to write
    the result back as an edge attribute. `networks_decorate.py`'s
    `_write_lanes_per_direction` calls this for every consolidated
    edge and stores the result; `clean_consolidated_edges` does NOT
    (consolidate is cleanup-only; per-edge derived attrs belong in
    decorate).
    """
    lanes = _parse_lanes(edge_data.get("lanes"))
    oneway = bool(edge_data.get("oneway", False))
    if lanes is None:
        return 1.0
    if oneway or lanes <= 1:
        return max(1.0, lanes)
    return lanes / 2.0


# Edge attributes dropped post-consolidation by `clean_consolidated_edges`
# (callers can override). `name` is the main offender: it lists across
# merged edges, costs disk space in `.graphml`, and isn't used anywhere
# in the routing pipeline.
_DEFAULT_DROP_EDGE_ATTRS = ["name"]


def clean_consolidated_edges(
    graph,
    *,
    drop_edge_attrs: list[str] | None = None,
    edge_attr_aggs: dict | None = None,
) -> None:
    """Post-consolidation edge cleanup. Mutates `graph` in place.

    Three operations per edge:

      1. Drop unwanted attrs (`drop_edge_attrs`; defaults to
         `_DEFAULT_DROP_EDGE_ATTRS`). Saves disk space + avoids
         round-trip ambiguity for non-aggregated list-valued attrs.
      2. Collapse list-valued attrs whose key is in `edge_attr_aggs`
         (defaults to `_DEFAULT_EDGE_ATTR_AGGS`) to a single value
         via the per-attr aggregator. OSMnx 2.x doesn't expose
         `edge_attr_aggs` to `consolidate_intersections`, so this is
         the post-pass that catches `lanes` / `maxspeed` / `highway` /
         `bridge` / `tunnel` lists. The `highway` aggregator uses
         `OSM_HIGHWAY_RANKS` for max-rank tie-breaking. `bridge` and
         `tunnel` arrive as floats from `collapse_degree_2_chains`
         (length-weighted fractions in `networks_from_pbf.py`); the
         numeric mean here handles the rare parallel-edge merge case.
      3. Recompute `length` from `geometry.length` (metric CRS). OSMnx
         sums `length` across merged source edges, which inflates it
         for parallel-path merges — the merged edge's actual geometry
         is shorter than that sum.

    Per-edge derived attributes like `lanes_per_direction` are NOT
    written here — those belong in `networks_decorate.py`, which owns
    every per-edge / per-node attribute addition the pipeline cares
    about. This function is strictly cleanup (drop / collapse /
    re-length) so the consolidate step has a single, narrow purpose.

    Suitable both for graphs produced by `aperta.network_processing.
    consolidate_intersections` and for graphs from a direct
    `ox.consolidate_intersections` call.

    Args:
        graph: an OSMnx-style consolidated MultiDiGraph in a metric CRS
            (so `geometry.length` gives meters).
        drop_edge_attrs: edge attribute keys to remove. Defaults to
            `_DEFAULT_DROP_EDGE_ATTRS` (`['name']`).
        edge_attr_aggs: `{attr: callable}` for collapsing list-valued
            attrs. Defaults to `_DEFAULT_EDGE_ATTR_AGGS`.
    """
    drop_attrs = _DEFAULT_DROP_EDGE_ATTRS if drop_edge_attrs is None else drop_edge_attrs
    eff_edge_aggs = _DEFAULT_EDGE_ATTR_AGGS if edge_attr_aggs is None else edge_attr_aggs
    for _, _, _, d in graph.edges(keys=True, data=True):
        for attr in drop_attrs:
            d.pop(attr, None)
        for attr, aggregator in eff_edge_aggs.items():
            if isinstance(d.get(attr), list):
                d[attr] = aggregator(d[attr])
        geom = d.get("geometry")
        if geom is not None:
            d["length"] = float(geom.length)


# =====================================================================
# OSM highway-rank machinery + per-node OSM classification.
# Moved from `aperta.network_processing` 2026-06-07 as part of the
# sweep that pushed all OSM-aware code to aperta-atlas; aperta is now
# OSM-agnostic and consumes whatever attributes the caller provides.
# =====================================================================


# OSM highway-type ranking. Used by `_highest_rank_highway` (above —
# the aggregator that `clean_consolidated_edges` applies to list-valued
# `highway` attrs) and by `flag_node_osm_classification` (below).
# Higher value = more major road. Anything not listed (or `None`) is
# treated as rank -1 ("not a real motor-vehicle road").
OSM_HIGHWAY_RANKS: dict[str, int] = {
    "motorway": 7,
    "motorway_link": 7,
    "trunk": 6,
    "trunk_link": 6,
    "primary": 5,
    "primary_link": 5,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 3,
    "tertiary_link": 3,
    "residential": 2,
    "road": 2,
    "living_street": 1,
    "pedestrian": 1,
    "unclassified": -1,
    "service": -1,
    "busway": -1,
    "cycleway": -1,
    "footway": -1,
    "path": -1,
    "track": -1,
    "steps": -1,
    "crossing": -1,
    "disused": -1,
}


def _osm_highway_rank(value) -> int:
    """Rank lookup tolerant of strings, lists (OSMnx-merged), and None."""
    if value is None:
        return -1
    if isinstance(value, list):
        return max((OSM_HIGHWAY_RANKS.get(v, -1) for v in value), default=-1)
    return OSM_HIGHWAY_RANKS.get(value, -1)


def flag_node_osm_classification(graph) -> None:
    """Mutate `graph` in place to add **OSM-tag-based** per-node classification
    attributes derived from the per-edge `highway` tag (OSM convention).

    Reads the per-edge `highway` attribute via `OSM_HIGHWAY_RANKS` and the
    per-node `is_t_junction` / `is_4way` flags. Call
    `flag_node_intersection_topology` first so those flags are present.

    Per-node attributes written:

    - `max_highway_rank` — max `OSM_HIGHWAY_RANKS` value over edges
      incident to this node (`-1` for unknown / not-a-real-road, e.g.
      footways).
    - `min_highway_rank` — same with min.
    - `is_t_junction_major` — `is_t_junction` AND `min_highway_rank >= 3`
      (every incident edge is tertiary or better — a "fully classified"
      T-junction with no minor branches).
    - `is_4way_major` — `is_4way` AND `min_highway_rank >= 3`.
    - `is_t_junction_anchor` — `is_t_junction` AND `max_highway_rank >= 3`
      AND `min_highway_rank <= 5` (at least one tertiary-or-better edge,
      and not exclusively trunk / motorway — a trip-anchor T-junction
      where car trips can naturally begin or end).
    - `is_4way_anchor` — `is_4way` AND the same rank condition.

    The two OSM-derived intersection tiers — `_major`, `_anchor` —
    capture progressively different selection criteria for downstream
    snap targets and edge-weight features:

    - **`_major`**: intersections where every connecting street is at
      least tertiary class. Used when only "real road" junctions matter
      (e.g., generating a coarse zone-snap candidate set).
    - **`_anchor`**: intersections that touch at least one main road
      (tertiary or better) and aren't purely highway interchanges. Used
      as priority snap targets for car routing — trips begin and end at
      anchor nodes.

    **Non-OSM networks**: this function only fires on graphs whose edges
    carry the OSM `highway` attribute (or `OSM_HIGHWAY_RANKS`-compatible
    string values for it). For networks with a different classification
    scheme (e.g., LUMOS's simplified 3-tier network with `highway` /
    `autostrasse` / `main_street` tiers), write a project-specific
    classifier that follows the same per-node-attribute pattern.
    """
    is_multi = graph.is_multigraph()

    # Per-node max / min highway rank from incident edges.
    node_max = {n: float("-inf") for n in graph.nodes}
    node_min = {n: float("inf") for n in graph.nodes}
    if is_multi:
        for u, v, _, d in graph.edges(keys=True, data=True):
            rank = _osm_highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank
    else:
        for u, v, d in graph.edges(data=True):
            rank = _osm_highway_rank(d.get("highway"))
            for endpoint in (u, v):
                if rank > node_max[endpoint]:
                    node_max[endpoint] = rank
                if rank < node_min[endpoint]:
                    node_min[endpoint] = rank

    for nid in graph.nodes():
        is_t = bool(graph.nodes[nid].get("is_t_junction", 0))
        is_4 = bool(graph.nodes[nid].get("is_4way", 0))
        mx = node_max[nid]
        mn = node_min[nid]
        max_rank = int(mx) if mx != float("-inf") else -1
        min_rank = int(mn) if mn != float("inf") else -1
        is_major = min_rank >= 3
        is_anchor = (max_rank >= 3) and (min_rank <= 5)

        graph.nodes[nid]["max_highway_rank"] = max_rank
        graph.nodes[nid]["min_highway_rank"] = min_rank
        graph.nodes[nid]["is_t_junction_major"] = int(is_t and is_major)
        graph.nodes[nid]["is_4way_major"] = int(is_4 and is_major)
        graph.nodes[nid]["is_t_junction_anchor"] = int(is_t and is_anchor)
        graph.nodes[nid]["is_4way_anchor"] = int(is_4 and is_anchor)


# =====================================================================
# Legacy OSM-specific wrappers (transitional — retire with the extended
# notebook). Moved here 2026-06-07. These all assume OSMnx-shaped graphs
# and read OSM tag conventions; the boundary rule "aperta is OSM-agnostic"
# now puts them in aperta-atlas even though they're transitional. When
# `aperta-atlas/atlas/` replaces `aperta/examples/extended/`,
# this whole section can be deleted alongside the notebook.
# =====================================================================


def _int_via_float(value) -> int:
    """`int(float(v))` — tolerates both `'0'`/`'1'` and `'0.0'`/`'1.0'` strings.

    Plain `int()` raises on float-formatted strings (`int('0.0')` → ValueError).
    Used as the cast for graphml-loaded `is_*` flags so older saves (where
    these were written as floats) and newer saves (ints) both round-trip
    cleanly.
    """
    return int(float(value))


def _int_via_bool_or_float(value) -> int:
    """Tolerant graphml-cast for attrs that may have been written as Python
    `bool` (round-trips as `'True'` / `'False'` strings) or as `int`."""
    if isinstance(value, (bool, int)):
        return int(value)
    if value in ("True", "true"):
        return 1
    if value in ("False", "false"):
        return 0
    return int(float(value))


# Per-node attribute dtypes that `consolidate_intersections` writes as
# ints. OSMnx's own `default_node_dtypes` only knows about its built-in
# attrs (elevation, x, y, osmid, street_count, lat, lon), so without
# this constant our custom `is_*` and `*_highway_rank` flags round-trip
# as strings — and `int('0.0')` would raise downstream.
_CONSOLIDATED_NODE_DTYPES: dict = {
    "n_streets": _int_via_float,
    "is_t_junction": _int_via_float,
    "is_4way": _int_via_float,
    "is_t_junction_major": _int_via_float,
    "is_4way_major": _int_via_float,
    "is_t_junction_anchor": _int_via_float,
    "is_4way_anchor": _int_via_float,
    "max_highway_rank": _int_via_float,
    "min_highway_rank": _int_via_float,
    "is_traffic_signal": _int_via_float,
    "is_stop": _int_via_float,
    "is_yield": _int_via_float,
    "is_roundabout": _int_via_float,
}

_CONSOLIDATED_EDGE_DTYPES: dict = {
    "lanes_per_direction": float,
    "density_norm": float,
    "is_t_junction": float,
    "is_4way": float,
    "is_traffic_signal": float,
}


# Prefix-scan attribute prefixes for dynamic per-mode bool flags
# written by `prepare_network` / `compute_snap_eligibility` /
# `insert_projected_nodes`. See `load_consolidated_graphml` below.
_PREFIX_SCAN_NODE = ("is_snap_eligible_", "is_virtual")
_PREFIX_SCAN_EDGE = ("cost_excluded_",)


def _scan_graphml_keys(filepath, prefixes, scope):
    """Return the names of `<key for=scope>` graphml attributes that
    start with any of `prefixes` (or equal one — handles the no-suffix
    `is_virtual` case via the prefix-match condition).
    """
    import xml.etree.ElementTree as ET

    ns = "{http://graphml.graphdrawing.org/xmlns}"
    out: list[str] = []
    for _event, elem in ET.iterparse(filepath, events=("end",)):
        if elem.tag == f"{ns}key" and elem.get("for") == scope:
            name = elem.get("attr.name", "")
            if any(name.startswith(p) or name == p for p in prefixes):
                out.append(name)
        elif elem.tag == f"{ns}graph":
            break
        elem.clear()
    return out


def load_consolidated_graphml(
    filepath, *, node_dtypes: dict | None = None,
    edge_dtypes: dict | None = None, **kwargs,
):
    """Load a graphml saved by `consolidate_intersections`, casting
    aperta-atlas's custom `is_*` / `*_highway_rank` / `is_snap_eligible_<mode>`
    / `cost_excluded_<mode>` / `is_virtual` attrs back to their proper
    types.

    Thin wrapper around `osmnx.load_graphml` that merges:
      1. `_CONSOLIDATED_NODE_DTYPES` / `_CONSOLIDATED_EDGE_DTYPES`
         (fixed casts for the attrs `consolidate_intersections` writes).
      2. Prefix-scan of the graphml `<key>` schema for the dynamic
         per-mode flags written by `prepare_network` etc.
      3. Caller overrides via `node_dtypes` / `edge_dtypes` (win).
    """
    import osmnx as ox

    auto_node = {
        name: _int_via_bool_or_float
        for name in _scan_graphml_keys(filepath, _PREFIX_SCAN_NODE, "node")
    }
    auto_edge = {
        name: _int_via_bool_or_float
        for name in _scan_graphml_keys(filepath, _PREFIX_SCAN_EDGE, "edge")
    }
    merged_node = {**_CONSOLIDATED_NODE_DTYPES, **auto_node, **(node_dtypes or {})}
    merged_edge = {**_CONSOLIDATED_EDGE_DTYPES, **auto_edge, **(edge_dtypes or {})}
    return ox.load_graphml(filepath, node_dtypes=merged_node, edge_dtypes=merged_edge, **kwargs)


def extract_obstacle_locations(
    graph,
    *,
    obstacle_node_tags: dict[str, tuple[str, str]] | None = None,
    detect_roundabouts: bool = True,
) -> tuple[dict[str, list[tuple[float, float]]], list[tuple[float, float]]]:
    """Pull obstacle + roundabout `(x, y)` locations from a raw OSMnx graph.

    Companion to `consolidate_intersections`. Returns the two structures
    the consolidator needs (`obstacle_xy`, `roundabout_xy`) so callers
    can extract obstacles *once* from a canonical source (typically the
    raw car graph — the most signal-complete) and reuse for every
    network type's consolidation. This matters because OSMnx's per-
    network-type filters drop ways that signals sit on (e.g. trunk roads
    excluded from walk graphs), losing those signal nodes entirely from
    the walk graph's node set; passing the union of locations via
    `obstacle_locations=` / `roundabout_locations=` to
    `consolidate_intersections` reattaches them to whichever consolidated
    node is nearest in each network.
    """
    if obstacle_node_tags is None:
        obstacle_node_tags = {
            "traffic_signal": ("highway", "traffic_signals"),
            "stop": ("highway", "stop"),
            "yield": ("highway", "give_way"),
        }
    obstacle_xy: dict[str, list[tuple[float, float]]] = {
        name: [] for name in obstacle_node_tags}
    for _, ndata in graph.nodes(data=True):
        for obstacle_name, (key, value) in obstacle_node_tags.items():
            tag_value = ndata.get(key)
            if (tag_value == value
                    or (isinstance(tag_value, list) and value in tag_value)):
                obstacle_xy[obstacle_name].append((ndata["x"], ndata["y"]))
    roundabout_xy: list[tuple[float, float]] = []
    if detect_roundabouts:
        for u, v, _, edata in graph.edges(keys=True, data=True):
            j = edata.get("junction")
            if j == "roundabout" or (isinstance(j, list) and "roundabout" in j):
                u_attr, v_attr = graph.nodes[u], graph.nodes[v]
                roundabout_xy.append(
                    ((u_attr["x"] + v_attr["x"]) / 2,
                     (u_attr["y"] + v_attr["y"]) / 2))
    return obstacle_xy, roundabout_xy


def consolidate_intersections(
    graph,
    tolerance: float,
    *,
    obstacle_buffer: float = 30.0,
    obstacle_node_tags: dict[str, tuple[str, str]] | None = None,
    obstacle_locations: dict[str, list[tuple[float, float]]] | None = None,
    detect_roundabouts: bool = True,
    roundabout_locations: list[tuple[float, float]] | None = None,
    node_attr_aggs: dict | None = None,
    edge_attr_aggs: dict | None = None,
    drop_edge_attrs: list[str] | None = None,
):
    """OSMnx intersection consolidation + obstacle-aware re-flagging.

    Wraps `osmnx.consolidate_intersections(rebuild_graph=True)` with the
    post-processing OSMnx alone misses: traffic-signal / stop / give-way
    nodes typically sit a few metres off the geometric intersection
    centre, so OSMnx's `tolerance`-based merge can throw those nodes away
    rather than carrying the `highway=traffic_signals` tag onto the
    surviving consolidated node.

    This wrapper captures obstacle locations from the *original* graph
    before consolidation, then spatially re-attaches them to the nearest
    surviving consolidated node within `obstacle_buffer` metres. Same
    trick for roundabouts (whose `junction=roundabout` tag lives on
    edges in OSM and is otherwise lost when the roundabout collapses).

    Post-consolidation cleanup delegates to `clean_consolidated_edges`
    (drop noise attrs, collapse list-valued `lanes`/`maxspeed`/`highway`,
    recompute `length`); per-edge `lanes_per_direction` is written via
    the canonical `lanes_per_direction` helper.

    The returned graph has per-node attributes set by
    `flag_node_intersection_topology` (`n_streets`, `is_t_junction`,
    `is_4way`) plus one `is_<name>` per requested obstacle type plus
    `is_roundabout` if `detect_roundabouts=True`. As of 2026-06-07 we
    no longer also call `flag_node_osm_classification` here (its output
    — `max_highway_rank`, `_major`, `_anchor` — was unused by the
    extended notebook); callers that need those flags should invoke
    `flag_node_osm_classification` separately on the result.

    Args:
        graph: an OSMnx MultiDiGraph (projected; `tolerance` in metres).
        tolerance: nodes within this distance are merged. 5-15 m urban;
            ~25 m sparse.
        obstacle_buffer: max distance (m) for the obstacle re-attach.
            Should be >= `tolerance`; default 30 m.
        obstacle_node_tags: `{flag_name -> (osm_key, osm_value)}` —
            obstacle node tags to extract. Default:
            `{'traffic_signal': ('highway', 'traffic_signals'),
              'stop': ('highway', 'stop'),
              'yield': ('highway', 'give_way')}`.
        obstacle_locations: pre-supplied `{flag_name -> [(x, y), ...]}`.
        detect_roundabouts: extract `junction=roundabout` midpoints.
        roundabout_locations: pre-supplied roundabout midpoints.
        node_attr_aggs / edge_attr_aggs / drop_edge_attrs: passed
            through to `ox.consolidate_intersections` /
            `clean_consolidated_edges`.

    Returns:
        Consolidated `nx.MultiDiGraph` with new integer node IDs.
    """
    from typing import cast
    import networkx as nx
    import osmnx as ox

    # Local import to avoid a circular dep with the aperta library at
    # module load time (aperta-atlas imports from aperta; aperta does not
    # import from aperta-atlas, but the function lookup happens at call
    # time anyway).
    from aperta.network_processing import flag_node_intersection_topology

    # 1. Obstacle + roundabout locations (auto-extract if not supplied).
    if (obstacle_locations is None
            or (detect_roundabouts and roundabout_locations is None)):
        auto_obstacle_xy, auto_roundabout_xy = extract_obstacle_locations(
            graph,
            obstacle_node_tags=obstacle_node_tags,
            detect_roundabouts=detect_roundabouts,
        )
        if obstacle_locations is None:
            obstacle_locations = auto_obstacle_xy
        if detect_roundabouts and roundabout_locations is None:
            roundabout_locations = auto_roundabout_xy

    # 2. Consolidate.
    consolidated = cast(
        nx.MultiDiGraph,
        ox.consolidate_intersections(
            graph,
            tolerance=tolerance,
            rebuild_graph=True,
            reconnect_edges=True,
            node_attr_aggs=node_attr_aggs,
        ),
    )

    # 3. Cleanup (canonical helper — drops noise, collapses list-valued
    #    lanes/maxspeed/highway via `_DEFAULT_EDGE_ATTR_AGGS`, recomputes
    #    length from geometry).
    clean_consolidated_edges(
        consolidated,
        drop_edge_attrs=drop_edge_attrs,
        edge_attr_aggs=edge_attr_aggs,
    )

    # 4. Per-edge `lanes_per_direction` (canonical helper). Reads the
    #    already-collapsed `lanes` + `oneway`.
    for _, _, _, d in consolidated.edges(keys=True, data=True):
        d["lanes_per_direction"] = lanes_per_direction(d)

    # 5. Topology flags (network-agnostic, in aperta).
    flag_node_intersection_topology(consolidated)

    # 6. Spatial re-attachment: nearest consolidated node within
    #    obstacle_buffer gets the obstacle / roundabout flag.
    from aperta.network_processing import snap_features_to_nodes
    for name, locs in obstacle_locations.items():
        snap_features_to_nodes(consolidated, locs, flag_name=name, max_distance=obstacle_buffer)
    if detect_roundabouts and roundabout_locations is not None:
        snap_features_to_nodes(
            consolidated, roundabout_locations,
            flag_name="roundabout", max_distance=obstacle_buffer)

    return consolidated


# =====================================================================
# POI category-map helpers (pure logic — no network access).
# Moved from `aperta_atlas.osm_helpers` 2026-06-07 (file deleted).
# Used by `aperta/examples/extended/prepare/1_download.py` (the small-
# scale Overpass-API path) for turning a
# `{category -> [(tag:value, weight), ...]}` map into either an OSMnx
# tag query or per-feature category columns on a POI GeoDataFrame.
#
# The PBF-based `preparation/world/osm/pois_download.py` doesn't use
# these — it streams via pyosmium with an inverted (key, value) ->
# categories lookup table that matches the stream-processing access
# pattern. The PBF script is the canonical production path; the API
# path lives in the aperta extended notebook as a "simplest way to get
# started" pedagogical example, not in aperta-atlas.
# =====================================================================


# A category map: `{user_category -> [(osm_tag_pair, weight), ...]}`
# where `osm_tag_pair` is a `'key:value'` string like `'shop:supermarket'`.
CategoryMap = dict[str, list[tuple[str, float]]]


def osm_tag_query_for_categories(
    category_map: CategoryMap,
) -> dict[str, bool | str | list[str]]:
    """Build the osmnx `tags=` argument from a category map.

    Unions every `key:value` pair across all categories and groups by key.
    The result is the minimal query that returns *every* feature any
    category could match.

    Args:
        category_map: `{user_category -> [(tag_pair, weight), ...]}` where
            `tag_pair` is a `'key:value'` string.

    Returns:
        `{osm_tag_key -> [values, ...]}` with values sorted (deterministic
        for caching / hashing). Pass directly as the `tags` argument to
        `osmnx.features_from_polygon` / `osmnx.features_from_place`.
    """
    out: dict[str, set[str]] = {}
    for tags in category_map.values():
        for tag_pair, _weight in tags:
            if ":" not in tag_pair:
                raise ValueError(
                    f"Tag pair {tag_pair!r} must be 'key:value' "
                    f"(e.g. 'shop:supermarket').")
            key, value = tag_pair.split(":", 1)
            out.setdefault(key, set()).add(value)
    return {k: sorted(v) for k, v in out.items()}


def categorize_pois(
    pois,
    category_map: CategoryMap,
    *,
    weight_suffix: str = "_weight",
    drop_unmatched: bool = True,
):
    """Add per-category count + weighted-count columns to a POI GeoDataFrame.

    For each `(category, [(tag:value, weight), ...])` entry, two new columns
    are appended:

    - **`{category}`** (int): number of listed `(tag:value)` pairs this row
      matches. Usually 0 or 1; can be >= 2 if multiple of the listed pairs
      match for one feature (a feature with both `amenity=school` and
      `school=primary` for a `schools` category that lists both).
    - **`{category}{weight_suffix}`** (float): sum of weights across all
      matching pairs. Equal to the count if every weight is 1.

    Features matching no category at all are dropped if
    `drop_unmatched=True` (the typical case — saves carrying around OSM
    features that aren't of interest).

    Args:
        pois: GeoDataFrame containing the OSM tag columns referenced by
            `category_map` (e.g. `amenity`, `shop`, `leisure`). Tags missing
            from the DataFrame are silently treated as never-matching, so
            partial input works.
        category_map: `{category -> [(tag:value, weight), ...]}`.
        weight_suffix: suffix for the per-category weight column. Default
            `'_weight'`. Set to e.g. `'_w'` for shorter columns.
        drop_unmatched: drop rows matching no listed `(tag:value)` pair.
            Default `True`.

    Returns:
        A copy of `pois` with two new columns per category. Original
        columns + index preserved.
    """
    pois = pois.copy()
    count_cols: list[str] = []
    for category, tags in category_map.items():
        weight_col = f"{category}{weight_suffix}"
        if category in pois.columns or weight_col in pois.columns:
            raise ValueError(
                f"Category {category!r} would overwrite an existing column "
                f"(have {category!r} / {weight_col!r}). Rename the category "
                f"or use a different `weight_suffix`.")
        pois[category] = 0
        pois[weight_col] = 0.0
        for tag_pair, weight in tags:
            key, value = tag_pair.split(":", 1)
            if key not in pois.columns:
                continue
            match = pois[key] == value
            pois.loc[match, category] += 1
            pois.loc[match, weight_col] += float(weight)
        count_cols.append(category)
    if drop_unmatched and count_cols:
        any_match = pois[count_cols].sum(axis=1) > 0
        pois = pois[any_match]
    return pois


