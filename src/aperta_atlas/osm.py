"""
OSM way classification rules per network type (walk / bike / drive / all),
plus OSM-specific post-consolidation helpers.

Single source of truth for "is this OSM way usable for routing in this
mode": `is_usable` filters the OSM way stream at network extraction
(`preparation/world/osm/networks_from_pbf.py`).

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
areas). One deliberate exception: the bike network leaves out
mountain-hiking trails, which bikes may use but the everyday cyclist the
networks stand for does not (layer 3).

Four rule layers (ALL must pass for a way to be usable):

  1. **Universal access**: ways tagged `access=private`, `access=no`, or
     `area=yes` excluded for any mode. First two are explicit "no public
     access", unless a mode-specific tag allows the mode (OSM's access
     hierarchy: `access=no` + `bicycle=yes` is a street closed to motor
     traffic but open to bikes; `_mode_access_override`); third is a
     polygon (plaza, parking lot) that isn't a linear route.

  2. **Universal highway-value exclusion**: highway values that aren't
     routes at all for any mode (`abandoned`, `construction`, `proposed`,
     `via_ferrata`, etc.) — see `_UNIVERSAL_HIGHWAY_EXCLUDE`.

  3. **Mode-specific access exclusions**: explicit "this mode excluded":
       - walk: `foot=no`; or `motorroad=yes` (motorway-equivalent)
       - bike: `bicycle=no`; or `motorroad=yes`; or a mountain-hiking
         trail (`sac_scale` T2 or harder) that is not part of a
         mountain-bike route (`_SAC_SCALES_NOT_FOR_BIKES`)
       - drive: `motor_vehicle=no` OR `motorcar=no`

  4. **Mode-specific highway-value rule** (`_HIGHWAY_RULES`):
     per-mode whitelist (`keep`) or blacklist (`exclude`) of
     `highway=...` values. Walk uses a narrow blacklist (motor-only +
     bus-only); bike a slightly wider blacklist (+ steps, escalator);
     drive a positive whitelist of car-drivable categories incl.
     `service`.
"""


# 2. Universal highway-value exclusions — apply to every network_type.
# Match OSMnx's convention in `_overpass.py`, plus `via_ferrata`. These represent things
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
    'via_ferrata',   # protected climbing route, needs climbing gear
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
# etc.) are the caller's separate concern. `is_usable` also reads
# `mtb_route` ('yes' on ways in a `route=mtb` relation), which is not a
# way tag: callers set it from the relations.
NEEDED_TAGS: tuple[str, ...] = (
    'highway',
    'service',
    'access',
    'area',
    'foot',
    'bicycle',
    'sac_scale',
    'motor_vehicle',
    'motorcar',
    'motorroad',
    # `sidepath_rule` (sidewalks mapped as their own ways):
    'footway',
    'sidewalk',
    'sidewalk:both',
    'sidewalk:left',
    'sidewalk:right',
)


# Mode-specific tags that override `access=no|private` (layer 1).
_MODE_ACCESS_KEYS: dict[str, tuple[str, ...]] = {
    'walk': ('foot',), 'bike': ('bicycle',), 'drive': ('motor_vehicle', 'motorcar'),
}
# Values that admit general car traffic; `forestry`, `agricultural`, `permit`, `delivery`, …
# admit only some vehicles, so the road stays closed to cars.
_DRIVE_ACCESS_ALLOWED: frozenset[str] = frozenset({
    'yes', 'designated', 'permissive', 'destination', 'customers',
})


def _mode_access_override(tags: dict, network_type: str) -> bool:
    """Whether a mode-specific tag allows `network_type` on a way tagged `access=no|private`.
    Walk / bike err on the side of keeping the way: any explicit `foot` / `bicycle` value
    other than `no` / `private` (incl. `dismount`, `unknown`) overrides."""
    values = [tags[k] for k in _MODE_ACCESS_KEYS.get(network_type, ()) if tags.get(k)]
    if not values:
        return False
    if network_type == 'drive':
        return any(v in _DRIVE_ACCESS_ALLOWED for v in values)
    return all(v not in ('no', 'private') for v in values)


# Hiking-trail grades (`sac_scale`, OSM wiki) left out of the bike network: T2 (mountain hiking)
# and harder. Bikes are often allowed there, but the networks stand for everyday cyclists, who
# do not ride (or push) up mountain trails. Ways in a mountain-bike route (`mtb_route`) stay;
# the hard ones are pushed (`bike_push`). T1 (`hiking`) stays: often a gravel path.
_SAC_SCALES_NOT_FOR_BIKES: frozenset[str] = frozenset({
    'mountain_hiking', 'demanding_mountain_hiking', 'alpine_hiking',
    'demanding_alpine_hiking', 'difficult_alpine_hiking',
})


def is_usable(tags: dict, network_type: str) -> bool:
    """True iff a way (or edge) with these OSM tags is usable for
    routing in `network_type`. Applies all four rule layers — see
    module docstring.

    Tag values are compared with string `==` / `in`, matching OSM's
    convention that tag values are strings. A list-valued `highway`
    (merged edges, e.g. OSMnx-simplified) is handled for `highway` only —
    other tags shouldn't be list-valued by construction.

    Raises `ValueError` for unknown `network_type`.
    """
    if network_type not in _HIGHWAY_RULES:
        raise ValueError(
            f"Unknown network_type {network_type!r}. Known: "
            f"{sorted(_HIGHWAY_RULES)}.")

    # Layer 1: universal access exclusions.
    if tags.get('access') in ('private', 'no') and not _mode_access_override(tags, network_type):
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
        if (tags.get('sac_scale') in _SAC_SCALES_NOT_FOR_BIKES
                and tags.get('mtb_route') != 'yes'):
            return False
    elif network_type == 'drive':
        if tags.get('motor_vehicle') == 'no' or tags.get('motorcar') == 'no':
            return False

    # Layers 2 + 4: highway-value rule (universal first, then mode-specific).
    hw = tags.get('highway')
    if hw is None:
        return False
    rule = _HIGHWAY_RULES[network_type]

    # List-valued highway (merged edges): universal exclude filters per element; the
    # way is usable if ANY of its remaining highway values passes the mode rule.
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


# -----------------------------------------------------------------------
# Post-consolidation edge cleanup (OSM `lanes` / `oneway` conventions).
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
# Surface types (`surface_type` in `edges_<mode>_extended.csv`, main/02b). "Smooth" unpaved:
# rideable with most bikes.
SURFACE_TYPES: tuple[str, ...] = ('paved', 'cobble', 'unpaved_smooth', 'unpaved_rough')
# OSM `surface` values (OSM wiki, Key:surface) by type. Values not listed (rare, ambiguous or
# several, e.g. `asphalt;gravel`) count as unknown.
_SURFACE_TYPE_BY_SURFACE: dict[str, str] = {
    **dict.fromkeys((
        'paved', 'asphalt', 'chipseal', 'concrete', 'concrete:lanes', 'concrete:plates',
        'paving_stones', 'paving_stones:lanes', 'bricks', 'metal', 'metal_grid', 'wood',
        'rubber', 'tiles'), 'paved'),
    **dict.fromkeys(
        ('sett', 'cobblestone', 'unhewn_cobblestone', 'cobblestone:flattened'), 'cobble'),
    **dict.fromkeys(('unpaved', 'compacted', 'fine_gravel', 'gravel'), 'unpaved_smooth'),
    **dict.fromkeys((
        'pebblestone', 'rock', 'shells', 'ground', 'dirt', 'earth', 'grass', 'grass_paver',
        'mud', 'sand', 'woodchips', 'clay', 'snow', 'ice', 'salt'), 'unpaved_rough'),
}
# `tracktype` (where `surface` is missing): grade1 solid (paved), grade2 solid but unpaved
# (gravel, compacted), grade3-5 mixed to soft.
_SURFACE_TYPE_BY_TRACKTYPE: dict[str, str] = {
    'grade1': 'paved', 'grade2': 'unpaved_smooth', 'grade3': 'unpaved_rough',
    'grade4': 'unpaved_rough', 'grade5': 'unpaved_rough',
}
# `smoothness` (OSM wiki): `intermediate` and better is fine for a city bike.
_SURFACE_TYPE_BY_SMOOTHNESS: dict[str, str] = {
    **dict.fromkeys(('excellent', 'good', 'intermediate'), 'unpaved_smooth'),
    **dict.fromkeys(('bad', 'very_bad', 'horrible', 'very_horrible', 'impassable'),
                    'unpaved_rough'),
}


def surface_type(data: dict) -> str | None:
    """An edge's surface type (`SURFACE_TYPES`), None if unknown: from `surface`, else from
    `tracktype`; for unpaved ways, `smoothness` (where tagged) decides smooth or rough. Merged
    edges with several values: the first one with a type."""
    def first(key: str, table: dict[str, str]) -> str | None:
        return next((table[v] for v in _tag_values(data, key)
                     if isinstance(v, str) and v in table), None)

    kind = first('surface', _SURFACE_TYPE_BY_SURFACE) or first(
        'tracktype', _SURFACE_TYPE_BY_TRACKTYPE)
    if kind in ('unpaved_smooth', 'unpaved_rough'):
        kind = first('smoothness', _SURFACE_TYPE_BY_SMOOTHNESS) or kind
    return kind


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
    "steps": _mean_numeric,
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
    if isinstance(raw, str):
        raw = raw.split(";")[0]                 # `2;3` (one value per part of the way)
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
         (defaults to `_DEFAULT_EDGE_ATTR_AGGS`, see its comment) to a
         single value; OSMnx 2.x doesn't expose `edge_attr_aggs` to
         `consolidate_intersections`.
      3. Recompute `length` from `geometry.length` (metric CRS; see the
         `_DEFAULT_EDGE_ATTR_AGGS` comment).

    Derived per-edge attributes (e.g. `lanes_per_direction`) are written by
    `networks_decorate.py`, not here.

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


# Sidepaths: a sidewalk or cycle track mapped as its own way beside a road, so that the street is
# in the walk / bike network twice. `drop_sidepath_ways` removes the way the mode does not use.
SIDEPATH_ROAD_HIGHWAYS: frozenset[str] = frozenset({
    "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service",
    "road",
})
# `bicycle=*` values under which cyclists may ride a path or footway.
BICYCLE_ALLOWED: frozenset[str] = frozenset({"yes", "designated", "permissive"})
# A way the mode does use runs "alongside" if it covers this share of the dropped way's length
# within this distance.
SIDEPATH_ALONGSIDE_M = 20.0
SIDEPATH_ALONGSIDE_SHARE = 0.8


def sidepath_rule(tags: dict, network_type: str) -> str | None:
    """Why `network_type` does not use this way although `is_usable` keeps it, or None.

      - bike: a sidewalk (`footway=sidewalk`) without any `bicycle` tag (cyclists ride on
        the road beside it; an explicit value, even `dismount`, keeps it); a road with
        `bicycle=use_sidepath` (cyclists must take the way beside it). Crossings stay: they
        link footways to the road (stubs left at dropped sidewalks are short dead ends that
        `networks_from_pbf`'s dead-end pruning removes);
      - walk: a road with sidewalks mapped separately on both sides (`sidewalk=separate`,
        `sidewalk:both=separate`, or left and right `separate`) or `foot=use_sidepath`.
    """
    highway = tags.get("highway")
    if network_type == "bike":
        if (highway == "footway" and tags.get("footway") == "sidewalk"
                and tags.get("bicycle") is None):
            return "footway=sidewalk"
        if highway in SIDEPATH_ROAD_HIGHWAYS and tags.get("bicycle") == "use_sidepath":
            return "bicycle=use_sidepath"
    elif network_type == "walk" and highway in SIDEPATH_ROAD_HIGHWAYS:
        if (tags.get("foot") == "use_sidepath"
                or "separate" in (tags.get("sidewalk"), tags.get("sidewalk:both"))
                or (tags.get("sidewalk:left") == "separate"
                    and tags.get("sidewalk:right") == "separate")):
            return "sidewalks mapped separately"
    return None


def _sidepath_partner(tags: dict, rule: str) -> bool:
    """Whether a way the mode keeps can stand in for a way dropped under `rule`."""
    highway = tags.get("highway")
    if rule.startswith("footway="):                 # bike: the road beside the sidewalk
        return highway in SIDEPATH_ROAD_HIGHWAYS
    if rule == "bicycle=use_sidepath":              # bike: the track beside the road
        return highway == "cycleway" or tags.get("bicycle") in BICYCLE_ALLOWED
    return highway in ("footway", "path", "pedestrian", "cycleway", "bridleway", "steps")


def drop_sidepath_ways(graph, network_type: str) -> dict[str, float]:
    """Remove ways that `network_type` does not use from a freshly built graph. Mutates `graph`.

    Errs on the side of keeping a way. An OSM way (all edges with its `osmid`) is dropped only
    if (1) `sidepath_rule` names it, (2) a way the mode does use runs alongside it, in both
    directions (`two_way`): within
    `SIDEPATH_ALONGSIDE_M` over `SIDEPATH_ALONGSIDE_SHARE` of its length, so no route is lost,
    only a duplicate, and (3) dropping it disconnects nothing: dropped ways that touch a part
    of the graph that lost its connection to the largest component are put back, until none
    does. Whole ways go or stay, so streets do not fragment.

    Expects the graph as `networks_from_pbf.build_multidigraph` returns it: WGS84 edge
    geometries, `osmid` = OSM way id, OSM tags as edge attributes. No-op for other network
    types than walk and bike. Returns counts and km (one direction) of what was dropped, kept
    for lack of a way alongside, and put back.
    """
    import networkx as nx
    import numpy as np
    import osmnx as ox
    import shapely

    stats = {"ways_dropped": 0, "km_dropped": 0.0, "ways_kept_no_alongside": 0,
             "ways_put_back": 0}
    if network_type not in ("walk", "bike"):
        return stats
    edges = ox.projection.project_gdf(ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True))
    attrs = {(u, v, k): d for u, v, k, d in graph.edges(keys=True, data=True)}
    rules = {e: sidepath_rule(d, network_type) for e, d in attrs.items()}
    candidates = {e for e, r in rules.items() if r}
    if not candidates:
        return stats

    # (2) A way the mode uses alongside, per OSM way and rule.
    way_of = {e: attrs[e].get("osmid") for e in attrs}
    geom = dict(zip(edges.index, edges.geometry))
    keep_ways = set()
    def two_way(e) -> bool:
        """The way also runs the other way in this graph (so a duplicate dropped beside it
        loses no direction, e.g. a sidewalk beside a road that is one-way for bikes)."""
        back = graph.get_edge_data(e[1], e[0]) or {}
        return any(d.get("osmid") == way_of[e] for d in back.values())

    for rule in {rules[e] for e in candidates}:
        partner_edges = [e for e in attrs if not rules[e] and _sidepath_partner(attrs[e], rule)
                         and two_way(e)]
        tree = shapely.STRtree([geom[e] for e in partner_edges])
        by_way: dict = {}
        for e in candidates:
            if rules[e] == rule:
                by_way.setdefault(way_of[e], []).append(e)
        for way, way_edges in by_way.items():
            line = shapely.union_all([geom[e] for e in way_edges])
            near = tree.query(line, predicate="dwithin", distance=SIDEPATH_ALONGSIDE_M)
            covered = 0.0
            if len(near):
                cover = shapely.union_all(
                    shapely.buffer(np.asarray(tree.geometries)[near], SIDEPATH_ALONGSIDE_M))
                covered = line.intersection(cover).length
            if covered < SIDEPATH_ALONGSIDE_SHARE * line.length:
                keep_ways.add(way)
    stats["ways_kept_no_alongside"] = len(keep_ways)
    drop = {e for e in candidates if way_of[e] not in keep_ways}

    # (3) Drop, then put back dropped ways that touch anything that lost its connection.
    largest = max(nx.weakly_connected_components(graph), key=len)
    graph.remove_edges_from(drop)
    put_back = set()
    while drop:
        connected = max(nx.weakly_connected_components(graph), key=len)
        stranded = {n for n in largest - connected if graph.degree(n) > 0}
        ways = {way_of[e] for e in drop if e[0] in stranded or e[1] in stranded}
        if not ways:
            break
        back = {e for e in drop if way_of[e] in ways}
        graph.add_edges_from((u, v, k, attrs[(u, v, k)]) for u, v, k in back)
        drop -= back
        put_back |= ways
    stats["ways_put_back"] = len(put_back)

    stats["ways_dropped"] = len({way_of[e] for e in drop})
    # Two-way ways have an edge per direction: count each way's length once.
    seen = set()
    for e in drop:
        if (way_of[e], frozenset(e[:2])) not in seen:
            seen.add((way_of[e], frozenset(e[:2])))
            stats["km_dropped"] += geom[e].length / 1_000
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
    return stats


# ---------------------------------------------------------------------------
# Speed limits and bike traffic stress
# ---------------------------------------------------------------------------

_MPH_KMH = 1.609344
# Implicit limits (`maxspeed=CH:urban` etc.), by the part after the country prefix; `rural`
# differs by country.
_MAXSPEED_IMPLICIT_KMH: dict[str, float] = {
    "urban": 50, "zone30": 30, "zone:30": 30, "zone20": 20, "zone:20": 20, "zone": 30,
    "living_street": 20, "walk": 7, "motorway": 120, "trunk": 100, "rural": 80,
}
_MAXSPEED_RURAL_KMH: dict[str, float] = {
    "DE": 100, "AT": 100, "IT": 90, "FR": 80, "CH": 80,
}
_MAXSPEED_GB_KMH: dict[str, float] = {      # national speed limits, in mph
    "GB:nsl_single": 60 * _MPH_KMH, "GB:nsl_dual": 70 * _MPH_KMH,
    "GB:nsl_restricted": 30 * _MPH_KMH, "GB:motorway": 70 * _MPH_KMH,
}


def parse_maxspeed_kmh(value) -> float | None:
    """An OSM `maxspeed` value in km/h: numbers (`50`, `30 mph`), implicit limits
    (`CH:urban`, `DE:rural`, `GB:nsl_single`, `CH:zone30`), several values (`50;80`: the
    highest). None if missing, not understood or not above 0 (`maxspeed=0` is a tagging error,
    not a limit)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None    # NaN compares False
    text = str(value).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace("|", ";").split(";") if p.strip()]
    if len(parts) > 1:
        speeds = [s for s in (parse_maxspeed_kmh(p) for p in parts) if s is not None]
        return max(speeds) if speeds else None
    if text in _MAXSPEED_GB_KMH:
        return _MAXSPEED_GB_KMH[text]
    lower = text.lower()
    try:
        if lower.endswith("mph"):
            return parse_maxspeed_kmh(float(lower[:-3].strip()) * _MPH_KMH)
        return parse_maxspeed_kmh(
            float(lower.removesuffix("km/h").removesuffix("kmh").strip().replace(",", ".")))
    except ValueError:
        pass
    country, _, kind = text.partition(":")
    kind = (kind or country).lower()
    if kind == "rural":
        return _MAXSPEED_RURAL_KMH.get(country.upper(), 80)
    return _MAXSPEED_IMPLICIT_KMH.get(kind)


def _tag_values(data: dict, key: str) -> list:
    """A tag's values on an edge; merged edges carry lists after consolidation."""
    value = data.get(key)
    return value if isinstance(value, list) else [value]


# Ways a bike can be pushed along but not ridden: footways, pedestrian ways, corridors and steps
# without bike access, and anything tagged `bicycle=dismount`. The bike network keeps them where
# they carry connectivity (`drop_sidepath_ways`).
_PUSH_HIGHWAYS: frozenset[str] = frozenset({"footway", "pedestrian", "corridor", "steps"})
# Mountain-bike trails rated S2 or harder (`mtb:scale`, OSM wiki: steps, large roots and rocks)
# are pushed too, whatever the bike access: everyday cyclists walk them.
_PUSH_MTB_SCALE = 2


def _hard_mtb_trail(value) -> bool:
    """Whether an `mtb:scale` value is `_PUSH_MTB_SCALE` or harder, by its leading digit (`2+`;
    a float when read back from CSV)."""
    s = str(value)
    return s[:1].isdigit() and int(s[0]) >= _PUSH_MTB_SCALE


def bike_push(data: dict) -> bool:
    """Whether a bike has to be pushed on this edge. Merged edges count as ridable if any of
    their values is."""
    if all(_hard_mtb_trail(v) for v in _tag_values(data, "mtb:scale")):
        return True
    bicycle = _tag_values(data, "bicycle")
    if any(b in BICYCLE_ALLOWED for b in bicycle):
        return False
    if "dismount" in bicycle:
        return True
    return all(h in _PUSH_HIGHWAYS for h in _tag_values(data, "highway"))


# Bike traffic stress, 0 (none) to 5 (very high), loosely after the Level of Traffic Stress
# (Mekuria, Furth & Nixon 2012; Furth's LTS 2.0 criteria: speed limit and lanes per direction,
# by type of bike infrastructure), with road class standing in for traffic volume (main roads
# busy, local streets quiet). The speed limit (and lanes) set a row, whose baseline is the level
# with a painted lane; separation from motor traffic takes one off, mixed traffic adds one. With
# a painted lane or in mixed traffic, a one-way street (for motor traffic) moves one row towards
# less stress (no oncoming traffic), a climb one row towards more (the speed difference to motor
# traffic is largest uphill; descents have stresses of their own, so they stay neutral). A way
# separated from motor traffic is rated by the road beside it (`bike_stress_beside`), not as
# no stress throughout: riding next to a busy road is more stressful than on a quiet path, and
# the level also stands in for the stress of crossing that road's junctions.
STRESS_MAIN_ROADS: frozenset[str] = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
})
_CYCLEWAY_SEPARATED: frozenset[str] = frozenset({"track", "opposite_track", "share_sidewalk"})
_CYCLEWAY_PAINTED: frozenset[str] = frozenset({     # incl. bus lanes shared with bikes
    "lane", "lanes", "opposite_lane", "yes", "right", "left", "both",
    "share_busway", "opposite_share_busway",
})
# Ways without motor traffic.
_CAR_FREE_HIGHWAYS: frozenset[str] = frozenset({
    "cycleway", "path", "track", "bridleway", "footway", "pedestrian", "steps", "corridor",
})
# `cycleway*` values that mean "no bike infrastructure on this carriageway" (`separate`: the
# bike way is mapped as its own way).
_CYCLEWAY_NONE: frozenset[str] = frozenset({"no", "none", "separate", "use_sidepath"})


def _motor_traffic_direction(tags: dict) -> bool | None:
    """Direction of motor traffic on a way: True along it, False against it (`oneway=-1`),
    None both ways. Roundabouts are one-way unless tagged otherwise."""
    oneway = tags.get("oneway")
    if oneway in ("yes", "true", "1") or (
            tags.get("junction") in ("roundabout", "circular") and oneway != "no"):
        return True
    if oneway == "-1":
        return False
    return None


def motor_traffic_oneway(tags: dict) -> bool:
    """Whether motor traffic on this way is one-way (bike traffic stress)."""
    return _motor_traffic_direction(tags) is not None


def cycleway_in_direction(tags: dict, forward: bool, left_hand_traffic: bool = False) -> str:
    """The bike infrastructure (`cycleway*` value) for travel along the OSM way (`forward`)
    or against it; `'no'` if there is none.

      - `cycleway` and `cycleway:both` serve both directions of a two-way road and the traffic
        direction of a one-way road (`cycleway:both` also the other one if bikes may ride
        against it, `oneway:bicycle=no`); contraflow values (`opposite*`) serve the direction
        against the traffic.
      - Two-way road: the lane on the traffic side (right; left with `left_hand_traffic`)
        serves the way's direction, the other side the opposite one.
      - One-way road: the traffic-side lane serves the traffic direction; a lane on the other
        side is the contraflow lane if its value says so or bikes may ride against the
        traffic, else it also serves the traffic direction.

    Several values for one direction: separated before painted before the rest.
    """
    traffic = _motor_traffic_direction(tags)
    contraflow_ok = tags.get("oneway:bicycle") == "no"
    near, far = (("cycleway:left", "cycleway:right") if left_hand_traffic
                 else ("cycleway:right", "cycleway:left"))
    if traffic is False:            # traffic against the way: its side is the way's other one
        near, far = far, near
    found: list[str] = []

    def add(value, serves: bool | None) -> None:        # serves: way direction, against, both
        if (isinstance(value, str) and value not in _CYCLEWAY_NONE
                and (serves is None or serves == forward)):
            found.append(value)

    for key in ("cycleway", "cycleway:both"):
        value = tags.get(key)
        if isinstance(value, str) and value.startswith("opposite"):
            add(value, False if traffic is None else not traffic)
        elif traffic is None:
            add(value, None)
        else:
            add(value, traffic)
            if key == "cycleway:both" and contraflow_ok:
                add(value, not traffic)
    if traffic is None:
        add(tags.get(near), True)
        add(tags.get(far), False)
    else:
        add(tags.get(near), traffic)
        value = tags.get(far)
        contraflow = isinstance(value, str) and (value.startswith("opposite") or contraflow_ok)
        add(value, (not traffic) if contraflow else traffic)
    for group in (_CYCLEWAY_SEPARATED, _CYCLEWAY_PAINTED):
        for value in found:
            if value in group:
                return value
    return found[0] if found else "no"


# Rows of levels when separated / with a painted lane or shared bus lane / in mixed traffic,
# from least to most stress. Untagged speed: 50 km/h on main roads, quiet (≤ 30) on local
# streets; untagged lanes: 1 per direction.
_STRESS_ROWS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 1),    # ≤ 20 km/h (living street, meeting zone), cycle street
    (0, 1, 2),    # local street ≤ 30 km/h or untagged
    (1, 2, 3),    # main road ≤ 30 km/h; local street > 30 km/h (tagged; rural: little traffic)
    (2, 3, 4),    # main road 50 km/h (or untagged), 1 lane per direction
    (3, 4, 5),    # main road 50 km/h with 2+ lanes per direction, or > 50 km/h
)
# Slope (rise / run, in the edge's direction) from which a climb moves a row. The run is at
# least `STRESS_SLOPE_MIN_LENGTH_M`: shorter edges are below what a 30 m DEM resolves (their
# slopes are mostly noise), and a short ramp hardly slows a cyclist.
STRESS_SLOPE = 0.05
STRESS_SLOPE_MIN_LENGTH_M = 100.0


def lanes_in_direction(tags: dict, forward: bool) -> float | None:
    """Motor-traffic lanes for travel along the OSM way (`forward`) or against it, without
    bus lanes: `lanes:forward|backward` where tagged, else half of `lanes` on two-way roads
    and all of them in a one-way road's direction (0 against it). None if `lanes` is not
    tagged."""
    traffic = _motor_traffic_direction(tags)
    total = _parse_lanes(tags.get("lanes"))
    if traffic is not None:
        if forward != traffic:
            return 0.0
        lanes = total
        side = "forward" if traffic else "backward"
        bus = next((n for n in (_parse_lanes(tags.get(k)) for k in (
            "lanes:bus", "lanes:psv", f"lanes:bus:{side}", f"lanes:psv:{side}")) if n), 0.0)
    else:
        side = "forward" if forward else "backward"
        lanes = _parse_lanes(tags.get(f"lanes:{side}"))
        if lanes is None and total is not None:
            lanes = total / 2
        bus = next((n for n in (_parse_lanes(tags.get(f"lanes:{k}:{side}"))
                                for k in ("bus", "psv")) if n), None)
        if bus is None:
            both = next((n for n in (_parse_lanes(tags.get(f"lanes:{k}"))
                                     for k in ("bus", "psv")) if n), 0.0)
            bus = both / 2
    return None if lanes is None else max(lanes - bus, 0.0)


def _road_row(data: dict) -> int:
    """Index into `_STRESS_ROWS` for riding on this road, before modifiers."""
    highways = [h for h in _tag_values(data, "highway") if h]
    speeds = [s for s in (parse_maxspeed_kmh(v) for v in _tag_values(data, "maxspeed"))
              if s is not None]
    speed = max(speeds) if speeds else None
    if ("living_street" in highways or (speed is not None and speed <= 20)
            or any(c in ("yes", True, "true", "1", 1) for c in _tag_values(data, "cyclestreet"))):
        return 0
    main = any(h in STRESS_MAIN_ROADS for h in highways)
    lanes = max([n for n in (_parse_lanes(v) for v in _tag_values(data, "lanes_dir"))
                 if n is not None] or [1.0])
    if speed is None:
        speed = 50.0 if main else 30.0
    if not main:
        return 1 if speed <= 30 else 2
    if speed <= 30:
        return 2
    return 4 if speed > 50 or lanes >= 2 else 3


def bike_stress_beside(data: dict) -> int:
    """The level a way separated from motor traffic gets beside this road: the table's
    `separated` column (0 beside a quiet street … 3 beside a fast or multi-lane main road)."""
    return _STRESS_ROWS[_road_row(data)][0]


def bike_stress(data: dict, beside: int | None = None, slope: float | None = None) -> float:
    """Traffic stress of riding this bike-network edge, 0 (none) to 5 (very high); NaN where
    the bike has to be pushed (`bike_push`).

      - ways without motor traffic (cycleways, paths, tracks, footways with bike access, roads
        closed to cars): `beside`, the `bike_stress_beside` level of the road running alongside,
        or 0 if none does;
      - roads: the row by speed limit, road class and lanes in the edge's direction
        (`lanes_dir`; `_STRESS_ROWS`), the column by the bike infrastructure in the edge's
        direction (`cycleway_dir`): a cycle track as separated, a lane or shared bus lane as
        painted; shared markings and contraflow without a lane as mixed traffic. Painted and
        mixed: one row less if motor traffic is one-way (`oneway_motor`), one row more if the
        edge climbs by `STRESS_SLOPE` or more (`slope`: rise in the edge's direction over
        its length, at least `STRESS_SLOPE_MIN_LENGTH_M`); both cancel out. `cycleway_dir`,
        `lanes_dir` and `oneway_motor` are set by `networks_from_pbf`.
    """
    if bike_push(data):
        return float("nan")
    highways = [h for h in _tag_values(data, "highway") if h]
    car_tags = [v for k in ("motor_vehicle", "motorcar") for v in _tag_values(data, k)]
    closed_to_cars = "no" in car_tags or (
        any(a in ("no", "private") for a in _tag_values(data, "access"))
        and not any(v in _DRIVE_ACCESS_ALLOWED for v in car_tags))
    if all(h in _CAR_FREE_HIGHWAYS for h in highways) or closed_to_cars:
        return float(beside if beside is not None else 0)
    cycleways = {c for c in _tag_values(data, "cycleway_dir") if isinstance(c, str)}
    row = _road_row(data)
    if cycleways & _CYCLEWAY_SEPARATED:
        return float(_STRESS_ROWS[row][0])
    oneway = [n for n in (_parse_lanes(v) for v in _tag_values(data, "oneway_motor"))
              if n is not None]
    shift = int(bool(oneway) and min(oneway) >= 0.5)
    if slope is not None and slope >= STRESS_SLOPE:          # NaN compares False
        shift -= 1
    row = min(max(row - shift, 0), len(_STRESS_ROWS) - 1)
    return float(_STRESS_ROWS[row][1 if cycleways & _CYCLEWAY_PAINTED else 2])


# =====================================================================
# OSM highway-rank machinery + per-node OSM classification.
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
    # Per-node max / min highway rank from incident edges (one item per parallel edge
    # on multigraphs too).
    node_max = {n: float("-inf") for n in graph.nodes}
    node_min = {n: float("inf") for n in graph.nodes}
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
