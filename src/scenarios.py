"""
Typed scenario definitions for the Accessibility Atlas project.

Each `Scenario` declares its `Area` (extent + buffers), pop/employment
data sources (private = STATPOP/STATENT dasymetric, public = GHS-POP
coefficient-based), default `Storage`, and per-coef provenance
(`Calibrate()` / `ImportFrom(...)` / `HandWritten()`).

Scenario names route outputs to `<DATA_DIR_*>/<PROJECT_NAME>/<scenario>/...`
via `init_context`. `DEFAULT_SCENARIO` is the fallback when no
`--scenario` is passed.
"""

from dataclasses import dataclass, field

from aperta_atlas.coefs import Calibrate, CoefSource, ImportFrom
from aperta_atlas.context import Storage
from mode_configs import MODE_CONFIGS, ModeConfig
from preparation.world.areas import AREAS


@dataclass(frozen=True)
class SnapConfig:
    """Per-scenario snap overrides for `02a_networks_snap.py`. Distances in
    `scenario.crs_main` units (meters). Defaults are Swiss-tuned; sparser
    networks (Cambridge UK, rural areas) may want larger radii.

    `max_cell_radius_m` / `max_zone_radius_m`: max search radius for
    snapping cell/zone centroids; beyond this the unit doesn't snap
    (affects `is_active`). `zone_priority_radius_m`: inside this radius
    prefer anchor/major intersections over nearest eligible node.
    `insert_max_radius_m`: virtual-node insertion search radius (should
    match `max_cell_radius_m`). `insert_node_spacing_m`: per-mode
    threshold below which the projection point snaps to the existing
    endpoint instead of inserting a new node.
    """
    max_cell_radius_m: float = 250.0
    max_zone_radius_m: float = 2_500.0
    zone_priority_radius_m: float = 500.0
    insert_max_radius_m: float = 250.0
    insert_node_spacing_m: dict[str, float] = field(default_factory=lambda: {
        'walk': 50.0, 'bike': 100.0, 'car': 150.0,
    })


@dataclass(frozen=True)
class AccessibilityGrid:
    """One accessibility grid for 10's three metric families. A scenario
    may declare several; each becomes a `--variant <grid_key>` in 10.

    Empty tuples DISABLE the corresponding metric family.

    `bin_edges_min`: cumulative-opportunity bin edges (minutes).
    `bin_edges_m`: cumulative-opportunity bin edges (metres). Applies to
        `dist_line` grids only.
    `nearest_k`: nearest-k grid (unitless opportunity counts).
    `gravity_half_decay_min`: half-decay times (minutes) for `exp(-β·t)`
        gravity, `β = ln(2) / (m · 60)`. Applies to `time_gross` grids
        only; ignored on `util` and `dist_line` grids.
    `gravity_util_betas`: β values applied directly to disutility for
        `util` grids (`exp(-β·D)`). `β = 1.0` gives the (exponentiated)
        mode-choice logsum. Ignored on `time_gross` and `dist_line` grids.
    `gravity_half_decay_m`: half-decay distances (metres) for `exp(-β·d)`
        gravity on `dist_line` grids, `β = ln(2) / m`. Ignored otherwise.
    `dest_cols`: subset of destination columns; `None` = pop + emp + poi
        union. Entries must be in the scenario's declared columns.
    `travel_cost`: `'time_gross'` (default, seconds inc. overheads),
        `'util'` (disutility; smaller = better), or `'dist_line'` (raw
        straight-line metres, mode-agnostic). Under `'util'`, coefs come
        from `utility_<utility>` — 09a's fitted β get sign-flipped in
        `build_disutility_spec` so the same downstream primitives
        (`nearest_k`, `floor_intrazonal_costs`, `exp(-β·D)`) apply
        directly. Under `'dist_line'`, the pre-computed straight-line
        distance ODM from 08b feeds the same primitives with metre-domain
        bin edges and decays. `'time_net'` is reserved.
    `utility`: 09a variant name for `'util'` grids; the coefs dict must
        declare a matching `utility_<name>` entry.
    """
    bin_edges_min: tuple[int, ...] = (0, 5, 10, 15, 30, 60)
    bin_edges_m: tuple[float, ...] = (0, 300, 1000, 3_000, 10_000, 30_000)
    nearest_k: tuple[int, ...] = (1, 3, 10, 30, 100)
    gravity_half_decay_min: tuple[float, ...] = (5, 10, 15, 20, 30)
    gravity_util_betas: tuple[float, ...] = (0.5, 0.8, 1.0, 1.5, 2.0)
    gravity_half_decay_m: tuple[float, ...] = (300, 1_000, 3_000, 10_000, 30_000)
    dest_cols: tuple[str, ...] | None = None
    travel_cost: str = 'time_gross'
    utility: str = 'default'


# First segment of on-disk output paths (`<DATA_DIR_*>/<PROJECT_NAME>/…`).
PROJECT_NAME: str = 'atlas'

# Runtime fallback when no `--scenario` is passed.
DEFAULT_SCENARIO: str = 'switzerland-h10'


@dataclass(frozen=True)
class Scenario:
    """Configuration for one atlas scenario.

    Key fields:
      - `name`: scenario id + output-folder name.
      - `area_name`: `preparation/world/areas.py` entry (buildings GPKG +
        per-mode network buffers).
      - `population_source`: `'statpop'` (Swiss high-fidelity STATPOP
        dasymetric, CH only), `'ghs'` (global GHS-POP dasymetric,
        works anywhere), or `'coef'` (per-OSM-tag coefficients from
        the STATPOP calibration, applied to OSM buildings anywhere).
      - `employment_source`: `'statent'` (Swiss high-fidelity STATENT
        dasymetric, CH only) or `'coef'` (per-OSM-tag coefficients from
        the STATENT calibration, applied to OSM buildings anywhere).
      - `edge_weights_sources`: per-mode ground truth for 04's edge-weight
        OLS. Keys are modes (`'walk'`, `'bike'`, `'car'`); values are one
        of `'mtmc'` (MZMV 2015 + 2021 self-reported), `'mobis_precovid'`
        (MOBIS pre-2020 GPS-tracked, ttf-filtered), or `'mobis_covid'`
        (MOBIS COVID-era 2020-wk13 through 2022-03 — anomalous traffic
        regime, use with care). Blending is intentionally NOT supported — pick one source per
        mode; MOBIS otherwise dominates by sample count.
      - `zone_h3_resolution`: `None` = load NPVM Swiss traffic zones;
        non-None = build H3 zones at that resolution (required for
        non-Swiss cases).
      - `crs_main`: metric CRS. Default LV95.
      - `storage`: default `Storage`. PRIVATE for restricted-data scenarios.
      - `coefs`: `{coef_name: CoefSource}` — Calibrate / ImportFrom / HandWritten.
      - `population_cols` / `employment_cols`: columns threaded through
        01's aggregation. Must contain ≥1 `*_total` (used to derive
        `combined_total`). STATPOP/STATENT scenarios can extend with
        age bands / industry codes.
      - `mode_configs`: active modes (dict keys) with their per-mode
        config. Override via `dataclasses.replace`; drop keys to
        deactivate modes.
      - `accessibility_grids`: `{grid_key: AccessibilityGrid}`; each grid
        becomes a `--variant` in 10.
      - `poi_cols` / `transit_stops_cols`: destination columns; the
        transit-stop set is opt-in via a grid's `dest_cols` (not part
        of the default union).
    """
    name: str
    area_name: str
    population_source: str       # 'statpop' | 'ghs' | 'coef'
    employment_source: str       # 'statent' | 'coef'
    statpop_year: str = '2025'
    statent_year: str = '2024'
    ghs_pop_area_name: str = 'switzerland'
    cell_source: str = 'h3'          # 'h3' | 'hectares' | 'buildings'
    cell_h3_resolution: int = 10     # only read when cell_source == 'h3'
    zone_h3_resolution: int | None = None
    crs_main: str = 'EPSG:2056'
    storage: Storage = Storage.PUBLIC
    coefs: dict[str, CoefSource] = field(default_factory=dict)
    population_cols: tuple[str, ...] = ('population_total',)
    employment_cols: tuple[str, ...] = (
        'employment_primary', 'employment_secondary',
        'employment_tertiary', 'employment_total',
    )
    mode_configs: dict[str, ModeConfig] = field(
        default_factory=lambda: dict(MODE_CONFIGS))
    edge_weights_sources: dict[str, str] = field(
        default_factory=lambda: {'walk': 'mtmc', 'bike': 'mtmc', 'car': 'mobis_precovid'})
    snap: SnapConfig = field(default_factory=SnapConfig)
    accessibility_grids: dict[str, AccessibilityGrid] = field(
        default_factory=lambda: {'default': AccessibilityGrid()})
    poi_source: str = 'osm'
    poi_cols: tuple[str, ...] = (
        'poi_errands_groceries', 'poi_errands_services', 'poi_healthcare',
        'poi_education_preschool', 'poi_education_school', 'poi_education_higher',
        'poi_leisure_gastronomy', 'poi_leisure_amenity', 'poi_leisure_hiking', 'poi_leisure_sports',
        'poi_tourism',
    )
    transit_stops_source: str = 'osm'
    transit_stops_cols: tuple[str, ...] = (
        'mobility_transit',
        'mobility_transit_train',
    )

    def __post_init__(self):
        if self.population_source not in ('statpop', 'ghs', 'coef'):
            raise ValueError(f"population_source must be 'statpop'|'ghs'|'coef', "
                             f"got {self.population_source!r}")
        if self.employment_source not in ('statent', 'coef'):
            raise ValueError(f"employment_source must be 'statent'|'coef', "
                             f"got {self.employment_source!r}")
        if self.cell_source not in ('h3', 'hectares', 'buildings'):
            raise ValueError(f"cell_source must be one of "
                             f"('h3','hectares','buildings'); got "
                             f"{self.cell_source!r}.")
        # Hectares are STATPOP/STATENT-native (100 m Swiss grid);
        # non-Swiss areas have no equivalent input data. `is_swiss` is
        # declared on each `Area` entry in `preparation/world/areas.py`.
        if self.cell_source == 'hectares' and not AREAS[self.area_name].is_swiss:
            raise ValueError(f"cell_source='hectares' requires a Swiss "
                             f"area_name (STATPOP is Swiss-specific); "
                             f"got area_name={self.area_name!r}. Set "
                             f"`is_swiss=True` on the Area in "
                             f"`preparation/world/areas.py` if this area "
                             f"is Swiss.")
        if self.zone_h3_resolution is not None and not (
                0 <= self.zone_h3_resolution <= 15):
            raise ValueError(f"zone_h3_resolution must be in [0, 15] or None, "
                             f"got {self.zone_h3_resolution!r}")
        if (self.cell_source == 'h3'
                and self.zone_h3_resolution is not None
                and self.cell_h3_resolution < self.zone_h3_resolution):
            raise ValueError(f"cell_h3_resolution ({self.cell_h3_resolution}) must be "
                             f"≥ zone_h3_resolution ({self.zone_h3_resolution}).")
        for label, cols in (('population_cols', self.population_cols),
                            ('employment_cols', self.employment_cols)):
            if not cols:
                raise ValueError(f"{label} must be non-empty.")
            if not any(c.endswith('_total') for c in cols):
                raise ValueError(f"{label} must contain a *_total column "
                                 f"(01 derives `combined_total`). Got {cols!r}.")
        if not self.mode_configs:
            raise ValueError("mode_configs must be non-empty.")
        unknown = [m for m in self.mode_configs if m not in ('walk', 'bike', 'car')]
        if unknown:
            raise ValueError(f"mode_configs keys must be ⊆ ('walk','bike','car'); "
                             f"got {unknown!r}.")
        _EW_SOURCES = ('mtmc', 'mobis_precovid', 'mobis_covid')
        missing_ew = [m for m in self.mode_configs
                      if m not in self.edge_weights_sources]
        if missing_ew:
            raise ValueError(f"edge_weights_sources missing entries for active "
                             f"mode(s) {missing_ew!r}.")
        bad_ew = {m: v for m, v in self.edge_weights_sources.items()
                  if v not in _EW_SOURCES}
        if bad_ew:
            raise ValueError(f"edge_weights_sources values must be ⊆ {_EW_SOURCES!r}; "
                             f"got {bad_ew!r}.")
        missing_spacing = [m for m in self.mode_configs
                           if m not in self.snap.insert_node_spacing_m]
        if missing_spacing:
            raise ValueError(f"snap.insert_node_spacing_m missing entries for active "
                             f"mode(s) {missing_spacing!r}.")
        for src_label, src_val in (('poi_source', self.poi_source),
                                    ('transit_stops_source',
                                     self.transit_stops_source)):
            if src_val != 'osm':
                raise ValueError(f"{src_label}={src_val!r} — only 'osm' is supported.")
        for label, cols, prefix in (
            ('poi_cols', self.poi_cols, 'poi_'),
            ('transit_stops_cols', self.transit_stops_cols, 'mobility_'),
        ):
            if not cols:
                raise ValueError(f"{label} must be non-empty.")
            bad = [c for c in cols if not c.startswith(prefix)]
            if bad:
                raise ValueError(f"{label} entries must start with {prefix!r}; "
                                 f"got {bad!r}.")
        if not self.accessibility_grids:
            raise ValueError("accessibility_grids must be non-empty.")
        for grid_key, g in self.accessibility_grids.items():
            for edges_name, edges in (('bin_edges_min', g.bin_edges_min),
                                       ('bin_edges_m', g.bin_edges_m)):
                if edges and (
                        len(edges) < 2
                        or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1))
                        or edges[0] < 0):
                    raise ValueError(f"accessibility_grids[{grid_key!r}].{edges_name} "
                                     f"must be strictly increasing, non-negative, ≥ 2 edges "
                                     f"(or empty to disable). Got {edges!r}.")
            if any(k <= 0 for k in g.nearest_k):
                raise ValueError(f"accessibility_grids[{grid_key!r}].nearest_k entries "
                                 f"must be > 0; got {g.nearest_k!r}.")
            if any(m <= 0 for m in g.gravity_half_decay_min):
                raise ValueError(f"accessibility_grids[{grid_key!r}].gravity_half_decay_min "
                                 f"entries must be > 0; got {g.gravity_half_decay_min!r}.")
            if any(b <= 0 for b in g.gravity_util_betas):
                raise ValueError(f"accessibility_grids[{grid_key!r}].gravity_util_betas "
                                 f"entries must be > 0; got {g.gravity_util_betas!r}.")
            if any(m <= 0 for m in g.gravity_half_decay_m):
                raise ValueError(f"accessibility_grids[{grid_key!r}].gravity_half_decay_m "
                                 f"entries must be > 0; got {g.gravity_half_decay_m!r}.")
            if g.travel_cost == 'util':
                gravity_decays = g.gravity_util_betas
                bin_edges = g.bin_edges_min
            elif g.travel_cost == 'dist_line':
                gravity_decays = g.gravity_half_decay_m
                bin_edges = g.bin_edges_m
            else:
                gravity_decays = g.gravity_half_decay_min
                bin_edges = g.bin_edges_min
            if not (bin_edges or g.nearest_k or gravity_decays):
                raise ValueError(f"accessibility_grids[{grid_key!r}] has all three metric "
                                 f"families disabled.")
            if g.dest_cols is not None:
                if not g.dest_cols:
                    raise ValueError(f"accessibility_grids[{grid_key!r}].dest_cols "
                                     f"must be None or non-empty tuple; got ().")
                available = (set(self.population_cols)
                             | set(self.employment_cols)
                             | set(self.poi_cols)
                             | set(self.transit_stops_cols))
                unknown = [c for c in g.dest_cols if c not in available]
                if unknown:
                    raise ValueError(f"accessibility_grids[{grid_key!r}].dest_cols "
                                     f"unknown: {unknown!r}. Available: {sorted(available)}.")
            valid_cost = {'time_gross', 'util', 'dist_line'}
            reserved_cost = {'time_net'}
            if g.travel_cost in reserved_cost:
                raise NotImplementedError(f"accessibility_grids[{grid_key!r}].travel_cost="
                                          f"{g.travel_cost!r} reserved; use {sorted(valid_cost)!r}.")
            if g.travel_cost not in valid_cost:
                raise ValueError(f"accessibility_grids[{grid_key!r}].travel_cost="
                                 f"{g.travel_cost!r} must be one of "
                                 f"{sorted(valid_cost | reserved_cost)!r}.")
            if g.travel_cost == 'util' and g.bin_edges_min:
                raise NotImplementedError(f"accessibility_grids[{grid_key!r}]: "
                                          f"cumulative on utility not implemented — "
                                          f"set `bin_edges_min=()` for util grids.")
        # All util grids share `utility` — 10 loads one coefs file up front.
        util_variants = {g.utility for g in self.accessibility_grids.values()
                         if g.travel_cost == 'util'}
        if len(util_variants) > 1:
            raise ValueError(f"Scenario {self.name!r}: util grids have mixed `utility` "
                             f"values {sorted(util_variants)} — must share one.")
        declared_util = sorted(k for k in self.coefs if k.startswith('utility_'))
        for grid_key, g in self.accessibility_grids.items():
            if g.travel_cost != 'util':
                continue
            coef_name = f'utility_{g.utility}'
            if coef_name not in self.coefs:
                raise ValueError(
                    f"Scenario {self.name!r}, grid {grid_key!r}: "
                    f"travel_cost='util' needs `{coef_name}` in the scenario's "
                    f"`coefs` dict. Currently declared: {declared_util}.")
        # Companion-sidecar validation. Any `<name>_stats` coef must have
        # its parent declared and share the parent's CoefSource type
        # (both Calibrate, or both ImportFrom to the same source). Catches
        # drift in the mirrored declaration at scenario-load time.
        for coef_name, source in self.coefs.items():
            if not coef_name.endswith('_stats'):
                continue
            parent_name = coef_name[:-len('_stats')]
            if parent_name not in self.coefs:
                raise ValueError(
                    f"Scenario {self.name!r}: companion coef {coef_name!r} "
                    f"declared without its parent {parent_name!r}.")
            parent_source = self.coefs[parent_name]
            if type(source) is not type(parent_source):
                raise ValueError(
                    f"Scenario {self.name!r}: companion coef {coef_name!r} "
                    f"has source type {type(source).__name__} but parent "
                    f"{parent_name!r} has {type(parent_source).__name__} — "
                    f"they must match.")
            if isinstance(source, ImportFrom) and isinstance(parent_source, ImportFrom):
                if source.scenario != parent_source.scenario:
                    raise ValueError(
                        f"Scenario {self.name!r}: companion coef {coef_name!r} "
                        f"imports from {source.scenario!r} but parent "
                        f"{parent_name!r} imports from {parent_source.scenario!r} "
                        f"— they must match.")


# switzerland-h10 is the calibration scenario; others import from it
# (cross-area ground-truth data isn't available outside CH).
_SWISS_CALIBRATED_COEFS: dict[str, CoefSource] = {
    'edge_weights_walk':            Calibrate(),
    'edge_weights_bike':            Calibrate(),
    'edge_weights_car':             Calibrate(),
    'overheads_road':               Calibrate(),
    'overheads_transit':            Calibrate(),
    'node_trip_weights_car':        Calibrate(),
    'flow_cost_bins_car':           Calibrate(),
    'utility_default':              Calibrate(),
    'utility_default_stats':        Calibrate(),
}
# Utility-spec comparison variants (09a `default_single` / `default_route`), fitted only in
# switzerland-h10. Kept out of the shared presets because 09b builds every declared
# `utility_*` variant, and the pipeline fits / transfers only `default`.
_SWISS_UTILITY_SPEC_VARIANTS: dict[str, CoefSource] = {
    'utility_default_single':       Calibrate(),
    'utility_default_single_stats': Calibrate(),
    'utility_default_route':        Calibrate(),
    'utility_default_route_stats':  Calibrate(),
}
_IMPORTED_FROM_SWISS: dict[str, CoefSource] = {
    'edge_weights_walk':            ImportFrom('switzerland-h10'),
    'edge_weights_bike':            ImportFrom('switzerland-h10'),
    'edge_weights_car':             ImportFrom('switzerland-h10'),
    'overheads_road':               ImportFrom('switzerland-h10'),
    'overheads_transit':            ImportFrom('switzerland-h10'),
    'node_trip_weights_car':        ImportFrom('switzerland-h10'),
    'flow_cost_bins_car':           ImportFrom('switzerland-h10'),
    'utility_default':              ImportFrom('switzerland-h10'),
    'utility_default_stats':        ImportFrom('switzerland-h10'),
}

SCENARIOS: dict[str, Scenario] = {
    # Switzerland-extent, Swiss high-fidelity STATPOP + STATENT sources.
    # PRIVATE because coefficient calibration uses restricted survey
    # microdata (MTMC + MOBIS + traffic counters) — not reproducible
    # end-to-end without survey access.
    'switzerland-h10': Scenario(
        name='switzerland-h10',
        area_name='switzerland',
        population_source='statpop',
        employment_source='statent',
        storage=Storage.PRIVATE,
        coefs={**_SWISS_CALIBRATED_COEFS, **_SWISS_UTILITY_SPEC_VARIANTS},
        accessibility_grids={
            'time': AccessibilityGrid(
                travel_cost='time_gross',
            ),
            'util': AccessibilityGrid(
                bin_edges_min=(),
                travel_cost='util',
            ),
            'dist': AccessibilityGrid(
                travel_cost='dist_line',
            ),
        },
    ),
    # Cambridge UK — GHS + coef-based; employment β still Swiss-derived.
    'cambridgeuk-public': Scenario(
        name='cambridgeuk-public',
        area_name='cambridgeuk',
        population_source='ghs',
        employment_source='coef',
        ghs_pop_area_name='cambridgeuk',
        zone_h3_resolution=8,
        crs_main='EPSG:27700',  # British National Grid
        coefs=_IMPORTED_FROM_SWISS,
    ),
}


# Technical Validation scenarios (for the aperta-atlas data descriptor
# paper) live in a sibling file to keep this one focused on core project
# scenarios. Merged in here; the sibling only imports the `Scenario`
# class + the `_IMPORTED_FROM_SWISS` preset (both already defined above),
# so no circular-import risk at load time.
from scenarios_technical_validation import SCENARIOS as _TV_SCENARIOS  # noqa: E402
SCENARIOS.update(_TV_SCENARIOS)


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown atlas scenario {name!r}. Available: "
                       f"{sorted(SCENARIOS.keys())}")
    return SCENARIOS[name]


def scenario_needs_survey_prep(scenario: Scenario) -> bool:
    """True if this scenario's coefs include any calibrator that fits
    from survey microdata (MZMV / MOBIS legs).

    Used by the four `survey/*` scripts to self-skip when the pipeline runs a
    fully-imported scenario (e.g. res-*): downstream calibration scripts
    short-circuit via `coefs.resolve`, so producing
    survey_legs/routed/overhead/summary would be wasted work — the outputs have
    no consumer in that run.

    Currently equivalent to "any Calibrate() in scenario.coefs" — every
    calibrator in this codebase is survey-driven. If a non-survey calibrator is
    added later (e.g. traffic-counter-only, GPS-only), narrow this check
    accordingly (e.g. inspect specific coef names).
    """
    return any(isinstance(s, Calibrate) for s in scenario.coefs.values())
