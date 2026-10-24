"""Technical Validation scenarios for the aperta-atlas data descriptor.

Purpose: isolate the effect of individual pipeline parameters on the
released accessibility surfaces, holding every other parameter constant.
Five axes, all feeding the Technical Validation section of the data
descriptor (``papers/atlas-pipeline``). Each scenario's prefix names what
varies:

- **`res-*`** — cell resolution ladder. Varies the cell layer
  (buildings / H3 res 9-11), holds OD-tier radii and everything else
  constant. Each grid is compared against `res-buildings`, the finest.
- **`radii-*`** — OD-tier radii ladder. Varies `(r_cells, r_medium)`
  on the H3-res-11 layer, holds everything else constant. Compared,
  with `res-h11` (default radii), against untiered (exact) accessibility
  by `technical_validation/access_exact_sample.py`.
- **`cv-*`** — spatial cross-validation across Swiss language regions.
  `cv-de-train` and `cv-fr-train` calibrate on 17 DE-speaking cantons
  and 4 FR-speaking cantons respectively (multilingual cantons + Ticino
  excluded — Ticino would be an isolated island). `cv-fr-test` /
  `cv-de-test` apply the swapped calibration via `ImportFrom`. Each test
  scenario is compared against its same-area train twin.
- **`data-*`** — population / employment data-source alternatives
  (bern-metro, H3 res 10). `data-coef` uses dasymetric-per-OSM-tag
  coefficients for BOTH pop and emp; `data-ghs` uses GHS-POP for pop
  (STATENT-native emp). Compared against `res-h10` (STATPOP + STATENT
  native).
- **`network-*`** — road network alternative. `network-tlmregio` routes cars
  on the swissTLMRegio main roads (no local streets) instead of OSM,
  calibrated nationally like `switzerland-h10` and compared against it (car,
  time-based).

What is compared against what is listed per headline scenario in
`TECHNICAL_VALIDATION` at the bottom of this module. The comparison scripts
(`technical_validation/access_exact_sample.py`, `access_tv_groups.py`,
`access_network_free.py`) run under the headline scenario (default
`switzerland-h10`), read the TV scenarios from their own storage, and write
their results under the headline scenario's `technical_validation/`.

Kept in a sibling file to avoid clogging ``scenarios.py`` — merged into
``scenarios.SCENARIOS`` at the bottom of that file.

Parameter choices shared by the res-*, radii-* and data-* scenarios
(the cv-* scenarios are described in their own section below):

- ``area_name='bern-metro'`` — Verwaltungskreis Bern-Mittelland,
  ~940 km², ~415k population (close proxy for BFS's "Agglomeration
  Bern" of ~410k pop). Defined in ``preparation/world/areas.py`` with
  buffers walk=5000 / bike=10000 / car=20000 metres — sized so a
  ~20 min radius and/or k=10 destinations are captured for all modes.
  About 6x smaller than Canton of Bern, so building-level runs remain
  tractable.

- ``population_source='statpop'``, ``employment_source='statent'``
  (except data-*) — the native hectare surfaces serve as the reference.
  ``storage=Storage.PUBLIC`` because the scenarios don't recalibrate:
  all coefs are ImportFrom, reproducible with the released coef bundle.

- ``coefs=_IMPORTED_FROM_SWISS`` — all coefficients inherit from
  ``switzerland-h10`` (exogenous). Testing a parameter's effect under
  fixed coefficients matches what a data reuser experiences: they get
  one released coef bundle and apply it with whatever data and
  resolution they have.

- ``mode_configs`` — walk, bike, car with one profile each (``rwalk``,
  ``rbike``, ``car_base``), plus transit (``transit_source='npvm'``:
  NPVM zone-to-zone times plus per-cell overheads).
  Reduced-mobility / e-bike variants are excluded (out of scope here).

- ``accessibility_grids`` — two compact grids with one setting per metric
  family at each of three reaches (counts up to 10 / 15 / 30 min, k = 1 /
  10 / 30, half-decays of 1 / 2 / 5 min, utility β = 2 / 1 / 0.7), matched
  by effective reach (see the grid definition below).
  Three destination types with contrasting spatial patterns
  (dense-urban clustered, mixed, tightly-clustered): grocery POIs,
  secondary employment, gastronomy POIs. Time-based and utility-based
  variants; distance-line is skipped as least-informative for resolution.

The res-* scenarios span the cell-resolution ladder:

    res-buildings   OSM building centroids  — dasymetric-per-building
    res-h11         H3 res 11 (~2 149 m²)   — very fine
    res-h10         H3 res 10 (~15 047 m²)  — atlas default baseline
    res-h9          H3 res 9 (~105 300 m²)  — coarse

A `res-hectares` scenario (100 m Swiss squares, STATPOP-native) is
supported by ``main/01_cells_zones.py`` but kept commented out below.
"""
from dataclasses import dataclass, replace

# The `_`-prefixed presets are module-private in `scenarios.py` but imported
# here on purpose, so both modules stay in sync when a preset changes.
from aperta_atlas.coefs import Calibrate, ImportFrom
from aperta_atlas.context import Storage
from mode_configs import MODE_CONFIGS, OdmRadii
from scenarios import (
    AccessibilityGrid,
    Scenario,
    SnapConfig,
    _IMPORTED_FROM_SWISS,
    _SWISS_ACCESSIBILITY_GRIDS,
    _SWISS_CALIBRATED_COEFS,
)


# MODE_CONFIGS with a single profile per mode; all other ModeConfig fields
# unchanged.
_TV_MODE_CONFIGS = {
    'walk': replace(MODE_CONFIGS['walk'], profiles=['rwalk']),
    'bike': replace(MODE_CONFIGS['bike'], profiles=['rbike']),
    'car':  replace(MODE_CONFIGS['car'],  profiles=['base']),
}


# Pared-down accessibility grid pair covering three reaches, one setting per metric
# family each (see `technical_validation.access_vs_reference.REACH`):
#   local   counts up to 10 min, k = 1,  1-min half-decay, utility β = 2
#   medium  counts up to 15 min, k = 10, 2-min half-decay, utility β = 1
#   broad   counts up to 30 min, k = 30, 5-min half-decay, utility β = 0.7
# All are released settings (counts: sums of the released bins), picked so that
# their effective reach (r50, main/11_access_reach) is about equal within a
# category on average over the modes (Bern-Mittelland: ~1, 1.6 and 3 km); per mode
# it still differs. Buffers were sized for ~k = 10, so walking k = 30 may hit the
# buffer edge for border cells. Three destination types with contrasting spatial
# patterns:
#   - poi_errands_groceries    — dense urban, sparse rural
#   - employment_secondary     — mixed / continuous
#   - poi_leisure_gastronomy   — tightly clustered in city centres
# Distance-based grid intentionally omitted (least informative here).
_TV_DEST_COLS = (
    'poi_errands_groceries',
    'employment_secondary',
    'poi_leisure_gastronomy',
)
_TV_ACCESSIBILITY_GRIDS = {
    'grid_time': AccessibilityGrid(
        bin_edges_min=(0, 10, 15, 30),
        nearest_k=(1, 10, 30),
        gravity_half_decay_min=(1, 2, 5),
        dest_cols=_TV_DEST_COLS,
        travel_cost='time_gross',
    ),
    'grid_util': AccessibilityGrid(
        bin_edges_min=(),
        nearest_k=(1, 10, 30),
        gravity_util_betas=(0.7, 1.0, 2.0),
        dest_cols=_TV_DEST_COLS,
        travel_cost='util',
    ),
}


# data-* change only population / employment, so their grids use only those
# destinations. Their baseline `res-h10` carries them in addition to the three
# resolution-test destinations (comparisons use the columns both sides share).
_DATA_DEST_COLS = ('population_total', 'employment_total')
_DATA_ACCESSIBILITY_GRIDS = {
    k: replace(g, dest_cols=_DATA_DEST_COLS) for k, g in _TV_ACCESSIBILITY_GRIDS.items()
}
_RES_H10_ACCESSIBILITY_GRIDS = {
    k: replace(g, dest_cols=_TV_DEST_COLS + _DATA_DEST_COLS)
    for k, g in _TV_ACCESSIBILITY_GRIDS.items()
}


# Fields shared across every res-* and radii-* scenario.
_TV_COMMON = dict(
    area_name='bern-metro',
    transit_source='npvm',
    population_source='statpop',
    employment_source='statent',
    storage=Storage.PUBLIC,
    coefs=_IMPORTED_FROM_SWISS,
    mode_configs=_TV_MODE_CONFIGS,
    accessibility_grids=_TV_ACCESSIBILITY_GRIDS,
)


# Alternate OD-tier radii for the `radii-*` scenarios. All use H3-11 cells, so
# only `(r_cells, r_medium)` varies against the default `_UNIFIED_RADII`
# (1500 / 7500 / 100000). `r_zones` stays at 100 km: reducing it would cut
# into transit z2z pairs that matter.
#
#   `_ALT_RADII_1` (750 / 3000)    both tighter.
#   `_ALT_RADII_2` (1000 / 7500)   tighter `r_cells` only.
#   `_ALT_RADII_3` (1500 / 5000)   tighter `r_medium` only.
#   `_ALT_RADII_4` (2500 / 15000)  more generous: checks that the default sits
#                                  at the accuracy plateau (car is the mode to
#                                  watch, where a 7500 m `r_medium` cuts into
#                                  the ~8-15 min reach).
_ALT_RADII_1 = OdmRadii(r_cells=750.0, r_medium=3_000.0,  r_zones=100_000.0)
_ALT_RADII_2 = OdmRadii(r_cells=1_000.0, r_medium=7_500.0,  r_zones=100_000.0)
_ALT_RADII_3 = OdmRadii(r_cells=1_500.0, r_medium=5_000.0,  r_zones=100_000.0)
_ALT_RADII_4 = OdmRadii(r_cells=2_500.0, r_medium=15_000.0, r_zones=100_000.0)


def _tv_mode_configs_with(radii: OdmRadii) -> dict:
    """`_TV_MODE_CONFIGS` with `radii` overridden on every mode. Keeps
    the unified-radii property (needed by cross-modal aggregation)."""
    return {m: replace(cfg, radii=radii) for m, cfg in _TV_MODE_CONFIGS.items()}


SCENARIOS: dict[str, Scenario] = {
    # ---- Cell resolution ladder (radii held at default) ---------------
    'res-buildings': Scenario(
        name='res-buildings',
        cell_source='buildings',
        **_TV_COMMON,
    ),
    # 'res-hectares': Scenario(
    #     name='res-hectares',
    #     cell_source='hectares',
    #     **_TV_COMMON,
    # ),
    'res-h11': Scenario(
        name='res-h11',
        cell_source='h3',
        cell_h3_resolution=11,
        **_TV_COMMON,
    ),
    'res-h10': Scenario(
        name='res-h10',
        cell_source='h3',
        cell_h3_resolution=10,
        **{**_TV_COMMON, 'accessibility_grids': _RES_H10_ACCESSIBILITY_GRIDS},
    ),
    'res-h9': Scenario(
        name='res-h9',
        cell_source='h3',
        cell_h3_resolution=9,
        **_TV_COMMON,
    ),
    # ---- Radii ladder (H11 cells, OD-tier radii varied) ----------------
    # Compared against untiered accessibility (`technical_validation/access_exact_sample.py`).
    'radii-750-3000': Scenario(
        name='radii-750-3000',
        cell_source='h3',
        cell_h3_resolution=11,
        **{**_TV_COMMON, 'mode_configs': _tv_mode_configs_with(_ALT_RADII_1)},
    ),
    'radii-1000-7500': Scenario(
        name='radii-1000-7500',
        cell_source='h3',
        cell_h3_resolution=11,
        **{**_TV_COMMON, 'mode_configs': _tv_mode_configs_with(_ALT_RADII_2)},
    ),
    'radii-1500-5000': Scenario(
        name='radii-1500-5000',
        cell_source='h3',
        cell_h3_resolution=11,
        **{**_TV_COMMON, 'mode_configs': _tv_mode_configs_with(_ALT_RADII_3)},
    ),
    'radii-2500-15000': Scenario(
        name='radii-2500-15000',
        cell_source='h3',
        cell_h3_resolution=11,
        **{**_TV_COMMON, 'mode_configs': _tv_mode_configs_with(_ALT_RADII_4)},
    ),
}


# ---------------------------------------------------------------------------
# Spatial cross-validation (`cv-*`)
# ---------------------------------------------------------------------------
#
# Four scenarios in two paired directions:
#
#   cv-de-train        → trains on 17 DE-speaking cantons (BE excluded)
#   cv-fr-test         → applies cv-de-train's coefs to 4 FR cantons
#
#   cv-fr-train        → trains on 4 FR cantons
#   cv-de-test         → applies cv-fr-train's coefs to 17 DE cantons
#
# The cv-*-train scenarios use `_SWISS_CALIBRATED_COEFS` (Calibrate() for
# everything survey/counter-driven). The cv-*-test scenarios ImportFrom
# the matching train scenario. Comparison targets for the paper are
# SELF-PAIRED — each cv-*-test compares against its same-area cv-*-train
# twin (not against `switzerland-h10`), so buffer / network extent
# are identical and only the training-region axis varies:
#
#   cv-fr-test (DE-trained coefs)  ↔  cv-fr-train (FR-trained coefs)
#   cv-de-test (FR-trained coefs)  ↔  cv-de-train (DE-trained coefs)
#
# `switzerland-h10` stays out of the CV validation and stands on its
# own (it uses wider buffers). Storage.PRIVATE because the training
# pipelines consume restricted MTMC + MOBIS microdata; the test
# scenarios inherit PRIVATE for output collocation.
#
# Region scoping: calibration stages (03a, 04, 07a, 07b, 09a) restrict their
# training legs / counters to the scenario's AOI via `aoi_filter.py`, requiring
# BOTH leg endpoints inside so cross-region legs don't leak out-of-region
# behaviour into training. For `switzerland-h10` the AOI is all of Switzerland,
# so the filter passes every in-country leg through.

# Full ImportFrom map from a given cv-*-train scenario. Mirrors
# `_IMPORTED_FROM_SWISS` in scenarios.py but points at the CV twin.
def _imported_from(src_scenario: str) -> dict:
    return {name: ImportFrom(src_scenario) for name in _SWISS_CALIBRATED_COEFS}


# CV scenarios pin `cell_source='h3'` + `cell_h3_resolution=10` explicitly
# (matches switzerland-h10) so the CV comparison is decoupled from any
# future change to the Scenario defaults. Mode configs are the full three-
# mode set (walk / bike / car with all profiles) — spatial generalisation
# is a full-model question, not a resolution one.
#
# Compact accessibility_grids: CV asks "does test-region accessibility
# match train-region accessibility on the same cells?", so we need a
# handful of representative metrics rather than the full default grid.
# `dist_line` is omitted because calibration doesn't affect straight-
# line distance — no signal there.
_CV_DEST_COLS = (
    'population_total',           # who can reach whom
    'employment_total',           # canonical accessibility target
    'poi_errands_groceries',      # daily-life POI
    'mobility_transit',           # transit-stop access
)
_CV_ACCESSIBILITY_GRIDS = {
    'cv_time': AccessibilityGrid(
        bin_edges_min=(0, 15, 30, 60),
        nearest_k=(1, 10),
        gravity_half_decay_min=(15, 30),
        dest_cols=_CV_DEST_COLS,
        travel_cost='time_gross',
    ),
    'cv_util': AccessibilityGrid(
        bin_edges_min=(),
        nearest_k=(1, 10),
        gravity_util_betas=(1.0,),
        dest_cols=_CV_DEST_COLS,
        travel_cost='util',
    ),
}
_CV_COMMON = dict(
    transit_source='npvm',
    cell_source='h3',
    cell_h3_resolution=10,
    population_source='statpop',
    employment_source='statent',
    storage=Storage.PRIVATE,
    accessibility_grids=_CV_ACCESSIBILITY_GRIDS,
)


SCENARIOS.update({
    # ---- Train pair: DE trains, FR applies -----------------------------
    'cv-de-train': Scenario(
        name='cv-de-train',
        area_name='switzerland-de',
        coefs=_SWISS_CALIBRATED_COEFS,
        **_CV_COMMON,
    ),
    'cv-fr-test': Scenario(
        name='cv-fr-test',
        area_name='switzerland-fr',
        coefs=_imported_from('cv-de-train'),
        **_CV_COMMON,
    ),
    # ---- Train pair: FR trains, DE applies -----------------------------
    'cv-fr-train': Scenario(
        name='cv-fr-train',
        area_name='switzerland-fr',
        coefs=_SWISS_CALIBRATED_COEFS,
        **_CV_COMMON,
    ),
    'cv-de-test': Scenario(
        name='cv-de-test',
        area_name='switzerland-de',
        coefs=_imported_from('cv-fr-train'),
        **_CV_COMMON,
    ),
})


# ---------------------------------------------------------------------------
# Data-source alternatives (`data-*`)
# ---------------------------------------------------------------------------
#
# Two scenarios that share `res-h10`'s spatial resolution, coefs and mode
# configs but swap the pop/emp DATA sources. Compared against `res-h10`
# (STATPOP + STATENT native) to isolate the data-source effect:
#
#   - data-coef: dasymetric-per-OSM-tag coefficients for BOTH pop and emp
#                (STATPOP-calibrated + STATENT-calibrated intensities × OSM
#                building area). Tests the full dasymetric-method vs the
#                native-microdata reference.
#   - data-ghs:  GHS-POP for population, STATENT native for employment.
#                Tests the GHS global-data alternative for pop only.
#
# Pop and emp accessibilities are essentially independent (pop uses pop
# columns as destinations, emp uses emp columns), so:
#   * res-h10 vs data-coef  → pop-access diff = coef-pop effect
#                             emp-access diff = coef-emp effect
#   * res-h10 vs data-ghs   → pop-access diff = GHS-pop effect
#                             (emp-access identical: same STATENT source)
#
# `storage=Storage.PUBLIC` — coefs are imported from `switzerland-h10` (as in
# res-h10), so no restricted-microdata dependency at run time.

_DATA_COMMON = dict(
    area_name='bern-metro',
    transit_source='npvm',
    # Public per-building pop / emp files for this area (default 'switzerland'
    # would need the Swiss-wide coef / GHS runs; per-building values are the same).
    ghs_pop_area_name='bern-metro',
    cell_source='h3',
    cell_h3_resolution=10,
    storage=Storage.PUBLIC,
    coefs=_IMPORTED_FROM_SWISS,
    mode_configs=_TV_MODE_CONFIGS,
    accessibility_grids=_DATA_ACCESSIBILITY_GRIDS,
)

SCENARIOS.update({
    'data-coef': Scenario(
        name='data-coef',
        population_source='coef',
        employment_source='coef',
        **_DATA_COMMON,
    ),
    'data-ghs': Scenario(
        name='data-ghs',
        population_source='ghs',
        employment_source='statent',
        **_DATA_COMMON,
    ),
})


# ---------------------------------------------------------------------------
# Road network alternative (`network-*`)
# ---------------------------------------------------------------------------
#
# `network-tlmregio`: Switzerland with cars routed on the swissTLMRegio main
# roads (`preparation/switzerland/public/tlmregio`; 1:200,000, motorways to minor
# roads, no local streets), as in LUMOS. Tests how much a coarse road dataset
# changes car accessibility, e.g. for regions without a detailed network.
#
#   - Car only (night / base / peak), no transit; compared against
#     `switzerland-h10` on the car time-based columns.
#   - Edge weights and road overheads are fitted on this network (04: 2 × 2
#     classes motorway / other × urban / rural plus congestion; 07a). The
#     flow model's trip weights and cost bins are taken from `switzerland-h10`
#     and only the flows are re-assigned on the coarse network; the traffic
#     counters serve to validate them (`technical_validation/flows_vs_counters.py`).
#   - Cells and survey endpoints snap up to 2 km, so that cells away from the
#     main roads still connect; the walk to the road is part of the fitted
#     `snap_dist` overheads.
#   - Storage.PRIVATE: calibrated on MTMC / MOBIS microdata.

SCENARIOS.update({
    'network-tlmregio': Scenario(
        name='network-tlmregio',
        area_name='switzerland',
        cell_source='h3',
        cell_h3_resolution=10,
        population_source='statpop',
        employment_source='statent',
        storage=Storage.PRIVATE,
        car_network_source='tlmregio',
        mode_configs={'car': MODE_CONFIGS['car']},
        snap=SnapConfig(max_cell_radius_m=2_000.0, insert_max_radius_m=2_000.0),
        coefs={
            'edge_weights_car':      Calibrate(),
            'overheads_road':        Calibrate(),
            'node_trip_weights_car': ImportFrom('switzerland-h10'),
            'flow_cost_bins_car':    ImportFrom('switzerland-h10'),
        },
        accessibility_grids={'time': _SWISS_ACCESSIBILITY_GRIDS['time']},
    ),
})


# ---------------------------------------------------------------------------
# Comparisons per headline scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TechnicalValidationPlan:
    """The accessibility comparisons behind one headline scenario's Technical
    Validation. The comparison scripts run under the headline scenario and
    write their results there.

    `exact_baseline`: scenario whose inputs (network, cells, coefficients,
        grids) build the untiered accessibility on a sample of origins
        (`access_exact_sample.py`).
    `exact_compare`: scenarios compared against that baseline; they must
        share its cell layer.
    `groups`: (target, reference) pairs per validation axis
        (`access_tv_groups.py`). Each reference differs from its targets only
        in the axis under test.
    `network_free`: scenario whose walking and cycling times are replaced by
        a function of straight-line distance (`access_network_free.py`).
    """
    exact_baseline: str
    exact_compare: tuple[str, ...]
    groups: dict[str, list[tuple[str, str]]]
    network_free: str


TECHNICAL_VALIDATION: dict[str, TechnicalValidationPlan] = {
    'switzerland-h10': TechnicalValidationPlan(
        # Default radii (res-h11) and the radii ladder, all on H3-11 cells.
        exact_baseline='res-h11',
        exact_compare=('res-h11', 'radii-750-3000', 'radii-1000-7500', 'radii-1500-5000',
                       'radii-2500-15000'),
        groups={
            # Each resolution against the finest layer.
            'resolution': [
                ('res-h11', 'res-buildings'),
                ('res-h10', 'res-buildings'),
                ('res-h9', 'res-buildings'),
            ],
            # Each test scenario against its same-area train twin.
            'cv': [
                ('cv-fr-test', 'cv-fr-train'),
                ('cv-de-test', 'cv-de-train'),
            ],
            # Public data sources against STATPOP + STATENT.
            'data': [
                ('data-coef', 'res-h10'),
                ('data-ghs', 'res-h10'),
            ],
            # swissTLMRegio main roads against OSM (car only).
            'network': [
                ('network-tlmregio', 'switzerland-h10'),
            ],
        },
        network_free='res-h10',
    ),
}


def technical_validation_plan(scenario_name: str) -> TechnicalValidationPlan:
    """The plan for `scenario_name`; raises if it is not a headline scenario."""
    if scenario_name not in TECHNICAL_VALIDATION:
        raise ValueError(
            f"No technical-validation plan for scenario {scenario_name!r}. The comparison "
            f"scripts run under a headline scenario: {sorted(TECHNICAL_VALIDATION)} "
            f"(omit --scenario for the default).")
    return TECHNICAL_VALIDATION[scenario_name]


