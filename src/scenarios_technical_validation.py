"""Technical Validation scenarios for the aperta-atlas data descriptor.

Purpose: isolate the effect of individual pipeline parameters on the
released accessibility surfaces, holding every other parameter constant.
Four axes so far, all feeding the ``papers/atlas-pipeline`` Technical
Validation section. Each scenario's prefix names what varies:

- **`res-*`** — cell resolution ladder. Varies the cell layer
  (buildings / hectares / H3 res 9-11), holds OD-tier radii and
  everything else constant.
- **`radii-*`** — OD-tier radii ladder. Varies `(r_cells, r_medium)`
  on the H3-res-11 baseline, holds cell layer constant. Compared
  against `res-h11`.
- **`cv-*`** — spatial cross-validation across Swiss language regions.
  `cv-de-train` and `cv-fr-train` calibrate on 17 DE-speaking cantons
  and 4 FR-speaking cantons respectively (multilingual cantons + Ticino
  excluded — Ticino would be an isolated island). `cv-fr-test` /
  `cv-de-test` apply the swapped calibration via `ImportFrom`. Compared
  self-paired (see the cv-* section below).
- **`data-*`** — pop/emp data-source alternatives (at bern-metro scale,
  H3 res 10). `data-coef` uses dasymetric-per-OSM-tag coefficients for
  BOTH pop and emp; `data-ghs` uses GHS-POP for pop (STATENT-native
  emp). Compared against `res-h10` (STATPOP+STATENT native reference).

Kept in a sibling file to avoid clogging ``scenarios.py`` — merged into
``scenarios.SCENARIOS`` at the bottom of that file.

Parameter choices constant across the four scenarios (rationale):

- ``area_name='bern-metro'`` — Verwaltungskreis Bern-Mittelland,
  ~940 km², ~415k population (close proxy for BFS's "Agglomeration
  Bern" of ~410k pop). Defined in ``preparation/world/areas.py`` with
  buffers walk=5000 / bike=10000 / car=20000 metres — sized so a
  ~20 min radius and/or k=10 destinations are captured for all modes.
  About 6x smaller than Canton of Bern, so building-level runs remain
  tractable.

- ``population_source='statpop'``, ``employment_source='statent'``,
  ``storage=Storage.PUBLIC`` — the brief's §4.3 item 2 reference is
  the "true hectare STATPOP/STATENT surfaces". Storage is PUBLIC
  because the scenarios don't recalibrate — all coefs are ImportFrom;
  reproducible by any collaborator with the shipped coef bundle.

- ``coefs=_IMPORTED_FROM_SWISS`` — all coefficients inherit from
  ``switzerland-h10`` (exogenous). Testing the pure resolution
  effect under fixed coefficients matches what a data reuser
  experiences: they get one released coef bundle and apply it at
  whatever resolution their data supports.

- ``mode_configs`` — walk, bike, car; one profile each (``rwalk``,
  ``rbike``, ``car_base``). Transit is excluded (NPVM-inherited, a
  separate validation topic). Reduced-mobility / e-bike variants are
  excluded (out of scope for the resolution question).

- ``accessibility_grids`` — two compact grids targeting the metric ×
  parameter combinations that maximise visible resolution sensitivity:
  short bins (0-15 min), low nearest-k (1, 3, 10), fast-decay gravity.
  Three destination types with contrasting spatial patterns
  (dense-urban clustered, mixed, tightly-clustered): grocery POIs,
  secondary employment, gastronomy POIs. Time-based and utility-based
  variants; distance-line is skipped as least-informative for resolution.

The res-* scenarios span the cell-resolution ladder:

    res-buildings   OSM building centroids  — dasymetric-per-building
    res-hectares    100 m Swiss squares     — STATPOP-native pass-through
    res-h11         H3 res 11 (~2 149 m²)   — very fine
    res-h10         H3 res 10 (~15 047 m²)  — atlas default baseline
    res-h9          H3 res 9 (~105 300 m²)  — coarse

The ``buildings`` and ``hectares`` scenarios require branching in
``main/01_cells_zones.py`` that has not yet been implemented (planned
as step 2 of the implementation plan). The H3 scenarios can already
run against the existing H3 code path.
"""

from dataclasses import replace

# NOTE: `_IMPORTED_FROM_SWISS` + `_SWISS_CALIBRATED_COEFS` are module-
# private in `scenarios.py` but intentionally reached into here so both
# modules stay in sync when the preset is updated. Alternative would be
# to duplicate the ~10-entry dicts; the import is the smaller footprint.
from aperta_atlas.coefs import ImportFrom
from aperta_atlas.context import Storage
from mode_configs import MODE_CONFIGS, OdmRadii
from scenarios import (
    AccessibilityGrid,
    Scenario,
    _IMPORTED_FROM_SWISS,
    _SWISS_CALIBRATED_COEFS,
)


# Per-mode single-profile MODE_CONFIGS override.
# Uses `dataclasses.replace` to keep the rest of each ModeConfig
# (calibrated_sources, min_route_*, radii, time_cutoff_s, snap-eligible
# node predicates) unchanged.
_TV_MODE_CONFIGS = {
    'walk': replace(MODE_CONFIGS['walk'], profiles=['rwalk']),
    'bike': replace(MODE_CONFIGS['bike'], profiles=['rbike']),
    'car':  replace(MODE_CONFIGS['car'],  profiles=['base']),
}


# Pared-down accessibility grid pair, targeting the parameter × metric
# combinations that are maximally sensitive to cell resolution: short
# bins (0-15 min), low nearest-k (1 is quantum-sensitive), fast-decay
# gravity. Three destination types with contrasting spatial patterns:
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
        bin_edges_min=(0, 5, 10, 15),
        nearest_k=(1, 3, 10),
        gravity_half_decay_min=(5, 10),
        dest_cols=_TV_DEST_COLS,
        travel_cost='time_gross',
    ),
    'grid_util': AccessibilityGrid(
        bin_edges_min=(),
        nearest_k=(1, 3, 10),
        gravity_util_betas=(1.0, 2.0),
        dest_cols=_TV_DEST_COLS,
        travel_cost='util',
    ),
}


# Fields shared across every res-* and radii-* scenario.
_TV_COMMON = dict(
    area_name='bern-metro',
    population_source='statpop',
    employment_source='statent',
    storage=Storage.PUBLIC,
    coefs=_IMPORTED_FROM_SWISS,
    mode_configs=_TV_MODE_CONFIGS,
    accessibility_grids=_TV_ACCESSIBILITY_GRIDS,
)


# Alternate OD-tier radii for the `radii-*` scenarios. Cell layer is
# held at H3 res 11 across all radii scenarios so the ONLY axis varying
# is `(r_cells, r_medium)` — probing whether the default `_UNIFIED_RADII`
# (1500 / 7500 / 100000) is generous or load-bearing for accessibility
# accuracy at the walk/bike scale. `r_zones` kept at 100 km on every
# alternate — reducing it would cut into transit z2z pairs that matter.
#
# `_ALT_RADII_3` (1500 / 5000) is an isolation test: same `r_cells` as
# the baseline, tighter `r_medium` alone. Comparing 3 vs baseline vs 1
# (1000 / 5000) decomposes whether walk sensitivity comes from `r_cells`
# reduction (expected — r_cells sits within walk's typical reach) or
# `r_medium` reduction (expected to affect bike/car more than walk since
# `r_medium=5000` is well beyond walk's typical range).
#
# `_ALT_RADII_4` (2000 / 10000) goes the OTHER direction — probes
# whether the default sits at the accuracy plateau or if further-
# generous radii still buy meaningful accuracy for car (where 7500 m
# `r_medium` cuts into the ~8-15 min car reach).
#
# `_ALT_RADII_5` (2500 / 15000) is a "very generous" ground-truth
# reference — analogue of the buildings scenario on the spatial-
# resolution axis. Used to check that the default (1500 / 7500) sits
# at the accuracy plateau: comparing default vs `_ALT_RADII_5` should
# show negligible improvement, confirming the default isn't leaving
# accuracy on the table.
_ALT_RADII_1 = OdmRadii(r_cells=750.0, r_medium=3_000.0,  r_zones=100_000.0)
_ALT_RADII_2 = OdmRadii(r_cells=1_000.0, r_medium=7_500.0,  r_zones=100_000.0)
_ALT_RADII_3 = OdmRadii(r_cells=1_500.0, r_medium=5_000.0,  r_zones=100_000.0)
_ALT_RADII_4 = OdmRadii(r_cells=2_500.0, r_medium=15_000.0, r_zones=100_000.0)


def _tv_mode_configs_with(radii: OdmRadii) -> dict:
    """`_TV_MODE_CONFIGS` with `radii` overridden on every mode. Keeps
    the unified-radii property (needed by cross-modal aggregation)."""
    from dataclasses import replace as _replace
    return {m: _replace(cfg, radii=radii) for m, cfg in _TV_MODE_CONFIGS.items()}


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
        **_TV_COMMON,
    ),
    'res-h9': Scenario(
        name='res-h9',
        cell_source='h3',
        cell_h3_resolution=9,
        **_TV_COMMON,
    ),
    # ---- Radii ladder (H11 baseline, OD-tier radii varied) ------------
    # Compare against `res-h11` to isolate the radii effect.
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
# Filter plumbing (still to implement — 5 small edits, one per script):
#   - main/04_edge_weights.py, main/07a_road_overhead_coefs.py,
#     main/09a_utility_estimation.py: filter legs to those where BOTH
#     origin AND destination fall inside the scenario's AOI polygon
#     (spatial join on `orig_x`/`orig_y` and `dest_x`/`dest_y` against a
#     cached AOI union geometry). Requiring both endpoints excludes
#     cross-region legs, which would leak out-of-region behaviour into
#     training.
#   - main/03a_flow_coefs.py: filter traffic counters to those inside
#     the AOI polygon before calibration.
#   - validation/flows_vs_counters.py: apply the same counter filter
#     so validation is scoped to the scenario's AOI — in-sample for
#     cv-*-train, out-of-sample for cv-*-test.
# The filter is region-agnostic — for `switzerland-h10` it becomes a
# spatial join against the Switzerland polygon, which naturally passes
# through all in-country legs / counters. Every other script inherits
# the AOI through the scenario's `area_name`.

# Full ImportFrom map from a given cv-*-train scenario. Mirrors
# `_IMPORTED_FROM_SWISS` in scenarios.py but points at the CV twin.
def _imported_from(src_scenario: str) -> dict:
    return {
        'edge_weights_walk':            ImportFrom(src_scenario),
        'edge_weights_bike':            ImportFrom(src_scenario),
        'edge_weights_car':             ImportFrom(src_scenario),
        'overheads_road':               ImportFrom(src_scenario),
        'overheads_transit':            ImportFrom(src_scenario),
        'node_trip_weights_car':        ImportFrom(src_scenario),
        'flow_cost_bins_car':           ImportFrom(src_scenario),
        'utility_default':              ImportFrom(src_scenario),
        'utility_default_stats':        ImportFrom(src_scenario),
    }


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
# `storage=Storage.PUBLIC` — coefs come from `res-h10`'s calibrated bundle
# (ImportFrom), so no restricted-microdata dependency at run time.

_DATA_COMMON = dict(
    area_name='bern-metro',
    cell_source='h3',
    cell_h3_resolution=10,
    storage=Storage.PUBLIC,
    coefs=_IMPORTED_FROM_SWISS,
    mode_configs=_TV_MODE_CONFIGS,
    accessibility_grids=_TV_ACCESSIBILITY_GRIDS,
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
