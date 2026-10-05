"""
Calibrate per-mode traffic-flow coefficients for stage 03b.

For each calibrated mode:
  - `node_trip_weights_<mode>`: per-cell Poisson-GLM coefs for trip
    generation. Exposure offset is `log(combined_total)`; features come
    from `_NODE_TRIP_FEATURES`. 03b applies the fit to weight cells
    when sampling origins (replaces the old hardcoded
    `combined_total × max(1 − density·coef, 0.25)` heuristic).
  - `flow_cost_bins_<mode>`: percentile bin edges over the observed
    travel-cost distribution. 03b's `bin_adjusted_dest_weights` reads
    them as the target P(C) shape per origin.

Only `'car'` is calibrated for now. Other modes follow when their
ground-truth cost distributions land.

Feature set is intentionally narrow and easy to extend — edit
`_NODE_TRIP_FEATURES` near the top to add density / topology /
demographic columns. Adding a new mode means a new `_load_mtmc_*_costs`
helper + a new block in `main()`.

`flow_cost_bins_car` uses MTMC self-reported car trip times, jittered
to undo 5-min reporting quantization; gross times are used as-is
(the gross→net overhead subtraction is currently disabled — see
`_MTMC_CAR_OVERHEAD_S` below for rationale).

Inputs:
    generic/survey_legs.csv                            (PRIVATE, from survey/02d)
    properties/cells_{population,employment,snap}.csv
    properties/nodes_car_extended.csv                  (density features)

Outputs (per scenario's coef declaration in `src/scenarios.py`):
    coefs/calibrated/node_trip_weights_car.csv
    coefs/calibrated/flow_cost_bins_car.csv

Run:
    python -m main.03a_flow_coefs --scenario <name>
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from aperta_atlas import coefs
from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step

from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from scenarios import get_scenario


# ---------- Trip-generation regression -----------------------------------
_TARGET_MODE = 'car'                # column value in `mode_simplified`
_MODE_COLUMN = 'mode_simplified'

# Raw node-level inputs joined from `properties/nodes_<mode>_extended.csv`
# via the cell's snap node. Used as inputs to derived features (e.g.
# interactions) — NOT necessarily included as standalone regressors.
# Add a raw input here whenever a new derived feature needs it.
_RAW_NODE_FEATURES: tuple[str, ...] = (
    'density_r500_norm',
)

# Features entering the Poisson GLM. To add: raw cell feature → append
# here; raw node feature → also add to `_RAW_NODE_FEATURES`; derived →
# append + implement in `_derive_features`. 03b mirrors this
# derivation — keep the two files in sync.
#
# `density_r500_norm` enters ONLY through the interactions — adding it
# standalone would double-count trip-generating mass. The interactions
# encode "density modulates the per-unit trip rate".
_NODE_TRIP_FEATURES: tuple[str, ...] = (
    'log1p_population',
    'log1p_employment',
    'log1p_pop_x_density',
    'log1p_emp_x_density',
)


def _derive_features(cells: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived columns in `_NODE_TRIP_FEATURES` from raw cell
    columns + the pre-joined `_RAW_NODE_FEATURES`. Mirrored in 03b —
    change both together."""
    cells = cells.copy()
    pop = cells['population_total'].astype(float)
    emp = cells['employment_total'].astype(float)
    density = cells['density_r500_norm'].astype(float)
    cells['log1p_population'] = np.log1p(pop)
    cells['log1p_employment'] = np.log1p(emp)
    cells['log1p_pop_x_density'] = cells['log1p_population'] * density
    cells['log1p_emp_x_density'] = cells['log1p_employment'] * density
    return cells


# ---------- Cost-bin extraction ------------------------------------------ Cost
# distribution derived from MTMC self-reported car trip times. MTMC values are
# (a) heaped on 5-min bins, and (b) gross (include parking + walking overheads).
# We jitter within the reporting half- width to undo the quantization
# (Uniform(-half, +half) — standard maximum-entropy fix for rounded continuous
# data). Constants are hardcoded for now — MTMC 2015 + 2021 both use ~5-min
# self-report.
#
# Gross→net overhead subtraction is currently DISABLED (overhead=0): Small bias
# (~3-7 % relative for typical trips); acceptable for now. Possible solution:
# piecewise overhead min(120, 0.3 * t) so short trips get less subtraction.
_COST_BIN_MIN_S = 30.0
_COST_BIN_MAX_S = 5400.0
_N_COST_BINS = 25
_MTMC_ROUND_HALF_S = 150.0    # ± 2.5 min jitter half-width (5-min bins)
_MTMC_CAR_OVERHEAD_S = 0.0    # gross→net deduction, disabled (see note above)
_JITTER_SEED = 0              # fixed for reproducibility


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _load_car_trip_counts_per_cell(context, cell_index: pd.Index) -> pd.Series:
    """Per-cell count of survey legs whose origin snapped to that cell.
    Cells with zero observed trips reindex to 0 — valid Poisson observations.
    Legs are AOI-filtered (both endpoints inside the scenario AOI) so
    trip counts reflect within-region behaviour only."""
    legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)
    legs = filter_legs_by_aoi(
        legs, aoi_polygon, crs=scenario.crs_main, label='car-trip legs')
    in_mode = legs[_MODE_COLUMN] == _TARGET_MODE
    legs_mode = legs.loc[in_mode]
    logging.info(f"  → {len(legs_mode):,} of {len(legs):,} legs are mode={_TARGET_MODE!r}")
    counts = legs_mode.groupby('orig_cell_id').size()
    return counts.reindex(cell_index).fillna(0).astype(int)


def _fit_node_trip_weights(context) -> pd.DataFrame:
    """Fit Poisson GLM: trips_per_cell ~ features (no offset; exposure
    enters as `log1p_population` + `log1p_employment` features so the
    elasticity is learned rather than forced to 1).

    Returns a param × profile DataFrame; the profile column is named after
    `_TARGET_MODE` so future modes can append columns side-by-side."""
    # `add_shapes=True` brings `is_in_ch` (and `is_aoi`) onto cells via
    # cells.gpkg — needed for the calibration-coverage filter below.
    cells = context.get_properties(
        'cells', ['population', 'employment', 'snap'], add_shapes=True)
    node_props = context.get_properties('nodes', f'{_TARGET_MODE}_extended')
    snap_col = f'node_id_{_TARGET_MODE}'
    cells = cells.join(node_props[list(_RAW_NODE_FEATURES)], on=snap_col, how='left')
    cells = _derive_features(cells)

    y = _load_car_trip_counts_per_cell(context, cells.index)

    # Calibration filter: `is_aoi` (training region — buffer + out-of-
    # region cells would enter as false-zero observations and bias βs);
    # snapped to this mode; non-zero exposure; features non-NaN. NOT
    # `is_active` — that's an output-side filter; here we want max
    # valid training data. Using `is_aoi` rather than `is_in_ch` scopes
    # the fit to the scenario's training region — for switzerland-h10
    # the two coincide; for CV scenarios `is_aoi` is the language region.
    mask = (
        cells['is_aoi'].astype(bool)
        & cells[snap_col].notna()
        & (cells['combined_total'] > 0)
        & cells[list(_NODE_TRIP_FEATURES)].notna().all(axis=1)
    )
    n_dropped = (~mask).sum()
    logging.info(
        f"  → fitting on {mask.sum():,} cells "
        f"({n_dropped:,} dropped: outside-AOI / unsnapped / zero-exposure / NA features)")

    X = sm.add_constant(cells.loc[mask, list(_NODE_TRIP_FEATURES)].astype(float))
    model = sm.GLM(
        y.loc[mask].astype(float),
        X,
        family=sm.families.Poisson(),
    ).fit()

    # Log fitted coefs + exp(β) ≈ multiplier per +1 unit of the feature.
    logging.info(f"  → Poisson GLM fit: deviance={model.deviance:.0f}, "
                 f"pseudo-R²={1 - model.deviance / model.null_deviance:.3f}, "
                 f"n={int(mask.sum()):,}")
    for name, beta in model.params.items():
        logging.info(f"      {name:.<30s} β={beta:+.4f}   exp(β)={np.exp(beta):.3f}")

    return pd.DataFrame({_TARGET_MODE: model.params})


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray,
) -> np.ndarray:
    """Weighted quantiles at levels `q` ∈ [0, 1]. Linearly interpolates
    on the empirical weighted CDF. Standard weighted-percentile
    definition (mid-rank; equivalent to `np.percentile` when all
    weights are equal)."""
    order = np.argsort(values)
    vals = values[order]
    ws = weights[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    if total <= 0:
        raise ValueError("weights sum to zero — cannot compute quantiles.")
    # Mid-rank position on the CDF: (cum − w/2) / total.
    cdf = (cum - 0.5 * ws) / total
    return np.interp(q, cdf, vals)


def _load_mtmc_car_costs(context) -> tuple[np.ndarray, np.ndarray]:
    """Load MTMC car-trip times from the scenario's `survey_legs.csv`,
    quality-gated + AOI-filtered + gross→net + jittered. Returns
    `(times_s, weights)` as parallel float arrays.

    Pipeline:
      1. Load survey_legs.csv (already gated on `is_valid_domestic_trip`,
         `is_land_based`, `is_within_elevation_band` in 02d).
      2. Filter to `mode_simplified == 'car'`.
      3. Filter on `is_within_speed_envelope` + `is_within_detour_envelope`
         (calibration-side quality flags 02d preserves but doesn't apply).
      4. AOI-filter (both endpoints inside the scenario's AOI polygon).
      5. Gross→net: subtract `_MTMC_CAR_OVERHEAD_S`; floor at 30 s.
      6. Jitter: add Uniform(-`_MTMC_ROUND_HALF_S`, +`_MTMC_ROUND_HALF_S`)
         to undo 5-min self-report rounding.
      7. Trim to `[_COST_BIN_MIN_S, _COST_BIN_MAX_S]` as a safety net
         against residual outliers.

    Weights come from `weight_person` (BFS stratification correction)
    so weighted quantiles downstream reflect the target population.
    """
    legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)

    n_before = len(legs)
    legs = legs[legs[_MODE_COLUMN] == _TARGET_MODE]
    logging.info(f"  → mode filter: {len(legs):,}/{n_before:,} car legs")

    for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
        if flag in legs.columns:
            before = len(legs)
            legs = legs[legs[flag] == 1]
            logging.info(f"  → {flag}: {len(legs):,}/{before:,} kept")

    legs = filter_legs_by_aoi(
        legs, aoi_polygon, crs=scenario.crs_main, label='MTMC car legs')

    times_gross = legs['time_measured'].to_numpy(dtype=float)
    weights = legs['weight_person'].to_numpy(dtype=float)
    finite = np.isfinite(times_gross) & np.isfinite(weights) & (weights > 0)
    times_gross = times_gross[finite]
    weights = weights[finite]

    times_net = np.maximum(times_gross - _MTMC_CAR_OVERHEAD_S, 30.0)
    rng = np.random.default_rng(_JITTER_SEED)
    jitter = rng.uniform(-_MTMC_ROUND_HALF_S, _MTMC_ROUND_HALF_S,
                         size=times_net.shape)
    times = np.maximum(times_net + jitter, 30.0)

    in_range = (times >= _COST_BIN_MIN_S) & (times <= _COST_BIN_MAX_S)
    times = times[in_range]
    weights = weights[in_range]

    logging.info(
        f"  → MTMC car cost survey: {len(times):,} trips "
        f"(after gross→net −{_MTMC_CAR_OVERHEAD_S:.0f} s, "
        f"±{_MTMC_ROUND_HALF_S:.0f} s jitter, "
        f"[{_COST_BIN_MIN_S:.0f}, {_COST_BIN_MAX_S:.0f}] s trim); "
        f"weighted median = "
        f"{_weighted_quantile(times, weights, np.array([0.5]))[0]:.0f} s, "
        f"weighted P95 = "
        f"{_weighted_quantile(times, weights, np.array([0.95]))[0]:.0f} s")
    return times, weights


def _fit_flow_cost_bins(context) -> pd.DataFrame:
    """Weighted-percentile bin edges (ascending) over the MTMC car cost
    survey. Returns a param × profile DataFrame indexed `edge_00`,
    `edge_01`, ... — readable on inspection and survives the standard
    `param`-keyed CSV."""
    times, weights = _load_mtmc_car_costs(context)
    q = np.linspace(0.0, 1.0, _N_COST_BINS + 1)
    bin_edges = _weighted_quantile(times, weights, q)
    logging.info(
        f"  → {len(bin_edges)} bin edges (s): "
        f"{[f'{e:.0f}' for e in bin_edges]}")
    index = pd.Index([f'edge_{i:02d}' for i in range(len(bin_edges))],
                     name='param')
    return pd.DataFrame({_TARGET_MODE: bin_edges}, index=index)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    logging.info(f"=== Calibrating flow coefs for scenario {scenario.name!r} ===")

    with step(f'calibrate node_trip_weights_{_TARGET_MODE} '
              f'(Poisson GLM on MZMV car legs)'):
        coefs.resolve(
            context,
            name=f'node_trip_weights_{_TARGET_MODE}',
            calibrate_fn=lambda: _fit_node_trip_weights(context),
        )

    with step(f'calibrate flow_cost_bins_{_TARGET_MODE} '
              f'(weighted percentile edges over jittered MTMC car legs)'):
        coefs.resolve(
            context,
            name=f'flow_cost_bins_{_TARGET_MODE}',
            calibrate_fn=lambda: _fit_flow_cost_bins(context),
        )

    context.close()


if __name__ == '__main__':
    main()
