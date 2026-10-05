"""
Compare atlas-predicted per-leg travel TIMES and DISTANCES against
measured values from a survey leg set.

One variant per leg set (`main.common.SURVEY_LEG_SETS`, prepared by
survey/02d → 05 → 08a):
  - `mtmc` (default): MZMV 2015 + 2021 self-reported legs. In-sample for
    walk + bike edge weights, road + transit overheads and utilities.
    Times are 5-min-rounded.
  - `mobis_precovid`: 100k-leg sample of MOBIS GPS-tracked legs, normal
    traffic regime. In-sample for car edge weights (car's training
    cohort); out-of-sample for walk, bike and transit. GPS times aren't
    rounded. No confirmed regular-bike legs in this cohort: bike legs are
    `anybike` / `anyebike` (sub-type unknown).
  - `mobis_covid`: same for the COVID-era cohort (2020-wk13 to 2022-03,
    anomalous traffic regime) — the only cohort with regular-bike legs.
    Keep it apart from precovid when interpreting results.

`anybike` legs are compared against the rbike prediction and `anyebike`
against ebike25. `anybike` mixes regular and e-bikes, so a positive bias
vs rbike is expected; an extra `anybike@ebike25` row compares the same
legs against the ebike25 prediction — measured times should fall between
the two.

Car legs additionally get `car@peak` / `car@base` / `car@night` rows, split by
`peak_str` — the same label that picks each leg's car profile in 05, 04 and 08a.

For each leg, `05_leg_times` + `08a_add_leg_overheads` produce a routed
time + per-endpoint overheads for the leg's actual chosen mode. Sum →
predicted GROSS door-to-door time (`time`); the routed time alone is
`net_time`. Compare against the leg's `time_measured`. Distance:
predicted `length_<profile>` from 05 vs `dist_measured` — no gross/net
distinction for distance (overheads add time, not distance). Transit has
no length column (z2z lookup only) so distance rows are skipped for
transit.

Residuals reflect combined model uncertainty AND inherent measurement
noise (self-reported precision, GPS scatter, trip-purpose noise). The
model predicts EXPECTED time; individual leg residuals are noise around
an unbiased signal, not signal error.

The `is_within_speed_envelope == 1` filter is MANDATORY — trips outside
the per-mode speed envelope are physically-implausible and are excluded
by 04's edge-weight calibration; comparing on them would validate model
behaviour against data the training pipeline itself rejects. Requires
`is_within_speed_envelope` in `survey_legs.csv` (added to 02d's
`_KEEP_COLS`).

**TEMPORARY**: legs are additionally restricted to `n_legs_in_trip == 1`
so measured `time_measured` is door-to-door (matches how model
predictions are constructed, esp. NPVM transit z2z). Multi-leg trips
report per-leg times that exclude mode-change overheads (walk-to-stop,
etc.) but the model predicts door-to-door. Downstream consequence:
transit sample collapses to ~0.5 % of the raw set (see
`validation/leg_composition.py`); car ~80 %, walk ~45 %, bike ~85 %.
Proper fix (deferred): re-add `trip_id` to 02d's `_KEEP_COLS` and
compare at trip level.

Stats reported per (mode × quantity × band). Every (mode × quantity)
combo gets an `ALL` row; non-walking modes (rbike, ebike25, ebike45,
anybike, anyebike, car, transit) additionally get distance-line bands (<5 km,
5-25 km, >25 km; same for every leg set). Walk is ALL-only because ~99 % of walk legs are <2 km —
no meaningful long-distance stratification.

Transit gets an extra `z2z` quantity alongside `time`: raw NPVM zone-
to-zone lookup with NO overhead adjustment. Comparing `bias(time)`
vs `bias(z2z)` decomposes the transit residual into an overhead
component (hand-written `overheads_transit`) and a z2z component
(NPVM lookup accuracy).

NOTE ON TRANSIT SEMANTICS: NPVM z2z times are already DOOR-TO-DOOR
(they include walk-to-stop, wait, in-vehicle, and transfer times).
This is UNLIKE walk / bike / car, where the routed time is net (edge
time only) and overheads add first/last-mile time. For transit,
`overheads_transit` acts as an ADJUSTMENT (typically small, often
negative) to correct residual bias in the NPVM baseline — NOT as an
additive first/last-mile overhead. Therefore, for transit:
  - `z2z` is comparable to measured gross times directly.
  - `time` = z2z + hand-tuned adjustment (positive OR negative).
  - Any systematic bias in `z2z` reflects NPVM lookup accuracy, not
    a missing overhead component.
  - n             — legs contributing
  - bias          — `mean(predicted − measured)`; ≈ 0 if unbiased.
                    Positive = model OVER-predicts; negative = UNDER-predicts.
                    In seconds for `time`, metres for `dist`.
  - bias%         — bias as a percentage of the mean measured value
  - MAE / RMSE    — spread (same units as bias)
  - slope         — OLS slope predicted~measured (≈ 1 when values scale)
  - R²            — 1:1 coefficient of determination, `1 − SS_res / SS_tot`
                    (same as 04's calibration log). Penalises bias and wrong
                    scale; < 0 = worse than predicting the mean measured value
  - MAE%          — 100 · MAE / mean(|measured|); WMAPE-style relative
                    error, robust to per-leg zeros

Inputs (PRIVATE, under `<scenario>/`; `_<leg_set>` suffix for MOBIS):
    generic/survey_legs[_<leg_set>].csv            # from survey/02d
    generic/survey_leg_times[_<leg_set>].csv       # from survey/05
    generic/survey_leg_overheads[_<leg_set>].csv   # from survey/08a

Outputs:
    RESULTS/times_vs_survey[_<leg_set>].csv        # summary table (mode + strata rows)
    Logs                                           # summary tables

Run (default variant `mtmc`):
    python -m validation.times_vs_survey --scenario <name>
MOBIS (after survey/02d, 05, 08a with the same `--variant`):
    python -m validation.times_vs_survey --scenario <name> --variant mobis_precovid
"""

import logging

import numpy as np
import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from main.common import (
    SURVEY_LEG_SETS,
    car_overhead_by_peak,
    car_routed_bias_by_peak,
    survey_file,
)
from validation.common import log_exclusion, plausibility_mask_bulk


# Mode → (routed_profile, overhead_profile) suffixes in survey_leg_times /
# survey_leg_overheads columns. Transit is asymmetric: the routed column
# is z2z-suffixed (`t_routed_transit_z2z`, from 05's NPVM z2z lookup)
# while the overhead columns are transit-generic (`t_overhead_{orig,dest}_transit`
# from 08a, matching what 09a uses). Car handled separately below: uses
# peak-aggregated `t_routed_car` + `car_overhead_by_peak(...)`.
#
# `anybike` / `anyebike` are MOBIS bike legs of unknown sub-type (before its
# 2020-07 bike/e-bike split). `anybike` mixes regular and e-bikes (MZMV 2021:
# ebike25 ≈ 15 % of rbike + ebike25), so a positive bias vs rbike is expected.
_MODE_TO_PROFILE: dict[str, tuple[str, str]] = {
    'walk':     ('rwalk',       'rwalk'),
    'rbike':    ('rbike',       'rbike'),
    'ebike25':  ('ebike25',     'ebike25'),
    'ebike45':  ('ebike45',     'ebike45'),
    'anybike':  ('rbike',       'rbike'),
    'anyebike': ('ebike25',     'ebike25'),
    'transit':  ('transit_z2z', 'transit'),
}

# Extra (mode, profile) comparisons, reported as mode `<mode>@<profile>`.
# `anybike` vs the ebike25 prediction brackets the mixture from the fast
# side: measured times should fall between the rbike and ebike25 rows.
_BRACKET_ROWS: list[tuple[str, str]] = [('anybike', 'ebike25')]

# Car legs are also reported per time of day (`peak_str`, the label that picks
# each leg's car profile), as modes `car@peak` / `car@base` / `car@night`.
_CAR_TIMES_OF_DAY: tuple[str, ...] = ('peak', 'base', 'night')

# Distance-line bands (meters) for the mode × band strata. Coarse on
# purpose: MTMC self-report is 5-min-rounded, so finer bands mostly add
# rounding noise; shared across leg sets so MTMC and MOBIS rows compare.
_DIST_BANDS: list[tuple[str, float, float]] = [
    ('   <5km',       0.0,    5_000.0),
    ('  5-25km',  5_000.0,   25_000.0),
    ('   >25km', 25_000.0,  np.inf),
]


# Modes that get per-band rows in addition to `ALL`. Walk is ALL-only
# because ~99 % of walk legs are <2 km — long-distance walk strata
# would be empty or dominated by noise.
_STRATIFIED_MODES: set[str] = {
    'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike', 'car', 'transit'}

def _predicted_gross(legs: pd.DataFrame, routed: pd.DataFrame, overhead: pd.DataFrame,
                     profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg predicted GROSS time (seconds), assembled from routed +
    overhead columns per the leg's chosen mode:
        gross = t_routed + orig_ov + dest_ov − t_routed_bias
    (t_routed_bias absorbs the fitted α scaling of routed time — see
    `main.common.route_time_alpha`. Zero for constrained-α fits.)
    NaN when the leg's mode isn't modelled, required columns are missing,
    or (for transit) the z2z lookup returned 0 (intrazone — no diagonal
    entry in NPVM).
    """
    pred = pd.Series(np.nan, index=legs.index, name='predicted_gross_s')
    for mode, (routed_profile, overhead_profile) in profiles.items():
        mask = legs['mode_simplified'] == mode
        if not mask.any():
            continue
        routed_col = f't_routed_{routed_profile}'
        ov_orig_col = f't_overhead_orig_{overhead_profile}'
        ov_dest_col = f't_overhead_dest_{overhead_profile}'
        bias_col = f't_routed_bias_{overhead_profile}'
        if routed_col not in routed.columns:
            logging.warning(
                f"  ⚠ mode={mode!r}: {routed_col!r} missing; "
                f"predictions for {int(mask.sum()):,} legs will be NaN.")
            continue
        if ov_orig_col not in overhead.columns or ov_dest_col not in overhead.columns:
            logging.warning(
                f"  ⚠ mode={mode!r}: overhead column(s) missing "
                f"({ov_orig_col!r}, {ov_dest_col!r}); "
                f"predictions for {int(mask.sum()):,} legs will be NaN.")
            continue
        r = routed.loc[mask, routed_col]
        o = overhead.loc[mask, ov_orig_col].fillna(0.0)
        d = overhead.loc[mask, ov_dest_col].fillna(0.0)
        bias = (overhead.loc[mask, bias_col].fillna(0.0)
                if bias_col in overhead.columns else 0.0)
        total = r + o + d - bias
        # Transit: NPVM z2z lookup returns 0 for intrazone pairs (no
        # diagonal). Those legs would collapse to overheads-only — drop.
        if mode == 'transit':
            n_intrazone = int((r == 0).sum())
            total = total.where(r > 0)
            if n_intrazone:
                logging.info(
                    f"  → transit: dropped {n_intrazone:,} intrazone legs "
                    f"(t_routed_transit_z2z == 0)")
        pred.loc[mask] = total

    # Car: peak-aggregated routed column + peak-appropriate overheads + bias.
    mask = legs['mode_simplified'] == 'car'
    if mask.any():
        if 't_routed_car' not in routed.columns:
            logging.warning("  ⚠ 't_routed_car' missing — car predictions will be NaN.")
        else:
            sub_overhead = overhead.loc[mask].join(legs.loc[mask, ['peak_str']])
            r = routed.loc[mask, 't_routed_car']
            o = car_overhead_by_peak(sub_overhead, 'orig').fillna(0.0)
            d = car_overhead_by_peak(sub_overhead, 'dest').fillna(0.0)
            bias = car_routed_bias_by_peak(sub_overhead).fillna(0.0)
            pred.loc[mask] = r + o + d - bias
    return pred


def _predicted_net(legs: pd.DataFrame, routed: pd.DataFrame,
                   profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg predicted NET time (seconds): the raw calibrated routed time
    `t_routed_<profile>` (car: peak-aggregated `t_routed_car`), with no
    overheads and no α correction. Comparing `net_time` with `time` shows
    how much the overheads contribute for a given ground truth — relevant
    for GPS legs, which may not include walking to and from the vehicle.
    Transit is covered by the `z2z` quantity instead → NaN here."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_net_s')
    for mode, (routed_profile, _overhead_profile) in profiles.items():
        if mode == 'transit':
            continue
        mask = legs['mode_simplified'] == mode
        routed_col = f't_routed_{routed_profile}'
        if mask.any() and routed_col in routed.columns:
            pred.loc[mask] = routed.loc[mask, routed_col]
    mask = legs['mode_simplified'] == 'car'
    if mask.any() and 't_routed_car' in routed.columns:
        pred.loc[mask] = routed.loc[mask, 't_routed_car']
    return pred


def _predicted_baseline(legs: pd.DataFrame, routed: pd.DataFrame,
                        profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg NAIVE-BASELINE time: `t_baseline_<profile>` from 05, routed on
    `length / baseline_speed` edge weights (OSM speed limit for car), with no
    calibration and NO overheads — what someone routing on raw OSM speeds would
    get. The gap to the `time` row quantifies the payoff of the full calibrated
    model. Car: `t_baseline_car`, picked by `peak_str` in 05. Transit has no
    baseline (NPVM z2z is the only source) → NaN."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_baseline_s')
    for mode, (routed_profile, _overhead_profile) in profiles.items():
        if mode == 'transit':
            continue
        mask = legs['mode_simplified'] == mode
        baseline_col = f't_baseline_{routed_profile}'
        if mask.any() and baseline_col in routed.columns:
            pred.loc[mask] = routed.loc[mask, baseline_col]
    mask = legs['mode_simplified'] == 'car'
    if mask.any() and 't_baseline_car' in routed.columns:
        pred.loc[mask] = routed.loc[mask, 't_baseline_car']
    return pred


def _predicted_transit_z2z(legs: pd.DataFrame, routed: pd.DataFrame) -> pd.Series:
    """Per-leg predicted TRANSIT z2z time (seconds), from the raw NPVM
    zone-to-zone lookup with NO overhead adjustment. Only defined for
    transit legs (NaN elsewhere). Intrazone legs (z2z lookup == 0) are
    also NaN.

    Compare against `time_measured` alongside the `time` (gross) row to
    decompose transit bias: `bias(time) − bias(z2z)` = the net effect of
    the hand-written `overheads_transit` adjustment. If `z2z` bias is
    strongly negative (model UNDER-predicts pure ride time) while `time`
    bias is positive, the hand-tuned overheads are too generous.
    """
    pred = pd.Series(np.nan, index=legs.index, name='predicted_transit_z2z_s')
    mask = legs['mode_simplified'] == 'transit'
    if not mask.any():
        return pred
    routed_col = 't_routed_transit_z2z'
    if routed_col not in routed.columns:
        logging.warning(f"  ⚠ {routed_col!r} missing; z2z predictions will be NaN.")
        return pred
    r = routed.loc[mask, routed_col]
    pred.loc[mask] = r.where(r > 0)  # drop intrazone (z2z == 0)
    return pred


def _predicted_length(legs: pd.DataFrame, routed: pd.DataFrame,
                      profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg predicted ROUTED distance (metres). Read from `length_<profile>`
    in survey_leg_times. Car: `length_car_{peak,base,night}` selected by `peak_str`
    (parallels the time-side aggregation in `05_leg_times`). Transit has
    no length column (z2z lookup) → NaN. No 'gross' distinction: overheads
    add time, not distance."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_length_m')
    for mode, (routed_profile, _overhead_profile) in profiles.items():
        if mode == 'transit':
            continue  # no length column for transit z2z lookup
        mask = legs['mode_simplified'] == mode
        if not mask.any():
            continue
        length_col = f'length_{routed_profile}'
        if length_col not in routed.columns:
            logging.warning(f"  ⚠ mode={mode!r}: {length_col!r} missing; "
                            f"distance predictions for {int(mask.sum()):,} legs will be NaN.")
            continue
        pred.loc[mask] = routed.loc[mask, length_col]

    # Car: pick length by peak_str, per-profile columns.
    mask = legs['mode_simplified'] == 'car'
    if mask.any():
        peak_str = legs.loc[mask, 'peak_str']
        col_map = {
            'peak':  'length_car_peak',
            'base':  'length_car_base',
            'night': 'length_car_night',
        }
        missing = [c for c in col_map.values() if c not in routed.columns]
        if missing:
            logging.warning(f"  ⚠ car length column(s) missing {missing}; "
                            f"car distance predictions will be partial.")
        vals = np.select(
            [peak_str.values == k for k in col_map],
            [routed.loc[mask, c].values if c in routed.columns else
             np.full(int(mask.sum()), np.nan) for c in col_map.values()],
            default=np.nan,
        )
        pred.loc[mask] = vals
    return pred


def _stats(measured: np.ndarray, predicted: np.ndarray) -> dict:
    """Bias / MAE / RMSE / slope / R² / MAE% + band-level means on a
    finite-in-both mask."""
    mask = np.isfinite(measured) & np.isfinite(predicted)
    n = int(mask.sum())
    if n < 2:
        return {'n': n, 'mean_pred': np.nan, 'mean_true': np.nan,
                'bias': np.nan, 'bias_pct': np.nan,
                'mae': np.nan, 'rmse': np.nan,
                'slope': np.nan, 'r2': np.nan, 'mae_pct': np.nan}
    m, p = measured[mask], predicted[mask]
    resid = p - m  # positive => model OVER-predicts
    m_mean = float(m.mean())
    p_mean = float(p.mean())
    bias = float(resid.mean())
    m_c = m - m_mean
    p_c = p - p_mean
    ss_m = float((m_c ** 2).sum())
    slope = float((m_c * p_c).sum() / ss_m) if ss_m > 0 else np.nan
    # 1:1 R² (`1 − SS_res / SS_tot`), as in `aperta.calibration`: penalises bias and
    # scale, can go negative (worse than predicting the mean measured value).
    r2 = 1.0 - float((resid ** 2).sum()) / ss_m if ss_m > 0 else np.nan
    mae = float(np.abs(resid).mean())
    m_abs_mean = float(np.abs(m).mean())
    mae_pct = 100.0 * mae / m_abs_mean if m_abs_mean > 0 else np.nan
    bias_pct = 100.0 * bias / m_mean if m_mean != 0 else np.nan
    return {
        'n':         n,
        'mean_pred': p_mean,
        'mean_true': m_mean,
        'bias':      bias,
        'bias_pct':  bias_pct,
        'mae':       mae,
        'rmse':      float(np.sqrt((resid ** 2).mean())),
        'slope':     slope,
        'r2':        r2,
        'mae_pct':   mae_pct,
    }


def _comparison_rows(legs: pd.DataFrame, routed: pd.DataFrame, overhead: pd.DataFrame,
                     profiles: dict[str, tuple[str, str]],
                     mode_suffix: str = '') -> list[dict]:
    """Predict per-leg times + lengths with `profiles` (survey mode →
    (routed, overhead) profile), apply the plausibility filter and return
    one stats row per (mode × quantity × band). `mode_suffix` is appended
    to the reported mode (e.g. `@ebike25` for a bracketing comparison)."""
    def label_of(mode: str) -> str:
        return f'{mode}{mode_suffix}'

    rows: list[dict] = []
    with step('assemble predicted gross times + routed lengths per leg'):
        pred_time = _predicted_gross(legs, routed, overhead, profiles)
        pred_net = _predicted_net(legs, routed, profiles)
        pred_baseline = _predicted_baseline(legs, routed, profiles)
        pred_transit_z2z = _predicted_transit_z2z(legs, routed)
        pred_dist = _predicted_length(legs, routed, profiles)
        meas_time = legs['time_measured']
        meas_dist = legs['dist_measured']
        n_time = int((pred_time.notna() & meas_time.notna() & (meas_time > 0)).sum())
        n_baseline = int((pred_baseline.notna() & meas_time.notna() & (meas_time > 0)).sum())
        n_z2z = int((pred_transit_z2z.notna() & meas_time.notna() & (meas_time > 0)).sum())
        n_dist = int((pred_dist.notna() & meas_dist.notna() & (meas_dist > 0)).sum())
        logging.info(f"  → {n_time:,} legs with finite predicted+measured time; "
                     f"{n_baseline:,} with baseline; "
                     f"{n_z2z:,} with transit z2z (no overheads); "
                     f"{n_dist:,} with finite predicted+measured distance")

    with step('plausibility filter (per-mode speed + detour ratio, both sides)'):
        # Applied symmetrically to ground truth AND model output — any leg
        # where either side is physically implausible is dropped from BOTH
        # the time and dist comparisons (unified subset, cleaner reading).
        keep_gt = plausibility_mask_bulk(
            legs['mode_simplified'], meas_dist, meas_time, legs['dist_line'])
        keep_model = plausibility_mask_bulk(
            legs['mode_simplified'], pred_dist, pred_time, legs['dist_line'])
        plausible = log_exclusion('plausibility', keep_gt, keep_model)
        # Zero out predictions/measurements that fail — downstream `_stats`
        # skips NaN via its finite-mask, so this excludes those legs cleanly.
        pred_time = pred_time.where(plausible)
        pred_net = pred_net.where(plausible)
        pred_baseline = pred_baseline.where(plausible)
        pred_transit_z2z = pred_transit_z2z.where(plausible)
        pred_dist = pred_dist.where(plausible)
        meas_time = meas_time.where(plausible)
        meas_dist = meas_dist.where(plausible)

    with step('compute per-mode stats (ALL + coarse distance bands)'):
        # Naive-baseline times (05's t_baseline_<profile>, no overheads) vs
        # measured: gap between 'time' and 'baseline_time' rows quantifies
        # the payoff of the calibrated model over speed-limit-only routing.
        dist_line_m = legs['dist_line'].astype(float)
        for quantity, meas, pred in (
            ('time',          meas_time, pred_time),
            ('net_time',      meas_time, pred_net),
            ('z2z',           meas_time, pred_transit_z2z),
            ('baseline_time', meas_time, pred_baseline),
            ('dist',          meas_dist, pred_dist),
        ):
            both = pred.notna() & meas.notna() & (meas > 0)
            for mode in sorted(legs.loc[both, 'mode_simplified'].unique()):
                mmask_all = both & (legs['mode_simplified'] == mode)
                rows.append({
                    'mode': label_of(mode), 'quantity': quantity, 'band': 'ALL',
                    **_stats(meas.loc[mmask_all].to_numpy(dtype=float),
                             pred.loc[mmask_all].to_numpy(dtype=float))})
                if mode not in _STRATIFIED_MODES:
                    continue
                for label, lo, hi in _DIST_BANDS:
                    bmask = (dist_line_m >= lo) & (dist_line_m < hi)
                    mmask = mmask_all & bmask
                    if not mmask.any():
                        continue
                    rows.append({
                        'mode': label_of(mode), 'quantity': quantity, 'band': label,
                        **_stats(meas.loc[mmask].to_numpy(dtype=float),
                                 pred.loc[mmask].to_numpy(dtype=float))})

    return rows


def main(variant):
    context = init_context(variant)
    leg_set = variant.leg_set

    with step(f'load survey_legs + survey_leg_times + survey_leg_overheads ({leg_set})'):
        legs = context.get_generic(
            survey_file('survey_legs.csv', leg_set), storage=Storage.PRIVATE)
        routed = context.get_generic(
            survey_file('survey_leg_times.csv', leg_set), storage=Storage.PRIVATE)
        overhead = context.get_generic(
            survey_file('survey_leg_overheads.csv', leg_set), storage=Storage.PRIVATE)
        logging.info(f"  → {len(legs):,} legs; "
                     f"routed {len(routed):,}, overhead {len(overhead):,}")
        # Defense-in-depth: after our recent leg_id-index refactor, all
        # three share leg_id as the index. Assert alignment.
        if not (legs.index.equals(routed.index) and legs.index.equals(overhead.index)):
            n_shared = int(legs.index.intersection(routed.index).intersection(overhead.index).size)
            logging.warning(
                f"  ⚠ indices don't fully agree — restricting to the "
                f"{n_shared:,} legs present in all three.")
            common = legs.index.intersection(routed.index).intersection(overhead.index)
            legs, routed, overhead = legs.loc[common], routed.loc[common], overhead.loc[common]

    with step('filter to within-speed-envelope legs (mandatory)'):
        # Trips outside the per-mode speed envelope are excluded from
        # 04's edge-weight calibration — validating against them would
        # measure model behaviour on data the training pipeline rejects
        # as physically-implausible. Same subset for both.
        if 'is_within_speed_envelope' not in legs.columns:
            raise KeyError(
                "Column 'is_within_speed_envelope' not in survey_legs.csv. "
                "Add it to `_KEEP_COLS` in survey/02d and re-run 02d.")
        n_before = len(legs)
        mask = legs['is_within_speed_envelope'] == 1
        legs = legs.loc[mask]
        routed = routed.loc[legs.index]
        overhead = overhead.loc[legs.index]
        logging.info(f"  → {len(legs):,} / {n_before:,} legs kept")

    with step('filter to single-leg trips (n_legs_in_trip == 1) — TEMPORARY'):
        # WORKAROUND: MTMC records TRIPS as one-or-more LEGS (mode change
        # ⇒ new leg). For multi-leg trips, `time_measured` on any leg is
        # for that leg's segment only, while model predictions (esp. NPVM
        # transit z2z) are door-to-door — an apples-to-oranges mismatch.
        # Restrict to `n_legs_in_trip == 1` so measured IS door-to-door.
        # Consequences (per `validation/leg_composition.py` on 2026-09-29):
        #   car ~80% kept, bike ~85%, walk ~45%, transit ~0.5% (!!).
        # Transit's tiny surviving sample is the correct trade-off for
        # apples-to-apples; the wider transit story needs trip-level
        # aggregation (requires re-adding `trip_id` to 02d's _KEEP_COLS
        # and grouping legs upstream) — deferred.
        if 'n_legs_in_trip' not in legs.columns:
            raise KeyError(
                "Column 'n_legs_in_trip' not in survey_legs.csv. "
                "Add it to `_KEEP_COLS` in survey/02d and re-run 02d.")
        n_before = len(legs)
        mask = legs['n_legs_in_trip'] == 1
        legs = legs.loc[mask]
        routed = routed.loc[legs.index]
        overhead = overhead.loc[legs.index]
        logging.info(f"  → {len(legs):,} / {n_before:,} legs kept "
                     f"(dropped {n_before - len(legs):,} legs in multi-leg trips)")

    rows = _comparison_rows(legs, routed, overhead, _MODE_TO_PROFILE)
    for mode, profile in _BRACKET_ROWS:
        sub = legs['mode_simplified'] == mode
        if not sub.any():
            continue
        with step(f'bracketing comparison: {mode} legs vs {profile} prediction'):
            rows += _comparison_rows(
                legs.loc[sub], routed.loc[sub], overhead.loc[sub],
                {mode: (profile, profile)}, mode_suffix=f'@{profile}')
    for tod in _CAR_TIMES_OF_DAY:
        sub = (legs['mode_simplified'] == 'car') & (legs['peak_str'] == tod)
        if not sub.any():
            continue
        with step(f'car legs by time of day: {tod}'):
            rows += _comparison_rows(
                legs.loc[sub], routed.loc[sub], overhead.loc[sub],
                _MODE_TO_PROFILE, mode_suffix=f'@{tod}')

    with step('summary'):
        _log_and_save(context, rows, leg_set)
    context.close()


def _log_and_save(context, rows: list[dict], leg_set: str) -> None:
    if not rows:
        logging.warning("no rows to summarize — check inputs.")
        return
    df = pd.DataFrame(rows)
    # Order: mode grouped; within each mode, time → net_time → z2z →
    # baseline_time → dist; within each quantity, ALL first, then the
    # distance bands. `z2z` is transit-only (raw NPVM lookup, no overheads);
    # bias(time) − bias(z2z) attributes the transit residual to overhead vs.
    # z2z prediction.
    quantity_order = {'time': 0, 'net_time': 1, 'z2z': 2, 'baseline_time': 3, 'dist': 4}
    band_order = {'ALL': 0, **{label: i + 1 for i, (label, _, _) in enumerate(_DIST_BANDS)}}
    df['_qorder'] = df['quantity'].map(quantity_order).fillna(99)
    df['_border'] = df['band'].map(band_order).fillna(99)
    df = df.sort_values(['mode', '_qorder', '_border']).drop(
        columns=['_qorder', '_border']).reset_index(drop=True)

    # Rename for readable table headers (unit-generic; unit inferred from
    # `quantity` column: 'time' → seconds, 'dist' → metres).
    df = df.rename(columns={
        'n': 'n_legs',
        'mean_pred': 'mean_pred', 'mean_true': 'mean_true',
        'bias': 'bias', 'bias_pct': 'bias%',
        'mae': 'MAE', 'rmse': 'RMSE',
        'slope': 'slope', 'r2': 'R²', 'mae_pct': 'MAE%',
    })
    display_cols = ['mode', 'quantity', 'band', 'n_legs',
                    'bias', 'bias%', 'MAE', 'RMSE',
                    'slope', 'R²', 'MAE%']

    # Save numeric-precise CSV to RESULTS (no console formatters applied).
    context.create_results(
        df[display_cols], survey_file('times_vs_survey.csv', leg_set),
        kws={'index': False, 'float_format': '%.4f'},
    )

    formatters = {
        'n_legs':     '{:>7,}'.format,
        'bias':       '{:+8.1f}'.format,
        'bias%':      '{:+.1f}%'.format,
        'MAE':        '{:7.1f}'.format,
        'RMSE':       '{:7.1f}'.format,
        'slope':      '{:+.3f}'.format,
        'R²':         '{:.4f}'.format,
        'MAE%':       '{:.1f}%'.format,
    }
    table = df[display_cols].to_string(index=False, formatters=formatters,
                                       justify='right')
    logging.info(
        f"Predicted vs measured per-leg values ({leg_set}), per (mode × quantity × band). "
        "Quantity 'time' = calibrated GROSS door-to-door in SECONDS; "
        "'net_time' = calibrated routed time alone (no overheads, no α "
        "correction) in SECONDS; "
        "'z2z' = raw NPVM zone-to-zone lookup, TRANSIT-ONLY, no overheads "
        "(bias(time)−bias(z2z) attributes the transit residual to "
        "overheads vs. z2z lookup accuracy); "
        "'baseline_time' = naive-baseline routing (speed limit only, no "
        "overheads) in SECONDS — gap between 'time' and 'baseline_time' "
        "rows quantifies calibration payoff; 'dist' = routed length in "
        "METRES (no gross/net — overheads add time, not distance).\n"
        "`bias > 0` = model OVER-predicts. Bands: `ALL` on every mode; "
        "non-walking modes additionally get distance-line strata "
        "(<5 km, 5-25 km, >25 km).\n"
        + table)


variants = Variants([('leg_set', str)])
for _leg_set in SURVEY_LEG_SETS:
    variants.add(name=_leg_set, leg_set=_leg_set)


if __name__ == '__main__':
    variants.run(main, default='mtmc')
