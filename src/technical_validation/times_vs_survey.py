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
    cohort); out-of-sample for walk and bike. GPS times aren't
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
predicted GROSS door-to-door time (`time`). `net_time` is the network time
as 04 calibrates it: the routed time plus 04's fitted endpoint term
(`snap_dist_orig` / `snap_dist_dest` coefs x the leg's snap distances), which
the released edge durations leave out. Compare against the leg's
`time_measured`. Distance:
predicted `length_<profile>` from 05 vs `dist_measured` — no gross/net
distinction for distance (overheads add time, not distance). Public transport
is compared at trip level (`technical_validation/times_vs_survey_transit.py`),
since NPVM zone-to-zone times are door to door.

Residuals reflect combined model uncertainty AND inherent measurement
noise (self-reported precision, GPS scatter, trip-purpose noise). The
model predicts EXPECTED time; individual leg residuals are noise around
an unbiased signal, not signal error.

Leg selection: each leg set is compared on the legs of the calibration step
it checks, selected as that step selects them (04's survey gates, AOI, and
the step's per-mode distance range and detour cap), so that the in-sample
comparisons are one-to-one with the fits:
  - `mtmc` → the road trip overheads (07a; door-to-door `time`), restricted
    to single-leg trips, whose reported time is door to door (07a also fits
    the legs of multi-leg trips, with an indicator).
  - MOBIS leg sets → the edge weights (04; `net_time`), including 04's
    snap-distance limit.
Out-of-sample rows (e-bikes, MOBIS walking, `anybike`) use the selection of
the network they are routed on (bike for e-bikes).

Rows per (mode × quantity × band), kept to what the data descriptor and the
WCTR paper report plus some context:
  - quantities: the calibrated time the leg set checks (`time`, door to door,
    for `mtmc`; `net_time` for the MOBIS leg sets, whose GPS legs record the
    movement only), `baseline_time` (uncalibrated: speed limits, no
    overheads) and `dist` (routed length; `ALL` only).
  - bands: `ALL` for every mode; the bicycle modes and car overall also get
    distance-line bands (<2 km, 2-5 km, 5-25 km, >25 km; same for every leg
    set). Walking (~99 % of legs < 2 km) and the car time-of-day rows are
    `ALL` only.

Columns:
  - n             — legs contributing
  - bias          — `mean(predicted − measured)`; ≈ 0 if unbiased.
                    Positive = model OVER-predicts; negative = UNDER-predicts.
                    In seconds for `time`, metres for `dist`.
  - bias%         — bias as a percentage of the mean measured value
  - med_resid     — `median(predicted − measured)`: where the typical leg lies
                    relative to the prediction (the fits target the mean, so
                    a right-skewed tail of long legs puts it above 0)
  - med_resid%    — med_resid as a percentage of the median measured value
  - R²_1to1       — R² relative to the 1:1 line, `1 − SS_res / SS_tot` (same
                    as 04's calibration log): do the values agree? Penalises
                    bias and wrong scale; < 0 = worse than the mean measured value
  - R²_fit        — R² relative to the best-fit line (= squared Pearson):
                    agreement up to offset and scale
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
    python -m technical_validation.times_vs_survey --scenario <name>
MOBIS (after survey/02d, 05, 08a with the same `--variant`):
    python -m technical_validation.times_vs_survey --scenario <name> --variant mobis_precovid
"""

import importlib
import inspect
import logging

import numpy as np
import pandas as pd

from aperta import calibration
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from main.common import (
    SURVEY_LEG_SETS,
    car_overhead_by_peak,
    car_routed_bias_by_peak,
    survey_file,
)
from scenarios import get_scenario
from technical_validation.common import agreement_stats

_m04 = importlib.import_module('main.04_edge_weights')
_m07a = importlib.import_module('main.07a_road_overhead_coefs')

# Calibration configs whose leg selection the comparisons reuse: 04 per network mode and scenario
# (`_m04._profile_config`), 07a per overhead mode (the three car configs share their distance
# range). 04 leaves the detour cap at `calibrate_edge_weights`' default.
_FIT_07A = {cfg.sub_modes[0]: cfg for cfg in _m07a._ROAD_CONFIGS}
_MAX_DETOUR_04 = inspect.signature(
    calibration.calibrate_edge_weights).parameters['max_dist_to_line_ratio'].default

# Survey mode → (network mode of 04, overhead mode of 07a) whose fit selection its legs follow.
_FIT_MODES: dict[str, tuple[str, str]] = {
    'walk':     ('walk', 'walk'),
    'rbike':    ('bike', 'rbike'),
    'ebike25':  ('bike', 'rbike'),
    'ebike45':  ('bike', 'rbike'),
    'anybike':  ('bike', 'rbike'),
    'anyebike': ('bike', 'rbike'),
    'car':      ('car', 'car'),
}

# Routing profile → network mode, for the endpoint term of 04's edge-weight coefs.
_PROFILE_NETWORK: dict[str, str] = {
    'rwalk': 'walk', 'rbike': 'bike', 'ebike25': 'bike', 'ebike45': 'bike',
    'car_peak': 'car', 'car_base': 'car', 'car_night': 'car',
}


# Mode → (routed_profile, overhead_profile) suffixes in survey_leg_times /
# survey_leg_overheads columns. Car handled separately below: uses
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
# <2 km and 2-5 km split the short legs, where per-trip overheads weigh most.
_DIST_BANDS: list[tuple[str, float, float]] = [
    ('   <2km',       0.0,    2_000.0),
    ('   2-5km',  2_000.0,    5_000.0),
    ('  5-25km',  5_000.0,   25_000.0),
    ('   >25km', 25_000.0,  np.inf),
]


# Modes that get per-band rows in addition to `ALL`. Walk is ALL-only
# because ~99 % of walk legs are <2 km — long-distance walk strata
# would be empty or dominated by noise.
_STRATIFIED_MODES: set[str] = {
    'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike', 'car'}

def _predicted_gross(legs: pd.DataFrame, routed: pd.DataFrame, overhead: pd.DataFrame,
                     profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg predicted GROSS time (seconds), assembled from routed +
    overhead columns per the leg's chosen mode:
        gross = t_routed + orig_ov + dest_ov − t_routed_bias
    (t_routed_bias absorbs the fitted α scaling of routed time — see
    `main.common.route_time_alpha`. Zero for constrained-α fits.)
    NaN when the leg's mode isn't modelled or required columns are missing.
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
        pred.loc[mask] = r + o + d - bias

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


def _endpoint_term(legs: pd.DataFrame, profile: pd.Series,
                   edge_coefs: dict[str, pd.DataFrame]) -> pd.Series:
    """Per-leg endpoint term of 04's edge-weight fit (seconds): the leg's profile's
    `snap_dist_orig` / `snap_dist_dest` coefs x its snap distances on that network."""
    term = pd.Series(np.nan, index=legs.index)
    for p, nmode in _PROFILE_NETWORK.items():
        mask = profile == p
        if not mask.any():
            continue
        c = edge_coefs[nmode][(p, 'coef')]
        term[mask] = (c.get('snap_dist_orig', 0.0) * legs.loc[mask, f'orig_snap_dist_{nmode}']
                      + c.get('snap_dist_dest', 0.0) * legs.loc[mask, f'dest_snap_dist_{nmode}'])
    return term


def _predicted_net(legs: pd.DataFrame, routed: pd.DataFrame,
                   profiles: dict[str, tuple[str, str]],
                   edge_coefs: dict[str, pd.DataFrame]) -> pd.Series:
    """Per-leg predicted NET time (seconds): the calibrated routed time
    `t_routed_<profile>` (car: peak-aggregated `t_routed_car`) plus the endpoint
    term fitted with the edge weights, i.e. the network time as 04 calibrates
    it; no overheads and no α correction. The comparison for GPS legs, which
    start and end with the movement."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_net_s')
    profile = pd.Series(None, index=legs.index, dtype=object)
    for mode, (routed_profile, _overhead_profile) in profiles.items():
        mask = legs['mode_simplified'] == mode
        routed_col = f't_routed_{routed_profile}'
        if mask.any() and routed_col in routed.columns:
            pred.loc[mask] = routed.loc[mask, routed_col]
            profile.loc[mask] = routed_profile
    mask = legs['mode_simplified'] == 'car'
    if mask.any() and 't_routed_car' in routed.columns:
        pred.loc[mask] = routed.loc[mask, 't_routed_car']
        profile.loc[mask] = 'car_' + legs.loc[mask, 'peak_str'].astype(str)
    return pred + _endpoint_term(legs, profile, edge_coefs)


def _predicted_baseline(legs: pd.DataFrame, routed: pd.DataFrame,
                        profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg NAIVE-BASELINE time: `t_baseline_<profile>` from 05, routed on
    `length / baseline_speed` edge weights (OSM speed limit for car), with no
    calibration and NO overheads — what someone routing on raw OSM speeds would
    get. The gap to the `time` row quantifies the payoff of the full calibrated
    model. Car: `t_baseline_car`, picked by `peak_str` in 05."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_baseline_s')
    for mode, (routed_profile, _overhead_profile) in profiles.items():
        mask = legs['mode_simplified'] == mode
        baseline_col = f't_baseline_{routed_profile}'
        if mask.any() and baseline_col in routed.columns:
            pred.loc[mask] = routed.loc[mask, baseline_col]
    mask = legs['mode_simplified'] == 'car'
    if mask.any() and 't_baseline_car' in routed.columns:
        pred.loc[mask] = routed.loc[mask, 't_baseline_car']
    return pred


def _predicted_length(legs: pd.DataFrame, routed: pd.DataFrame,
                      profiles: dict[str, tuple[str, str]]) -> pd.Series:
    """Per-leg predicted ROUTED distance (metres). Read from `length_<profile>`
    in survey_leg_times. Car: `length_car_{peak,base,night}` selected by `peak_str`
    (parallels the time-side aggregation in `05_leg_times`). No 'gross'
    distinction: overheads add time, not distance."""
    pred = pd.Series(np.nan, index=legs.index, name='predicted_length_m')
    for mode, (routed_profile, _overhead_profile) in profiles.items():
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


def _comparison_rows(legs: pd.DataFrame, routed: pd.DataFrame, overhead: pd.DataFrame,
                     profiles: dict[str, tuple[str, str]],
                     edge_coefs: dict[str, pd.DataFrame], quantities: tuple[str, ...],
                     mode_suffix: str = '', banded: bool = True) -> list[dict]:
    """Predict per-leg times + lengths with `profiles` (survey mode →
    (routed, overhead) profile) and return one stats row per (mode ×
    quantity × band) for `quantities`. `mode_suffix` is appended to the
    reported mode (e.g. `@ebike25` for a bracketing comparison); `banded=False`
    reports `ALL` rows only."""
    def label_of(mode: str) -> str:
        return f'{mode}{mode_suffix}'

    rows: list[dict] = []
    with step('assemble predicted gross times + routed lengths per leg'):
        pred_time = _predicted_gross(legs, routed, overhead, profiles)
        pred_net = _predicted_net(legs, routed, profiles, edge_coefs)
        pred_baseline = _predicted_baseline(legs, routed, profiles)
        pred_dist = _predicted_length(legs, routed, profiles)
        meas_time = legs['time_measured']
        meas_dist = legs['dist_measured']
        n_time = int((pred_time.notna() & meas_time.notna() & (meas_time > 0)).sum())
        n_baseline = int((pred_baseline.notna() & meas_time.notna() & (meas_time > 0)).sum())
        n_dist = int((pred_dist.notna() & meas_dist.notna() & (meas_dist > 0)).sum())
        logging.info(f"  → {n_time:,} legs with finite predicted+measured time; "
                     f"{n_baseline:,} with baseline; "
                     f"{n_dist:,} with finite predicted+measured distance")

    with step('compute per-mode stats (ALL + coarse distance bands)'):
        # Naive-baseline times (05's t_baseline_<profile>, no overheads) vs
        # measured: gap between 'time' and 'baseline_time' rows quantifies
        # the payoff of the calibrated model over speed-limit-only routing.
        dist_line_m = legs['dist_line'].astype(float)
        for quantity, meas, pred in (
            ('time',          meas_time, pred_time),
            ('net_time',      meas_time, pred_net),
            ('baseline_time', meas_time, pred_baseline),
            ('dist',          meas_dist, pred_dist),
        ):
            if quantity not in quantities:
                continue
            both = pred.notna() & meas.notna() & (meas > 0)
            for mode in sorted(legs.loc[both, 'mode_simplified'].unique()):
                mmask_all = both & (legs['mode_simplified'] == mode)
                rows.append({
                    'mode': label_of(mode), 'quantity': quantity, 'band': 'ALL',
                    **agreement_stats(meas.loc[mmask_all].to_numpy(dtype=float),
                             pred.loc[mmask_all].to_numpy(dtype=float))})
                if not banded or quantity == 'dist' or mode not in _STRATIFIED_MODES:
                    continue
                for label, lo, hi in _DIST_BANDS:
                    bmask = (dist_line_m >= lo) & (dist_line_m < hi)
                    mmask = mmask_all & bmask
                    if not mmask.any():
                        continue
                    rows.append({
                        'mode': label_of(mode), 'quantity': quantity, 'band': label,
                        **agreement_stats(meas.loc[mmask].to_numpy(dtype=float),
                                 pred.loc[mmask].to_numpy(dtype=float))})

    return rows


def _fit_legs(context, legs: pd.DataFrame, leg_set: str) -> pd.Series:
    """Legs of the calibration step `leg_set` checks, selected as that step
    selects them: 07a's overhead fit for `mtmc` (single-leg trips only), 04's
    edge-weight fit for the MOBIS leg sets (see the module docstring)."""
    scenario = get_scenario(context.scenario)
    gated = legs.index.isin(_m04._apply_survey_gates(legs).index)
    in_aoi = legs.index.isin(filter_legs_by_aoi(
        legs, load_aoi_polygon(context), crs=scenario.crs_main, label='validation legs').index)
    detour = legs['dist_measured'] / legs['dist_line']
    keep = pd.Series(False, index=legs.index)
    for smode, (nmode, ov_mode) in _FIT_MODES.items():
        sel = legs['mode_simplified'] == smode
        if leg_set == 'mtmc':
            cfg = _FIT_07A[ov_mode]
            sel &= ((legs['dist_line'] >= cfg.min_trip_distance)
                    & (legs['dist_line'] <= cfg.max_trip_distance)
                    & (detour < cfg.max_dist_measured_ratio))
        else:
            cfg = _m04._profile_config(nmode, scenario)
            sel &= ((legs['dist_line'] > cfg.min_trip_distance)
                    & (legs['dist_line'] <= cfg.max_trip_distance)
                    & (detour < _MAX_DETOUR_04)
                    & (legs[f'orig_snap_dist_{nmode}'] <= cfg.max_snap_distance)
                    & (legs[f'dest_snap_dist_{nmode}'] <= cfg.max_snap_distance))
        keep |= sel
    if leg_set == 'mtmc':
        keep &= legs['n_legs_in_trip'] == 1
    return keep & gated & in_aoi


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

    with step(f'select the legs of the calibration step {leg_set!r} checks'):
        n_before = len(legs)
        keep = _fit_legs(context, legs, leg_set)
        legs, routed, overhead = legs.loc[keep], routed.loc[keep], overhead.loc[keep]
        logging.info(f"  → {len(legs):,} / {n_before:,} legs kept")
    edge_coefs = {m: context.get_coefs(f'edge_weights_{m}')
                  for m in get_scenario(context.scenario).mode_configs}

    # GPS-tracked MOBIS legs record the movement only: compare network times there.
    quantities = ('time' if leg_set == 'mtmc' else 'net_time', 'baseline_time', 'dist')
    rows = _comparison_rows(legs, routed, overhead, _MODE_TO_PROFILE, edge_coefs, quantities)
    for mode, profile in _BRACKET_ROWS:
        sub = legs['mode_simplified'] == mode
        if not sub.any():
            continue
        with step(f'bracketing comparison: {mode} legs vs {profile} prediction'):
            rows += _comparison_rows(
                legs.loc[sub], routed.loc[sub], overhead.loc[sub],
                {mode: (profile, profile)}, edge_coefs, quantities,
                mode_suffix=f'@{profile}')
    for tod in _CAR_TIMES_OF_DAY:
        sub = (legs['mode_simplified'] == 'car') & (legs['peak_str'] == tod)
        if not sub.any():
            continue
        with step(f'car legs by time of day: {tod}'):
            rows += _comparison_rows(
                legs.loc[sub], routed.loc[sub], overhead.loc[sub],
                _MODE_TO_PROFILE, edge_coefs, quantities[:2],
                mode_suffix=f'@{tod}', banded=False)

    with step('summary'):
        _log_and_save(context, rows, leg_set)
    context.close()


def _log_and_save(context, rows: list[dict], leg_set: str) -> None:
    if not rows:
        logging.warning("no rows to summarize — check inputs.")
        return
    df = pd.DataFrame(rows)
    # Order: mode grouped; within each mode, time / net_time → baseline_time →
    # dist; within each quantity, ALL first, then the distance bands.
    quantity_order = {'time': 0, 'net_time': 1, 'baseline_time': 2, 'dist': 3}
    band_order = {'ALL': 0, **{label: i + 1 for i, (label, _, _) in enumerate(_DIST_BANDS)}}
    df['_qorder'] = df['quantity'].map(quantity_order).fillna(99)
    df['_border'] = df['band'].map(band_order).fillna(99)
    df = df.sort_values(['mode', '_qorder', '_border']).drop(
        columns=['_qorder', '_border']).reset_index(drop=True)

    # Rename for readable table headers (unit-generic; unit inferred from
    # `quantity` column: 'time' → seconds, 'dist' → metres).
    df = df.rename(columns={
        'n': 'n_legs',
        'bias_pct': 'bias%', 'med_resid_pct': 'med_resid%',
        'r2': 'R²_1to1', 'r2_fit': 'R²_fit', 'mae_pct': 'MAE%',
    })
    display_cols = ['mode', 'quantity', 'band', 'n_legs',
                    'bias', 'bias%', 'med_resid', 'med_resid%',
                    'R²_1to1', 'R²_fit', 'MAE%']

    # Save numeric-precise CSV to RESULTS (no console formatters applied).
    context.create_results(
        df[display_cols], 'technical_validation/' + survey_file('times_vs_survey.csv', leg_set),
        kws={'index': False, 'float_format': '%.4f'},
    )

    formatters = {
        'n_legs':     '{:>7,}'.format,
        'bias':       '{:+8.1f}'.format,
        'bias%':      '{:+.1f}%'.format,
        'med_resid':  '{:+8.1f}'.format,
        'med_resid%': '{:+.1f}%'.format,
        'R²_1to1':    '{:.4f}'.format,
        'R²_fit':     '{:.4f}'.format,
        'MAE%':       '{:.1f}%'.format,
    }
    table = df[display_cols].to_string(index=False, formatters=formatters,
                                       justify='right')
    logging.info(
        f"Predicted vs measured per-leg values ({leg_set}), per (mode × quantity × band). "
        "Quantity 'time' = calibrated GROSS door-to-door in SECONDS; "
        "'net_time' = calibrated routed time + 04's fitted endpoint term (no "
        "overheads, no α correction) in SECONDS; "
        "'baseline_time' = naive-baseline routing (speed limit only, no "
        "overheads) in SECONDS — gap between 'time' and 'baseline_time' "
        "rows quantifies calibration payoff; 'dist' = routed length in "
        "METRES (no gross/net — overheads add time, not distance).\n"
        "`bias > 0` = model OVER-predicts (mean); `med_resid > 0` = model above "
        "the median leg. Bands: `ALL` on every mode; bicycle modes and car "
        "overall additionally get distance-line strata "
        "(<2 km, 2-5 km, 5-25 km, >25 km).\n"
        + table)


variants = Variants([('leg_set', str)])
for _leg_set in SURVEY_LEG_SETS:
    variants.add(name=_leg_set, leg_set=_leg_set)


if __name__ == '__main__':
    variants.run(main, default='mtmc')
