"""
Calibrate trip-endpoint overhead coefficients — road modes only.

Fits, per road profile, on door-to-door TRIPS (`_main_mode_trips`: exactly
one leg of the profile's mode plus any walking legs, e.g. the walk from a
car park, which MTMC records as a leg of its own; for walking, single-leg
walking trips)::

    trip time = const + α · t_routed_<profile>(main leg)
                      + Σ β_i · endpoint_feature_i(trip start)
                      + Σ β_i · endpoint_feature_i(trip end)

Endpoint features are `density` + `snap_dist` of the mode's network. All
road configs set `share_endpoint_coefs`, which collapses each endpoint pair
into a single `β_i · (orig + dest)` regressor (overhead per unit of a
feature shouldn't depend on which end of the trip you're standing at).

Network times are calibrated per leg (04, the movement); overheads per
trip, so that walks to and from a parked vehicle and waiting between legs
count as overhead where they occur. Trip time runs from the first leg's
departure to the last leg's arrival.

α is fitted (α ≠ 1: residual scale bias in the routed times). The three
car profiles share one overhead (constant + endpoint features) and one α
(`_POOLED_PROFILES`): the overhead describes the place, and the profiles
differ only in their network times, fitted per time of day on MOBIS GPS
legs (04). Fitted separately on the self-reported MTMC trips, the
constants and α pick up which trips are made at each time of day (at night
mostly leisure trips, with harder parking), and the reports compress the
time-of-day differences that GPS shows; this holds in both MTMC waves
(2015, 2021), so it is not a COVID effect. walk_prm / ebike25 / ebike45
inherit rwalk / rbike at consume time via `resolve_road_overhead_column`.

**Rounded reports.** MTMC times are self-reported, mostly in 5-min steps,
and short trips are hardly ever reported as 0 min, so a short walk that
gets rounded becomes "5 min". OLS on the reports therefore inflates the
constant. `_fit_interval_em` treats each report as the interval of true
times it can stand for and fits by stochastic EM.

**Median fit.** Reported times have a long upper tail (stops, strolls),
heavier for walking than for cycling, which pulls mean fits up. The
production fit (`_FIT_METHOD = 'interval_median'`) is the interval EM with
a median regression in its M-step and a robust (MAD) error scale, i.e. it
targets the median reported time. `'interval_huber'` uses a Huber
regression on relative residuals instead (`_HUBER_T` robust standard
deviations: deviations within count as in a mean, larger ones linearly).
OLS and a plain median regression on the reports are logged alongside.

**Speed gate.** The survey flag `is_within_speed_envelope` (survey
preparation, shared with 04) tolerates rounded reports: a leg fails only if
every true time its report can stand for gives a speed outside its mode's
envelope (e.g. a 700 m walk reported as "5 min" passes); a trip is used
only if all its legs pass.

`fix_alpha` (per config) fixes the routed-time slope at 1 and fits only the
overhead terms, so long trips don't trade slope against constant.

Transit overheads are fitted in `main/07b_transit_overhead_coefs.py`.

Inputs (PRIVATE, under <scenario>/):
    generic/survey_legs.csv                  # from survey/02d
    generic/survey_leg_times.csv             # from survey/05

Inputs (scenario storage, under <scenario>/):
    properties/cells_snap.csv                # 02a → 02c — cell → node_id_<mode>, distance_<mode>
    properties/nodes_<mode>_extended.csv     # 02b — density_r500_norm
    shapes/cells.gpkg                        # 01 — AOI (`is_aoi`)

Outputs (scenario storage, under <scenario>/):
    coefs/<kind>/overheads_road.csv          # one column per road profile; <kind>
                                             # per the scenario's coef declaration

Run:
    python -m main.07a_road_overhead_coefs --scenario <name>
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, truncnorm

from aperta.errors import DataError
from aperta_atlas import coefs
from aperta_atlas.context import Context, Storage, init_context
from aperta_atlas.utils import step
from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from main.common import ROAD_OVERHEAD_DENSITY_COL
from preparation.switzerland.private.surveys.common import in_trip_order, report_intervals
from scenarios import get_scenario


# ---------------------------------------------------------------------------
# Shared config types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointFeature:
    """A per-endpoint regressor. `source(context, df)` returns
    `(orig_series, dest_series)` aligned to `df.index` (any frame with
    `orig_cell_id` / `dest_cell_id`; 07a passes trips). The calibrator
    fits `orig_<name>` + `dest_<name>` (or their sum when
    `share_endpoint_coefs`)."""
    name: str
    source: Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]


@dataclass(frozen=True)
class OverheadCalibConfig:
    """One overhead calibration — a road profile."""
    name: str                                # output column in the coefs CSV
    sub_modes: list[str]                     # `mode_simplified` values to include
    baseline_col: str                        # column in survey_leg_times.csv
    endpoint_features: list[EndpointFeature] = field(default_factory=list)
    peak_filter: str | None = None           # optional main leg `peak_str` filter (car)
    min_trip_distance: float = 1.0          # trip dist_line lower bound (m)
    max_trip_distance: float = 999_999.0     # trip dist_line upper bound (m)
    max_dist_measured_ratio: float = 5.0     # trip dist_measured / dist_line cap
    share_endpoint_coefs: bool = False       # fit β · (orig + dest) as a single regressor
    # Fix the routed-time slope α at 1 (fit only the overhead terms on report − t_routed).
    fix_alpha: bool = False


# ---------------------------------------------------------------------------
# Endpoint-feature source factories
# ---------------------------------------------------------------------------


def _cell_indexed(
    build_cell_series: Callable[[Context], pd.Series],
) -> Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]:
    """Wrap a per-cell Series builder into an EndpointFeature source —
    builds the cell-indexed Series once, then maps via orig / dest cell_id."""
    def _build(context: Context, legs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        cell_series = build_cell_series(context)
        return (legs['orig_cell_id'].map(cell_series),
                legs['dest_cell_id'].map(cell_series))
    return _build


def _density_source(mode: str) -> Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]:
    """Per-cell density: cells_snap[node_id_<mode>] → nodes_<mode>_extended[density]."""
    def _cell(context: Context) -> pd.Series:
        cells_snap = context.get_properties('cells', 'snap')
        node_props = context.get_properties('nodes', f'{mode}_extended')
        col = f'node_id_{mode}'
        df = cells_snap[[col]].dropna().copy()
        df[col] = df[col].astype(int)
        return df.join(node_props[[ROAD_OVERHEAD_DENSITY_COL]], on=col)[ROAD_OVERHEAD_DENSITY_COL]
    return _cell_indexed(_cell)


def _snap_dist_source(mode: str) -> Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]:
    """Per-cell snap distance to the mode graph."""
    return _cell_indexed(
        lambda context: context.get_properties('cells', 'snap')[f'distance_{mode}'])


# ---------------------------------------------------------------------------
# Fit method: OLS on the reported times, or interval EM on rounded reports
# ---------------------------------------------------------------------------

# Fit written to `overheads_road`: 'ols' | 'median' | 'interval' (mean) |
# 'interval_median' | 'interval_huber'. OLS and median are always logged too.
_FIT_METHOD = 'interval_median'
_EM_ESTIMATORS = {'interval': 'mean', 'interval_median': 'median', 'interval_huber': 'huber'}
_HUBER_T = 1.345               # Huber threshold in robust sd of the relative residuals
_EM_ITERATIONS = 50
_EM_BURN_IN = 20               # iterations discarded before averaging the coefficients
_EM_SEED = 0
_EM_MIN_PREDICTED_S = 30.0     # floor on predicted times (error sd scales with them)


def _interval_prob(lo, hi, mu, sd) -> np.ndarray:
    return norm.cdf((hi - mu) / sd) - norm.cdf((lo - mu) / sd)


def _draw_in_interval(lo, hi, mu, sd, rng) -> np.ndarray:
    """Normal(mu, sd) draws truncated to [lo, hi]; interval midpoint where
    the truncation is numerically degenerate (far tails)."""
    a, b = (lo - mu) / sd, (hi - mu) / sd
    with np.errstate(invalid='ignore', over='ignore'):
        t = truncnorm.rvs(a, b, loc=mu, scale=sd, random_state=rng)
    ok = np.isfinite(t) & (t >= lo) & (t <= hi)
    return np.where(ok, t, (lo + hi) / 2.0)


def _error_share(t: np.ndarray, mu: np.ndarray, estimator: str) -> float:
    """Error sd as a share of the predicted time: RMS of the relative errors for mean
    fits, else 1.4826 × their median absolute value (robust to the tail)."""
    rel = (t - mu) / mu
    if estimator == 'mean':
        return float(np.sqrt(np.mean(rel ** 2)))
    return float(1.4826 * np.median(np.abs(rel)))


def _m_step(Xv: np.ndarray, target: np.ndarray, estimator: str,
            mu: np.ndarray) -> np.ndarray:
    """Least squares ('mean'), median regression ('median'), or Huber regression on
    residuals relative to the current prediction `mu` ('huber')."""
    if estimator == 'median':
        return np.asarray(sm.QuantReg(target, Xv).fit(q=0.5, max_iter=5_000).params)
    if estimator == 'huber':
        w = 1.0 / mu
        model = sm.RLM(target * w, Xv * w[:, None], M=sm.robust.norms.HuberT(t=_HUBER_T))
        return np.asarray(model.fit().params)
    return np.linalg.lstsq(Xv, target, rcond=None)[0]


def _fit_interval_em(y: pd.Series, X: pd.DataFrame, start: pd.Series, *,
                     estimator: str = 'mean', offset: pd.Series | None = None) -> pd.Series:
    """Stochastic EM for `true time = offset + X·β + error`, observing only
    the interval each report stands for (`report_intervals`). Error sd is a
    constant share of the predicted time. `estimator` picks the M-step:
    'mean' (OLS), 'median' (median regression) or 'huber' (Huber regression
    on relative residuals); the latter two use a robust error scale.
    `offset` (e.g. the routed time when α is fixed at 1) enters the
    prediction with coefficient 1. Per iteration:

      1. Each report on a 5-min step is read as rounded or exact, drawn
         with probability ∝ p·P(true ∈ rounded interval) vs
         (1 − p)·P(true ∈ exact interval) under the current fit.
      2. A true time is drawn from the predicted distribution, truncated
         to the report's interval.
      3. The M-step regression on the drawn times → β; error share and the
         share of rounded reports p are updated from the draws.

    Starts from `start`, a fit on the reports (`_fit` passes OLS for 'mean',
    else the median regression); the result is β averaged over the
    iterations after `_EM_BURN_IN`."""
    rng = np.random.default_rng(_EM_SEED)
    y_s = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    off = np.zeros(len(y_s)) if offset is None else offset.to_numpy(dtype=float)
    exact_lo, exact_hi, rounded_lo, rounded_hi, on_step = report_intervals(y_s)
    beta = start.reindex(X.columns).to_numpy(dtype=float)
    mu = np.maximum(off + Xv @ beta, _EM_MIN_PREDICTED_S)
    err_share = _error_share(y_s, mu, estimator)
    p_rounded = float(on_step.mean())
    kept = []
    for it in range(1, _EM_ITERATIONS + 1):
        sd = err_share * mu
        w_rounded = np.where(
            on_step, p_rounded * _interval_prob(rounded_lo, rounded_hi, mu, sd), 0.0)
        w_exact = (1.0 - p_rounded) * _interval_prob(exact_lo, exact_hi, mu, sd)
        total = w_rounded + w_exact
        prob_rounded = np.where(total > 0, w_rounded / np.where(total > 0, total, 1.0), 1.0)
        rounded = on_step & (rng.random(len(y_s)) < prob_rounded)
        lo = np.where(rounded, rounded_lo, exact_lo)
        hi = np.where(rounded, rounded_hi, exact_hi)
        t = _draw_in_interval(lo, hi, mu, sd, rng)

        beta = _m_step(Xv, t - off, estimator, mu)
        mu = np.maximum(off + Xv @ beta, _EM_MIN_PREDICTED_S)
        err_share = _error_share(t, mu, estimator)
        p_rounded = float(rounded.mean())
        if it > _EM_BURN_IN:
            kept.append(beta)
        if it % 10 == 0:
            logging.info(f"     EM iter {it:>2}: const = {beta[0]:7.1f} s, "
                         f"rounded share = {p_rounded:.2f}, "
                         f"error share = {err_share:.2f}, mean drawn time = {t.mean():.0f} s")
    return pd.Series(np.mean(kept, axis=0), index=X.columns)


# ---------------------------------------------------------------------------
# Design + fits
# ---------------------------------------------------------------------------


def _fit_and_log_ols(
    y: pd.Series, X: pd.DataFrame, *,
    ref_series: pd.Series,
    name_width: int = 20,
) -> pd.Series:
    """Fit `sm.OLS(y, sm.add_constant(X))`, log R² + n + residual std +
    y-mean + ref_series-mean, then log each coefficient. Returns the
    coefficient Series."""
    result = sm.OLS(y, sm.add_constant(X, has_constant='add')).fit()
    logging.info(
        f"  → R² = {result.rsquared:.3f}, n = {int(result.nobs):,}; "
        f"residual std = {result.resid.std():.1f} s; "
        f"mean t_measured = {y.mean():.1f} s, "
        f"mean t_routed = {ref_series.mean():.1f} s")
    for name, coef in result.params.items():
        logging.info(f"     {str(name):>{name_width}s}: {coef:.4f}")
    return result.params


def _design(context: Context, cfg: OverheadCalibConfig, legs: pd.DataFrame,
            routed: pd.DataFrame) -> pd.DataFrame:
    """The profile's trips (`_main_mode_trips`) within its distance range, with the
    endpoint features at the trip's ends (summed when `share_endpoint_coefs`);
    rows with missing values dropped."""
    with step(f'{cfg.name}: trips (one {"/".join(cfg.sub_modes)} leg + walking legs) '
              f'+ features'):
        trips = _main_mode_trips(legs, routed, cfg)
        trips = trips[(trips['dist_line'] >= cfg.min_trip_distance)
                      & (trips['dist_line'] <= cfg.max_trip_distance)
                      & (trips['dist_measured'] / trips['dist_line']
                         < cfg.max_dist_measured_ratio)]
        df = trips[['time_measured', 't_routed', 'orig_cell_id', 'dest_cell_id']].copy()
        for f in cfg.endpoint_features:
            orig, dest = f.source(context, trips)
            df[f'orig_{f.name}'] = orig
            df[f'dest_{f.name}'] = dest
        before = len(df)
        df = df.dropna()
        if cfg.share_endpoint_coefs:
            for f in cfg.endpoint_features:
                df[f.name] = df[f'orig_{f.name}'] + df[f'dest_{f.name}']
        n_walk = int((trips.loc[df.index, 'n_walk_legs'] > 0).sum())
        logging.info(f"  → {len(df):,}/{before:,} trips after filter + dropna; "
                     f"{n_walk:,} with walking legs")
    return df


def _feature_cols(cfg: OverheadCalibConfig) -> list[str]:
    if cfg.share_endpoint_coefs:
        return [f.name for f in cfg.endpoint_features]
    return [c for f in cfg.endpoint_features for c in (f'orig_{f.name}', f'dest_{f.name}')]


def _fit(df: pd.DataFrame, X_cols: list[str], method: str, *,
         offset: pd.Series | None = None, label: str = '') -> pd.Series:
    """Fit `time_measured` (minus `offset`) on a constant + `X_cols` with `method`;
    OLS and a plain median regression are logged alongside."""
    y = df['time_measured'] - (offset if offset is not None else 0.0)
    X = sm.add_constant(df[X_cols], has_constant='add')
    ols = _fit_and_log_ols(y, df[X_cols], ref_series=df['t_routed'])
    fits = {'ols': ols, 'median': sm.QuantReg(y, X).fit(q=0.5, max_iter=5_000).params}
    if method in _EM_ESTIMATORS:
        estimator = _EM_ESTIMATORS[method]
        fits[method] = _fit_interval_em(
            df['time_measured'], X, start=fits['ols' if estimator == 'mean' else 'median'],
            estimator=estimator, offset=offset)
    logging.info(f"  → fits (writing {method!r}{label}):\n"
                 + pd.DataFrame(fits).round(4).to_string())
    return fits[method]


def _calibrate_overhead(
    context: Context, cfg: OverheadCalibConfig,
    legs: pd.DataFrame, routed: pd.DataFrame, method: str | None = None,
) -> pd.Series:
    """Fit one profile's overhead on its trips (`_design`) with `method` (default
    `_FIT_METHOD`). Returns the coefficient Series (`t_routed` = 1 when
    `cfg.fix_alpha`)."""
    method = method or _FIT_METHOD
    df = _design(context, cfg, legs, routed)
    with step(f'{cfg.name}: fit ({method})'):
        X_cols = ['t_routed'] + _feature_cols(cfg)
        offset = df['t_routed'] if cfg.fix_alpha else None
        if cfg.fix_alpha:
            X_cols.remove('t_routed')
        params = _fit(df, X_cols, method, offset=offset,
                      label=', α fixed at 1' if cfg.fix_alpha else '')
    if cfg.fix_alpha:
        params = pd.concat([params.iloc[:1], pd.Series({'t_routed': 1.0}), params.iloc[1:]])
    return params


def _calibrate_pooled(
    context: Context, cfgs: list[OverheadCalibConfig],
    legs: pd.DataFrame, routed: pd.DataFrame, method: str | None = None,
) -> dict[str, pd.Series]:
    """One overhead (constant + endpoint features) and one routed-time slope α
    shared by `cfgs` (one mode, one config per time-of-day profile); each trip
    enters with the routed time of its own profile, so the profiles differ only
    in their network times. Returns one coefficient Series per config name."""
    method = method or _FIT_METHOD
    df = pd.concat([_design(context, cfg, legs, routed) for cfg in cfgs])
    features = _feature_cols(cfgs[0])
    names = ', '.join(cfg.name for cfg in cfgs)
    with step(f'{names}: pooled fit ({method}; shared overhead and α)'):
        params = _fit(df, ['t_routed'] + features, method, label=f', pooled over {names}')
    return {cfg.name: params.copy() for cfg in cfgs}


# ---------------------------------------------------------------------------
# Road configs (production) + orchestration
# ---------------------------------------------------------------------------


def _road_config(
    name: str, mode: str, sub_modes: list[str], *,
    max_trip_distance: float = 999_999.0,
    peak_filter: str | None = None,
    fix_alpha: bool = False,
) -> OverheadCalibConfig:
    """Build a road overhead config with the standard shape: baseline
    `t_routed_<name>` + density + snap_dist endpoint features (both
    mode-specific), share_endpoint_coefs=True."""
    return OverheadCalibConfig(
        name=name,
        sub_modes=sub_modes,
        baseline_col=f't_routed_{name}',
        endpoint_features=[
            EndpointFeature('density',   _density_source(mode)),
            EndpointFeature('snap_dist', _snap_dist_source(mode)),
        ],
        peak_filter=peak_filter,
        max_trip_distance=max_trip_distance,
        share_endpoint_coefs=True,
        fix_alpha=fix_alpha,
    )


_ROAD_CONFIGS: list[OverheadCalibConfig] = [
    _road_config('rwalk',     'walk', ['walk'],  max_trip_distance=5_000.0),
    _road_config('rbike',     'bike', ['rbike'], max_trip_distance=25_000.0),
    _road_config('car_peak',  'car',  ['car'],   peak_filter='peak'),
    _road_config('car_base',  'car',  ['car'],   peak_filter='base'),
    _road_config('car_night', 'car',  ['car'],   peak_filter='night'),
]
# Car time-of-day profiles sharing one overhead and α (`_calibrate_pooled`).
_POOLED_PROFILES: tuple[str, ...] = ('car_peak', 'car_base', 'car_night')


def _fit_road_overheads(context: Context) -> pd.DataFrame:
    """Fit per-profile road overheads from MZMV survey legs + survey/05's
    routed times; the profiles in `_POOLED_PROFILES` share one overhead and α
    (`_calibrate_pooled`). Returns a `param × profile` DataFrame."""
    legs, routed = _load_survey_data(context)
    all_coefs: dict[str, pd.Series] = {}
    pooled: list[OverheadCalibConfig] = []
    for cfg in _ROAD_CONFIGS:
        if cfg.baseline_col not in routed.columns:
            logging.warning(
                f"profile={cfg.name}: {cfg.baseline_col!r} missing from "
                f"survey_leg_times.csv; skipping (run survey/05 for this profile).")
            continue
        if cfg.name in _POOLED_PROFILES:
            pooled.append(cfg)
            continue
        all_coefs[cfg.name] = _calibrate_overhead(context, cfg, legs, routed)
    if pooled:
        all_coefs.update(_calibrate_pooled(context, pooled, legs, routed))
    if not all_coefs:
        raise RuntimeError(
            "No overhead models fit — survey_leg_times.csv has no matching "
            "t_routed columns. Did survey/05 run successfully?")
    out = pd.DataFrame(all_coefs)
    out.index.name = 'param'
    return out


# ---------------------------------------------------------------------------
# Shared data loading + main
# ---------------------------------------------------------------------------


def _main_mode_trips(legs: pd.DataFrame, routed: pd.DataFrame,
                     cfg: OverheadCalibConfig) -> pd.DataFrame:
    """One row per trip of exactly one leg of `cfg.sub_modes` plus any walking
    legs (for walking: single-leg walking trips), with all its legs present
    (none dropped by the validity or AOI filters) and passing the gates, the
    trip within the gap envelope, and the main leg matching `cfg.peak_filter`.
    Columns: time_measured (first departure → last arrival, incl. waiting
    between legs; else the summed leg times), t_routed (the main leg's
    `cfg.baseline_col`), orig_cell_id / dest_cell_id (the trip's ends),
    dist_line (between the trip's ends), dist_measured (summed legs),
    n_walk_legs (besides the main leg)."""
    sub = in_trip_order(legs[legs['trip_id'].notna()])
    trip = sub['trip_id']
    is_main = sub['mode_simplified'].isin(cfg.sub_modes)
    is_other_walk = (sub['mode_simplified'] == 'walk') & ~is_main
    info = pd.DataFrame({
        'n_present': trip.groupby(trip).size(),
        'n_legs': sub.groupby('trip_id')['n_legs_in_trip'].first(),
        'n_main': is_main.groupby(trip).sum(),
        'n_walk': is_other_walk.groupby(trip).sum(),
        'all_pass': sub['passes_gates'].groupby(trip).all(),
    })
    keep = ((info['n_present'] == info['n_legs']) & (info['n_main'] == 1)
            & (info['n_main'] + info['n_walk'] == info['n_present']) & info['all_pass'])
    if 'is_within_gap_envelope' in sub.columns:
        keep &= sub['is_within_gap_envelope'].groupby(trip).min() == 1
    sub = sub[trip.isin(keep.index[keep])]
    g = sub.groupby('trip_id')
    first, last = g.head(1).set_index('trip_id'), g.tail(1).set_index('trip_id')
    main_leg = sub[sub['mode_simplified'].isin(cfg.sub_modes)]
    if cfg.peak_filter is not None:
        main_leg = main_leg[main_leg['peak_str'] == cfg.peak_filter]
    out = pd.DataFrame({'t_routed': routed.loc[main_leg.index, cfg.baseline_col].to_numpy()},
                       index=pd.Index(main_leg['trip_id'], name='trip_id'))
    idx = out.index
    span = (pd.to_datetime(last.loc[idx, 'datetime_arrival'], errors='coerce')
            - pd.to_datetime(first.loc[idx, 'datetime_departure'], errors='coerce')
            ).dt.total_seconds()
    span = span.where(span >= 0, span + 86_400)            # trips across midnight
    out['time_measured'] = span.fillna(g['time_measured'].sum().loc[idx])
    out['orig_cell_id'] = first.loc[idx, 'orig_cell_id']
    out['dest_cell_id'] = last.loc[idx, 'dest_cell_id']
    out['dist_line'] = np.hypot(last.loc[idx, 'dest_x'] - first.loc[idx, 'orig_x'],
                                last.loc[idx, 'dest_y'] - first.loc[idx, 'orig_y'])
    out['dist_measured'] = g['dist_measured'].sum().loc[idx]
    out['n_walk_legs'] = info.loc[idx, 'n_walk']
    return out


def _load_survey_data(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load survey legs + routed times (both PRIVATE). Context caches
    reads, so repeated calls are free after the first.

    Applies the same quality gates as `04_edge_weights.py`'s
    `_apply_survey_gates`: overhead = time_measured − routed time, so any
    noise / person-fitness bias that contaminates edge weights also
    contaminates the overhead residual.

    02d's `survey_legs.csv` already enforces `is_valid_domestic_trip`,
    `is_land_based`, `is_within_elevation_band`, and the mode-choice
    mode set — kept as defense-in-depth. The speed and detour envelope flags
    are kept as a column `passes_gates` rather than dropping legs, so that
    trips stay complete; `_main_mode_trips` uses only trips whose legs all
    pass.
    """
    with step('load survey legs + routed times'):
        legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
        routed = context.get_generic('survey_leg_times.csv', storage=Storage.PRIVATE)
        n_before = len(legs)
        if 'is_valid_domestic_trip' in legs.columns:
            legs = legs[legs['is_valid_domestic_trip'] == 1]
        if 'is_land_based' in legs.columns:
            legs = legs[legs['is_land_based'].fillna(1) == 1]
        if 'is_within_elevation_band' in legs.columns:
            legs = legs[legs['is_within_elevation_band'] == 1]
        n_after = len(legs)
        if n_after < n_before:
            logging.info(
                f"  Validity filters: {n_before:,} → {n_after:,} legs "
                f"kept ({n_before - n_after:,} dropped)")
        passes = pd.Series(True, index=legs.index)
        if 'is_within_speed_envelope' in legs.columns:
            passes &= legs['is_within_speed_envelope'] == 1
        if 'is_within_detour_envelope' in legs.columns:
            passes &= legs['is_within_detour_envelope'] == 1
        legs = legs.assign(passes_gates=passes)
        logging.info(f"  → {int(passes.sum()):,} / {len(legs):,} legs within the speed and "
                     f"detour envelopes")
        # AOI filter: scope training to legs whose BOTH endpoints fall
        # inside the scenario's AOI polygon. Pass-through for whole-
        # country scenarios, load-bearing for CV scenarios.
        scenario = get_scenario(context.scenario)
        aoi_polygon = load_aoi_polygon(context)
        legs = filter_legs_by_aoi(
            legs, aoi_polygon, crs=scenario.crs_main, label='survey legs')
        # Reindex by stable leg_id (both files carry it as their index).
        # Aligns routed to the post-gate legs subset AND surfaces any
        # stale-05 / re-run-02d mismatch as a low overlap ratio.
        n_overlap = int(routed.index.isin(legs.index).sum())
        if n_overlap < 0.5 * len(legs):
            raise DataError(
                f"survey_leg_times.csv covers only {n_overlap:,}/{len(legs):,} "
                f"post-gate legs — likely stale after a 02d re-run. "
                f"Re-run survey/05_leg_times.")
        routed = routed.reindex(legs.index)
        t_routed_cols = sorted(c for c in routed.columns if c.startswith('t_routed_'))
        logging.info(f"  → {len(legs):,} legs "
                     f"({n_overlap:,} matched in routed), "
                     f"t_routed columns: {t_routed_cols}")
    return legs, routed


def main():
    """Resolve `overheads_road` per the scenario's coef declaration."""
    context = init_context()
    coefs.resolve(
        context,
        name='overheads_road',
        calibrate_fn=lambda: _fit_road_overheads(context),
    )
    context.close()


if __name__ == '__main__':
    main()
