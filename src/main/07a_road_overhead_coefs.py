"""
Calibrate trip-endpoint overhead coefficients — road modes only.

Fits the OLS::

    time_measured = const + α · t_routed_<profile>
                          + Σ β_i · endpoint_feature_i(orig)
                          + Σ β_i · endpoint_feature_i(dest)

with the optional `share_endpoint_coefs` collapsing the endpoint pair
into a single `β_i · (orig + dest)` regressor per feature (physically:
overhead per unit of a feature shouldn't depend on which end of the
trip you're standing at).

Per-profile α lets the fit scale freely (α ≠ 1 → residual bias in 05's
edge weights). Endpoint features are `density` + `snap_dist` per mode.
ebike25 / ebike45 inherit rbike at consume-time via
`resolve_road_overhead_column`.

An `is_multi_leg_trip` (0/1) regressor is fit alongside as a NUISANCE
CONTROL — it absorbs the systematic overhead difference between legs
of single-leg trips (MTMC's `time_measured` is door-to-door because
the leg IS the trip) and legs of multi-leg trips (mode-change walking
is recorded in separate walk legs, so the leg carries less overhead).
Consumer semantics: `per_cell_road_overheads` in `main.common` does
NOT read this coefficient. That's intentional — ODM OD pairs represent
solo (single-leg) trips, so the applied overhead uses `is_multi_leg_trip
= 0` implicitly. The coefficient's role is to prevent the multi-leg
contamination from biasing the other estimates (`const`, `density`,
`snap_dist`).

Transit fitting lives in a sibling `main/07b_transit_overhead_coefs.py`
— it needs TRIP-level (not leg-level) data because NPVM z2z is door-to-
door while a transit LEG's `time_measured` is only stop-to-stop.

Inputs (PRIVATE, under <scenario>/):
    generic/survey_legs.csv                  # from survey/02d
    generic/survey_leg_times.csv             # from survey/05

Inputs (PUBLIC, under <scenario>/):
    properties/cells_snap.csv                # 02a → 02c — cell → node_id_<mode>, distance_<mode>
    properties/nodes_<mode>_extended.csv     # 02b — density_r*_norm

Outputs (PUBLIC, under <scenario>/):
    coefs/<kind>/overheads_road.csv          # one column per road profile

Run:
    python -m main.07a_road_overhead_coefs --scenario <name>
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import statsmodels.api as sm

from aperta.errors import DataError
from aperta_atlas import coefs
from aperta_atlas.context import Context, Storage, init_context
from aperta_atlas.utils import step
from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from scenarios import get_scenario


# Per-node density column joined to cells via cells_snap → nodes_<mode>_extended.
_DENSITY_COL = 'density_r500_norm'


# ---------------------------------------------------------------------------
# Shared config types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointFeature:
    """A per-endpoint OLS regressor. `source(context, legs)` returns
    `(orig_series, dest_series)` aligned to `legs.index`. The
    calibrator fits `orig_<name>` + `dest_<name>` (or their sum when
    `share_endpoint_coefs`)."""
    name: str
    source: Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]


@dataclass(frozen=True)
class OverheadCalibConfig:
    """One OLS overhead calibration — a road profile."""
    name: str                                # output column in the coefs CSV
    sub_modes: list[str]                     # `mode_simplified` values to include
    baseline_col: str                        # column in survey_leg_times.csv
    endpoint_features: list[EndpointFeature] = field(default_factory=list)
    peak_filter: str | None = None           # optional legs['peak_str'] filter (car)
    min_trip_distance: float = 50.0          # dist_line lower bound (m)
    max_trip_distance: float = 999_999.0     # dist_line upper bound (m)
    max_dist_measured_ratio: float = 3.0     # dist_measured / dist_line cap
    share_endpoint_coefs: bool = False       # fit β · (orig + dest) as a single regressor
    # Add `is_multi_leg_trip` (0/1) as a leg-level regressor. Captures
    # the systematic overhead difference between legs of single-leg
    # trips (where the leg IS the whole trip → full door-to-door
    # overhead applies) and legs of multi-leg trips (where mode-change
    # walking is recorded in separate walk legs, so this leg's overhead
    # should be smaller). Expected sign: negative — β_multi_leg reduces
    # the fitted overhead for multi-leg-trip legs.
    include_multi_leg_indicator: bool = False


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
        return df.join(node_props[[_DENSITY_COL]], on=col)[_DENSITY_COL]
    return _cell_indexed(_cell)


def _snap_dist_source(mode: str) -> Callable[[Context, pd.DataFrame], tuple[pd.Series, pd.Series]]:
    """Per-cell snap distance to the mode graph."""
    return _cell_indexed(
        lambda context: context.get_properties('cells', 'snap')[f'distance_{mode}'])


# ---------------------------------------------------------------------------
# OLS helper
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


def _calibrate_overhead(
    context: Context, cfg: OverheadCalibConfig,
    legs: pd.DataFrame, routed: pd.DataFrame,
) -> pd.Series:
    """Filter legs by config's mode/distance/peak, join baseline + endpoint
    features, drop NaNs, fit OLS. Returns the coefficient Series."""
    with step(f'{cfg.name}: subset legs + join baseline + features'):
        mask = (
            legs['mode_simplified'].isin(cfg.sub_modes)
            & (legs['dist_line'] >= cfg.min_trip_distance)
            & (legs['dist_line'] <= cfg.max_trip_distance)
            & (legs['dist_measured'] / legs['dist_line'] < cfg.max_dist_measured_ratio)
        )
        if cfg.peak_filter is not None:
            mask &= (legs['peak_str'] == cfg.peak_filter)
        legs_sub = legs.loc[mask]
        df = legs_sub[['time_measured', 'orig_cell_id', 'dest_cell_id']].copy()
        df['t_routed'] = routed.loc[df.index, cfg.baseline_col]
        for f in cfg.endpoint_features:
            orig, dest = f.source(context, legs_sub)
            df[f'orig_{f.name}'] = orig
            df[f'dest_{f.name}'] = dest
        if cfg.include_multi_leg_indicator:
            if 'n_legs_in_trip' not in legs_sub.columns:
                logging.warning(
                    f"  ⚠ {cfg.name}: include_multi_leg_indicator=True but "
                    f"'n_legs_in_trip' missing from legs — skipping the "
                    f"indicator (add it to survey/02d's `_KEEP_COLS`).")
            else:
                df['is_multi_leg_trip'] = (
                    legs_sub['n_legs_in_trip'] > 1).astype(int)
        before = len(df)
        df = df.dropna()
        logging.info(f"  → {len(df):,}/{before:,} legs after filter + dropna")
        if 'is_multi_leg_trip' in df.columns:
            n_multi = int(df['is_multi_leg_trip'].sum())
            logging.info(
                f"  → is_multi_leg_trip: {n_multi:,}/{len(df):,} "
                f"({100*n_multi/max(len(df),1):.1f} %) are multi-leg trips")

    with step(f'{cfg.name}: fit OLS'):
        if cfg.share_endpoint_coefs and cfg.endpoint_features:
            df = df.copy()
            for f in cfg.endpoint_features:
                df[f.name] = df[f'orig_{f.name}'] + df[f'dest_{f.name}']
            X_cols = ['t_routed'] + [f.name for f in cfg.endpoint_features]
        else:
            X_cols = ['t_routed']
            for f in cfg.endpoint_features:
                X_cols += [f'orig_{f.name}', f'dest_{f.name}']
        if 'is_multi_leg_trip' in df.columns:
            X_cols.append('is_multi_leg_trip')
        return _fit_and_log_ols(
            df['time_measured'], df[X_cols], ref_series=df['t_routed'],
            name_width=50 if any('_dev' in f.name for f in cfg.endpoint_features) else 20)


# ---------------------------------------------------------------------------
# Road configs (production) + orchestration
# ---------------------------------------------------------------------------


def _road_config(
    name: str, mode: str, sub_modes: list[str], *,
    max_trip_distance: float = 999_999.0,
    peak_filter: str | None = None,
) -> OverheadCalibConfig:
    """Build a road overhead config with the standard shape: baseline
    `t_routed_<name>` + density + snap_dist endpoint features (both
    mode-specific), share_endpoint_coefs=True. `include_multi_leg_indicator=True`
    adds an `is_multi_leg_trip` regressor to capture the systematic
    overhead difference for legs of multi-leg trips (MTMC records mode-
    change walking as separate walk legs, so a leg embedded in a multi-
    leg trip carries less door-to-door overhead than the same leg
    standing alone as a single-leg trip)."""
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
        include_multi_leg_indicator=True,
        share_endpoint_coefs=True,
    )


_ROAD_CONFIGS: list[OverheadCalibConfig] = [
    _road_config('rwalk',     'walk', ['walk'],  max_trip_distance=5_000.0),
    _road_config('rbike',     'bike', ['rbike'], max_trip_distance=25_000.0),
    _road_config('car_peak',  'car',  ['car'],   peak_filter='peak'),
    _road_config('car_base',  'car',  ['car'],   peak_filter='base'),
    _road_config('car_night', 'car',  ['car'],   peak_filter='night'),
]


def _fit_road_overheads(context: Context) -> pd.DataFrame:
    """Fit per-profile road overheads from MZMV survey legs + survey/05's
    routed times. Returns a `param × profile` DataFrame."""
    legs, routed = _load_survey_data(context)
    all_coefs: dict[str, pd.Series] = {}
    for cfg in _ROAD_CONFIGS:
        if cfg.baseline_col not in routed.columns:
            logging.warning(
                f"profile={cfg.name}: {cfg.baseline_col!r} missing from "
                f"survey_leg_times.csv; skipping (run survey/05 for this profile).")
            continue
        all_coefs[cfg.name] = _calibrate_overhead(context, cfg, legs, routed)
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


def _load_survey_data(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load survey legs + routed times (both PRIVATE). Context caches
    reads, so calling this per calibration pass is free after the first.

    Applies the same trip-quality gates as `04_edge_weights.py`'s
    `_apply_survey_gates` — this ensures 07's overhead calibration
    trains on the same subset of trips that 04's edge weights did.
    Rationale: overhead = time_measured − baseline_routed_time, so any
    noise / person-fitness bias that contaminates edge weights also
    contaminates the overhead residual.

    02d's `survey_legs.csv` already enforces `is_valid_domestic_trip`,
    `is_land_based`, `is_within_elevation_band`, and the mode-choice
    mode set — kept as defense-in-depth. `is_within_speed_envelope` +
    `is_within_detour_envelope` are calibration-specific (not enforced
    in 02d because mode-choice modeling doesn't need them) so they're
    re-applied here.
    """
    with step('load survey legs + routed times'):
        legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
        routed = context.get_generic('survey_leg_times.csv', storage=Storage.PRIVATE)
        n_before = len(legs)
        if 'is_valid_domestic_trip' in legs.columns:
            legs = legs[legs['is_valid_domestic_trip'] == 1]
        if 'is_land_based' in legs.columns:
            legs = legs[legs['is_land_based'].fillna(1) == 1]
        if 'is_within_speed_envelope' in legs.columns:
            legs = legs[legs['is_within_speed_envelope'] == 1]
        if 'is_within_detour_envelope' in legs.columns:
            legs = legs[legs['is_within_detour_envelope'] == 1]
        if 'is_within_elevation_band' in legs.columns:
            legs = legs[legs['is_within_elevation_band'] == 1]
        n_after = len(legs)
        if n_after < n_before:
            logging.info(
                f"  Trip-quality gates: {n_before:,} → {n_after:,} legs "
                f"kept ({n_before - n_after:,} dropped)")
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
    """Resolve `overheads_road` per scenario's coefs declaration.
    `overheads_transit` (calibration + diagnostic) lives in the sibling
    `main/07b_transit_overhead_coefs.py`."""
    context = init_context()
    coefs.resolve(
        context,
        name='overheads_road',
        calibrate_fn=lambda: _fit_road_overheads(context),
    )
    context.close()


if __name__ == '__main__':
    main()
