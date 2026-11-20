"""
Estimate mode-choice utility coefficients on the door-to-door survey trips via
biogeme.

One observation per trip of survey/02d's trip set (`main.common.survey_trips`,
the trips 07a and 07b fit the overheads on): the chosen mode is the trip's main
mode, and every alternative's time runs door to door between the trip's ends
(survey/05, overheads from survey/08a), as the atlas computes it between cells.

Per-mode utility functions with:
  - mode-specific ASC (walk is the base, ASC_walk fixed at 0)
  - travel time in MINUTES (`time_form` axis below; piecewise coefs
    `b_time_pw_<mode>_<lo>_<hi>`), centered at T = `UTILITY_T_REF_MIN` (main.common),
    so each ASC is the utility of a T-minute trip; T is stored as `t_ref_min`
  - location features (`location_features` axis below)
  - sociodemographic covariates (weather, sex, age bins, income bins),
    with per-mode applicability declared in `_SD_COL_MODES`
  - per-feature "base mode" logic: location + SD terms are added ONLY
    to modes listed in the feature's free-modes tuple; the omitted mode
    is the feature's base (β = 0 implicit)
  - peak/night shifts (`car_peak_night`, `peak_night_asc_modes` axes below)
  - `weight_person` for weighted MLE

Variants: `default`, and `default_route` (route aggregates instead of endpoint
features; a spec comparison, declared in switzerland-h10 only and run by hand).
`default_linlog` (linear + log time terms, the form before 2026-11-15) is declared in no
scenario, so `main` skips it; misc/utility_time_shape fits it next to `default`.

Variant axes:
  - `time`:              `net` | `gross` (adds survey/08a overheads)
  - `ownership`:         `endogenous` (time-only availability) |
                         `exogenous` (AND vehicle ownership from sd cols)
  - `ebikes`:            `exclude` | `separate` (5th mode) | `nested`
                         (bike/ebike nest with shared MU)
  - `subset_fraction`:   uniform random sample of the filtered rows
  - `location_features`: `endpoints_combined` (orig+dest summed, one
                         coef per feature) | `endpoints_separate` (two
                         coefs per feature) | `route` (per-trip route
                         aggregates from survey/05)
  - `time_form`:         `piecewise` (one slope per time band, knots per mode
                         in `_PIECEWISE_KNOTS_MIN`) | `linlog` (linear + log
                         time, centered at T; fitted only, 09b reads piecewise)
  - `car_peak_night`:    comma-separated tokens combining additively;
                         `asc`, `time`, or a biogeme variable name.
                         Empty = no peak/night effect. See
                         `_parse_car_peak_night` for the full grammar.
  - `peak_night_asc_modes`: comma-separated non-car modes (`rbike`,
                         `transit`, `ebike25`) that also get peak and
                         night ASC shifts, relative to walking (the base,
                         whose ASC is 0). Their ODMs (09b) are at base
                         hours; 10 applies the shifts in the cross-modal
                         profiles. Empty = car only.
  - `darkness`:          departure in darkness, for bike, transit and car
                         vs walk (`_DARK_SD_COL_MODES`). Off in all variants.
  - `seasons`:           season terms on the same modes
                         (`_SEASON_SD_COL_MODES`). Off in all variants.

Inputs (PRIVATE):
    generic/survey_trips.csv                         # survey/02d (--variant mtmc_trips)
    generic/survey_trip_times.csv                    # survey/05 (--variant mtmc_trips)
    generic/survey_trip_overheads.csv                # survey/08a (--variant mtmc_trips)

Inputs (scenario storage, under <scenario>/):
    properties/nodes_{walk,bike,car}_extended.csv    # 02b
    properties/nodes_car_flows_avg.csv               # 03b
    shapes/cells.gpkg                                # AOI (`is_aoi` cells)

Outputs (scenario storage, under <scenario>/):
    coefs/<kind>/utility_<variant>.csv               # β table (rows: b_*, asc_*,
                                                      # t_ref_min; cols: value,
                                                      # rob_std_err, rob_t_test,
                                                      # rob_p_value)
    coefs/<kind>/utility_<variant>_stats.csv         # companion sidecar (rows:
                                                      # sd_avg_<sd_col>; two-col
                                                      # `key, value`); same coef
                                                      # source as the parent
    biogeme/<variant>/                               # SCRATCH — biogeme logs + reports

Run (default variant `default`; variants not in `scenario.coefs` are skipped):
    python -m main.09a_utility_estimation --scenario <name>
Single variant, or all:
    python -m main.09a_utility_estimation --scenario <name> --variant <name|all>
"""

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import biogeme.database as db
from biogeme import biogeme, models
from biogeme.expressions import Beta, Numeric, Variable
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

from aperta_atlas import coefs
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from main.common import (
    CHOICE_QUALITY_FLAGS,
    TRIP_SET,
    UTILITY_T_REF_MIN,
    car_overhead_by_peak,
    car_routed_bias_by_peak,
    compute_sd_averages,
    piecewise_time_coef_name,
    survey_file,
)
from main.utility_config import FEATURE_SCALE as _FEATURE_SCALE
from main.utility_config import POINT_FEATURE_SOURCES as _POINT_FEATURE_SOURCES
from mode_configs import MODE_CONFIGS, TRANSIT_MODE_CONFIG
from scenarios import get_scenario


# Per-mode physical floor on per-trip time (seconds); protects the fit from
# below-floor values (e.g. short routed times + negative overheads). NPVM has no
# same-zone transit times; such trips have no transit time and transit is unavailable.
_MIN_ROUTE_TIME_S_PER_MODE: dict[str, float] = {
    'walk':    MODE_CONFIGS['walk'].min_route_time_s,
    'rbike':   MODE_CONFIGS['bike'].min_route_time_s,
    'ebike25': MODE_CONFIGS['bike'].min_route_time_s,
    'car':     MODE_CONFIGS['car'].min_route_time_s,
    'transit': TRANSIT_MODE_CONFIG.min_route_time_s,
}


# Keys match the survey's `mode_simplified` values directly (no aliasing).
# ebike45 is dropped by the active-mode filter (too few observations);
# ebike25 participates only when `variant.ebikes != 'exclude'`.
_MODE_IDS: dict[str, int] = {
    'walk':    1,
    'rbike':   2,
    'transit': 3,
    'car':     4,
    'ebike25': 5,   # only present when variant.ebikes != 'exclude'
}

# Mode → routing profile in survey_trip_times.csv + survey_trip_overheads.csv.
# Transit uses `t_routed_transit_z2z` (NPVM z2z lookup); car uses the
# peak-str-aggregated `t_routed_car` column.
_MODE_TO_PROFILE: dict[str, str] = {
    'walk':    'rwalk',
    'rbike':   'rbike',
    'ebike25': 'ebike25',
    'car':     'car',
}

# Root feature name → free-modes tuple. `_features_for_anchors` suffixes
# each root by anchor: `_orig`+`_dest` (endpoints_separate) or `_od`
# (endpoints_combined). Modes NOT listed are the feature's base (β = 0
# implicit). Values looked up per feature's source graph
# (`_POINT_FEATURE_SOURCES`).
_ENDPOINT_FEATURES: dict[str, tuple[str, ...]] = {
    'density_r500_norm':          ('rbike', 'transit', 'car'),
    # Bike traffic stress around the endpoint: shares of edges at levels 0-1 (none / very low)
    # and 4-5 (high / very high); 2-3 is the reference. Walk is the base mode, as for density
    # and traffic flow, so the rbike coefficient is the cycling-specific effect; against car,
    # both shares mostly mark car-unfriendly places (paths, pedestrian areas; arterials).
    'bike_stress_share_01_r250':  ('rbike', 'transit', 'car'),
    'bike_stress_share_45_r250':  ('rbike', 'transit', 'car'),
    'speed_limit_avg_r250':       ('rbike', 'transit', 'car'),
    'traffic_flow_avg_r250':      ('rbike', 'car', 'transit'),
    'mean_abs_slope_r250':        ('walk', 'rbike', 'transit'),
}

# ROUTE variant: the endpoint features, as length-weighted means along a route (survey/05's
# per-profile aggregates in `survey_trip_times.csv`), with the same free modes. Each copies from
# ONE profile's column per `_ROUTE_FEATURE_DEFAULTS`: the car at base hours (a representative
# urban route) for density, speed limit and traffic flow; cycling for bike stress and slope.
_ROUTE_FEATURES: dict[str, tuple[str, ...]] = {
    f'{root}_route': modes for root, modes in _ENDPOINT_FEATURES.items()}
_ROUTE_FEATURE_DEFAULTS: dict[str, str] = {
    'density_r500_norm_route':          'density_r500_norm_car_base',
    'bike_stress_share_01_r250_route':  'bike_stress_share_01_r250_rbike',
    'bike_stress_share_45_r250_route':  'bike_stress_share_45_r250_rbike',
    'speed_limit_avg_r250_route':       'speed_limit_avg_r250_car_base',
    'traffic_flow_avg_r250_route':      'traffic_flow_avg_r250_car_base',
    'mean_abs_slope_r250_route':        'mean_abs_slope_r250_rbike',
}

# Sociodemographic covariates → free-coef modes. Weather is bike-only (car
# and transit users don't switch daily based on weather).
_SD_COL_MODES: dict[str, tuple[str, ...]] = {
    'sd_bool_weather_good':         ('rbike',),
    'sd_bool_weather_bad':          ('rbike',),
    'sd_cat_sex_m':                 ('rbike', 'transit', 'car'),
    'sd_cat_age_7to13':             ('rbike', 'transit', 'car'),
    'sd_cat_age_14to18':            ('rbike', 'transit', 'car'),
    'sd_cat_age_19to25':            ('rbike', 'transit', 'car'),
    'sd_cat_age_51to65':            ('rbike', 'transit', 'car'),
    'sd_cat_age_66to75':            ('rbike', 'transit', 'car'),
    'sd_cat_age_76+':               ('rbike', 'transit', 'car'),
    'sd_cat_income_leq4000':        ('rbike', 'transit', 'car'),
    'sd_cat_income_4001to8000':     ('rbike', 'transit', 'car'),
    'sd_cat_income_12001to16000':   ('rbike', 'transit', 'car'),
    'sd_cat_income_16001+':         ('rbike', 'transit', 'car'),
}
_SD_COLS: tuple[str, ...] = tuple(_SD_COL_MODES)

# `darkness`: departure in darkness (sun below civil twilight, survey `is_daylight` = 0),
# relative to walking, like the covariates above. `seasons`: season terms (summer is the
# reference) on the same modes, to control for confounding: darkness goes with winter, which
# also changes mode choice (cold, snow), and the night profile is dark by the hour, in every
# season. Named `sd_*` so that 09b evaluates them at their averages over the trips, like the
# covariates (sidecar `sd_avg_*`; `add_condition_columns` in `_compute_utility_stats`), so the
# season terms drop out of the released utilities.
# Tested 2026-11-13 (switzerland-h10) for the night profile of the cross-modal logsum (67 % of
# its legs depart in darkness, 7 % at base and peak hours). With seasons, cycling only:
# darkness +0.03 (ns); seasons vs summer −0.55 (winter), −0.20 (spring), −0.16 (fall). Without
# seasons: darkness −0.05 cycling, +0.10 transit, +0.01 car (ns). Both off: even the largest
# naive effect shifts night utilities by ~0.06, and the released utilities are annual averages.
_DARK_SD_COL_MODES: dict[str, tuple[str, ...]] = {
    'sd_bool_dark':          ('rbike', 'transit', 'car'),
}
_SEASON_SD_COL_MODES: dict[str, tuple[str, ...]] = {
    'sd_cat_season_winter':  ('rbike', 'transit', 'car'),
    'sd_cat_season_spring':  ('rbike', 'transit', 'car'),
    'sd_cat_season_fall':    ('rbike', 'transit', 'car'),
}


def add_condition_columns(trips: pd.DataFrame) -> pd.DataFrame:
    """The trips with the darkness and season columns (`_DARK_SD_COL_MODES`,
    `_SEASON_SD_COL_MODES`), from the survey's `is_daylight` and `season` at departure (NaN
    where those are missing)."""
    trips = trips.copy()
    trips['sd_bool_dark'] = 1.0 - pd.to_numeric(trips['is_daylight'], errors='coerce')
    season = trips['season']
    for s in ('winter', 'spring', 'fall'):
        trips[f'sd_cat_season_{s}'] = (season == s).astype(float).where(season.notna())
    return trips


def _sd_col_modes(variant) -> dict[str, tuple[str, ...]]:
    """The covariates of `variant`: `_SD_COL_MODES`, plus darkness for `darkness` and the
    season terms for `seasons`."""
    return {**_SD_COL_MODES,
            **(_DARK_SD_COL_MODES if variant.darkness else {}),
            **(_SEASON_SD_COL_MODES if variant.seasons else {})}


def _node_feature_anchors(location_features: str) -> tuple[str, ...]:
    """Anchor tuple for the active `location_features` variant."""
    if location_features == 'endpoints_separate':
        return ('orig', 'dest')
    if location_features == 'endpoints_combined':
        return ('od',)
    if location_features == 'route':
        return ('route',)
    raise ValueError(
        f"location_features must be 'endpoints_separate', "
        f"'endpoints_combined', or 'route'; got {location_features!r}.")


def _features_for_anchors(anchors: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Flat `biogeme_name → free-modes` dict for `anchors`. Endpoint anchors
    suffix each `_ENDPOINT_FEATURES` root; route uses `_ROUTE_FEATURES`
    verbatim (its keys already carry the `_route` suffix)."""
    if anchors in (('orig', 'dest'), ('od',)):
        return {
            f'{root}_{suffix}': modes
            for root, modes in _ENDPOINT_FEATURES.items()
            for suffix in anchors
        }
    if anchors == ('route',):
        return _ROUTE_FEATURES
    raise ValueError(
        f"Unknown anchor tuple {anchors!r} — expected `('orig', 'dest')`, "
        f"`('od',)`, or `('route',)`.")

_TIME_CAP_MIN = 5 * 60
_MAX_TRIP_DIST_LINE_M = 200_000

# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _load_and_join(context) -> pd.DataFrame:
    """Load the survey trips (`TRIP_SET`) + routed times + overheads. Left-join on the
    trip index; missing columns become NaN and are handled downstream by the
    availability + dropna logic. Scopes trips to the scenario AOI so
    utility estimation trains on trips within the calibration region."""
    trips = context.get_generic(survey_file('survey_legs.csv', TRIP_SET),
                                storage=Storage.PRIVATE)
    routed = context.get_generic(survey_file('survey_leg_times.csv', TRIP_SET),
                                 storage=Storage.PRIVATE)
    overhead = context.get_generic(survey_file('survey_leg_overheads.csv', TRIP_SET),
                                   storage=Storage.PRIVATE)
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)
    trips = filter_legs_by_aoi(
        trips, aoi_polygon, crs=scenario.crs_main, label='survey trips')
    df = trips.join(routed, how='left').join(overhead, how='left')
    logging.info(f"  → {len(df):,} trips joined (trips {len(trips):,}, "
                 f"routed {len(routed):,}, overhead {len(overhead):,})")
    return df


def _compute_mode_times(
    df: pd.DataFrame, time_type: str, include_ebikes: bool,
) -> pd.DataFrame:
    """Populate `t_<mode>_min` (minutes) per mode for all modes in the choice
    set. `net` = t_routed only; `gross` adds origin + destination overheads."""
    df = df.copy()   # defragment before per-column inserts (pandas warning)
    modes = list(_MODE_TO_PROFILE) + ['transit']
    if not include_ebikes:
        modes = [m for m in modes if m != 'ebike25']
    for mode in modes:
        profile = _MODE_TO_PROFILE.get(mode, mode)   # transit: 'transit'
        net_s = df['t_routed_transit_z2z' if mode == 'transit' else f't_routed_{profile}']
        total_s = net_s
        if time_type == 'gross' and mode == 'car':
            total_s = (net_s
                       + car_overhead_by_peak(df, 'orig').fillna(0)
                       + car_overhead_by_peak(df, 'dest').fillna(0)
                       - car_routed_bias_by_peak(df).fillna(0))
        elif time_type == 'gross':
            # gross = t_routed + orig_ov + dest_ov − t_routed_bias (α scaling of the routed
            # time; see main.common.route_time_alpha).
            bias = df.get(f't_routed_bias_{profile}')
            total_s = (net_s
                       + df[f't_overhead_orig_{profile}'].fillna(0)
                       + df[f't_overhead_dest_{profile}'].fillna(0)
                       - (bias.fillna(0) if bias is not None else 0.0))
        min_s = _MIN_ROUTE_TIME_S_PER_MODE.get(mode, 0.0)
        if min_s > 0:
            total_s = np.maximum(total_s, min_s)
        df[f't_{mode}_min'] = total_s / 60.0
        # Pre-compute log(t) as a column — safer than biogeme's `log()` on
        # data that may transiently be non-positive during optimisation.
        t = df[f't_{mode}_min']
        df[f't_{mode}_min_log'] = np.log(t.where(t > 0))
    return df


def _join_point_node_features(
    df: pd.DataFrame, context, anchors: tuple[str, ...],
) -> pd.DataFrame:
    """Build every biogeme feature column implied by the variant. Endpoint
    anchors look each feature up on its source graph's node file (per
    `_POINT_FEATURE_SOURCES`); route anchors copy from one default profile column
    per `_ROUTE_FEATURE_DEFAULTS`. Meters-scale features are divided by
    `_FEATURE_SCALE`.
    """
    df = df.copy()   # defragment before per-column inserts (pandas warning)
    features = _features_for_anchors(anchors)

    if anchors in (('orig', 'dest'), ('od',)):
        nodes_by_data_name: dict[str, pd.DataFrame] = {}
        def _load(data_name: str) -> pd.DataFrame:
            if data_name not in nodes_by_data_name:
                nodes_by_data_name[data_name] = context.get_properties(
                    'nodes', data_name)
            return nodes_by_data_name[data_name]

        for src_col, (source_mode, data_name) in _POINT_FEATURE_SOURCES.items():
            nodes_src = _load(data_name)
            if src_col not in nodes_src.columns:
                logging.warning(f"  → point-feature source {src_col!r} missing from "
                                f"nodes_{data_name}.csv; utilities using it drop rows.")
                continue
            s = nodes_src[src_col]
            # Always populate _orig/_dest; combined mode also adds _od = _orig + _dest.
            for side in ('orig', 'dest'):
                node_id_col = f'{side}_node_id_{source_mode}'
                bn = f'{src_col}_{side}'
                scale = _FEATURE_SCALE.get(src_col, 1.0)
                df[bn] = df[node_id_col].map(s) / scale
            if anchors == ('od',):
                bn_od = f'{src_col}_od'
                df[bn_od] = df[f'{src_col}_orig'] + df[f'{src_col}_dest']
    elif anchors == ('route',):
        for bn, source_col in _ROUTE_FEATURE_DEFAULTS.items():
            if source_col not in df.columns:
                logging.warning(f"  → route-feature source {source_col!r} not in "
                                f"survey_trip_times.csv; {bn!r} rows dropped.")
                continue
            s = df[source_col]
            scale = _FEATURE_SCALE.get(bn, 1.0)
            df[bn] = s / scale
    else:
        raise ValueError(f"Unknown anchor tuple {anchors!r}.")

    have = [bn for bn in features if bn in df.columns]
    missing = [bn for bn in features if bn not in df.columns]
    logging.info(f"  → biogeme feature columns built: {len(have)}/{len(features)} "
                 f"({'; '.join(have[:6])}{'…' if len(have) > 6 else ''})"
                 + (f"; MISSING: {missing}" if missing else ''))
    return df


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def _ownership_available(df: pd.DataFrame, mode: str) -> pd.Series:
    """Boolean per trip: is the household eligible for this mode based on
    ownership? Used only when `ownership='exogenous'`."""
    if mode == 'car':
        # sd_cat_n_cars_0 == 0 means "NOT in the zero-cars bin".
        if 'sd_cat_n_cars_0' in df.columns:
            return df['sd_cat_n_cars_0'] == 0
        return df['sd_ordinal_n_cars'].astype(str) != '0'
    if mode == 'rbike':
        if 'sd_cat_n_bikes_0' in df.columns:
            return df['sd_cat_n_bikes_0'] == 0
        return df['sd_ordinal_n_bikes'].astype(str) != '0'
    if mode == 'ebike25':
        if 'sd_cat_n_ebikes_0' in df.columns:
            return df['sd_cat_n_ebikes_0'] == 0
        return df['sd_ordinal_n_ebikes'].astype(str) != '0'
    if mode == 'transit':
        pass_cols = [c for c in df.columns if c.startswith('sd_bool_transitpass_')]
        if not pass_cols:
            return pd.Series(True, index=df.index)
        return df[pass_cols].sum(axis=1) > 0
    return pd.Series(True, index=df.index)   # walk always available


def _compute_availability(
    df: pd.DataFrame, ownership: str, modes: list[str],
) -> pd.DataFrame:
    """Populate `availability_<mode>` + `availability_mismatch` +
    `n_available_modes`. Endogenous = time-only; exogenous = time AND
    ownership check."""
    df = df.copy()   # defragment before per-column inserts (pandas warning)
    for mode in modes:
        t_ok = df[f't_{mode}_min'].notna() & (df[f't_{mode}_min'] < _TIME_CAP_MIN)
        if ownership == 'exogenous':
            t_ok = t_ok & _ownership_available(df, mode)
        df[f'availability_{mode}'] = t_ok.astype(int)
    # Mismatch: the chosen mode is not marked available. `mode_simplified`
    # values not in `_MODE_IDS` (e.g. ebike45) are dropped later.
    chosen = df['mode_simplified']
    df['_chosen_mode'] = chosen
    mismatch = pd.Series(False, index=df.index)
    for mode in modes:
        m = (chosen == mode) & (df[f'availability_{mode}'] == 0)
        mismatch = mismatch | m
    df['availability_mismatch'] = mismatch.astype(int)
    df['n_available_modes'] = df[[f'availability_{m}' for m in modes]].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Biogeme model build + estimate
# ---------------------------------------------------------------------------
def _time_beta_names(mode: str) -> tuple[str, str]:
    """Coef names for the linear + log time term of a mode."""
    return f'b_time_{mode}', f'b_time_log_{mode}'


# Knots (minutes) of the `piecewise` time form: one slope per band, the last band open. One
# grid (5, 10, 20, 45, 90), each mode using the knots within its times: in a finer test
# (2026-11-15) slopes changed smoothly in log time, and neighboring bands 1.5× apart were often
# equal, while finer bands picked up trip composition (non-monotonic slopes). Transit has no
# knot at 20 min: a separate 0–20 band was uninformative (+0.014 ± 0.007; short transit trips
# vary little in modeled time and compete with walking), so 0–45 shares one slope.
_PIECEWISE_KNOTS_MIN: dict[str, tuple[float, ...]] = {
    'walk':    (0, 5, 10, 20, 45),
    'rbike':   (0, 5, 10, 20, 45),
    'transit': (0, 45, 90),
    'car':     (0, 10, 20, 45, 90),
}


def _piecewise_bands(mode: str) -> list[tuple[float, float]]:
    """A mode's `piecewise` time bands (minutes), the last one open (`hi` = inf)."""
    knots = _PIECEWISE_KNOTS_MIN[mode]
    return list(zip(knots, [*knots[1:], math.inf]))


def _piecewise_beta_names(mode: str) -> list[str]:
    """Coef names of a mode's `piecewise` time form, one per band (`…_<last knot>_inf`)."""
    return [piecewise_time_coef_name(mode, lo, hi) for lo, hi in _piecewise_bands(mode)]


def _parse_car_peak_night(spec: str) -> tuple[bool, bool, list[str]]:
    """Parse comma-separated `car_peak_night` spec → `(has_asc, has_time,
    feat_names)`. Tokens combine ADDITIVELY. Reserved: `asc` (ASC modifier),
    `time` (time-coef interaction); anything else is a biogeme variable
    name that gets one interaction pair. Feature variables must be
    registered by the active `location_features` variant. Empty string
    = no peak/night effect.

    Examples:
      `'asc'`                                  — ASC modifier only
      `'asc,time'`                             — ASC + time interactions
      `'asc,traffic_flow_avg_r250_od'`         — ASC + one feature (combined endpoints)
    """
    tokens = [t.strip() for t in spec.split(',') if t.strip()]
    has_asc = 'asc' in tokens
    has_time = 'time' in tokens
    feat_names = [t for t in tokens if t not in ('asc', 'time')]
    return has_asc, has_time, feat_names


def _parse_peak_night_asc_modes(spec: str) -> tuple[str, ...]:
    """Parse the comma-separated `peak_night_asc_modes` spec. Walking is the base
    (ASC 0) and the car is covered by `car_peak_night`, so neither is allowed."""
    modes = tuple(t.strip() for t in spec.split(',') if t.strip())
    bad = [m for m in modes if m in ('walk', 'car')]
    if bad:
        raise ValueError(f"peak_night_asc_modes: {bad} not allowed (walk is the base; "
                         f"car uses `car_peak_night`).")
    return modes


def _utility_expression(
    mode: str, betas: dict, vars_: dict,
    features: dict[str, tuple[str, ...]],
    car_peak_night: str,
    peak_night_asc_modes: tuple[str, ...],
    sd_col_modes: dict[str, tuple[str, ...]],
    time_form: str,
) -> Any:
    """Biogeme utility expression for one mode. Location + SD terms are
    added only when the mode is in the feature's free-modes tuple.
    See `_parse_car_peak_night` for the peak/night interaction grammar.
    """
    has_asc, has_time, feat_names = _parse_car_peak_night(car_peak_night)
    exprs = []
    if mode != 'walk':
        exprs.append(betas[f'asc_{mode}'])
        if (mode == 'car' and has_asc) or mode in peak_night_asc_modes:
            exprs.append(betas[f'asc_{mode}_peak'] * vars_['hour_peak'])
            exprs.append(betas[f'asc_{mode}_night'] * vars_['hour_night'])
    # Time terms centered at UTILITY_T_REF_MIN (see the module docstring).
    t_c = vars_[f't_{mode}_min'] - UTILITY_T_REF_MIN
    if time_form == 'piecewise':
        pw_betas = [betas[n] for n in _piecewise_beta_names(mode)]
        exprs.append(models.piecewise_formula(
            vars_[f't_{mode}_min'], [*_PIECEWISE_KNOTS_MIN[mode], None], pw_betas))
        # Centering: minus the term's value at T (a constant per band).
        for beta, (lo, hi) in zip(pw_betas, _piecewise_bands(mode)):
            at_ref = min(max(UTILITY_T_REF_MIN - lo, 0.0), hi - lo)
            if at_ref > 0:
                exprs.append(-at_ref * beta)
    else:
        exprs.append(betas[f'b_time_{mode}'] * t_c)
        exprs.append(betas[f'b_time_log_{mode}']
                     * (vars_[f't_{mode}_min_log'] - math.log(UTILITY_T_REF_MIN)))
    if mode == 'car' and has_time:
        exprs.append(betas['b_time_car_peak'] * vars_['hour_peak'] * t_c)
        exprs.append(betas['b_time_car_night'] * vars_['hour_night'] * t_c)
    if mode == 'car' and feat_names:
        missing = [f for f in feat_names if f not in vars_]
        if missing:
            raise KeyError(f"car_peak_night refs variables not on biogeme frame: "
                           f"{missing}. Load them via `location_features`.")
        for var in feat_names:
            exprs.append(betas[f'b_car_peak_x_{var}']
                         * vars_['hour_peak'] * vars_[var])
            exprs.append(betas[f'b_car_night_x_{var}']
                         * vars_['hour_night'] * vars_[var])
    for bn, feat_modes in features.items():
        if mode in feat_modes:
            exprs.append(betas[f'b_{bn}_{mode}'] * vars_[bn])
    for sd, sd_modes in sd_col_modes.items():
        if mode in sd_modes:
            exprs.append(betas[f'b_{sd}_{mode}'] * vars_[sd])
    return sum(exprs[1:], exprs[0])


def _register_betas(
    modes: list[str], features: dict[str, tuple[str, ...]],
    car_peak_night: str,
    peak_night_asc_modes: tuple[str, ...],
    sd_col_modes: dict[str, tuple[str, ...]],
    time_form: str,
) -> dict:
    """Instantiate all Beta objects. Walk's ASC is fixed at 0 (base mode).
    Location + SD betas registered only for modes with a free coef.
    Car peak/night adds extra betas per `_parse_car_peak_night`; the modes in
    `peak_night_asc_modes` get peak/night ASC shifts.
    """
    betas: dict[str, Beta] = {}
    for mode in modes:
        if mode != 'walk':
            betas[f'asc_{mode}'] = Beta(f'asc_{mode}', 0, None, None, 0)
        if mode in peak_night_asc_modes:
            for name in (f'asc_{mode}_peak', f'asc_{mode}_night'):
                betas[name] = Beta(name, 0, None, None, 0)
        time_names = (_piecewise_beta_names(mode) if time_form == 'piecewise'
                      else _time_beta_names(mode))
        for name in time_names:
            betas[name] = Beta(name, 0, None, None, 0)
        for bn, feat_modes in features.items():
            if mode in feat_modes:
                name = f'b_{bn}_{mode}'
                betas[name] = Beta(name, 0, None, None, 0)
        for sd, sd_modes in sd_col_modes.items():
            if mode in sd_modes:
                name = f'b_{sd}_{mode}'
                betas[name] = Beta(name, 0, None, None, 0)
    if 'car' in modes:
        has_asc, has_time, feat_names = _parse_car_peak_night(car_peak_night)
        peak_night_names: list[str] = []
        if has_asc:
            peak_night_names += ['asc_car_peak', 'asc_car_night']
        if has_time:
            peak_night_names += ['b_time_car_peak', 'b_time_car_night']
        for var in feat_names:
            peak_night_names += [f'b_car_peak_x_{var}',
                                 f'b_car_night_x_{var}']
        for name in peak_night_names:
            betas[name] = Beta(name, 0, None, None, 0)
    return betas


def _biogeme_variables(
    modes: list[str], features: dict[str, tuple[str, ...]],
    sd_cols: tuple[str, ...] = _SD_COLS,
) -> dict:
    """Instantiate all biogeme `Variable` handles referenced by the model."""
    v: dict[str, Variable] = {}
    for mode in modes:
        v[f't_{mode}_min'] = Variable(f't_{mode}_min')
        v[f't_{mode}_min_log'] = Variable(f't_{mode}_min_log')
        v[f'availability_{mode}'] = Variable(f'availability_{mode}')
    for bn in features:
        v[bn] = Variable(bn)
    for sd in sd_cols:
        v[sd] = Variable(sd)
    v['hour_peak'] = Variable('hour_peak')
    v['hour_night'] = Variable('hour_night')
    v['weight_person'] = Variable('weight_person')
    v['choice'] = Variable('choice')
    return v


def _fit_and_extract(
    df: pd.DataFrame, variant,
    features: dict[str, tuple[str, ...]],
    run_dir: Path,
) -> tuple[pd.DataFrame, float, float]:
    """Set up + estimate the biogeme MNL / nested-logit model. Returns the
    coefficient table and rho² (bar) vs the null model and vs a constants-only
    MNL (ASCs only, same trips, weights and availability), which reproduces the
    mode shares and so shows what times, features and SD terms add."""
    modes = ['walk', 'rbike', 'transit', 'car']
    if variant.ebikes != 'exclude':
        modes.append('ebike25')

    # biogeme wants a numeric `choice` column matching the alternative IDs.
    df = df.copy()
    df['choice'] = df['_chosen_mode'].map(_MODE_IDS).astype(int)

    sd_col_modes = _sd_col_modes(variant)
    v = _biogeme_variables(modes, features, tuple(sd_col_modes))
    pn_modes = _parse_peak_night_asc_modes(variant.peak_night_asc_modes)
    missing = [m for m in pn_modes if m not in modes]
    if missing:
        raise ValueError(f"peak_night_asc_modes: {missing} not among the fitted modes {modes}.")
    if variant.time_form not in ('linlog', 'piecewise'):
        raise ValueError(f"time_form: {variant.time_form!r} not in ('linlog', 'piecewise').")
    betas = _register_betas(modes, features, variant.car_peak_night, pn_modes, sd_col_modes,
                            time_form=variant.time_form)

    utilities = {
        _MODE_IDS[mode]: _utility_expression(
            mode, betas, v, features,
            car_peak_night=variant.car_peak_night, peak_night_asc_modes=pn_modes,
            sd_col_modes=sd_col_modes, time_form=variant.time_form)
        for mode in modes
    }
    availability = {
        _MODE_IDS[mode]: v[f'availability_{mode}'] for mode in modes
    }

    if variant.ebikes == 'nested':
        mu = Beta('MU', 1.5, 1.0, 10.0, 0)
        bike_nest = OneNestForNestedLogit(
            nest_param=mu,
            list_of_alternatives=[_MODE_IDS['rbike'], _MODE_IDS['ebike25']],
            name='bikes',
        )
        nests = NestsForNestedLogit(
            choice_set=list(utilities), tuple_of_nests=(bike_nest,))
        log_prob = models.lognested(utilities, availability, nests, v['choice'])
    else:
        log_prob = models.loglogit(utilities, availability, v['choice'])

    # Subset to numeric cols; biogeme's audit iterates every column and
    # any leftover string column raises `Cannot interpret 'StringDtype'`.
    biogeme_cols = ['choice', 'weight_person', 'hour_peak', 'hour_night']
    for mode in modes:
        biogeme_cols += [f't_{mode}_min', f't_{mode}_min_log',
                         f'availability_{mode}']
    biogeme_cols += list(features)
    biogeme_cols += [c for c in sd_col_modes if c in df.columns]
    biogeme_cols = [c for c in dict.fromkeys(biogeme_cols) if c in df.columns]
    df_bg = df[biogeme_cols].copy()

    # Per-variant subdir isolates biogeme's iteration log / pickle / HTML.
    run_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        database = db.Database('trips', df_bg)
        bg = biogeme.BIOGEME(database, {
            'loglike': log_prob, 'weight': v['weight_person'],
        })
        bg.modelName = f'utility_{variant.name}'
        bg.calculate_null_loglikelihood(availability)
        results = bg.estimate()

        asc = {mode: Beta(f'asc_{mode}', 0, None, None, 0) for mode in modes if mode != 'walk'}
        utilities_c = {_MODE_IDS[mode]: asc.get(mode, Numeric(0)) for mode in modes}
        bg_c = biogeme.BIOGEME(db.Database('trips_constants', df_bg), {
            'loglike': models.loglogit(utilities_c, availability, v['choice']),
            'weight': v['weight_person'],
        })
        bg_c.modelName = f'utility_{variant.name}_constants'
        results_c = bg_c.estimate()
    finally:
        os.chdir(cwd)

    params = results.get_estimated_parameters()
    rho2 = float(results.data.rhoBarSquareNull)
    rho2_const = 1.0 - ((results.data.logLike - results.data.nparam)
                        / (results_c.data.logLike - results_c.data.nparam))
    logging.info(f"  → final log-likelihood {results.data.logLike:,.1f} "
                 f"({results.data.nparam} parameters); constants only "
                 f"{results_c.data.logLike:,.1f}")
    # Rename biogeme's default columns to snake_case so the on-disk CSV
    # follows this codebase's convention.
    params = params.rename(columns={
        'Value':        'value',
        'Rob. Std err': 'rob_std_err',
        'Rob. t-test':  'rob_t_test',
        'Rob. p-value': 'rob_p_value',
    })
    return params, rho2, rho2_const


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(variant):
    context = init_context()
    v = variant.name
    scenario = get_scenario(context.scenario)
    if f'utility_{v}' not in scenario.coefs:
        logging.info(f"utility_{v} not declared in scenario {scenario.name!r} "
                     f"(no utility grid); skipping")
        context.close()
        return
    coefs.resolve(
        context,
        name=f'utility_{v}',
        calibrate_fn=lambda: _fit_utility(context, variant),
    )
    # Companion sidecar — SD averages. Same source declaration
    # as the parent (validated in Scenario.__post_init__).
    coefs.resolve(
        context,
        name=f'utility_{v}_stats',
        calibrate_fn=lambda: _compute_utility_stats(context),
    )
    context.close()


def _compute_utility_stats(context) -> pd.DataFrame:
    """SD averages, as a two-column `key, value` DataFrame for the
    `utility_<variant>_stats` sidecar. Only called on `Calibrate()` scenarios."""
    raw_trips = context.get_generic(survey_file('survey_legs.csv', TRIP_SET),
                                    storage=Storage.PRIVATE)
    # Scope SD averages to the AOI so utility-stats align with the
    # region trained on in `_fit_utility` / `_load_and_join`.
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)
    raw_trips = filter_legs_by_aoi(
        raw_trips, aoi_polygon, crs=scenario.crs_main, label='SD-stats trips')
    if 'is_daylight' in raw_trips.columns:   # averages for `darkness` / `seasons`
        raw_trips = add_condition_columns(raw_trips)
    sd_averages = compute_sd_averages(raw_trips)
    stats = pd.DataFrame({'value': list(sd_averages.values())},
                         index=[f'sd_avg_{col}' for col in sd_averages])
    stats.index.name = 'key'
    logging.info(f"  → utility stats: {len(sd_averages)} sd_avg rows")
    return stats


def _modeled_sample(
    context, variant,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """The trips the model is fitted on, after every filter (weights normalized to mean 1),
    with per-mode times, availability and the variant's location features; also returns
    those features with their free modes. Also read by misc/mtmc_modal_split."""
    logging.info(f"variant: time={variant.time}, ownership={variant.ownership}, "
                 f"ebikes={variant.ebikes}, "
                 f"subset_fraction={variant.subset_fraction}, "
                 f"location_features={variant.location_features}, "
                 f"time_form={variant.time_form}, "
                 f"car_peak_night={variant.car_peak_night}, "
                 f"peak_night_asc_modes={variant.peak_night_asc_modes}, "
                 f"darkness={variant.darkness}, seasons={variant.seasons}")

    anchors = _node_feature_anchors(variant.location_features)
    features = _features_for_anchors(anchors)

    with step('load survey trips + routed times + overheads'):
        df = _load_and_join(context)
        if variant.darkness or variant.seasons:
            df = add_condition_columns(df)

    with step(f'compute per-mode times ({variant.time}, minutes)'):
        df = _compute_mode_times(df, variant.time, include_ebikes=(variant.ebikes != 'exclude'))

    with step(f'join per-node point features (anchors={anchors})'):
        df = _join_point_node_features(df, context, anchors)

    modes_for_avail = ['walk', 'rbike', 'transit', 'car']
    if variant.ebikes != 'exclude':
        modes_for_avail.append('ebike25')

    with step(f'compute availability flags ({variant.ownership})'):
        df = _compute_availability(df, variant.ownership, modes_for_avail)

    with step('plausibility filter (CHOICE_QUALITY_FLAGS)'):
        n_before = len(df)
        for flag in CHOICE_QUALITY_FLAGS:
            df = df[df[flag] == 1]
        logging.info(f"  → {len(df):,}/{n_before:,} kept "
                     f"({100*len(df)/max(n_before,1):.1f} %)")

    with step('filter to modeled sample'):
        n0 = len(df)
        df = df[df['dist_line'] <= _MAX_TRIP_DIST_LINE_M]
        active_modes = set(modes_for_avail)
        df = df[df['_chosen_mode'].isin(active_modes)]
        df = df[df['availability_mismatch'] == 0]
        df = df[df['n_available_modes'] >= 2]
        # Drop rows with NaN in any biogeme-evaluated feature column.
        need = (
            list(features)
            + ['hour_peak', 'hour_night', 'weight_person']
            + list(_sd_col_modes(variant))
        )
        need = [c for c in need if c in df.columns]
        df = df.dropna(subset=need)
        logging.info(f"  → n = {len(df):,} / {n0:,}  after filters")
        df = df.copy()
        # NaN per-mode times → above-cap placeholder; availability_<mode>=0
        # already excludes them from that trip's choice set.
        placeholder_t = float(_TIME_CAP_MIN + 1)
        placeholder_t_log = float(np.log(placeholder_t))
        for m in modes_for_avail:
            df[f't_{m}_min'] = df[f't_{m}_min'].fillna(placeholder_t)
            df[f't_{m}_min_log'] = df[f't_{m}_min_log'].fillna(placeholder_t_log)
        if variant.subset_fraction < 1.0:
            n_before = len(df)
            df = df.sample(frac=variant.subset_fraction, random_state=42)
            logging.info(f"  → subset_fraction={variant.subset_fraction}: "
                         f"{len(df):,}/{n_before:,} kept")
        df['weight_person'] = df['weight_person'] / df['weight_person'].sum() * len(df)
    return df, features


def _fit_utility(context, variant) -> pd.DataFrame:
    """Fit the biogeme MNL / nested-logit model and return the coef
    DataFrame for `coefs.resolve` to persist. Only called on
    `Calibrate()` scenarios."""
    df, features = _modeled_sample(context, variant)

    with step('fit biogeme model'):
        scratch_root = Path(context.path_for(Storage.SCRATCH, 'biogeme'))
        params, rho2, rho2_const = _fit_and_extract(
            df, variant, features, run_dir=scratch_root / variant.name)
        logging.info(f"  → {len(df):,} trips; rho² (bar) vs null = {rho2:.4f}, "
                     f"vs constants only = {rho2_const:.4f}")
        with pd.option_context('display.max_rows', None):
            logging.info('\n' + params[['value', 'rob_p_value']].round(4).to_string())

    out = params[['value', 'rob_std_err', 'rob_t_test', 'rob_p_value']].copy()
    out.loc['t_ref_min', 'value'] = UTILITY_T_REF_MIN
    return out


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
variants = Variants([
    ('time', str),
    ('ownership', str),
    ('ebikes', str),
    ('subset_fraction', float),
    ('location_features', str),
    ('time_form', str),
    ('car_peak_night', str),
    ('peak_night_asc_modes', str),
    ('darkness', bool),
    ('seasons', bool),
])
# No peak / night shifts: one all-day constant per mode, so the released utilities hold at all
# hours and the car's hour profiles differ only by their congested times. The shifts mostly
# reflect who travels when (commuting at peak), not how well a mode serves those hours; with
# them, the car's peak constant offset the peak congestion. Fitted 2026-11-14 (switzerland-h10,
# all p < 0.05): peak transit +0.40, cycling +0.20, car +0.13; night cycling +0.19, car +0.10,
# transit +0.06.
_DEFAULT = dict(
    time='gross',
    ownership='endogenous',
    ebikes='exclude',
    subset_fraction=1.0,
    location_features='endpoints_combined',
    time_form='piecewise',
    car_peak_night='',
    peak_night_asc_modes='',
    darkness=False,
    seasons=False,
)
variants.add(name='default', **_DEFAULT)
variants.add(name='default_route', **{**_DEFAULT, 'location_features': 'route'})
variants.add(name='default_linlog', **{**_DEFAULT, 'time_form': 'linlog'})

if __name__ == '__main__':
    variants.run(main, default='default')
