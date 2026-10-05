"""
Estimate mode-choice utility coefficients on the survey legs via biogeme.

Per-mode utility functions with:
  - mode-specific ASC (walk is the base, ASC_walk fixed at 0)
  - travel time as `β_t · t + β_ln · log(t)` (t in MINUTES)
  - location features driven by the `location_features` variant axis
    (endpoints_combined / endpoints_separate / route — see below)
  - sociodemographic covariates (weather, sex, age bins, income bins),
    with per-mode applicability declared in `_SD_COL_MODES`
  - per-feature "base mode" logic: location + SD terms are added ONLY
    to modes listed in the feature's free-modes tuple; the omitted mode
    is the feature's base (β = 0 implicit)
  - optional car peak/night interactions (ASC / time / feature — see
    `_parse_car_peak_night`)
  - `weight_person` for weighted MLE

Variant axes:
  - `time`:              `net` | `gross` (adds survey/08a overheads)
  - `ownership`:         `endogenous` (time-only availability) |
                         `exogenous` (AND vehicle ownership from sd cols)
  - `ebikes`:            `exclude` | `separate` (5th mode) | `nested`
                         (bike/ebike nest with shared MU)
  - `single_leg_only`:   drop multi-leg-trip legs (transit exempt)
  - `subset_fraction`:   uniform random sample of the filtered rows
  - `location_features`: `endpoints_combined` (orig+dest summed, one
                         coef per feature) | `endpoints_separate` (two
                         coefs per feature) | `route` (per-leg route
                         aggregates from survey/05)
  - `car_peak_night`:    comma-separated tokens combining additively;
                         `asc`, `time`, or a biogeme variable name.
                         Empty = no peak/night effect. See
                         `_parse_car_peak_night` for the full grammar.

Inputs (PRIVATE):
    generic/survey_legs.csv                          # survey/02d
    generic/survey_leg_times.csv                     # survey/05
    generic/survey_leg_overheads.csv                 # survey/08a
    generic/survey_summary.csv                       # survey/08c — for t_cut_min bake-in

Inputs (PUBLIC):
    properties/nodes_{walk,bike,car}_extended.csv    # 02b
    properties/nodes_car_flows_avg.csv               # 03b
    preparation/switzerland/npvm/odm/npvm_2023_transit_{idx,travel_time}.npz

Outputs (PUBLIC):
    coefs/<kind>/utility_<variant>.csv               # β table (rows: b_*, asc_*;
                                                      # cols: value, rob_std_err,
                                                      # rob_t_test, rob_p_value).
    coefs/<kind>/utility_<variant>_stats.csv         # companion sidecar (rows:
                                                      # sd_avg_<sd_col> +
                                                      # t_cut_min_<mode>; two-col
                                                      # `key, value`). Source
                                                      # mirrors the parent's coef
                                                      # declaration (validated in
                                                      # Scenario.__post_init__).

Run:
    python -m main.09a_utility_estimation --scenario <name>
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import biogeme.database as db
from biogeme import biogeme, models
from biogeme.expressions import Beta, Variable
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

from aperta_atlas import coefs
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from main.common import car_overhead_by_peak, car_routed_bias_by_peak
from mode_configs import MODE_CONFIGS, TRANSIT_MODE_CONFIG
from scenarios import get_scenario


# Per-mode physical floor on per-leg time (seconds); protects the fit from
# below-floor values (e.g. NPVM same-zone transit = 0 s + negative overheads).
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

# Mode → routing profile in survey_leg_times.csv + survey_leg_overheads.csv.
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
    'bike_infra_score_avg_r250':  ('walk', 'rbike', 'transit'),
    'speed_limit_avg_r250':       ('rbike', ),
    'traffic_flow_avg_r250':      ('rbike', 'car', 'transit'),
    'mean_abs_slope_r250':        ('walk', 'rbike', 'transit'),
}

# Per-feature source graph (single source of truth in main.utility_config,
# shared with 09b / 10 via `common.join_util_node_features`).
from main.utility_config import POINT_FEATURE_SOURCES as _POINT_FEATURE_SOURCES

# ROUTE variant: features from survey/05's per-profile route aggregates in
# `survey_leg_times.csv`. Each biogeme feature copies from ONE default-profile
# column per `_ROUTE_FEATURE_DEFAULTS`. Elevation here is cumulative-along-
# route (survey/05 sums), not endpoint Δ; n_traffic_signals is route-only.
_ROUTE_FEATURES: dict[str, tuple[str, ...]] = {
    # 'elev_gain_route':                  ('walk', 'rbike', 'transit'),  # CAR = base
    # 'elev_loss_route':                  ('walk', 'rbike', 'transit'),
    'density_r500_norm_route':          ('rbike', 'transit', 'car'),   # walk = base
    'bike_infra_score_avg_r250_route':  ('rbike', 'transit', 'car'),   # walk = base
    'speed_limit_avg_r250_route':       ('walk', 'rbike', ),
    'vc_beta_2.0_route':                ('car', ),
    # 'traffic_flow_avg_r250_route':      ('car', ),
    'mean_abs_slope_r250_route':        ('walk', 'rbike', 'transit'),  # CAR = base
    'n_traffic_signals_route':          ('transit', ),  # CAR = base
}

# Which survey/05 column each route feature copies from. Default-mode picks:
# rwalk for elev (walk-experienced climb); car_base for density / signals /
# speed / flow (representative urban route); rbike for bike_infra + slope.
_ROUTE_FEATURE_DEFAULTS: dict[str, str] = {
    'elev_gain_route':                  'elev_gain_rwalk',
    'elev_loss_route':                  'elev_loss_rwalk',
    'density_r500_norm_route':          'density_r500_norm_car_base',
    'bike_infra_score_avg_r250_route':  'bike_infra_score_avg_r250_rbike',
    'speed_limit_avg_r250_route':       'speed_limit_avg_r250_car_base',
    'vc_beta_2.0_route':                'vc_beta_2.0_car_base',
    'mean_abs_slope_r250_route':        'mean_abs_slope_r250_rbike',
    'n_traffic_signals_route':          'n_traffic_signals_car_base',
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
    'sd_cat_income_leq4000':        ('rbike', 'transit', 'car'),
    'sd_cat_income_4001to8000':     ('rbike', 'transit', 'car'),
    'sd_cat_income_12001to16000':   ('rbike', 'transit', 'car'),
    'sd_cat_income_16001+':         ('rbike', 'transit', 'car'),
}
_SD_COLS: tuple[str, ...] = tuple(_SD_COL_MODES)


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


# Meters-scale feature scaling (single source of truth in main.utility_config).
from main.utility_config import FEATURE_SCALE as _FEATURE_SCALE

_TIME_CAP_MIN = 5 * 60
_MIN_LEG_DIST_LINE_M = 100.0
_MAX_LEG_DIST_LINE_M = 200_000

# Endpoint elevation cap (m). Drops alpine legs whose mode-choice character
# skews the fit. None = disable. Looked up on nodes_walk_extended.csv.
_MAX_ENDPOINT_ELEVATION_M: float | None = 1_000.0

# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _load_and_join(context) -> pd.DataFrame:
    """Load survey legs + routed times + overheads. Left-join on the legs
    index; missing columns become NaN and are handled downstream by the
    availability + dropna logic. Scopes legs to the scenario AOI so
    utility estimation trains on trips within the calibration region."""
    legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
    routed = context.get_generic('survey_leg_times.csv', storage=Storage.PRIVATE)
    overhead = context.get_generic('survey_leg_overheads.csv', storage=Storage.PRIVATE)
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)
    legs = filter_legs_by_aoi(
        legs, aoi_polygon, crs=scenario.crs_main, label='survey legs')
    df = legs.join(routed, how='left').join(overhead, how='left')
    logging.info(f"  → {len(df):,} legs joined (legs {len(legs):,}, "
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
        if mode == 'transit':
            net_s = df['t_routed_transit_z2z']
            if time_type == 'gross':
                # gross = t_routed + orig_ov + dest_ov − t_routed_bias
                # (α scaling of z2z; see main.common.route_time_alpha).
                ov_orig = df.get('t_overhead_orig_transit')
                ov_dest = df.get('t_overhead_dest_transit')
                bias = df.get('t_routed_bias_transit')
                total_s = (net_s
                           + ov_orig.fillna(0) + ov_dest.fillna(0)
                           - (bias.fillna(0) if bias is not None else 0.0))
            else:
                total_s = net_s
        elif mode == 'car':
            net_s = df['t_routed_car']
            if time_type == 'gross':
                total_s = (net_s
                           + car_overhead_by_peak(df, 'orig').fillna(0)
                           + car_overhead_by_peak(df, 'dest').fillna(0)
                           - car_routed_bias_by_peak(df).fillna(0))
            else:
                total_s = net_s
        else:
            profile = _MODE_TO_PROFILE[mode]
            net_s = df[f't_routed_{profile}']
            if time_type == 'gross':
                bias = df.get(f't_routed_bias_{profile}')
                total_s = (net_s
                           + df[f't_overhead_orig_{profile}'].fillna(0)
                           + df[f't_overhead_dest_{profile}'].fillna(0)
                           - (bias.fillna(0) if bias is not None else 0.0))
            else:
                total_s = net_s
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
    `_POINT_FEATURE_SOURCES`) and derive `elev_gain_net` / `elev_loss_net`
    from Δelevation; route anchors copy from one default profile column
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
                # Prefer the fully-oriented biogeme name, else fall back to the
                # un-oriented source column so one dict entry covers both sides.
                scale = _FEATURE_SCALE.get(bn, _FEATURE_SCALE.get(src_col, 1.0))
                df[bn] = df[node_id_col].map(s) / scale
            if anchors == ('od',):
                bn_od = f'{src_col}_od'
                df[bn_od] = df[f'{src_col}_orig'] + df[f'{src_col}_dest']
        # Net elev gain / loss — walk-graph elevation at both endpoints (any
        # mode's node file would work; DEM sampling is graph-agnostic).
        nodes_walk = _load('walk_extended')
        if 'elevation' in nodes_walk.columns:
            elev_lookup = nodes_walk['elevation']
            elev_orig = df['orig_node_id_walk'].map(elev_lookup)
            elev_dest = df['dest_node_id_walk'].map(elev_lookup)
            delta = elev_dest - elev_orig
            gain_scale = _FEATURE_SCALE.get('elev_gain_net', 1.0)
            loss_scale = _FEATURE_SCALE.get('elev_loss_net', 1.0)
            df['elev_gain_net'] = np.maximum(delta, 0.0) / gain_scale
            df['elev_loss_net'] = np.maximum(-delta, 0.0) / loss_scale
        else:
            logging.warning("  → `elevation` missing from nodes_walk_extended.csv; "
                            "elev_gain_net / elev_loss_net unavailable.")
    elif anchors == ('route',):
        for bn, source_col in _ROUTE_FEATURE_DEFAULTS.items():
            if source_col not in df.columns:
                logging.warning(f"  → route-feature source {source_col!r} not in "
                                f"survey_leg_times.csv; {bn!r} rows dropped.")
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
    """Boolean per leg: is the household eligible for this mode based on
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


def _utility_expression(
    mode: str, betas: dict, vars_: dict,
    features: dict[str, tuple[str, ...]],
    is_walk: bool,
    car_peak_night: str = 'asc',
) -> Any:
    """Biogeme utility expression for one mode. Location + SD terms are
    added only when the mode is in the feature's free-modes tuple.
    See `_parse_car_peak_night` for the peak/night interaction grammar.
    """
    has_asc, has_time, feat_names = _parse_car_peak_night(car_peak_night)
    exprs = []
    if not is_walk:
        exprs.append(betas[f'asc_{mode}'])
        if mode == 'car' and has_asc:
            exprs.append(betas['asc_car_peak'] * vars_['hour_peak'])
            exprs.append(betas['asc_car_night'] * vars_['hour_night'])
    exprs.append(betas[f'b_time_{mode}'] * vars_[f't_{mode}_min'])
    exprs.append(betas[f'b_time_log_{mode}'] * vars_[f't_{mode}_min_log'])
    if mode == 'car' and has_time:
        exprs.append(betas['b_time_car_peak']
                     * vars_['hour_peak'] * vars_['t_car_min'])
        exprs.append(betas['b_time_car_night']
                     * vars_['hour_night'] * vars_['t_car_min'])
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
    for sd, sd_modes in _SD_COL_MODES.items():
        if mode in sd_modes:
            exprs.append(betas[f'b_{sd}_{mode}'] * vars_[sd])
    return sum(exprs[1:], exprs[0])


def _register_betas(
    modes: list[str], features: dict[str, tuple[str, ...]],
    car_peak_night: str = 'asc',
) -> dict:
    """Instantiate all Beta objects. Walk's ASC is fixed at 0 (base mode).
    Location + SD betas registered only for modes with a free coef.
    Car peak/night adds extra betas per `_parse_car_peak_night`.
    """
    betas: dict[str, Beta] = {}
    for mode in modes:
        if mode != 'walk':
            betas[f'asc_{mode}'] = Beta(f'asc_{mode}', 0, None, None, 0)
        b_t, b_ln = _time_beta_names(mode)
        betas[b_t] = Beta(b_t, 0, None, None, 0)
        betas[b_ln] = Beta(b_ln, 0, None, None, 0)
        for bn, feat_modes in features.items():
            if mode in feat_modes:
                name = f'b_{bn}_{mode}'
                betas[name] = Beta(name, 0, None, None, 0)
        for sd, sd_modes in _SD_COL_MODES.items():
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
) -> dict:
    """Instantiate all biogeme `Variable` handles referenced by the model."""
    v: dict[str, Variable] = {}
    for mode in modes:
        v[f't_{mode}_min'] = Variable(f't_{mode}_min')
        v[f't_{mode}_min_log'] = Variable(f't_{mode}_min_log')
        v[f'availability_{mode}'] = Variable(f'availability_{mode}')
    for bn in features:
        v[bn] = Variable(bn)
    for sd in _SD_COLS:
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
) -> tuple[pd.DataFrame, float]:
    """Set up + estimate the biogeme MNL / nested-logit model. Returns the
    coefficient table and rho² (bar, vs null)."""
    modes = ['walk', 'rbike', 'transit', 'car']
    if variant.ebikes != 'exclude':
        modes.append('ebike25')

    # biogeme wants a numeric `choice` column matching the alternative IDs.
    df = df.copy()
    df['choice'] = df['_chosen_mode'].map(_MODE_IDS).astype(int)

    v = _biogeme_variables(modes, features)
    betas = _register_betas(modes, features, variant.car_peak_night)

    utilities = {
        _MODE_IDS[mode]: _utility_expression(
            mode, betas, v, features, is_walk=(mode == 'walk'),
            car_peak_night=variant.car_peak_night)
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
    biogeme_cols += [c for c in _SD_COLS if c in df.columns]
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
        bg.modelName = f'utility_{variant_name(variant)}'
        bg.calculate_null_loglikelihood(availability)
        results = bg.estimate()
    finally:
        os.chdir(cwd)

    params = results.get_estimated_parameters()
    rho2 = float(results.data.rhoBarSquareNull)
    # Rename biogeme's default columns to snake_case so the on-disk CSV
    # follows this codebase's convention.
    params = params.rename(columns={
        'Value':        'value',
        'Rob. Std err': 'rob_std_err',
        'Rob. t-test':  'rob_t_test',
        'Rob. p-value': 'rob_p_value',
    })
    return params, rho2




def variant_name(variant) -> str:
    """Registered variant name; matches the `--variant` CLI selector."""
    return variant.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(variant):
    context = init_context()
    v = variant_name(variant)
    coefs.resolve(
        context,
        name=f'utility_{v}',
        calibrate_fn=lambda: _fit_utility(context, variant),
    )
    # Companion sidecar — SD averages + t_cut_min. Mirrors the parent's
    # source declaration (validated in Scenario.__post_init__); the
    # calibrate_fn only fires on `Calibrate()`.
    coefs.resolve(
        context,
        name=f'utility_{v}_stats',
        calibrate_fn=lambda: _compute_utility_stats(context),
    )
    context.close()


def _compute_utility_stats(context) -> pd.DataFrame:
    """SD averages + t_cut_min per mode, as a two-column `key, value`
    DataFrame. Written to the `utility_<variant>_stats` companion CSV
    (see `main` above). Only called on `Calibrate()` scenarios;
    `ImportFrom` copies the file from the source scenario."""
    from main.common import compute_sd_averages, load_util_cutoffs_from_summary
    raw_legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
    # Scope SD averages to the AOI so utility-stats align with the
    # region trained on in `_fit_utility` / `_load_and_join`.
    scenario = get_scenario(context.scenario)
    aoi_polygon = load_aoi_polygon(context)
    raw_legs = filter_legs_by_aoi(
        raw_legs, aoi_polygon, crs=scenario.crs_main, label='SD-stats legs')
    sd_averages = compute_sd_averages(raw_legs)
    try:
        survey_summary = context.get_generic('survey_summary.csv', storage=Storage.PRIVATE)
        t_cut_min_per_mode = load_util_cutoffs_from_summary(survey_summary)
    except FileNotFoundError:
        t_cut_min_per_mode = {}
        logging.warning(
            "  ⚠ survey_summary.csv missing — t_cut_min NOT baked into "
            "stats. 09b's log-time extrapolation will fall back to pure "
            "log. Re-run survey/08b to enable.")
    stats = pd.DataFrame(
        {'value': list(sd_averages.values()) + list(t_cut_min_per_mode.values())},
        index=(
            [f'sd_avg_{col}' for col in sd_averages]
            + [f't_cut_min_{mode}' for mode in t_cut_min_per_mode]
        ),
    )
    stats.index.name = 'key'
    logging.info(f"  → utility stats: {len(sd_averages)} sd_avg + "
                 f"{len(t_cut_min_per_mode)} t_cut_min rows")
    return stats


def _fit_utility(context, variant) -> pd.DataFrame:
    """Fit the biogeme MDCL model and return the coef DataFrame for
    `coefs.resolve` to persist. Only called on `Calibrate()` scenarios;
    `ImportFrom`/`HandWritten` short-circuit before reaching this."""
    logging.info(f"variant: time={variant.time}, ownership={variant.ownership}, "
                 f"ebikes={variant.ebikes}, single_leg_only={variant.single_leg_only}, "
                 f"subset_fraction={variant.subset_fraction}, "
                 f"location_features={variant.location_features}, "
                 f"car_peak_night={variant.car_peak_night}")

    anchors = _node_feature_anchors(variant.location_features)
    features = _features_for_anchors(anchors)

    with step('load survey legs + routed times + overheads'):
        df = _load_and_join(context)

    with step(f'compute per-mode times ({variant.time}, minutes)'):
        df = _compute_mode_times(df, variant.time, include_ebikes=(variant.ebikes != 'exclude'))

    with step(f'join per-node point features (anchors={anchors})'):
        df = _join_point_node_features(df, context, anchors)

    modes_for_avail = ['walk', 'rbike', 'transit', 'car']
    if variant.ebikes != 'exclude':
        modes_for_avail.append('ebike25')

    with step(f'compute availability flags ({variant.ownership})'):
        df = _compute_availability(df, variant.ownership, modes_for_avail)

    if _MAX_ENDPOINT_ELEVATION_M is not None:
        with step(f'elevation filter (≤ {_MAX_ENDPOINT_ELEVATION_M:.0f} m at orig + dest)'):
            walk_nodes = context.get_properties('nodes', 'walk_extended')
            if 'elevation' not in walk_nodes.columns:
                raise KeyError("`_MAX_ENDPOINT_ELEVATION_M` set but `elevation` "
                               "missing from nodes_walk_extended.csv.")
            elev = walk_nodes['elevation']
            elev_orig = df['orig_node_id_walk'].map(elev)
            elev_dest = df['dest_node_id_walk'].map(elev)
            mask = (
                (elev_orig <= _MAX_ENDPOINT_ELEVATION_M)
                & (elev_dest <= _MAX_ENDPOINT_ELEVATION_M)
            )
            n_before = len(df)
            df = df[mask.fillna(False)]
            logging.info(f"  → {len(df):,}/{n_before:,} kept ({100*len(df)/max(n_before,1):.1f} %)")

    with step('speed + detour envelope filter (from survey standardization)'):
        # `is_within_speed_envelope` / `is_within_detour_envelope` are
        # per-mode plausibility gates from the survey preparation. Same
        # subset that 04's edge-weight calibration and 07's overhead
        # calibration train on — so the utility model fits on
        # data-quality-consistent legs.
        n_before = len(df)
        for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
            if flag in df.columns:
                df = df[df[flag] == 1]
        logging.info(f"  → {len(df):,}/{n_before:,} kept "
                     f"({100*len(df)/max(n_before,1):.1f} %)")

    with step('filter to modeled sample'):
        n0 = len(df)
        df = df[
            (df['dist_line'] >= _MIN_LEG_DIST_LINE_M)
            & (df['dist_line'] <= _MAX_LEG_DIST_LINE_M)
        ]
        df = df[df['_chosen_mode'].isin(list(_MODE_IDS))]
        active_modes = set(modes_for_avail)
        df = df[df['_chosen_mode'].isin(active_modes)]
        df = df[df['availability_mismatch'] == 0]
        df = df[df['n_available_modes'] >= 2]
        # Transit is exempt from single_leg_only — walk-to-stop + vehicle +
        # walk-from-stop makes n_legs > 1 by construction.
        if variant.single_leg_only:
            if 'n_legs_in_trip' not in df.columns:
                raise KeyError("single_leg_only=True requires n_legs_in_trip in "
                               "survey_legs.csv.")
            n_before = len(df)
            df = df[
                (df['n_legs_in_trip'] == 1)
                | (df['mode_simplified'] == 'transit')
            ]
            logging.info(f"  → single_leg_only filter: {len(df):,}/{n_before:,} kept")
        # Drop rows with NaN in any biogeme-evaluated feature column.
        need = (
            list(features)
            + ['hour_peak', 'hour_night', 'weight_person']
            + list(_SD_COLS)
        )
        need = [c for c in need if c in df.columns]
        df = df.dropna(subset=need)
        logging.info(f"  → n = {len(df):,} / {n0:,}  after filters")
        df = df.copy()
        # NaN per-mode times → above-cap placeholder; availability_<mode>=0
        # already excludes them from that leg's choice set.
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

    with step('fit biogeme model'):
        scratch_root = Path(context.path_for(Storage.SCRATCH, 'biogeme'))
        params, rho2 = _fit_and_extract(
            df, variant, features, run_dir=scratch_root / variant_name(variant))
        logging.info(f"  → rho² (bar, vs null) = {rho2:.4f}")
        with pd.option_context('display.max_rows', None):
            logging.info('\n' + params[['value', 'rob_p_value']].round(4).to_string())

    return params[['value', 'rob_std_err', 'rob_t_test', 'rob_p_value']].copy()


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
variants = Variants([
    ('time', str),
    ('ownership', str),
    ('ebikes', str),
    ('single_leg_only', bool),
    ('subset_fraction', float),
    ('location_features', str),
    ('car_peak_night', str),
])
variants.add(
    name='default',
    time='gross',
    ownership='endogenous',
    ebikes='exclude',
    single_leg_only=False,
    subset_fraction=1.0,
    location_features='endpoints_combined',
    # car_peak_night='asc,traffic_flow_avg_r250_od',
    car_peak_night='asc',
)
variants.add(
    name='default_single',
    time='gross',
    ownership='endogenous',
    ebikes='exclude',
    single_leg_only=True,
    subset_fraction=1.0,
    location_features='endpoints_combined',
    # car_peak_night='asc,traffic_flow_avg_r250_od',
    car_peak_night='asc',
)
variants.add(
    name='default_route',
    time='gross',
    ownership='endogenous',
    ebikes='exclude',
    single_leg_only=False,
    subset_fraction=1.0,
    location_features='route',
    # car_peak_night='asc,vc_beta_2.0_route',
    car_peak_night='asc',
)

if __name__ == '__main__':
    variants.run(main, default='default')
