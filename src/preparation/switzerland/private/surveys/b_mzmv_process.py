"""
Process the preprocessed MZMV survey (legs, trips, daytrips, routes) into final
per-year tables with mode classifications, peak/night flags, GHG accounting,
externalities, and zone metadata. Also writes per-mode / per-purpose GHG summary
CSVs.

> **Private data.** Consumes preprocessed MZMV outputs (Switzerland's
> federal mobility survey, BFS-restricted); not redistributable.
> Published as documentation of the method.

Inputs (under <DATA_DIR_PRIVATE>/preparation/.../mzmv_<year>/preprocessed/):
    trips.csv, legs.csv, people.csv, households.csv         # from a_mzmv_preprocess
Inputs (under <DATA_DIR_PRIVATE>/raw/switzerland/mzmv/):
    MZMV<year>_mit_Geo/5_Routen(Geometriefiles)/CH_Routen/Routen_CH.shp
Inputs (cross-namespace; not yet migrated — TODO):
    preparation/switzerland/zones :: properties+shapes for zones, shapes for lfi_regions.
    -> TODO: finalize. Moved zone-based information away from here so that this script doesn't need zones.

Outputs (PRIVATE):
    mzmv_<year>/trips.csv
    mzmv_<year>/legs.csv
    mzmv_<year>/daytrips.csv
    mzmv_<year>/households.csv
    mzmv_<year>/legs.routes.measured.npy
    mzmv_<year>/ghg_stats/<frame>_<mode>_<age>_<group>_<year>.csv

Run all years sequentially (default):
    python -m preparation.switzerland.private.surveys.b_mzmv_process
Single year:
    python -m preparation.switzerland.private.surveys.b_mzmv_process --variant 2021
"""

import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

# Survey-prep DataFrames accumulate ~200 columns via sequential
# `df['col'] = ...` assignments. Pandas emits a `PerformanceWarning`
# for each write past a fragmentation threshold — hundreds of lines
# of noise. This is a research prep script that runs a few times a
# year on ~300k rows; the "slow" writes are milliseconds. Silence.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from aperta import data_processing, geo_processing
from aperta_atlas import utils
from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage
from aperta_atlas.variant import Variants

from preparation.switzerland.common import CRS_CH, CRS_LATLON
from preparation.switzerland.private.surveys import common, standardize
from preparation.switzerland.private.surveys.common import (
    F51300_TO_STR, F51700W_TO_STR, F52900A_TO_STR,
    MODE_TO_SIMPLIFIED, WMITTEL2_TO_STR, FRACTION_COVERED,
)


MZMV_YEARS = (2015, 2021)

variants = Variants([('mzmv_year', int)])
for _y in MZMV_YEARS:
    variants.add(name=str(_y), mzmv_year=_y)


# Bin boundaries used in load_people / load_households. Defined at module scope so
# they're listable for downstream code.
BINS_AGE = (0, 6, 13, 18, 25, 35, 50, 65, 75, 120)
BINS_N_VEHICLES = (-1, 0, 1, 2, 99)
BINS_N_EBIKES = (-1, 0, 1, 99)


# MZMV education-level remap: f40120 → string.
_EDUCATION_VALUES: dict[int, str] = {
    -99: 'na', -98: 'na', -97: 'na',
    1: 'none',     # Keine
    2: 'basic',    # Obligatorische Schule
    3: 'general',  # Allgemeinbildung ohne Maturität
    4: 'general',  # Berufliche Grundbildung / Berufslehre
    5: 'higher',   # Maturität / Lehrkräfte-Seminar
    6: 'higher',   # Höhere Berufsbildung
    7: 'higher',   # Höhere Fachschule
    8: 'uni',      # FH / Uni
    9: 'uni',      # Doktorat (only in 2021)
}

_WEATHER_VALUES: dict[int, str] = {
    -99: 'na', -98: 'na', -97: 'na',
    1: 'sunny', 2: 'cloudy', 3: 'cloudy', 4: 'foggy',
    5: 'rainy', 6: 'snowy', 7: 'unstable', 8: 'hot', 9: 'cold',
}

_WEATHER_VALUES_SIMPLE: dict[int, str] = {
    -99: 'na', -98: 'na', -97: 'na',
    1: 'good', 2: 'good', 3: 'medium', 4: 'medium',
    5: 'bad', 6: 'bad', 7: 'medium', 8: 'good', 9: 'bad',
}

_INCOME_VALUES: dict[int, str] = {
    -99: 'na', -98: 'na', -97: 'na',
    1: 'leq4000', 2: 'leq4000',
    3: '4001to8000', 4: '4001to8000',
    5: '8001to12000', 6: '8001to12000',
    7: '12001to16000', 8: '12001to16000',
    9: '16001+',
}

_HHTYPE_VALUES: dict[int, str] = {
    -98: 'na', -97: 'na',
    10: 'single_nokids',     # Einpersonenhaushalt
    30: 'other',             # Nichtfamilienhaushalte
    210: 'coupe_nokids',     # Paare ohne Kinder
    220: 'couple_kids',      # Paare mit Kindern
    230: 'single_kids',      # Einelternhaushalte mit Kindern
}

_EMISSION_CASES: dict[str, tuple[str, str, str | None]] = {
    'operating_icev':     ('operating', 'icev_mix', None),
    'operating_ev_ch':    ('operating', 'bev_mix', 'ch'),
    'operating_ev_global':('operating', 'bev_mix', 'global'),
    'total_icev':         ('total',     'icev_mix', None),
    'total_ev_ch':        ('total',     'bev_mix', 'ch'),
    'total_ev_global':    ('total',     'bev_mix', 'global'),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_leg_id(year: int, hhnr: int | str, etnr: int | str) -> str:
    """Stable per-leg ID across years."""
    return f"{year}_{hhnr}_{etnr}"


def _vectorized_leg_ids(year: int, hhnr: pd.Series, etnr: pd.Series) -> pd.Series:
    """Vectorized counterpart to `get_leg_id` for a whole DataFrame."""
    return f"{year}_" + hhnr.astype(str) + '_' + etnr.astype(str)


def _car_ratio_vectorized(n_adults: pd.Series, n_cars: pd.Series) -> np.ndarray:
    """Vectorized car-to-adult ratio classification: '1+' if cars >= adults,
    '<1' if cars > 0, otherwise 'none'.
    """
    return np.select(
        [n_cars >= n_adults, n_cars > 0],
        ['1+', '<1'],
        default='none',
    )


# ---------------------------------------------------------------------------
# People / households
# ---------------------------------------------------------------------------

def load_people(context: Context, mzmv_year: int) -> pd.DataFrame:
    people = context.get_generic(
        f'mzmv_{mzmv_year}/preprocessed/people.csv',
    ).set_index('HHNR')

    people['sd_bool_weekday'] = people['tag'].isin([1, 2, 3, 4, 5]).astype(int)
    people['sd_age'] = people['alter']
    people['sd_ordinal_age'] = pd.cut(
        people['alter'], BINS_AGE, right=True, labels=common.bin_labels(BINS_AGE),
    )
    people['sd_nominal_sex'] = people['gesl'].replace({1: 'm', 2: 'w'})
    people['sd_bool_license_car']           = (people['f20400a'] == 1).astype(int)
    people['sd_bool_license_motorbike']     = (people['f20400b'] == 1).astype(int)
    people['sd_bool_transitpass_ga']        = (people['f41600_01a'] == 1).astype(int)
    people['sd_bool_transitpass_verbund']   = (people['f41600_01c'] == 1).astype(int)
    people['sd_bool_transitpass_halbtax']   = (people['f41600_01b'] == 1).astype(int)
    people['sd_bool_transitpass_anderes']   = (
        people[['f41600_01d', 'f41600_01e', 'f41600_01f', 'f41600_01g']].sum(axis=1) >= 1
    ).astype(int)
    people['sd_nominal_education'] = people['f40120'].replace(_EDUCATION_VALUES)
    people['sd_nominal_weather']   = people['f50100a'].replace(_WEATHER_VALUES)
    weather_simple = people['f50100a'].replace(_WEATHER_VALUES_SIMPLE)
    people['sd_bool_weather_good']   = (weather_simple == 'good').astype(int)
    people['sd_bool_weather_medium'] = (weather_simple == 'medium').astype(int)
    people['sd_bool_weather_bad']    = (weather_simple == 'bad').astype(int)
    date_format = {2015: '%m/%d/%Y', 2021: '%d.%m.%Y'}[mzmv_year]
    people['datetime_day'] = pd.to_datetime(people['VSTag'], format=date_format)
    return people


def load_households(context: Context, mzmv_year: int) -> pd.DataFrame:

    households = context.get_generic(
        f'mzmv_{mzmv_year}/preprocessed/households.csv',
    ).set_index('HHNR')

    households['sd_ordinal_n_cars']        = pd.cut(
        households['f30100'], BINS_N_VEHICLES, right=True,
        labels=common.bin_labels(BINS_N_VEHICLES),
    )
    households['sd_ordinal_n_car_parking'] = pd.cut(
        households['f31100'], BINS_N_VEHICLES, right=True,
        labels=common.bin_labels(BINS_N_VEHICLES),
    )
    households['sd_ordinal_n_bikes']       = pd.cut(
        households['f32200a'], BINS_N_VEHICLES, right=True,
        labels=common.bin_labels(BINS_N_VEHICLES),
    )
    households['sd_ordinal_n_ebikes']      = pd.cut(
        households['f32200b'], BINS_N_EBIKES, right=True,
        labels=common.bin_labels(BINS_N_EBIKES),
    )
    households['sd_ordinal_n_pedelecs']    = pd.cut(
        households['f32200c'], BINS_N_EBIKES, right=True,
        labels=common.bin_labels(BINS_N_EBIKES),
    )

    # Household composition (vectorized — was a per-row apply previously).
    hh_size_valid = (households['hhgr'] > 0) & (households['hhgr'] < 12)
    households['hh_n_people'] = np.where(
        hh_size_valid, households['hhgr'].astype(int), -999_999,
    )
    households['hh_n_adults'] = np.select(
        [households['hhtyp'] == 230, households['hhtyp'] == 220],
        [1, 2],
        default=households['hhgr'],
    )
    cars_valid = (households['f30100'] > 0) & (households['f30100'] < 10)
    households['hh_n_cars'] = np.where(cars_valid, households['f30100'].astype(int), 0)
    households['sd_ordinal_car_ratio'] = _car_ratio_vectorized(
        households['hh_n_adults'], households['f30100'],
    )

    households['sd_nominal_hhtype']  = households['hhtyp'].replace(_HHTYPE_VALUES)
    households['sd_ordinal_income']  = households['f20601'].replace(_INCOME_VALUES)
    households['sd_bool_second_home'] = (households['f10700'] == 1).astype(int)

    people = load_people(context, mzmv_year)
    sd_cols_people = [c for c in people.columns
                      if c.startswith('sd_') or c.startswith('datetime_')]
    households = households.join(people[sd_cols_people])

    # One-hot encode all sd_nominal_*/sd_ordinal_* columns. (Drop-first is left
    # disabled — kept consistent with the legacy behaviour.)
    sd_cat_cols = [c for c in households.columns
                   if c.startswith(('sd_nominal', 'sd_ordinal'))]
    for col in sd_cat_cols:
        prefix = col.replace('_nominal', '_cat').replace('_ordinal', '_cat')
        dummies = pd.get_dummies(
            households[col], prefix=prefix, dummy_na=False, drop_first=False,
        ).astype(int)
        households = households.join(dummies)

    # PhD column is only captured by MZMV 2021; insert zero column for 2015 so
    # concatenated multi-year tables have aligned columns.
    if 'sd_cat_education_phd' not in households.columns:
        households['sd_cat_education_phd'] = 0

    households = households.rename(columns={'W_X_LV95': 'hh_x', 'W_Y_LV95': 'hh_y'})
    # households = surveys.add_zone_columns_for_location(
    #     households, zones, CRS_CH, 'hh',
    #     extra_zones=lfi_regions, extra_zone_cols=['lfi_region_id'],
    # )
    # households = common.add_swiss_location_subsets(households, 'hh')
    households['hh_id'] = households.index
    return households


# ---------------------------------------------------------------------------
# GHG and externalities
# ---------------------------------------------------------------------------

def save_ghg_stats(context: Context, df: pd.DataFrame, mzmv_year: int,
                   n_households: int) -> None:
    df = df.copy()
    purpose_replace = {
        'leisure_other': 'leisure_other_na',
        'leisure_unknown': 'leisure_other_na',
        'other': 'other_na',
        'na': 'other_na',
        'accompany': 'other_na',
    }
    df['purpose_detailed'] = df['purpose_detailed'].replace(purpose_replace).fillna('other_na')
    df['all'] = 1

    n_people = n_households / FRACTION_COVERED
    landbased_modes = {'car', 'coach', 'rbike', 'ebike25', 'ebike45', 'micro',
                       'motorbike', 'transit', 'truck', 'walk'}
    group_field_to_short = {
        'mode_simplified': 'mode', 'purpose_detailed': 'purpose',
        'sd_ordinal_age': 'age', 'all': 'all',
    }
    base_index_name = df.index.name.replace('_id', '')

    for group_field in ('all', 'mode_simplified', 'purpose_detailed', 'sd_ordinal_age'):
        for mode in ('all', 'landbased'):
            mode_filter = (df['mode_simplified'].isin(landbased_modes) if mode == 'landbased'
                           else pd.Series(True, index=df.index))
            for age in ('7+', '19+'):
                f = mode_filter & ((df['sd_age'] >= 19) if age == '19+' else True)
                tmp = df[f].groupby(group_field, dropna=False).agg({
                    'dist_line':    lambda x: x[x >= 0].sum(),
                    'dist_measured': lambda x: x[x >= 0].sum(),
                    'time_measured': lambda x: x[x >= 0].sum(),
                    'car_occupancy': utils.get_weighted_agg_function_bounded(df, 'dist_measured', 10, 1),
                    'kgco2eq_total_mix': lambda x: x[x >= 0].sum(),
                    'gco2eq_per_km_total_mix': utils.get_weighted_agg_function_bounded(df, 'dist_measured', 9_999, 1),
                    'externalities_chf': lambda x: x[x >= -100].sum(),
                    'hh_id': lambda x: len(set(x)),
                })
                gco2eq = tmp['gco2eq_per_km_total_mix']
                tmp = tmp.drop(columns='gco2eq_per_km_total_mix')
                tmp['chance_of_travel']               = tmp['hh_id'] / n_households * 100
                tmp['km_per_day_if_traveled']         = tmp['dist_measured'] / tmp['hh_id'] / 1e3
                tmp['gco2eq_per_km_total_mix']        = gco2eq
                tmp['emissions_share']                = tmp['kgco2eq_total_mix'] / tmp['kgco2eq_total_mix'].sum() * 100
                tmp['line_km_per_day_if_traveled']    = tmp['dist_line'] / tmp['hh_id'] / 1e3
                tmp['min_per_day_if_traveled']        = tmp['time_measured'] / tmp['hh_id'] / 60
                tmp['km_per_day_contribution']        = tmp['dist_measured'] / n_people * 100
                tmp['dist_line_share']                = tmp['dist_line'] / tmp['dist_line'].sum() * 100
                tmp['dist_measured_share']            = tmp['dist_measured'] / tmp['dist_measured'].sum() * 100
                tmp['time_reported_share']            = tmp['time_measured'] / tmp['time_measured'].sum() * 100
                tmp['tco2eq_per_person_year_if_traveled']    = tmp['kgco2eq_total_mix'] / tmp['hh_id'] * 365 / 1e3
                tmp['tco2eq_per_person_year_contribution']   = tmp['kgco2eq_total_mix'] / n_people * 365 / 1e3
                tmp['mtco2eq_per_year']               = tmp['tco2eq_per_person_year_contribution'] * 9e6 / 1e6
                tmp['chf_ext_per_person_year_if_traveled']  = tmp['externalities_chf'] / tmp['hh_id'] * 365
                tmp['chf_ext_per_person_year_contribution'] = tmp['externalities_chf'] / n_people * 365
                tmp['mchf_ext_per_year']              = tmp['chf_ext_per_person_year_contribution'] * 9e6 / 1e6
                tmp = tmp.drop(columns=['dist_line', 'dist_measured', 'time_measured'])
                tmp = tmp.loc[~np.isclose(tmp.sum(axis=1), 0)]
                short_name = group_field_to_short[group_field]
                file_name = (f"mzmv_{mzmv_year}/ghg_stats/"
                             f"{base_index_name}_{mode}_{age}_{short_name}_{mzmv_year}.csv")
                context.create_generic(
                    tmp, file_name,
                    verify_unique_index=False, kws={'float_format': '%.2f'},
                )


def get_effective_occupancy(df: pd.DataFrame) -> pd.Series:
    """Adjust car occupancy downward for non-work trips, assuming 10% of additional
    passengers (beyond the first) are children under 7 (and so already covered by
    survey-exclusion of that age band).
    """
    res = df['car_occupancy'].astype(float)
    has_occupancy = df['car_occupancy'].notna()
    before = np.average(res[has_occupancy], weights=df.loc[has_occupancy, 'dist_measured'])
    non_work_with_passengers = (df['car_occupancy'] > 1) & ~df['purpose_detailed'].str.startswith('work')
    res.loc[non_work_with_passengers] = 1 + (res.loc[non_work_with_passengers] - 1) * 0.9
    after = np.average(res[has_occupancy], weights=df.loc[has_occupancy, 'dist_measured'])
    logging.info(
        f"Average car occupancy adjusted {before:.2f} -> {after:.2f} after removing "
        f"estimated children under 7.")
    return res.fillna(1)


def _compute_purpose_columns(df: pd.DataFrame) -> None:
    """In-place: derive `purpose`, `purpose_detailed`, and `purpose_<group>_*` columns
    from MZMV's `f52900` / `wzweck1` / `f51700*` columns.
    """
    if 'f52900' in df.columns:
        df['purpose'] = df['f52900'].replace(F52900A_TO_STR)
        df['purpose_return'] = df['f52950'].replace(F52900A_TO_STR)
        leisure_col = 'f51700'
    else:
        df['purpose'] = df['wzweck1'].replace(F52900A_TO_STR)
        leisure_col = 'f51700_weg'
    df['purpose_detailed'] = df['purpose']
    purpose_leisure = df[leisure_col].replace(F51700W_TO_STR)
    leisure_mask = purpose_leisure.notna()
    df.loc[leisure_mask, 'purpose_detailed'] = purpose_leisure[leisure_mask]
    df.loc[df['purpose_detailed'] == 'leisure', 'purpose_detailed'] = 'leisure_unknown'
    df.loc[(df['purpose'] == 'education') & (df['sd_age'] < 19), 'purpose_detailed'] = 'education_school'
    df.loc[(df['purpose'] == 'education') & (df['sd_age'] >= 19), 'purpose_detailed'] = 'education_adult'

    for purpose in ('work_commute', 'work_business', 'education_school', 'education_adult',
                    'errands_groceries', 'errands_services', 'leisure_gastronomy',
                    'leisure_amenity', 'leisure_active'):
        df[f'purpose_{purpose}'] = (df['purpose_detailed'] == purpose).astype(int)
    for purpose_group in ('work', 'errands', 'leisure', 'education'):
        prefix = f'purpose_{purpose_group}_'
        df[f'{prefix}any'] = df[[c for c in df.columns if c.startswith(prefix)]].max(axis=1)


def add_indicators(df: pd.DataFrame, emissions_data: pd.DataFrame | None = None) -> pd.DataFrame:
    _compute_purpose_columns(df)

    if emissions_data is None:
        # Compute emissions per row. Cars use a speed-dependent formula; non-cars
        # are speed-independent (lookup). The pandas `apply` here is the same shape
        # as the legacy code and remains the bottleneck — see TODO below.
        # TODO(perf): vectorize emissions by groupby('mode') + per-mode formula.
        no_speed = (df['dist_measured'] <= 0) | (df['time_measured'] <= 0)
        for case_name, (case_which, case_veh, case_region) in _EMISSION_CASES.items():
            df[f'gco2eq_per_km_{case_name}'] = df.apply(
                lambda r: common.get_gco2eq_per_km(
                    r['mode'], r['speed_kmh'], case_which, case_veh, case_region,
                ), axis=1,
            )
            car_mask = (df['mode_simplified'] == 'car') & (df['dist_measured'] > 1) & (df['time_measured'] > 1)
            w = df.loc[car_mask, 'weight_person'] * df.loc[car_mask, 'dist_measured']
            v = df.loc[car_mask, f'gco2eq_per_km_{case_name}']
            logging.info(f"   {case_name}: avg car emissions {np.average(v, weights=w):.1f} gCO2eq/km")
            occ = get_effective_occupancy(df)
            df[f'gco2eq_per_km_{case_name}'] = df[f'gco2eq_per_km_{case_name}'] / occ
            df.loc[no_speed, f'gco2eq_per_km_{case_name}'] = np.nan
            df[f'kgco2eq_{case_name}'] = df[f'gco2eq_per_km_{case_name}'] * df['dist_measured'] / 1_000_000
            v = df.loc[car_mask, f'gco2eq_per_km_{case_name}']
            logging.info(f"      ...after occupancy: {np.average(v, weights=w):.1f} gCO2eq/km")
        df['gco2eq_per_km_total_mix'] = 0.95 * df['gco2eq_per_km_total_icev'] + 0.05 * df['gco2eq_per_km_total_ev_ch']
        df['kgco2eq_total_mix']      = 0.95 * df['kgco2eq_total_icev']      + 0.05 * df['kgco2eq_total_ev_ch']
        df['ext_rp_per_km'] = df.apply(
            lambda r: common.get_ext_rp_per_km(r['mode'], r['speed_kmh']), axis=1,
        )
        df.loc[no_speed, 'ext_rp_per_km'] = np.nan
        df['externalities_chf'] = df['ext_rp_per_km'] * df['dist_measured'] / 100_000
    else:
        agg_fns: dict[str, object] = {f'kgco2eq_{c}': 'sum' for c in _EMISSION_CASES}
        agg_fns['gco2eq_per_km_total_mix'] = utils.get_weighted_agg_function(emissions_data, 'dist_measured')
        agg_fns['kgco2eq_total_mix']      = 'sum'
        agg_fns['externalities_chf']      = 'sum'
        agg_fns['car_occupancy']          = utils.get_weighted_agg_function_bounded(emissions_data, 'dist_measured', 10, 1)
        agg_fns['ext_rp_per_km']          = utils.get_weighted_agg_function(emissions_data, 'dist_measured')
        df = df.join(emissions_data.groupby('trip_id').agg(agg_fns), on='trip_id')

    df['metric_is_mode_walk']    = (df['mode'] == 'walk').astype(int)
    df['metric_is_mode_ebike']   = df['mode'].isin(['ebike25', 'ebike45']).astype(int)
    df['metric_is_mode_rbike']   = (df['mode'] == 'rbike').astype(int)
    df['metric_is_mode_bike']    = (df['mode_simplified'] == 'bike').astype(int)
    df['metric_is_mode_active']  = df['mode'].isin(['walk', 'rbike', 'ebike25', 'ebike45', 'micro']).astype(int)
    df['metric_is_mode_transit'] = (df['mode_simplified'] == 'transit').astype(int)
    df['metric_is_mode_car']     = (df['mode'] == 'car').astype(int)
    df['is_mode_access_model']   = df[[c for c in df.columns if c.startswith('metric_is_mode_')]].max(axis=1)
    df['metric_is_purpose_leisure'] = (df['purpose'] == 'leisure').astype(int)
    if 'wzweck2' in df.columns:
        df['is_outward'] = (df['wzweck2'] == 1).astype(int)
        df['is_return'] = (df['wzweck2'] == 2).astype(int)

    standardize.attach_hour_flags(df)
    return df


# ---------------------------------------------------------------------------
# Trips, daytrips, legs
# ---------------------------------------------------------------------------

_HH_JOIN_COLS = (
    'hh_x', 'hh_y', 'hh_id',
    # 'hh_municipality_id', 
    # 'hh_is_in_metro', 'hh_is_in_100k_metro', 'hh_is_in_plateau', 'hh_is_in_extended_plateau',
    'hh_n_adults', 'hh_n_people', 'hh_n_cars',
)


def _set_plane_mode_detail(df: pd.DataFrame) -> None:
    """In-place: refine generic 'plane' to plane_short/medium/long based on distance."""
    is_plane = df['mode'] == 'plane'
    if is_plane.any():
        df.loc[is_plane, 'mode'] = common.vectorized_plane_mode(
            df.loc[is_plane, 'dist_measured'].to_numpy(),
        )


def process_trips_of_year(context: Context, mzmv_year: int, legs: pd.DataFrame) -> pd.DataFrame:
    trips = context.get_generic(
        f'mzmv_{mzmv_year}/preprocessed/trips.csv',
    )
    trips['time_measured'] = (trips['dauer2'] * 60).round().astype(int)
    trips['dist_measured'] = (trips['w_rdist'] * 1000).round().astype(int)
    trips['mode'] = trips['wmittel2'].replace(WMITTEL2_TO_STR)
    trips['mode_simplified'] = trips['mode'].replace(MODE_TO_SIMPLIFIED)
    _set_plane_mode_detail(trips)
    trips['is_land_based'] = (~trips['mode'].isin(['plane', 'boat'])).astype(int)
    trips['trip_id'] = _vectorized_leg_ids(mzmv_year, trips['HHNR'], trips['WEGNR'])

    trips = trips[(trips['w_rdist'] > 0) & (trips['dauer1'] > 0)].copy()

    trips['speed_kmh'] = (trips['w_rdist'] / (trips['dauer2'] / 60)).round(2)
    trips['hour_of_day'] = np.floor(trips['f51100'] / 60).fillna(-1).astype(int)
    trips.loc[trips['hour_of_day'] >= 24, 'hour_of_day'] = -1

    households = load_households(context, mzmv_year)
    sd_cols_hh = [c for c in households.columns if c.startswith('sd_')]
    trips = trips.join(households[sd_cols_hh + list(_HH_JOIN_COLS)], on='HHNR')
    trips = add_indicators(trips, legs)
    # Drop respondents whose age band is only partially covered.
    trips = trips[trips['sd_ordinal_age'] != 'leq6']

    longest_leg = legs.groupby('trip_id')['dist_measured'].idxmax().rename('longest_leg_id')
    trips = trips.join(longest_leg, on='trip_id')

    trips['n_legs'] = trips['w_etappen']
    trips['weight_person'] = trips['WP']

    for col in [c for c in trips.columns if c.startswith('sd_')]:
        try:
            mean_unweighted = np.average(trips[col])
            mean_weighted = np.average(trips[col], weights=trips['weight_person'])
            logging.info(f"   {col}: {mean_unweighted:.2f} (w: {mean_weighted:.2f})")
        except TypeError:
            pass

    trips = trips.rename(columns={'S_X_LV95': 'orig_x', 'Z_X_LV95': 'dest_x',
                                  'S_Y_LV95': 'orig_y', 'Z_Y_LV95': 'dest_y'})
    trips = common.add_lat_lon(trips, 'orig', CRS_CH, CRS_LATLON)
    trips = common.add_lat_lon(trips, 'dest', CRS_CH, CRS_LATLON)
    # trips = surveys.add_zone_columns_for_endpoints(trips, zones, crs_main)
    # trips = common.add_swiss_city_subsets_trips(trips)
    trips = data_processing.add_straight_line_dist(trips)
    trips = trips.set_index('trip_id')

    # Many trips with purpose='leisure' are flagged as 'return' in legs and end up
    # 'leisure_unknown' in trips. Re-impute from the immediately preceding trip per
    # household so the original leisure detail is preserved when obvious.
    def impute_return_purpose(g: pd.DataFrame) -> pd.Series:
        result = g['purpose_detailed'].copy()
        for i in range(1, len(result)):
            if result.iloc[i] == 'leisure_unknown' and str(result.iloc[i - 1]).startswith('leisure_'):
                result.iloc[i] = result.iloc[i - 1]
        return result

    imputed = (trips[['hh_id', 'dist_measured', 'purpose_detailed', 'f51100']]
               .groupby('hh_id').apply(impute_return_purpose, include_groups=False)
               .droplevel(0))
    leisure_unknown = trips['purpose_detailed'] == 'leisure_unknown'
    trips.loc[leisure_unknown, 'purpose_detailed'] = imputed.loc[leisure_unknown]

    save_ghg_stats(context, trips, mzmv_year, n_households=households.index.nunique())

    keep_prefixes = common.ALLOWED_SURVEY_PREFIXES
    cols = [c for c in trips.columns if c.startswith(keep_prefixes)]
    context.create_generic(trips[cols], f'mzmv_{mzmv_year}/trips.csv')
    return trips


def process_daytrips_of_year(context: Context, mzmv_year: int, trips: pd.DataFrame) -> None:
    trips = trips.copy()
    trips['n_trips'] = 1
    agg_fns: dict[str, str] = {}
    for col in trips.columns:
        if 'metric_is' in col or col in ('is_outward', 'is_return'):
            agg_fns[col] = 'mean'
        elif col in ('metric_ghg_kg', 'externalities_chf', 'time_measured',
                     'dist_measured', 'n_trips'):
            agg_fns[col] = 'sum'
        elif col in ('is_land_based', 'subset_cities_within'):
            agg_fns[col] = 'min'
        elif col.startswith('subset_'):
            agg_fns[col] = 'max'
    daytrips = trips.groupby('HHNR').agg(agg_fns)
    households = load_households(context, mzmv_year)
    sd_cols_hh = [c for c in households.columns if c.startswith('sd_')]
    daytrips = daytrips.join(households[sd_cols_hh + list(_HH_JOIN_COLS)])
    context.create_generic(daytrips, f'mzmv_{mzmv_year}/daytrips.csv')


def process_legs_of_year(context: Context, mzmv_year: int,
                         measured_routes: pd.DataFrame, max_geom_size: int) -> pd.DataFrame:
    legs = context.get_generic(
        f'mzmv_{mzmv_year}/preprocessed/legs.csv',
    )

    legs['time_measured'] = (legs['e_dauer'] * 60).round().astype(int)
    legs['dist_measured'] = (legs['rdist'] * 1000).round().astype(int)
    legs['speed_kmh'] = legs['rdist'] / (legs['e_dauer'] / 60)

    legs['mode'] = legs['f51300'].replace(F51300_TO_STR)
    legs['mode_simplified'] = legs['mode'].replace(MODE_TO_SIMPLIFIED)
    legs['mode_is_passenger'] = legs['f51300'].isin([8, 10]).astype(int)
    _set_plane_mode_detail(legs)
    legs['is_land_based'] = (~legs['mode'].isin(['plane', 'boat'])).astype(int)
    legs['car_occupancy'] = legs['f51320']
    invalid_occupancy = (
        (legs['car_occupancy'] <= 0) | (legs['car_occupancy'] > 9) |
        (legs['mode_simplified'] != 'car')
    )
    legs.loc[invalid_occupancy, 'car_occupancy'] = np.nan

    # +1 min so that ":30" consistently rounds up — self-reported start times bunch
    # heavily on the half-hour. Hour 24 wraps to 0; trips after 24:29 dropped.
    legs['hour_of_day'] = ((legs['f51100'] + 1) / 60).round().fillna(-1).astype(int)
    legs.loc[legs['hour_of_day'] == 24, 'hour_of_day'] = 0
    legs.loc[legs['f51100'] > (24 * 60), 'hour_of_day'] = -1

    households = load_households(context, mzmv_year)
    other_cols = list(_HH_JOIN_COLS) + ['datetime_day']
    sd_cols_hh = [c for c in households.columns if c.startswith('sd_')]
    legs = legs.join(households[sd_cols_hh + other_cols], on='HHNR')
    context.create_generic(households, f'mzmv_{mzmv_year}/households.csv')

    # Build per-leg datetimes by combining the day-level datetime with a derived
    # H:MM. (Modulo 24 lets MZMV's "next day" minutes flow naturally back to hour 0+.)
    for raw_col, label in (('f51100', 'departure'), ('f51400', 'arrival')):
        hour = ((legs[raw_col] // 60) % 24).round().astype(int).astype(str).str.zfill(2)
        minute = (legs[raw_col] % 60).round().astype(int).astype(str).str.zfill(2)
        legs[f'datetime_{label}'] = pd.to_datetime(
            legs['datetime_day'].astype(str) + ' ' + hour + ':' + minute + ':00',
        )
        legs[f'datetime_{label}_tz'] = common.attach_time_zone(
            legs[f'datetime_{label}'], 'Europe/Zurich',
        )

    legs['n_legs'] = 1
    legs['weight_person'] = legs['WP']
    legs['trip_id'] = _vectorized_leg_ids(mzmv_year, legs['HHNR'], legs['WEGNR'])
    legs['leg_id'] = _vectorized_leg_ids(mzmv_year, legs['HHNR'], legs['ETNR'])
    legs['n_legs_in_trip'] = legs.groupby('trip_id')['n_legs'].transform('count')

    geom_size = measured_routes['geometry_size'].to_dict()
    legs['route_geo_size'] = legs['leg_id'].map(geom_size).fillna(-1).astype(int)
    logging.info(
        f"{(legs['route_geo_size'] <= 0).sum() / len(legs) * 100:.1f}% "
        f"of routes have no route geometry")

    legs = legs.set_index('leg_id')
    legs = common.add_group_id(legs, 'HHNR')
    legs = add_indicators(legs, None)
    legs = legs[legs['sd_ordinal_age'] != 'leq6']

    legs = legs.rename(columns={'S_X_LV95': 'orig_x', 'Z_X_LV95': 'dest_x',
                                'S_Y_LV95': 'orig_y', 'Z_Y_LV95': 'dest_y'})
    legs = standardize.attach_lat_lon_and_dist_line(legs)
    # Sample the Swiss DEM at each leg's endpoints — supplies
    # `elev_orig` / `elev_dest` for downstream feature engineering,
    # alpine-trip filtering, and person-environment analyses.
    dem_ctx = context.source('preparation/world/elevation', storage=Storage.PUBLIC)
    dem_path = dem_ctx.path_for(Storage.PUBLIC, 'dem_switzerland.tif')
    standardize.attach_endpoint_elevation(legs, dem_path=dem_path)
    # Composite elevation gate (data-quality + alpine exclusion) —
    # single source of truth consumed by 04, 07, 02d.
    standardize.attach_within_elevation_band_flag(legs)
    save_ghg_stats(context, legs, mzmv_year, n_households=households.index.nunique())

    car = legs['mode'] == 'car'
    if car.any():
        logging.info(f"Peak (car):  {legs.loc[car, 'hour_peak'].sum() / car.sum() * 100:.1f}%")
        logging.info(f"Night (car): {legs.loc[car, 'hour_night'].sum() / car.sum() * 100:.1f}%")
    logging.info(f"Originally: {len(legs):,} rows ({legs['HHNR'].nunique():,} households)")

    f_dist_time = (legs['rdist'] <= 0) | (legs['e_dauer'] <= 0) | (legs['dist_line'] < 1)
    f_outside_ch = (legs['S_LND'] != 8100) | (legs['Z_LND'] != 8100)
    # Speed + detour envelopes: flags only. Downstream (04_edge_weights,
    # 07a_road_overhead_coefs, 09a_utility_estimation) filter on
    # `is_within_speed_envelope==1 & is_within_detour_envelope==1`
    # alongside `is_valid_domestic_trip==1`.
    # MTMC self-reports times in ~5-min steps + may include unlock /
    # trip-overhead time — this deflates apparent speed on very short
    # trips (esp. walks under ~1 km). Skip the LOWER envelope bound
    # below 1 km; the upper bound still catches gross misclassifications.
    standardize.attach_speed_envelope_flag(legs, short_trip_dist_m=1_000.0)
    standardize.attach_within_detour_envelope_flag(legs)
    # Per-reason validity flags + composite `is_valid_domestic_trip`.
    standardize.attach_validity_flags(legs, validity_checks={
        'is_valid_dist_time': f_dist_time,
        'is_in_switzerland': f_outside_ch,
    })
    valid = legs['is_valid_domestic_trip'] == 1
    for mode, n in legs.loc[valid, 'mode'].value_counts().items():
        logging.info(f"   -> {mode}: {n:,}")
    standardize.attach_include_in_route_matching(legs, max_size=max_geom_size)
    logging.info(f"      of which have route: {legs['include_in_route_matching'].sum():,}")

    # legs = surveys.add_zone_columns_for_endpoints(legs, zones, crs_main)
    # legs = common.add_swiss_city_subsets_trips(legs)

    context.create_generic(standardize.filter_columns_by_prefix(legs),
                           f'mzmv_{mzmv_year}/legs.csv')
    return legs


# ---------------------------------------------------------------------------
# Routes (raw shapefile -> per-leg simplified coordinate arrays)
# ---------------------------------------------------------------------------

def _process_one_route(mzmv_year: int, row: pd.Series,
                       target_size: int, max_size: int) -> pd.Series:
    """Simplify one row's route geometry. OSRM `--max-matching-size` should match
    `target_size`. Invalid geometries return -1 sentinels."""
    leg_id = get_leg_id(mzmv_year, row['HHNR'], row['ETNR'])
    try:
        simplified, new_size, ratio = geo_processing.simplify_geometry(
            row['geometry'], target_size, max_size,
        )
        x, y = simplified.xy
        coords = np.vstack([x, y]).T
        return pd.Series({'leg_id': leg_id, 'route': coords,
                          'geometry_size': new_size, 'geometry_size_ratio': ratio})
    except NotImplementedError:
        # Raised by `.xy` on certain invalid geometries.
        return pd.Series({'leg_id': leg_id, 'route': np.nan,
                          'geometry_size': -1, 'geometry_size_ratio': -1})


def process_routes_of_year(context: Context, mzmv_year: int,
                           target_size: int, max_size: int) -> pd.DataFrame:
    raw = (f'switzerland/mzmv/MZMV{mzmv_year}_mit_Geo/'
           f'5_Routen(Geometriefiles)/CH_Routen/Routen_CH.shp')
    routes = gpd.read_file(context.raw_path(Storage.PRIVATE, raw))
    out = routes.apply(
        lambda r: _process_one_route(mzmv_year, r, target_size, max_size), axis=1,
    ).set_index('leg_id')
    context.create_generic(
        out['route'].to_dict(), f'mzmv_{mzmv_year}/legs.routes.measured.npy',
    )
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(variant) -> None:
    context = init_context(variant)
    # TODO(zones-migration): the zones / lfi_regions data hasn't been moved into the
    # new preparation tree yet. When `preparation/switzerland/zones` lands, switch
    # the .source(...) namespace below.
    # zones_src = context.source('preparation/switzerland/zones')
    # zones = zones_src.get_properties('zones', ['core', 'relations'], add_shapes=True)
    # lfi_regions = zones_src.get_shapes('lfi_regions')

    measured_routes = process_routes_of_year(context,
                                             variant.mzmv_year,
                                             common.TARGET_GEOMETRY_SIZE,
                                             common.MAXIMUM_GEOMETRY_SIZE,
    )
    legs = process_legs_of_year(context,
                                variant.mzmv_year,
                                measured_routes,
                                common.MAXIMUM_GEOMETRY_SIZE)
    trips = process_trips_of_year(context, variant.mzmv_year, legs)
    logging.info(
        f"Per-mode mean/median distance:\n"
        f"{trips.groupby('mode').agg({'dist_measured': ['mean', 'median']})}")
    process_daytrips_of_year(context, variant.mzmv_year, trips)
    context.close()


if __name__ == '__main__':
    variants.run(main)
