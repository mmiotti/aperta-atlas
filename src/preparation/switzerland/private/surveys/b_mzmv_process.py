"""
Turn the standardized MTMC (MZMV) legs into the survey legs the atlas calibrates on: simplified
modes, peak / night flags, daylight / late-night / season flags, validity and plausibility flags
(`standardize.py`), endpoint elevation, and each traveller's sociodemographic dummies.

> **Private data.** Consumes the standardized MZMV tables (Switzerland's federal mobility
> survey, BFS-restricted); not redistributable. Published as documentation of the method.

Reads the cross-year tables written by `a_mzmv_standardize.py` (columns and caveats:
`mzmv/README.md`) and keeps one row per leg, including legs that fail a check: the flags say
which, and each consumer (`main/04_edge_weights.py`, `survey/02d_prepare_survey_legs.py`)
applies the ones it needs. Also simplifies the BFS route geometries of the legs, for map
matching.

Inputs (PRIVATE, under preparation/switzerland/surveys/mzmv_<year>/standardized/):
    legs.csv, people.csv, households.csv            # from a_mzmv_standardize
Inputs (PRIVATE, under raw/switzerland/mzmv/):
    MZMV<year>_mit_Geo/5_Routen(Geometriefiles)/CH_Routen/Routen_CH.shp
Inputs (PUBLIC, under preparation/world/elevation/):
    dem_switzerland.tif                             # endpoint elevation

Outputs (PRIVATE, under preparation/switzerland/surveys/):
    mzmv_<year>/legs.csv                            # one row per leg, indexed by leg_id
    mzmv_<year>/legs.routes.measured.npy            # {leg_id: simplified route coordinates}

Run all years sequentially (default):
    python -m preparation.switzerland.private.surveys.b_mzmv_process
Single year:
    python -m preparation.switzerland.private.surveys.b_mzmv_process --variant 2021
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta import geo_processing
from aperta_atlas.context import Context, Storage, init_context
from aperta_atlas.variant import Variants

from preparation.switzerland.private.surveys import common, standardize
from preparation.switzerland.private.surveys.common import MODE_TO_SIMPLIFIED
from preparation.switzerland.private.surveys.mzmv import schema


# Years with route geometries; the atlas pools both.
MZMV_YEARS = (2015, 2021)

variants = Variants([('mzmv_year', int)])
for _y in MZMV_YEARS:
    variants.add(name=str(_y), mzmv_year=_y)


# Bins of the sd_ordinal_* covariates (right-closed; labels from `common.bin_labels`).
BINS_AGE = (0, 6, 13, 18, 25, 35, 50, 65, 75, 120)
BINS_N_VEHICLES = (-1, 0, 1, 2, 99)
BINS_N_EBIKES = (-1, 0, 1, 99)
# Gross monthly household income (CHF), cut on the lower bound of the surveyed band.
BINS_INCOME = (-1, 4_000, 8_000, 12_000, 16_000, np.inf)

# Weather on the survey day, in three levels (`sd_bool_weather_<level>`).
_WEATHER_LEVELS = {
    'good': ('sunny', 'partly_cloudy', 'hot'),
    'medium': ('cloudy', 'foggy', 'unstable'),
    'bad': ('rainy', 'snowy', 'cold'),
}


# ---------------------------------------------------------------------------
# Travellers: covariates of each target person and their household
# ---------------------------------------------------------------------------

def load_travellers(context: Context, mzmv_year: int) -> pd.DataFrame:
    """Survey day, home and `sd_*` covariates of each target person, indexed by `person_id`.
    Every category of an `sd_nominal_*` / `sd_ordinal_*` covariate gets an `sd_cat_*` dummy,
    also when no one is in it, so the survey years have the same columns."""
    people = context.get_generic(f'mzmv_{mzmv_year}/standardized/people.csv')
    households = context.get_generic(f'mzmv_{mzmv_year}/standardized/households.csv')
    hh = households.reindex(people['hh_id']).set_axis(people.index)

    out = pd.DataFrame(index=people.index)
    out['datetime_day'] = pd.to_datetime(people['survey_date'])
    out['sd_bool_weekday'] = people['weekday'].between(1, 5).astype(int)
    out['sd_age'] = people['age']
    out['sd_ordinal_age'] = pd.cut(people['age'], BINS_AGE, labels=common.bin_labels(BINS_AGE))
    out['sd_nominal_sex'] = pd.Categorical(people['sex'].map({'male': 'm', 'female': 'w'}),
                                           categories=['m', 'w'])
    for col in ('license_car', 'license_motorbike'):
        out[f'sd_bool_{col}'] = people[col].eq(1).astype(int)
    for col, name in (('pass_ga', 'ga'), ('pass_halbtax', 'halbtax'),
                      ('pass_regional', 'verbund'), ('pass_season', 'season')):
        out[f'sd_bool_transitpass_{name}'] = people[col].eq(1).astype(int)
    out['sd_nominal_education'] = pd.Categorical(people['education'], categories=schema.EDUCATION)
    out['sd_nominal_weather'] = pd.Categorical(people['weather'], categories=schema.WEATHER)
    for level, values in _WEATHER_LEVELS.items():
        out[f'sd_bool_weather_{level}'] = people['weather'].isin(values).astype(int)

    out['hh_x'], out['hh_y'] = hh['home_x'], hh['home_y']
    out['hh_n_people'] = hh['n_people']
    out['hh_n_cars'] = hh['n_cars']
    # Adults: single parents 1, couples with kids 2, any other household all its members.
    out['hh_n_adults'] = np.select(
        [hh['hh_type'] == 'single_parent', hh['hh_type'] == 'couple_kids'], [1, 2],
        default=hh['n_people'])
    out['sd_ordinal_n_cars'] = _cut(hh['n_cars'], BINS_N_VEHICLES)
    out['sd_ordinal_n_car_parking'] = _cut(hh['n_car_parking'], BINS_N_VEHICLES)
    bikes = hh['n_bikes_any'] - hh['n_ebikes_slow'] - hh['n_ebikes_fast']   # without assist
    out['sd_ordinal_n_bikes'] = _cut(bikes, BINS_N_VEHICLES)
    out['sd_ordinal_n_ebikes'] = _cut(hh['n_ebikes_slow'], BINS_N_EBIKES)
    out['sd_ordinal_n_ebikes_fast'] = _cut(hh['n_ebikes_fast'], BINS_N_EBIKES)
    out['sd_ordinal_car_ratio'] = pd.Categorical(
        np.select([hh['n_cars'] >= out['hh_n_adults'], hh['n_cars'] > 0], ['1+', '<1'],
                  default='none'),
        categories=['none', '<1', '1+'])
    out['sd_nominal_hhtype'] = pd.Categorical(hh['hh_type'], categories=schema.HH_TYPES)
    income = _cut(hh['income_min_chf'], BINS_INCOME)
    out['sd_ordinal_income'] = income.cat.add_categories('na').fillna('na')
    out['sd_bool_second_home'] = hh['has_second_home'].eq(1).astype(int)

    for col in [c for c in out.columns if c.startswith(('sd_nominal_', 'sd_ordinal_'))]:
        prefix = col.replace('_nominal', '_cat').replace('_ordinal', '_cat')
        out = out.join(pd.get_dummies(out[col], prefix=prefix).astype(int))
    return out


def _cut(values: pd.Series, bins: tuple) -> pd.Series:
    return pd.cut(values, bins, labels=common.bin_labels(bins))


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def process_legs_of_year(context: Context, mzmv_year: int,
                         measured_routes: pd.DataFrame) -> None:
    legs = context.get_generic(f'mzmv_{mzmv_year}/standardized/legs.csv')
    logging.info(f"Loaded {len(legs):,} legs ({legs['hh_id'].nunique():,} households)")

    legs['time_measured'] = (legs['duration_min'] * 60).round()
    legs['dist_measured'] = (legs['dist_km'] * 1000).round()
    legs['speed_kmh'] = legs['dist_km'] / (legs['duration_min'] / 60)
    legs['mode_simplified'] = legs['mode'].replace(MODE_TO_SIMPLIFIED)
    # Taxi riders are passengers too (`taxi` → `car` in `mode_simplified`; the survey asks
    # driver / passenger only for cars and motorbikes).
    legs['mode_is_passenger'] = (legs['is_passenger'].eq(1)
                                 | legs['mode'].eq('taxi')).astype(int)
    legs['is_land_based'] = (~legs['mode'].isin(['plane', 'boat', 'aerialway'])).astype(int)
    legs['weight_person'] = legs['weight']
    legs['n_legs_in_trip'] = legs.groupby('trip_id')['trip_id'].transform('size')
    legs = legs.join(load_travellers(context, mzmv_year), on='person_id')

    # Departure hour, rounded; +1 min so that most ":30" reports round up (self-reported start
    # times bunch on the half-hour). Times count minutes from the survey day's midnight, so
    # departures after 24:00 wrap to the next day's hours. Missing → -1.
    minutes = legs['dep_min'] % (24 * 60)
    legs['hour_of_day'] = (((minutes + 1) / 60).round() % 24).fillna(-1).astype(int)
    standardize.attach_hour_flags(legs)
    # Clock times past 24:00 fall on the next day; missing → NaT.
    for col, label in (('dep_min', 'departure'), ('arr_min', 'arrival')):
        legs[f'datetime_{label}'] = (legs['datetime_day']
                                     + pd.to_timedelta(legs[col].round(), unit='min'))

    geom_size = measured_routes['geometry_size'].to_dict()
    legs['route_geo_size'] = legs.index.map(geom_size).fillna(-1).astype(int)
    logging.info(f"{(legs['route_geo_size'] <= 0).mean() * 100:.1f}% of legs have no route "
                 f"geometry")
    legs = common.add_group_id(legs, 'hh_id')

    # Target persons are 6 and older; age 6 alone is too partial a band to keep.
    legs = legs[~(legs['sd_age'] <= 6)].copy()
    legs = standardize.attach_lat_lon_and_dist_line(legs)
    standardize.attach_daylight_and_season(legs)
    # Endpoint elevation from the Swiss DEM (`elev_orig` / `elev_dest`).
    dem_ctx = context.source('preparation/world/elevation', storage=Storage.PUBLIC)
    dem_path = dem_ctx.path_for(Storage.PUBLIC, 'dem_switzerland.tif')
    standardize.attach_endpoint_elevation(legs, dem_path=dem_path)
    # Composite elevation gate (data quality + alpine exclusion).
    standardize.attach_within_elevation_band_flag(legs)

    car = legs['mode'] == 'car'
    if car.any():
        logging.info(f"Peak (car):  {legs.loc[car, 'hour_peak'].mean() * 100:.1f}%")
        logging.info(f"Night (car): {legs.loc[car, 'hour_night'].mean() * 100:.1f}%")
    logging.info(f"Legs kept: {len(legs):,} ({legs['hh_id'].nunique():,} households)")

    # Missing distance, time or coordinates count as invalid; so does an unknown country.
    f_dist_time = (~(legs['dist_km'] > 0) | ~(legs['duration_min'] > 0)
                   | ~(legs['dist_line'] >= 1))
    f_outside_ch = (legs['orig_in_switzerland'] != 1) | (legs['dest_in_switzerland'] != 1)
    # Speed + detour envelopes: flags only. MTMC self-reports times mostly in 5-min steps,
    # so the speed gate accepts a leg if any time its report can stand for gives an
    # in-envelope speed (`rounded_reports=True`).
    standardize.attach_speed_envelope_flag(legs, rounded_reports=True)
    standardize.attach_within_detour_envelope_flag(legs)
    standardize.attach_within_gap_envelope_flag(
        legs, departure_col='datetime_departure', arrival_col='datetime_arrival')
    # Per-reason validity flags + composite `is_valid_domestic_trip`.
    standardize.attach_validity_flags(legs, validity_checks={
        'is_valid_dist_time': f_dist_time,
        'is_in_switzerland': f_outside_ch,
    })
    valid = legs['is_valid_domestic_trip'] == 1
    for mode, n in legs.loc[valid, 'mode'].value_counts().items():
        logging.info(f"   -> {mode}: {n:,}")
    standardize.attach_include_in_route_matching(legs)
    logging.info(f"      of which have route: {legs['include_in_route_matching'].sum():,}")

    context.create_generic(standardize.filter_columns_by_prefix(legs),
                           f'mzmv_{mzmv_year}/legs.csv')


# ---------------------------------------------------------------------------
# Routes (raw shapefile -> per-leg simplified coordinate arrays)
# ---------------------------------------------------------------------------

def get_leg_id(year: int, hhnr: int | str, etnr: int | str) -> str:
    """The standardized leg id (`mzmv/schema.py`): from 2015 the target person is always
    no. 1, and ETNR numbers the legs over the survey day."""
    return f"{year}_{hhnr}_1_{etnr}"


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
    measured_routes = process_routes_of_year(context, variant.mzmv_year,
                                             common.TARGET_GEOMETRY_SIZE,
                                             common.MAXIMUM_GEOMETRY_SIZE)
    process_legs_of_year(context, variant.mzmv_year, measured_routes)
    context.close()


if __name__ == '__main__':
    variants.run(main)
