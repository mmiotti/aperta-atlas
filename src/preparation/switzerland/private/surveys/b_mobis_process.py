"""
Process the MOBIS travel-tracking dataset into per-leg tables for two cohorts:
pre-Covid (`mobis_precovid`, year_month < 2020) and Covid (`mobis_covid`, week 13 of 2020
through end-of-March 2022).

> **Private data.** Consumes MOBIS trajectory dumps (Switzerland's
> continuous mobility tracking panel); not redistributable. Published
> as documentation of the method.

Reads the raw `legs.csv` plus the simplified trajectories produced by
`a_mobis_trajectories_process.py`, attaches mode / peak / weekday / daylight /
season flags and the per-leg validity / plausibility flags (see `standardize.py`),
joins the route geometry sizes, and writes the result in the same format as the MTMC legs
(`b_mzmv_process.py`): indexed by `leg_id` (the MOBIS leg id), local clock times in
`datetime_departure` / `datetime_arrival`, the columns of `ALLOWED_SURVEY_PREFIXES`.
MOBIS has no person covariates here; `weight_person` is 1.

Inputs (PRIVATE, under raw/switzerland/mobis/tracking/):
    legs.csv                                                 # raw MOBIS legs
Inputs (PRIVATE, under preparation/switzerland/surveys/):
    mobis_all/trajectories.npy / trajectories.csv            # simplified routes
Inputs (PUBLIC, under preparation/world/elevation/):
    dem_switzerland.tif                                      # endpoint elevation

Outputs (PRIVATE, under preparation/switzerland/surveys/):
    <survey>/legs.csv                              # one row per leg, indexed by leg_id
    <survey>/legs.routes.measured.npy              # {leg_id: route coordinates}
where <survey> ∈ {'mobis_precovid', 'mobis_covid'}.

Run all variants sequentially (default):
    python -m preparation.switzerland.private.surveys.b_mobis_process
Single variant:
    python -m preparation.switzerland.private.surveys.b_mobis_process --variant pre_covid
"""

import logging
import warnings

import pandas as pd

# Silence pandas' DataFrame-fragmentation warnings from the many column inserts.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from aperta_atlas.context import Context, Storage, init_context
from aperta_atlas.variant import Variants

from preparation.switzerland.private.surveys import common, standardize


variants = Variants([('pre_covid', bool)])
variants.add(name='pre_covid', pre_covid=True)
variants.add(name='covid', pre_covid=False)


# MOBIS raw `mode` column → simplified mode string. Bike family: MOBIS added `Mode::Ebicycle`
# in 2020-07 (`src/misc/mobis_mode_over_time.py`) and never tells ebike25 from ebike45, so
# Ebicycle is 'anyebike'; Bicycle is 'anybike' before the split (mechanical + e-bike mixed) and
# 'rbike' (confirmed mechanical) after it, set by the date-conditional override in `process`.
_MOBIS_MODE_MAP: dict[str, str] = {
    'Mode::Walk': 'walk',
    'Mode::Car': 'car',
    'Mode::Bicycle': 'anybike',          # overridden to 'rbike' post-2020-07 below
    'Mode::Ebicycle': 'anyebike',
    'Mode::Bikesharing': 'anybike',      # shared-bike fleet mixes mechanical + ebike
    'Mode::Motorbike': 'motorbike',
    'Mode::MotorbikeScooter': 'motorbike',
    'Mode::TaxiUber': 'car',
    'Mode::CarsharingMobility': 'car',
    'Mode::RidepoolingPikmi': 'car',
    'Mode::Escooter': 'micro',
    'Mode::Etrottinett': 'micro',
    # Water / air modes — not on the modeled network. Feed `is_land_based`.
    # Ferry lumped with boat; cablecar lumped with aerialway (semantics
    # match at the mode_simplified level).
    'Mode::Airplane': 'plane',
    'Mode::Boat': 'boat',
    'Mode::Ferry': 'boat',
    'Mode::Cablecar': 'aerialway',
    'Mode::Aerialway': 'aerialway',
    'Mode::Bus': 'transit',
    'Mode::Train': 'transit',
    'Mode::Tram': 'transit',
    'Mode::LightRail': 'transit',
    'Mode::RegionalTrain': 'transit',
    'Mode::Subway': 'transit',
}

_MOBIS_EBICYCLE_SPLIT_YM: float = 2020.07     # year + month / 100 of the Ebicycle split

# Covid cohort window: from week 13 of 2020 (first week with new participants) to
# end of March 2022 (end of Schutzmassnahmen). Encoded in the same year+month/100 and
# year+week/100 scheme the original legs CSV uses for fast numeric filtering.
_COVID_YEAR_WEEK_FROM = 2020.13 - 1e-3
_COVID_YEAR_MONTH_TO = 2022.03 + 1e-3


def _filter_to_cohort(legs: pd.DataFrame, pre_covid: bool) -> pd.DataFrame:
    if pre_covid:
        return legs[legs['year_month'] < 2020]
    return legs[
        (legs['year_week'] >= _COVID_YEAR_WEEK_FROM) &
        (legs['year_month'] <= _COVID_YEAR_MONTH_TO)
    ]


def process(context: Context, pre_covid: bool, traj_routes: pd.Series,
            traj_df: pd.DataFrame) -> None:
    logging.info(f"Starting processing (pre_covid={pre_covid})")
    survey_name = 'mobis_precovid' if pre_covid else 'mobis_covid'

    raw_path = context.raw_path(Storage.PRIVATE, 'switzerland/mobis/tracking/legs.csv')
    legs = pd.read_csv(raw_path)

    # Raw timestamps are UTC (`…Z`); hours, weekdays and cohort weeks are local Swiss time.
    for col in ('started_at', 'finished_at'):
        legs[col] = pd.to_datetime(legs[col], utc=True).dt.tz_convert('Europe/Zurich')
    # Written out as local clock times, like the MTMC legs (the tz-aware columns above
    # stay for the gap flag, which needs exact durations across DST changes).
    legs['datetime_departure'] = legs['started_at'].dt.tz_localize(None)
    legs['datetime_arrival'] = legs['finished_at'].dt.tz_localize(None)
    legs['year_month'] = (legs['started_at'].dt.year +
                          legs['started_at'].dt.month / 100).round(2)
    legs['year_week'] = (legs['started_at'].dt.year +
                         legs['started_at'].dt.isocalendar().week / 100).round(2)

    n_before = len(legs)
    legs = _filter_to_cohort(legs, pre_covid)
    logging.info(f"Cohort filter pre_covid={pre_covid}: {n_before:,} -> {len(legs):,} "
                 f"samples ({legs['participant_id'].nunique():,} participants)")

    legs['sd_bool_weekday'] = (legs['started_at'].dt.dayofweek < 5).astype(int)
    # Departure hour, rounded to the nearest hour, on the local clock (rounding tz-aware
    # times fails in the repeated hour when DST ends).
    legs['hour_of_day'] = legs['datetime_departure'].dt.round('h').dt.hour
    standardize.attach_hour_flags(legs)

    legs['time_measured'] = legs['duration'].round().astype(int)
    legs['dist_measured'] = legs['length'].round().astype(int)
    legs['speed_kmh'] = legs['length'] * 3.6 / legs['duration']
    legs = common.add_group_id(legs, 'participant_id')

    # Raw modes missing from `_MOBIS_MODE_MAP` pass through untranslated: add entries for them.
    unmapped = set(legs['mode'].dropna().unique()) - set(_MOBIS_MODE_MAP)
    if unmapped:
        counts = legs['mode'].value_counts()
        preview = {m: int(counts.get(m, 0)) for m in sorted(unmapped)}
        logging.warning(
            f"{len(unmapped)} raw MOBIS mode value(s) missing from "
            f"_MOBIS_MODE_MAP — passing through untranslated: {preview}")
    legs['mode_simplified'] = legs['mode'].replace(_MOBIS_MODE_MAP)
    post_split_bicycle = (
        (legs['mode'] == 'Mode::Bicycle')
        & (legs['year_month'] >= _MOBIS_EBICYCLE_SPLIT_YM)
    )
    legs.loc[post_split_bicycle, 'mode_simplified'] = 'rbike'
    legs['weight_person'] = 1

    car = legs['mode_simplified'] == 'car'
    if car.any():
        logging.info(f"Peak (car):  {legs.loc[car, 'hour_peak'].sum() / car.sum() * 100:.1f}%")
        logging.info(f"Night (car): {legs.loc[car, 'hour_night'].sum() / car.sum() * 100:.1f}%")

    f_dist_time = (legs['dist_measured'] <= 0) | (legs['time_measured'] <= 0) | legs['trip_id'].isna()
    f_implausible = legs['implausible'] == True  # noqa: E712 — keep explicit for clarity on Series
    f_outside_ch = legs['in_switzerland'] != True  # noqa: E712
    # Land-based: excludes air / water / aerial-cable modes (as in MZMV);
    # relies on `_MOBIS_MODE_MAP` mapping them to 'plane' / 'boat' / 'aerialway'.
    legs['is_land_based'] = (
        ~legs['mode_simplified'].isin(['plane', 'boat', 'aerialway'])).astype(int)
    # Speed + detour envelopes: flags only (detour below, once `dist_line` exists).
    standardize.attach_speed_envelope_flag(legs)
    # Per-reason validity flags + composite `is_valid_domestic_trip`.
    # Order sets the cumulative log breakdown; total is order-independent.
    standardize.attach_validity_flags(legs, validity_checks={
        'is_valid_dist_time': f_dist_time,
        'is_plausible': f_implausible,
        'is_in_switzerland': f_outside_ch,
    })
    valid = legs['is_valid_domestic_trip'] == 1
    for mode, n in legs.loc[valid, 'mode_simplified'].value_counts().items():
        logging.info(f"   -> {mode}: {n:,}")

    legs = legs.rename(columns={
        'start_x': 'orig_x', 'end_x': 'dest_x',
        'start_y': 'orig_y', 'end_y': 'dest_y',
    })
    legs = standardize.attach_lat_lon_and_dist_line(legs)
    standardize.attach_daylight_and_season(legs)
    standardize.attach_within_detour_envelope_flag(legs)
    standardize.attach_within_gap_envelope_flag(
        legs, departure_col='started_at', arrival_col='finished_at')
    # Endpoint elevation from the Swiss DEM (`elev_orig` / `elev_dest`).
    dem_ctx = context.source('preparation/world/elevation', storage=Storage.PUBLIC)
    dem_path = dem_ctx.path_for(Storage.PUBLIC, 'dem_switzerland.tif')
    standardize.attach_endpoint_elevation(legs, dem_path=dem_path)
    # Composite elevation gate (data quality + alpine exclusion).
    standardize.attach_within_elevation_band_flag(legs)
    legs = legs.set_index('leg_id')
    n_before = len(legs)
    legs = legs[~legs.index.duplicated(keep='first')].copy()
    if len(legs) < n_before:
        logging.warning(
            f"{n_before - len(legs):,} duplicate trajectories removed (of {n_before:,}).")
    legs['n_legs_in_trip'] = legs.groupby('trip_id')['trip_id'].transform('size')
    legs['route_geo_size'] = traj_df['geometry_size'].reindex(legs.index).fillna(-1).astype(int)
    standardize.attach_include_in_route_matching(legs)
    n_routed = legs['include_in_route_matching'].sum()
    logging.info(f"...containing route: {n_routed:,} ({n_routed/len(legs)*100:.1f}%)")

    context.create_generic(standardize.filter_columns_by_prefix(legs), f'{survey_name}/legs.csv')
    routes = traj_routes[traj_routes.index.isin(legs.index)]
    context.create_generic(routes.to_dict(), f'{survey_name}/legs.routes.measured.npy')


def main(variant) -> None:
    context = init_context(variant)
    traj_routes = pd.Series(context.get_generic('mobis_all/trajectories.npy'), name='route')
    traj_df = context.get_generic('mobis_all/trajectories.csv')

    process(context, variant.pre_covid, traj_routes, traj_df)
    context.close()


if __name__ == '__main__':
    variants.run(main)
