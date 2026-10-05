"""
Process the MOBIS travel-tracking dataset into per-leg tables for two cohorts:
pre-Covid (`mobis_precovid`, year_month < 2020) and Covid (`mobis_covid`, week 13 of 2020
through end-of-March 2022).

> **Private data.** Consumes MOBIS trajectory dumps (Switzerland's
> continuous mobility tracking panel); not redistributable. Published
> as documentation of the method.

Reads the raw `legs.csv` plus the simplified trajectories produced by
`a_mobis_trajectories_process.py`, attaches mode / peak / weekday flags, joins
zone metadata, applies the inclusion filters for the mode-choice and travel-time
models, and writes the result.

Inputs (under <DATA_DIR_PRIVATE>/raw/switzerland/mobis/tracking/):
    legs.csv                                                 # raw MOBIS legs
Inputs (prepared, prior stage):
    mobis_all/trajectories.npy / trajectories.csv            # simplified routes
Inputs (cross-namespace; not yet migrated — TODO):
    preparation/switzerland/zones :: properties + shapes for zones.
    -> TODO: finalize. Moved zone-based information away from here so that this script doesn't need zones.

Outputs:
    <survey>/legs.csv                              # PRIVATE — full legs
    <survey>/legs.sample.csv                       # PRIVATE — first 1000 legs
    <survey>/legs.routes.measured.npy              # PRIVATE — per-leg route arrays
where <survey> ∈ {'mobis_precovid', 'mobis_covid'}.

Run all variants sequentially (default):
    python -m preparation.switzerland.private.surveys.b_mobis_process
Single variant:
    python -m preparation.switzerland.private.surveys.b_mobis_process --variant pre_covid
"""

import logging
import warnings

import geopandas as gpd
import pandas as pd

# Same PerformanceWarning silence as in `b_mzmv_process.py` — see that
# file's comment for rationale.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage
from aperta_atlas.variant import Variants

from preparation.switzerland.private.surveys import common, standardize


variants = Variants([('pre_covid', bool)])
variants.add(name='pre_covid', pre_covid=True)
variants.add(name='covid', pre_covid=False)


# MOBIS raw `mode` column → simplified mode string.
#
# Bike-family notes (from `src/misc/mobis_mode_over_time.py` diagnostic):
#
#   `Mode::Ebicycle` was introduced in MOBIS on 2020-07. Before that,
#   Mode::Bicycle was an inherently mixed mechanical+ebike bucket (no
#   way to disentangle). After that, users can classify as Ebicycle if
#   they want, but MOBIS never distinguishes ebike25 from ebike45.
#
#   Naming reflects the confidence dimension:
#     'rbike'    — confirmed mechanical (only from post-split Bicycle)
#     'anybike'  — bike, sub-type unknown (pre-split Bicycle)
#     'anyebike' — ebike, assist limit unknown (Ebicycle: 25 or 45?)
#
#   The pre/post-split distinction for Mode::Bicycle happens after the
#   dict lookup (see the year_month-conditional override below), because
#   `.replace()` can't do date-conditional mapping. Mode::Bicycle's dict
#   value here is the pre-split default; the override upgrades post-split
#   rows to 'rbike'.
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

# MOBIS introduced the Mode::Ebicycle category in 2020-07 (verified via
# src/misc/mobis_mode_over_time.py). Post-split Bicycle rows are treated
# as confirmed mechanical ('rbike'); pre-split rows stay 'anybike'.
_MOBIS_EBICYCLE_SPLIT_YM: float = 2020.07

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


def process(context: Context, pre_covid: bool, zones: gpd.GeoDataFrame,
            traj_routes: pd.Series, traj_df: pd.DataFrame) -> None:
    logging.info(f"Starting processing (pre_covid={pre_covid})")
    survey_name = 'mobis_precovid' if pre_covid else 'mobis_covid'

    raw_path = context.raw_path(Storage.PRIVATE, 'switzerland/mobis/tracking/legs.csv')
    legs = pd.read_csv(raw_path)

    legs['started_at'] = pd.to_datetime(legs['started_at'])
    legs['finished_at'] = pd.to_datetime(legs['finished_at'])
    legs['year_month'] = (legs['started_at'].dt.year +
                          legs['started_at'].dt.month / 100).round(2)
    legs['year_week'] = (legs['started_at'].dt.year +
                         legs['started_at'].dt.isocalendar().week / 100).round(2)

    n_before = len(legs)
    legs = _filter_to_cohort(legs, pre_covid)
    logging.info(f"Cohort filter pre_covid={pre_covid}: {n_before:,} -> {len(legs):,} "
                 f"samples ({legs['participant_id'].nunique():,} participants)")

    legs['sd_bool_weekday'] = (legs['started_at'].dt.dayofweek < 5).astype(int)
    # Vectorized round-to-hour and hour extraction.
    legs['hour_of_day'] = legs['started_at'].dt.round('h').dt.hour
    standardize.attach_hour_flags(legs)

    legs['time_measured'] = legs['duration'].round().astype(int)
    legs['dist_measured'] = legs['length'].round().astype(int)
    legs['speed_kmh'] = legs['length'] * 3.6 / legs['duration']
    legs = common.add_group_id(legs, 'participant_id')

    # Warn on raw MOBIS mode values missing from `_MOBIS_MODE_MAP` — they
    # pass through untranslated and would either bias downstream (if
    # they collide with a training-mode simplified value) or clutter the
    # output CSV. Add explicit entries when they appear.
    unmapped = set(legs['mode'].dropna().unique()) - set(_MOBIS_MODE_MAP)
    if unmapped:
        counts = legs['mode'].value_counts()
        preview = {m: int(counts.get(m, 0)) for m in sorted(unmapped)}
        logging.warning(
            f"{len(unmapped)} raw MOBIS mode value(s) missing from "
            f"_MOBIS_MODE_MAP — passing through untranslated: {preview}")
    legs['mode_str'] = legs['mode'].replace(_MOBIS_MODE_MAP)
    # Date-conditional override: post-split Mode::Bicycle rows are
    # confirmed mechanical → upgrade from 'anybike' to 'rbike'.
    post_split_bicycle = (
        (legs['mode'] == 'Mode::Bicycle')
        & (legs['year_month'] >= _MOBIS_EBICYCLE_SPLIT_YM)
    )
    legs.loc[post_split_bicycle, 'mode_str'] = 'rbike'
    # No further collapse — mode_simplified retains the specific label
    # (anybike / rbike / anyebike / walk / car / …). Downstream chooses
    # which subset to include; see 04_edge_weights.py bike filters.
    legs['mode_simplified'] = legs['mode_str']
    legs['n_legs'] = 1
    legs['weight_person'] = 1

    car = legs['mode_str'] == 'car'
    if car.any():
        logging.info(f"Peak (car):  {legs.loc[car, 'hour_peak'].sum() / car.sum() * 100:.1f}%")
        logging.info(f"Night (car): {legs.loc[car, 'hour_night'].sum() / car.sum() * 100:.1f}%")

    f_dist_time = (legs['dist_measured'] <= 0) | (legs['time_measured'] <= 0) | legs['trip_id'].isna()
    f_implausible = legs['implausible'] == True  # noqa: E712 — keep explicit for clarity on Series
    f_outside_ch = legs['in_switzerland'] != True  # noqa: E712
    # Land-based: excludes air / water modes (matches MZMV's semantic).
    # Depends on the extended `_MOBIS_MODE_MAP` above mapping the
    # non-land raw modes to short canonical names.
    non_land_modes = ('plane', 'boat', 'aerialway')
    legs['is_land_based'] = (~legs['mode_str'].isin(non_land_modes)).astype(int)
    # Speed + detour envelopes: flags only. Downstream (04_edge_weights,
    # 07a_road_overhead_coefs, 09a_utility_estimation) filter on
    # `is_within_speed_envelope==1 & is_within_detour_envelope==1`
    # alongside `is_valid_domestic_trip==1`. Detour envelope needs
    # `dist_line` (attached below via `attach_lat_lon_and_dist_line`),
    # so the detour call happens after that.
    standardize.attach_speed_envelope_flag(legs)
    # Per-reason validity flags + composite `is_valid_domestic_trip`.
    # Order sets the cumulative log breakdown; total is order-independent.
    standardize.attach_validity_flags(legs, validity_checks={
        'is_valid_dist_time': f_dist_time,
        'is_plausible': f_implausible,
        'is_in_switzerland': f_outside_ch,
    })
    valid = legs['is_valid_domestic_trip'] == 1
    for mode, n in legs.loc[valid, 'mode_str'].value_counts().items():
        logging.info(f"   -> {mode}: {n:,}")

    legs = legs.rename(columns={
        'start_x': 'orig_x', 'end_x': 'dest_x',
        'start_y': 'orig_y', 'end_y': 'dest_y',
    })
    legs = standardize.attach_lat_lon_and_dist_line(legs)
    # Now `dist_line` exists — attach the detour envelope flag.
    standardize.attach_within_detour_envelope_flag(legs)
    # Sample the Swiss DEM at each leg's endpoints — supplies
    # `elev_orig` / `elev_dest` for downstream feature engineering,
    # alpine-trip filtering, and person-environment analyses.
    dem_ctx = context.source('preparation/world/elevation', storage=Storage.PUBLIC)
    dem_path = dem_ctx.path_for(Storage.PUBLIC, 'dem_switzerland.tif')
    standardize.attach_endpoint_elevation(legs, dem_path=dem_path)
    # Composite elevation gate (data-quality + alpine exclusion) —
    # single source of truth consumed by 04, 07, 02d.
    standardize.attach_within_elevation_band_flag(legs)
    # TODO(zones-migration): both calls below need the endpoint zone spatial
    # join that isn't migrated yet. Re-enable together with the join.
    # legs = common.add_zone_columns_for_endpoints(legs, zones, crs_main)
    # legs = common.add_swiss_city_subsets_trips(legs)

    legs = standardize.filter_columns_by_prefix(legs)

    # Vectorized id composition (was a row-apply with f-string).
    legs['id_column'] = (legs['participant_id'].astype(str) + '__' +
                        legs['leg_id'].astype(str))
    leg_id_to_new_id = dict(zip(legs['leg_id'], legs['id_column']))
    legs = legs.set_index('id_column')
    n_before = len(legs)
    legs = legs[~legs.index.duplicated(keep='first')]
    if len(legs) < n_before:
        logging.warning(
            f"{n_before - len(legs):,} duplicate trajectories removed (of {n_before:,}).")
    legs = legs.join(traj_df, on='leg_id')
    legs['n_legs_in_trip'] = legs.groupby('trip_id')['n_legs'].transform('count')

    legs = legs.rename(columns={'geometry_size': 'route_geo_size'})
    standardize.attach_include_in_route_matching(legs)
    n_routed = legs['include_in_route_matching'].sum()
    logging.info(f"...containing route: {n_routed:,} ({n_routed/len(legs)*100:.1f}%)")

    context.create_generic(legs, f'{survey_name}/legs.csv')
    context.create_generic(legs.iloc[:1000], f'{survey_name}/legs.sample.csv')

    routes = traj_routes[traj_routes.index.isin(legs['leg_id'])]
    routes.index = routes.index.map(leg_id_to_new_id)
    context.create_generic(routes.to_dict(), f'{survey_name}/legs.routes.measured.npy')


def main(variant) -> None:
    context = init_context(variant)
    # TODO(zones-migration): zones / lfi_regions data isn't migrated to the new
    # preparation tree yet. When `preparation/switzerland/zones` lands, switch the
    # `.source(...)` namespace below to point at it.
    # zones_src = context.source('preparation/switzerland/zones')
    # zones = zones_src.get_properties('zones', ['core', 'relations'], add_shapes=True)

    traj_routes = context.get_generic('mobis_all/trajectories.npy')
    traj_routes = pd.Series(traj_routes, name='route')
    traj_df = context.get_generic('mobis_all/trajectories.csv')

    process(context, variant.pre_covid, None, traj_routes, traj_df)
    context.close()


if __name__ == '__main__':
    variants.run(main)
