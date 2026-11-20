"""
Prepare survey trip legs for the Aperta Accessibility Atlas — takes a
standardized survey source and produces a scenario-scoped legs table for
the downstream survey scripts (05, 08a–c) and the calibration stages.

One variant per leg set (`main.common.SURVEY_LEG_SETS`):
  - `mtmc` (default): MZMV 2015 + 2021 → `survey_legs.csv`, the legs read by 03a.
  - `mtmc_trips`: the same legs as door-to-door trips (`main.common.
    survey_trips`: main mode plus walking, or transit plus feeder legs) →
    `survey_trips.csv`, read (with its survey/05 and 08a outputs) by 07a/b,
    09a and the validation of the times. Each trip is snapped and
    located at its ends, like a leg. Its legs are filtered by scope only (in
    Switzerland, a modeled mode); plausibility is checked on the whole trip
    (`main.common.TRIP_QUALITY_FLAGS`), not on each leg as for the legs.
  - `mobis_precovid` / `mobis_covid`: validation-only MOBIS GPS legs → a
    fixed-seed random sample of 100k legs in `survey_legs_<leg_set>.csv`.
    Keep the cohorts apart: precovid is normal traffic (and car's training
    cohort); covid is an anomalous traffic regime but the only one with
    confirmed regular-bike legs.

Concatenates the source vintages (MZMV with a `year` column), filters to
mode-choice-model rows, then enriches each with:

- Per mode (walk/bike/car): nearest snap-eligible node + snap distance
  for both origin AND destination → 4 columns per mode.
- Cell + zone assignment for both endpoints → 4 columns. Endpoints
  outside every cell / zone (beyond a small nearest-neighbour grace
  radius) get NaN.

A final completeness filter drops trips where any endpoint snap failed
OR any cell/zone id is NaN — makes downstream (survey/05) simpler
(no per-mode snap-presence checks). Uses the same graphs 02a saves,
so survey snaps land on the same SCC subset that cell snaps did (via
per-mode `cost_excluded_<mode>` flags).

Inputs (PRIVATE, under `preparation/switzerland/surveys/`):
    mzmv_2015/legs.csv, mzmv_2021/legs.csv   # mtmc
    mobis_precovid/legs.csv                  # mobis_precovid
    mobis_covid/legs.csv                     # mobis_covid

Inputs (scenario storage, under `<scenario>/`):
    nw/<mode>.graphml + properties/edges_<mode>_core.csv
    shapes/cells.gpkg, shapes/zones.gpkg

Output (PRIVATE, under `<scenario>/`):
    generic/survey_legs.csv                  # mtmc
    generic/survey_trips.csv                 # mtmc_trips
    generic/survey_legs_<leg_set>.csv        # MOBIS leg sets

Run (default variant `mtmc`):
    python -m survey.02d_prepare_survey_legs --scenario <name>
Trip set, or a MOBIS validation leg set:
    python -m survey.02d_prepare_survey_legs --scenario <name> --variant mtmc_trips
    python -m survey.02d_prepare_survey_legs --scenario <name> --variant mobis_precovid
"""

import logging

import geopandas as gpd
import pandas as pd

from aperta import geo_mapping, network_snap, routing_prep
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from main.common import MOBIS_LEG_SETS, SURVEY_LEG_SETS, TRIP_SET, survey_file, survey_trips
from preparation.switzerland.private.surveys.common import MODES_MODE_CHOICE_MODEL
from scenarios import get_scenario, scenario_needs_survey_prep


_MODES = ['walk', 'bike', 'car']

# MOBIS leg sets are validation-only: a fixed-seed random sample keeps survey/05's routing cheap.
_MOBIS_SAMPLE_N = 100_000
_MOBIS_SAMPLE_SEED = 0

# Uniform snap radius across modes. Generous enough to catch most
# residential origins; endpoints beyond it get NaN node_id + snap_dist for
# that mode (dropped by the completeness filter). Raised to the scenario's
# cell snap radius where that is larger (sparse networks).
_MAX_SNAP_RADIUS = 500.0

# Snap-eligibility setup, as in 02a.
_DIRECTEDNESS = 'directed_scc'

# Columns kept from each year's legs.csv. `sd_*` numeric columns are
# discovered + appended programmatically (varies by survey year).
# `is_within_{speed,detour,gap}_envelope` are kept, not filtered on: each
# consumer (03a, 07a/b, 09a, the validation scripts) applies the ones it needs.
_KEEP_COLS = [
    'orig_x', 'orig_y', 'dest_x', 'dest_y',
    'dist_measured', 'time_measured',
    'dist_line',
    'mode_simplified',
    'peak_str',
    'hour_peak', 'hour_night', 'hour_base',
    'weight_person',
    'trip_id',
    'n_legs_in_trip',
    'is_within_speed_envelope',
    'is_within_detour_envelope',
    'is_within_gap_envelope',
    'is_valid_dist_time',          # main-mode legs of the trip set (main.common.survey_trips)
    # Local clock times: trip-level door-to-door time incl. waiting between legs
    # (main.common.survey_trips) and timetable-based transit routing (survey/08d).
    'datetime_departure', 'datetime_arrival',
    # Conditions at departure (standardize.attach_daylight_and_season); not used by the
    # models yet.
    'is_daylight', 'is_late_night', 'season',
]
# Kept for the trip set only (MTMC legs; `survey_trips` takes them from the main leg).
_TRIP_KEEP_COLS = ['mode_is_passenger']


def _load_legs(surveys_ctx, survey_name: str) -> pd.DataFrame:
    """Read one source's legs.csv from PRIVATE, indexed by `leg_id`. MZMV vintages are tagged
    with an int `year`."""
    legs = surveys_ctx.get_generic(
        f'{survey_name}/legs.csv', storage=Storage.PRIVATE)
    if survey_name.startswith('mzmv_'):
        legs['year'] = int(survey_name.split('_')[-1])
    return legs


def _points_from_xy(df: pd.DataFrame, x_col: str, y_col: str, crs: str) -> gpd.GeoDataFrame:
    """GeoDataFrame of points preserving `df.index`. `crs` is the
    scenario's `crs_main` since `orig_x`/`orig_y`/`dest_x`/`dest_y` are
    pre-projected to it upstream (LV95 for Swiss scenarios)."""
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        index=df.index,
        crs=crs,
    )


def main(variant):
    context = init_context(variant)
    scenario = get_scenario(context.scenario)
    if not scenario_needs_survey_prep(scenario):
        logging.info(
            f"Scenario {scenario.name!r} has no survey-driven calibrators — "
            f"skipping (survey prep outputs would have no consumer).")
        context.close()
        return
    crs_main = scenario.crs_main
    leg_set = variant.leg_set
    sources = SURVEY_LEG_SETS[leg_set]

    # -------- Load + concat survey legs -----------------------------------
    with step(f'load + concat {leg_set} legs ({", ".join(sources)})'):
        surveys_ctx = context.source('preparation/switzerland/surveys', storage=Storage.PRIVATE)
        per_source = [_load_legs(surveys_ctx, s) for s in sources]
        # `verify_integrity=True` asserts leg_id uniqueness across MZMV years
        # (get_leg_id namespaces with `year`, so this is safe).
        legs = pd.concat(per_source, verify_integrity=True)
        by_year = legs.groupby('year').size().to_dict() if 'year' in legs.columns else {}
        logging.info(f"  → {len(legs):,} legs total" + (f" ({by_year})" if by_year else ""))

    with step('filter to valid legs in mode-choice mode set'):
        before = len(legs)
        # `is_land_based` is defense-in-depth: the mode filter already
        # excludes plane / boat / aerialway, but taxonomy changes can't
        # sneak non-land modes through. `is_within_elevation_band` gates
        # on DEM validity + alpine exclusion (standardize.py), as in 04.
        # Envelope flags are kept, not filtered on (see `_KEEP_COLS`). The trip
        # set filters its legs by scope only: a leg's distance, time or
        # elevation does not invalidate the trip (`main.common.survey_trips`).
        keep = ((legs['is_land_based'] == 1)
                & legs['mode_simplified'].isin(MODES_MODE_CHOICE_MODEL))
        if leg_set == TRIP_SET:
            keep &= legs['is_in_switzerland'] == 1
        else:
            keep &= ((legs['is_valid_domestic_trip'] == 1)
                     & (legs['is_within_elevation_band'] == 1))
        legs = legs.loc[keep].copy()
        logging.info(f"  → {len(legs):,} / {before:,} legs kept")

    if leg_set in MOBIS_LEG_SETS and len(legs) > _MOBIS_SAMPLE_N:
        with step(f'random sample of {_MOBIS_SAMPLE_N:,} legs (validation-only leg set)'):
            legs = legs.sample(n=_MOBIS_SAMPLE_N, random_state=_MOBIS_SAMPLE_SEED)
            logging.info(f"  → modes: {legs['mode_simplified'].value_counts().to_dict()}")

    with step('select columns (incl. numeric sd_* columns)'):
        sd_cols = [
            c for c in legs.columns
            if c.startswith('sd_') and pd.api.types.is_numeric_dtype(legs[c])
        ]
        cols = _KEEP_COLS + sd_cols + (['year'] if 'year' in legs.columns else [])
        if leg_set == TRIP_SET:
            cols += _TRIP_KEEP_COLS
        missing = [c for c in cols if c not in legs.columns]
        if missing:
            raise KeyError(f"Expected columns missing from legs: {missing}")
        legs = legs[cols]
        logging.info(f"  → {len(legs.columns)} columns "
                     f"({len(sd_cols)} sd_* numeric)")

    if leg_set == TRIP_SET:
        # From here on, each row is a trip: snapped and located at its ends.
        with step('aggregate legs → door-to-door trips (main.common.survey_trips)'):
            legs = survey_trips(legs)

    endpoints = {e: _points_from_xy(legs, f'{e}_x', f'{e}_y', crs_main)
                 for e in ('orig', 'dest')}

    # -------- Per-mode snap (orig + dest) ---------------------------------
    max_snap_radius = max(_MAX_SNAP_RADIUS, scenario.snap.max_cell_radius_m)
    modes = [m for m in _MODES if m in scenario.mode_configs]
    for mode in modes:
        with step(f'mode={mode}: load graph + compute snap-eligible nodes'):
            # `core` edge properties carry 02a's `cost_excluded_<mode>` flags.
            graph = context.get_nw(
                data_name=mode,
                add_node_properties='core',
                add_edge_properties='core',
            )
            snap_eligible = routing_prep.compute_snap_eligible_nodes(
                graph, directedness=_DIRECTEDNESS,
                cost_excluded_flag=f'cost_excluded_{mode}',
            )
            logging.info(
                f"  → graph {graph.number_of_nodes():,} nodes, "
                f"snap-eligible {len(snap_eligible):,} "
                f"({100*len(snap_eligible)/graph.number_of_nodes():.1f} %)")

        for endpoint, pts in endpoints.items():
            with step(f'mode={mode}: snap {endpoint}'):
                ids, dists = network_snap.snap_to_network_nodes(
                    pts, graph,
                    max_distance=max_snap_radius,
                    eligible_node_ids=snap_eligible,
                )
                legs[f'{endpoint}_node_id_{mode}']  = ids.values
                legs[f'{endpoint}_snap_dist_{mode}'] = dists.values
                logging.info(f"  → matched {ids.notna().sum():,}/{len(legs):,}, median dist {dists.median():.1f} m")

    # -------- Cell + zone assignment (orig + dest) ------------------------
    with step('load cells + zones'):
        cells = context.get_shapes('cells').to_crs(crs_main)
        zones = context.get_shapes('zones').to_crs(crs_main)
        logging.info(f"  → {len(cells):,} cells, {len(zones):,} zones")

    for endpoint, pts in endpoints.items():
        with step(f'{endpoint}: assign cell + zone (within only; foreign → NaN)'):
            cell_ids, _ = geo_mapping.map_points_to_polygons(pts, cells, allow_nearest=True,
                                                             max_distance=500)
            zone_ids, _ = geo_mapping.map_points_to_polygons(pts, zones, allow_nearest=True,
                                                             max_distance=2000)
            legs[f'{endpoint}_cell_id'] = cell_ids.values
            legs[f'{endpoint}_zone_id'] = zone_ids.values
            logging.info(
                f"  → {cell_ids.notna().sum():,}/{len(legs):,} matched to cells, "
                f"{zone_ids.notna().sum():,}/{len(legs):,} matched to zones")

    # Completeness filter: drop trips with ANY NaN mode-snap node id or
    # cell/zone id, so every remaining leg is routable on every mode and
    # located in the atlas (survey/05 needs no per-mode snap checks).
    # Drops are reported per cause.
    with step('drop trips with missing snap / cell-zone (two-batch report)'):
        before = len(legs)
        node_cols = [f'{e}_node_id_{m}' for e in ('orig', 'dest') for m in modes]
        network_missing = legs[node_cols].isna().any(axis=1)
        cell_zone_missing = (
            legs['orig_cell_id'].isna() | legs['dest_cell_id'].isna()
            | legs['orig_zone_id'].isna() | legs['dest_zone_id'].isna()
        )
        n_network    = int(network_missing.sum())
        n_cell_zone  = int((cell_zone_missing & ~network_missing).sum())
        legs = legs.loc[~network_missing & ~cell_zone_missing].copy()
        logging.info(
            f"  → dropped {n_network:,} (network node not found) "
            f"+ {n_cell_zone:,} (cell/zone not found) "
            f"= {before - len(legs):,} of {before:,} "
            f"({100*(before - len(legs))/max(before, 1):.2f} %); "
            f"{len(legs):,} kept")

    out_name = survey_file('survey_legs.csv', leg_set)
    with step(f'save PRIVATE/generic/{out_name}'):
        context.create_generic(legs, out_name, storage=Storage.PRIVATE)
        logging.info(
            f"  → saved {len(legs):,} legs × {len(legs.columns)} cols")

    context.close()


variants = Variants([('leg_set', str)])
for _leg_set in SURVEY_LEG_SETS:
    variants.add(name=_leg_set, leg_set=_leg_set)


if __name__ == '__main__':
    variants.run(main, default='mtmc')
