"""
Trip-level transit-time predictions for validation.

For every survey TRIP that contains at least one transit leg, aggregate
the trip's constituent legs, then produce a door-to-door transit
prediction as `α × NPVM z2z(trip_orig_zone, trip_dest_zone) +
overheads_transit(trip_orig_cell) + overheads_transit(trip_dest_cell)`.

Why trip-level (not leg-level) for transit: NPVM z2z is door-to-door
(includes walk-to-stop and walk-from-stop), while MTMC's per-leg
`time_measured` for a transit leg covers only the ride (stop-to-stop),
so a leg-level comparison is biased (z2z ≈ +47 % vs measured). The
trip's measured time (first departure → last arrival) is door-to-door.

Access-mode context columns are added so downstream validation can
stratify by walk-only / park-and-ride / bike-and-ride patterns.

Legs within a trip are ordered by `leg_id` (the index of
`survey_legs.csv`, `<year>_<HHNR>_<ETNR>` from survey prep); the first
leg's origin and the last leg's destination are the trip endpoints
(`main.common.aggregate_transit_trips`).

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_legs.csv                      # from survey/02d (needs trip_id)

Inputs (scenario storage, under `<scenario>/`):
    properties/cells_{population,employment,snap,pois,transit_access}.csv
    shapes/cells.gpkg
    coefs/<kind>/overheads_transit.csv           # from main/07b (calibrated in switzerland-h10)

Inputs (external): NPVM z2z table via `main.common.npvm_transit_z2z_lookup`.

Output (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit.csv              # one row per transit trip

Output columns:
    trip_id (index)
    trip_time_measured                           # last arrival − first departure (s;
                                                 # door to door incl. waiting)
    trip_time_legs, trip_dist_measured           # sum across legs (seconds, metres)
    is_within_gap_envelope                       # survey prep (standardize.py)
    trip_dist_line                               # straight-line trip_orig → trip_dest (metres)
    trip_orig_{x,y}, trip_dest_{x,y}             # first-leg orig, last-leg dest
    trip_orig_cell, trip_dest_cell               # cell ids for overhead lookup
    trip_orig_zone, trip_dest_zone               # zone ids for NPVM z2z lookup
    weight_person                                # first-leg weight (constant within trip)
    peak_str                                     # first-leg peak label
    datetime_departure                           # first-leg departure (MZMV legs only)
    n_legs_in_trip, n_transit_legs
    has_walk_access, has_car_access, has_bike_access   # 0/1
    access_pattern                               # walk_only / park_and_ride / bike_and_ride / mixed
    is_within_speed_envelope, is_within_detour_envelope
                                                 # min across legs (0 if ANY leg fails)
    pred_z2z                                     # NPVM z2z(trip_orig_zone, trip_dest_zone)
    pred_overhead_orig, pred_overhead_dest       # cell-level overheads_transit at trip endpoints
    pred_time                                    # α × pred_z2z + orig_ov + dest_ov
                                                 # (α from overheads_transit's `t_routed` row,
                                                 #  = 1.0 if the coefs are constrained-α)

Run:
    python -m survey.08b_trip_transit_times --scenario <name>
"""

import logging

import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step

from main.common import (
    aggregate_transit_trips,
    npvm_transit_z2z_lookup,
    per_cell_transit_overhead,
    route_time_alpha,
)
from scenarios import get_scenario, scenario_needs_survey_prep


def _predict_transit(trips: pd.DataFrame, context) -> pd.DataFrame:
    """Add pred_z2z, pred_overhead_{orig,dest}, pred_time columns."""
    with step('NPVM z2z lookup per trip'):
        lookup = npvm_transit_z2z_lookup(context)
        keys = list(zip(trips['trip_orig_zone'], trips['trip_dest_zone']))
        trips['pred_z2z'] = pd.Series(
            [lookup.get(k, float('nan')) for k in keys],
            index=trips.index, dtype=float)
        n_hit = int(trips['pred_z2z'].notna().sum())
        logging.info(
            f"  → {n_hit:,}/{len(trips):,} trips with NPVM z2z entry "
            f"({100*n_hit/max(len(trips),1):.1f} %); "
            f"median pred_z2z = {trips['pred_z2z'].median():.0f} s")

    with step('per-cell transit overhead lookup at trip endpoints'):
        transit_coefs = context.get_coefs('overheads_transit')
        cells = context.get_properties(
            'cells',
            ['population', 'employment', 'snap', 'pois', 'transit_access'],
            add_shapes=True)
        per_cell_ov = per_cell_transit_overhead(cells, transit_coefs)
        trips['pred_overhead_orig'] = trips['trip_orig_cell'].map(per_cell_ov).astype(float)
        trips['pred_overhead_dest'] = trips['trip_dest_cell'].map(per_cell_ov).astype(float)
        n_ov = int((trips['pred_overhead_orig'].notna()
                    & trips['pred_overhead_dest'].notna()).sum())
        logging.info(
            f"  → {n_ov:,}/{len(trips):,} trips with both endpoint overheads; "
            f"mean orig+dest overhead = "
            f"{(trips['pred_overhead_orig'] + trips['pred_overhead_dest']).mean():.1f} s")

    # α (the `t_routed` row of overheads_transit) scales z2z directly: the
    # trip-level counterpart of survey/08a's per-leg `t_routed_bias_transit`.
    alpha_transit = route_time_alpha(transit_coefs['transit'])
    logging.info(
        f"  → α (t_routed slope) = {alpha_transit:.4f} applied to pred_z2z")
    trips['pred_time'] = (
        alpha_transit * trips['pred_z2z']
        + trips['pred_overhead_orig'].fillna(0.0)
        + trips['pred_overhead_dest'].fillna(0.0))
    return trips


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    if scenario.transit_source is None:
        logging.info(f"Scenario {scenario.name!r} has no transit; skipping.")
        context.close()
        return
    if not scenario_needs_survey_prep(scenario):
        logging.info(
            f"Scenario {scenario.name!r} has no survey-driven calibrators — "
            f"skipping (survey prep outputs would have no consumer).")
        context.close()
        return

    with step('load survey_legs.csv'):
        legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
        if 'trip_id' not in legs.columns:
            raise KeyError(
                "Column 'trip_id' not in survey_legs.csv. "
                "Verify it's in 02d's `_KEEP_COLS` and rerun 02d.")
        logging.info(f"  → {len(legs):,} legs")

    with step('filter to transit-containing trips + aggregate to trip rows'):
        trips = aggregate_transit_trips(legs)
        n_by_pattern = trips['access_pattern'].value_counts().to_dict()
        logging.info(
            f"  → {len(trips):,} transit trips; access patterns: {n_by_pattern}")

    with step('predict per-trip: z2z + endpoint overheads'):
        trips = _predict_transit(trips, context)

    with step('save survey_trip_transit.csv'):
        context.create_generic(trips, 'survey_trip_transit.csv',
                               storage=Storage.PRIVATE,
                               kws={'float_format': '%.3f'})
        logging.info(f"  → saved {len(trips):,} trips × {len(trips.columns)} cols")

    context.close()


if __name__ == '__main__':
    main()
