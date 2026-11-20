"""
Trip-level transit-time predictions for validation.

Predicts the door-to-door transit time of the trips main/07b fits its overheads on (07b's
`fit_trips`: the trip set's transit trips, plausibility-filtered, within its distance
range and the AOI), as the released times are built: `α × NPVM z2z(orig_zone, dest_zone)
+ overheads_transit(orig_cell) + overheads_transit(dest_cell)`.

Why trip-level (not leg-level) for transit: NPVM z2z is door-to-door
(includes walk-to-stop and walk-from-stop), while MTMC's per-leg
`time_measured` for a transit leg covers only the ride (stop-to-stop),
so a leg-level comparison is biased (z2z ≈ +47 % vs measured). The
trip's measured time (first departure → last arrival) is door-to-door.

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_trips.csv                     # from survey/02d (--variant mtmc_trips)

Inputs (scenario storage, under `<scenario>/`):
    properties/cells_{population,employment,snap,pois,transit_access}.csv
    shapes/cells.gpkg
    coefs/<kind>/overheads_transit.csv           # from main/07b (calibrated in switzerland-h10)

Inputs (external): NPVM z2z table via `main.common.npvm_transit_z2z_lookup`.

Output (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit.csv              # one row per transit trip

Output columns (index `trip_id`): the trip set's (`main.common.survey_trips`)
    time_measured                                # last arrival − first departure (s;
                                                 # door to door incl. waiting)
    dist_line, orig_{x,y}, dest_{x,y}, {orig,dest}_cell_id, {orig,dest}_zone_id,
    datetime_departure, peak_str, weight_person, n_transit_legs,
    transit_feeder                               # walk / bike / car / bike_car
and
    raw_z2z                                      # NPVM z2z(orig_zone, dest_zone) (s)
    pred_overhead_orig, pred_overhead_dest       # cell-level overheads_transit at the ends
    pred_time                                    # α × raw_z2z + orig_ov + dest_ov
                                                 # (α from overheads_transit's `t_routed` row,
                                                 #  = 1.0 if the coefs are constrained-α)

Run:
    python -m survey.08b_trip_transit_times --scenario <name>
"""

import importlib
import logging

import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step

from main.common import per_cell_transit_overhead, route_time_alpha
from scenarios import get_scenario, scenario_needs_survey_prep

_m07b = importlib.import_module('main.07b_transit_overhead_coefs')

_COLUMNS: tuple[str, ...] = (
    'time_measured', 'dist_line', 'orig_x', 'orig_y', 'dest_x', 'dest_y',
    'orig_cell_id', 'dest_cell_id', 'orig_zone_id', 'dest_zone_id',
    'datetime_departure', 'peak_str', 'weight_person', 'n_transit_legs', 'transit_feeder',
    'raw_z2z',
)


def _predict_transit(trips: pd.DataFrame, context) -> pd.DataFrame:
    """Add pred_overhead_{orig,dest}, pred_time columns."""
    with step('per-cell transit overhead lookup at trip endpoints'):
        transit_coefs = context.get_coefs('overheads_transit')
        cells = context.get_properties(
            'cells',
            ['population', 'employment', 'snap', 'pois', 'transit_access'],
            add_shapes=True)
        per_cell_ov = per_cell_transit_overhead(cells, transit_coefs)
        trips['pred_overhead_orig'] = trips['orig_cell_id'].map(per_cell_ov).astype(float)
        trips['pred_overhead_dest'] = trips['dest_cell_id'].map(per_cell_ov).astype(float)
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
        f"  → α (t_routed slope) = {alpha_transit:.4f} applied to raw_z2z")
    trips['pred_time'] = (
        alpha_transit * trips['raw_z2z']
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

    trips = _m07b.fit_trips(context, scenario)[list(_COLUMNS)].copy()
    logging.info(f"  → transit feeders: {trips['transit_feeder'].value_counts().to_dict()}")

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
