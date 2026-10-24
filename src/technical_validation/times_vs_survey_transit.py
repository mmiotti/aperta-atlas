"""
Compare atlas-predicted TRIP-LEVEL transit times against MTMC-observed
door-to-door totals.

Complements `technical_validation/times_vs_survey.py`, which is leg-level and
walk/bike/car-shaped. Transit z2z is inherently door-to-door (includes
walk-to-stop + walk-from-stop), so the correct comparator is a
TRIP-level total (Σ leg time_measured), not a per-leg time. This script
reads `survey_trip_transit.csv` (from `survey/08b_trip_transit_times.py`)
and reports the same statistics as `times_vs_survey.py`.

Sample scope: ~37 k trips (per 2026-09 leg-composition run) that
contain at least one transit leg — far larger than the ~280 single-leg
transit legs surviving the `n_legs_in_trip == 1` workaround in
`times_vs_survey.py`.

Quantities:
  - `time` = α × pred_z2z + pred_overhead_orig + pred_overhead_dest
             (α from overheads_transit's `t_routed` row)
  - `time_cv_year` = `time`, but with the adjustment fitted on the OTHER
             MTMC survey wave (2015 ↔ 2021; same model and fit as 07b,
             applied as in 08b): an out-of-sample check of `time`, whose
             production adjustment is fitted on these trips.
  - `z2z`  = pred_z2z alone (raw, no α + no overhead adjustment) —
             comparing time vs z2z rows attributes the transit residual
             to the α + endpoint corrections vs. raw NPVM lookup.
  - `r5_<v>_p<P>` = timetable-based R5 times from survey/08d, unfitted:
             `d2d` door to door, `z2z` zone point to zone point (cf. `z2z`);
             p10 ≈ departure timed to the timetable, p50 = random departure
             (30-min window). Only when `survey_trip_transit_r5.csv` exists;
             08d's other variants (p1, the composed stop-based `k3a` / `k3b`)
             are left out of the summary (`_R5_QUANTITIES`).

Two samples (`sample` column): `all` = `time` / `z2z` on every
plausibility-filtered trip (the headline validation); `r5_subset` = all
quantities incl. R5 on the trips with every prediction available (the R5
sample, endpoints beyond walking reach), so rows within it are comparable.

Rows, kept to what the data descriptor and the WCTR paper report plus some
context (`band`: straight-line trip distance, <2 km, 2-5 km, 5-25 km, >25 km):
  - `all`: `time`, `time_cv_year` and `z2z` for all trips and per distance
    band; `time` per access pattern (walk_only / park_and_ride /
    bike_and_ride).
  - `r5_subset`: every quantity on all trips of the R5 sample (the
    descriptor's R5 table); `time` and R5 door to door also per distance
    band (the sample has no trips < 2 km).

R²_1to1 is R² relative to the 1:1 line (`1 − SS_res / SS_tot`), R²_fit relative
to the best-fit line (= squared Pearson), as in `times_vs_survey.py`.

Plausibility filter: keeps trips where all constituent legs passed
`is_within_speed_envelope` and `is_within_detour_envelope` (aggregated
as `min` in 08b).

Inputs (PUBLIC, under `<scenario>/`):
    properties/cells_transit_access.csv   # 06 — zone-dev features for `time_cv_year`
Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit.csv    # from survey/08b_trip_transit_times (needs
                                       # `datetime_departure` for the survey wave)
    generic/survey_trip_transit_r5.csv # from survey/08d_trip_transit_r5 (optional)

Outputs:
    RESULTS/times_vs_survey_transit.csv
    Logs

Run:
    python -m technical_validation.times_vs_survey_transit --scenario <name>
"""

import argparse
import importlib
import logging

import numpy as np
import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step

from main.common import TRIP_QUALITY_FLAGS, per_cell_transit_overhead, route_time_alpha
from technical_validation.common import agreement_stats

_m07b = importlib.import_module('main.07b_transit_overhead_coefs')
_WAVE_SPLIT_YEAR = 2018   # departures before → MTMC 2015 wave, from → 2021 wave


_DIST_BANDS: list[tuple[str, float, float]] = [
    ('   <2km',       0.0,    2_000.0),
    ('   2-5km',  2_000.0,    5_000.0),
    ('  5-25km',  5_000.0,   25_000.0),
    ('   >25km', 25_000.0,  np.inf),
]


# R5 variants in the summary (08d also computes p1 and the composed `k3a` / `k3b` models).
_R5_QUANTITIES: tuple[str, ...] = ('r5_d2d_p10', 'r5_d2d_p50', 'r5_z2z_p10', 'r5_z2z_p50')

# Order for the summary table.
_ACCESS_PATTERN_ORDER: dict[str, int] = {
    'ALL':             0,
    'walk_only':       1,
    'park_and_ride':   2,
    'bike_and_ride':   3,
    'mixed':           4,
}


def main():
    argparse.ArgumentParser(
        description="Compare predicted vs measured TRIP-LEVEL transit times.",
    ).parse_known_args()

    context = init_context()

    with step('load survey_trip_transit.csv'):
        trips = context.get_generic(
            'survey_trip_transit.csv', storage=Storage.PRIVATE)
        logging.info(f"  → {len(trips):,} transit trips")

    with step('quality filter (both endpoint flags == 1)'):
        n_before = len(trips)
        for flag in TRIP_QUALITY_FLAGS:
            if flag in trips.columns:
                trips = trips[trips[flag] == 1]
        logging.info(f"  → {len(trips):,} / {n_before:,} trips kept")

    with step('year-wise cross-validation of the transit adjustment (time_cv_year)'):
        trips['pred_time_cv'] = _cross_validated_prediction(context, trips)
    base = {'time': 'pred_time', 'time_cv_year': 'pred_time_cv', 'z2z': 'pred_z2z'}
    with step('stats on all plausibility-filtered trips (sample = all)'):
        rows = _stats_rows(trips, base, 'all', band_quantities=tuple(base),
                           access_quantities=('time',))

    try:
        r5 = context.get_generic('survey_trip_transit_r5.csv', storage=Storage.PRIVATE)
    except FileNotFoundError:
        r5 = None
        logging.info("  → no survey_trip_transit_r5.csv — R5 rows skipped")
    if r5 is not None:
        with step('R5 comparison on the common trip subset (sample = r5_subset)'):
            r5.index = r5.index.astype(str)
            sub = trips.copy()
            sub.index = sub.index.astype(str)
            r5_cols = [f't_{q}' for q in _R5_QUANTITIES if f't_{q}' in r5.columns]
            sub = sub.join(r5[r5_cols], how='inner')
            quantities = {**base, **{c.removeprefix('t_'): c for c in r5_cols}}
            n_before = len(sub)
            sub = sub.dropna(subset=list(quantities.values()))
            logging.info(f"  → {len(sub):,} / {n_before:,} R5-sample trips with all predictions")
            r5_d2d = [q for q in ('r5_d2d_p10', 'r5_d2d_p50') if q in quantities]
            rows += _stats_rows(sub, quantities, 'r5_subset', band_quantities=('time', *r5_d2d),
                                access_quantities=())

    with step('summary'):
        _log_and_save(context, rows)
    context.close()


def _cross_validated_prediction(context, trips: pd.DataFrame) -> pd.Series:
    """Transit time with the adjustment fitted on the other survey wave: 07b's
    fit (`_fit_calibrated_transit_overhead`, production column `transit`) on
    one wave's trips, applied to the other's as 08b does (α × z2z + per-cell
    overheads at both ends)."""
    year = pd.to_datetime(trips['datetime_departure'], errors='coerce').dt.year
    wave = pd.Series(np.where(year < _WAVE_SPLIT_YEAR, 2015, 2021),
                     index=trips.index).where(year.notna())
    cells = context.get_properties('cells', 'transit_access')
    fit_input = trips.assign(raw_z2z=trips['pred_z2z'])
    pred = pd.Series(np.nan, index=trips.index)
    for test in (2015, 2021):
        train = fit_input[wave.notna() & (wave != test) & fit_input['raw_z2z'].notna()]
        coefs = _m07b._fit_calibrated_transit_overhead(train, cells)
        overhead = per_cell_transit_overhead(cells, coefs)
        alpha = route_time_alpha(coefs['transit'])
        test_rows = wave == test
        pred[test_rows] = (alpha * trips.loc[test_rows, 'pred_z2z']
                           + trips.loc[test_rows, 'trip_orig_cell'].map(overhead).fillna(0.0)
                           + trips.loc[test_rows, 'trip_dest_cell'].map(overhead).fillna(0.0))
        logging.info(f"  → {test} trips ({int(test_rows.sum()):,}) predicted with the "
                     f"adjustment fitted on the other wave ({len(train):,} trips; "
                     f"α = {alpha:.3f}, const = {coefs.loc['const_s', 'transit']:.0f} s)")
    return pred


def _stats_rows(trips: pd.DataFrame, quantities: dict[str, str], sample: str, *,
                band_quantities: tuple[str, ...], access_quantities: tuple[str, ...]) -> list[dict]:
    """Agreement stats for `quantities` (name → prediction column) on all trips; also per
    distance band for `band_quantities` and per access pattern for `access_quantities`."""
    meas = trips['trip_time_measured'].astype(float)
    dist_line = trips['trip_dist_line'].astype(float)
    every = pd.Series(True, index=trips.index)
    rows: list[dict] = []

    def add(quantity: str, mask: pd.Series, access: str = 'ALL', band: str = 'ALL') -> None:
        pred = trips[quantities[quantity]].astype(float)
        m = mask & pred.notna() & meas.notna() & (meas > 0)
        if m.any():
            rows.append({'sample': sample, 'access_pattern': access, 'quantity': quantity,
                         'band': band,
                         **agreement_stats(meas[m].to_numpy(dtype=float),
                                           pred[m].to_numpy(dtype=float))})

    for quantity in quantities:
        add(quantity, every)
        if quantity in band_quantities:
            for label, lo, hi in _DIST_BANDS:
                add(quantity, (dist_line >= lo) & (dist_line < hi), band=label)
    for quantity in access_quantities:
        for access in sorted(trips['access_pattern'].dropna().unique(),
                             key=lambda x: _ACCESS_PATTERN_ORDER.get(x, 99)):
            add(quantity, trips['access_pattern'] == access, access=access)
    return rows


def _log_and_save(context, rows: list[dict]) -> None:
    if not rows:
        logging.warning("no rows to summarize — check inputs.")
        return
    df = pd.DataFrame(rows)
    quantity_order = {q: i for i, q in enumerate(dict.fromkeys(df['quantity']))}
    sample_order = {'all': 0, 'r5_subset': 1}
    band_order = {'ALL': 0, **{label: i + 1 for i, (label, _, _) in enumerate(_DIST_BANDS)}}
    df['_sorder']  = df['sample'].map(sample_order).fillna(99)
    df['_aporder'] = df['access_pattern'].map(_ACCESS_PATTERN_ORDER).fillna(99)
    df['_qorder']  = df['quantity'].map(quantity_order).fillna(99)
    df['_border']  = df['band'].map(band_order).fillna(99)
    df = df.sort_values(['_sorder', '_aporder', '_qorder', '_border'], kind='stable').drop(
        columns=['_sorder', '_aporder', '_qorder', '_border']).reset_index(drop=True)

    df = df.rename(columns={
        'n': 'n_trips',
        'bias_pct': 'bias%', 'med_resid_pct': 'med_resid%',
        'r2': 'R²_1to1', 'r2_fit': 'R²_fit', 'mae_pct': 'MAE%',
    })
    display_cols = ['sample', 'access_pattern', 'quantity', 'band', 'n_trips',
                    'bias', 'bias%', 'med_resid', 'med_resid%', 'R²_1to1', 'R²_fit', 'MAE%']

    context.create_results(
        df[display_cols], 'technical_validation/times_vs_survey_transit.csv',
        kws={'index': False, 'float_format': '%.4f'},
    )
    formatters = {
        'n_trips': '{:>7,}'.format,
        'bias':    '{:+8.1f}'.format,
        'bias%':   '{:+.1f}%'.format,
        'med_resid':  '{:+8.1f}'.format,
        'med_resid%': '{:+.1f}%'.format,
        'R²_1to1': '{:.4f}'.format,
        'R²_fit':  '{:.4f}'.format,
        'MAE%':    '{:.1f}%'.format,
    }
    table = df[display_cols].to_string(index=False, formatters=formatters,
                                       justify='right')
    logging.info(
        "TRIP-LEVEL transit: predicted vs summed-measured, per "
        "(access_pattern × quantity × band).\n"
        "Quantity 'time' = NPVM z2z(trip endpoints) + hand-tuned "
        "overheads_transit at trip endpoints;\n"
        "'z2z'           = NPVM z2z alone (no overhead adjustment); "
        "bias(time)−bias(z2z) attributes residual to overheads.\n"
        "`bias > 0` = model OVER-predicts (mean); `med_resid > 0` = model above "
        "the median trip. Bands over trip_dist_line "
        "(straight-line trip origin → destination).\n"
        + table)


if __name__ == '__main__':
    main()
