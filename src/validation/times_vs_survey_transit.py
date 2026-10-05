"""
Compare atlas-predicted TRIP-LEVEL transit times against MTMC-observed
door-to-door totals.

Complements `validation/times_vs_survey.py`, which is leg-level and
walk/bike/car-shaped. Transit z2z is inherently door-to-door (includes
walk-to-stop + walk-from-stop), so the correct comparator is a
TRIP-level total (Σ leg time_measured), not a per-leg time. This script
reads `survey_trip_transit.csv` (from `survey/08b_trip_transit_times.py`)
and produces the same-shaped stats table as `times_vs_survey.py`.

Sample scope: ~37 k trips (per 2026-09 leg-composition run) that
contain at least one transit leg — far larger than the ~280 single-leg
transit legs surviving the `n_legs_in_trip == 1` workaround in
`times_vs_survey.py`.

Two quantities reported per (access_pattern × band):
  - `time` = α × pred_z2z + pred_overhead_orig + pred_overhead_dest
             (α from overheads_transit's `t_routed` row)
  - `z2z`  = pred_z2z alone (raw, no α + no overhead adjustment) —
             comparing time vs z2z rows attributes the transit residual
             to the α + endpoint corrections vs. raw NPVM lookup.

Distance bands from `trip_dist_line` (straight-line trip origin →
destination) matching `times_vs_survey.py`'s coarse strata: <5 km,
5-25 km, >25 km. Every access_pattern gets an ALL row.

Also stratifies by `access_pattern` (walk_only / park_and_ride /
bike_and_ride / mixed) and by `n_transit_legs` (single-transit vs.
≥ 2 transfers).

R² is the 1:1 coefficient of determination (`1 − SS_res / SS_tot`), as in
`times_vs_survey.py`: penalises bias and scale, < 0 = worse than the mean.

Plausibility filter: keeps trips where all constituent legs passed
`is_within_speed_envelope` and `is_within_detour_envelope` (aggregated
as `min` in 08b).

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit.csv    # from survey/08b_trip_transit_times

Outputs:
    RESULTS/times_vs_survey_transit.csv
    Logs

Run:
    python -m validation.times_vs_survey_transit --scenario <name>
"""

import argparse
import logging

import numpy as np
import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step


_DIST_BANDS: list[tuple[str, float, float]] = [
    ('   <5km',       0.0,    5_000.0),
    ('  5-25km',  5_000.0,   25_000.0),
    ('   >25km', 25_000.0,  np.inf),
]

# Order for the summary table.
_ACCESS_PATTERN_ORDER: dict[str, int] = {
    'ALL':             0,
    'walk_only':       1,
    'park_and_ride':   2,
    'bike_and_ride':   3,
    'mixed':           4,
}


def _stats(measured: np.ndarray, predicted: np.ndarray) -> dict:
    mask = np.isfinite(measured) & np.isfinite(predicted)
    n = int(mask.sum())
    if n < 2:
        return {'n': n, 'bias': np.nan, 'bias_pct': np.nan,
                'mae': np.nan, 'rmse': np.nan,
                'slope': np.nan, 'r2': np.nan, 'mae_pct': np.nan}
    m, p = measured[mask], predicted[mask]
    resid = p - m
    m_mean = float(m.mean())
    bias = float(resid.mean())
    m_c = m - m_mean
    p_c = p - float(p.mean())
    ss_m = float((m_c ** 2).sum())
    slope = float((m_c * p_c).sum() / ss_m) if ss_m > 0 else np.nan
    # 1:1 R² (`1 − SS_res / SS_tot`), as in `times_vs_survey` and `aperta.calibration`.
    r2 = 1.0 - float((resid ** 2).sum()) / ss_m if ss_m > 0 else np.nan
    mae = float(np.abs(resid).mean())
    m_abs_mean = float(np.abs(m).mean())
    mae_pct = 100.0 * mae / m_abs_mean if m_abs_mean > 0 else np.nan
    bias_pct = 100.0 * bias / m_mean if m_mean != 0 else np.nan
    return {
        'n': n, 'bias': bias, 'bias_pct': bias_pct,
        'mae': mae, 'rmse': float(np.sqrt((resid ** 2).mean())),
        'slope': slope, 'r2': r2, 'mae_pct': mae_pct,
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
        for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
            if flag in trips.columns:
                trips = trips[trips[flag] == 1]
        logging.info(f"  → {len(trips):,} / {n_before:,} trips kept")

    meas = trips['trip_time_measured'].astype(float)
    pred_time = trips['pred_time'].astype(float)
    pred_z2z = trips['pred_z2z'].astype(float)
    dist_line = trips['trip_dist_line'].astype(float)

    with step('compute per-(access_pattern × band) stats'):
        rows: list[dict] = []
        for quantity, pred in (('time', pred_time), ('z2z', pred_z2z)):
            for access_label in ['ALL'] + sorted(
                    trips['access_pattern'].dropna().unique(),
                    key=lambda x: _ACCESS_PATTERN_ORDER.get(x, 99)):
                if access_label == 'ALL':
                    ap_mask = pd.Series(True, index=trips.index)
                else:
                    ap_mask = trips['access_pattern'] == access_label
                both = ap_mask & pred.notna() & meas.notna() & (meas > 0)
                if not both.any():
                    continue
                # ALL band
                rows.append({
                    'access_pattern': access_label, 'quantity': quantity,
                    'band': 'ALL',
                    **_stats(meas.loc[both].to_numpy(dtype=float),
                             pred.loc[both].to_numpy(dtype=float))})
                for band_label, lo, hi in _DIST_BANDS:
                    bmask = (dist_line >= lo) & (dist_line < hi)
                    mmask = both & bmask
                    if not mmask.any():
                        continue
                    rows.append({
                        'access_pattern': access_label, 'quantity': quantity,
                        'band': band_label,
                        **_stats(meas.loc[mmask].to_numpy(dtype=float),
                                 pred.loc[mmask].to_numpy(dtype=float))})

    with step('summary'):
        _log_and_save(context, rows)
    context.close()


def _log_and_save(context, rows: list[dict]) -> None:
    if not rows:
        logging.warning("no rows to summarize — check inputs.")
        return
    df = pd.DataFrame(rows)
    quantity_order = {'time': 0, 'z2z': 1}
    band_order = {'ALL': 0, **{label: i + 1 for i, (label, _, _) in enumerate(_DIST_BANDS)}}
    df['_aporder'] = df['access_pattern'].map(_ACCESS_PATTERN_ORDER).fillna(99)
    df['_qorder']  = df['quantity'].map(quantity_order).fillna(99)
    df['_border']  = df['band'].map(band_order).fillna(99)
    df = df.sort_values(['_aporder', '_qorder', '_border']).drop(
        columns=['_aporder', '_qorder', '_border']).reset_index(drop=True)

    df = df.rename(columns={
        'n': 'n_trips',
        'bias': 'bias', 'bias_pct': 'bias%',
        'mae': 'MAE', 'rmse': 'RMSE',
        'slope': 'slope', 'r2': 'R²', 'mae_pct': 'MAE%',
    })
    display_cols = ['access_pattern', 'quantity', 'band', 'n_trips',
                    'bias', 'bias%', 'MAE', 'RMSE',
                    'slope', 'R²', 'MAE%']

    context.create_results(
        df[display_cols], 'times_vs_survey_transit.csv',
        kws={'index': False, 'float_format': '%.4f'},
    )
    formatters = {
        'n_trips': '{:>7,}'.format,
        'bias':    '{:+8.1f}'.format,
        'bias%':   '{:+.1f}%'.format,
        'MAE':     '{:7.1f}'.format,
        'RMSE':    '{:7.1f}'.format,
        'slope':   '{:+.3f}'.format,
        'R²':      '{:.4f}'.format,
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
        "`bias > 0` = model OVER-predicts. Bands over trip_dist_line "
        "(straight-line trip origin → destination).\n"
        + table)


if __name__ == '__main__':
    main()
