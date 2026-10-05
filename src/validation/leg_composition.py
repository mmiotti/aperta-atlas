"""
Diagnostic: distribution of `n_legs_in_trip` per survey mode.

Answers a data-consistency question raised while investigating transit
bias in `times_vs_survey.py`: MTMC records TRIPS as one-or-more LEGS
(mode changes and/or purpose changes trigger a new leg). For a
multi-modal trip like walk → train → walk, the transit LEG's
`time_measured` covers only the train segment (platform-to-platform,
including wait), while NPVM z2z is DOOR-TO-DOOR (includes walk-to-
stop and walk-from-stop). Comparing them 1:1 conflates these.

This script quantifies how often each mode's legs sit inside multi-leg
trips vs. as the whole trip on their own, so the transit `bias(z2z)`
observation can be interpreted correctly.

Uses `n_legs_in_trip` (preserved in 02d's `_KEEP_COLS`) — the actual
trip_id was dropped, so we can't reconstruct trip identity, but this
column gives per-leg the size of the parent trip.

Two views of "average legs per trip" per mode:
  - LEG-VIEW mean:  mean(n_legs_in_trip) over all legs of that mode.
                    Biased upward — big trips contribute more rows.
  - TRIP-VIEW mean: weighted by 1/n_legs_in_trip. Unbiased estimate of
                    "the average trip that contains a leg of this mode
                    has how many legs?"

Inputs (PRIVATE):
    generic/survey_legs.csv         # from survey/02d

Outputs:
    RESULTS/leg_composition.csv     # per-mode diagnostic table
    Logs                            # summary tables

Run:
    python -m validation.leg_composition --scenario <name>
"""

import argparse
import logging

import numpy as np
import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step


def _per_mode_stats(sub: pd.DataFrame) -> dict:
    n = sub['n_legs_in_trip'].dropna().astype(int)
    if n.empty:
        return {'n_legs': 0, 'min': np.nan, 'p50': np.nan, 'p90': np.nan,
                'p99': np.nan, 'max': np.nan, 'mean_leg_view': np.nan,
                'mean_trip_view': np.nan, 'frac_n1': np.nan, 'frac_n2': np.nan,
                'frac_n3': np.nan, 'frac_n_ge4': np.nan}
    # Trip-view weight = 1 / n_legs_in_trip; sum of weights = n_trips (that
    # contain a leg of this mode); weighted mean of n_legs_in_trip gives
    # unbiased average trip size for trips containing this mode.
    w = 1.0 / n.to_numpy(dtype=float)
    trip_view_mean = float((n.to_numpy(dtype=float) * w).sum() / w.sum())
    return {
        'n_legs':          int(len(n)),
        'min':             int(n.min()),
        'p50':             int(np.median(n)),
        'p90':             int(np.percentile(n, 90)),
        'p99':             int(np.percentile(n, 99)),
        'max':             int(n.max()),
        'mean_leg_view':   float(n.mean()),
        'mean_trip_view':  trip_view_mean,
        'frac_n1':         float((n == 1).mean()),
        'frac_n2':         float((n == 2).mean()),
        'frac_n3':         float((n == 3).mean()),
        'frac_n_ge4':      float((n >= 4).mean()),
    }


def main():
    argparse.ArgumentParser(
        description="Per-mode diagnostics of n_legs_in_trip.",
    ).parse_known_args()

    context = init_context()

    with step('load survey_legs.csv'):
        legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
        if 'n_legs_in_trip' not in legs.columns:
            raise KeyError(
                "Column 'n_legs_in_trip' not in survey_legs.csv. "
                "Verify it's in 02d's `_KEEP_COLS` and rerun 02d.")
        logging.info(f"  → {len(legs):,} legs")

    with step('compute per-mode composition stats'):
        rows = []
        # Overall row first, then per-mode alphabetically.
        rows.append({'mode': 'ALL', **_per_mode_stats(legs)})
        for mode in sorted(legs['mode_simplified'].dropna().unique()):
            sub = legs[legs['mode_simplified'] == mode]
            rows.append({'mode': mode, **_per_mode_stats(sub)})

    with step('summary (per-mode leg-view stats)'):
        df = pd.DataFrame(rows)
        _log_and_save(context, df)

    # ---- Trip-level composition (requires trip_id) ------------------------
    if 'trip_id' in legs.columns:
        with step('trip-level composition (mode presence, multimodal breakdown)'):
            _log_trip_composition(context, legs)
    else:
        logging.info(
            "trip_id not in survey_legs.csv — skipping trip-level composition. "
            "Add 'trip_id' to 02d's `_KEEP_COLS` and rerun 02d to enable.")

    context.close()


def _log_trip_composition(context, legs: pd.DataFrame) -> None:
    """Trip-level composition: how often each mode appears in a trip,
    and among multimodal trips (n_legs > 1) how many contain transit vs.
    do not. Answers 'how many multimodal trips have no transit leg?'"""
    trips = legs.groupby('trip_id').agg(
        n_legs=('mode_simplified', 'size'),
        modes=('mode_simplified', lambda s: frozenset(s.dropna())),
    )
    n_trips = len(trips)
    n_single = int((trips['n_legs'] == 1).sum())
    n_multi = int((trips['n_legs'] > 1).sum())
    trips_multi = trips.loc[trips['n_legs'] > 1]

    all_modes = sorted({m for mset in trips['modes'] for m in mset})
    presence_rows = []
    for mode in all_modes:
        has_mode = trips['modes'].apply(lambda mset, mode=mode: mode in mset)
        has_mode_multi = trips_multi['modes'].apply(
            lambda mset, mode=mode: mode in mset)
        presence_rows.append({
            'mode':                 mode,
            'trips_with':           int(has_mode.sum()),
            'frac_of_all':          float(has_mode.mean()),
            'trips_with_multi':     int(has_mode_multi.sum()),
            'frac_of_multi':        float(has_mode_multi.mean())
                                    if len(trips_multi) else np.nan,
        })

    # Multimodal breakdown: single-mode-multi-leg vs true-multimodal;
    # among true-multimodal, with/without transit.
    modes_per_trip = trips_multi['modes']
    single_mode_multi_leg = int(
        modes_per_trip.apply(lambda mset: len(mset) == 1).sum())
    true_multimodal = int(
        modes_per_trip.apply(lambda mset: len(mset) > 1).sum())
    multimodal_with_transit = int(
        modes_per_trip.apply(
            lambda mset: (len(mset) > 1) and ('transit' in mset)).sum())
    multimodal_no_transit = int(
        modes_per_trip.apply(
            lambda mset: (len(mset) > 1) and ('transit' not in mset)).sum())

    header = (
        f"Trip-level composition ({n_trips:,} trips total):\n"
        f"  single-leg trips        : {n_single:,} ({100*n_single/max(n_trips,1):5.1f} %)\n"
        f"  multi-leg trips (n>1)   : {n_multi:,}  ({100*n_multi /max(n_trips,1):5.1f} %)\n"
        f"    ↳ single-mode multi-leg (all legs same mode) : {single_mode_multi_leg:,} "
        f"({100*single_mode_multi_leg/max(n_multi,1):5.1f} %)\n"
        f"    ↳ true multimodal (≥ 2 distinct modes)        : {true_multimodal:,} "
        f"({100*true_multimodal/max(n_multi,1):5.1f} %)\n"
        f"        ↳ WITH transit    : {multimodal_with_transit:,} "
        f"({100*multimodal_with_transit/max(true_multimodal,1):5.1f} % of true multimodal)\n"
        f"        ↳ WITHOUT transit : {multimodal_no_transit:,} "
        f"({100*multimodal_no_transit/max(true_multimodal,1):5.1f} % of true multimodal)\n")

    presence_df = pd.DataFrame(presence_rows)
    formatters = {
        'trips_with':       '{:>10,}'.format,
        'frac_of_all':      '{:>7.1%}'.format,
        'trips_with_multi': '{:>10,}'.format,
        'frac_of_multi':    '{:>7.1%}'.format,
    }
    presence_table = presence_df.to_string(index=False, formatters=formatters,
                                           justify='right')
    logging.info(
        header
        + "\nPer-mode presence in trips: how often each mode appears "
          "in at least one leg of a trip.\n"
          "  trips_with       : count of trips containing at least one leg of this mode.\n"
          "  frac_of_all      : that count over ALL trips.\n"
          "  trips_with_multi : count of MULTI-LEG trips containing this mode.\n"
          "  frac_of_multi    : that count over MULTI-LEG trips.\n"
        + presence_table)

    # Save trip-composition to a companion CSV.
    context.create_results(
        presence_df, 'trip_composition_per_mode.csv',
        kws={'index': False, 'float_format': '%.4f'},
    )


def _log_and_save(context, df: pd.DataFrame) -> None:
    display_cols = [
        'mode', 'n_legs',
        'min', 'p50', 'p90', 'p99', 'max',
        'mean_leg_view', 'mean_trip_view',
        'frac_n1', 'frac_n2', 'frac_n3', 'frac_n_ge4',
    ]
    context.create_results(
        df[display_cols], 'leg_composition.csv',
        kws={'index': False, 'float_format': '%.4f'},
    )
    formatters = {
        'n_legs':          '{:>8,}'.format,
        'min':             '{:>3d}'.format,
        'p50':             '{:>3d}'.format,
        'p90':             '{:>3d}'.format,
        'p99':             '{:>3d}'.format,
        'max':             '{:>3d}'.format,
        'mean_leg_view':   '{:>6.2f}'.format,
        'mean_trip_view':  '{:>6.2f}'.format,
        'frac_n1':         '{:>5.1%}'.format,
        'frac_n2':         '{:>5.1%}'.format,
        'frac_n3':         '{:>5.1%}'.format,
        'frac_n_ge4':      '{:>5.1%}'.format,
    }
    table = df[display_cols].to_string(index=False, formatters=formatters,
                                       justify='right')
    logging.info(
        "Per-mode composition of `n_legs_in_trip`. Percentiles taken over "
        "LEGS (rows), so a trip with 5 legs contributes 5 rows to the stats.\n"
        "  mean_leg_view:  mean(n_legs_in_trip) over legs (biased upward).\n"
        "  mean_trip_view: 1/n-weighted mean → avg trip size for trips "
        "containing this mode (unbiased).\n"
        "  frac_n1: fraction of legs whose parent trip is single-leg "
        "(this leg IS the whole trip → time_measured is door-to-door).\n"
        "  frac_n2/n3/n_ge4: fraction in 2-leg / 3-leg / ≥4-leg trips.\n"
        "For transit, high frac_n>1 means MTMC time_measured excludes "
        "walk-to-stop / walk-from-stop, while NPVM z2z includes them — "
        "explains part of any z2z over-prediction.\n"
        + table)


if __name__ == '__main__':
    main()
