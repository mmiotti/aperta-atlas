"""
Per-mode summary statistics for the survey legs.

For each mode (walk, rbike, ebike25, ebike45, transit, car), filters
to legs whose chosen mode matches, then reports person-weighted mean +
percentiles (p50, p90, p95, p99) of:
  - dist_line_m: straight-line OD distance
  - t_net_min:   net routed travel time
  - t_gross_min: gross (net + orig + dest overheads)

Weighted percentiles use `weight_person` via cumulative-weight
interpolation (inverse weighted empirical CDF).

Provides `t_gross_min_p95` per mode → consumed by 09a as the
LINEAR-EXTRAPOLATION SWITCHOVER point for the utility-vs-time fit.
Beyond `t_cut`, `V(t) = V(t_cut) + s_cut · (t − t_cut)` with
`s_cut = β_t + β_ln/t_cut` — prevents the log-shaped tail from
mis-behaving in the survey's data-thin long-time region.

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_legs.csv                          # from survey/02d
    generic/survey_leg_times.csv                     # from survey/05
    generic/survey_leg_overheads.csv                 # from survey/08a

Output (PRIVATE, under `<scenario>/`):
    generic/survey_summary.csv     # long: mode, metric, p50, p90, p95, p99, mean, n

Run:
    python -m survey.08c_survey_stats --scenario <name>
"""

import logging

import numpy as np
import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step

from main.common import car_overhead_by_peak
from scenarios import get_scenario, scenario_needs_survey_prep


# Modes to summarise (`mode_simplified` labels). `ebike45` stays in the
# summary even though 09a drops it (too few observations to identify) —
# the distributional stats are worth reporting.
_MODES: tuple[str, ...] = (
    'walk', 'rbike', 'ebike25', 'ebike45', 'transit', 'car',
)

# Mode → routing-profile column in survey_leg_times.csv. Car uses
# `t_routed_car` (peak-str-aware from survey/05); transit uses
# `t_routed_transit_z2z` (NPVM z2z).
# TODO: rename 'walk' to 'rwalk' earlier so this map isn't needed.
_MODE_TO_PROFILE: dict[str, str] = {
    'walk':    'rwalk',
    'rbike':   'rbike',
    'ebike25': 'ebike25',
    'ebike45': 'ebike45',
    'car':     'car',
    'transit': 'transit_z2z',
}

_PERCENTILES: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float,
) -> float:
    """Weighted `q`-quantile via cumulative-weight interpolation on the
    sorted values (equivalent to the inverse weighted empirical CDF).
    Non-finite values + non-positive weights are filtered out; returns
    NaN if nothing survives.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float('nan')
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float('nan')
    return float(np.average(values[mask], weights=weights[mask]))


def _mode_times_seconds(
    df: pd.DataFrame, mode: str,
) -> tuple[pd.Series, pd.Series]:
    """Return `(t_net_s, t_gross_s)` for `mode` — seconds. Car needs
    special handling because its overhead is peak-str-aware (per leg's
    `peak_str` picks one of three columns); all other modes read net
    from `t_routed_<profile>` + overheads uniformly."""
    zero = pd.Series(0.0, index=df.index)
    if mode == 'car':
        net_s = df['t_routed_car']
        gross_s = (net_s
                   + car_overhead_by_peak(df, 'orig').fillna(0)
                   + car_overhead_by_peak(df, 'dest').fillna(0))
    else:
        profile = _MODE_TO_PROFILE[mode]
        net_s = df[f't_routed_{profile}']
        # Transit overheads live under `_transit`, NOT `_transit_z2z`,
        # in survey_leg_overheads.csv (08a keys by utility-model mode
        # label, not routing-profile column name).
        ov_profile = 'transit' if mode == 'transit' else profile
        ov_orig = df.get(f't_overhead_orig_{ov_profile}', zero).fillna(0)
        ov_dest = df.get(f't_overhead_dest_{ov_profile}', zero).fillna(0)
        gross_s = net_s + ov_orig + ov_dest
    return net_s, gross_s


def _summarise(
    values: pd.Series, weights: pd.Series, metric_name: str,
) -> dict[str, float | int | str]:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    row: dict[str, float | int | str] = {'metric': metric_name}
    for q in _PERCENTILES:
        row[f'p{int(q*100):02d}'] = _weighted_quantile(v, w, q)
    row['mean'] = _weighted_mean(v, w)
    row['n'] = int(np.isfinite(v).sum())
    return row


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    if not scenario_needs_survey_prep(scenario):
        logging.info(
            f"Scenario {scenario.name!r} has no survey-driven calibrators — "
            f"skipping (survey prep outputs would have no consumer).")
        context.close()
        return

    with step('load survey_legs + survey_leg_times + survey_leg_overheads'):
        legs = context.get_generic(
            'survey_legs.csv', storage=Storage.PRIVATE)
        routed = context.get_generic(
            'survey_leg_times.csv', storage=Storage.PRIVATE)
        overhead = context.get_generic(
            'survey_leg_overheads.csv', storage=Storage.PRIVATE)
        # All three share leg_id; left-join preserves every leg.
        df = legs.join(routed, how='left').join(overhead, how='left')
        logging.info(f"  → {len(df):,} legs joined")
        chose_counts = df['mode_simplified'].value_counts()
        logging.info(f"  → chosen-mode counts: {chose_counts.to_dict()}")

    weights = df['weight_person'].astype(float)
    dist_line = df['dist_line'].astype(float)

    all_rows: list[dict[str, float | int | str]] = []
    for mode in _MODES:
        chosen_mask = (df['mode_simplified'] == mode)
        n_chosen = int(chosen_mask.sum())
        if n_chosen == 0:
            logging.warning(f"  ⚠ mode_simplified={mode!r}: 0 legs in survey — skipped.")
            continue
        with step(f'summarise mode={mode} (n={n_chosen:,})'):
            net_s, gross_s = _mode_times_seconds(df, mode)
            w_sub = weights.loc[chosen_mask]
            rows = [
                _summarise(dist_line.loc[chosen_mask], w_sub, 'dist_line_m'),
                _summarise(net_s.loc[chosen_mask] / 60.0, w_sub, 't_net_min'),
                _summarise(gross_s.loc[chosen_mask] / 60.0, w_sub, 't_gross_min'),
            ]
            for row in rows:
                row['mode'] = mode
                all_rows.append(row)
            logging.info(
                f"  → t_gross_min p95={rows[2]['p95']:.1f}, "
                f"mean={rows[2]['mean']:.1f}; "
                f"dist_line_m mean={rows[0]['mean']:.0f}")

    with step('save survey_summary.csv'):
        summary = pd.DataFrame(all_rows)[
            ['mode', 'metric'] + [f'p{int(q*100):02d}' for q in _PERCENTILES] + ['mean', 'n']
        ]
        context.create_generic(
            summary, 'survey_summary.csv',
            storage=Storage.PRIVATE,
            kws={'float_format': '%.2f', 'index': False},
        )
        logging.info(f"  → {len(summary)} rows ({summary['mode'].nunique()} modes × {summary['metric'].nunique()} metrics)")

    context.close()


if __name__ == '__main__':
    main()
