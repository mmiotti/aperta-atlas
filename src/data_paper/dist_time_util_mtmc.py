"""
Distance vs. door-to-door time vs. disutility per mode on MTMC legs
(Figure 1 of the atlas data descriptor).

Every leg is evaluated for every mode, not just the chosen one, so the
curves compare the three cost types on the same set of trips. Times are
composed as in 09a (routed time + overheads − routed bias, from survey/05 +
08a); disutilities apply `utility_default` through `build_disutility_spec`,
the same spec 09b bakes into the released utility ODMs (SD covariates at
population averages, car peak / night ASC, log time extrapolated beyond
`t_cut_min`). Legs pass 09a's quality filters (AOI, endpoint elevation,
speed + detour envelopes, straight-line distance); a mode with no route or
a time above 09a's cap counts as unavailable for that leg.

Car: each leg is evaluated under all three car profiles. The narrow band
spans the three profiles' medians; the wide band is the 12.5–87.5th
percentile range pooled over the profiles. Other modes: median line and
12.5–87.5th percentile band. Statistics are unweighted; bins with fewer
than `_MIN_LEGS_PER_BIN` legs are left out. Each curve (medians and
percentiles, per bin) is smoothed with a Savitzky–Golay filter
(`_SMOOTH_WINDOW` bins, quadratic): a local polynomial fit, so unlike a
moving average it doesn't pull the curve ends towards the interior. Times
are smoothed in log space, matching their log axis.

Rows: all legs; legs in the top `_DENSE_SHARE` by endpoint density
(`density_r500_norm`, mean of origin and destination).

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_legs.csv                    # survey/02d
    generic/survey_leg_times.csv               # survey/05
    generic/survey_leg_overheads.csv           # survey/08a
Inputs (PUBLIC, under `<scenario>/`):
    properties/nodes_{walk,bike,car}_extended.csv, nodes_car_flows_avg.csv
    coefs/<kind>/utility_default.csv, utility_default_stats.csv   # 09a

Outputs (RESULTS, under `<scenario>/`):
    dist_time_util_mtmc.pdf

Run:
    python -m data_paper.dist_time_util_mtmc --scenario switzerland-h10
"""

import importlib
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from main.common import (
    build_disutility_spec, extract_sd_averages_from_coefs, extract_t_cut_min_from_coefs,
)

_u09a = importlib.import_module('main.09a_utility_estimation')

_UTILITY_VARIANT = 'default'

# Mode → (label, colour, routing profiles evaluated per leg).
_MODES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    'walk':    ('Walking',          '#1a2540', ('rwalk',)),
    'rbike':   ('Bicycle',          '#4a7be0', ('rbike',)),
    'transit': ('Public transport', '#7fc23a', ('transit',)),
    'car':     ('Car',              '#e02b3a', ('car_night', 'car_base', 'car_peak')),
}

_DENSE_SHARE = 0.2
# Log-spaced bins (widening towards long trips, which are rarer), so the bins at the
# upper end still hold enough legs; uniform in log space, as the smoothing assumes.
_DIST_EDGES_KM = np.geomspace(0.1, 100.0, 21)   # up to the 100-km tier radius
_TIME_EDGES_MIN = np.geomspace(3.0, 60.0, 17)
_PCT_LOW, _PCT_HIGH = 12.5, 87.5
_MIN_LEGS_PER_BIN = 100
_SMOOTH_WINDOW, _SMOOTH_ORDER = 5, 2
_FIG_WIDTH_IN = 6.3      # the article's text width, so fonts print at their nominal size
_FONT_SIZE = 8

_LABELS = {
    'd': 'Straight-line distance [km]',
    't': 'Door-to-door time [min]',
    'u': 'Disutility',
}
_PANELS = (('d', 't'), ('d', 'u'), ('t', 'u'))


def _load_legs(context) -> pd.DataFrame:
    """Survey legs with per-mode gross times and endpoint features, filtered
    as in 09a (without the choice-specific filters)."""
    df = _u09a._load_and_join(context)
    df = _u09a._compute_mode_times(df, 'gross', include_ebikes=False)
    df = _u09a._join_point_node_features(df, context, ('od',))

    elev = context.get_properties('nodes', 'walk_extended')['elevation']
    max_elev = _u09a._MAX_ENDPOINT_ELEVATION_M
    keep = ((df['orig_node_id_walk'].map(elev) <= max_elev)
            & (df['dest_node_id_walk'].map(elev) <= max_elev))
    for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
        if flag in df.columns:
            keep &= df[flag] == 1
    keep &= df['dist_line'].between(_u09a._MIN_LEG_DIST_LINE_M, _u09a._MAX_LEG_DIST_LINE_M)
    logging.info(f"  → {int(keep.sum()):,} / {len(df):,} legs after 09a's quality filters")
    return df[keep.fillna(False)].copy()


def _car_gross_min(df: pd.DataFrame, profile: str) -> pd.Series:
    """Car gross time for one profile, for every leg (09a picks the leg's own
    profile via `peak_str`; here each leg gets all three)."""
    bias_col = f't_routed_bias_{profile}'
    bias = df[bias_col].fillna(0.0) if bias_col in df.columns else 0.0
    total_s = (df[f't_routed_{profile}']
               + df[f't_overhead_orig_{profile}'].fillna(0.0)
               + df[f't_overhead_dest_{profile}'].fillna(0.0)
               - bias)
    return np.maximum(total_s, _u09a._MIN_ROUTE_TIME_S_PER_MODE['car']) / 60.0


def _disutility(df: pd.DataFrame, t_min: pd.Series, spec) -> pd.Series:
    """Per-leg disutility; same terms as `build_disutility_odm` +
    `apply_log_time_utility` (spec coefficients are per second)."""
    u = spec.disutility
    t_sec = t_min * 60.0
    d = u.constant + u.cost_coefficient * t_sec
    for feat, beta in u.origin_features.items():
        d = d + beta * df[f'{feat}_orig']
    for feat, beta in u.destination_features.items():
        d = d + beta * df[f'{feat}_dest']
    if spec.b_time_log != 0.0:
        log_t = np.log(t_sec)
        if spec.t_cut_min is not None:
            t_cut = spec.t_cut_min * 60.0
            log_t = log_t.where(t_sec <= t_cut, math.log(t_cut) + (t_sec - t_cut) / t_cut)
        d = d + spec.b_time_log * log_t
    return d


def _costs_per_profile(df: pd.DataFrame, context) -> dict[str, pd.DataFrame]:
    """`{profile: DataFrame[d, t, u]}` (km, min, disutility); NaN where the
    mode is unavailable for the leg."""
    coefs = context.get_coefs(f'utility_{_UTILITY_VARIANT}')
    stats = context.get_coefs(f'utility_{_UTILITY_VARIANT}_stats')
    sd_averages = extract_sd_averages_from_coefs(stats)
    t_cut_min = extract_t_cut_min_from_coefs(stats) or None

    out = {}
    for mode, (_label, _colour, profiles) in _MODES.items():
        for profile in profiles:
            t_min = _car_gross_min(df, profile) if mode == 'car' else df[f't_{mode}_min']
            t_min = t_min.where(t_min < _u09a._TIME_CAP_MIN)
            spec = build_disutility_spec(coefs, profile, sd_averages=sd_averages,
                                         t_cut_min_per_mode=t_cut_min)
            if spec is None:
                raise ValueError(f"No utility spec for profile {profile!r}.")
            out[profile] = pd.DataFrame({
                'd': df['dist_line'] / 1_000.0,
                't': t_min,
                'u': _disutility(df, t_min, spec),
            }).dropna()
            logging.info(f"  → {profile}: {len(out[profile]):,} legs; median time "
                         f"{out[profile]['t'].median():.1f} min, median disutility "
                         f"{out[profile]['u'].median():.2f}")
    return out


def _smooth(values: np.ndarray, log: bool) -> np.ndarray:
    """Savitzky–Golay filter over each run of consecutive finite bins (runs
    shorter than the window are left as they are)."""
    v = np.log(values) if log else values.astype(float)
    out = v.copy()
    finite = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= _SMOOTH_WINDOW:
            out[start:stop] = savgol_filter(v[start:stop], _SMOOTH_WINDOW, _SMOOTH_ORDER,
                                            mode='interp')
    return np.exp(out) if log else out


def _binned(costs: pd.DataFrame, x: str, y: str, edges: np.ndarray) -> pd.DataFrame:
    """Smoothed median and percentile band of `y` per bin of `x`; NaN for thin bins."""
    bins = pd.cut(costs[x], edges)
    grouped = costs.groupby(bins, observed=False)[y]
    stats = pd.DataFrame({
        'median': grouped.median(),
        'low': grouped.quantile(_PCT_LOW / 100),
        'high': grouped.quantile(_PCT_HIGH / 100),
    })
    stats.loc[grouped.size() < _MIN_LEGS_PER_BIN] = np.nan
    return stats.apply(lambda col: _smooth(col.to_numpy(), log=(y == 't')))


def _plot_panel(ax, costs: dict[str, pd.DataFrame], legs: pd.Index, x: str, y: str) -> None:
    edges = _DIST_EDGES_KM if x == 'd' else _TIME_EDGES_MIN
    centres = np.sqrt(edges[:-1] * edges[1:])   # geometric bin centres (log-spaced bins)
    for _mode, (_label, colour, profiles) in _MODES.items():
        per_profile = [costs[p].loc[costs[p].index.intersection(legs)] for p in profiles]
        pooled = _binned(pd.concat(per_profile), x, y, edges)
        ax.fill_between(centres, pooled['low'], pooled['high'],
                        color=colour, alpha=0.15, linewidth=0)
        if len(profiles) == 1:
            ax.plot(centres, pooled['median'], color=colour, linewidth=1.2)
        else:
            medians = pd.concat([_binned(c, x, y, edges)['median'] for c in per_profile], axis=1)
            ax.fill_between(centres, medians.min(axis=1, skipna=False),
                            medians.max(axis=1, skipna=False),
                            color=colour, alpha=0.7, linewidth=0)
    plain = FuncFormatter(lambda v, _pos: f'{v:g}')
    if x == 'd':
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plain)
    if y == 't':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel(_LABELS[x])


def _figure(costs: dict[str, pd.DataFrame], rows: dict[str, pd.Index]) -> plt.Figure:
    plt.rcParams.update({'font.size': _FONT_SIZE})
    fig, axes = plt.subplots(len(rows), len(_PANELS), layout='constrained',
                             figsize=(_FIG_WIDTH_IN, len(rows) * 2.0 + 0.3))
    for i, (row_label, legs) in enumerate(rows.items()):
        for j, (x, y) in enumerate(_PANELS):
            ax = axes[i, j]
            _plot_panel(ax, costs, legs, x, y)
            ax.set_ylabel(f'{row_label}\n\n{_LABELS[y]}' if j == 0 else _LABELS[y])
    handles = []
    for _mode, (label, colour, profiles) in _MODES.items():
        if len(profiles) == 1:
            handles.append(Line2D([], [], color=colour, linewidth=1.2, label=label))
        else:
            handles.append(Patch(color=colour, alpha=0.7, label=f'{label} (night, base, peak)'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False)
    return fig


def main():
    context = init_context()

    with step('load survey legs + per-mode times + endpoint features'):
        df = _load_legs(context)

    with step(f'per-leg times + disutilities (utility_{_UTILITY_VARIANT})'):
        costs = _costs_per_profile(df, context)

    with step(f'high-density subset (top {_DENSE_SHARE:.0%} by mean endpoint density)'):
        density = (df['density_r500_norm_orig'] + df['density_r500_norm_dest']) / 2
        threshold = density.quantile(1 - _DENSE_SHARE)
        dense = density.index[density >= threshold]
        logging.info(f"  → {len(dense):,} of {density.notna().sum():,} legs "
                     f"(mean density_r500_norm ≥ {threshold:.3f})")
        rows = {
            'All trips': df.index,
            f'Highest-density {_DENSE_SHARE:.0%} of trips': dense,
        }

    with step('plot + save'):
        fig = _figure(costs, rows)
        context.create_results(fig, 'dist_time_util_mtmc.pdf')
        plt.close(fig)

    context.close()


if __name__ == '__main__':
    main()
