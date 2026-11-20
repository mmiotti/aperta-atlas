"""
Distance vs. door-to-door time vs. disutility per mode on MTMC trips
(`dist_time_util_mtmc.pdf`, atlas data descriptor).

Every trip is evaluated for every mode, not just the chosen one, so the
curves compare the three cost types on the same set of trips. Times are
composed as in 09a (routed time + overheads − routed bias, from survey/05 +
08a); disutilities apply `utility_default` through `build_disutility_spec`,
the same spec 09b bakes into the released utility ODMs (SD covariates at
population averages, piecewise-linear time term). Trips are 09a's (survey/02d's trip set) and pass its quality
filters (AOI, `CHOICE_QUALITY_FLAGS`, straight-line distance); a mode with no
route or a time above 09a's cap counts as unavailable for that trip.

Car: each trip is evaluated at peak and at night hours (solid and dotted;
base hours lie between them). The car's disutility depends on its time alone,
not the hour (09a has no period constants), so the time–disutility panels
pool both into one (solid) curve. Every curve: median line and 12.5–87.5th
percentile band, transparent so that overlaps show. Statistics are unweighted; bins with fewer
than `_MIN_LEGS_PER_BIN` trips are left out. Each curve (medians and
percentiles, per bin) is smoothed with a Savitzky–Golay filter
(`_SMOOTH_WINDOW` bins, quadratic): a local polynomial fit, so unlike a
moving average it doesn't pull the curve ends towards the interior. Times
are smoothed in log space, matching their log axis.

Rows: all trips; trips that start or end in a city of more than `_CITY_MIN_RESIDENTS`
residents (the municipality itself, not its agglomeration: the NPVM zones' municipality,
residents summed over its zones).

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_trips.csv                   # survey/02d (--variant mtmc_trips)
    generic/survey_trip_times.csv              # survey/05 (--variant mtmc_trips)
    generic/survey_trip_overheads.csv          # survey/08a (--variant mtmc_trips)
Inputs (PUBLIC, under `<scenario>/`):
    properties/nodes_{walk,bike,car}_extended.csv, nodes_car_flows_avg.csv
    properties/zones_population.csv
    coefs/<kind>/utility_default.csv, utility_default_stats.csv   # 09a

Inputs (PUBLIC, preparation): switzerland/general/traffic_zones_centroids.gpkg  # municipality

Outputs (RESULTS, under `<scenario>/`):
    dist_time_util_mtmc.pdf

Run:
    python -m data_paper.dist_time_util_mtmc --scenario switzerland-h10
"""

import importlib
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from data_paper.access_maps import _FIG_WIDTH_IN
from main.common import (
    CHOICE_QUALITY_FLAGS, build_disutility_spec, extract_sd_averages_from_coefs,
)

_u09a = importlib.import_module('main.09a_utility_estimation')

_UTILITY_VARIANT = 'default'

# Mode → (label, colour, routing profiles evaluated per trip; the first is drawn solid).
_MODES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    'walk':    ('Walking',          '#1a2540', ('rwalk',)),
    'rbike':   ('Bicycle',          '#4a7be0', ('rbike',)),
    'transit': ('Public transport', '#7fc23a', ('transit',)),
    'car':     ('Car',              '#e02b3a', ('car_peak', 'car_night')),
}
_PROFILE_LABELS = {'car_peak': 'peak hours', 'car_night': 'night'}
_LINESTYLES = {'car_night': (0, (1, 1.2))}   # dotted; other profiles solid
_LINEWIDTH = 1.0
_BAND_ALPHA = 0.15

_CITY_MIN_RESIDENTS = 100_000
# Log-spaced bins (widening towards long trips, which are rarer), so the bins at the
# upper end still hold enough trips; uniform in log space, as the smoothing assumes.
_DIST_EDGES_KM = np.geomspace(0.1, 100.0, 21)   # up to the 100-km tier radius
_TIME_EDGES_MIN = np.geomspace(3.0, 60.0, 17)
_PCT_LOW, _PCT_HIGH = 12.5, 87.5
_MIN_LEGS_PER_BIN = 100
_SMOOTH_WINDOW, _SMOOTH_ORDER = 5, 2
_FONT_SIZE = 8

_LABELS = {
    'd': 'Straight-line distance [km]',
    't': 'Door-to-door time [min]',
    'u': 'Disutility',
}
_PANELS = (('d', 't'), ('d', 'u'), ('t', 'u'))


def _load_legs(context) -> pd.DataFrame:
    """Survey trips with per-mode gross times and endpoint features, filtered
    as in 09a (without the choice-specific filters)."""
    df = _u09a._load_and_join(context)
    df = _u09a._compute_mode_times(df, 'gross', include_ebikes=False)
    df = _u09a._join_point_node_features(df, context, ('od',))

    keep = ((df['dist_line'] <= _u09a._MAX_TRIP_DIST_LINE_M)
            & df[list(CHOICE_QUALITY_FLAGS)].eq(1).all(axis=1))
    logging.info(f"  → {int(keep.sum()):,} / {len(df):,} trips after 09a's quality filters")
    return df[keep].copy()


def _car_gross_min(df: pd.DataFrame, profile: str) -> pd.Series:
    """Car gross time for one profile, for every trip (09a picks the trip's own
    profile via `peak_str`; here each trip gets every profile of `_MODES['car']`)."""
    bias_col = f't_routed_bias_{profile}'
    bias = df[bias_col].fillna(0.0) if bias_col in df.columns else 0.0
    total_s = (df[f't_routed_{profile}']
               + df[f't_overhead_orig_{profile}'].fillna(0.0)
               + df[f't_overhead_dest_{profile}'].fillna(0.0)
               - bias)
    return np.maximum(total_s, _u09a._MIN_ROUTE_TIME_S_PER_MODE['car']) / 60.0


def _disutility(df: pd.DataFrame, t_min: pd.Series, spec) -> pd.Series:
    """Per-trip disutility; same terms as `build_disutility_odm`."""
    u = spec.disutility
    d = u.constant + pd.Series(spec.time_disutility(t_min * 60.0), index=t_min.index)
    for feat, beta in u.origin_features.items():
        d = d + beta * df[f'{feat}_orig']
    for feat, beta in u.destination_features.items():
        d = d + beta * df[f'{feat}_dest']
    return d


def _costs_per_profile(df: pd.DataFrame, context) -> dict[str, pd.DataFrame]:
    """`{profile: DataFrame[d, t, u]}` (km, min, disutility), without the trips
    for which the mode is unavailable."""
    coefs = context.get_coefs(f'utility_{_UTILITY_VARIANT}')
    stats = context.get_coefs(f'utility_{_UTILITY_VARIANT}_stats')
    sd_averages = extract_sd_averages_from_coefs(stats)

    out = {}
    for mode, (_label, _colour, profiles) in _MODES.items():
        for profile in profiles:
            t_min = _car_gross_min(df, profile) if mode == 'car' else df[f't_{mode}_min']
            t_min = t_min.where(t_min < _u09a._TIME_CAP_MIN)
            spec = build_disutility_spec(coefs, profile, sd_averages=sd_averages)
            if spec is None:
                raise ValueError(f"No utility spec for profile {profile!r}.")
            out[profile] = pd.DataFrame({
                'd': df['dist_line'] / 1_000.0,
                't': t_min,
                'u': _disutility(df, t_min, spec),
            }).dropna()
            logging.info(f"  → {profile}: {len(out[profile]):,} trips; median time "
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
    for _label, colour, profiles in _MODES.values():
        per_profile = {p: costs[p].loc[costs[p].index.intersection(legs)] for p in profiles}
        if (x, y) == ('t', 'u'):
            # The same disutility at every hour for a given time: one curve per mode.
            per_profile = {profiles[0]: pd.concat(per_profile.values())}
        for profile, c in per_profile.items():
            stats = _binned(c, x, y, edges)
            ax.fill_between(centres, stats['low'], stats['high'],
                            color=colour, alpha=_BAND_ALPHA, linewidth=0)
            ax.plot(centres, stats['median'], color=colour, linewidth=_LINEWIDTH,
                    linestyle=_LINESTYLES.get(profile, '-'))
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
    handles = [
        Line2D([], [], color=colour, linewidth=_LINEWIDTH, linestyle=_LINESTYLES.get(p, '-'),
               label=f'{label}, {_PROFILE_LABELS[p]}' if p in _PROFILE_LABELS else label)
        for label, colour, profiles in _MODES.values() for p in profiles]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False)
    return fig


def main():
    context = init_context()

    with step('load survey trips + per-mode times + endpoint features'):
        df = _load_legs(context)

    with step(f'per-trip times + disutilities (utility_{_UTILITY_VARIANT})'):
        costs = _costs_per_profile(df, context)

    with step(f'trips to or from cities of more than {_CITY_MIN_RESIDENTS:,} residents'):
        zones = context.source('preparation/switzerland/general').get_generic(
            'traffic_zones_centroids.gpkg').set_index('zone_id')
        municipality = zones.loc[zones['is_domestic'].astype(bool), 'municipality']
        residents = context.get_properties('zones', 'population')['population_total']
        per_municipality = residents.groupby(municipality.reindex(residents.index)).sum()
        cities = per_municipality[per_municipality > _CITY_MIN_RESIDENTS]
        city_zones = municipality.index[municipality.isin(cities.index)]
        in_city = df['orig_zone_id'].isin(city_zones) | df['dest_zone_id'].isin(city_zones)
        logging.info(f"  → {int(in_city.sum()):,} of {len(df):,} trips; cities: "
                     + ', '.join(f'{c} ({n:,.0f})' for c, n in
                                 cities.sort_values(ascending=False).items()))
        rows = {
            'All trips': df.index,
            f'Trips to or from cities\nover {_CITY_MIN_RESIDENTS:,} residents': df.index[in_city],
        }

    with step('plot + save'):
        fig = _figure(costs, rows)
        context.create_results(fig, 'dist_time_util_mtmc.pdf')
        plt.close(fig)

    context.close()


if __name__ == '__main__':
    main()
