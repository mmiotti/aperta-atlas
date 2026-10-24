"""
LaTeX table bodies for the accessibility tests of the Technical Validation.

Reads the per-column results of the three comparison scripts
(`access_exact_sample.py`, `access_tv_groups.py`, `access_network_free.py`)
and the fitted coefficients of the headline and cv training scenarios, and
writes the body rows (between `\\midrule` and `\\bottomrule`) of the
Technical Validation tables in the data descriptor (`papers/atlas-pipeline`):

    tiering      R²_1:1 per radii setting (the plan's `exact_compare`), by cost ×
                 metric × reach; the default radii (`exact_baseline`) in bold
    resolution   R²_1:1 per cell layer against buildings, median [worst mode]; the
                 headline scenario's resolution in bold
    data         R²_1:1 (slope) per input alternative, time-based only: one column
                 per (data-* scenario, destination it changes), then no network
                 for walking and cycling, then each `network` scenario (car; left
                 out until its comparison has run)
    cv           R²_1:1 and R²_fit per test region, by cost × metric; utility also in
                 time-equivalent units (each side divided by its marginal utility of
                 time at the reference time)
    cv_coefs     selected coefficients (`_CV_COEFS`) per mode, fitted on all of
                 Switzerland (headline scenario) and on each cv training region
    utility      the headline scenario's mode-choice coefficients with
                 significance stars (Methods; n and ρ̄² are in 09a's log)

All values are medians over destinations per mode, then over modes; the cv
table takes the median over all columns of its grid (destination × parameter ×
profile; its own count bins, not summed) instead, as its grid has no reach
categories. Counts are compared cumulatively from 0 (`access_vs_reference.
column_stats`); columns outside the reach categories (e.g. single bins beyond the
first) are dropped, and so are transit counts up to 10 min, which are almost
empty (door-to-door transit times include access and waiting). The fast e-bike
is left out of the cv table (see the paper).

Inputs (RESULTS, under the headline scenario):
    technical_validation/access_exact_<n>_columns.csv
    technical_validation/access_{resolution,data,network,cv}_columns.csv
    technical_validation/access_network_free_columns.csv
and (cv_coefs; the headline and cv-*-train scenarios' own storage):
    coefs/<kind>/{overheads_road,overheads_transit,edge_weights_<mode>,utility_default}.csv
and (utility; the headline scenario's own storage):
    coefs/<kind>/utility_default.csv

Outputs (RESULTS, under the headline scenario; also logged):
    technical_validation/tables/tv_<table>.tex

Run all tables (headline scenario `switzerland-h10` by default):
    python -m data_paper.tv_tables [--n-origins 1000]
Selected tables:
    python -m data_paper.tv_tables --tables tiering resolution
"""

import argparse
import logging
import math
import os

import pandas as pd

from aperta_atlas.context import Storage, init_context

import scenarios  # noqa: F401 — load first: it imports scenarios_technical_validation
from scenarios import get_scenario
from scenarios_technical_validation import technical_validation_plan

_REACHES = ('local', 'medium', 'broad')
_METRICS = {'counts': 'Counts', 'gravity': 'Gravity', 'nearest_k': 'Nearest-$k$'}
_COSTS = {'time': 'Time', 'util': 'Utility', 'util_min': 'Utility, in minutes'}
_EMPTY_BINS = {'transit': ('0_10min',)}
_CV_EXCLUDED_PROFILES = ('ebike45',)
_DATA_DESTS = {'population': 'population_total', 'employment': 'employment_total'}
_NETWORK_FREE_PROFILES = ('rwalk', 'rbike')

_CV_MODES = ('walk', 'bike', 'transit', 'car')
_P = {'walk': 'walk', 'bike': 'rbike', 'transit': 'transit', 'car': 'car'}


def _time_utility(t_min: float):
    """Time part of the utility at `t_min`: constant + β_t·(t − T) +
    β_ln·log(t / T), with the reference time T stored by 09a (walk: reference
    mode, no constant), on the zero point of the released utilities (a 1-min
    walk; see `main.common.build_disutility_spec`). Shows level and steepness
    in one number per time."""
    def value(coefs: dict[str, pd.DataFrame], mode: str) -> float:
        u, m = coefs['utility_default']['value'], _P[mode]
        asc = u[f'asc_{m}'] if f'asc_{m}' in u.index else 0.0
        t_ref = u['t_ref_min']
        walk_offset = u['b_time_walk'] * t_ref + u['b_time_log_walk'] * math.log(t_ref)
        return (asc + u[f'b_time_{m}'] * (t_min - t_ref)
                + u[f'b_time_log_{m}'] * math.log(t_min / t_ref) + walk_offset)
    return value


# Coefficients compared between calibration regions: (label, number format,
# {mode: (coef set, row, column) or a function of all coef sets}). Car: base
# profile. The constants set the level of times and utilities; climbing and
# density vary between places.
_CV_COEFS: list[tuple[str, str, dict[str, tuple[str, str, object]]]] = [
    ('Overhead constant (s)', '{:.0f}', {
        'walk': ('overheads_road', 'const', 'rwalk'),
        'bike': ('overheads_road', 'const', 'rbike'),
        'transit': ('overheads_transit', 'const_s', 'transit'),
        'car': ('overheads_road', 'const', 'car_base')}),
    ('Network time factor', '{:.3f}', {
        'walk': ('edge_weights_walk', 'baseline_time', ('rwalk', 'coef')),
        'bike': ('edge_weights_bike', 'baseline_time', ('rbike', 'coef')),
        'car': ('edge_weights_car', 'baseline_time', ('car_base', 'coef'))}),
    ('Climbing (s/m)', '{:.2f}', {
        'walk': ('edge_weights_walk', 'elevation_gain', ('rwalk', 'coef')),
        'bike': ('edge_weights_bike', 'elevation_gain', ('rbike', 'coef'))}),
    ('Time utility, 10 min', '{:.2f}', {m: _time_utility(10) for m in _CV_MODES}),
    ('Time utility, 40 min', '{:.2f}', {m: _time_utility(40) for m in _CV_MODES}),
    ('Density (normalized)', '{:.3f}', {
        m: ('utility_default', f'b_density_r500_norm_od_{p}', 'value')
        for m, p in (('bike', 'rbike'), ('transit', 'transit'), ('car', 'car'))}),
]
_REGION_LABELS = {'cv-de-train': 'German-speaking', 'cv-fr-train': 'French-speaking'}

# Mode-choice coefficients: row groups of (label, utility parameter per mode,
# decimals); modes not listed are not in the model for that row.
_UTILITY_ROWS: list[list[tuple[str, dict[str, str], int]]] = [
    [('Alternative-specific constant',
      {m: f'asc_{_P[m]}' for m in ('bike', 'transit', 'car')}, 3),
     ('\\quad shift, peak hours', {'car': 'asc_car_peak'}, 3),
     ('\\quad shift, night hours', {'car': 'asc_car_night'}, 3),
     ('Time (min)', {m: f'b_time_{_P[m]}' for m in _CV_MODES}, 4),
     ('Log(time)', {m: f'b_time_log_{_P[m]}' for m in _CV_MODES}, 3)],
    [('Density (normalized)',
      {m: f'b_density_r500_norm_od_{_P[m]}' for m in ('bike', 'transit', 'car')}, 3),
     ('Mean absolute slope (m/m)',
      {m: f'b_mean_abs_slope_r250_od_{_P[m]}' for m in ('walk', 'bike', 'transit')}, 3),
     ('Speed limit (km/h)', {'bike': 'b_speed_limit_avg_r250_od_rbike'}, 3),
     ('Traffic volume (10,000 veh/day)',
      {m: f'b_traffic_flow_avg_r250_od_{_P[m]}' for m in ('bike', 'transit', 'car')}, 3),
     ('Bicycle-infrastructure score',
      {m: f'b_bike_infra_score_avg_r250_od_{_P[m]}' for m in ('walk', 'bike', 'transit')}, 3)],
    [(label, {m: f'b_sd_{key}_{_P[m]}' for m in ('bike', 'transit', 'car')}, 3)
     for label, key in (('Male', 'cat_sex_m'), ('Age 7--13', 'cat_age_7to13'),
                        ('Age 14--18', 'cat_age_14to18'), ('Age 19--25', 'cat_age_19to25'),
                        ('Age 51--65', 'cat_age_51to65'), ('Age 66--75', 'cat_age_66to75'),
                        ('Age $>$ 75', 'cat_age_76+'),
                        ('Income $\\leq$ 4,000', 'cat_income_leq4000'),
                        ('Income 4,001--8,000', 'cat_income_4001to8000'),
                        ('Income 12,001--16,000', 'cat_income_12001to16000'),
                        ('Income $>$ 16,000', 'cat_income_16001+'))]
    + [('Good weather', {'bike': 'b_sd_bool_weather_good_rbike'}, 3),
       ('Bad weather', {'bike': 'b_sd_bool_weather_bad_rbike'}, 3)],
]


def _load(context, name: str) -> pd.DataFrame:
    # get_results indexes by the first column; reset_index also leaves the cached frame alone.
    c = context.get_results(f'technical_validation/{name}_columns.csv').reset_index()
    c['cost'] = c['grid'].map(lambda g: 'util' if 'util' in g else 'time')
    empty = pd.Series(False, index=c.index)
    for profile, bins in _EMPTY_BINS.items():
        empty |= (c['profile'] == profile) & (c['metric'] == 'counts') & c['param'].isin(bins)
    return c[~empty]


def _by_mode(c: pd.DataFrame, keys: list[str], value: str, agg: str = 'median') -> pd.Series:
    """Median over destinations per mode, then `agg` over modes."""
    c = c[c['reach'].isin(_REACHES)]
    per_mode = c.groupby(keys + ['cost', 'metric', 'reach', 'profile'])[value].median()
    return per_mode.groupby(level=list(range(len(keys) + 3))).agg(agg)


def _num(v: float, digits: int = 3) -> str:
    if pd.isna(v):
        return '---'
    s = f'{v:.{digits}f}'
    return f'${s}$' if v < 0 else s


def _bold(s: str, on: bool) -> str:
    return f'\\textbf{{{s}}}' if on else s


def _rows(cells: dict, row_keys, with_cost: bool = True) -> list[str]:
    """`cells[(cost, metric, reach)]` → list of cell strings, as table rows with
    the cost and metric names written once per block."""
    lines, last_cost, last_metric = [], None, None
    for key in row_keys:
        if key not in cells:
            continue
        cost, metric, reach = key
        label = [_COSTS[cost] if cost != last_cost else '',
                 _METRICS[metric] if (cost, metric) != (last_cost, last_metric) else '', reach]
        if not with_cost:
            label = label[1:]
        lines.append(' & '.join(label + cells[key]) + ' \\\\')
        last_cost, last_metric = cost, metric
    return lines


def _row_keys(costs=('time', 'util')):
    return [(c, m, r) for c in costs for m in _METRICS for r in _REACHES]


def _tiering(context, plan, n_origins: int) -> list[str]:
    c = _load(context, f'access_exact_{n_origins}')
    med = _by_mode(c, ['scenario'], 'R²_1to1').unstack('scenario')

    def radii(name):
        r = get_scenario(name).mode_configs['walk'].radii
        return r.r_cells, r.r_medium

    order = sorted(plan.exact_compare, key=radii)
    logging.info('tiering columns: ' + ', '.join(
        f'{n} ({radii(n)[0]:.0f}/{radii(n)[1]:.0f})' for n in order))
    cells = {key: [_bold(_num(med.at[key, n]), n == plan.exact_baseline) for n in order]
             for key in med.index}
    return _rows(cells, _row_keys())


def _resolution(context, plan, headline) -> list[str]:
    c = _load(context, 'access_resolution')
    med = _by_mode(c, ['target'], 'R²_1to1').unstack('target')
    worst = _by_mode(c, ['target'], 'R²_1to1', 'min').unstack('target')
    targets = [t for t, _ in plan.groups['resolution']]
    released = {t for t in targets
                if (get_scenario(t).cell_source, get_scenario(t).cell_h3_resolution)
                == (headline.cell_source, headline.cell_h3_resolution)}
    logging.info(f'resolution columns: {targets} (bold: {sorted(released)})')
    cells = {key: [_bold(f'{_num(med.at[key, t])} [{_num(worst.at[key, t])}]', t in released)
                   for t in targets] for key in med.index}
    return _rows(cells, _row_keys())


def _data(context, plan) -> list[str]:
    c = _load(context, 'access_data')
    c = c[c['cost'] == 'time']
    # One column per (data-* scenario, destination whose source it changes).
    columns = []
    for target, reference in plan.groups['data']:
        tgt, ref = get_scenario(target), get_scenario(reference)
        for kind, dest in _DATA_DESTS.items():
            if getattr(tgt, f'{kind}_source') != getattr(ref, f'{kind}_source'):
                columns.append((target, dest))
    network = None
    if plan.groups.get('network'):
        if os.path.exists(context.path_for(
                Storage.RESULTS, 'technical_validation/access_network_columns.csv')):
            network = _load(context, 'access_network')
            network = network[network['cost'] == 'time']
        else:
            logging.info('data: no network comparison yet (technical_validation.access_tv_groups '
                         '--variant network); its columns are left out')
    stats = {}
    for value in ('R²_1to1', 'slope'):
        s = {col: _by_mode(c[(c['target'] == col[0]) & (c['dest'] == col[1])], [], value)
             for col in columns}
        nf = _load(context, 'access_network_free')
        s.update({('no network', p): _by_mode(nf[nf['profile'] == p], [], value)
                  for p in _NETWORK_FREE_PROFILES})
        if network is not None:
            # All destinations; the car profiles count as the modes of `_by_mode`.
            s.update({(target, 'car'): _by_mode(network[network['target'] == target], [], value)
                      for target, _ in plan.groups['network']})
        stats[value] = s
    logging.info(f'data columns: {list(stats["R²_1to1"])}')
    cells = {}
    for key in _row_keys(('time',)):
        if all(key not in s for s in stats['R²_1to1'].values()):
            continue
        cells[key] = [f'{_num(stats["R²_1to1"][col].get(key))} '
                      f'({_num(stats["slope"][col].get(key), 2)})'
                      for col in stats['R²_1to1']]
    return _rows(cells, _row_keys(('time',)), with_cost=False)


def _cv(context, plan) -> list[str]:
    c = context.get_results('technical_validation/access_cv_columns.csv').reset_index()
    # `<grid>_min`: utility-based columns in time-equivalent units (access_tv_groups).
    c['cost'] = c['grid'].map(
        lambda g: 'util_min' if g.endswith('_min') else 'util' if 'util' in g else 'time')
    if 'summed' in c.columns:   # the cv table pools the grid's own bins
        c = c[~c['summed'].astype(bool)]
    c = c[~c['profile'].isin(_CV_EXCLUDED_PROFILES)]
    targets = [t for t, _ in plan.groups['cv']]
    logging.info(f'cv columns: {targets} × (R²_1:1, R²_fit)')
    med = c.groupby(['cost', 'metric', 'target'])[['R²_1to1', 'R²_fit']].median()
    lines, last_cost = [], None
    for cost in _COSTS:
        for metric in _METRICS:
            if (cost, metric, targets[0]) not in med.index:
                continue
            vals = [_num(med.at[(cost, metric, t), v], 2 if med.at[(cost, metric, t), v] < -1
                         else 3)
                    for t in targets for v in ('R²_1to1', 'R²_fit')]
            lines.append(' & '.join([_COSTS[cost] if cost != last_cost else '',
                                     _METRICS[metric]] + vals) + ' \\\\')
            last_cost = cost
    return lines


def _cv_coefs(context, regions: dict[str, str]) -> list[str]:
    """Rows per coefficient × region (`{label: scenario}`), one column per mode."""
    sources = {label: context.source(f'{context.project}/{name}')
               for label, name in regions.items()}
    names = {'utility_default'} | {spec[0] for _, _, modes in _CV_COEFS
                                   for spec in modes.values() if isinstance(spec, tuple)}
    coefs = {label: {name: src.get_coefs(name) for name in names}
             for label, src in sources.items()}
    lines = []
    for coef_label, fmt, modes in _CV_COEFS:
        for i, region in enumerate(regions):
            vals = []
            for mode in _CV_MODES:
                if mode not in modes:
                    vals.append('')
                    continue
                spec = modes[mode]
                if callable(spec):
                    v = float(spec(coefs[region], mode))
                else:
                    name, row, col = spec
                    v = float(coefs[region][name].at[row, col])
                vals.append(f'${fmt.format(v)}$' if v < 0 else fmt.format(v))
            lines.append(' & '.join([coef_label if i == 0 else '', region] + vals) + ' \\\\')
    return lines


def _utility(context) -> list[str]:
    """Coefficients with stars (*** p < 0.001, ** p < 0.01, * p < 0.05, robust)."""
    coefs = context.get_coefs('utility_default')
    shown = {name for group in _UTILITY_ROWS for _, names, _ in group for name in names.values()}
    missing = sorted(set(coefs.index) - shown - {'t_ref_min'})
    if missing:
        logging.warning(f'utility parameters not in the table: {missing}')
    lines = []
    for i, group in enumerate(_UTILITY_ROWS):
        if i:
            lines.append('\\midrule')
        for label, names, digits in group:
            cells = []
            for mode in _CV_MODES:
                if mode not in names:
                    cells.append('')
                    continue
                v, pv = coefs.at[names[mode], 'value'], coefs.at[names[mode], 'rob_p_value']
                stars = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
                cells.append(f'${v:.{digits}f}' + (f'^{{{stars}}}$' if stars else '$'))
            lines.append(' & '.join([label] + cells) + ' \\\\')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--n-origins', type=int, default=1000,
                        help='sample size of the exact-baseline run (file name suffix)')
    parser.add_argument('--tables', nargs='*', default=None,
                        help='tables to write (default: all)')
    args, _ = parser.parse_known_args()

    context = init_context()
    plan = technical_validation_plan(context.scenario)
    headline = get_scenario(context.scenario)
    regions = {'Switzerland': context.scenario,
               **{_REGION_LABELS.get(r, r): r for _, r in plan.groups['cv']}}
    builders = {
        'tiering': lambda: _tiering(context, plan, args.n_origins),
        'resolution': lambda: _resolution(context, plan, headline),
        'data': lambda: _data(context, plan),
        'cv': lambda: _cv(context, plan),
        'cv_coefs': lambda: _cv_coefs(context, regions),
        'utility': lambda: _utility(context),
    }
    tables = {name: build() for name, build in builders.items()
              if args.tables is None or name in args.tables}
    for name, lines in tables.items():
        path = context.path_for(Storage.RESULTS, f'technical_validation/tables/tv_{name}.tex')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logging.info(f'{name} → {path}\n' + '\n'.join(lines))
    context.close()


if __name__ == '__main__':
    main()
