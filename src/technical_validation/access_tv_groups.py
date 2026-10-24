"""
Compare accessibility outputs for the technical-validation scenario groups.

Runs `access_vs_reference.compare_scenarios` for every (target, baseline)
pair in the headline scenario's plan
(`scenarios_technical_validation.TECHNICAL_VALIDATION`), grouped by
validation axis (one variant per group). For `switzerland-h10`:

    resolution   res-h11 / res-h10 / res-h9         vs res-buildings
    cv           cv-fr-test / cv-de-test            vs cv-fr-train / cv-de-train
    data         data-coef / data-ghs               vs res-h10
    network      network-tlmregio                   vs switzerland-h10

Scenarios differing in cell layer are compared on shared OSM buildings
(`buildings` mode), the rest on shared cell ids (see `access_vs_reference`).
Pairs whose target or baseline has no accessibility outputs yet are
skipped with a warning.

Statistics are per accessibility column (destination × bin / k / decay; see
`access_vs_reference.column_stats`). Per group, the log shows R²_1to1, R²_fit
and MAE% as medians over destinations, one row per (grid, metric, profile,
reach, parameter) and one column per target scenario.

Inputs (via read-only source contexts, each scenario's own storage):
    <scenario>/properties/cells_access_*.csv
    preparation/world/osm/shapes/buildings_<area>.gpkg     # buildings mode

Outputs (RESULTS, under the headline scenario):
    technical_validation/access_<group>.csv           # medians per pair × grid × metric ×
                                                      # profile × parameter
    technical_validation/access_<group>_columns.csv   # one row per matched column

Run all groups (default; headline scenario `switzerland-h10`):
    python -m technical_validation.access_tv_groups
Single group:
    python -m technical_validation.access_tv_groups --variant resolution
"""

import logging

import pandas as pd

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

import scenarios  # noqa: F401 — load first: it imports scenarios_technical_validation
from scenarios_technical_validation import TECHNICAL_VALIDATION, technical_validation_plan
from technical_validation.access_vs_reference import by_param, compare_scenarios


# Building → cell mappings, shared across pairs and groups within one run
# (the baselines recur in several pairs).
_B2C_CACHE: dict[str, pd.Series] = {}

_KEYS = ['target', 'reference', 'mode', 'grid', 'metric', 'profile']


def _pivot_table(summary: pd.DataFrame, value: str, fmt: str, targets: list[str]) -> str:
    """`value` (median over destinations) per (grid, metric, profile, reach,
    parameter) × target, formatted for the log."""
    pv = summary.pivot_table(index=['grid', 'metric', 'profile', 'reach', 'param'],
                             columns='target', values=value, aggfunc='first', sort=False)
    pv = pv[[t for t in targets if t in pv.columns]]
    return pv.map(lambda v: fmt.format(v) if pd.notna(v) else '—').to_string()


def main(variant):
    context = init_context(variant)
    group = variant.group
    pairs = technical_validation_plan(context.scenario).groups.get(group)
    if not pairs:
        logging.warning(f"group {group!r}: not in the plan of {context.scenario!r}.")
        context.close()
        return

    rows: list[dict] = []
    for target, reference in pairs:
        with step(f'{group}: {target} vs {reference}'):
            # cv: the two sides use separately estimated utility models, so their
            # utilities are also compared in minutes (see compare_scenarios).
            pair_rows, info = compare_scenarios(
                context, target, reference, b2c_cache=_B2C_CACHE,
                util_in_minutes=(group == 'cv'))
            if info['n_total'] == 0:
                logging.warning(f"  ⚠ {target!r} has no accessibility outputs — skipped "
                                f"(run its pipeline first).")
                continue
            if not pair_rows and info['n_no_pair']:
                logging.warning(f"  ⚠ no counterparts in {reference!r} — skipped "
                                f"(run its pipeline first).")
                continue
            logging.info(f"  → {len(pair_rows)} column(s) compared, mode={info['mode']!r}")
            for r in pair_rows:
                rows.append({'group': group, 'target': target, 'reference': reference,
                             'mode': info['mode'], **r})

    if not rows:
        logging.warning(f"group {group!r}: nothing to compare.")
        context.close()
        return

    with step(f'{group}: summary'):
        columns = pd.DataFrame(rows)
        df = by_param(columns, _KEYS)
        context.create_results(
            columns, f'technical_validation/access_{group}_columns.csv',
            kws={'index': False, 'float_format': '%.4f'})
        context.create_results(
            df, f'technical_validation/access_{group}.csv',
            kws={'index': False, 'float_format': '%.4f'})
        targets = [t for t, _ in pairs]
        baselines = ', '.join(dict.fromkeys(f'{t} vs {r}' for t, r in pairs
                                            if t in set(df['target'])))
        logging.info(f"Group {group!r} ({baselines})\n"
                     f"R²_1to1 (vs the 1:1 line: do values agree?):\n"
                     + _pivot_table(df, 'R²_1to1', '{:.4f}', targets))
        logging.info(f"Group {group!r} — R²_fit (vs the best-fit line: does the pattern agree?):\n"
                     + _pivot_table(df, 'R²_fit', '{:.4f}', targets))
        logging.info(f"Group {group!r} — MAE% (MAE / mean |baseline|):\n"
                     + _pivot_table(df, 'MAE%', '{:.1f}%', targets))
    context.close()


variants = Variants([('group', str)])
for _group in dict.fromkeys(g for plan in TECHNICAL_VALIDATION.values() for g in plan.groups):
    variants.add(name=_group, group=_group)


if __name__ == '__main__':
    variants.run(main, default='all')
