"""
Compare accessibility outputs between the current (target) scenario and
a reference scenario, on their overlapping (grid × metric × profile ×
destination × bin/k/decay × unit) subspace.

For each grid in the target, its counterpart in the reference is picked
by matching `(travel_cost, utility)` — the two dimensions that identify
a grid's "type". For each `(grid_pair, metric, profile)` where BOTH
scenarios produced a `properties/cells_access_*.csv`, the two frames
are intersected on:

  - **Columns**: destination × bin/k/decay combinations present in both.
    A target grid with fewer bins / fewer dest_cols contributes exactly
    those to the comparison; extra columns on the reference are ignored.
  - **Rows** (spatial unit ids): shared H3 cell ids when both scenarios
    use the same cell layer, OR shared building ids when they don't
    (buildings mode; see `--reference-mode`).

Reference modes (`--reference-mode`, default `auto`):
  - `cells` — direct join on cell ids. Fast; requires both scenarios
    to share `(cell_source, cell_h3_resolution)`.
  - `buildings` — remap each scenario's cell-indexed accessibility onto
    OSM buildings (spatial join buildings.centroid → cells), then join
    on the shared building id. Used when the two scenarios' cell layers
    differ (e.g. res-h10 vs res-h9). For
    `cell_source='buildings'`, cells ARE buildings — pass-through.
  - `auto` — `cells` if `(cell_source, cell_h3_resolution)` matches on
    both sides, else `buildings`.

Statistics are computed per accessibility column (one destination × one
bin / k / decay), not pooled over columns: pooling columns of different
levels (k = 1 vs k = 30, groceries vs population) inflates R² by their
between-column spread and hides which settings fail. Each column also gets
its parameter (`param`, e.g. `k10`) and reach (`REACH`: local / medium /
broad); summaries are medians over destinations per parameter (`by_param`).

Reported per column:
  - `n_units`     — matched-index size (before finite filter)
  - `n_finite`    — units finite in both
  - `R²_1to1`     — R² relative to the 1:1 line (`1 − SS_res / SS_tot`,
                    SS_tot from the reference): do the values agree?
                    Penalises offset, scale and scatter; < 0 = worse than
                    the reference's own mean
  - `R²_fit`      — R² relative to the best-fit line (= squared Pearson
                    correlation): does the spatial pattern agree? Ignores
                    offset and scale
  - `offset`      — mean difference target − reference (level shift)
  - `slope`       — OLS slope tgt~ref (≈ 1 when values agree in scale)
  - `MAE` / `RMSE` — target − reference in the metric's native units

Inputs (via context):
    <target>/    PUBLIC / properties/cells_access_*.csv
    <reference>/ PUBLIC / properties/cells_access_*.csv     # read cross-scenario
    preparation/world/osm/shapes/buildings_<area>.gpkg      # buildings mode

Outputs (RESULTS, under the target scenario):
    access_vs_<reference>_<mode>.csv     # one row per matched column

The comparison itself is `compare_scenarios(...)`, also used by
`technical_validation/access_tv_groups.py` to run the technical-validation groups
(resolution / radii / cv / data) in one go.

Run:
    python -m technical_validation.access_vs_reference --scenario <target> \
        [--reference <name>] [--reference-mode {auto,cells,buildings}]
"""

import argparse
import contextlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aperta import NOTE as _NOTE_LEVEL

from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import _CAR_HOUR_MODIFIER, UTILITY_T_REF_MIN, profile_to_util_mode
from scenarios import get_scenario


@contextlib.contextmanager
def _quiet_logs():
    """Suppress INFO + NOTE messages inside the block — used to hide the
    per-file `Loaded ...` + `Dependency note: ...` spam during the
    comparison loop, so the final summary table isn't buried."""
    logging.disable(_NOTE_LEVEL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


_DEFAULT_REFERENCE = 'switzerland-h10'

# Longest-first so `access_util_nearest_k_rbike` peels the metric before
# splitting the profile. `counts` matched last since it can appear as a
# substring (currently doesn't, but future-proofing).
_METRICS: tuple[str, ...] = ('nearest_k', 'gravity', 'counts')

# Column names are `<destination>_<bin | k | decay>` (e.g. `population_total_0_5min`,
# `poi_errands_groceries_k3`, `employment_total_exp1.0`); stripping the suffix
# gives the destination; the suffix itself is the parameter.
_PARAM_SUFFIX = re.compile(r'_(\d+_\d+(?:min|m)|k\d+|exp[\d.]+(?:min|m)?)$')


def dest_of(column: str) -> str:
    """Destination name of an accessibility column (suffix stripped)."""
    return _PARAM_SUFFIX.sub('', column)


def param_of(column: str) -> str:
    """Parameter of an accessibility column (bin, k or decay suffix)."""
    m = _PARAM_SUFFIX.search(column)
    return m.group(1) if m else ''


# Settings grouped by how far they reach: within a category, the effective reach
# (r50, main/11_access_reach) of counts (cumulative from 0), nearest-k, gravity and
# utility gravity is about equal on average over the modes (Bern-Mittelland, grocery
# stores, secondary-sector jobs and gastronomy: about 1, 1.6 and 3 km). Per mode
# they still differ: on foot, counts are the most local; by car, nearest-k is.
# Parameters not listed (e.g. single bins beyond the first) report as '—'.
REACH: dict[str, str] = {
    '0_10min': 'local',  'k1':  'local',  'exp1min': 'local',  'exp2.0': 'local',
    '0_15min': 'medium', 'k10': 'medium', 'exp2min': 'medium', 'exp1.0': 'medium',
    '0_30min': 'broad',  'k30': 'broad',  'exp5min': 'broad',  'exp0.7': 'broad',
}
_REACH_ORDER = {'local': 0, 'medium': 1, 'broad': 2, '—': 3}


_COUNT_BIN = re.compile(r'^(.*)_(\d+)_(\d+)(min|m)$')


def _cumulative_counts(cols: list[str]) -> dict[str, list[str]]:
    """`{<dest>_0_<hi><unit>: [bin columns]}`: count bins contiguous from 0 summed up
    to each of their upper edges but the first (that one is the first bin itself)."""
    bins: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for c in cols:
        m = _COUNT_BIN.match(c)
        if m:
            bins.setdefault((m[1], m[4]), []).append((int(m[2]), int(m[3]), c))
    out: dict[str, list[str]] = {}
    for (dest, unit), found in bins.items():
        edge, summed = 0, []
        for lo, hi, c in sorted(found):
            if lo != edge:
                break
            summed.append(c)
            edge = hi
            if len(summed) > 1:
                out[f'{dest}_0_{hi}{unit}'] = list(summed)
    return out


def column_stats(tgt: pd.DataFrame, ref: pd.DataFrame, idx, cols: list[str]) -> pd.DataFrame:
    """One row of statistics per column, with its destination, parameter and reach.
    Count bins are also summed into cumulative counts from 0, as users form them
    (rows flagged `summed`)."""
    sums = _cumulative_counts(cols)
    if sums:
        tgt, ref = (df.loc[idx].assign(**{name: df.loc[idx, bins].sum(axis=1, min_count=len(bins))
                                         for name, bins in sums.items()})
                    for df in (tgt, ref))
    rows = []
    for c in list(cols) + list(sums):
        s = _stats_from_arrays(tgt.loc[idx, c].to_numpy(dtype=float),
                               ref.loc[idx, c].to_numpy(dtype=float))
        param = param_of(c)
        rows.append({'column': c, 'dest': dest_of(c), 'param': param,
                     'reach': REACH.get(param, '—'), 'summed': c in sums,
                     'n_finite': s['n_finite'],
                     'R²_1to1': s['r2'], 'R²_fit': s['r2_fit'], 'offset': s['offset'],
                     'slope': s['slope'], 'MAE': s['mae'], 'RMSE': s['rmse'],
                     'MAE%': s['mae_pct']})
    return pd.DataFrame(rows)


def by_param(columns: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median over destinations of the scale-free statistics, per `keys` ×
    parameter (offset / MAE are in each destination's own units, so not
    summarised)."""
    group = columns.groupby(keys + ['reach', 'param'], sort=False)
    out = group[['R²_1to1', 'R²_fit', 'slope', 'MAE%']].median()
    out['n_dests'] = group.size()
    out = out.reset_index()
    out['_order'] = out['reach'].map(_REACH_ORDER)
    return (out.sort_values(keys + ['_order', 'param'], kind='stable')
            .drop(columns='_order').reset_index(drop=True))


def _list_access_files(context) -> list[str]:
    """Basenames (`access_*`) of every `properties/cells_access_*.csv`
    on disk for this context's scenario."""
    props_dir = Path(context.path_for(context.default_storage, 'properties'))
    return sorted(
        f.stem[len('cells_'):]
        for f in props_dir.glob('cells_access_*.csv')
    )


def _auto_reference_mode(target_scenario, reference_scenario) -> str:
    """Pick `cells` when both scenarios share the same cell layer keys,
    else `buildings`. Keys: `(cell_source, cell_h3_resolution)`."""
    def layer_key(s):
        return (s.cell_source, s.cell_h3_resolution)
    return 'cells' if layer_key(target_scenario) == layer_key(reference_scenario) else 'buildings'


def _bldg_to_cell(scenario_ctx, scenario) -> pd.Series:
    """`building_id → cell_id` for a scenario. For `cell_source='buildings'`,
    identity (cells ARE buildings). Otherwise spatial join buildings'
    centroids into cells (H3 or hectare polygons); buildings outside all
    cells (typically at the buffer boundary) get NaN and drop out at the
    finite-mask step downstream.

    `allow_cache=False` on `get_shapes('cells')` is load-bearing: the
    context cache is keyed only by `(relative_path, storage)`, NOT by
    `(project, scenario)`, so back-to-back reads from two scenarios of
    the same file (`shapes/cells.gpkg`) would silently return the
    first scenario's cells for the second — making both bldg→cell
    mappings identical.
    """
    osm_ctx = scenario_ctx.source('preparation/world/osm')
    buildings = osm_ctx.get_shapes('buildings', data_name=scenario.area_name)
    if scenario.cell_source == 'buildings':
        return pd.Series(buildings.index, index=buildings.index, name='cell_id')
    cells = scenario_ctx.get_shapes('cells', allow_cache=False)
    b_pts = buildings[['geometry']].to_crs(cells.crs)
    b_pts.geometry = b_pts.geometry.centroid
    joined = b_pts.sjoin(cells[['geometry']], how='left', predicate='within')
    # `sjoin` names the right-hand index column `index_right` when the
    # right side has an unnamed index, or the right's index.name if set.
    right_col = 'index_right' if 'index_right' in joined.columns else cells.index.name
    return joined[right_col].rename('cell_id')


def _load_access_on_units(
    scenario_ctx, basename: str, mode: str,
    bldg_to_cell: pd.Series | None,
) -> pd.DataFrame:
    """Load a `cells_access_<basename>.csv` and return it keyed by the
    active reference-unit id: cell id (mode='cells') or building id
    (mode='buildings', via `bldg_to_cell`).

    `allow_cache=False` is load-bearing when target and reference share
    storage AND grid-name (e.g. res-h9 vs res-h10,
    both use grid `grid_time` → same relative path). The context cache
    is keyed only by `(relative_path, storage)`; without this flag, the
    second scenario's read silently returns the first's data.
    """
    access = scenario_ctx.get_properties('cells', basename, allow_cache=False)
    if mode == 'cells':
        return access
    assert bldg_to_cell is not None, "buildings mode requires a bldg_to_cell mapping"
    # Buildings mode — remap: each building inherits its containing cell.
    # Drop buildings whose cell is NaN (outside cell coverage — edge/buffer
    # buildings) and those whose cell isn't in `access`.
    b2c = bldg_to_cell.dropna()
    numeric = access.select_dtypes(include='number')
    values = numeric.reindex(b2c.values)
    values.index = pd.Index(b2c.index, name=str(bldg_to_cell.index.name or 'building_id'))
    return values


def _parse_access_basename(basename: str) -> tuple[str, str, str] | None:
    """Parse `access_<grid>_<metric>_<profile>` into (grid, metric, profile).
    `<metric>` is one of `_METRICS`; `<profile>` may be empty (mode-agnostic
    dist grids). Returns `None` on unparseable inputs."""
    if not basename.startswith('access_'):
        return None
    rest = basename[len('access_'):]
    # `_metric` token — grid keys and profile names can contain
    # underscores, so use `_<metric>` as the split anchor. Use the
    # LAST occurrence so a `_gravity_` inside a grid key doesn't
    # confuse the split.
    for metric in _METRICS:
        needle = f'_{metric}'
        i = rest.rfind(needle)
        if i < 0:
            continue
        grid = rest[:i]
        after = rest[i + len(needle):]
        profile = after[1:] if after.startswith('_') else ''
        return grid, metric, profile
    return None


def _match_grids(target_scenario, reference_scenario) -> dict[str, str]:
    """Map target grid_key → reference grid_key by matching (travel_cost,
    utility). Warns and skips when a target grid has 0 or >1 candidates
    on the reference."""
    def grid_type(g):
        return (g.travel_cost, g.utility if g.travel_cost == 'util' else None)

    ref_by_type: dict[tuple, list[str]] = {}
    for ref_key, ref_g in reference_scenario.accessibility_grids.items():
        ref_by_type.setdefault(grid_type(ref_g), []).append(ref_key)

    matches: dict[str, str] = {}
    for tgt_key, tgt_g in target_scenario.accessibility_grids.items():
        candidates = ref_by_type.get(grid_type(tgt_g), [])
        if len(candidates) == 1:
            matches[tgt_key] = candidates[0]
        else:
            logging.warning(
                f"  ⚠ target grid {tgt_key!r} (type={grid_type(tgt_g)}) has "
                f"{len(candidates)} reference candidate(s); skipping.")
    return matches


def _stats_from_arrays(tgt: np.ndarray, ref: np.ndarray) -> dict:
    """Descriptive stats for a matched (target, reference) pair after
    finite-in-both masking. Slope is OLS through the joint mean;
    `r2` is relative to the 1:1 line (`1 − SS_res / SS_tot`, as in
    `times_vs_survey`); `r2_fit` relative to the best-fit line (squared
    Pearson). `mae_pct`
    is a WMAPE-style relative error — `MAE / mean(|ref|)` — dimensionless
    and robust to per-cell zeros
    (undefined only when the reference is uniformly zero). Returns
    `n_finite=0` when insufficient data."""
    mask = np.isfinite(tgt) & np.isfinite(ref)
    n = int(mask.sum())
    if n < 2:
        return {'n_finite': n, 'r2': float('nan'), 'r2_fit': float('nan'),
                'offset': float('nan'), 'slope': float('nan'),
                'mae': float('nan'), 'rmse': float('nan'),
                'mae_pct': float('nan')}
    tgt_f, ref_f = tgt[mask], ref[mask]
    ref_c = ref_f - ref_f.mean()
    tgt_c = tgt_f - tgt_f.mean()
    ss_ref = float((ref_c ** 2).sum())
    slope = float((ref_c * tgt_c).sum() / ss_ref) if ss_ref > 0 else float('nan')
    resid = tgt_f - ref_f
    r2 = 1.0 - float((resid ** 2).sum()) / ss_ref if ss_ref > 0 else float('nan')
    # Pattern agreement (ignores level and scale) + the level shift itself.
    ss_tgt = float((tgt_c ** 2).sum())
    r2_fit = (float((ref_c * tgt_c).sum()) ** 2 / (ss_ref * ss_tgt)
                  if ss_ref > 0 and ss_tgt > 0 else float('nan'))
    offset = float(resid.mean())
    mae = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    ref_abs_mean = float(np.abs(ref_f).mean())
    mae_pct = 100.0 * mae / ref_abs_mean if ref_abs_mean > 0 else float('nan')
    return {
        'n_finite': n,
        'r2': r2,
        'r2_fit': r2_fit,
        'offset': offset,
        'slope': slope,
        'mae': mae,
        'rmse': rmse,
        'mae_pct': mae_pct,
    }


def _common(target_df: pd.DataFrame, reference_df: pd.DataFrame) -> tuple[list[str], pd.Index]:
    """Numeric columns and units both frames share. `get_properties` copies
    the H3 index into a `cell_id` column; the numeric filter drops it."""
    tgt_numeric = target_df.select_dtypes(include='number').columns
    ref_numeric = reference_df.select_dtypes(include='number').columns
    return (sorted(set(tgt_numeric) & set(ref_numeric)),
            target_df.index.intersection(reference_df.index))


def _minutes_per_utility(ctx, variant: str, profile: str) -> float | None:
    """Minutes of door-to-door time worth one utility unit for `profile` under the
    scenario's utility coefficients (`utility_<variant>`): 1 / |∂v/∂t| at the
    reference time of 20 min (the derivative does not depend on whether the time
    terms were centered). `None` if the profile has no utility model."""
    util_mode = profile_to_util_mode(profile)
    u = ctx.get_coefs(f'utility_{variant}')['value']
    if util_mode is None or f'b_time_{util_mode}' not in u.index:
        return None
    slope = (float(u[f'b_time_{util_mode}'])
             + float(u[f'b_time_log_{util_mode}']) / UTILITY_T_REF_MIN)
    b_time_modifier = _CAR_HOUR_MODIFIER.get(profile, (None, None))[1]
    if b_time_modifier in u.index:
        slope += float(u[b_time_modifier])
    return 1.0 / abs(slope)


def compare_scenarios(
    context, target: str, reference: str, *, mode: str = 'auto',
    b2c_cache: dict[str, pd.Series] | None = None, util_in_minutes: bool = False,
) -> tuple[list[dict], dict]:
    """Compare every `cells_access_*` file of `target` with its counterpart
    in `reference`. Both are read through read-only source contexts, so
    `context` can be any project context (its own scenario doesn't matter).

    Returns `(rows, info)`: one row per matched column (grid, metric,
    profile, destination × parameter; see `column_stats`), and `info` with the resolved `mode` plus file counts (`n_total`,
    `n_no_pair`, `n_too_few`). `b2c_cache` (scenario → building→cell
    Series) lets callers comparing many pairs build each buildings-mode
    mapping only once.

    `util_in_minutes`: for scenarios with different utility coefficients, also
    compare the utility-based columns in time-equivalent units, each side divided
    by its own marginal utility of time (`_minutes_per_utility`); these rows get
    the grid name suffix `_min`. Removes differences in the utility scale of
    separately estimated models, not differences in their levels.
    """
    tgt_scenario = get_scenario(target)
    ref_scenario = get_scenario(reference)
    if mode == 'auto':
        mode = _auto_reference_mode(tgt_scenario, ref_scenario)
    info = {'mode': mode, 'n_total': 0, 'n_no_pair': 0, 'n_too_few': 0}

    grid_matches = _match_grids(tgt_scenario, ref_scenario)
    if not grid_matches:
        logging.warning(f"  ⚠ {target!r} vs {reference!r}: no matched grids.")
        return [], info

    tgt_ctx = context.source(f'{context.project}/{target}')
    ref_ctx = context.source(f'{context.project}/{reference}')

    # Buildings mode: building → cell mapping, built once per scenario.
    b2c: dict[str, pd.Series] = {}
    if mode == 'buildings':
        cache = b2c_cache if b2c_cache is not None else {}
        for name, ctx, scenario in ((target, tgt_ctx, tgt_scenario),
                                    (reference, ref_ctx, ref_scenario)):
            if name not in cache:
                with _quiet_logs():
                    cache[name] = _bldg_to_cell(ctx, scenario)
                logging.info(f"  → {name}: {cache[name].notna().sum():,} of "
                             f"{len(cache[name]):,} buildings mapped to a cell")
            b2c[name] = cache[name]

    tgt_files = _list_access_files(tgt_ctx)
    info['n_total'] = len(tgt_files)
    rows: list[dict] = []
    with _quiet_logs():
        for basename in tgt_files:
            parsed = _parse_access_basename(basename)
            if parsed is None:
                continue
            grid, metric, profile = parsed
            if grid not in grid_matches:
                continue
            ref_grid = grid_matches[grid]
            ref_basename = (f'access_{ref_grid}_{metric}'
                            + (f'_{profile}' if profile else ''))
            try:
                tgt_df = _load_access_on_units(tgt_ctx, basename, mode, b2c.get(target))
            except FileNotFoundError:
                continue
            try:
                ref_df = _load_access_on_units(ref_ctx, ref_basename, mode, b2c.get(reference))
            except FileNotFoundError:
                info['n_no_pair'] += 1
                continue
            cols, idx = _common(tgt_df, ref_df)
            columns = column_stats(tgt_df, ref_df, idx, cols) if cols and len(idx) else None
            if columns is None or (columns['n_finite'] >= 2).sum() == 0:
                info['n_too_few'] += 1
                continue
            base = {'grid': grid, 'grid_pair': f'{grid} ↔ {ref_grid}', 'metric': metric,
                    'profile': profile or '—', 'n_units': len(idx)}
            rows.extend({**base, **r} for r in columns.to_dict('records'))
            tgt_grid = tgt_scenario.accessibility_grids[grid]
            if util_in_minutes and tgt_grid.travel_cost == 'util':
                scale_tgt = _minutes_per_utility(tgt_ctx, tgt_grid.utility, profile)
                scale_ref = _minutes_per_utility(ref_ctx, tgt_grid.utility, profile)
                if scale_tgt is not None and scale_ref is not None:
                    minutes = column_stats(tgt_df[cols] * scale_tgt, ref_df[cols] * scale_ref,
                                           idx, cols)
                    rows.extend({**base, 'grid': f'{grid}_min',
                                 'grid_pair': f'{grid}_min ↔ {ref_grid}_min', **r}
                                for r in minutes.to_dict('records'))
    return rows, info


def main():
    parser = argparse.ArgumentParser(
        description="Compare access_* between current and reference scenario.")
    parser.add_argument(
        '--reference', default=_DEFAULT_REFERENCE,
        help=f"Scenario to compare against (default: {_DEFAULT_REFERENCE!r}).")
    parser.add_argument(
        '--reference-mode', choices=('auto', 'cells', 'buildings'), default='auto',
        help="Shared-index basis: 'cells' (direct cell-id join; requires "
             "matching cell layers), 'buildings' (remap via OSM buildings; "
             "works across cell resolutions), or 'auto' (pick based on the "
             "two scenarios' cell configs). Default: auto.")
    args, _ = parser.parse_known_args()

    context = init_context()
    target = context.scenario
    if target == args.reference:
        logging.warning(f"target == reference == {target!r}; nothing to compare.")
        context.close()
        return

    with step(f'compare {target!r} against {args.reference!r}'):
        rows, info = compare_scenarios(
            context, target, args.reference, mode=args.reference_mode)

    with step('summary'):
        _log_summary_table(target, args.reference, rows,
                           info['n_no_pair'], info['n_too_few'],
                           n_total=info['n_total'],
                           context=context, mode=info['mode'])
    context.close()


def _log_summary_table(target: str, reference: str, rows: list[dict],
                       n_skipped_no_pair: int, n_skipped_too_few: int,
                       n_total: int, context=None, mode: str = 'cells') -> None:
    """Log the median statistics per (grid, metric, profile) × parameter and,
    with `context`, save the per-column rows as a CSV under RESULTS."""
    n_files = len({(r['grid'], r['metric'], r['profile']) for r in rows})
    logging.info(f"Comparison: {target!r} vs {reference!r}  "
                 f"({n_files}/{n_total} file(s) matched; "
                 f"{n_skipped_no_pair} no counterpart, "
                 f"{n_skipped_too_few} too few finite)")
    if not rows:
        return
    df = pd.DataFrame(rows)
    if context is not None:
        # Filename encodes reference + mode so invocations don't overwrite each other.
        context.create_results(df, f'access_vs_{reference}_{mode}.csv',
                               kws={'index': False, 'float_format': '%.4f'})
    summary = by_param(df, ['grid_pair', 'metric', 'profile'])
    logging.info('Median over destinations, per parameter:\n'
                 + summary.to_string(index=False, float_format='{:.3f}'.format))


if __name__ == '__main__':
    main()
