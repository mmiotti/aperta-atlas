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

Reported per matched CSV pair:
  - `n_units`     — matched-index size (before finite filter)
  - `n_cols`      — matched-column count
  - `n_finite`    — unit × column entries finite in both
  - `R²`          — squared Pearson between target and reference
  - `slope`       — OLS slope tgt~ref (≈ 1 when values agree in scale)
  - `MAE` / `RMSE` — target − reference in the metric's native units

Inputs (via context):
    <target>/    PUBLIC / properties/cells_access_*.csv
    <reference>/ PUBLIC / properties/cells_access_*.csv     # read cross-scenario
    preparation/world/osm/shapes/buildings_<area>.gpkg      # buildings mode

Outputs:
    Logs only — one line per matched (grid × metric × profile).

Run:
    python -m validation.access_vs_reference --scenario <target> \
        [--reference <name>] [--reference-mode {auto,cells,buildings}]
"""

import argparse
import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aperta import NOTE as _NOTE_LEVEL

from aperta_atlas.context import init_context
from aperta_atlas.utils import step

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

# Aggregate R² below this triggers a per-column breakdown for that row
# (worst-N columns by R²) — quick auto-diagnosis of which destination ×
# bin combo drags the aggregate down (usually boundary-effect columns).
_LOW_R2_THRESHOLD: float = 0.95
_LOW_R2_TOP_N: int = 3


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
    `R² = Pearson²`. `mae_pct` is a WMAPE-style relative error —
    `MAE / mean(|ref|)` — dimensionless and robust to per-cell zeros
    (undefined only when the reference is uniformly zero). Returns
    `n_finite=0` when insufficient data."""
    mask = np.isfinite(tgt) & np.isfinite(ref)
    n = int(mask.sum())
    if n < 2:
        return {'n_finite': n, 'r2': float('nan'), 'slope': float('nan'),
                'mae': float('nan'), 'rmse': float('nan'),
                'mae_pct': float('nan')}
    tgt_f, ref_f = tgt[mask], ref[mask]
    ref_c = ref_f - ref_f.mean()
    tgt_c = tgt_f - tgt_f.mean()
    ss_ref = float((ref_c ** 2).sum())
    ss_tgt = float((tgt_c ** 2).sum())
    slope = float((ref_c * tgt_c).sum() / ss_ref) if ss_ref > 0 else float('nan')
    if ss_ref > 0 and ss_tgt > 0:
        r2 = float(((ref_c * tgt_c).sum() / (ss_ref * ss_tgt) ** 0.5) ** 2)
    else:
        r2 = float('nan')
    resid = tgt_f - ref_f
    mae = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    ref_abs_mean = float(np.abs(ref_f).mean())
    mae_pct = 100.0 * mae / ref_abs_mean if ref_abs_mean > 0 else float('nan')
    return {
        'n_finite': n,
        'r2': r2,
        'slope': slope,
        'mae': mae,
        'rmse': rmse,
        'mae_pct': mae_pct,
    }


def _compare(target_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    """Column ∩ + row ∩ + finite-in-both mask; then descriptive stats
    aggregated across ALL matched columns.

    `get_properties` copies the H3 index into a `cell_id` column on the
    way out; restricting to numeric columns drops it (plus any other
    incidental non-numeric metadata columns).
    """
    tgt_numeric = target_df.select_dtypes(include='number').columns
    ref_numeric = reference_df.select_dtypes(include='number').columns
    common_cols = sorted(set(tgt_numeric) & set(ref_numeric))
    common_idx = target_df.index.intersection(reference_df.index)
    base = {'n_units': int(len(common_idx)), 'n_cols': len(common_cols),
            'common_cols': common_cols, 'common_idx': common_idx}
    if not common_cols or len(common_idx) == 0:
        return {**base, 'n_finite': 0, 'r2': float('nan'),
                'slope': float('nan'), 'mae': float('nan'),
                'rmse': float('nan')}
    tgt = target_df.loc[common_idx, common_cols].to_numpy(dtype=float).ravel()
    ref = reference_df.loc[common_idx, common_cols].to_numpy(dtype=float).ravel()
    return {**base, **_stats_from_arrays(tgt, ref)}


def _compare_per_column(
    target_df: pd.DataFrame, reference_df: pd.DataFrame,
    common_idx, common_cols: list[str],
) -> list[dict]:
    """Same stats as `_compare`, computed independently per column.
    Called only when the aggregate R² falls below `_LOW_R2_THRESHOLD` —
    identifies which destination × bin combo is dragging the aggregate
    down (usually boundary-effect columns at the widest time bins)."""
    out = []
    tgt_sub = target_df.loc[common_idx, common_cols]
    ref_sub = reference_df.loc[common_idx, common_cols]
    for col in common_cols:
        stats = _stats_from_arrays(
            tgt_sub[col].to_numpy(dtype=float),
            ref_sub[col].to_numpy(dtype=float),
        )
        out.append({'column': col, **stats})
    return out


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

    tgt_scenario = get_scenario(target)
    ref_scenario = get_scenario(args.reference)
    mode = (args.reference_mode
            if args.reference_mode != 'auto'
            else _auto_reference_mode(tgt_scenario, ref_scenario))
    logging.info(f"Comparing target={target!r} against reference={args.reference!r} "
                 f"(reference-mode={mode!r})")

    with step('match target grids to reference by (travel_cost, utility)'):
        grid_matches = _match_grids(tgt_scenario, ref_scenario)
        logging.info(f"  → {len(grid_matches)} grid pair(s): {grid_matches}")
        if not grid_matches:
            logging.warning("no matched grids — nothing to compare.")
            context.close()
            return

    ref_ctx = context.source(f'{context.project}/{args.reference}')

    # Pre-compute the buildings-mode building→cell mapping ONCE per
    # scenario. Passed into every per-metric load below.
    tgt_b2c: pd.Series | None = None
    ref_b2c: pd.Series | None = None
    if mode == 'buildings':
        with step('build buildings → cell mapping (spatial join, once per scenario)'):
            tgt_b2c = _bldg_to_cell(context, tgt_scenario)
            ref_b2c = _bldg_to_cell(ref_ctx, ref_scenario)
            logging.info(f"  → target buildings: {len(tgt_b2c):,} "
                         f"({tgt_b2c.notna().sum():,} mapped to a cell)")
            logging.info(f"  → reference buildings: {len(ref_b2c):,} "
                         f"({ref_b2c.notna().sum():,} mapped to a cell)")

    with step('enumerate target access files + compare each to reference'):
        tgt_files = _list_access_files(context)
        logging.info(f"  → {len(tgt_files)} target access file(s); "
                     f"loading + comparing (quietly)…")
        rows: list[dict] = []
        n_skipped_no_pair = 0
        n_skipped_too_few = 0
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
                    tgt_df = _load_access_on_units(context, basename, mode, tgt_b2c)
                except FileNotFoundError:
                    continue
                try:
                    ref_df = _load_access_on_units(ref_ctx, ref_basename, mode, ref_b2c)
                except FileNotFoundError:
                    n_skipped_no_pair += 1
                    continue
                stats = _compare(tgt_df, ref_df)
                if stats.get('n_finite', 0) < 2:
                    n_skipped_too_few += 1
                    continue
                # Always compute per-column stats — cheap because dfs are
                # already loaded, and it lets us surface `MAE%_max` on the
                # summary row AND still power the low-R² breakdown below.
                per_col = _compare_per_column(
                    tgt_df, ref_df,
                    stats['common_idx'], stats['common_cols'],
                )
                finite_per_col = [
                    c for c in per_col
                    if np.isfinite(c.get('mae_pct', float('nan')))
                ]
                if finite_per_col:
                    worst = max(finite_per_col, key=lambda c: c['mae_pct'])
                    mae_pct_max = worst['mae_pct']
                    mae_pct_max_col = worst['column']
                else:
                    mae_pct_max = float('nan')
                    mae_pct_max_col = ''
                # Emit per-column detail only when aggregate R² is low
                # (`_per_col=None` suppresses the breakdown block below).
                per_col_for_breakdown = (
                    per_col if np.isfinite(stats['r2'])
                    and stats['r2'] < _LOW_R2_THRESHOLD else None
                )
                rows.append({
                    'grid_pair':    f'{grid} ↔ {ref_grid}',
                    'metric':       metric,
                    'profile':      profile or '—',
                    'n_units':      stats['n_units'],
                    'n_cols':       stats['n_cols'],
                    'n_finite':     stats['n_finite'],
                    'R²':           stats['r2'],
                    'slope':        stats['slope'],
                    'MAE':          stats['mae'],
                    'RMSE':         stats['rmse'],
                    'MAE%':         stats['mae_pct'],
                    'MAE%_max':     mae_pct_max,
                    'MAE%_max_col': mae_pct_max_col,
                    '_per_col':     per_col_for_breakdown,
                })

    with step('summary'):
        _log_summary_table(target, args.reference, rows,
                           n_skipped_no_pair, n_skipped_too_few,
                           n_total=len(tgt_files),
                           context=context, mode=mode)
    context.close()


def _log_summary_table(target: str, reference: str, rows: list[dict],
                       n_skipped_no_pair: int, n_skipped_too_few: int,
                       n_total: int, context=None, mode: str = 'cells') -> None:
    """Emit the final comparison table + counts + low-R² breakdowns.
    One row per matched (grid_pair, metric, profile), sorted. Also
    persists the summary as a CSV under RESULTS when `context` is given."""
    logging.info(f"Comparison: {target!r} vs {reference!r}  "
                 f"({len(rows)}/{n_total} file(s) matched; "
                 f"{n_skipped_no_pair} no counterpart, "
                 f"{n_skipped_too_few} too few finite)")
    if not rows:
        return
    display_cols = ['grid_pair', 'metric', 'profile', 'n_units', 'n_cols',
                    'n_finite', 'R²', 'slope', 'MAE', 'RMSE',
                    'MAE%', 'MAE%_max', 'MAE%_max_col']
    df = pd.DataFrame(rows).sort_values(
        ['grid_pair', 'metric', 'profile']).reset_index(drop=True)

    if context is not None:
        # Save the summary (numeric-precise, no formatters) to RESULTS.
        # Filename encodes reference + mode so multiple invocations
        # don't overwrite each other.
        rel = f'access_vs_{reference}_{mode}.csv'
        context.create_results(df[display_cols], rel, kws={'index': False, 'float_format': '%.3f'})
    formatters = {
        'n_units':  '{:>7,}'.format,
        'n_cols':   '{:>6,}'.format,
        'n_finite': '{:>9,}'.format,
        'R²':       '{:.4f}'.format,
        'slope':    '{:+.3f}'.format,
        'MAE':      '{:.3g}'.format,
        'RMSE':     '{:.3g}'.format,
        'MAE%':     '{:.1f}%'.format,
        'MAE%_max': '{:.1f}%'.format,
    }
    table = df[display_cols].to_string(index=False, formatters=formatters,
                                       justify='right')
    # Emit as one logging call — the level prefix only appears on the
    # first line, so the table rows stay column-aligned in the output.
    logging.info('\n' + table)

    # Per-column breakdown for any row whose aggregate R² fell below
    # `_LOW_R2_THRESHOLD` — worst `_LOW_R2_TOP_N` columns by R², to
    # spot boundary-effect columns without dumping the whole grid.
    low = [r for r in rows if r.get('_per_col') is not None]
    if not low:
        return
    col_formatters = {
        'n_finite': '{:>9,}'.format,
        'R²':       '{:.4f}'.format,
        'slope':    '{:+.3f}'.format,
        'MAE':      '{:.3g}'.format,
        'RMSE':     '{:.3g}'.format,
        'MAE%':     '{:.1f}%'.format,
    }
    for r in low:
        worst = sorted(
            r['_per_col'],
            key=lambda d: (float('inf') if not np.isfinite(d['r2']) else d['r2']),
        )[:_LOW_R2_TOP_N]
        worst_df = pd.DataFrame(worst).rename(columns={'r2': 'R²',
                                                       'slope': 'slope',
                                                       'mae': 'MAE',
                                                       'rmse': 'RMSE',
                                                       'mae_pct': 'MAE%'})
        col_table = worst_df[['column', 'n_finite', 'R²', 'slope',
                              'MAE', 'RMSE', 'MAE%']].to_string(
            index=False, formatters=col_formatters, justify='right')
        logging.info(
            f"\nWorst {len(worst)} column(s) by R² for "
            f"{r['grid_pair']}  {r['metric']}  {r['profile']} "
            f"(aggregate R²={r['R²']:.4f} < {_LOW_R2_THRESHOLD:.2f}):\n"
            + col_table)


if __name__ == '__main__':
    main()
