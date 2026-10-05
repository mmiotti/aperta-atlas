"""
Unified per-cell accessibility computation across all modes.

Every mode's cell-baked ODM has the same origin + destination shape,
so the metric loop is symmetric across walk / bike / car / transit.
Per (grid × mode × profile):
  1. Load the cell-baked ODM (fully-gross for road, or util variant
     from 09b) and its geo pair index (from 08a).
  2. Apply per-mode floor (`min_route_time_s` / `min_route_disutility`).
  3. Run three metric families (cumulative, nearest_k, gravity) per
     destination column; batches destinations to cap weight memory.
  4. Save `access_<grid>_<metric>_<profile>.csv`. Profile alone is
     unique across modes.

Inputs (PUBLIC, under `<scenario>/`):
    odm/<mode>_time_gross_<profile>.npz     # 08a — road gross-time
    odm/<mode>_geo_pairs.npz                # 08a — road geo pair index
    odm/transit_time_gross_npvm.npz         # 08a — transit gross-time
    odm/transit_geo_pairs.npz               # 08a — transit geo pair index
    odm/<mode>_dist_line.npz                # 08b — straight-line distance (dist_line grids only)
    odm/<mode>_util_<profile>_<var>.npz     # 09b — road disutility per variant
    odm/transit_util_npvm_<var>.npz         # 09b — transit disutility per variant
    properties/cells_*.csv + shapes/cells.gpkg
    properties/zones_*.csv + shapes/zones.gpkg

Output (PUBLIC, under `<scenario>/`):
    properties/cells_access_<grid>_<metric>_<profile>.csv

Run all grids (default):
    python -m main.10_accessibilities --scenario <name>
Single grid:
    python -m main.10_accessibilities --scenario <name> --variant <grid>
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

from aperta import accessibility, od_pairs, routing
from aperta.accessibility import exp_decay
from aperta.od_pairs import TieredODGeoPairs, TieredODPairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import (
    DEST_BATCH_SIZE,
    bins_from_edges_m,
    bins_from_edges_min,
    flatten_bin_dest_columns,
    flatten_decay_dest_columns,
    flatten_k_dest_columns,
    gravity_decays_from_half_decay_m,
    gravity_decays_from_half_decay_min,
    scaled_weights_for_nearest_k,
)
from mode_configs import TRANSIT_MODE_CONFIG, ModeConfig
from scenarios import get_scenario


def _iter_plan(scenario) -> list[tuple[str, str, ModeConfig]]:
    """Return `(mode, profile_label, case)` triples covering every road
    profile plus transit. `profile_label` is the CalibratedSource name
    for road (e.g. `'rwalk'`, `'car_peak'`) and `'transit'` for transit."""
    plan: list[tuple[str, str, ModeConfig]] = []
    for case in scenario.mode_configs.values():
        for profile in case.profiles:
            plan.append((case.mode, case.source_for(profile).name, case))
    plan.append(('transit', 'transit', TRANSIT_MODE_CONFIG))
    return plan


def _compute_and_save_metrics(
    context,
    costs_final: TieredODGeoPairs,
    pairs_geo: TieredODGeoPairs,
    cells: pd.DataFrame,
    cells_m: pd.DataFrame,
    zones_m: pd.DataFrame,
    cell_to_zone: dict,
    grid_key: str,
    grid,
    is_util: bool,
    mode: str,
    profile_label: str,
    dest_cols: list[str],
    scaled_dests: set[str],
    bins,
    ks,
    decays,
) -> None:
    """Run all three metric families for one (grid, mode, profile) triple
    and save one CSV per family. Batches destinations to cap
    simultaneous weight-ODM memory."""
    compute_cum = bool(bins)
    compute_nk = bool(ks)
    compute_grav = bool(decays)
    parts_cum: list = []
    parts_nk: list = []
    parts_grav: list = []
    n_batches = (len(dest_cols) + DEST_BATCH_SIZE - 1) // DEST_BATCH_SIZE
    for b_idx in range(0, len(dest_cols), DEST_BATCH_SIZE):
        batch = dest_cols[b_idx:b_idx + DEST_BATCH_SIZE]
        batch_no = b_idx // DEST_BATCH_SIZE + 1
        with step(f'grid={grid_key} mode={mode} profile={profile_label}: batch {batch_no}/{n_batches}'):
            batch_weights: dict[str, TieredODPairs] = {
                d: od_pairs.lookup_dest_column_geo(
                    d, pairs_geo, cells_m, zones=zones_m)
                for d in batch
            }
            batch_weights_nk = (
                scaled_weights_for_nearest_k(batch_weights, scaled_dests)
                if compute_nk else None
            )
            if compute_cum:
                parts_cum.append(accessibility.cumulative_opportunities(
                    costs_final, batch_weights, cell_to_zone, bins))
            if compute_nk:
                assert batch_weights_nk is not None
                # Default aggregator = cost_mean (weight-weighted mean cost
                # over the first k weight-units).
                parts_nk.append(accessibility.nearest_k(
                    costs_final, batch_weights_nk, cell_to_zone, ks=ks))
            if compute_grav:
                # Time / distance: raw Hansen gravity `Σ w·exp(-β·cost)`
                # (0 = no reachable destinations, a valid value).
                # Utility: `log(Σ w·exp(-β·D))` — at β=1 this is the
                # weighted logsum in native utility units; 0 → NaN
                # since log is undefined for unreachable cells.
                raw = accessibility.gravity(
                    costs_final, batch_weights, cell_to_zone, decays)
                if is_util:
                    parts_grav.append(np.log(raw.where(raw > 0)))
                else:
                    parts_grav.append(raw)

    # Concatenate + save per metric family. Empty profile_label
    # (dist_line, mode-agnostic) drops the trailing suffix.
    pref = 'access_'
    suffix = f'_{profile_label}' if profile_label else ''
    with step(f'grid={grid_key} mode={mode} profile={profile_label}: concat + save'):
        if compute_cum:
            cum = flatten_bin_dest_columns(pd.concat(parts_cum, axis=1))
            cum = cum.reindex(cells.index, fill_value=np.nan)
            context.create_properties(cum, f'{pref}{grid_key}_counts{suffix}', float_format='%.0f')
        if compute_nk:
            nk = flatten_k_dest_columns(pd.concat(parts_nk, axis=1))
            nk = nk.reindex(cells.index)
            fmt = '%.2f' if is_util else '%.1f'
            context.create_properties(nk, f'{pref}{grid_key}_nearest_k{suffix}', float_format=fmt)
        if compute_grav:
            gv = flatten_decay_dest_columns(pd.concat(parts_grav, axis=1))
            gv = gv.reindex(cells.index, fill_value=np.nan)
            context.create_properties(gv, f'{pref}{grid_key}_gravity{suffix}', float_format='%.3g')


def main():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--scenario', default=None,
                   help='(consumed by init_context)')
    p.add_argument('--variant', default='all',
                   help="'all' (default) or a specific grid key.")
    args, _ = p.parse_known_args()

    context = init_context()
    scenario = get_scenario(context.scenario)

    if args.variant == 'all':
        selected_grids = scenario.accessibility_grids
    else:
        if args.variant not in scenario.accessibility_grids:
            raise ValueError(
                f"--variant={args.variant!r} not in "
                f"scenario.accessibility_grids "
                f"(available: {sorted(scenario.accessibility_grids)}).")
        selected_grids = {
            args.variant: scenario.accessibility_grids[args.variant]}
    logging.info(f"running {len(selected_grids)} grid(s): {list(selected_grids)}")

    with step('load cells + zones (all properties + shapes)'):
        prop_cols = ['population', 'snap', 'pois', 'employment']
        cells_full = context.get_properties('cells', prop_cols, add_shapes=True)
        zones = context.get_properties(
            'zones', ['population', 'snap', 'pois', 'employment'],
            add_shapes=True)
        # `cells_m` covers ALL cells (including cross-border buffer) so
        # `lookup_dest_column_geo` can resolve buffer-cell destinations
        # produced by 08a/08b. `cells` is the is_active subset, used
        # only as the output index at each metric-family save (so CSVs
        # still cover Swiss-only origins).
        cells_m = cells_full.copy()
        zones_m = zones.copy()
        cells = cells_full[cells_full['is_active'] == 1].copy()
        cell_to_zone = cells_m['zone_id'].to_dict()
        logging.info(f"  → {len(cells_m):,} cells ({len(cells):,} active for output), "
                     f"{len(zones):,} zones")

    with step('assemble destination columns (pop + emp + poi from scenario)'):
        all_dest_cols: list[str] = (
            list(scenario.population_cols)
            + list(scenario.employment_cols)
            + list(scenario.poi_cols)
            + list(scenario.transit_stops_cols)
        )
        scaled_dests: set[str] = (
            set(scenario.population_cols) | set(scenario.employment_cols)
        )
        missing = [c for c in all_dest_cols if c not in cells.columns]
        if missing:
            raise ValueError(
                f"scenario declares destination columns {missing!r} that "
                f"aren't in cells (available: {sorted(cells.columns)}).")
        logging.info(
            f"  → {len(all_dest_cols)} destination columns available")

    # Overheads (road + transit) are pre-baked in 08a's gross ODMs.
    for grid_key, grid in selected_grids.items():
        is_util = grid.travel_cost == 'util'
        is_dist_line = grid.travel_cost == 'dist_line'
        ks = list(grid.nearest_k)
        if is_util:
            bins = bins_from_edges_min(grid.bin_edges_min)  # empty for util
            decays = [exp_decay(f'exp{b}', beta=b)
                      for b in grid.gravity_util_betas]
        elif is_dist_line:
            bins = bins_from_edges_m(grid.bin_edges_m)
            decays = gravity_decays_from_half_decay_m(grid.gravity_half_decay_m)
        else:
            bins = bins_from_edges_min(grid.bin_edges_min)
            decays = gravity_decays_from_half_decay_min(grid.gravity_half_decay_min)
        dest_cols = (list(grid.dest_cols) if grid.dest_cols is not None else all_dest_cols)
        logging.info(
            f"grid={grid_key!r} ({grid.travel_cost}): "
            f"cumulative={len(bins)} bins, nearest_k={len(ks)}, "
            f"gravity={len(decays)} decays, dest_cols={len(dest_cols)}")

        # dist_line is mode-agnostic — 08b produced one shared pair set +
        # distance ODM (`dist_line_geo_pairs.npz` / `dist_line_dist_line.npz`),
        # so a single iteration with `network_name='dist_line'` covers it.
        # Empty label drops the profile suffix from output filenames.
        if is_dist_line:
            iter_targets = [('dist_line', '', None)]
        else:
            iter_targets = _iter_plan(scenario)

        for mode, label, case in iter_targets:
            source = 'npvm' if mode == 'transit' else label
            if is_util:
                data_name = f'util_{source}_{grid.utility}'
            elif is_dist_line:
                data_name = 'dist_line'
            else:
                data_name = f'time_gross_{source}'

            # Missing util spec / skipped-upstream ODM → FileNotFoundError = skip.
            try:
                costs = cast(TieredODGeoPairs, context.get_tiered_odm(
                    network_name=mode, data_name=data_name))
            except FileNotFoundError:
                logging.info(f"  → grid={grid_key} mode={mode} profile={label}: ODM not found; skipping")
                continue

            # Geo pair index (dest ids). `lookup_dest_column_geo` needs
            # this — a cost ODM's value arrays would NaN-poison every
            # lookup. Shared across all profiles + util variants.
            pairs_geo = cast(TieredODGeoPairs, context.get_tiered_odm(
                network_name=mode, data_name='geo_pairs'))

            with step(f'grid={grid_key} mode={mode} profile={label}: floor'):
                if is_dist_line:
                    min_val = None  # straight-line distance has no natural floor
                else:
                    assert case is not None  # only dist_line passes case=None
                    min_val = (case.min_route_disutility if is_util
                               else case.min_route_time_s)
                costs_floored = (
                    routing.floor_intrazonal_costs(costs, min_cost=min_val)
                    if min_val is not None else costs)

            _compute_and_save_metrics(
                context, costs_floored, pairs_geo, cells, cells_m,
                zones_m, cell_to_zone, grid_key, grid, is_util,
                mode, label, dest_cols, scaled_dests, bins, ks, decays)

    context.close()


if __name__ == '__main__':
    main()
