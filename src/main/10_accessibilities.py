"""
Unified per-cell accessibility computation across all modes.

Every mode's cell-baked ODM has the same origin + destination shape,
so the metric loop is symmetric across walk / bike / car / transit.
Per (grid × mode × profile):
  1. Load the cell-baked cost ODM (gross time from 08a, disutility from
     09b, or straight-line distance from 08b) and its geo pair index.
  2. Apply the per-mode time floor (`min_route_time_s`; disutilities are built
     from floored times in 09b).
  3. Run three metric families (bin counts, nearest_k, gravity) per
     destination column; batches destinations to cap weight memory.
  4. Save `access_<grid>_<metric>_<profile>.csv`. Profile alone is
     unique across modes; mode-agnostic dist_line grids have no suffix.
Util grids with `crossmodal_car_profiles` add one profile
`crossmodal_<car profile>` each, over the four modes of the mode-choice
model and in the order consistent with utility maximization: for β ≤ 1,
the disutilities are combined per OD pair (logsum), then gravity runs
over the destinations (see `_crossmodal_odm`); for β > 1, the per-mode
logsums are combined (`_logsum_across_modes`). Both agree at β = 1.
Cycling and transit have one profile (base hours); in the night and peak
cross-modal profiles, their utilities shift by 09a's hour constants
(`crossmodal_hour_shift`), on top of the floored disutilities, so that
both orders shift alike.

Inputs (scenario storage, under `<scenario>/`):
    odm/<mode>_time_gross_<profile>.npz     # 08a — road gross-time
    odm/<mode>_geo_pairs.npz                # 08a — road geo pair index
    odm/transit_time_gross_npvm.npz         # 08a — transit gross-time
    odm/transit_geo_pairs.npz               # 08a — transit geo pair index
    odm/dist_line_{geo_pairs,dist_line}.npz # 08b — straight-line distance (dist_line grids only)
    odm/<mode>_util_<profile>_<var>.npz     # 09b — road disutility per variant
    odm/transit_util_npvm_<var>.npz         # 09b — transit disutility per variant
    properties/cells_*.csv + shapes/cells.gpkg
    properties/zones_*.csv + shapes/zones.gpkg

Output (scenario storage, under `<scenario>/`):
    properties/cells_access_<grid>_<metric>[_<profile>].csv
    properties/cells_access_util_gravity_crossmodal_<car profile>.csv   # cross-modal

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
from aperta.errors import DataError
from aperta.od_pairs import TieredODGeoPairs, TieredODPairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import (
    DEST_BATCH_SIZE,
    bins_from_edges_m,
    bins_from_edges_min,
    crossmodal_hour_shift,
    flatten_bin_dest_columns,
    flatten_decay_dest_columns,
    flatten_k_dest_columns,
    gravity_decays_from_half_decay_m,
    gravity_decays_from_half_decay_min,
    gravity_decays_util,
    scaled_weights_for_nearest_k,
    util_logsums,
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
    if scenario.transit_source is not None:
        plan.append(('transit', 'transit', TRANSIT_MODE_CONFIG))
    return plan


_TIERS = ('cells_to_cells', 'cells_to_zones', 'zones_to_zones')


def _crossmodal_profiles(car_profile: str) -> tuple[str, ...]:
    """The four modes of the mode-choice model, as profile labels. E-bike and
    reduced-mobility walking are not in the model."""
    return ('rwalk', 'rbike', f'car_{car_profile}', 'transit')


def _util_beta(column: str) -> float:
    """β of a util gravity column `<destination>_exp<β>`."""
    return float(column.rsplit('_exp', 1)[1])


def _logsum_across_modes(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """`ln Σ_m exp(a_m)` over per-mode logsums (frames with the same cells and
    columns): the cross-modal logsum with modes above destinations, consistent
    with utility maximization for β ≥ 1. A mode without a value contributes
    nothing; NaN where no mode has one."""
    assert all(p.columns.equals(parts[0].columns) and p.index.equals(parts[0].index)
               for p in parts)
    stacked = np.exp(np.stack([p.to_numpy(dtype=float) for p in parts]))
    with np.errstate(divide='ignore'):
        out = np.log(np.nansum(stacked, axis=0))
    out[np.isnan(stacked).all(axis=0)] = np.nan
    return pd.DataFrame(out, index=parts[0].index, columns=parts[0].columns)


def _shift_costs(costs: TieredODGeoPairs, shift: float) -> TieredODGeoPairs:
    """`costs − shift` on every pair: a disutility at another hour (non-finite entries stay)."""
    def tier(t):
        return None if t is None else {o: a - shift for o, a in t.items()}
    return TieredODGeoPairs(**{t: tier(getattr(costs, t)) for t in _TIERS})


def _crossmodal_inputs(context, scenario, grid, car_profile: str
                       ) -> dict[str, tuple[TieredODGeoPairs, TieredODGeoPairs]]:
    """`{profile: (pairs, disutility)}` of the four modes of the mode-choice model,
    each shifted to the car's hour (`crossmodal_hour_shift`; also used by 11)."""
    wanted = set(_crossmodal_profiles(car_profile))
    coefs = context.get_coefs(f'utility_{grid.utility}')
    odms: dict[str, tuple[TieredODGeoPairs, TieredODGeoPairs]] = {}
    for mode, label, case in _iter_plan(scenario):
        if label not in wanted:
            continue
        source = 'npvm' if mode == 'transit' else label
        with step(f'crossmodal_{car_profile}: load {label}'):
            costs = cast(TieredODGeoPairs, context.get_tiered_odm(
                network_name=mode, data_name=f'util_{source}_{grid.utility}'))
            shift = crossmodal_hour_shift(coefs, label, car_profile)
            if shift:
                costs = _shift_costs(costs, shift)
                logging.info(f"  → {label}: utility shift {shift:+.3f} at {car_profile} hours")
            pairs = context.get_tiered_odm(network_name=mode, data_name='geo_pairs')
        # The loader returns node-pair objects; these ODMs are cell/zone-keyed. Dest IDs stay
        # the loader's (cached) lists: converting them to arrays would copy every ID.
        odms[label] = cast(tuple[TieredODGeoPairs, TieredODGeoPairs], tuple(
            TieredODGeoPairs(**{t: getattr(x, t) for t in _TIERS}) for x in (pairs, costs)))
    if set(odms) != wanted:
        raise DataError(f"crossmodal_{car_profile}: profiles {sorted(wanted - set(odms))} "
                        f"missing from the scenario's mode plan.")
    return odms


def _crossmodal_odm(context, scenario, grid, car_profile: str
                    ) -> tuple[TieredODGeoPairs, TieredODGeoPairs]:
    """`(pairs, disutility)` per OD pair across the four modes of the mode-choice
    model: `-ln Σ_m exp(-D_m)` over walking (`rwalk`), cycling (`rbike`), car
    (`car_<car_profile>`) and transit; a mode that lacks a pair contributes
    nothing.

    Combining the modes per pair keeps mode choice as estimated (09a, given the
    destination); gravity with β ≤ 1 on the result is the logsum of a nested
    choice with destinations above modes."""
    odms = _crossmodal_inputs(context, scenario, grid, car_profile)
    with step(f'crossmodal_{car_profile}: combine modes per OD pair'):
        return od_pairs.aggregate_across_modes(odms, aggregator='logsum')


def _compute_metrics(
    costs_final: TieredODGeoPairs,
    pairs_geo: TieredODGeoPairs,
    cells: pd.DataFrame,
    cells_m: pd.DataFrame,
    zones_m: pd.DataFrame,
    cell_to_zone: dict,
    grid_key: str,
    is_util: bool,
    mode: str,
    profile_label: str,
    dest_cols: list[str],
    scaled_dests: set[str],
    bins,
    ks,
    decays,
) -> dict[str, pd.DataFrame]:
    """Run all three metric families for one (grid, mode, profile) triple;
    returns `{family: cells × columns}` for the families with parameters
    (`counts`, `nearest_k`, `gravity`). Batches destinations to cap
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
            if compute_cum:
                parts_cum.append(accessibility.cumulative_opportunities(
                    costs_final, batch_weights, cell_to_zone, bins))
            if compute_nk:
                # Default aggregator = cost_mean (weight-weighted mean cost
                # over the first k weight-units).
                parts_nk.append(accessibility.nearest_k(
                    costs_final, scaled_weights_for_nearest_k(batch_weights, scaled_dests),
                    cell_to_zone, ks=ks))
            if compute_grav:
                # Time / distance: raw Hansen gravity `Σ w·exp(-β·cost)`
                # (0 = no reachable destinations, a valid value).
                # Utility: logsum `(1/β)·ln Σ w·exp(-β·D)` in utility
                # units for every β; NaN for unreachable cells.
                raw = accessibility.gravity(
                    costs_final, batch_weights, cell_to_zone, decays)
                parts_grav.append(util_logsums(raw) if is_util else raw)

    metrics: dict[str, pd.DataFrame] = {}
    if compute_cum:
        cum = flatten_bin_dest_columns(pd.concat(parts_cum, axis=1))
        metrics['counts'] = cum.reindex(cells.index, fill_value=np.nan)
    if compute_nk:
        nk = flatten_k_dest_columns(pd.concat(parts_nk, axis=1))
        metrics['nearest_k'] = nk.reindex(cells.index)
    if compute_grav:
        gv = flatten_decay_dest_columns(pd.concat(parts_grav, axis=1))
        metrics['gravity'] = gv.reindex(cells.index, fill_value=np.nan)
    return metrics


def _round_significant(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    """`df` rounded to `digits` significant digits (NaN and 0 kept)."""
    x = df.to_numpy(dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        exponent = np.floor(np.log10(np.abs(x)))
        scale = np.where(np.isfinite(exponent), 10.0 ** (digits - 1 - exponent), 1.0)
        rounded = np.round(x * scale) / scale
    return pd.DataFrame(np.where(np.isfinite(exponent), rounded, x),
                        index=df.index, columns=df.columns)


def _save_metrics(context, metrics: dict[str, pd.DataFrame], grid_key: str, is_util: bool,
                  profile_label: str) -> None:
    """One `cells_access_<grid>_<family>[_<profile>]` CSV per metric family.
    Empty `profile_label` (dist_line, mode-agnostic) drops the suffix.

    Precision: counts whole; nearest-k in whole seconds / metres (utility: 2 decimals);
    gravity 3 significant digits, written out rather than in exponent notation from 1,000 up
    (logsums: 2 decimals, as they are read as differences)."""
    formats = {'counts': '%.0f', 'nearest_k': '%.2f' if is_util else '%.0f',
               'gravity': '%.2f' if is_util else '%.10g'}
    suffix = f'_{profile_label}' if profile_label else ''
    for family, df in metrics.items():
        if family == 'gravity' and not is_util:
            df = _round_significant(df, 3)
        context.create_properties(df, f'access_{grid_key}_{family}{suffix}',
                                  float_format=formats[family])


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
        zones = context.get_properties('zones', prop_cols, add_shapes=True)
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
            decays = gravity_decays_util(grid.gravity_util_betas)
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
        iter_targets = [('dist_line', '', None)] if is_dist_line else _iter_plan(scenario)
        # Per-mode util gravity for β > 1, kept for the cross-modal profiles.
        crossmodal_inputs = {p for c in grid.crossmodal_car_profiles
                             for p in _crossmodal_profiles(c)}
        above_one: dict[str, pd.DataFrame] = {}

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
                # Distance has no natural floor; disutilities are built from floored
                # times (09b).
                costs_floored = (
                    costs if is_dist_line or is_util
                    else routing.floor_intrazonal_costs(
                        costs, min_cost=cast(ModeConfig, case).min_route_time_s))

            metrics = _compute_metrics(
                costs_floored, pairs_geo, cells, cells_m, zones_m, cell_to_zone,
                grid_key, is_util, mode, label, dest_cols, scaled_dests, bins, ks, decays)
            with step(f'grid={grid_key} mode={mode} profile={label}: save'):
                _save_metrics(context, metrics, grid_key, is_util, label)
            if label in crossmodal_inputs and 'gravity' in metrics:
                g = metrics['gravity']
                above_one[label] = g[[c for c in g.columns if _util_beta(c) > 1]]

        # Cross-modal (util grids, gravity only), in the order consistent with utility
        # maximization: the less sensitive choice sits higher in the nesting. For β ≤ 1
        # (destinations chosen no more sensitively than modes), combine the modes per OD
        # pair, then the destinations; for β > 1, combine the per-mode logsums.
        betas_per_pair = tuple(b for b in grid.gravity_util_betas if b <= 1)
        for car_profile in grid.crossmodal_car_profiles:
            label = f'crossmodal_{car_profile}'
            pairs_xm, costs_xm = _crossmodal_odm(context, scenario, grid, car_profile)
            metrics = _compute_metrics(
                costs_xm, pairs_xm, cells, cells_m, zones_m, cell_to_zone,
                grid_key, is_util, 'crossmodal', label, dest_cols, scaled_dests, [], [],
                gravity_decays_util(betas_per_pair))
            del pairs_xm, costs_xm
            # A per-mode logsum shifts by exactly the mode's hour shift (as in
            # `_crossmodal_inputs`, applied after the floor).
            util_coefs = context.get_coefs(f'utility_{grid.utility}')
            parts = [above_one[p] + crossmodal_hour_shift(util_coefs, p, car_profile)
                     for p in _crossmodal_profiles(car_profile)]
            if len(parts[0].columns):
                metrics['gravity'] = pd.concat(
                    [metrics.get('gravity'), _logsum_across_modes(parts)], axis=1)
            metrics['gravity'] = metrics['gravity'][
                [f'{d}_exp{b}' for d in dest_cols for b in grid.gravity_util_betas]]
            with step(f'grid={grid_key} profile={label}: save'):
                _save_metrics(context, metrics, grid_key, is_util, label)

    context.close()


if __name__ == '__main__':
    main()
