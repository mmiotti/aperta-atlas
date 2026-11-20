"""
Effective reach of each released accessibility setting.

Count bins, nearest-k and gravity weight destinations very differently, so
their parameters are not comparable by their numbers (a 300-m half-decay is
not a 300-m bin). For each setting, this script measures the straight-line
distance and the door-to-door travel time within which half of the metric's
weight lies (r50), from the actual destinations and costs:

    count bin (lo, hi]   weight w_d for destinations in the bin
    cumulative (0, hi]   weight w_d for destinations up to hi (sums of bins)
    nearest-k            the k nearest destination units (fractional at the k-th;
                         residents and jobs in units of 100, as in 10)
    gravity              w_d · exp(−β c_d)

Weights are stitched across the three OD tiers as in aperta's metric functions,
with the same per-mode cost floors as 10. Per origin, r50 is the weighted median
of the straight-line distance (and of the mode's door-to-door time) over the
weighted destinations; the output gives its median over a random sample of
active AOI cells, for all of them and split by the residents and jobs within
1 km straight-line (`DENSITY_CLASSES`). Settings with similar r50 measure
accessibility at a similar scale, whatever their metric family. The cross-modal
utility profiles of 10 get r50 in km only, as they have no single travel time.

Inputs (scenario storage, under `<scenario>/`):
    odm/<mode>_{geo_pairs,time_gross_<profile>,util_<profile>_default}.npz
    odm/transit_{geo_pairs,time_gross_npvm,util_npvm_default}.npz
    odm/dist_line_geo_pairs.npz
    properties/cells_*.csv + shapes/cells.gpkg, zones likewise

Output (scenario storage, under `<scenario>/`):
    generic/access_reach.csv    # one row per grid × profile × metric × parameter ×
                                # destination × origin class: r50_km, r50_min

Run:
    python -m main.11_access_reach --scenario <name> [--n-origins 2000] [--seed 0]
"""

import argparse
import importlib
import logging
from typing import cast

import numpy as np
import pandas as pd

from aperta import routing
from aperta.accessibility import _stitched_for, _stitched_weights
from aperta.od_pairs import (
    TieredODGeoPairs,
    aggregate_across_modes,
    get_euclidean_dists,
    lookup_dest_column_geo,
)
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from scenarios import get_scenario

_m10 = importlib.import_module('main.10_accessibilities')

_TIERS = ('cells_to_cells', 'cells_to_zones', 'zones_to_zones')
_LN2 = np.log(2.0)
_QUANTILES = (0.5, 0.9)
_GRID_NAMES = {'time_gross': 'time', 'util': 'util', 'dist_line': 'dist'}
# Origin classes by residents + jobs within 1 km (straight line); ~300 and
# ~3,000 per km².
DENSITY_CLASSES = (('rural', 0, 1_000), ('suburban', 1_000, 10_000),
                   ('urban', 10_000, np.inf))
_STATS = ('r50_km', 'r90_km', 'r50_min', 'r90_min')


def _kernels(grid) -> list[tuple[str, str, object]]:
    """(metric, parameter label, kernel) per setting of a grid, labelled like the
    released columns (main.common). A kernel maps the costs of the grid's own unit
    (s, disutility, m) to per-destination factors; nearest-k kernels are the
    integer k (see `_nearest_k_take`)."""
    out: list[tuple[str, str, object]] = []
    # Bins are half-open, lo <= cost < hi, as in `aperta.accessibility` (stage 10's counts).
    if grid.travel_cost == 'time_gross':
        e = grid.bin_edges_min
        out += [('counts', f'{a}_{b}min', (lambda c, lo=a * 60, hi=b * 60: (c >= lo) & (c < hi)))
                for a, b in zip(e, e[1:])]
        out += [('cumulative', f'0_{b}min', (lambda c, hi=b * 60: c < hi)) for b in e[2:]]
        out += [('gravity', f'exp{h}min', (lambda c, beta=_LN2 / (h * 60): np.exp(-beta * c)))
                for h in grid.gravity_half_decay_min]
    elif grid.travel_cost == 'dist_line':
        e = grid.bin_edges_m
        out += [('counts', f'{int(a)}_{int(b)}m', (lambda c, lo=a, hi=b: (c >= lo) & (c < hi)))
                for a, b in zip(e, e[1:])]
        out += [('cumulative', f'0_{int(b)}m', (lambda c, hi=b: c < hi)) for b in e[2:]]
        out += [('gravity', f'exp{int(h)}m', (lambda c, beta=_LN2 / h: np.exp(-beta * c)))
                for h in grid.gravity_half_decay_m]
    else:  # util
        out += [('gravity', f'exp{b}', (lambda c, beta=b: np.exp(-beta * c)))
                for b in grid.gravity_util_betas]
    out += [('nearest_k', f'k{k}', k) for k in grid.nearest_k]
    return out


def _nearest_k_take(order: np.ndarray, w: np.ndarray, k: float) -> np.ndarray:
    """Units of each destination among the k nearest (by the grid's cost)."""
    w_sorted = w[order]
    before = np.cumsum(w_sorted) - w_sorted
    take_sorted = np.clip(k - before, 0.0, w_sorted)
    take = np.empty_like(w)
    take[order] = take_sorted
    return take


def _weighted_quantiles(x_sorted: np.ndarray, order: np.ndarray, weights: np.ndarray,
                        qs) -> list[float]:
    """Weighted quantiles of x (pre-sorted, with its sort order)."""
    cum = np.cumsum(weights[order])
    total = cum[-1] if cum.size else 0.0
    if not np.isfinite(total) or total <= 0:
        return [np.nan] * len(qs)
    idx = np.searchsorted(cum, np.asarray(qs) * total, side='left')
    return [float(x_sorted[min(i, len(x_sorted) - 1)]) for i in idx]


def _stitch(odm: TieredODGeoPairs, origin, zone, n: tuple[int, int, int]) -> np.ndarray:
    return _stitched_weights(origin, zone, odm, *n, dtype=np.float64)


def _reach_for(cost: TieredODGeoPairs, dist: TieredODGeoPairs, time: TieredODGeoPairs | None,
               weights: dict[str, TieredODGeoPairs], kernels, origins, cell_to_zone,
               scaled: set[str]) -> dict[tuple[str, str, str], np.ndarray]:
    """`{(dest, metric, param): origins × (r50_km, r90_km, r50_min, r90_min)}`.
    Origins missing from `cost` (no accessibility in 10 either) stay NaN."""
    out: dict[tuple[str, str, str], np.ndarray] = {}
    missing = [o for o in origins if o not in cost.cells_to_cells]
    if missing:
        logging.warning(f"  ⚠ {len(missing)} of {len(origins)} sampled origins missing from the "
                        f"cost ODM (e.g. {missing[:3]}); left out")
    for i, origin in enumerate(origins):
        if origin not in cost.cells_to_cells:
            continue
        c, cell_sl, c2z_sl, zone_sl = _stitched_for(origin, cost, cell_to_zone)
        n = (cell_sl.stop, c2z_sl.stop - c2z_sl.start, zone_sl.stop - zone_sl.start)
        zone = cell_to_zone.get(origin)
        d = _stitch(dist, origin, zone, n)
        t = _stitch(time, origin, zone, n) / 60.0 if time is not None else None
        ok = np.isfinite(c) & np.isfinite(d)
        c, d = c[ok], d[ok] / 1_000.0
        t = t[ok] if t is not None else None
        order = np.argsort(c, kind='stable')
        d_order = np.argsort(d)
        d_sorted = d[d_order]
        if t is not None:
            t_order = np.argsort(t)
            t_sorted = t[t_order]
        for dest, odm in weights.items():
            w = np.nan_to_num(_stitch(odm, origin, zone, n)[ok])
            w = np.where(w > 0, w, 0.0)
            if not w.any():
                continue
            w_k = w / 100.0 if dest in scaled else w
            for metric, param, kernel in kernels:
                if metric == 'nearest_k':
                    if w_k.sum() < kernel:
                        continue
                    f = _nearest_k_take(order, w_k, float(kernel))
                else:
                    f = w * kernel(c)
                arr = out.setdefault((dest, metric, param),
                                     np.full((len(origins), 4), np.nan, dtype=np.float32))
                arr[i, :2] = _weighted_quantiles(d_sorted, d_order, f, _QUANTILES)
                if t is not None:
                    arr[i, 2:] = _weighted_quantiles(t_sorted, t_order, f, _QUANTILES)
    return out


def _subset_origins(odm: TieredODGeoPairs, cells, zones) -> TieredODGeoPairs:
    """`odm` restricted to the origin cells `cells` (and their zones `zones`)."""
    def sub(tier, keys):
        return None if tier is None else {k: tier[k] for k in keys if k in tier}
    return TieredODGeoPairs(cells_to_cells=sub(odm.cells_to_cells, cells),
                            cells_to_zones=sub(odm.cells_to_zones, cells),
                            zones_to_zones=sub(odm.zones_to_zones, zones))


def _crossmodal_reach_for(costs: list[TieredODGeoPairs], dist: TieredODGeoPairs,
                          weights: dict[str, TieredODGeoPairs], betas, origins,
                          cell_to_zone) -> dict[tuple[str, str, str], np.ndarray]:
    """As `_reach_for`, for 10's cross-modal profile (gravity, km only: there is no
    single travel time). `costs` are the per-mode disutilities D_m on shared pairs.
    A destination's weight follows 10's construction: w·(Σ_m e^(−D_m))^β for β ≤ 1
    (modes combined per pair), and for β > 1 (modes above destinations) the mode
    shares ∝ S_m^(1/β), S_m = Σ_d w·e^(−β·D_m), times each mode's destination
    distribution, i.e. w·Σ_m S_m^(1/β − 1)·e^(−β·D_m)."""
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for i, origin in enumerate(origins):
        if origin not in costs[0].cells_to_cells:
            continue
        stitched = [_stitched_for(origin, c, cell_to_zone) for c in costs]
        _, cell_sl, c2z_sl, zone_sl = stitched[0]
        n = (cell_sl.stop, c2z_sl.stop - c2z_sl.start, zone_sl.stop - zone_sl.start)
        zone = cell_to_zone.get(origin)
        d = _stitch(dist, origin, zone, n) / 1_000.0
        e = np.nan_to_num(np.exp(-np.stack([s[0] for s in stitched]).astype(np.float64)))
        ok = np.isfinite(d) & (e.sum(axis=0) > 0)
        d, e = d[ok], e[:, ok]
        d_order = np.argsort(d)
        d_sorted = d[d_order]
        for dest, odm in weights.items():
            w = np.nan_to_num(_stitch(odm, origin, zone, n)[ok])
            w = np.where(w > 0, w, 0.0)
            if not w.any():
                continue
            for b in betas:
                if b <= 1:
                    f = w * e.sum(axis=0) ** b
                else:
                    eb = e ** b
                    s = eb @ w
                    use = s > 0
                    f = w * (s[use] ** (1 / b - 1) @ eb[use])
                arr = out.setdefault((dest, 'gravity', f'exp{b}'),
                                     np.full((len(origins), 4), np.nan, dtype=np.float32))
                arr[i, :2] = _weighted_quantiles(d_sorted, d_order, f, _QUANTILES)
    return out


def _crossmodal_part(context, scenario, grid, car_profile: str, origins, cell_to_zone,
                     centroids, cells_m, zones_m, dests) -> dict:
    """Reach of the cross-modal profile `crossmodal_<car_profile>` on the sampled
    origins: the four modes' ODMs, cut to those origins, aligned on their union pairs."""
    zones = {cell_to_zone[o] for o in origins if o in cell_to_zone}
    odms = {label: tuple(_subset_origins(x, origins, zones) for x in pc)
            for label, pc in _m10._crossmodal_inputs(context, scenario, grid,
                                                      car_profile).items()}
    costs = []
    for m in range(len(odms)):   # each mode's D on the union pairs (missing: +inf)
        union, aligned = aggregate_across_modes(odms, aggregator=lambda s, m=m: s[m])
        costs.append(aligned)
    node_d = get_euclidean_dists(centroids, union)
    dist = TieredODGeoPairs(**{t: getattr(node_d, t) for t in _TIERS})
    weights = {d: lookup_dest_column_geo(d, union, cells_m, zones=zones_m) for d in dests
               if grid.dest_cols is None or d in grid.dest_cols}
    return _crossmodal_reach_for(costs, dist, weights, grid.gravity_util_betas, origins,
                                 cell_to_zone)


def _append_rows(rows: list[dict], part: dict, strata, grid_name: str, label: str) -> None:
    """One row per (destination, metric, parameter, origin class) of `part`: the
    median and quartiles over the origins of each statistic."""
    for (dest, metric, param), arr in part.items():
        for stratum, mask in strata:
            sub = arr[mask]
            r = {'grid': grid_name, 'profile': label, 'metric': metric,
                 'param': param, 'dest': dest, 'origins': stratum,
                 'n_origins': int(np.isfinite(sub[:, 0]).sum())}
            for j, name in enumerate(_STATS):
                finite = np.isfinite(sub[:, j])
                for stat, q in (('median', 50), ('p25', 25), ('p75', 75)):
                    r[f'{name}_{stat}'] = (float(np.percentile(sub[finite, j], q))
                                           if finite.any() else np.nan)
            rows.append(r)


def _people_within(dist: TieredODGeoPairs, weights: dict[str, TieredODGeoPairs], origins,
                   cell_to_zone, radius_m: float, dests: list[str]) -> np.ndarray:
    """Sum of `dests` (residents + jobs) within `radius_m` of each origin."""
    out = np.zeros(len(origins))
    for i, origin in enumerate(origins):
        d, cell_sl, c2z_sl, zone_sl = _stitched_for(origin, dist, cell_to_zone)
        n = (cell_sl.stop, c2z_sl.stop - c2z_sl.start, zone_sl.stop - zone_sl.start)
        zone = cell_to_zone.get(origin)
        near = np.isfinite(d) & (d <= radius_m)
        out[i] = sum(np.nansum(_stitch(weights[c], origin, zone, n)[near]) for c in dests)
    return out


def compute_reach(context, n_origins: int, seed: int) -> pd.DataFrame:
    """Reach of every setting of the scenario's grids: one row per grid × profile ×
    metric × parameter × destination × origin class, with the median and quartiles
    over the sampled origins of r50 / r90 in km and min (time and utility grids)."""
    scenario = get_scenario(context.scenario)
    grids = {_GRID_NAMES[g.travel_cost]: g for g in scenario.accessibility_grids.values()}

    with step('load cells + zones, sample origins'):
        prop_cols = ['population', 'snap', 'pois', 'employment']
        cells_m = context.get_properties('cells', prop_cols, add_shapes=True)
        zones_m = context.get_properties('zones', prop_cols, add_shapes=True)
        cell_to_zone = cells_m['zone_id'].to_dict()
        centroids = pd.concat([
            cells_m.assign(geometry=cells_m.geometry.centroid)[['geometry']],
            zones_m.assign(geometry=zones_m.geometry.centroid)[['geometry']],
        ])
        dests = [c for c in (list(scenario.population_cols) + list(scenario.employment_cols)
                             + list(scenario.poi_cols) + list(scenario.transit_stops_cols))
                 if c in cells_m.columns]
        scaled = set(scenario.population_cols) | set(scenario.employment_cols)
        totals = [c for c in ('population_total', 'employment_total') if c in cells_m.columns]
        pool = cells_m.index[(cells_m['is_active'] == 1) & (cells_m['is_aoi'] == 1)]
        rng = np.random.default_rng(seed)
        origins = list(rng.choice(pool.to_numpy(), size=min(n_origins, len(pool)),
                                  replace=False))
        logging.info(f"  → {len(origins):,} origins of {len(pool):,}; {len(dests)} destinations")

    pair_cache: dict[str, tuple] = {}

    def _pairs(mode: str) -> tuple:
        if mode not in pair_cache:
            pairs = cast(TieredODGeoPairs, context.get_tiered_odm(
                network_name=mode, data_name='geo_pairs'))
            node_d = get_euclidean_dists(centroids, pairs)
            dist = TieredODGeoPairs(**{t: getattr(node_d, t) for t in _TIERS})
            weights = {d: lookup_dest_column_geo(d, pairs, cells_m, zones=zones_m)
                       for d in dests}
            pair_cache[mode] = (dist, weights)
        return pair_cache[mode]

    with step('origin classes (residents + jobs within 1 km)'):
        dist, weights = _pairs('dist_line')
        density = _people_within(dist, weights, origins, cell_to_zone, 1_000.0, totals)
        strata = [('all', np.ones(len(origins), bool))] + [
            (name, (density >= lo) & (density < hi)) for name, lo, hi in DENSITY_CLASSES]
        logging.info('  → ' + ', '.join(f'{name} {int(mask.sum())}' for name, mask in strata))

    targets = [('dist', 'dist_line', '', None)] if 'dist' in grids else []
    targets += [(grid, mode, label, case) for mode, label, case in _m10._iter_plan(scenario)
                for grid in ('time', 'util') if grid in grids]
    rows: list[dict] = []
    time_cache: dict[str, TieredODGeoPairs] = {}
    for grid_name, mode, label, case in targets:
        with step(f'{grid_name} / {mode} {label}'):
            grid = grids[grid_name]
            dist, weights = _pairs(mode)
            if grid.dest_cols is not None:
                weights = {d: w for d, w in weights.items() if d in grid.dest_cols}
            source = 'npvm' if mode == 'transit' else label
            if grid_name == 'dist':
                cost, time = dist, None   # metres
            else:
                if grid_name == 'util':
                    # As in 10: profiles without a utility model have no util ODM.
                    try:
                        cost = cast(TieredODGeoPairs, context.get_tiered_odm(
                            network_name=mode, data_name=f'util_{source}_{grid.utility}'))
                    except FileNotFoundError:
                        logging.info(f"  → no utility ODM for {label}; skipping")
                        continue
                if label not in time_cache:   # the time and util grids share it
                    time_cache.clear()
                    time_cache[label] = routing.floor_intrazonal_costs(
                        cast(TieredODGeoPairs, context.get_tiered_odm(
                            network_name=mode, data_name=f'time_gross_{source}')),
                        min_cost=case.min_route_time_s)
                time = time_cache[label]
                if grid_name == 'time':
                    cost = time
            part = _reach_for(cost, dist, time, weights, _kernels(grid), origins,
                              cell_to_zone, scaled)
            _append_rows(rows, part, strata, grid_name, label)
    for car_profile in (grids['util'].crossmodal_car_profiles if 'util' in grids else ()):
        label = f'crossmodal_{car_profile}'
        with step(f'util / {label}'):
            part = _crossmodal_part(context, scenario, grids['util'], car_profile, origins,
                                    cell_to_zone, centroids, cells_m, zones_m, dests)
            _append_rows(rows, part, strata, 'util', label)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--n-origins', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args, _ = parser.parse_known_args()

    context = init_context()
    full = compute_reach(context, args.n_origins, args.seed)

    with step('save generic/access_reach.csv'):
        out = full[['grid', 'profile', 'metric', 'param', 'dest', 'origins', 'n_origins',
                    'r50_km_median', 'r50_min_median']].rename(
            columns={'r50_km_median': 'r50_km', 'r50_min_median': 'r50_min'})
        context.create_generic(out, 'access_reach.csv',
                               kws={'index': False, 'float_format': '%.3f'})
        groceries = out[(out['dest'] == 'poi_errands_groceries') & (out['metric'] != 'counts')]
        for origins in ('all', 'urban', 'rural'):
            pv = groceries[groceries['origins'] == origins].pivot_table(
                index=['grid', 'metric', 'param'], columns='profile', values='r50_km',
                sort=False)
            logging.info(f'r50 (km), groceries, {origins} origins:\n' + pv.round(2).to_string())
    context.close()


if __name__ == '__main__':
    main()
