"""
Network-free accessibility: how much do the walking and cycling networks add?

For each walk / bike profile of the plan's `network_free` scenario (for
`switzerland-h10`: `res-h10`, Bern-Mittelland; see
`scenarios_technical_validation.TECHNICAL_VALIDATION`), the gross
door-to-door times are replaced by a function of
straight-line distance alone, on the same OD pairs (distance between cell /
zone centroids, as in 08b): the median network time of all pairs in the same
distance bin (`_DIST_BIN_M`), interpolated between bins and kept monotone. That
is the best network-free predictor of the scenario's own times, so the
comparison isolates what the network adds (routes, barriers, per-cell
overheads) rather than a difference in level. A straight line `a + b · d` is
not used: its intercept (~5-6 min) leaves the 0-5 min bin empty by
construction. The time grids are then computed exactly as in 10 (same floor,
cutoff, destinations and parameters) and compared cell by cell with the
network-based surfaces.

Inputs (the network-free scenario's own storage):
    odm/<mode>_time_gross_<profile>.npz, odm/<mode>_geo_pairs.npz     # 08a
    properties/cells_*.csv + shapes/cells.gpkg, zones likewise         # as in 10
    properties/cells_access_<grid>_<metric>_<profile>.csv              # 10 (baseline)

Outputs (RESULTS, under the headline scenario; the network-free surfaces
themselves are kept in memory):
    technical_validation/access_network_free.csv          # medians over destinations per
        # profile × grid × metric × parameter, plus the distance model's R²
    technical_validation/access_network_free_columns.csv  # one row per column

Run (headline scenario `switzerland-h10` by default):
    python -m technical_validation.access_network_free
"""

import importlib
import logging
from typing import cast

import numpy as np
import pandas as pd

from aperta import routing
from aperta.od_pairs import TieredODGeoPairs, get_euclidean_dists
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import bins_from_edges_min, gravity_decays_from_half_decay_min
from scenarios import get_scenario
from scenarios_technical_validation import technical_validation_plan
from technical_validation.access_vs_reference import by_param, column_stats

_m10 = importlib.import_module('main.10_accessibilities')

_MODES = ('walk', 'bike')
_DIST_BIN_M = 50.0      # distance bin width of the network-free time model
_REPORT_DIST_M = (250, 1_000, 3_000)   # distances at which the model's times are logged
_TIERS = ('cells_to_cells', 'cells_to_zones', 'zones_to_zones')


def _flat(odm: TieredODGeoPairs) -> np.ndarray:
    return np.concatenate([arr for t in _TIERS for arr in (getattr(odm, t) or {}).values()])


def _map_values(odm: TieredODGeoPairs, fn) -> TieredODGeoPairs:
    return TieredODGeoPairs(**{
        t: ({o: fn(arr) for o, arr in getattr(odm, t).items()}
            if getattr(odm, t) is not None else None)
        for t in _TIERS})


def _distance_model(gross: TieredODGeoPairs, dist: TieredODGeoPairs):
    """Median gross time per `_DIST_BIN_M` distance bin, made monotone; returns
    `(predict, r2)` with `predict(d_m) -> t_s` (linear interpolation between bin
    centres) and the R² of the predicted against the network times."""
    t, d = _flat(gross).astype(float), _flat(dist).astype(float)
    keep = np.isfinite(t) & np.isfinite(d)
    t, d = t[keep], d[keep]
    bins = (d // _DIST_BIN_M).astype(int)
    medians = pd.Series(t).groupby(bins).median()
    centres = (medians.index.to_numpy() + 0.5) * _DIST_BIN_M
    times = np.maximum.accumulate(medians.to_numpy())

    def predict(d_m: np.ndarray) -> np.ndarray:
        return np.interp(d_m, centres, times)

    resid = t - predict(d)
    r2 = 1.0 - (resid ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return predict, float(r2)


def _compare(inputs, grid_key: str, metric: str, label: str, tgt: pd.DataFrame) -> list[dict]:
    """Stats rows per column of the network-free `tgt` against the scenario's
    network-based surface (see `access_vs_reference.column_stats`)."""
    ref = inputs.get_properties('cells', f'access_{grid_key}_{metric}_{label}')
    cols = sorted(set(ref.select_dtypes('number').columns)
                  & set(tgt.select_dtypes('number').columns))
    idx = ref.index.intersection(tgt.index)
    return column_stats(tgt, ref, idx, cols).to_dict('records')


def main():
    context = init_context()
    plan = technical_validation_plan(context.scenario)
    # Inputs come from the network-free scenario; results go to the headline scenario.
    inputs = context.source(f'{context.project}/{plan.network_free}')
    scenario = get_scenario(plan.network_free)
    grids = {k: g for k, g in scenario.accessibility_grids.items()
             if g.travel_cost == 'time_gross'}

    with step(f'load cells + zones ({plan.network_free}, as in 10)'):
        prop_cols = ['population', 'snap', 'pois', 'employment']
        cells_m = inputs.get_properties('cells', prop_cols, add_shapes=True)
        zones_m = inputs.get_properties('zones', prop_cols, add_shapes=True)
        cells = cells_m[cells_m['is_active'] == 1].copy()
        cell_to_zone = cells_m['zone_id'].to_dict()
        centroids = pd.concat([
            cells_m.assign(geometry=cells_m.geometry.centroid)[['geometry']],
            zones_m.assign(geometry=zones_m.geometry.centroid)[['geometry']],
        ])
        all_dest_cols = (list(scenario.population_cols) + list(scenario.employment_cols)
                         + list(scenario.poi_cols) + list(scenario.transit_stops_cols))
        scaled_dests = set(scenario.population_cols) | set(scenario.employment_cols)

    rows: list[dict] = []
    for mode in _MODES:
        case = scenario.mode_configs.get(mode)
        if case is None:
            continue
        pairs_geo = cast(TieredODGeoPairs, inputs.get_tiered_odm(
            network_name=mode, data_name='geo_pairs'))
        with step(f'{mode}: straight-line distances on the network pairs'):
            node_dists = get_euclidean_dists(centroids, pairs_geo)
            dist = TieredODGeoPairs(**{t: getattr(node_dists, t) for t in _TIERS})

        for profile in case.profiles:
            label = case.source_for(profile).name
            gross = cast(TieredODGeoPairs, inputs.get_tiered_odm(
                network_name=mode, data_name=f'time_gross_{label}'))
            predict, r2 = _distance_model(gross, dist)
            at = predict(np.array(_REPORT_DIST_M, dtype=float)) / 60
            logging.info(f"  → {label}: network-free time model (R² = {r2:.3f}): "
                         + ', '.join(f'{d:,} m → {m:.1f} min' for d, m in zip(_REPORT_DIST_M, at)))
            cutoff = case.time_cutoff_s
            line = _map_values(dist, lambda d: np.where(
                predict(d) <= cutoff, predict(d), np.nan).astype(np.float32))
            line = routing.floor_intrazonal_costs(line, min_cost=case.min_route_time_s)

            for grid_key, grid in grids.items():
                with step(f'{label}: grid={grid_key} on straight-line times'):
                    dest_cols = (list(grid.dest_cols) if grid.dest_cols is not None
                                 else all_dest_cols)
                    metrics = _m10._compute_metrics(
                        line, pairs_geo, cells, cells_m, zones_m, cell_to_zone,
                        grid_key, False, mode, f'{label} (network-free)', dest_cols,
                        scaled_dests, bins_from_edges_min(grid.bin_edges_min),
                        list(grid.nearest_k),
                        gravity_decays_from_half_decay_min(grid.gravity_half_decay_min))
                for metric, tgt in metrics.items():
                    for r in _compare(inputs, grid_key, metric, label, tgt):
                        rows.append({'profile': label, 'grid': grid_key, 'metric': metric,
                                     'model_r2': r2, **r})

    with step('summary'):
        columns = pd.DataFrame(rows)
        df = by_param(columns, ['profile', 'model_r2', 'grid', 'metric'])
        context.create_results(columns, 'technical_validation/access_network_free_columns.csv',
                               kws={'index': False, 'float_format': '%.4f'})
        context.create_results(df, 'technical_validation/access_network_free.csv',
                               kws={'index': False, 'float_format': '%.4f'})
        pv = df.pivot_table(index=['grid', 'metric', 'reach', 'param'], columns='profile',
                            values=['R²_1to1', 'R²_fit', 'MAE%'], sort=False)
        logging.info('Network-free vs network-based (median over destinations):\n'
                     + pv.round(3).to_string())
    context.close()


if __name__ == '__main__':
    main()
