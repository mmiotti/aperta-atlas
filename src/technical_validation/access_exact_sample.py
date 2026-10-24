"""
Untiered ("exact") accessibility on a sample of origins, as a ground truth
for the tiered OD approximation (radii-* / res-h11 technical validation).

For a random sample of AOI origin cells, every destination cell reachable
within the mode's time cutoff is routed exactly (no cells_to_zones /
zones_to_zones tiers, no radii). The routed net times then go through the
same steps as the pipeline — floor, α scaling, per-cell road overheads
(08a), floor and metric functions (10) — so the only difference from a
scenario's own accessibility is the tiering. Each comparison scenario's
`cells_access_*` values on the sampled cells are then compared against
this exact baseline (R²_1to1, R²_fit, offset, slope, MAE%).

Routing uses a prototype Dijkstra (numba, indexed binary heap) that stops
at the cutoff or once all of an origin's targets are settled, with origins
routed in parallel across threads. A subset of origins is cross-checked
against scipy's Dijkstra on the same CSR graph — two independent
shortest-path implementations — and the per-origin timings are logged
(scipy, prototype on one thread, prototype in parallel).

**Prototype for aperta.** The functions in the "MIGRATION CANDIDATES"
section below are intended to move into aperta (planned for 0.5.0a) as
the "flat" OD alternative to the tiered pairs, once this prototype has
proven itself. Everything after that section is atlas-specific.

Scope: road modes, `time_gross` and `util` grids. Utility grids apply 09b's
disutility (time, log time, origin / destination endpoint features — no
route features) to the same gross times. Transit (NPVM z2z lookup, no
routing) and `dist_line` grids are skipped with a log line.

Destination cells are restricted to those with a positive value in at
least one of the grids' destination columns — exact for all three metric
families, since zero-weight destinations contribute nothing.

Inputs (the plan's `exact_baseline` scenario, e.g. res-h11, its own storage):
    nw/<mode>.graphml + properties/edges_<mode>_calibrated.csv
    properties/cells_{population,snap,zone_ids,pois,employment}.csv
    properties/nodes_<mode>_extended.csv (+ the util endpoint-feature node files)
    properties/zones_snap.csv
    coefs/<kind>/overheads_road.csv, utility_<variant>.csv, utility_<variant>_stats.csv
    <compare scenario>/properties/cells_access_<grid>_<metric>_<profile>.csv

Outputs (RESULTS, under the headline scenario):
    technical_validation/access_exact_<n>.csv           # medians over destinations per
                                                        # scenario × grid × metric × profile × parameter
    technical_validation/access_exact_<n>_columns.csv   # one row per compared column

The baseline and the compared scenarios come from the headline scenario's
plan (`scenarios_technical_validation.TECHNICAL_VALIDATION`); for
`switzerland-h10`, res-h11 cells as the baseline, compared with res-h11
(default radii) and the radii-* scenarios.

Run (headline scenario `switzerland-h10` by default):
    python -m technical_validation.access_exact_sample \\
        [--n-origins 1000] [--seed 0] [--batch-size 100] [--check-scipy 20] \\
        [--compare res-h11 radii-750-3000 ...]
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from numba import get_num_threads, get_thread_id, njit, prange

from aperta import accessibility, overhead, routing
from aperta.accessibility import exp_decay
from aperta.data_processing import weighted_group_mean
from aperta.network_processing import attach_edge_properties
from aperta.od_pairs import TieredODGeoPairs, lookup_dest_column_geo
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

import scenarios  # noqa: F401 — load first: it imports scenarios_technical_validation
from main.common import (
    ROAD_OVERHEAD_DENSITY_COL,
    bins_from_edges_min,
    build_disutility_odm,
    build_disutility_spec,
    extract_sd_averages_from_coefs,
    extract_t_cut_min_from_coefs,
    flatten_bin_dest_columns,
    flatten_decay_dest_columns,
    flatten_k_dest_columns,
    gravity_decays_from_half_decay_min,
    join_util_node_features,
    per_cell_road_overheads,
    profile_to_util_mode,
    resolve_road_overhead_column,
    route_time_alpha,
    scaled_weights_for_nearest_k,
)
from scenarios import get_scenario
from scenarios_technical_validation import technical_validation_plan
from technical_validation.access_vs_reference import by_param, column_stats


# ===========================================================================
# MIGRATION CANDIDATES (→ aperta 0.5.0a, "flat" OD routing)
#
# Self-contained, atlas-agnostic: CSR arrays + node indices in, per-origin
# costs / geo-keyed pairs out. If the prototype holds up, these move to
# `aperta.routing` (kernel + driver) and `aperta.od_pairs` (flat pairs),
# with unit tests against scipy's Dijkstra.
# ===========================================================================

@njit(nogil=True, cache=True)
def _sift_up(heap, pos, dist, i):
    """Move heap[i] up until its parent's key is no larger."""
    node = heap[i]
    key = dist[node]
    while i > 0:
        parent = (i - 1) >> 1
        p_node = heap[parent]
        if dist[p_node] <= key:
            break
        heap[i] = p_node
        pos[p_node] = i
        i = parent
    heap[i] = node
    pos[node] = i


@njit(nogil=True, cache=True)
def _sift_down(heap, pos, dist, i, size):
    """Move heap[i] down until both children's keys are no smaller."""
    node = heap[i]
    key = dist[node]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        right = child + 1
        if right < size and dist[heap[right]] < dist[heap[child]]:
            child = right
        c_node = heap[child]
        if dist[c_node] >= key:
            break
        heap[i] = c_node
        pos[c_node] = i
        i = child
    heap[i] = node
    pos[node] = i


@njit(nogil=True, cache=True)
def dijkstra_to_targets(indptr, indices, weights, source, cutoff, targets, out,
                        dist, pos, heap, touched, slot):
    """Single-source Dijkstra on a CSR graph, writing the cost of each node
    in `targets` (unique node indices) into `out` (`inf` if not reached).
    Stops at `cutoff` or as soon as every target is settled.

    Indexed binary heap with decrease-key (keys = `dist`). Work arrays,
    each of length n_nodes and caller-owned so they're allocated once:
    `dist` (inf), `pos` (-1 = never queued, -2 = settled), `heap`,
    `touched`, `slot` (-1). Only entries this search touched are reset on
    exit, so a call costs O(search size), not O(n_nodes).
    """
    n_t = targets.shape[0]
    for j in range(n_t):
        slot[targets[j]] = j
        out[j] = np.inf
    remaining = n_t

    dist[source] = 0.0
    touched[0] = source
    n_touched = 1
    heap[0] = source
    pos[source] = 0
    size = 1

    while size > 0 and remaining > 0:
        u = heap[0]
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            _sift_down(heap, pos, dist, 0, size)
        pos[u] = -2
        du = dist[u]
        j = slot[u]
        if j >= 0:
            out[j] = du
            remaining -= 1
        for e in range(indptr[u], indptr[u + 1]):
            v = indices[e]
            if pos[v] == -2:
                continue
            nd = du + weights[e]
            if nd <= cutoff and nd < dist[v]:
                if dist[v] == np.inf:
                    touched[n_touched] = v
                    n_touched += 1
                dist[v] = nd
                if pos[v] == -1:
                    heap[size] = v
                    pos[v] = size
                    size += 1
                _sift_up(heap, pos, dist, pos[v])

    for i in range(n_touched):
        dist[touched[i]] = np.inf
        pos[touched[i]] = -1
    for j in range(n_t):
        slot[targets[j]] = -1


@njit(parallel=True, cache=True)
def _route_batch(indptr, indices, weights, sources, cutoff, t_ptr, t_nodes, out,
                 dist, pos, heap, touched, slot):
    """`dijkstra_to_targets` for every source, in parallel. Origin `i`'s
    targets are `t_nodes[t_ptr[i]:t_ptr[i+1]]`, results go to the same
    slice of `out`. Work arrays are per thread (first axis)."""
    for i in prange(sources.shape[0]):
        k = get_thread_id()
        a, b = t_ptr[i], t_ptr[i + 1]
        dijkstra_to_targets(indptr, indices, weights, sources[i], cutoff,
                            t_nodes[a:b], out[a:b],
                            dist[k], pos[k], heap[k], touched[k], slot[k])


class CsrRouter:
    """Routes many origins to per-origin target sets on one CSR graph,
    in parallel. Holds the int64 / float64 CSR arrays and per-thread
    work arrays, so repeated calls (batches, profiles of the same graph
    with `set_weights`) don't reallocate."""

    def __init__(self, csr):
        self.indptr = csr.indptr.astype(np.int64)
        self.indices = csr.indices.astype(np.int64)
        self.weights = csr.data.astype(np.float64)
        n = csr.shape[0]
        n_threads = get_num_threads()
        self.dist = np.full((n_threads, n), np.inf)
        self.pos = np.full((n_threads, n), -1, dtype=np.int64)
        self.heap = np.empty((n_threads, n), dtype=np.int64)
        self.touched = np.empty((n_threads, n), dtype=np.int64)
        self.slot = np.full((n_threads, n), -1, dtype=np.int64)

    def route(self, sources: np.ndarray, t_ptr: np.ndarray, t_nodes: np.ndarray,
              cutoff: float) -> np.ndarray:
        """Costs aligned with `t_nodes` (`inf` = not reached within `cutoff`).
        Each origin's target list must not contain duplicates."""
        out = np.empty(t_nodes.shape[0], dtype=np.float64)
        _route_batch(self.indptr, self.indices, self.weights,
                     np.asarray(sources, dtype=np.int64), float(cutoff),
                     np.asarray(t_ptr, dtype=np.int64), np.asarray(t_nodes, dtype=np.int64),
                     out, self.dist, self.pos, self.heap, self.touched, self.slot)
        return out

    def route_shared(self, sources: np.ndarray, targets: np.ndarray,
                     cutoff: float) -> np.ndarray:
        """Flat case: every origin routes to the same `targets`. Returns a
        dense `(len(sources), len(targets))` cost matrix."""
        n_s, n_t = len(sources), len(targets)
        t_ptr = np.arange(n_s + 1, dtype=np.int64) * n_t
        return self.route(sources, t_ptr, np.tile(targets, n_s), cutoff).reshape(n_s, n_t)


class NodeToUnits:
    """Expand node-level results to geo units: maps each node (CSR index)
    to the destination units (e.g. cells) snapped to it."""

    def __init__(self, unit_ids: np.ndarray, unit_nodes: np.ndarray, n_nodes: int):
        order = np.argsort(unit_nodes, kind='stable')
        self.units = np.asarray(unit_ids, dtype=object)[order]
        sorted_nodes = unit_nodes[order]
        nodes, starts, counts = np.unique(sorted_nodes, return_index=True, return_counts=True)
        self.start = np.zeros(n_nodes, dtype=np.int64)
        self.count = np.zeros(n_nodes, dtype=np.int64)
        self.start[nodes] = starts
        self.count[nodes] = counts
        self.nodes = nodes

    def expand(self, nodes: np.ndarray, costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(unit_ids, unit_costs)` for every unit snapped to `nodes`, each
        unit inheriting its node's cost."""
        reps = self.count[nodes]
        if reps.sum() == 0:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float64)
        first = self.start[nodes]
        offsets = np.repeat(first - np.cumsum(np.r_[0, reps[:-1]]), reps) + np.arange(reps.sum())
        return self.units[offsets], np.repeat(costs, reps)


def flat_geo_pairs(
    cost_matrix: np.ndarray,
    origin_units: pd.Index,
    target_nodes: np.ndarray,
    dests: NodeToUnits,
    dtype=np.float32,
) -> tuple[TieredODGeoPairs, TieredODGeoPairs]:
    """Build a flat (cells_to_cells-only) geo-keyed pair index + cost ODM
    from a dense origin × target-node cost matrix (`CsrRouter.route_shared`).
    Row `i` belongs to `origin_units[i]`; unreached targets are dropped.
    Same structures as the tiered pipeline, so overhead baking and the
    accessibility metrics apply unchanged."""
    pairs: dict = {}
    costs: dict = {}
    for unit, row in zip(origin_units, cost_matrix):
        reached = np.isfinite(row)
        ids, vals = dests.expand(target_nodes[reached], row[reached])
        pairs[unit] = ids
        costs[unit] = vals.astype(dtype)
    return TieredODGeoPairs(cells_to_cells=pairs), TieredODGeoPairs(cells_to_cells=costs)


# ===========================================================================
# Atlas-specific: inputs, pipeline-consistent post-processing, comparison
# ===========================================================================

def _supported_grids(scenario) -> dict:
    grids = {k: g for k, g in scenario.accessibility_grids.items()
             if g.travel_cost in ('time_gross', 'util')}
    skipped = sorted(set(scenario.accessibility_grids) - set(grids))
    if skipped:
        logging.info(f"  → skipping grids not covered here (dist_line): {skipped}")
    return grids


class UtilInputs:
    """09b's inputs for the utility grids: cells / zones with endpoint
    features joined, and the utility coefs + stats per variant."""

    def __init__(self, context, cells: pd.DataFrame, variants: set[str]):
        zones = context.get_properties('zones', ['snap'])
        self.cells = cells.copy()
        self.zones = zones.copy()
        join_util_node_features(self.cells, self.zones, context)
        self.cells['unit_id'] = self.cells.index
        self.zones['unit_id'] = self.zones.index
        self.coefs = {v: context.get_coefs(f'utility_{v}') for v in variants}
        stats = {v: context.get_coefs(f'utility_{v}_stats') for v in variants}
        self.sd_avg = {v: extract_sd_averages_from_coefs(stats[v]) for v in variants}
        self.t_cut = {v: extract_t_cut_min_from_coefs(stats[v]) or None for v in variants}

    def spec(self, variant: str, profile_label: str):
        """09b's disutility spec for one profile, or None if not modelled."""
        if profile_to_util_mode(profile_label) is None:
            return None
        return build_disutility_spec(self.coefs[variant], profile_label,
                                     sd_averages=self.sd_avg[variant],
                                     t_cut_min_per_mode=self.t_cut[variant])


def _scipy_check(csr, router: CsrRouter, origin_nodes, targets, cutoff) -> dict:
    """Route `origin_nodes` with the prototype on ONE thread and with
    scipy's Dijkstra (same CSR, same cutoff), compare costs at `targets`.
    Returns max |Δ|, reachability mismatches and serial ms per origin."""
    from numba import set_num_threads
    from scipy.sparse.csgraph import dijkstra
    n_threads = get_num_threads()
    set_num_threads(1)
    t0 = time.perf_counter()
    mine = router.route_shared(origin_nodes, targets, cutoff)
    t_proto = time.perf_counter() - t0
    set_num_threads(n_threads)
    max_abs, n_mismatch, t_scipy = 0.0, 0, 0.0
    for o, row in zip(origin_nodes, mine):
        t0 = time.perf_counter()
        ref = dijkstra(csr, directed=True, indices=int(o), limit=cutoff)[targets]
        t_scipy += time.perf_counter() - t0
        n_mismatch += int((np.isfinite(ref) != np.isfinite(row)).sum())
        both = np.isfinite(ref) & np.isfinite(row)
        if both.any():
            max_abs = max(max_abs, float(np.abs(ref[both] - row[both]).max()))
    n = max(len(origin_nodes), 1)
    return {'max_abs_diff_s': max_abs, 'reach_mismatches': n_mismatch,
            'proto_serial_ms': 1000 * t_proto / n, 'scipy_ms': 1000 * t_scipy / n}


def _metrics_for_batch(costs_final, pairs_geo, cells, cell_to_zone, grid,
                       scaled_dests, is_util: bool) -> dict[str, pd.DataFrame]:
    """Run 10's three metric families for one grid on a batch of origins."""
    bins = bins_from_edges_min(grid.bin_edges_min)
    if is_util:
        decays = [exp_decay(f'exp{b}', beta=b) for b in grid.gravity_util_betas]
    else:
        decays = gravity_decays_from_half_decay_min(grid.gravity_half_decay_min)
    ks = list(grid.nearest_k)
    weights = {d: lookup_dest_column_geo(d, pairs_geo, cells) for d in grid.dest_cols}
    out = {}
    if bins:
        out['counts'] = flatten_bin_dest_columns(
            accessibility.cumulative_opportunities(costs_final, weights, cell_to_zone, bins))
    if ks:
        out['nearest_k'] = flatten_k_dest_columns(accessibility.nearest_k(
            costs_final, scaled_weights_for_nearest_k(weights, scaled_dests),
            cell_to_zone, ks=ks))
    if decays:
        raw = accessibility.gravity(costs_final, weights, cell_to_zone, decays)
        # As in 10: utility gravity is the logsum; 0 → NaN (unreachable).
        out['gravity'] = flatten_decay_dest_columns(np.log(raw.where(raw > 0)) if is_util else raw)
    return out


def _pair_check(pipe_pairs, pipe_costs, pairs_geo, gross) -> dict:
    """Compare the exact gross times with the pipeline's stored cells_to_cells
    gross times (08a) on the pairs both contain. Within `r_cells` the two must
    agree up to float32 rounding — any larger difference means the exact
    baseline doesn't reproduce the pipeline."""
    diffs = []
    for origin, ex_ids in pairs_geo.cells_to_cells.items():
        p_ids = pipe_pairs.cells_to_cells.get(origin)
        if p_ids is None or len(ex_ids) == 0:
            continue
        p_vals = pd.Series(pipe_costs.cells_to_cells[origin], index=p_ids)
        ex = pd.Series(gross.cells_to_cells[origin], index=ex_ids)
        common = ex.index.intersection(p_vals.index)
        d = (ex[common].to_numpy(float) - p_vals[common].to_numpy(float))
        diffs.append(d[np.isfinite(d)])
    d = np.concatenate(diffs) if diffs else np.empty(0)
    return {'n_pairs': len(d), 'max_abs_s': float(np.abs(d).max()) if len(d) else np.nan,
            'share_over_0.5s': float((np.abs(d) > 0.5).mean()) if len(d) else np.nan,
            'mean_s': float(d.mean()) if len(d) else np.nan}


def _exact_for_profile(context, scenario, case, profile, cells, coefs, sample,
                       grids, batch_size, n_check, util: UtilInputs | None,
                       check_pairs: bool = False) -> tuple[dict, dict]:
    """Exact accessibility for one road profile on the sampled origins.
    Returns `({(grid, metric): DataFrame}, routing_info)`."""
    mode = case.mode
    src = case.source_for(profile)
    label = src.name
    node_col = f'node_id_{mode}'
    dest_cols = sorted({c for g in grids.values() for c in g.dest_cols})
    scaled_dests = set(scenario.population_cols) | set(scenario.employment_cols)
    specs = {}
    for grid_key, grid in grids.items():
        if grid.travel_cost == 'util':
            assert util is not None
            specs[grid_key] = util.spec(grid.utility, label)
            if specs[grid_key] is None:
                logging.info(f"  → {grid_key}: no utility spec for {label!r}; skipped")

    with step(f'profile={label}: graph + CSR'):
        graph = context.get_nw(data_name=mode, allow_cache=False)
        calibrated = context.get_properties('edges', src.edges_data_name)
        attach_edge_properties(graph, calibrated[[src.edge_column]])
        csr, nx_to_seq, _ = routing._graph_to_csr(graph, src.edge_column)
        del graph
        router = CsrRouter(csr)

    with step(f'profile={label}: origins, destinations, overheads'):
        # Destination eligibility as in 05 / 08a, plus "positive weight in
        # some grid dest column" (exact: zero weights contribute nothing).
        cells_m = cells[cells[node_col].notna() & cells['zone_id'].notna()].copy()
        has_weight = (cells_m[dest_cols].fillna(0) > 0).any(axis=1)
        dest_cells = cells_m[has_weight]
        dest_nodes = dest_cells[node_col].astype(np.int64).map(nx_to_seq).to_numpy()
        dests = NodeToUnits(dest_cells.index.to_numpy(), dest_nodes, csr.shape[0])
        origin_nodes = sample.to_frame().join(cells[[node_col]])[node_col]
        origin_seq = origin_nodes.astype(np.int64).map(nx_to_seq)
        logging.info(f"  → {len(origin_seq):,} origins, {len(dest_cells):,} weighted "
                     f"destination cells on {len(dests.nodes):,} nodes")

        node_props = context.get_properties('nodes', f'{mode}_extended')
        dens = f'_density_{mode}'
        cells_m = cells_m.join(node_props[[ROAD_OVERHEAD_DENSITY_COL]].rename(columns={ROAD_OVERHEAD_DENSITY_COL: dens}),
                               on=node_col)
        cells_m[dens] = cells_m[dens].fillna(0.0)
        same_mode = [case.source_for(p).name for p in case.profiles]
        ov_label = resolve_road_overhead_column(label, coefs.columns,
                                                same_mode_profile_labels=same_mode)
        alpha = route_time_alpha(coefs[ov_label])
        orig_ov, dest_ov = per_cell_road_overheads(
            cells_m, coefs[ov_label], density_col=dens, snap_dist_col=f'distance_{mode}')
        cell_to_zone = cells['zone_id'].to_dict()
        zone_ov_o = weighted_group_mean(orig_ov, cells_m['combined_total'], cells_m['zone_id'])
        zone_ov_d = weighted_group_mean(dest_ov, cells_m['combined_total'], cells_m['zone_id'])

    parts: dict[tuple[str, str], list[pd.DataFrame]] = {}
    t_route, n_routed = 0.0, 0
    if check_pairs:
        pipe_pairs = context.get_tiered_odm(network_name=mode, data_name='geo_pairs')
        pipe_costs = context.get_tiered_odm(network_name=mode, data_name=f'time_gross_{label}')
        checks = []
    for b0 in range(0, len(origin_seq), batch_size):
        batch = origin_seq.iloc[b0:b0 + batch_size]
        with step(f'profile={label}: origins {b0 + 1}-{b0 + len(batch)} of {len(origin_seq)}'):
            # Cells sharing a snap node share one search.
            u_nodes, inverse = np.unique(batch.to_numpy(), return_inverse=True)
            t0 = time.perf_counter()
            matrix = router.route_shared(u_nodes, dests.nodes, case.time_cutoff_s)
            t_route += time.perf_counter() - t0
            n_routed += len(u_nodes)

            pairs_geo, costs = flat_geo_pairs(matrix[inverse], batch.index, dests.nodes, dests)
            # Same post-processing as 08a (net) and 10 (gross).
            costs = routing.floor_intrazonal_costs(costs, min_cost=case.min_route_time_s)
            if alpha != 1.0:
                costs = TieredODGeoPairs(cells_to_cells={
                    k: v * alpha for k, v in costs.cells_to_cells.items()})
            gross = overhead.add_geo_overheads(
                costs, pairs_geo, origin_cell=orig_ov, dest_cell=dest_ov,
                origin_zone=zone_ov_o, dest_zone=zone_ov_d, cell_to_zone=cell_to_zone)
            if check_pairs:
                checks.append(_pair_check(pipe_pairs, pipe_costs, pairs_geo, gross))

            for grid_key, grid in grids.items():
                is_util = grid.travel_cost == 'util'
                if is_util:
                    if specs[grid_key] is None:
                        continue
                    # 09b: disutility from 08a's gross times; then 10's floor.
                    assert util is not None
                    final = build_disutility_odm(
                        cost_odm_sec=gross, pairs_geo=pairs_geo, spec=specs[grid_key],
                        cells_m=util.cells, zones_m=util.zones)
                    final = routing.floor_intrazonal_costs(
                        final, min_cost=case.min_route_disutility)
                else:
                    final = routing.floor_intrazonal_costs(gross, min_cost=case.min_route_time_s)
                res = _metrics_for_batch(final, pairs_geo, cells, cell_to_zone, grid,
                                         scaled_dests, is_util)
                for metric, df in res.items():
                    parts.setdefault((grid_key, metric), []).append(df)

    info = {'profile': label, 'threads': get_num_threads(),
            'proto_parallel_ms': 1000 * t_route / max(n_routed, 1)}
    if check_pairs:
        c = pd.DataFrame(checks)
        n = c['n_pairs'].sum()
        info.update({'pairs_checked': int(n), 'pair_max_abs_s': c['max_abs_s'].max(),
                     'pair_share_over_0.5s': float((c['share_over_0.5s'] * c['n_pairs']).sum() / n),
                     'pair_mean_s': float((c['mean_s'] * c['n_pairs']).sum() / n)})
        logging.info(f"  → pair check vs pipeline c2c gross times: {n:,} pairs, max |Δ| "
                     f"{info['pair_max_abs_s']:.3f} s, share |Δ| > 0.5 s "
                     f"{info['pair_share_over_0.5s']:.2%}, mean Δ {info['pair_mean_s']:+.3f} s")
    check_nodes = np.unique(origin_seq.to_numpy())[:n_check]
    with step(f'profile={label}: cross-check {len(check_nodes)} origins against scipy'):
        info.update(_scipy_check(csr, router, check_nodes, dests.nodes, case.time_cutoff_s))
        logging.info(
            f"  → max |Δ| = {info['max_abs_diff_s']:.2e} s, reachability mismatches = "
            f"{info['reach_mismatches']}; ms/origin: scipy {info['scipy_ms']:.2f}, "
            f"prototype serial {info['proto_serial_ms']:.2f}, "
            f"parallel ({info['threads']} threads) {info['proto_parallel_ms']:.2f}")
    return {k: pd.concat(v) for k, v in parts.items()}, info


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--n-origins', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--check-scipy', type=int, default=20,
                        help='origins per profile cross-checked against scipy')
    parser.add_argument('--check-pairs', action='store_true',
                        help="compare exact gross times with the pipeline's stored "
                             "cells_to_cells gross times on shared pairs")
    parser.add_argument('--compare', nargs='*', default=None,
                        help="scenarios to compare (default: the plan's exact_compare)")
    args, _ = parser.parse_known_args()

    context = init_context()
    plan = technical_validation_plan(context.scenario)
    # Inputs come from the baseline scenario; results go to the headline scenario.
    inputs = context.source(f'{context.project}/{plan.exact_baseline}')
    scenario = get_scenario(plan.exact_baseline)
    compare = args.compare or list(plan.exact_compare)
    grids = _supported_grids(scenario)

    with step(f'load cells + overhead coefs ({plan.exact_baseline}), sample origins'):
        cells = inputs.get_properties(
            'cells', ['population', 'snap', 'zone_ids', 'pois', 'employment'])
        coefs = inputs.get_coefs('overheads_road')
        pool = cells.index[(cells['is_active'] == 1) & (cells['is_aoi'] == 1)]
        rng = np.random.default_rng(args.seed)
        sample_ids = rng.choice(pool.to_numpy(), size=min(args.n_origins, len(pool)),
                                replace=False)
        sample = pd.Series(sample_ids, index=sample_ids, name='cell_id')
        logging.info(f"  → {len(sample):,} origins sampled from {len(pool):,} active AOI cells")
        util_variants = {g.utility for g in grids.values() if g.travel_cost == 'util'}
        util = UtilInputs(inputs, cells, util_variants) if util_variants else None

    exact: dict[tuple[str, str, str], pd.DataFrame] = {}
    routing_info = []
    for case in scenario.mode_configs.values():
        for profile in case.profiles:
            res, info = _exact_for_profile(inputs, scenario, case, profile, cells, coefs,
                                           sample, grids, args.batch_size, args.check_scipy,
                                           util, args.check_pairs)
            routing_info.append(info)
            for (grid_key, metric), df in res.items():
                exact[(grid_key, metric, info['profile'])] = df
    logging.info("  → transit skipped (NPVM lookup, no routing)")

    with step(f'compare {compare} against the exact baseline'):
        rows = []
        for name in compare:
            ctx = context.source(f'{context.project}/{name}')
            for (grid_key, metric, label), ref in exact.items():
                try:
                    tgt = ctx.get_properties('cells', f'access_{grid_key}_{metric}_{label}',
                                             allow_cache=False)
                except FileNotFoundError:
                    logging.warning(f"  ⚠ {name}: no access_{grid_key}_{metric}_{label}")
                    continue
                cols = [c for c in ref.columns if c in tgt.columns]
                idx = ref.index.intersection(tgt.index)
                base = {'scenario': name, 'grid': grid_key, 'metric': metric,
                        'profile': label, 'n_origins': len(idx)}
                rows.extend({**base, **r} for r in
                            column_stats(tgt, ref, idx, cols).to_dict('records'))

    with step('summary'):
        columns = pd.DataFrame(rows)
        df = by_param(columns, ['scenario', 'grid', 'metric', 'profile'])
        context.create_results(columns,
                               f'technical_validation/access_exact_{len(sample)}_columns.csv',
                               kws={'index': False, 'float_format': '%.4f'})
        context.create_results(df, f'technical_validation/access_exact_{len(sample)}.csv',
                               kws={'index': False, 'float_format': '%.4f'})
        for value, fmt in (('R²_1to1', '{:.4f}'), ('R²_fit', '{:.4f}'), ('MAE%', '{:.1f}%')):
            pv = df.pivot_table(
                index=['grid', 'metric', 'profile', 'reach', 'param'], columns='scenario',
                values=value, aggfunc='first', sort=False)
            pv = pv[[c for c in compare if c in pv.columns]]
            logging.info(f"{value} vs exact (untiered) baseline, {len(sample):,} origins:\n"
                         + pv.map(lambda v: fmt.format(v) if pd.notna(v) else '—').to_string())
        logging.info("Routing cross-check (prototype vs scipy):\n"
                     + pd.DataFrame(routing_info).to_string(index=False))
    context.close()


if __name__ == '__main__':
    main()
