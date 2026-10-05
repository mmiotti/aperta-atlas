"""
Fork-based multiprocessing wrapper around `aperta.routing` primitives,
plus a `lean_graph` helper that strips an OSMnx-loaded networkx graph to
just topology + the routing weight(s).

The wrapper relies on the `fork` start method (Mac/Linux only) so that
the graph the parent process holds is inherited by worker processes via
copy-on-write — no pickling. For best results pass a graph that's been
through `lean_graph` first: less per-worker memory and fewer COW pages
dirtied during routing.

Single-threaded callers should use `aperta.routing.shortest_path_costs_one_to_one`
directly. This module exists to amortize that function over many trips
across cores on a shared graph.

Windows is NOT supported (no `fork`). On Windows, fall back to the
single-threaded `aperta.routing` variant.
"""

import multiprocessing as mp
import os

import networkx as nx
import numpy as np
import pandas as pd

from aperta import routing as _routing


# Module-level globals populated by the mp wrappers before spawning
# workers. After fork, each child sees these via memory inheritance —
# no pickling, no per-call IPC for the graph. Costs-only wrappers use
# _GRAPH + _WEIGHT; the metrics wrapper additionally uses
# _EDGE_FEATURES + _LENGTH_ATTR.
_GRAPH: nx.Graph | None = None
_WEIGHT: str | None = None
_EDGE_FEATURES: dict[str, str] | None = None
_LENGTH_ATTR: str | None = None


def lean_graph(graph: nx.Graph, weight_attrs: str | list[str]) -> nx.Graph:
    """Return a copy of `graph` with only `weight_attrs` on edges.

    Strips every other per-edge attribute (geometry, length, bridge,
    tunnel, lanes, ...). Nodes keep their attributes (commonly `x`/`y`).

    Use this BEFORE passing a graph to a multiprocessing routing wrapper.
    Two wins:
      1. Cuts per-worker memory (fork-COW pages stay shared longer when
         the graph's reachable Python objects are fewer).
      2. Reduces the in-memory graph size in the parent too — useful even
         single-threaded if you ran into memory pressure.

    Args:
        graph: networkx graph (any variant).
        weight_attrs: edge attribute name (or list of names) to KEEP.
            Edges missing every named attr are dropped.

    Returns:
        New graph of the same `__class__` as input.
    """
    if isinstance(weight_attrs, str):
        weight_attrs = [weight_attrs]
    keep = set(weight_attrs)

    lean = graph.__class__()
    lean.add_nodes_from(graph.nodes(data=True))
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, k, d in graph.edges(keys=True, data=True):
            attrs = {a: d[a] for a in keep if a in d}
            if not attrs:
                continue
            lean.add_edge(u, v, key=k, **attrs)
    else:
        for u, v, d in graph.edges(data=True):
            attrs = {a: d[a] for a in keep if a in d}
            if not attrs:
                continue
            lean.add_edge(u, v, **attrs)
    return lean


def _route_chunk(chunk):
    """Worker entry point. `_GRAPH` and `_WEIGHT` are inherited from the
    parent via fork."""
    trip_ids, origins, dests = chunk
    return _routing.shortest_path_costs_one_to_one(
        _GRAPH, trip_ids, origins, dests, _WEIGHT,
    )


def shortest_path_costs_one_to_one_mp(
    graph: nx.Graph,
    trip_ids,
    origins,
    destinations,
    weight: str,
    *,
    n_workers: int | None = None,
) -> pd.Series:
    """Fork-multiprocessing wrapper around `routing.shortest_path_costs_one_to_one`.

    Splits the trip list into `n_workers` chunks and routes each chunk in
    a forked child process. The graph is shared via memory inheritance
    (no per-worker pickling), so passing a `lean_graph(graph, weight)`
    keeps per-worker memory low.

    Args:
        graph: networkx graph; edges carry `weight`. Will be set as a
            module global before fork so workers can access it via
            inheritance — DO NOT mutate `graph` while this function runs.
        trip_ids, origins, destinations: same-length sequences.
        weight: edge attribute name (single column).
        n_workers: number of worker processes. Default = `max(1, cpu_count - 1)`,
            which is also the ceiling — larger values are clamped down.
            Pass `1` for a single-threaded fallback (no fork overhead).

    Returns:
        `pd.Series[float]` indexed by trip_id, named `'cost'`. Drop
        semantics match the underlying nx variant.

    Raises:
        RuntimeError on platforms without `fork` support (Windows).
    """
    global _GRAPH, _WEIGHT
    _cap = max((os.cpu_count() or 2) - 1, 1)
    n_workers = min(n_workers or _cap, _cap)
    if n_workers <= 1 or len(trip_ids) == 0:
        return _routing.shortest_path_costs_one_to_one(
            graph, trip_ids, origins, destinations, weight,
        )

    try:
        ctx = mp.get_context('fork')
    except ValueError as e:
        raise RuntimeError(
            "shortest_path_costs_one_to_one_mp requires the 'fork' start "
            "method, which isn't available on this platform (typically Windows). "
            "Use routing.shortest_path_costs_one_to_one single-threaded instead."
        ) from e

    trip_ids_arr = np.asarray(trip_ids)
    origins_arr = np.asarray(origins)
    dests_arr = np.asarray(destinations)
    chunk_idxs = np.array_split(np.arange(len(trip_ids_arr)), n_workers)
    chunks = [
        (trip_ids_arr[c], origins_arr[c], dests_arr[c])
        for c in chunk_idxs if len(c) > 0
    ]

    _GRAPH = graph
    _WEIGHT = weight
    try:
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_route_chunk, chunks)
    finally:
        # Don't keep a multi-GB graph reference alive in this module's
        # globals after the call returns.
        _GRAPH = None
        _WEIGHT = None

    if not results:
        return pd.Series(dtype=float, name='cost')
    return pd.concat(results)


def _metrics_chunk(chunk):
    """Worker entry for `shortest_path_metrics_one_to_one_mp`. Reads
    `_GRAPH`, `_WEIGHT`, `_LENGTH_ATTR`, `_EDGE_FEATURES` from module
    globals inherited via fork."""
    trip_ids, origins, dests = chunk
    return _routing.shortest_path_metrics_one_to_one(
        _GRAPH, trip_ids, origins, dests, _WEIGHT,
        length_attr=_LENGTH_ATTR or 'length',
        edge_features=_EDGE_FEATURES,
    )


def shortest_path_metrics_one_to_one_mp(
    graph: nx.Graph,
    trip_ids,
    origins,
    destinations,
    weight: str,
    *,
    length_attr: str = 'length',
    edge_features: dict[str, str] | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Fork-multiprocessing wrapper around
    `routing.shortest_path_metrics_one_to_one`.

    Same chunk-and-fork scaffolding as `shortest_path_costs_one_to_one_mp`
    but the worker calls the metrics variant, so each per-chunk result is
    a DataFrame indexed by trip_id with columns `distance`, `cost`, and
    one column per entry in `edge_features`. Per-chunk DataFrames are
    concatenated on return.

    Path extraction + per-edge aggregation is more expensive than a
    cost-only Dijkstra — expect this variant to run ~1.5–2× longer than
    the cost-only wrapper on the same trip set.

    Args:
        graph, trip_ids, origins, destinations, weight, n_workers: see
            `shortest_path_costs_one_to_one_mp`.
        length_attr: edge attribute used as path-length (default
            `'length'`); required only if any `edge_features` value is
            `'length_weighted'`, but always summed into a `distance`
            column of the result.
        edge_features: mapping `attr_name → aggregation`, where
            aggregation is one of `'sum'`, `'length_weighted'`, or
            `'duration_weighted'`. Adds one column per entry to the
            result. `None` (default) or `{}` = no extra features (still
            returns `distance` + `cost`).

    Returns:
        `pd.DataFrame` indexed by trip_id. Columns: `distance`, `cost`,
        plus one column per key in `edge_features`. Rows where routing
        found no path are dropped (drop semantics match the underlying
        nx variant).

    Raises:
        RuntimeError on platforms without `fork` support (Windows).
    """
    global _GRAPH, _WEIGHT, _EDGE_FEATURES, _LENGTH_ATTR
    _cap = max((os.cpu_count() or 2) - 1, 1)
    n_workers = min(n_workers or _cap, _cap)
    if n_workers <= 1 or len(trip_ids) == 0:
        return _routing.shortest_path_metrics_one_to_one(
            graph, trip_ids, origins, destinations, weight,
            length_attr=length_attr,
            edge_features=edge_features,
        )

    try:
        ctx = mp.get_context('fork')
    except ValueError as e:
        raise RuntimeError(
            "shortest_path_metrics_one_to_one_mp requires the 'fork' "
            "start method, which isn't available on this platform "
            "(typically Windows). Use "
            "routing.shortest_path_metrics_one_to_one single-threaded "
            "instead."
        ) from e

    trip_ids_arr = np.asarray(trip_ids)
    origins_arr = np.asarray(origins)
    dests_arr = np.asarray(destinations)
    chunk_idxs = np.array_split(np.arange(len(trip_ids_arr)), n_workers)
    chunks = [
        (trip_ids_arr[c], origins_arr[c], dests_arr[c])
        for c in chunk_idxs if len(c) > 0
    ]

    _GRAPH = graph
    _WEIGHT = weight
    _EDGE_FEATURES = edge_features or {}
    _LENGTH_ATTR = length_attr
    try:
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_metrics_chunk, chunks)
    finally:
        # Don't keep the graph reference alive after the call returns.
        _GRAPH = None
        _WEIGHT = None
        _EDGE_FEATURES = None
        _LENGTH_ATTR = None

    if not results:
        cols = ['distance', 'cost'] + list(edge_features or [])
        return pd.DataFrame(columns=cols)
    return pd.concat(results)
