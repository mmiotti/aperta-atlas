"""Project-scoped context: filesystem paths, typed data I/O.

A `Context` (obtained from `init_context()`) is the single object scripts
pass through to do filesystem work. It owns:

  - **Namespace + role**: `preparation/<...>` (scenario-free; produces
    reusable prepared assets, shared across projects) or `<project>/<sub>`
    (project-bound; consumes prepared data + applies project logic).
    Inferred from the caller's path under `src/`: a script in
    `src/preparation/<region>/<sub>/x.py` gets namespace
    `preparation/<region>/<sub>`; a script anywhere else under `src/`
    gets namespace `<PROJECT_NAME>/<first-path-segment>` where
    `PROJECT_NAME` is read from the top-level `scenarios` module.

  - **Resolved paths** via `Context.path_for(storage, relative_path)` for
    typed assets and `Context.raw_path(...)` for read-only external
    inputs. The four `Storage` classes map onto distinct on-disk roots:

        preparation/<remainder>/x.py:
          PUBLIC    -> <DATA_DIR_PUBLIC>/preparation/<remainder>/<rel>
          PRIVATE   -> <DATA_DIR_PRIVATE>/preparation/<remainder>/<rel>
          SCRATCH   -> <WORKING_DIR>/scratch/preparation/<remainder>/<rel>
          RESULTS   -> not allowed

        <project> scripts (any subfolder under src/ other than preparation/):
          PUBLIC    -> <DATA_DIR_PUBLIC>/<project>/<scenario>/<rel>
          PRIVATE   -> <DATA_DIR_PRIVATE>/<project>/<scenario>/<rel>
          SCRATCH   -> <WORKING_DIR>/scratch/<project>/<scenario>/<rel>
          RESULTS   -> <WORKING_DIR>/results/<project>/<scenario>/<rel>

        raw_path (callable from any namespace; SCRATCH and RESULTS rejected):
          PUBLIC    -> <DATA_DIR_PUBLIC>/raw/<rel>
          PRIVATE   -> <DATA_DIR_PRIVATE>/raw/<rel>

    `WORKING_DIR` is the repo root (source code, config, status.json
    live there). `SCRATCH` is the storage class for transient outputs
    inside that root — distinct from the env-var name.

  - **Scenario + storage defaults** are read from the top-level
    `scenarios` module (typed Python): `PROJECT_NAME` is the project
    identifier used in on-disk paths; `DEFAULT_SCENARIO` is the
    scenario used when `--scenario` isn't passed; each
    `SCENARIOS[<name>]` entry's `.storage` field provides the project
    context's `default_storage`. No YAML.

  - **Per-namespace preparation storage**: each preparation sub-package's
    `__init__.py` declares `STORAGE = Storage.PUBLIC` (or `PRIVATE`),
    used as that namespace's `default_storage`. So a script writing
    Swiss historic data writes to PRIVATE without needing
    `storage=Storage.PRIVATE` on every call; a script writing OSM
    derivatives writes to PUBLIC the same way.

  - **Typed I/O methods** — `Context.create_/get_shapes`, `_properties`,
    `_odm`, `_tiered_odm`, `_nw`, `_generic`, `_results`. Each enforces
    the `aperta_atlas` filesystem conventions (typed subfolders, index-name
    → geo_unit mapping, ODM CSR layout, network skeleton + companion
    properties). Geo-units are a fixed canonical set defined in this
    module (`CANONICAL_UNITS`: `cells`, `zones`, `nodes`, `edges`).

Cross-namespace reads use `context.source('<path>', storage=...)` where
`<path>` mirrors the on-disk layout:
    context.source('preparation/switzerland/historic')        # preparation
    context.source(f'{context.project}/switzerland-h10')      # same project,
                                                                # other scenario
    context.source('lumos/2020')                              # cross-project (rare)
The optional `storage=` kwarg pins the source ctx's default_storage
(useful when a namespace's STORAGE constant is wrong for this read, or
unreachable cross-repo).

Optional dependency tracking lives in `aperta_atlas.tracking`; `Context`
delegates to it from `register_used_data`, `close`, and the start-of-run
banner emitted by `init_context`. Activate via
`APERTA_TRACK_DEPENDENCIES=1` in `.env`.
"""

import inspect
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from aperta.errors import ContextError, DataError
from aperta_atlas import tracking
from aperta_atlas.utils import tracked_namedtuple


# Load .env once. The .env determines where config files live, so it can't itself
# be in the YAML config.
env_values = dotenv_values()


# Reserved sentinel for status.json's scenario position when a script is scenario-free
# (i.e. preparation). Real scenario names cannot collide with this.
_NO_SCENARIO = '_'


# Custom NOTE log level. Registered by the `aperta` library at import
# (level number + plain "NOTE" name + `logging.note()` helper). See
# `aperta/__init__.py`. Here we just re-export the level number for the
# summary handler below; the yellow "⚠ NOTE   " label is installed in
# `init_context` (which overrides `aperta`'s plain "NOTE" name).
from aperta import NOTE as NOTE_LEVEL   # noqa: E402


class _LevelCounterHandler(logging.Handler):
    """Root-logger handler that counts records emitted per level.
    `Context.close()` reads from `.counts` to print a per-script summary.
    """
    def __init__(self):
        super().__init__(level=logging.NOTSET)
        self.counts: dict[int, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[record.levelno] = self.counts.get(record.levelno, 0) + 1


_LEVEL_COUNTER: _LevelCounterHandler | None = None


def _install_level_counter() -> None:
    """Attach a single `_LevelCounterHandler` to the root logger. Idempotent
    (subsequent `init_context` calls in the same process won't stack handlers).
    """
    global _LEVEL_COUNTER
    if _LEVEL_COUNTER is not None:
        return
    _LEVEL_COUNTER = _LevelCounterHandler()
    logging.getLogger().addHandler(_LEVEL_COUNTER)


# =============================================================================
# Storage classes — where a typed asset lives on disk
# =============================================================================


class Storage(Enum):
    """Where a typed asset lives on disk.

    PUBLIC    — DATA_DIR_PUBLIC/{preparation/<remainder> | <project>/<scenario>}/...
    PRIVATE   — DATA_DIR_PRIVATE/...                                                  (same shape)
                (proprietary inputs / outputs derived from them; not for redistribution)
    SCRATCH   — WORKING_DIR/scratch/...                                               (same shape)
                (transient/intermediate; OK to delete)
    RESULTS   — WORKING_DIR/results/<project>/<scenario>/...
                (only callable from project namespaces; preparation produces
                prepared data, not results)
    """

    PUBLIC = "public"
    PRIVATE = "private"
    SCRATCH = "scratch"
    RESULTS = "results"


# Process-lifetime cache. Key is `(relative_path, storage)` so the same relative
# path under different storage classes doesn't alias.
cache: dict[tuple[str, Storage], any] = {}


# =============================================================================
# Canonical geo-units — registry consumed by the typed I/O methods
# =============================================================================
#
# Four canonical units in two conceptual groups, mapped to the on-disk index
# column they use:
#
#   Aggregation: `cells` (finest spatial unit; the only level routed
#                cell-to-cell) and `zones` (coarser, many-cells-per-zone).
#   Network:     `nodes` and `edges` of a routable mode-specific graph.
#
# Anything else (municipalities, cantons, buildings, locations, …) is
# *project-specific source data* that gets mapped into one of these four
# during preparation. The library knows about these four and only these four.

CANONICAL_UNITS: dict[str, str] = {
    "cells": "cell_id",
    "zones": "zone_id",
    "nodes": "node_id",
    "edges": "edge_id",
    # Scaffolding-only addition (2026-05-06). `buildings` is NOT a
    # routing primitive in the aperta algorithm library — that contract
    # remains tight at cells/zones/nodes/edges. This entry is here so
    # the typed-I/O methods (`create_shapes` / `create_properties` etc.)
    # recognise buildings, dedupe geometry across multiple per-building
    # property files, and follow the same `shapes/`+`properties/`
    # layout cells/zones use. Index values are OSM way IDs (globally
    # unique + stable across PBF refreshes); column name stays
    # consistent with the `<unit>_id` pattern.
    "buildings": "building_id",
}


def _unit_id_col(unit: str) -> str:
    """ID column / index name for `unit`. Raises `DataError` on unknown units."""
    if unit not in CANONICAL_UNITS:
        raise DataError(
            f"No registered id_col for geo_unit '{unit}'. "
            f"Known units: {sorted(CANONICAL_UNITS)}.",
        )
    return CANONICAL_UNITS[unit]


def _unit_for_id_col(col: str) -> str | None:
    """Reverse lookup: unit whose `id_col` is `col`, else `None`."""
    for name, id_col in CANONICAL_UNITS.items():
        if id_col == col:
            return name
    return None


def _warn_if_unknown_unit(unit: str, where: str = "") -> None:
    """Log a warning if `unit` is not canonical. Doesn't raise — preserves
    flexibility for ad-hoc geo_types — but flags typos at the call site.
    """
    if unit in CANONICAL_UNITS:
        return
    suffix = f" in {where}" if where else ""
    logging.warning(
        f"Unknown geo_unit '{unit}'{suffix}. "
        f"Known: {sorted(CANONICAL_UNITS)}. "
        f"Map your source data to one of these in preparation, or add a new "
        f"canonical unit to `aperta_atlas.context.CANONICAL_UNITS` if generally useful.",
    )


# =============================================================================
# Module-level helpers for the typed-I/O methods
# =============================================================================


def _require_osmnx():
    """Import osmnx lazily, with a clear install hint on failure.

    `osmnx` is an optional dependency — only needed for graph I/O (`.graphml`
    persistence and `graph_to_gdfs` for nodes/edges GeoDataFrame extraction).
    Core routing / accessibility / OD-pairs work without it.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise DataError(
            "osmnx is required for aperta's graph I/O (network .graphml + "
            "nodes/edges GeoDataFrame extraction) but is not installed. "
            "Install with `pip install aperta[osm]` or `pip install osmnx`.",
        ) from exc
    return ox


def _file_suffix(data_name: str | None) -> str:
    return f"_{data_name}" if data_name else ""


def _get_geo_type_str(
    geo_type: str | list[str] | None = None,
    df: pd.DataFrame | gpd.GeoDataFrame | None = None,
) -> str:
    """Resolve geo_type to its on-disk string form.

    If `df` is provided, look up the geo_type by reverse-mapping
    `df.index.name` through `CANONICAL_UNITS` (find the unit whose
    registered `id_col` is `df.index.name`).
    """
    if geo_type is not None and df is not None:
        raise ValueError("Specify either `df` or `geo_type`, not both.")
    if df is not None:
        if df.index.name is None:
            raise DataError("Index name must be set before creating file.")
        unit = _unit_for_id_col(df.index.name)
        if unit is None:
            raise DataError(
                f"DataFrame index name '{df.index.name}' is not a registered id_col "
                f"for any geo_unit. Known units: {sorted(CANONICAL_UNITS)}. "
                f"Set `df.index.name` to a registered id_col (e.g. 'cell_id').",
            )
        return unit
    if isinstance(geo_type, str):
        return geo_type
    if isinstance(geo_type, list):
        return '__'.join(set(geo_type))
    raise DataError(f"`geo_type` must be str or list[str] (got {type(geo_type)}).")


def _verify_index_matches_geo_type(df, geo_type: str, where: str) -> None:
    """If `geo_type` is registered, verify `df.index.name` matches its `id_col`.

    Catches files that were written under a different convention, or callers
    passing a mismatched geo_type. Skipped silently for unregistered geo_types
    — those are already flagged by `_warn_if_unknown_unit` at the call site.
    """
    if geo_type not in CANONICAL_UNITS:
        return
    expected = _unit_id_col(geo_type)
    if df.index.name != expected:
        raise DataError(
            f"{where}: loaded {geo_type!r} frame has index.name={df.index.name!r}, "
            f"expected {expected!r} (the registered id_col for '{geo_type}'). "
            f"File may have been written under a different convention.",
        )


def _verify_unique_index(df: pd.DataFrame) -> None:
    if len(df.index.unique()) != len(df.index):
        raise DataError(f"DataFrame index `{df.index.name}` is not unique.")


def _verify_index_type(df: pd.DataFrame | gpd.GeoDataFrame) -> None:
    # `pd.api.types.is_float_dtype` handles BOTH numpy dtypes and pandas
    # extension dtypes (e.g. `StringDtype` from `read_csv(low_memory=False)`
    # on a fully-string column). `np.issubdtype` raises TypeError on the
    # extension dtypes — don't use it here.
    if pd.api.types.is_float_dtype(df.index.dtype):
        raise DataError(
            "Index of most recent file was loaded as float; check integrity. "
            "It was likely saved with a float-type index after a merge-aggregate.",
        )


# =============================================================================
# Network-skeleton helpers (used by Context.create_nw / Context.get_nw)
# =============================================================================

# Skeleton attribute allowlists. Anything else on the graph belongs in companion
# property files — `.graphml`'s type system is weak, so types of computed metrics,
# weights, etc. don't round-trip reliably. Properties go through CSV (pandas),
# which preserves dtypes correctly.
NW_SKELETON_NODE_ATTRS = frozenset({'x', 'y'})
NW_SKELETON_EDGE_ATTRS = frozenset({'geometry', 'length'})


def _edge_id(u, v, k=None) -> str:
    """Synthesize a stable string edge_id from a (u, v[, key]) tuple.

    Used by `Context.create_nw` / `Context.get_nw` to fit edge tables into the
    typed-properties single-`id_col` convention. Limitation: assumes node IDs
    don't contain ':'.
    """
    if k is None:
        return f'{u}:{v}'
    return f'{u}:{v}:{k}'


def _to_skeleton(network):
    """Return a copy of `network` with only structural attributes preserved."""
    skel = network.__class__()
    skel.graph.update(network.graph)  # preserve graph-level attrs (e.g. crs)
    for n, data in network.nodes(data=True):
        skel.add_node(n, **{k: v for k, v in data.items() if k in NW_SKELETON_NODE_ATTRS})
    if network.is_multigraph():
        for u, v, k, data in network.edges(keys=True, data=True):
            attrs = {kk: vv for kk, vv in data.items() if kk in NW_SKELETON_EDGE_ATTRS}
            skel.add_edge(u, v, key=k, **attrs)
    else:
        for u, v, data in network.edges(data=True):
            attrs = {kk: vv for kk, vv in data.items() if kk in NW_SKELETON_EDGE_ATTRS}
            skel.add_edge(u, v, **attrs)
    return skel


def _nodes_to_gdf(network) -> gpd.GeoDataFrame:
    """Extract nodes as a GeoDataFrame indexed on `node_id` (the registered id_col)."""
    ox = _require_osmnx()
    gdf = ox.graph_to_gdfs(network, edges=False, nodes=True)
    gdf.index.name = 'node_id'
    return gdf


def _edges_to_gdf(network) -> gpd.GeoDataFrame:
    """Extract edges as a GeoDataFrame indexed on synthetic `edge_id` (string).

    The natural (u, v[, key]) MultiIndex is flattened so it fits the
    single-`id_col` properties convention; reverse the flattening with
    `aperta.network_processing.parse_edge_id`.
    """
    ox = _require_osmnx()
    gdf = ox.graph_to_gdfs(network, edges=True, nodes=False, fill_edge_geometry=False)
    gdf = gdf.reset_index()
    if network.is_multigraph():
        gdf['edge_id'] = [_edge_id(u, v, k) for u, v, k in zip(gdf['u'], gdf['v'], gdf['key'])]
        gdf = gdf.drop(columns=['u', 'v', 'key'])
    else:
        gdf['edge_id'] = [_edge_id(u, v) for u, v in zip(gdf['u'], gdf['v'])]
        gdf = gdf.drop(columns=['u', 'v'])
    gdf = gdf.set_index('edge_id')
    return gdf


def _expand_nw_property_names(nw_name: str, value: str | list[str]) -> str | list[str]:
    """Expand short property suffixes to fully-qualified property names by
    prefixing the network's `data_name`.

    A value equal to `nw_name` itself passes through unchanged — this is
    how callers ask for the network's own base property file (the
    `properties/{nodes,edges}_<nw_name>.csv` written by
    `create_nw(save_properties=True)` with no `properties_name`), as
    promised by `get_nw`'s docstring. Without this carve-out the bare
    `nw_name` would expand to `<nw_name>_<nw_name>`, which has no file.
    """
    def expand(s: str) -> str:
        return s if s == nw_name else f'{nw_name}_{s}'
    if isinstance(value, str):
        return expand(value)
    return [expand(s) for s in value]


# =============================================================================
# ODM-shape helpers (used by Context.create_odm / Context.get_odm)
# =============================================================================

# `.npz` sub-array names for tiered ODMs. One file holds all populated tiers;
# each tier contributes three CSR-style arrays under a `<tier>__*` prefix:
# `<tier>__keys` (origin IDs preserved at original dtype — int OSM node IDs
# round-trip as ints, not strings), `<tier>__offsets` (k+1 int64 cumulative
# bounds; origin i's slice is `values[offsets[i]:offsets[i+1]]`), and
# `<tier>__values` (one concatenated array, all origins back-to-back).
# Plus a top-level `__tier_names__` array listing which tiers are present.
_ODM_TIER_NAMES_SENTINEL = '__tier_names__'
_ODM_KEYS_SUFFIX = '__keys'
_ODM_OFFSETS_SUFFIX = '__offsets'
_ODM_VALUES_SUFFIX = '__values'


def _odm_dict_value_len(odm_values) -> int:
    if isinstance(odm_values, np.ndarray):
        return odm_values.shape[-1] if odm_values.ndim >= 2 else odm_values.size
    return len(odm_values)


def _odm_n_items(odm: dict) -> int:
    return sum(_odm_dict_value_len(v) for v in odm.values())


def _pack_odm(odm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a dict-of-arrays ODM into CSR-style (keys, offsets, values).

    `keys` preserves the original dtype of the dict keys (int OSM node IDs
    round-trip as ints, not strings). Per-origin arrays must share a dtype
    so they can be concatenated.
    """
    keys = list(odm.keys())
    arrays = [v if isinstance(v, np.ndarray) else np.asarray(v) for v in odm.values()]
    lengths = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    offsets = np.empty(len(arrays) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    return np.asarray(keys), offsets, values


def _unpack_odm(
    keys: np.ndarray, offsets: np.ndarray, values: np.ndarray,
) -> dict:
    """Inverse of `_pack_odm`. Numeric arrays come back as read-only views
    into the shared `values` buffer (mutate via `.copy()`); string-dest
    arrays come back as native lists (no shared storage)."""
    is_string = values.dtype.kind in ('U', 'S')
    if not is_string:
        # Slices share storage with `values`; lock to prevent surprise
        # cross-origin mutation. Callers that need to mutate should
        # `.copy()` the slice first.
        values.flags.writeable = False
    odm: dict = {}
    for i, key in enumerate(keys.tolist()):
        sl = values[offsets[i]:offsets[i + 1]]
        odm[key] = sl.tolist() if is_string else sl
    return odm


def verify_match(a, b) -> None:
    """Verify two datasets are aligned by index (DataFrames) or keys (ODM dicts).

    Index comparison is order-insensitive (uses set semantics via
    `symmetric_difference`): two DataFrames with the same set of index
    values in different orders are considered matching. This matters for
    layered loads where one source (e.g. `networks_decorate.py`'s
    `_decorated` overlay) re-extracts edges from a graph after a
    `project_graph` round-trip, which changes osmnx's edge iteration
    order without changing the edge set.
    """
    if len(a) != len(b):
        raise DataError(f"Datasets are of different size ({len(a)} vs {len(b)}).")
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            raise DataError("Mismatch in keys between the two ODMs.")
        if any(_odm_dict_value_len(a[k]) != _odm_dict_value_len(b[k]) for k in a):
            raise DataError("Mismatch in destination count for at least one origin.")
        logging.info("Verified ODM-dict structure match.")
        return
    a_index = pd.Index(a.keys()) if isinstance(a, dict) else a.index
    b_index = pd.Index(b.keys()) if isinstance(b, dict) else b.index
    if not isinstance(a, dict) and not isinstance(b, dict):
        if a.index.name != b.index.name:
            raise DataError(f"Index names differ ({a.index.name}, {b.index.name}).")
    diff = a_index.symmetric_difference(b_index)
    if not diff.empty:
        sample = diff[: min(5, len(diff))].tolist()
        raise DataError(
            f"Indices do not match: {len(diff)} entries differ "
            f"(first few: {sample})."
        )


# Tiered-ODM tier order. Tuple of `TieredODNodePairs` field names —
# used both for iteration order and as the per-tier prefix inside the
# saved `.npz` archive (`<tier>__keys` etc.). `cells_to_cells` is
# always required; the others are optional.
_TIER_ODM_NAMES: tuple[str, ...] = (
    'cells_to_cells',
    'cells_to_zones',
    'zones_to_zones',
)


# =============================================================================
# Properties composition (no Context required — composes already-loaded frames)
# =============================================================================


def combine_properties(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Index-match and column-concatenate already-loaded property DataFrames.

    Use this to compose properties from multiple sources (or to add a transformed
    frame to a loaded one):

        historic = context.source('preparation/switzerland/historic')
        surveys  = context.source('preparation/switzerland/surveys')
        df = combine_properties(
            historic.get_properties('cells', 'core'),
            surveys.get_properties('cells', 'survey_extras').rename(columns={'foo': 'survey_foo'}),
        )

    All inputs must share the same index. The duplicated index-as-column (added
    by `get_properties`) is collapsed automatically.
    """
    if not dfs:
        raise DataError("combine_properties requires at least one DataFrame.")
    cleaned: list[pd.DataFrame] = []
    for df in dfs:
        if df.index.name and df.index.name in df.columns:
            cleaned.append(df.drop(columns=df.index.name))
        else:
            cleaned.append(df.copy())
    for i in range(len(cleaned) - 1):
        verify_match(cleaned[i], cleaned[i + 1])
    out = pd.concat(cleaned, axis=1)
    if len(out.columns) > len(set(out.columns)):
        vc = out.columns.value_counts()
        raise DataError(f"Duplicate columns after combine_properties: {list(vc.index[vc > 1])}")
    if out.index.name:
        out[out.index.name] = out.index
    return out


@dataclass(frozen=True)
class Context:
    """Frozen-ish dataclass holding context info: paths, parameters, dep tracking.

    Get one via `init_context()`; do not instantiate directly. `cached_property`
    properties memoize lazily without breaking frozen-ness.
    """

    parent: Self | None
    variant: any
    project: str | None
    scenario: str | None
    namespace: str   # e.g. 'preparation/switzerland/historic', 'atlas/main'
    caller_root_path: str
    caller_file_path: str
    caller_base_name: str
    data_dir_public: str
    data_dir_private: str
    working_dir: str
    read_only: bool
    track_dependencies: bool   # opt-in via APERTA_TRACK_DEPENDENCIES=1 in .env
    start_time: float
    env: dict
    created_data: set
    used_data: set
    # Per-source override for `default_storage`. Set by `source(path,
    # storage=...)` to pin a source context to a specific storage class
    # (useful when a namespace's STORAGE constant is wrong for this read,
    # or unreachable cross-repo). None → fall back to dynamic-import lookup.
    default_storage_override: 'Storage | None' = None

    # ----------------------- namespace introspection ----------------------

    @cached_property
    def role(self) -> str:
        """`'preparation'` for `preparation/...` namespaces; `'project'` for
        project namespaces (anything where `project` is set)."""
        if self.project is None:
            if not self.namespace.startswith('preparation/'):
                raise ContextError(
                    f"Context with project=None must have a preparation/... namespace "
                    f"(got '{self.namespace}').")
            return 'preparation'
        return 'project'

    @cached_property
    def namespace_remainder(self) -> str:
        """Namespace with leading role-segment stripped.

        For preparation: `'preparation/switzerland/historic'` → `'switzerland/historic'`.
        For project namespaces: strips the leading `<project>/` segment so a script
        in `src/main/x.py` (namespace `atlas/main`) yields remainder `'main'`.
        """
        if self.namespace.startswith('preparation/'):
            return self.namespace[len('preparation/'):]
        if self.project is not None and self.namespace.startswith(f'{self.project}/'):
            return self.namespace[len(self.project) + 1:]
        return self.namespace

    # ----------------------- path resolution ------------------------------

    def path_for(self, storage: 'Storage', relative_path: str) -> str:
        """Resolve a typed asset path. See the module docstring for the path
        templates. The public/private distinction is implicit in which root
        is used (DATA_DIR_PUBLIC vs DATA_DIR_PRIVATE), so it does not appear
        as a path segment.
        """
        # Sub-path under the storage root depends on the namespace role.
        if self.project is None:
            # preparation namespace
            sub = os.path.join('preparation', self.namespace_remainder)
        else:
            assert self.scenario is not None
            sub = os.path.join(self.project, self.scenario)

        # Storage-class → on-disk root. PUBLIC / PRIVATE / SCRATCH are uniform
        # across roles; RESULTS is role-specific (forbidden under preparation).
        if storage == Storage.PUBLIC:
            base = os.path.join(self.data_dir_public, sub)
        elif storage == Storage.PRIVATE:
            base = os.path.join(self.data_dir_private, sub)
        elif storage == Storage.SCRATCH:
            base = os.path.join(self.working_dir, 'scratch', sub)
        elif storage == Storage.RESULTS:
            if self.project is None:
                raise DataError(
                    f"Storage.RESULTS is not allowed from preparation/* namespaces "
                    f"(current namespace: '{self.namespace}'). Preparation produces "
                    f"prepared data, not results.")
            assert self.scenario is not None
            base = os.path.join(self.working_dir, 'results', self.project, self.scenario)
        else:
            raise ValueError(f"Unknown storage: {storage!r}")
        return os.path.join(base, relative_path)

    def raw_path(self, storage, relative_path: str) -> str:
        """Resolve a path under the raw-data tree.

        Raw data is read-only external/upstream input that no aperta-atlas script
        produced (e.g. swisstopo TLM, OFS census downloads). Lives at:

            <DATA_DIR_PUBLIC>/raw/<rel>     (Storage.PUBLIC)
            <DATA_DIR_PRIVATE>/raw/<rel>    (Storage.PRIVATE)

        Unlike `path_for`, raw paths do not include role / namespace / scenario
        segments — the caller supplies the full relative path under `raw/`.

        Storage.SCRATCH and Storage.RESULTS are invalid here — raw is by definition
        external read-only data.
        """
        if storage == Storage.PUBLIC:
            return os.path.join(self.data_dir_public, 'raw', relative_path)
        if storage == Storage.PRIVATE:
            return os.path.join(self.data_dir_private, 'raw', relative_path)
        raise DataError(
            f"`raw_path` accepts only Storage.PUBLIC or Storage.PRIVATE; "
            f"got {storage}. Raw data is read-only external input — SCRATCH and "
            f"RESULTS storage classes don't apply.")

    # ----------------------- status / dep tracking ------------------------

    @cached_property
    def initial_status(self):
        return tracking.read_status(self)

    @cached_property
    def status_script_name(self) -> str:
        return f'{self.namespace}/{self.caller_base_name}'

    @cached_property
    def status_scenario_name(self) -> str:
        """Scenario name for status.json keying. `_` for scenario-free (preparation)."""
        return self.scenario if self.scenario is not None else _NO_SCENARIO

    @cached_property
    def status_variant_name(self) -> str:
        return self.variant_suffix or 'default'

    @cached_property
    def variant_suffix(self) -> str:
        if self.variant:
            return '_'.join(str(p) for p in self.variant)
        return ''

    @cached_property
    def default_storage(self):
        """Default `Storage` class for I/O calls that don't specify
        `storage=`.

        Lookup chain (highest priority first):
          1. `default_storage_override` — set by `init_context` from the
             caller-script's own `STORAGE` constant (for preparation
             scripts), or by `source(path, storage=...)` on the caller
             side (for cross-namespace reads).
          2. For project namespaces: `SCENARIOS[scenario].storage` via
             dynamic import of the top-level `scenarios` module.
          3. Fallback: `Storage.PUBLIC`. For preparation source contexts
             this means: if the caller of `source()` didn't pass `storage=`,
             reads go to the PUBLIC root. Reads from PRIVATE preparation
             namespaces must pass `storage=Storage.PRIVATE` explicitly.
        """
        if self.default_storage_override is not None:
            return self.default_storage_override
        if self.project is None:
            return Storage.PUBLIC
        if self.scenario is None:
            return Storage.PUBLIC
        storage = _read_scenario_storage(self.scenario)
        return storage if storage is not None else Storage.PUBLIC


    def _path_label(self, storage, relative_path: str) -> str:
        """Human-readable storage label used in status.json
        (`STORAGE[<namespace>__<scenario>]/<rel>`)."""
        s = f"__{self.scenario}" if self.scenario else ''
        return f"{storage.name}[{self.namespace}{s}]/{relative_path}"

    def _register_data(
        self, relative_path: str, storage, n: int | None, *, action: str,
    ) -> str:
        """Log the I/O event and (when tracking is on) record the labelled path
        on the root context's `{created,used}_data` set. Returns the path label
        for any caller-side follow-up (e.g. dependency check).
        """
        path_str = self._path_label(storage, relative_path)
        size = f" (n = {n:,})" if n else ""
        logging.info(f"{action} `{path_str}`{size}")
        if self.track_dependencies:
            target = self.parent if self.parent else self
            (target.created_data if action == "Created" else target.used_data).add(path_str)
        return path_str

    def register_created_data(self, relative_path: str, storage, n: int | None) -> None:
        self._register_data(relative_path, storage, n, action="Created")

    def register_used_data(self, relative_path: str, storage, n: int | None) -> None:
        path_str = self._register_data(relative_path, storage, n, action="Loaded")
        if self.track_dependencies:
            tracking.check_dependencies(self, [path_str], False)

    def close(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        script = self.caller_base_name
        if not self.track_dependencies:
            logging.info(f"Task `{script}` completed; took {elapsed:.1f} seconds.")
        else:
            status = tracking.update_status(self)
            cur = status[self.status_script_name][self.status_scenario_name][self.status_variant_name]
            logging.info(f"Task `{script}` completed; took {cur['runtime']:.1f} seconds.")
        self._log_level_summary()

    @staticmethod
    def _log_level_summary() -> None:
        """Emit a one-line color-coded summary of log records emitted at
        each severity during this script run (WARNING / NOTE / INFO).
        Counts are collected by the module-level `_LEVEL_COUNTER` handler
        attached in `_install_level_counter`."""
        if _LEVEL_COUNTER is None:
            return
        n_err = _LEVEL_COUNTER.counts.get(logging.ERROR, 0)
        n_warn = _LEVEL_COUNTER.counts.get(logging.WARNING, 0)
        n_note = _LEVEL_COUNTER.counts.get(NOTE_LEVEL, 0)
        n_info = _LEVEL_COUNTER.counts.get(logging.INFO, 0)
        # Green when the count is 0 (nothing to worry about); the
        # severity color otherwise (bright-red bg for errors, red fg for
        # warnings, yellow fg for notes).
        GREEN = "\033[1;32m"
        RESET = "\033[0m"
        parts = []
        if n_err:
            parts.append(f"\033[1;41m{n_err} error{'s' if n_err != 1 else ''}{RESET}")
        warn_color = GREEN if n_warn == 0 else "\033[1;31m"
        parts.append(f"{warn_color}{n_warn} warning{'s' if n_warn != 1 else ''}{RESET}")
        note_color = GREEN if n_note == 0 else "\033[1;33m"
        parts.append(f"{note_color}{n_note} note{'s' if n_note != 1 else ''}{RESET}")
        parts.append(f"{n_info} info line{'s' if n_info != 1 else ''}")
        logging.info(f"Log summary — {'  '.join(parts)}")

    # ----------------------- cross-namespace reads ------------------------

    def source(
        self,
        path: str,
        *,
        storage: 'Storage | None' = None,
    ) -> 'Context':
        """Read-only Context for another namespace; used to read data created
        elsewhere. The `path` argument mirrors the on-disk layout:

          context.source('preparation/switzerland/historic')
              → preparation read (scenario-free).

          context.source('atlas/cambridgeuk-public')
              → same-project cross-scenario read (when calling from atlas)
                or cross-project read (when calling from another project).
                The foreign project's `scenarios.py` doesn't need to be
                importable — paths are constructed from the strings.

          context.source(f'{context.project}/{other_scenario}')
              → idiomatic same-project, different-scenario form.

        `storage=` pins the source ctx's `default_storage` to a specific
        class. Useful when:

          - A preparation namespace mixes PUBLIC and PRIVATE data and you
            know which one this read targets.
          - You're sourcing cross-repo and the foreign STORAGE / SCENARIOS
            lookup falls back to PUBLIC but the data is actually PRIVATE.
          - You want the source's typed I/O calls to land in a specific
            root without passing `storage=` on every single get_/create_.
        """
        if not path or path.startswith('/') or path.endswith('/'):
            raise ValueError(
                f"`source` path must be a non-empty, slash-separated string "
                f"with no leading/trailing slash. Got {path!r}.")
        parts = path.split('/')
        first = parts[0]

        if first == 'preparation':
            if len(parts) < 2:
                raise ValueError(
                    f"Preparation source path needs at least one segment "
                    f"after 'preparation/'. Got {path!r}.")
            sub_project = None
            sub_scenario = None
            sub_namespace = path
        else:
            # Project source: '<project>/<scenario>[/<extra>]'
            if len(parts) < 2:
                raise ValueError(
                    f"Project source path must be '<project>/<scenario>' "
                    f"(got {path!r}). For same-project cross-scenario reads "
                    f"use f'{{context.project}}/{{other_scenario}}'.")
            sub_project = first
            sub_scenario = parts[1]
            sub_namespace = path

        return replace(
            self,
            parent=self.parent if self.parent else self,
            project=sub_project,
            scenario=sub_scenario,
            namespace=sub_namespace,
            read_only=True,
            created_data=set(),
            used_data=set(),
            default_storage_override=storage,
        )

    # =====================================================================
    # Typed I/O — shapes, properties, ODMs, networks, generic, results
    # =====================================================================

    def _verify_writable(self, storage: Storage) -> None:
        if self.read_only:
            raise DataError("Data creation not allowed for read-only context.")
        if storage == Storage.RESULTS and self.project is None:
            raise DataError(
                f"Storage.RESULTS is only allowed from project namespaces; "
                f"current namespace: '{self.namespace}'.",
            )

    def _generic_relative_path(self, storage: Storage, relative_path: str) -> str:
        """Resolve where a generic file lives, relative to its storage root.

        In project namespaces, generic files live under a `generic/` typed-
        subfolder so they don't collide with shapes/properties/odm conventions.
        Preparation outputs are scenario-free and skip the segment. Results
        files live under their own root and keep their relative path verbatim.
        """
        if storage == Storage.RESULTS or self.project is None:
            return relative_path
        p = Path(relative_path)
        return f"generic/{p.parent}/{p.stem}{p.suffix}".replace('/./', '/')

    # ------------------------ shapes (geometries) ------------------------

    def create_shapes(
        self,
        gdf: gpd.GeoDataFrame,
        data_name: str | None = None,
        storage: Storage | None = None,
        extra_columns: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Write a GeoDataFrame to `shapes/<geo_type>[_<data_name>].gpkg`.

        Columns are filtered to a minimal allowlist (`geometry`,
        `centroid_*`, `area_m2`, `*_id_int`) so shape files stay
        compact; non-geometry attributes belong in companion
        `properties/<geo_type>_<data_name>.csv` files via
        `create_properties`. Pass `extra_columns` to opt-in additional
        identity-level attributes — typical use is buildings'
        `building` (OSM tag), which is fundamental to the unit's
        identity and shouldn't require a two-file load to access.
        """
        storage = storage or self.default_storage
        if 'geometry' not in gdf.columns or not isinstance(gdf, gpd.GeoDataFrame):
            raise DataError("`create_shapes` requires a GeoDataFrame with a `geometry` column.")
        self._verify_writable(storage)
        _verify_unique_index(gdf)

        geo_type = _get_geo_type_str(df=gdf)
        _warn_if_unknown_unit(geo_type, where='create_shapes')
        relative = f"shapes/{geo_type}{_file_suffix(data_name)}.gpkg"
        full = self.path_for(storage, relative)
        Path(full).parent.mkdir(parents=True, exist_ok=True)

        allowed = {'geometry', 'centroid_x', 'centroid_y', 'centroid_lat', 'centroid_lon',
                   'area_m2', 'area_m2_geometry'}
        if extra_columns:
            allowed = allowed | set(extra_columns)
        allowed_suffix = ('_id_int',)
        cols = [c for c in gdf.columns if c in allowed or c.endswith(allowed_suffix)]
        gdf[cols].to_file(full)
        self.register_created_data(relative, storage, len(gdf))

    def get_shapes(
        self,
        geo_type: str | list[str],
        data_name: str | None = None,
        storage: Storage | None = None,
        allow_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        """Load a `shapes/<geo_type>[_<data_name>].gpkg` and return as GeoDataFrame
        indexed by the file's first column (the registered `id_col`).
        """
        storage = storage or self.default_storage
        geo_type_str = _get_geo_type_str(geo_type=geo_type)
        _warn_if_unknown_unit(geo_type_str, where='get_shapes')
        relative = f"shapes/{geo_type_str}{_file_suffix(data_name)}.gpkg"
        cache_key = (relative, storage)
        if cache_key in cache:
            return cache[cache_key]
        full = self.path_for(storage, relative)
        gdf = gpd.read_file(full)
        gdf = gdf.set_index(gdf.columns[0])
        _verify_index_matches_geo_type(gdf, geo_type_str, where='get_shapes')
        gdf[gdf.index.name] = gdf.index
        if allow_cache:
            cache[cache_key] = gdf
        self.register_used_data(relative, storage, len(gdf))
        _verify_index_type(gdf)
        return gdf

    # ----------- properties (tabular per-unit attrs, indexed by id) ------

    def create_properties(
        self,
        df: pd.DataFrame | gpd.GeoDataFrame,
        data_name: str | None = None,
        storage: Storage | None = None,
        float_format: str | None = None,
    ) -> None:
        """Write a DataFrame to `properties/<geo_type>[_<data_name>].csv`.

        `float_format` is passed verbatim to `DataFrame.to_csv` — typical
        values: `'%.4f'` (fixed-point), `'%.6g'` (general), or
        `'{:.3f}'.format`. `None` keeps pandas' default (full precision),
        which produces the largest files.
        """
        storage = storage or self.default_storage
        self._verify_writable(storage)
        _verify_unique_index(df)
        geo_type = _get_geo_type_str(df=df)
        _warn_if_unknown_unit(geo_type, where='create_properties')
        relative = f"properties/{geo_type}{_file_suffix(data_name)}.csv"
        full = self.path_for(storage, relative)
        Path(full).parent.mkdir(parents=True, exist_ok=True)
        df[[c for c in df.columns if c != 'geometry']].to_csv(full, float_format=float_format)
        self.register_created_data(relative, storage, len(df))

    def _load_properties(
        self,
        geo_type: str,
        data_name: str,
        storage: Storage,
        allow_cache: bool,
    ) -> pd.DataFrame:
        geo_type_str = _get_geo_type_str(geo_type=geo_type)
        _warn_if_unknown_unit(geo_type_str, where='get_properties')
        relative = f"properties/{geo_type_str}{_file_suffix(data_name)}.csv"
        cache_key = (relative, storage)
        if cache_key in cache:
            return cache[cache_key]
        full = self.path_for(storage, relative)
        # `low_memory=False` disables pandas' chunked-read dtype-inference,
        # which throws `IndexError: list index out of range` in
        # `_concatenate_chunks` on multi-million-row property files where
        # dtypes differ between chunks (a known pandas bug). Cost: peak
        # parse memory roughly equals final DataFrame size — fine for our
        # property files, which are bounded by node/edge count.
        df = pd.read_csv(full, index_col=0, low_memory=False)
        _verify_index_matches_geo_type(df, geo_type_str, where='get_properties')
        if allow_cache:
            cache[cache_key] = df
        self.register_used_data(relative, storage, len(df))
        _verify_index_type(df)
        return df

    @overload
    def get_properties(
        self,
        geo_type: str,
        data_name: str | list[str],
        storage: Storage | None = ...,
        allow_cache: bool = ...,
        add_shapes: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @overload
    def get_properties(
        self,
        geo_type: str,
        data_name: str | list[str],
        storage: Storage | None = ...,
        allow_cache: bool = ...,
        add_shapes: Literal[True] | str = ...,
    ) -> gpd.GeoDataFrame: ...

    def get_properties(
        self,
        geo_type: str,
        data_name: str | list[str],
        storage: Storage | None = None,
        allow_cache: bool = True,
        add_shapes: bool | str = False,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """Load properties for `geo_type`; optionally join multiple `data_name`
        files (column-wise) and attach shapes.

        For cross-source composition, load each frame separately (typically via
        `context.source(...).get_properties(...)`) and merge with `combine_properties`.

        Return type narrows on `add_shapes` (via overloads): `pd.DataFrame` when
        False (default), `gpd.GeoDataFrame` when True or a str (the shapes data_name).
        """
        storage = storage or self.default_storage
        geo_type_str = _get_geo_type_str(geo_type=geo_type)
        names = [data_name] if isinstance(data_name, str) else list(data_name)
        dfs = [self._load_properties(geo_type_str, n, storage, allow_cache) for n in names]

        if len(dfs) == 1:
            df = dfs[0]
        else:
            for i in range(len(dfs) - 1):
                verify_match(dfs[i], dfs[i + 1])
            # "Later wins" on column-name collisions — supports the canonical
            # overlay-layering pattern (e.g., `networks_decorate.py`'s
            # `_decorated` CSV intentionally re-emits some base columns with
            # refined values: collapsed `highway`, NaN-filled `lanes`).
            # Walk dfs from last back to first; drop any column from an
            # earlier df that is also present in a later one.
            seen: set[str] = set()
            kept: list[pd.DataFrame] = []
            for d in reversed(dfs):
                cols = [c for c in d.columns if c not in seen]
                kept.append(d[cols])
                seen.update(cols)
            kept.reverse()
            # `pd.concat(axis=1)` does an outer-join on index — handles the
            # case where verify_match accepted same-set/different-order indices.
            df = pd.concat(kept, axis=1)

        # Defensive backstop: after the override dedup above, duplicates
        # should be impossible from layered loads. Still useful if a single
        # property file somehow ships duplicate columns.
        if len(df.columns) > len(set(df.columns)):
            vc = df.columns.value_counts()
            raise DataError(f"Duplicate columns in loaded data: {list(vc.index[vc > 1])}")
        df[df.index.name] = df.index

        if add_shapes:
            shapes_name = add_shapes if isinstance(add_shapes, str) else None
            s = self.get_shapes(geo_type, shapes_name, storage=storage)
            verify_match(df, s)
            df = df.join(s[[c for c in s.columns if c not in df.columns and c != 'geometry']])
            return gpd.GeoDataFrame(df, geometry=s.geometry, crs=s.crs)
        return df

    # ------------------- ODMs (sparse OD dicts) --------------------------

    def create_tiered_odm(
        self,
        pairs,                              # aperta.od_pairs.TieredODPairs
        network_name: str,
        data_name: str,
        storage: Storage | None = None,
        allow_cache: bool = True,
    ) -> None:
        """Write a `TieredODPairs` to a single `.npz` on disk.

        File on disk: `odm/<network_name>_<data_name>.npz` — one file per
        (network, data_name) pair, holding every populated tier.
        Conceptually an ODM "belongs" to a network (more precisely, a set
        of nodes for that network).

        The three tiers have fixed origin / destination key types — the
        tier name tells you unambiguously what each row/column is:
            `cells_to_cells`  — origin = cell_id, destination = cell_id
                                (close-range pairs at cell resolution).
            `cells_to_zones`  — origin = cell_id, destination = zone_id
                                (medium-range; cell-origin precision
                                preserved, dest aggregated to zones).
            `zones_to_zones`  — origin = zone_id, destination = zone_id
                                (long-range pairs at zone resolution).
        (This is why there's no `'type'` data_name anymore — the tier name
        already encodes the destination geo-type.)

        `data_name` describes what the per-OD-pair values mean across all
        tiers:
            `'idx'`         — destination IDs (str or int); list[str]
                              values are cast to U-dtype string arrays on
                              save and converted back to lists on load so
                              helpers using `.index(...)` keep working.
            `<other>`       — typically np.float32 (travel times, line
                              distances, generalised utilities, …).

        If you genuinely need only one tier (`cells_to_cells`), construct
        the `TieredODNodePairs` with `cells_to_zones=None` /
        `zones_to_zones=None` and pass it here — only the populated tier
        is written. Variants of the same logical ODM go via the
        `data_name` (e.g. `'travel_time_detoured'`).

        On-disk format: `.npz` (numpy zipped archive) with
        `allow_pickle=False` — pickle-free (safe to load from untrusted
        sources), preserves array dtypes (int8/float32), zlib-compressed
        for compactness. Each populated tier `<t>` contributes three
        CSR-style arrays under a `<t>__*` prefix (`<t>__keys`,
        `<t>__offsets`, `<t>__values`); a top-level `__tier_names__` array
        lists which tiers are present.

        To check an ODM matches the network it claims to belong to, call
        `aperta.network_processing.verify_odm_against_network(odm, nodes)`
        on each populated tier dict.
        """
        storage = storage or self.default_storage
        self._verify_writable(storage)
        relative = f"odm/{network_name}_{data_name}.npz"
        full = self.path_for(storage, relative)
        Path(full).parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {}
        populated: list[str] = []
        total_items = 0
        for field in _TIER_ODM_NAMES:
            tier = getattr(pairs, field)
            if tier is None:
                continue
            if not isinstance(tier, dict):
                raise DataError(
                    f"Tier {field!r} must be a dict (got {type(tier).__name__}).")
            keys, offsets, values = _pack_odm(tier)
            payload[f'{field}{_ODM_KEYS_SUFFIX}']    = keys
            payload[f'{field}{_ODM_OFFSETS_SUFFIX}'] = offsets
            payload[f'{field}{_ODM_VALUES_SUFFIX}']  = values
            populated.append(field)
            total_items += _odm_n_items(tier)
        if not populated:
            raise DataError(
                "TieredODPairs has no populated tiers — refusing to write an "
                "empty file. Populate `cells_to_cells` at minimum.")
        payload[_ODM_TIER_NAMES_SENTINEL] = np.asarray(populated)

        # `**payload` unpacking confuses pyright (it can't statically prove
        # no key collides with `np.savez_compressed`'s `allow_pickle` kwarg).
        # Runtime is correct — our keys are all our own sentinels/suffixes.
        np.savez_compressed(full, **payload)  # type: ignore[arg-type]
        if allow_cache:
            cache[(relative, storage)] = pairs
        self.register_created_data(relative, storage, total_items)

    def get_tiered_odm(
        self,
        network_name: str,
        data_name: str,
        storage: Storage | None = None,
        allow_cache: bool = True,
    ):
        """Load a `TieredODPairs` from the single `.npz` written by
        `create_tiered_odm`.

        Returned numeric arrays are read-only views into one shared
        backing buffer (the `.npz`'s `values` array). Mutate via
        `arr.copy()` — direct in-place writes raise `ValueError`.
        String-dest ODMs (`data_name='idx'`) come back as native lists
        and don't share storage.

        At least one tier must be populated on-disk — raises `DataError`
        otherwise. Mirrors `create_tiered_odm`'s "any populated tier
        suffices" behavior; tiers not present in the file load as
        `None` on the returned `TieredODPairs`. (Example: the NPVM PT
        zone-to-zone matrix has only `zones_to_zones` populated; both
        cell-tier fields come back as `None`.)
        """
        from aperta.od_pairs import TieredODNodePairs  # local import — avoid cycle
        storage = storage or self.default_storage
        relative = f"odm/{network_name}_{data_name}.npz"
        cache_key = (relative, storage)
        if cache_key in cache:
            return cache[cache_key]
        full = self.path_for(storage, relative)

        loaded: dict[str, dict | None] = {f: None for f in _TIER_ODM_NAMES}
        total_items = 0
        with np.load(full, allow_pickle=False) as z:
            if _ODM_TIER_NAMES_SENTINEL not in z.files:
                raise DataError(
                    f"Tiered ODM file at {full!r} is missing the "
                    f"{_ODM_TIER_NAMES_SENTINEL!r} manifest. Re-create it.")
            present = {str(t) for t in z[_ODM_TIER_NAMES_SENTINEL]}
            for field in _TIER_ODM_NAMES:
                if field not in present:
                    continue
                odm = _unpack_odm(
                    z[f'{field}{_ODM_KEYS_SUFFIX}'],
                    z[f'{field}{_ODM_OFFSETS_SUFFIX}'],
                    z[f'{field}{_ODM_VALUES_SUFFIX}'],
                )
                loaded[field] = odm
                total_items += _odm_n_items(odm)
        if all(loaded[f] is None for f in _TIER_ODM_NAMES):
            raise DataError(
                f"Tiered ODM at {full!r} has no populated tiers. "
                f"Re-create it.")

        pairs = TieredODNodePairs(
            cells_to_cells=loaded['cells_to_cells'],
            cells_to_zones=loaded['cells_to_zones'],
            zones_to_zones=loaded['zones_to_zones'],
        )
        if allow_cache:
            cache[cache_key] = pairs
        self.register_used_data(relative, storage, total_items)
        return pairs

    # ----------- networks (.graphml skeleton + companion shapes/properties) -

    def create_nw(
        self,
        network,
        data_name: str | None = None,
        storage: Storage | None = None,
        save_skeleton: bool = True,
        save_shapes: bool | str = False,
        save_properties: bool | str = False,
        properties_name: str | None = None,
        properties_columns: list[str] | None = None,
    ) -> None:
        """Write a network as a `.graphml` skeleton, with optional companion files.

        The .graphml stores only structural attributes (`NW_SKELETON_NODE_ATTRS`
        for nodes, `NW_SKELETON_EDGE_ATTRS` for edges, plus graph-level attrs
        like `crs`). Type-fragile attributes — computed metrics, weights, etc.
        — are written as separate CSVs via `save_properties` (or via follow-up
        `create_properties` calls) for type fidelity that `.graphml` doesn't
        provide reliably.

        Args:
            network: networkx graph (typically a MultiDiGraph from osmnx/momepy).
            data_name: identifier for the network on disk (e.g. 'driving', '2020').
                Maps to `nw/<data_name>.graphml` (defaults to `'network'`).
            save_skeleton: write the .graphml. Set False to skip — useful when
                layering an additional property table onto an existing network
                without rewriting the skeleton (e.g. after computing extra edge
                metrics in a later script).
            save_shapes: False | True | 'nodes' | 'edges'. If truthy, also write
                node and/or edge geometries via `create_shapes`
                (`shapes/{nodes,edges}_<data_name>.gpkg`).
            save_properties: False | True | 'nodes' | 'edges'. If truthy, also
                write node and/or edge non-skeleton attributes via
                `create_properties`
                (`properties/{nodes,edges}_<data_name>[_<properties_name>].csv`).
            properties_name: **suffix** appended to `data_name` for property
                files. When None, the file is just
                `properties/edges_<data_name>.csv`; otherwise it becomes
                `properties/edges_<data_name>_<properties_name>.csv`. This
                matches the suffix convention used by `get_nw`'s
                `add_*_properties=`, so a `properties_name='core'` write pairs
                cleanly with an `add_edge_properties='core'` read.
            properties_columns: optional column allowlist when `save_properties`
                is set. Restricts the written CSV to these columns (applies to
                nodes and/or edges, whichever side `save_properties` enables).
                Use this when saving a layered property table containing only
                newly-computed columns — otherwise base columns from the source
                graph would be re-written and conflict on later joins via
                `add_*_properties`.
        """
        import networkx as nx
        storage = storage or self.default_storage
        if not isinstance(network, nx.Graph):
            raise DataError("`create_nw` requires a networkx graph.")
        self._verify_writable(storage)
        nw_name = data_name or 'network'

        if save_skeleton:
            relative = f'nw/{nw_name}.graphml'
            full = self.path_for(storage, relative)
            Path(full).parent.mkdir(parents=True, exist_ok=True)
            ox = _require_osmnx()
            ox.save_graphml(_to_skeleton(network), filepath=full)
            self.register_created_data(relative, storage, None)

        if save_shapes:
            if save_shapes is True or save_shapes == 'nodes':
                self.create_shapes(_nodes_to_gdf(network), data_name=data_name, storage=storage)
            if save_shapes is True or save_shapes == 'edges':
                self.create_shapes(_edges_to_gdf(network), data_name=data_name, storage=storage)

        if save_properties:
            # Suffix semantics: properties_name is appended to data_name to keep
            # the network's own properties under one prefix, matching the
            # `get_nw` reader.
            props_name = nw_name if properties_name is None else f'{nw_name}_{properties_name}'
            if save_properties is True or save_properties == 'nodes':
                n_gdf = _nodes_to_gdf(network)
                drop = NW_SKELETON_NODE_ATTRS | {'geometry'}
                n_props = n_gdf.drop(columns=[c for c in drop if c in n_gdf.columns])
                if properties_columns is not None:
                    n_props = n_props[[c for c in properties_columns if c in n_props.columns]]
                if len(n_props.columns) > 0:
                    self.create_properties(n_props, data_name=props_name, storage=storage)
            if save_properties is True or save_properties == 'edges':
                e_gdf = _edges_to_gdf(network)
                drop = NW_SKELETON_EDGE_ATTRS | {'geometry'}
                e_props = e_gdf.drop(columns=[c for c in drop if c in e_gdf.columns])
                if properties_columns is not None:
                    e_props = e_props[[c for c in properties_columns if c in e_props.columns]]
                if len(e_props.columns) > 0:
                    self.create_properties(e_props, data_name=props_name, storage=storage)

    def get_nw(
        self,
        data_name: str | None = None,
        storage: Storage | None = None,
        add_node_properties: str | list[str] | None = None,
        add_edge_properties: str | list[str] | None = None,
        allow_cache: bool = False,
    ):
        """Read `nw/<data_name>.graphml` and optionally merge same-context
        property tables.

        For cross-source property merges (e.g. props from a `.source(...)`
        namespace), load the network here and attach via
        `aperta.network_processing.attach_node_properties` /
        `attach_edge_properties` afterwards.

        Args:
            add_node_properties: data_name (or list) for `get_properties('nodes', ...)`
                on `self`, attached to the loaded graph as node attributes.
                Short forms (`'core'`, `'land_use'`) are auto-prefixed with
                `<data_name>_` so you don't have to repeat the network name;
                fully-qualified names (`'nw_2020_core'`, or a bare `'nw_2020'`
                for the property file written by `create_nw(save_properties=True)`)
                are also accepted.
            add_edge_properties: same, for edges.
        """
        storage = storage or self.default_storage
        nw_name = data_name or 'network'
        relative = f'nw/{nw_name}.graphml'
        cache_key = (relative, storage)
        if allow_cache and cache_key in cache:
            nw = cache[cache_key]
        else:
            full = self.path_for(storage, relative)
            ox = _require_osmnx()
            nw = ox.load_graphml(full)
            self.register_used_data(relative, storage, None)
            if allow_cache:
                cache[cache_key] = nw

        # If layering properties, work on a copy so the cache stays clean.
        if add_node_properties is not None or add_edge_properties is not None:
            from aperta.network_processing import (
                attach_edge_properties,
                attach_node_properties,
            )
            nw = nw.copy()
            if add_node_properties is not None:
                attach_node_properties(nw, self.get_properties(
                    'nodes',
                    _expand_nw_property_names(nw_name, add_node_properties),
                    storage=storage, allow_cache=allow_cache,
                ))
            if add_edge_properties is not None:
                attach_edge_properties(nw, self.get_properties(
                    'edges',
                    _expand_nw_property_names(nw_name, add_edge_properties),
                    storage=storage, allow_cache=allow_cache,
                ))
        return nw

    # -------------- generic files (.csv .gpkg .npy .graphml .json .asc) --

    def create_generic(
        self,
        payload,
        relative_path: str,
        storage: Storage | None = None,
        verify_unique_index: bool = True,
        kws: dict | None = None,
    ) -> None:
        """Write `payload` to `<typed_sub>/<relative_path>` based on the file
        extension.

        Dispatches on the file extension of `relative_path`:
          .csv, .gpkg, .npy, .asc, .graphml, .json — via the obvious writer
          .png/.pdf/.svg/.jpg — `payload` is a matplotlib Figure (saved via .savefig)

        In project namespaces, generic files live under a `generic/` typed-
        subfolder so they don't collide with shapes/properties/odm conventions.
        Preparation outputs are scenario-free and skip the segment. Results
        files live under their own root and keep their relative path verbatim.
        """
        storage = storage or self.default_storage
        self._verify_writable(storage)
        kws = kws or {}
        p = Path(relative_path)
        relative = self._generic_relative_path(storage, relative_path)
        full = self.path_for(storage, relative)
        Path(full).parent.mkdir(parents=True, exist_ok=True)

        n: int | None
        match p.suffix:
            case '.csv':
                if verify_unique_index:
                    _verify_unique_index(payload)
                payload.to_csv(full, **kws)
                n = len(payload)
            case '.gpkg':
                if verify_unique_index:
                    _verify_unique_index(payload)
                payload.to_file(full, **kws)
                n = len(payload)
            case '.asc':
                fmt = "%.1f" if kws.get('as_float') else "%d"
                np.savetxt(full, payload, fmt=fmt)
                n = len(payload)
            case '.npy':
                np.save(full, payload, allow_pickle=True)
                n = None
            case '.graphml':
                ox = _require_osmnx()
                ox.save_graphml(payload, filepath=full)
                n = None
            case '.json':
                with open(full, 'w') as fp:
                    json.dump(payload, fp)
                n = len(payload)
            case '.png' | '.pdf' | '.svg' | '.jpg' | '.jpeg':
                # `payload` is a matplotlib Figure. `bbox_inches='tight'` is the
                # near-universal default for publication-ready figures; override
                # via `kws` if you need a different bounding box.
                kws.setdefault('bbox_inches', 'tight')
                payload.savefig(full, **kws)
                n = None
            case _:
                raise DataError(f"File extension not supported: {relative_path}")
        self.register_created_data(relative, storage, n)

    def get_generic(
        self,
        relative_path: str,
        storage: Storage | None = None,
        allow_cache: bool = True,
        kws: dict | None = None,
    ) -> any:
        """Load a previously-saved generic file by extension. Inverse of
        `create_generic`. See its docstring for the dispatch table and the
        typed-subfolder rules.
        """
        storage = storage or self.default_storage
        p = Path(relative_path)
        relative = self._generic_relative_path(storage, relative_path)
        cache_key = (relative, storage)
        if cache_key in cache:
            return cache[cache_key]
        full = self.path_for(storage, relative)

        n: int | None
        match p.suffix:
            case '.csv':
                kws = dict(kws) if kws else {}
                kws.setdefault('index_col', 0)
                payload = pd.read_csv(full, **kws)
                n = len(payload)
            case '.gpkg':
                payload = gpd.read_file(full)
                n = len(payload)
            case '.npy':
                payload = np.load(full, allow_pickle=True).item()
                n = None
            case '.graphml':
                ox = _require_osmnx()
                payload = ox.load_graphml(full)
                n = None
            case '.json':
                with open(full) as fp:
                    payload = json.load(fp)
                n = None
            case _:
                raise DataError(f"File extension not supported: {relative_path}")
        if allow_cache:
            cache[cache_key] = payload
        self.register_used_data(relative, storage, n)
        return payload

    # -------- results — sugar on top of generic; projects/ namespaces only -

    def create_results(self, payload, relative_path: str, kws: dict | None = None) -> None:
        """Convenience for `create_generic(..., storage=Storage.RESULTS, ...)`.
        Only callable from project namespaces — preparation produces prepared
        data, not results.
        """
        self.create_generic(payload, relative_path, storage=Storage.RESULTS, kws=kws)

    def get_results(self, relative_path: str, kws: dict | None = None) -> any:
        """Convenience for `get_generic(..., storage=Storage.RESULTS, ...)`."""
        return self.get_generic(
            relative_path, storage=Storage.RESULTS, allow_cache=True, kws=kws,
        )

    # ------------------------ coefficient I/O ----------------------------

    def get_coefs(self, name: str) -> pd.DataFrame:
        """Load coef `name` as a DataFrame.

        Path depends on context type:

        - **Project context** — subfolder is determined by the scenario's
          declaration in `scenarios.SCENARIOS[<this scenario>].coefs[<name>]`:
            - `Calibrate()`   → reads from `coefs/calibrated/<name>.csv`
            - `ImportFrom(X)` → reads from `coefs/transferred/<name>.csv`
            - `HandWritten()` → reads from `coefs/manual/<name>.csv`
        - **Namespace context** (preparation) — reads from
          `coefs/<name>.csv` (no `<kind>/` subfolder; preparation has a
          single source, "computed during this run").

        Raises `FileNotFoundError` if the file doesn't exist — for
        `Calibrate()` / `ImportFrom`, that means the calibration step
        hasn't been run for this scenario yet; for `HandWritten()`, the
        user needs to place the file; for preparation, the script that
        produces the coef hasn't been run yet.
        """
        if self.project is None or self.scenario is None:
            rel = f'coefs/{name}.csv'
            source_repr = 'preparation-namespace coef'
        else:
            from aperta_atlas import coefs as _coefs
            source = _coefs._get_source(self, name)
            kind = _coefs._kind_for(source)
            rel = f'coefs/{kind}/{name}.csv'
            source_repr = repr(source)
        path = self.path_for(self.default_storage, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Coef {name!r} expected at {path} but not found. "
                f"Source declaration: {source_repr}.")
        # Header-row count: detect MultiIndex columns (calibrated bike's
        # multi-level (profile, kind) layout) by sniffing the second row.
        df = _read_coefs_csv(path)
        self.register_used_data(rel, self.default_storage, n=len(df))
        return df

    def create_coefs(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        kind: str = 'calibrated',
    ) -> None:
        """Write a coef DataFrame to `coefs/<kind>/<name>.csv`.

        - **Project context** — `coefs/<kind>/<name>.csv`, where `kind`
          is `'calibrated'` (default) or `'transferred'` (used internally
          by the `coefs.resolve` dispatcher when copying from another
          scenario). Manual files are placed by hand, not via this method.
        - **Namespace context** (preparation) — `coefs/<name>.csv`
          (no `<kind>/` subfolder). The `kind` argument is rejected for
          namespace contexts: preparation has a single source ("computed
          during this run"), no dispatcher choosing between options.
        """
        storage = self.default_storage
        self._verify_writable(storage)
        if self.project is None or self.scenario is None:
            if kind != 'calibrated':
                raise DataError(
                    f"`create_coefs(kind=...)` is project-only. Preparation "
                    f"namespaces have a single source ('computed during this "
                    f"run') and don't use the <kind>/ subfolder. "
                    f"Drop the `kind=` argument. Got kind={kind!r}.")
            rel = f'coefs/{name}.csv'
        else:
            if kind not in ('calibrated', 'transferred'):
                raise DataError(
                    f"kind must be 'calibrated' or 'transferred' (manual files "
                    f"are placed by hand, not via create_coefs). Got {kind!r}.")
            rel = f'coefs/{kind}/{name}.csv'
        out_path = self.path_for(storage, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _write_coefs_csv(df, out_path)
        self.register_created_data(rel, storage, n=len(df))


def _read_coefs_csv(path: str) -> pd.DataFrame:
    """Load a coef CSV with auto-detected header layout.

    The standard layout is `param` index + one column per profile (single
    header row). Bike's calibrated file uses a 2-row header to bundle
    `coef` / `p` per profile — detected by peeking at the second row's
    first cell (non-numeric → 2-row header).
    """
    with open(path) as f:
        f.readline()  # header row 0
        peek = f.readline().split(',', 2)
    is_2row = len(peek) >= 2 and peek[1].strip() and not _looks_numeric(peek[1])
    header = [0, 1] if is_2row else 0
    return pd.read_csv(path, index_col=0, header=header)


def _write_coefs_csv(df: pd.DataFrame, path: str) -> None:
    """Write a coef DataFrame with `param` index + profile columns."""
    df.to_csv(path, float_format='%.4f')


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_dir(default: str, alt: str | None, os_name_alt: str | None) -> str:
    if alt and os_name_alt and os.name == os_name_alt:
        return alt
    return default


def _read_scenarios_module():
    """Dynamic-import the top-level `scenarios` module (None on failure).

    Each repo's `src/scenarios.py` declares `PROJECT_NAME`, `DEFAULT_SCENARIO`,
    `SCENARIOS`. The module is expected to live on `sys.path` (typical for
    scripts run via `python -m main.X` from `src/`).
    """
    import importlib
    try:
        return importlib.import_module('scenarios')
    except ImportError:
        return None


def _read_project_name() -> str | None:
    """Read `PROJECT_NAME` from the top-level `scenarios` module (None if
    the module / constant isn't defined)."""
    mod = _read_scenarios_module()
    return getattr(mod, 'PROJECT_NAME', None) if mod else None


def _read_default_scenario() -> str | None:
    """Read `DEFAULT_SCENARIO` from the top-level `scenarios` module."""
    mod = _read_scenarios_module()
    return getattr(mod, 'DEFAULT_SCENARIO', None) if mod else None


def _read_scenario_storage(scenario: str) -> 'Storage | None':
    """Read `SCENARIOS[scenario].storage` from the top-level `scenarios`
    module. Returns `None` if any link is missing (caller falls back to
    `Storage.PUBLIC`)."""
    mod = _read_scenarios_module()
    if mod is None:
        return None
    scenarios_dict = getattr(mod, 'SCENARIOS', None)
    if scenarios_dict is None:
        return None
    s = scenarios_dict.get(scenario)
    if s is None:
        return None
    return getattr(s, 'storage', None)


def _read_storage_from_module(dotted_module_path: str) -> 'Storage | None':
    """Import a preparation sub-package and return its `STORAGE` attribute.

    Called by `init_context` on the CALLER'S OWN module path (the
    physical location of the running script), so `dotted_module_path`
    already contains any `private`/`public` segment — no path inference.
    Returns `None` if the import fails or no `STORAGE` is declared
    (caller decides the fallback).
    """
    import importlib
    try:
        mod = importlib.import_module(dotted_module_path)
    except ImportError:
        return None
    return getattr(mod, 'STORAGE', None)


def init_context(
    variant=None,
    scenario: str | None = None,
    namespace: str | None = None,
    caller_base_name: str | None = None,
) -> Context:
    """Initialize context. Most arguments are inferred from the caller path and CLI.

    Namespace is derived from the caller's path under `src/`:

      `src/preparation/<region>/<sub>/x.py`
          → namespace `preparation/<region>/<sub>` (scenario-free).
      `src/<subfolder>/<...>/x.py`        (any folder other than preparation/)
          → namespace `<PROJECT_NAME>/<subfolder>` (project-bound).

    `PROJECT_NAME` is read from the top-level `scenarios` module
    (`src/scenarios.py`). One repo = one project; the project name is in
    on-disk paths (`<DATA_DIR_*>/<project>/<scenario>/...`) so multiple
    project repos can share a `DATA_DIR_*` without collision.

    Scenario resolution for project namespaces (first non-None wins):
      1. explicit `scenario=` kwarg
      2. `--scenario` CLI flag
      3. `DEFAULT_SCENARIO` constant in the top-level `scenarios` module

    Cross-namespace reads use `context.source(...)`.
    """
    logging.basicConfig(level=env_values.get('LOGGING_LEVEL') or 'INFO',
                        format='%(levelname)-9s %(message)s [%(name)s]',
                        stream=sys.stdout)
    # Level-name strings are padded manually to 9 visible columns so
    # messages line up with the plain-text INFO output (whose 4-char
    # label is padded to 9 by the `%(levelname)-9s` format spec — but
    # that spec counts the ANSI-code bytes toward width, so colored
    # names get no automatic padding).
    logging.addLevelName(logging.WARNING, "\033[1;31m⚠ WARNING\033[1;0m")     # 9 visible
    logging.addLevelName(logging.ERROR,   "\033[1;41m⚠ ERROR\033[1;0m  ")     # 7 + 2 spaces
    logging.addLevelName(NOTE_LEVEL,      "\033[1;33m⚠ NOTE\033[1;0m   ")     # 6 + 3 spaces
    # pyogrio emits its own "Created N records" INFO on every write — duplicates
    # the `register_created_data` log we already emit. Silence its INFO traffic.
    logging.getLogger('pyogrio').setLevel(logging.WARNING)
    _install_level_counter()

    # init_context always returns a writable context. Read-only contexts come only
    # from `.source(...)`, which constructs Context() directly (not via init_context).
    read_only = False

    if caller_base_name is None:
        caller_base_name = os.path.basename(inspect.stack()[1][1])

    caller_full_path = os.path.abspath(inspect.stack()[1][1])
    re_module = re.compile(r'(.+?/)(src/(.+?)/[^/]+\.py)$')
    res = re_module.match(caller_full_path.replace('\\', '/'))
    caller_storage: 'Storage | None' = None
    if res is None:
        if namespace is None:
            raise ValueError(
                f"Invalid caller path: {caller_full_path}. Cannot infer namespace. "
                f"Pass `namespace=...` explicitly if calling from outside src/.")
        caller_root_path = ''
        caller_file_path = caller_base_name
    else:
        caller_root_path = res.group(1)
        caller_file_path = res.group(2)
        if namespace is None:
            parts = res.group(3).split('/')
            if parts[0] == 'preparation':
                # preparation/<region>/<sub>/<file>.py → preparation/<region>/<sub>.
                # `private`/`public` segments are **organizational** — they
                # group scripts by data-visibility class but don't appear in
                # the on-disk namespace, so consumers can keep using stable
                # paths like `preparation/switzerland/historic` regardless of
                # where the producing script physically lives.
                #
                # STORAGE is read directly from the caller's __init__.py at
                # its physical location (private/public preserved) and
                # stashed as `default_storage_override` so the running script's
                # writes go to the right root — no reverse lookup needed.
                caller_storage = _read_storage_from_module('.'.join(parts))
                parts = [p for p in parts if p not in ('private', 'public')]
                namespace = '/'.join(parts)
            else:
                # Any other layout is a project script. The namespace is
                # `<project>/<first-segment-under-src>`. The project name comes
                # from the top-level scenarios module (not the path).
                project_name = _read_project_name()
                if project_name is None:
                    raise ContextError(
                        f"Could not infer project name for caller {caller_full_path}. "
                        f"Add `PROJECT_NAME = '<name>'` to src/scenarios.py, or pass "
                        f"`namespace=...` explicitly.")
                if project_name == 'preparation':
                    raise ContextError(
                        f"`PROJECT_NAME = 'preparation'` is reserved (clashes with "
                        f"the shared preparation namespace). Pick a different name "
                        f"in src/scenarios.py.")
                namespace = f'{project_name}/{parts[0]}'
    assert namespace is not None

    # Determine project from namespace. preparation/... → no project.
    if namespace.startswith('preparation/'):
        project = None
    else:
        project = namespace.split('/', 1)[0] or None
        if project is None:
            raise ContextError(
                f"Could not derive project from namespace '{namespace}'.")

    working_dir = _get_dir(env_values['WORKING_DIR'],
                           env_values.get('WORKING_DIR_MACHINE2'),
                           env_values.get('OS_NAME_MACHINE2'))

    # Scenario resolution: kwarg > --scenario CLI > project DEFAULT_SCENARIO.
    if scenario is None:
        import argparse as _argparse
        _cli_parser = _argparse.ArgumentParser(add_help=False)
        _cli_parser.add_argument('--scenario', default=None)
        try:
            _cli_args, _ = _cli_parser.parse_known_args()
            scenario = _cli_args.scenario
        except SystemExit:
            scenario = None

    if project is not None:
        if scenario is None:
            scenario = _read_default_scenario()
        if scenario is None:
            raise ContextError(
                f"Project namespace '{namespace}' requires a scenario. Pass "
                f"`--scenario <name>` on the CLI or declare "
                f"`DEFAULT_SCENARIO = '<name>'` in src/scenarios.py.")
        if scenario == _NO_SCENARIO:
            raise ContextError(
                f"Scenario name '{_NO_SCENARIO}' is reserved. Pick a different name.")
    else:
        # preparation: scenario must not be set.
        if scenario is not None:
            raise ContextError(
                f"Preparation namespace '{namespace}' is scenario-free; passing "
                f"`--scenario {scenario}` is not allowed. Run without --scenario.")

    context = Context(
        parent=None,
        variant=variant,
        project=project,
        scenario=scenario,
        namespace=namespace,
        caller_root_path=caller_root_path,
        caller_file_path=caller_file_path,
        caller_base_name=caller_base_name,
        data_dir_public=_get_dir(env_values['DATA_DIR_PUBLIC'],
                                 env_values.get('DATA_DIR_PUBLIC_MACHINE2'),
                                 env_values.get('OS_NAME_MACHINE2')),
        data_dir_private=_get_dir(env_values['DATA_DIR_PRIVATE'],
                                  env_values.get('DATA_DIR_PRIVATE_MACHINE2'),
                                  env_values.get('OS_NAME_MACHINE2')),
        working_dir=working_dir,
        read_only=read_only,
        track_dependencies=_parse_bool_env('APERTA_TRACK_DEPENDENCIES'),
        start_time=time.perf_counter(),
        env=env_values,
        created_data=set(),
        used_data=set(),
        default_storage_override=caller_storage,
    )
    if not read_only:
        tracking.log_context_info(context)
    return context


def _parse_bool_env(key: str, default: bool = False) -> bool:
    """Read a boolean env var (`'1'`, `'true'`, `'yes'`, `'on'` → True; otherwise False)."""
    val = env_values.get(key)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')
