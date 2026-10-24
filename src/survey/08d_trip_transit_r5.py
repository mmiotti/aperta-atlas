"""
Timetable-based transit times (R5) for the MTMC transit trips: an
independent, unfitted reference for `technical_validation/times_vs_survey_transit.py`.

Each trip from survey/08b is routed with r5py on the Swiss GTFS timetable +
OSM street network, departing at the reported time of day. Variants:

  - `d2d` (door to door): the trip's own origin → destination. Full
    timetable routing incl. walk access / egress — the reference for the
    atlas's (NPVM-based) transit times.
  - `z2z` (zone to zone): representative points of the trip's origin and
    destination zones (population + employment-weighted centroid of the
    zone's cells). What a GTFS-based zone-to-zone table would give, to
    compare with NPVM.
  - `k<K>` (composed, e.g. `k3`): the access-point model planned for
    aperta (`planning/aperta_refactor_plan.md`, D7). The K nearest stop
    places at each trip end, the calibrated walk-network time (`rwalk`)
    from the trip's cell to each, and R5 between the stop places; then
    (a) the minimum over origin stops with the nearest egress stop, and
    (b) the minimum over origin and egress stops.

Timing: trips are mapped to the same weekday of a reference week within
the GTFS validity (`_REFERENCE_MONDAY`; survey dates are 2015 / 2021, so
timetable changes since then add noise), and grouped into `_SLOT_MIN`-minute
departure slots. Each slot is routed with a `_WINDOW`-long departure window
starting at the slot start, reporting the 1st, 10th and 50th percentile:
p1 ≈ best-timed departure (no initial wait), p10 ≈ departure timed to the
timetable, p50 = random departure. Within a slot, each variant routes its
unique origin and destination points once and looks each trip's pair up.

Transit only: R5 cannot exclude walk-only trips (its "direct" mode must equal
the access mode), so results for endpoint pairs closer than
`_WALK_ONLY_REACH_M` (the most walking R5 allows) are dropped — beyond that,
every result is a transit trip. Park-and-ride / bike-and-ride trips are
routed with walk access only, as in the atlas's transit model. Trips are
restricted to those passing `TRIP_QUALITY_FLAGS`.

R5 gotchas handled here: ids get an `o_` / `d_` prefix (r5py zeroes the
travel time when an origin and a destination share an id); the GTFS zip's
taxi routes (route_type 1500, rejected by R5) were recoded to 715 — see
`raw/switzerland/gtfs/README.txt`.

Prerequisites (not in the base env): `conda install -c conda-forge
"openjdk=21"` + `pip install r5py`; the zipped Swiss GTFS feed at
`raw/switzerland/gtfs/<_GTFS_FILE>` under DATA_DIR_PUBLIC. Builds the
national network (cached by r5py under ~/.cache/r5py; ~20 GB JVM heap).

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit.csv        # survey/08b (needs `datetime_departure`)
    shapes/cells.gpkg + properties/cells_{population,snap}.csv
    nw/walk.graphml + properties/edges_walk_calibrated.csv   # k<K> access
Inputs (PUBLIC):
    raw/global/osm/<area_name>_buffered-latest.osm.pbf + raw/switzerland/gtfs/<_GTFS_FILE>

Output (PRIVATE, under `<scenario>/`):
    generic/survey_trip_transit_r5.csv     # trip_id; t_r5_<variant>_p{1,10,50}
                                           # (seconds; NaN = unreachable within
                                           # `_MAX_TIME` or within walking reach)

Run (all trips and variants by default; `--n-trips N` for a random sample;
`--variants` subset (`all` = default; note this is 08d's own option, not the
scaffolding's `--variant`);
`--same-trips` adds variants to an earlier run's trips without re-routing it):
    python -m survey.08d_trip_transit_r5 --scenario switzerland-h10 --n-trips 5000
    python -m survey.08d_trip_transit_r5 --scenario switzerland-h10 --variants k3 --same-trips
"""

import argparse
import datetime as dt
import logging
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from aperta import network_snap, routing, routing_prep
from aperta.network_processing import attach_edge_properties
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step

from main.common import TRIP_QUALITY_FLAGS
from mode_configs import MODE_CONFIGS
from scenarios import get_scenario


_GTFS_FILE = 'gtfs_fp2026.zip'
_JVM_MAX_MEMORY = '20G'
_SPEED_WALKING_KMH = 4.5
_MAX_TIME_WALKING = dt.timedelta(minutes=30)   # access / egress / walk-only
_WALK_ONLY_REACH_M = _SPEED_WALKING_KMH / 3.6 * _MAX_TIME_WALKING.total_seconds()

_REFERENCE_MONDAY = dt.date(2026, 10, 12)   # no national holidays that week
_SLOT_MIN = 15
_WINDOW = dt.timedelta(minutes=30)
_MAX_TIME = dt.timedelta(hours=3)
_PERCENTILES = (1, 10, 50)
_POINT_VARIANTS = ('d2d', 'z2z')
_DEFAULT_VARIANTS = ('d2d', 'z2z', 'k3')
_WALK_SNAP_RADIUS_M = 500.0
_SEED = 42

OUTPUT_FILE = 'survey_trip_transit_r5.csv'


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _zone_points(context) -> pd.DataFrame:
    """Population + employment-weighted centroid (x, y) per zone, from cells."""
    # `zone_id` lives on shapes/cells.gpkg; `combined_total` = residents + jobs.
    cells = context.get_properties('cells', 'population', add_shapes=True)
    cen = cells.geometry.centroid
    df = pd.DataFrame({
        'zone_id': cells['zone_id'], 'w': cells['combined_total'].fillna(0),
        'x': cen.x, 'y': cen.y,
    }).dropna(subset=['zone_id', 'x', 'y'])
    df['w'] = df['w'] + 1e-6   # zones without residents / jobs → plain centroid
    df['wx'], df['wy'] = df['w'] * df['x'], df['w'] * df['y']
    g = df.groupby('zone_id')[['w', 'wx', 'wy']].sum()
    out = pd.DataFrame({'x': g['wx'] / g['w'], 'y': g['wy'] / g['w']})
    out.index = out.index.astype(str)
    return out


def _stop_areas(context, crs: str) -> pd.DataFrame:
    """GTFS stop places (Swiss feed: the `didok` number, shared by a parent
    station row and all its platforms / poles): id → mean (x, y) in `crs`."""
    path = context.raw_path(Storage.PUBLIC, f'switzerland/gtfs/{_GTFS_FILE}')
    with zipfile.ZipFile(path) as zf, zf.open('stops.txt') as f:
        stops = pd.read_csv(f, dtype={'stop_id': str, 'didok': str}, encoding='utf-8-sig')
    g = stops.groupby(stops['didok'].fillna(stops['stop_id']))[['stop_lon', 'stop_lat']].mean()
    pts = gpd.GeoSeries(gpd.points_from_xy(g['stop_lon'], g['stop_lat']),
                        crs='EPSG:4326').to_crs(crs)
    return pd.DataFrame({'x': pts.x.to_numpy(), 'y': pts.y.to_numpy()}, index=g.index)


def _kstop_access(context, trips: pd.DataFrame, areas: pd.DataFrame, k: int,
                  crs: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per trip end (`orig` / `dest`): the `k` nearest stop places (straight
    line) and the calibrated walk time (s; `rwalk`) between the trip's cell
    and each, as `(area_ids (n, k), walk_s (n, k))`. inf = beyond the
    walking limit, NaN = cell or stop place not snapped. The egress walk is
    routed from the destination cell (walk network ≈ symmetric)."""
    from technical_validation.access_exact_sample import CsrRouter   # numba prototype router
    tree = cKDTree(areas[['x', 'y']].to_numpy())
    cand = {}
    for side in ('orig', 'dest'):
        _, idx = tree.query(trips[[f'trip_{side}_x', f'trip_{side}_y']].to_numpy(), k=k)
        cand[side] = areas.index.to_numpy()[idx.reshape(len(trips), k)]

    src = MODE_CONFIGS['walk'].source_for('rwalk')
    graph = context.get_nw(data_name='walk', add_node_properties='core',
                           add_edge_properties='core', allow_cache=False)
    attach_edge_properties(graph, context.get_properties(
        'edges', src.edges_data_name)[[src.edge_column]])
    eligible = routing_prep.compute_snap_eligible_nodes(
        graph, directedness='directed_scc', cost_excluded_flag='cost_excluded_walk')
    used = np.unique(np.concatenate([cand['orig'].ravel(), cand['dest'].ravel()]))
    pts = gpd.GeoSeries(gpd.points_from_xy(areas.loc[used, 'x'], areas.loc[used, 'y']),
                        index=used, crs=crs)
    area_node, _ = network_snap.snap_to_network_nodes(
        pts, graph, max_distance=_WALK_SNAP_RADIUS_M, eligible_node_ids=eligible)
    csr, nx_to_seq, _ = routing._graph_to_csr(graph, src.edge_column)
    del graph
    area_seq = area_node.map(nx_to_seq)
    cell_node = context.get_properties('cells', 'snap')['node_id_walk']
    router = CsrRouter(csr)

    out = {}
    for side in ('orig', 'dest'):
        src_seq = trips[f'trip_{side}_cell'].map(cell_node).map(nx_to_seq).to_numpy()
        tgt_seq = area_seq.reindex(cand[side].ravel()).to_numpy().reshape(len(trips), k)
        walk = np.full((len(trips), k), np.nan)
        # Per-origin target lists must be duplicate-free: route each row's
        # unique snapped nodes, then map back to the k candidates.
        sources, t_ptr, t_nodes, back = [], [0], [], []
        for r in np.flatnonzero(np.isfinite(src_seq)):
            ok = np.isfinite(tgt_seq[r])
            if not ok.any():
                continue
            uniq, inv = np.unique(tgt_seq[r][ok].astype(np.int64), return_inverse=True)
            sources.append(int(src_seq[r]))
            t_nodes.extend(uniq.tolist())
            t_ptr.append(len(t_nodes))
            back.append((r, np.flatnonzero(ok), inv, t_ptr[-2]))
        costs = router.route(np.array(sources), np.array(t_ptr), np.array(t_nodes),
                             _MAX_TIME_WALKING.total_seconds())
        for r, cols, inv, start in back:
            walk[r, cols] = costs[start + inv]
        out[side] = (cand[side], walk)
        nearest = np.nanmin(np.where(np.isfinite(walk), walk, np.nan), axis=1)
        logging.info(f"  → {side}: walk to nearest of {k} stop places: median "
                     f"{np.nanmedian(nearest) / 60:.1f} min; "
                     f"{np.isnan(walk).all(axis=1).mean() * 100:.1f}% of trips without access")
    return out


# ---------------------------------------------------------------------------
# R5
# ---------------------------------------------------------------------------
def _r5_network(context, area_name: str):
    """Import r5py (JVM sized via `_JVM_MAX_MEMORY`) and build the R5
    network from the area's clipped OSM PBF + the Swiss GTFS feed."""
    if '--max-memory' not in sys.argv:   # r5py reads JVM options at import time
        sys.argv.extend(['--max-memory', _JVM_MAX_MEMORY])
    import r5py
    pbf = context.raw_path(Storage.PUBLIC, f'global/osm/{area_name}_buffered-latest.osm.pbf')
    gtfs = context.raw_path(Storage.PUBLIC, f'switzerland/gtfs/{_GTFS_FILE}')
    logging.info(f"  → R5 network from {pbf} + {gtfs}")
    return r5py, r5py.TransportNetwork(pbf, [gtfs])


def _travel_times(r5py, network, origins, destinations, departure) -> pd.DataFrame:
    """R5 matrix (minutes, percentiles `_PERCENTILES`) indexed by (from_id, to_id)."""
    return r5py.TravelTimeMatrix(
        network,
        origins=origins,
        destinations=destinations,
        departure=departure,
        departure_time_window=_WINDOW,
        percentiles=list(_PERCENTILES),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=_MAX_TIME,
        max_time_walking=_MAX_TIME_WALKING,
        speed_walking=_SPEED_WALKING_KMH,
    ).set_index(['from_id', 'to_id'])


def _points(keys: pd.Index, coords: pd.DataFrame, prefix: str, crs: str) -> gpd.GeoDataFrame:
    """R5 point set; ids prefixed (`o_` / `d_`) — see module docstring."""
    xy = coords.loc[keys]
    return gpd.GeoDataFrame({'id': prefix + keys.astype(str)},
                            geometry=gpd.points_from_xy(xy['x'], xy['y']),
                            crs=crs).to_crs('EPSG:4326')


def _route_points(r5py, network, departure, sub, o_key, d_key, coords, crs) -> dict:
    """`d2d` / `z2z` for one slot: one R5 matrix over the slot's unique
    endpoints; per trip its own pair (seconds), NaN within walking reach."""
    o_key, d_key = o_key.loc[sub.index], d_key.loc[sub.index]
    ttm = _travel_times(r5py, network,
                        _points(pd.Index(o_key.unique()), coords['o'], 'o_', crs),
                        _points(pd.Index(d_key.unique()), coords['d'], 'd_', crs), departure)
    pairs = pd.MultiIndex.from_arrays(
        ['o_' + o_key.astype(str).to_numpy(), 'd_' + d_key.astype(str).to_numpy()])
    walkable = np.hypot(*(coords['o'].loc[o_key].to_numpy()
                          - coords['d'].loc[d_key].to_numpy()).T) <= _WALK_ONLY_REACH_M
    out = {}
    for p in _PERCENTILES:
        vals = np.array(ttm[f'travel_time_p{p}'].reindex(pairs), dtype=float) * 60.0
        vals[walkable] = np.nan   # possibly a walk-only result
        out[p] = vals
    return out


def _route_kstop(r5py, network, departure, sub, trips, acc, areas, crs) -> dict:
    """Composed `k<K>` for one slot: R5 between the slot's candidate stop
    places, then per trip (a) min over origin stops with the nearest egress
    stop (by walk), (b) min over origin and egress stops (seconds). Stop-place
    pairs within walking reach are dropped (possibly walk-only results)."""
    rows = trips.index.get_indexer(sub.index)
    o_ids, o_walk = acc['orig'][0][rows], acc['orig'][1][rows]
    d_ids, d_walk = acc['dest'][0][rows], acc['dest'][1][rows]
    m, k = o_ids.shape
    ttm = _travel_times(r5py, network,
                        _points(pd.Index(np.unique(o_ids)), areas, 'o_', crs),
                        _points(pd.Index(np.unique(d_ids)), areas, 'd_', crs), departure)
    frm = np.repeat(o_ids, k, axis=1).ravel()          # (m, k_orig, k_dest) flattened
    to = np.tile(d_ids, (1, k)).ravel()
    pairs = pd.MultiIndex.from_arrays(['o_' + frm.astype(str), 'd_' + to.astype(str)])
    sep = np.hypot(areas.loc[frm, 'x'].to_numpy() - areas.loc[to, 'x'].to_numpy(),
                   areas.loc[frm, 'y'].to_numpy() - areas.loc[to, 'y'].to_numpy())
    a = np.where(np.isfinite(o_walk), o_walk, np.inf)[:, :, None]
    e = np.where(np.isfinite(d_walk), d_walk, np.inf)
    egress = np.argmin(e, axis=1)                       # nearest egress stop by walk
    out = {}
    for p in _PERCENTILES:
        t = np.array(ttm[f'travel_time_p{p}'].reindex(pairs), dtype=float) * 60.0
        t[~np.isfinite(t) | (sep <= _WALK_ONLY_REACH_M)] = np.inf
        tot = a + t.reshape(m, k, k) + e[:, None, :]
        for suffix, best in (('a', tot[np.arange(m), :, egress].min(axis=1)),
                             ('b', tot.min(axis=(1, 2)))):
            out[(suffix, p)] = np.where(np.isfinite(best), best, np.nan)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="R5 times for MTMC transit trips.")
    parser.add_argument('--n-trips', type=int, default=None,
                        help="Route a random sample of N trips.")
    parser.add_argument('--variants', default=','.join(_DEFAULT_VARIANTS),
                        help="Comma-separated subset of d2d, z2z, k<K> (composed, K "
                             "candidate stop places per trip end), or 'all' (the default).")
    parser.add_argument('--same-trips', action='store_true',
                        help="Route exactly the trips already in the output file (to add "
                             "variants to an earlier run); --n-trips is ignored.")
    args, _ = parser.parse_known_args()
    variants = (list(_DEFAULT_VARIANTS) if args.variants.strip() == 'all'
                else [v.strip() for v in args.variants.split(',') if v.strip()])
    point_variants = [v for v in variants if v in _POINT_VARIANTS]
    k_variants = [v for v in variants if v.startswith('k') and v[1:].isdigit()]
    unknown = set(variants) - set(point_variants) - set(k_variants)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    context = init_context()
    scenario = get_scenario(context.scenario)
    crs = scenario.crs_main

    with step('load trips (plausibility-filtered) + assign departure slots'):
        trips = context.get_generic('survey_trip_transit.csv', storage=Storage.PRIVATE)
        trips.index = trips.index.astype(str)
        if 'datetime_departure' not in trips.columns:
            raise KeyError("`datetime_departure` missing from survey_trip_transit.csv — "
                           "rerun survey/02d and survey/08b.")
        for flag in TRIP_QUALITY_FLAGS:
            if flag in trips.columns:
                trips = trips[trips[flag] == 1]
        trips = trips[pd.to_datetime(trips['datetime_departure'], errors='coerce').notna()]
        existing = None
        if args.same_trips:
            existing = context.get_generic(OUTPUT_FILE, storage=Storage.PRIVATE)
            existing.index = existing.index.astype(str)
            trips = trips[trips.index.isin(existing.index)]
            logging.info(f"  → --same-trips: {len(trips):,} of {len(existing):,} earlier "
                         f"trips still pass the plausibility flags")
        elif args.n_trips is not None and args.n_trips < len(trips):
            trips = trips.sample(n=args.n_trips, random_state=_SEED)
        trips = trips.copy()
        dep = pd.to_datetime(trips['datetime_departure'])
        trips['_weekday'] = dep.dt.dayofweek
        trips['_slot'] = ((dep.dt.hour * 60 + dep.dt.minute) // _SLOT_MIN) * _SLOT_MIN
        n_slots = trips.groupby(['_weekday', '_slot']).ngroups
        logging.info(f"  → {len(trips):,} trips in {n_slots:,} (weekday × "
                     f"{_SLOT_MIN}-min) departure slots")

    endpoints, access, areas = {}, {}, None
    if 'd2d' in point_variants:
        endpoints['d2d'] = (
            trips.index.to_series(), trips.index.to_series(),
            {'o': trips[['trip_orig_x', 'trip_orig_y']].set_axis(['x', 'y'], axis=1),
             'd': trips[['trip_dest_x', 'trip_dest_y']].set_axis(['x', 'y'], axis=1)})
    if 'z2z' in point_variants:
        with step('z2z: zone points (residents + jobs-weighted centroids)'):
            zones = _zone_points(context)
            endpoints['z2z'] = (trips['trip_orig_zone'].astype(str),
                                trips['trip_dest_zone'].astype(str), {'o': zones, 'd': zones})
    if k_variants:
        areas = _stop_areas(context, crs)
        logging.info(f"  → {len(areas):,} GTFS stop places")
        for kv in k_variants:   # before R5, so the walk graph and the JVM don't overlap
            with step(f'{kv}: {kv[1:]} candidate stop places per trip end + walk access'):
                access[kv] = _kstop_access(context, trips, areas, int(kv[1:]), crs)

    with step('build R5 transport network (national OSM + GTFS)'):
        r5py, network = _r5_network(context, scenario.area_name)

    cols = [f't_r5_{v}_p{p}' for v in point_variants for p in _PERCENTILES]
    cols += [f't_r5_{kv}{s}_p{p}' for kv in k_variants for s in 'ab' for p in _PERCENTILES]
    out = pd.DataFrame(np.nan, index=trips.index, columns=cols)
    with step(f'route slot by slot ({", ".join(variants)}; window {_WINDOW}, '
              f'percentiles {_PERCENTILES})'):
        for i, ((weekday, slot), sub) in enumerate(trips.groupby(['_weekday', '_slot'])):
            departure = (dt.datetime.combine(_REFERENCE_MONDAY, dt.time())
                         + dt.timedelta(days=int(weekday), minutes=int(slot)))
            for v in point_variants:
                o_key, d_key, coords = endpoints[v]
                res = _route_points(r5py, network, departure, sub, o_key, d_key, coords, crs)
                for p, vals in res.items():
                    out.loc[sub.index, f't_r5_{v}_p{p}'] = vals
            for kv in k_variants:
                res = _route_kstop(r5py, network, departure, sub, trips, access[kv], areas, crs)
                for (s, p), vals in res.items():
                    out.loc[sub.index, f't_r5_{kv}{s}_p{p}'] = vals
            if (i + 1) % 25 == 0 or i + 1 == n_slots:
                logging.info(f"  → {i + 1:,} / {n_slots:,} slots")

    with step('save'):
        if existing is not None:   # add / overwrite this run's columns only
            merged = existing.copy()
            for col in out.columns:
                merged[col] = out[col].reindex(merged.index)
            out = merged
        out.index.name = 'trip_id'
        context.create_generic(out, OUTPUT_FILE, storage=Storage.PRIVATE)
        for col in [c for c in out.columns if c.startswith('t_r5_')]:
            logging.info(f"  → {col}: {out[col].notna().mean() * 100:.1f}% reachable, "
                         f"median {out[col].median() / 60:.1f} min")

    context.close()


if __name__ == '__main__':
    main()
