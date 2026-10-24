"""
Materialize the z2z transit pair-index + z2z travel-time ODM from NPVM.

Two sections:
  A. **Pair index** — z2z-only `TieredODGeoPairs`, keyed by zone_id
     strings (matches NPVM's key format): all pairs of zones with a
     car-graph snap whose centroids lie within `r_zones`.
  B. **NPVM z2z fill** — look up travel times in NPVM's country-wide
     matrix for every pair in the scenario set; missing pairs and pairs
     beyond `TRANSIT_MODE_CONFIG.time_cutoff_s` → NaN (as road pairs beyond
     their routing cutoff).

Sibling to `05_road_od_times.py` — same pipeline stage but a
fundamentally different mechanism (matrix lookup, no routing). A
future GTFS-based transit variant would replace THIS file wholesale
without touching the road script.

Downstream 08a lifts the z2z ODM to cell tiers (via `cell_to_zone`)
and bakes the per-cell overhead — the cell-tier equivalent of what
road's Dijkstra provides directly.

Inputs (scenario storage, under <scenario>/):
    properties/zones_snap.csv + shapes/zones.gpkg

Inputs (PUBLIC, preparation):
    preparation/switzerland/npvm/odm/npvm_2023_transit_{idx,travel_time}.npz

Outputs (scenario storage, under <scenario>/):
    odm/transit_z2z_pairs.npz       # zone-id-keyed z2z pair index
    odm/transit_time_net_npvm.npz   # z2z travel times (seconds); `npvm`
                                    # names the source (GTFS variant → `_gtfs`)

Run:
    python -m main.05_transit_od_times --scenario <name>
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from aperta.od_pairs import TieredODGeoPairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import npvm_transit_z2z_lookup
from mode_configs import TRANSIT_MODE_CONFIG
from scenarios import get_scenario


# ---------------------------------------------------------------------------
# Section A: transit pair index
# ---------------------------------------------------------------------------


def _build_transit_pairs(zones) -> TieredODGeoPairs:
    """z2z-only pair index — for each zone, every zone within
    `r_zones` Euclidean distance on centroid, keyed by zone_id.

    A direct KDTree query: `od_pairs.get_pairs` is cells-first, and a
    zone-only output from it needs `r_cells = r_medium = 0` plus a reindex.
    TODO: support zone-only pairs in `get_pairs`.

    Zones without a car snap are excluded — matches the origin /
    destination set NPVM's atlas subset was built against.
    """
    r = TRANSIT_MODE_CONFIG.radii.r_zones
    with step(f'mode=transit: build z2z pair set (r_zones={r:.0f})'):
        zones_m = zones[zones['node_id_car'].notna()]
        centroids = zones_m.geometry.centroid
        coords = np.column_stack([
            centroids.x.to_numpy(), centroids.y.to_numpy(),
        ])
        z_ids = zones_m.index.to_numpy()
        tree = cKDTree(coords)
        z2z_out: dict = {}
        for i, zid in enumerate(z_ids):
            hits = tree.query_ball_point(coords[i], r=r)
            # Sorted for reproducibility across runs; downstream lookups
            # are dict-based so order carries no other meaning.
            z2z_out[zid] = np.asarray(sorted(z_ids[j] for j in hits))
        pairs = TieredODGeoPairs(zones_to_zones=z2z_out)
        n_pairs = sum(len(a) for a in z2z_out.values())
        logging.info(f"  → z2z: {len(z2z_out):,} zone origins / {n_pairs:,} OD pairs (c2c + c2z: None by design)")
        return pairs


# ---------------------------------------------------------------------------
# Section B: NPVM z2z fill
# ---------------------------------------------------------------------------


def _fill_transit_z2z_from_npvm(
    context, transit_pairs: TieredODGeoPairs,
) -> None:
    """Fill z2z travel times from NPVM's country-wide matrix.
    `transit_pairs` must be geo-keyed (zone_id strings) so the NPVM
    lookup matches. Missing scenario pairs and pairs beyond the cutoff
    → NaN (downstream treats as unreachable)."""
    z2z_idx = transit_pairs.zones_to_zones or {}

    with step('mode=transit: load NPVM 2023 z2z travel times'):
        lookup = npvm_transit_z2z_lookup(context)
        logging.info(f"  → NPVM covers {len(lookup):,} z2z pairs")

    cutoff_s = TRANSIT_MODE_CONFIG.time_cutoff_s
    with step(f'mode=transit: filter NPVM to scenario pair set (cutoff {cutoff_s:.0f} s) + save'):
        z2z_out: dict = {}
        n_missing = n_beyond = 0
        for orig, dests in z2z_idx.items():
            cost = np.empty(len(dests), dtype=np.float32)
            for i, dest in enumerate(dests):
                v = lookup.get((orig, dest))
                if v is None:
                    n_missing += 1
                    cost[i] = np.nan
                elif v > cutoff_s:
                    n_beyond += 1
                    cost[i] = np.nan
                else:
                    cost[i] = v
            z2z_out[orig] = cost
        logging.info(
            f"  → {n_beyond:,} z2z pairs beyond the {cutoff_s / 60:.0f}-min cutoff → NaN")
        if n_missing:
            n_total = sum(len(a) for a in z2z_idx.values())
            pct = 100 * n_missing / max(n_total, 1)
            logging.warning(
                f"  ⚠ {n_missing:,} of {n_total:,} scenario z2z pairs "
                f"({pct:.1f}%) missing from NPVM — stored as NaN. "
                f"If pct is near 100%, check the zone-id key format: "
                f"NPVM uses 'ZD…'/'ZF…' strings.")
        costs = TieredODGeoPairs(
            cells_to_cells=None,
            cells_to_zones=None,
            zones_to_zones=z2z_out,
        )
        context.create_tiered_odm(
            costs, network_name='transit', data_name='time_net_npvm')
        finite = np.concatenate([
            a[np.isfinite(a)] for a in z2z_out.values() if len(a)
        ])
        if len(finite):
            logging.info(
                f"  → z2z travel time (finite entries only): "
                f"median={float(np.median(finite)):.0f} s, "
                f"P95={float(np.quantile(finite, 0.95)):.0f} s")


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    if scenario.transit_source is None:
        logging.info(f"Scenario {scenario.name!r} has no transit; skipping.")
        context.close()
        return

    with step('load zones (snap + shapes)'):
        zones = context.get_properties(
            'zones', ['snap'], add_shapes=True)
        logging.info(f"  → {len(zones):,} zones loaded")

    # ---- Section A: build the transit pair index ------------------------
    transit_pairs = _build_transit_pairs(zones)
    with step('mode=transit: save z2z-only tiered OD pair index'):
        context.create_tiered_odm(transit_pairs, network_name='transit', data_name='z2z_pairs')

    # ---- Section B: fill z2z from NPVM using the in-memory pair index ---
    _fill_transit_z2z_from_npvm(context, transit_pairs)

    context.close()


if __name__ == '__main__':
    main()
