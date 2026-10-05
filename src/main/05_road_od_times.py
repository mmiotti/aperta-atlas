"""
Materialize tiered OD pair-index + net-time cost ODMs for ONE road mode.

Two sections, run in order in `main`:

  A. **Pair index** — build a `TieredODNodePairs` for the selected mode
     (walk / bike / car), save as `odm/<mode>_node_pairs.npz`.
  B. **Road routing** — for each profile of the mode (walk: `rwalk`;
     bike: `rbike / ebike25 / ebike45`; car: `base / peak / night`),
     scipy Dijkstra on the per-mode lean graph using the
     `duration_calibrated_<profile>` edge weights, save as
     `odm/<mode>_time_net_<profile>.npz`.

Split into per-mode variants so bike / car / walk can be re-run
selectively, parallelised across shells, or memory-isolated (only one
mode's pair index in RAM). No cross-mode data flow — split is purely
mechanical. Default (`--variant all` / omitted) runs every mode
sequentially. Variant list comes from `mode_configs.MODE_CONFIGS` at
import time; scenarios with a mode subset skip the missing ones.

Transit lives in a sibling `05_transit_od_times.py` — different
mechanism (NPVM z2z lookup, no routing).

Tier classification (from `aperta.od_pairs.get_pairs`):
    d(Z, Z') < r_cells          → cells_to_cells  (close)
    r_cells ≤ d < r_medium      → cells_to_zones  (medium)
    r_medium ≤ d < r_zones      → zones_to_zones  (far)
    d ≥ r_zones                 → dropped

Cells are filtered to those with mode-specific snap + `zone_id` in the
prepared zones layer. Origins are further restricted to `is_active == 1`
(cells that snapped to ALL network types); destinations include the
buffer-only cells that only snap to some networks.

Inputs (PUBLIC, under <scenario>/):
    properties/cells_{population,snap}.csv + shapes/cells.gpkg
    properties/zones_{population,snap}.csv + shapes/zones.gpkg
    nw/<mode>.graphml + properties/edges_<mode>_calibrated.csv   # from 04

Outputs (PUBLIC, under <scenario>/):
    odm/<mode>_node_pairs.npz                             # section A — node-keyed tiered pair index
    odm/walk_time_net_rwalk.npz                           # section B (mode=walk)
    odm/bike_time_net_{rbike,ebike25,ebike45}.npz         # section B (mode=bike)
    odm/car_time_net_{base,peak,night}.npz                # section B (mode=car)

Run all modes sequentially (default):
    python -m main.05_road_od_times --scenario <name>
Single mode:
    python -m main.05_road_od_times --scenario <name> --variant <walk|bike|car>
"""

import logging
from typing import cast

import numpy as np

from aperta import od_pairs, routing
from aperta.network_processing import attach_edge_properties
from aperta.od_pairs import TieredODNodePairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from mode_configs import MODE_CONFIGS, CalibratedSource, ModeConfig
from scenarios import get_scenario


# Variants: one per road mode. Populated from `MODE_CONFIGS`
# (canonical enumeration in `mode_configs.py`); scenario subsets are
# honoured at runtime in `main()`.
variants = Variants([('mode', str)])
for _mode_name in MODE_CONFIGS:
    variants.add(name=_mode_name, mode=_mode_name)


def _tier_origins(tier: dict | None) -> int:
    return len(tier or {})


def _tier_pairs(tier: dict | None) -> int:
    return sum(len(a) for a in (tier or {}).values())


def _median_tier_cost(tier: dict | None) -> float:
    """Median over finite values across a tier — for a quick log sanity check."""
    if not tier:
        return float('nan')
    vals = np.concatenate(list(tier.values()))
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float('nan')


# ---------------------------------------------------------------------------
# Section A: pair index
# ---------------------------------------------------------------------------


def _build_road_pairs(
    mode_config: ModeConfig, cells, zones,
) -> TieredODNodePairs:
    """Build the tiered OD-pair index for one road mode. Filters cells
    + zones to those with valid mode-specific snap, then classifies each
    (origin, dest) into c2c / c2z / z2z per the mode's radii."""
    mode = mode_config.mode
    radii = mode_config.radii
    node_col = f'node_id_{mode}'

    with step(f'mode={mode}: filter cells/zones for valid {node_col}'):
        cells_m = cells.copy()
        zones_m = zones.copy()
        # `get_pairs` reads `node_column` from a single column; remap the
        # per-mode snap into the generic name.
        cells_m['node_id'] = cells_m[node_col]
        zones_m['node_id'] = zones_m[node_col]

        n_cells_before = len(cells_m)
        n_zones_before = len(zones_m)
        # Filter zones first — its survivors set the `zone_id.isin(...)`
        # predicate for cells.
        zones_m = zones_m[zones_m['node_id'].notna()]
        # Keep all routable cells as DESTINATIONS (no combined_total > 0
        # filter — POI-only cells are legitimate destinations). The
        # `orig_cells=is_active` mask below restricts ORIGINS.
        no_snap = cells_m['node_id'].isna()
        no_zone = ~no_snap & cells_m['zone_id'].isna()
        zone_dropped = ~no_snap & ~no_zone & ~cells_m['zone_id'].isin(zones_m.index)
        keep = ~(no_snap | no_zone | zone_dropped)
        aoi_before = cells_m['is_aoi']
        cells_m = cells_m[keep]
        n_aoi = int(cells_m['is_aoi'].sum())
        logging.info(
            f"  → cells: {len(cells_m):,}/{n_cells_before:,} kept "
            f"({n_aoi:,} AOI origins, {len(cells_m) - n_aoi:,} buffer-only destinations); "
            f"zones: {len(zones_m):,}/{n_zones_before:,} kept")
        logging.info(
            f"  → drops by cause [all / of which AOI]: "
            f"no_snap={int(no_snap.sum()):,} / {int((no_snap & aoi_before).sum()):,}, "
            f"no_zone={int(no_zone.sum()):,} / {int((no_zone & aoi_before).sum()):,}, "
            f"zone_dropped={int(zone_dropped.sum()):,} / "
            f"{int((zone_dropped & aoi_before).sum()):,}")

    with step(
        f'mode={mode}: build tiered OD pairs '
        f'(r_cells={radii.r_cells:.0f}, r_medium={radii.r_medium:.0f}, '
        f'r_zones={radii.r_zones:.0f})'
    ):
        pairs = od_pairs.get_pairs(
            cells_m,
            r_cells=radii.r_cells,
            node_column='node_id',
            zones=zones_m,
            r_zones=radii.r_zones,
            r_medium=radii.r_medium,
            orig_cells=cells_m['is_active'] == 1,
        )
        logging.info(
            f"  → tiers (origins / OD pairs): "
            f"c2c={_tier_origins(pairs.cells_to_cells):,} / "
            f"{_tier_pairs(pairs.cells_to_cells):,}, "
            f"c2z={_tier_origins(pairs.cells_to_zones):,} / "
            f"{_tier_pairs(pairs.cells_to_zones):,}, "
            f"z2z={_tier_origins(pairs.zones_to_zones):,} / "
            f"{_tier_pairs(pairs.zones_to_zones):,}")
        return pairs


# ---------------------------------------------------------------------------
# Section B: road routing
# ---------------------------------------------------------------------------


def _load_lean_graph(
    context, mode_config: ModeConfig, src: CalibratedSource,
):
    """Load `<mode>.graphml` + attach one profile's calibrated-duration
    edge column. Returns `(graph, weight_col)`."""
    graph = context.get_nw(data_name=mode_config.mode, allow_cache=False)
    calibrated = context.get_properties('edges', src.edges_data_name)
    attach_edge_properties(graph, calibrated[[src.edge_column]])
    return graph, src.edge_column


def _route_road_profile(
    context, mode_config: ModeConfig, src: CalibratedSource,
    pairs: TieredODNodePairs,
) -> None:
    """Route one road profile with scipy Dijkstra and save the resulting
    net-time cost ODM. Uses the per-mode Dijkstra cutoff to bound the
    per-origin frontier without changing in-bound results."""
    with step(f'profile={src.name}: load lean graph + route + save'):
        graph, weight_col = _load_lean_graph(context, mode_config, src)
        # `cast`: `tiered_path_costs` returns the subclass matching `pairs`.
        costs = cast(TieredODNodePairs, routing.tiered_path_costs(graph, pairs,
            weight=weight_col,
            cutoff=mode_config.time_cutoff_s,
        ))
        context.create_tiered_odm(
            costs,
            network_name=mode_config.mode,
            data_name=src.cost_data_name,
        )
        logging.info(
            f"  → tier medians (s): "
            f"c2c={_median_tier_cost(costs.cells_to_cells):.0f}, "
            f"c2z={_median_tier_cost(costs.cells_to_zones):.0f}, "
            f"z2z={_median_tier_cost(costs.zones_to_zones):.0f}")


def main(variant) -> None:
    context = init_context(variant)
    scenario = get_scenario(context.scenario)

    # Scenarios may declare a subset of MODE_CONFIGS; log + skip if this
    # variant's mode is absent, so `--variant all` sails past missing modes.
    mode_configs = scenario.mode_configs
    if variant.mode not in mode_configs:
        logging.info(f"mode={variant.mode!r} not in scenario mode_configs ({list(mode_configs)}); skipping")
        context.close()
        return
    mode_config = mode_configs[variant.mode]

    with step('load cells + zones (population + per-mode snap + shapes)'):
        # `allow_cache=False` so `del` below actually frees these —
        # context's default cache holds a strong ref otherwise.
        cells = context.get_properties(
            'cells', ['population', 'snap'], add_shapes=True,
            allow_cache=False)
        zones = context.get_properties(
            'zones', ['population', 'snap'], add_shapes=True,
            allow_cache=False)
        logging.info(f"  → {len(cells):,} cells, {len(zones):,} zones loaded")

    # ---- Section A: build the mode's pair index (kept in memory for B) ----
    pairs = _build_road_pairs(mode_config, cells, zones)
    with step(f'mode={mode_config.mode}: save node-keyed tiered OD pair index'):
        context.create_tiered_odm(
            pairs, network_name=mode_config.mode, data_name='node_pairs')

    # Free cells + zones — Section B doesn't touch them. `pairs` must
    # stay: each profile's `tiered_path_costs` consumes it.
    del cells, zones

    # ---- Section B: route each profile of this mode -----------------------
    for profile in mode_config.profiles:
        _route_road_profile(
            context, mode_config, mode_config.source_for(profile),
            pairs,
        )

    context.close()


if __name__ == '__main__':
    variants.run(main)
