"""
Materialize cell-baked GROSS-TIME ODMs — one per (road mode, profile),
plus one for transit. 09b / 10 read these as-is (no further overhead).

**Road**: reindex 05's node-keyed NET-TIME ODMs to cell/zone keys, scale
by the fitted α (`route_time_alpha`) and bake the full per-cell road
overhead (const + density + snap-distance) at both endpoints.

**Transit**: build a cell-tier pair index by spatial proximity, fill
each pair's base cost from 05's z2z NPVM time via `cell_to_zone`, scale
by α, then bake the per-cell transit overhead. All cells in a zone share the
z2z base; per-cell overhead creates within-zone variation — the
correction that makes transit accessibility genuinely per-cell.

Inputs (scenario storage, under `<scenario>/`):
    odm/<mode>_node_pairs.npz                                  # from 05 (node-keyed tiered pair index)
    odm/<mode>_time_net_<profile>.npz                          # from 05 (per-profile node net-time)
    odm/transit_z2z_pairs.npz + transit_time_net_npvm.npz      # from 05 (z2z-only)
    properties/cells_*.csv + shapes/cells.gpkg + zones_*.csv + shapes/zones.gpkg
    properties/nodes_<mode>_extended.csv                       # from 02b — ROAD_OVERHEAD_DENSITY_COL per snap node
    coefs/<kind>/overheads_road.csv                            # from 07a — per-profile road coefs
    coefs/<kind>/overheads_transit.csv                         # from 07b — transit coefs

Outputs (scenario storage, under `<scenario>/`):
    odm/<mode>_geo_pairs.npz                                   # geo-keyed tiered pair index (road)
    odm/<mode>_time_gross_<profile>.npz                        # cell-baked gross-time ODM per road profile
                                                               # (parallels 05's `<mode>_time_net_<profile>.npz`)
    odm/transit_geo_pairs.npz                                  # geo-keyed tiered pair index (transit)
    odm/transit_time_gross_npvm.npz                            # cell-baked gross-time ODM for transit
                                                               # (parallels 05's `transit_time_net_npvm.npz`)

Run:
    python -m main.08a_gross_od_times --scenario <name>
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

from aperta import od_pairs, overhead
from aperta.od_pairs import TieredODGeoPairs, TieredODNodePairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from aperta.data_processing import weighted_group_mean

from main.common import (
    ROAD_OVERHEAD_DENSITY_COL,
    per_cell_road_overheads,
    per_cell_transit_overhead,
    resolve_road_overhead_column,
    route_time_alpha,
)
from mode_configs import TRANSIT_MODE_CONFIG
from scenarios import get_scenario


def _scale_tiered_costs_in_place(costs, factor: float) -> None:
    """Multiply every value in a TieredOD*Pairs by `factor`, in place.
    Used to apply the fitted α (`t_routed` slope) from overhead coefs to
    the routed ODM before adding per-cell overheads. `factor == 1.0`
    (constrained-α fits or missing coef) is a no-op."""
    if factor == 1.0:
        return
    for tier_name in ('cells_to_cells', 'cells_to_zones', 'zones_to_zones'):
        tier = getattr(costs, tier_name, None)
        if not tier:
            continue
        for k in tier:
            tier[k] = tier[k] * factor


def _z2z_lookup(context) -> dict[tuple, float]:
    """Flatten 05's transit z2z pair index + net cost into a single
    `{(orig_zone, dest_zone): cost_s}` dict for O(1) per-pair lookup."""
    idx = context.get_tiered_odm(network_name='transit', data_name='z2z_pairs')
    cost = context.get_tiered_odm(network_name='transit', data_name='time_net_npvm')
    z2z_idx = idx.zones_to_zones or {}
    z2z_cost = cost.zones_to_zones or {}
    lookup: dict[tuple, float] = {}
    for orig, dests in z2z_idx.items():
        vals = z2z_cost[orig]
        for i, dest in enumerate(dests):
            lookup[(orig, dest)] = float(vals[i])
    return lookup


def _fill_transit_base_costs(
    transit_pairs: TieredODGeoPairs,
    z2z: dict[tuple, float],
    cell_to_zone: dict,
) -> TieredODGeoPairs:
    """Fill each cell/zone-tier pair with its base transit cost, looked
    up in the z2z table via `cell_to_zone`. Missing pairs get NaN."""
    def _fill_tier(tier, orig_is_cell, dest_is_cell):
        if not tier:
            return None
        out: dict = {}
        for orig, dests in tier.items():
            o_zone = cell_to_zone[orig] if orig_is_cell else orig
            arr = np.empty(len(dests), dtype=np.float32)
            for i, dest in enumerate(dests):
                d_zone = cell_to_zone[dest] if dest_is_cell else dest
                arr[i] = z2z.get((o_zone, d_zone), np.nan)
            out[orig] = arr
        return out

    return TieredODGeoPairs(
        cells_to_cells=_fill_tier(
            transit_pairs.cells_to_cells, orig_is_cell=True, dest_is_cell=True),
        cells_to_zones=_fill_tier(
            transit_pairs.cells_to_zones, orig_is_cell=True, dest_is_cell=False),
        zones_to_zones=_fill_tier(
            transit_pairs.zones_to_zones, orig_is_cell=False, dest_is_cell=False),
    )


def _build_transit_gross(
    context, cells: pd.DataFrame, zones: pd.DataFrame, cell_to_zone: dict,
) -> None:
    """Materialize `odm/transit_time_gross_npvm.npz` — cell-tier transit ODM with
    the per-cell transit overhead baked in. See the module docstring."""
    radii = TRANSIT_MODE_CONFIG.radii

    with step('mode=transit: build cell-tier pair index (proximity)'):
        # Reuse car-graph snap for connectivity filter (transit zones
        # snap to car nodes upstream). Dest-side filter matches the
        # road loop: cells with valid car snap + valid zone + zone in
        # zones_m. Origins are restricted via `orig_cells=is_active`.
        cells_m = cells.copy()
        cells_m['node_id'] = cells_m['node_id_car']
        zones_m = zones.copy()
        zones_m['node_id'] = zones_m['node_id_car']
        zones_m = zones_m[zones_m['node_id'].notna()]
        keep_dest = (
            cells_m['node_id'].notna()
            & cells_m['zone_id'].notna()
            & cells_m['zone_id'].isin(zones_m.index)
        )
        cells_m = cells_m[keep_dest]
        transit_pairs_node = cast(TieredODNodePairs, od_pairs.get_pairs(
            cells_m,
            r_cells=radii.r_cells,
            node_column='node_id',
            zones=zones_m,
            r_zones=radii.r_zones,
            r_medium=radii.r_medium,
            orig_cells=cells_m['is_active'] == 1,
        ))
        # Reindex node → geo keys so `_fill_transit_base_costs` +
        # `add_geo_overheads` can look up by cell / zone id.
        transit_pairs, _ = od_pairs.reindex_by_geo_unit(
            transit_pairs_node, None, cells_m,
            cell_node_column='node_id',
            zones=zones_m,
            zone_node_column='node_id',
            r_cells=radii.r_cells,
            r_medium=radii.r_medium,
            r_zones=radii.r_zones,
        )
        n_c2c = sum(len(a) for a in (transit_pairs.cells_to_cells or {}).values())
        n_c2z = sum(len(a) for a in (transit_pairs.cells_to_zones or {}).values())
        n_z2z = sum(len(a) for a in (transit_pairs.zones_to_zones or {}).values())
        logging.info(f"  → pairs (cell/zone tiers): c2c={n_c2c:,}, c2z={n_c2z:,}, z2z={n_z2z:,}")

    with step('mode=transit: load z2z lookup + fill cell-tier base costs'):
        z2z = _z2z_lookup(context)
        base_costs = _fill_transit_base_costs(transit_pairs, z2z, cell_to_zone)

    # The minimum trip time applies to the door-to-door cost (10_accessibilities), as in the
    # overhead calibration (07b) and the utility model (09a), not to the NPVM base time.
    with step('mode=transit: scale by α + bake per-cell transit overhead'):
        transit_coefs = context.get_coefs('overheads_transit')
        alpha_transit = route_time_alpha(transit_coefs['transit'])
        _scale_tiered_costs_in_place(base_costs, alpha_transit)
        per_cell_ov = per_cell_transit_overhead(cells_m, transit_coefs)
        cb_weight = cells_m['combined_total']
        zone_ids_series = cells_m['zone_id']
        zone_mean_ov = weighted_group_mean(per_cell_ov, cb_weight, zone_ids_series)

        costs_gross = overhead.add_geo_overheads(
            base_costs, transit_pairs,
            origin_cell=per_cell_ov, dest_cell=per_cell_ov,
            origin_zone=zone_mean_ov, dest_zone=zone_mean_ov,
            cell_to_zone=cell_to_zone,
        )
        logging.info(
            f"  → α = {alpha_transit:.4f} scaled onto z2z base costs; "
            f"per-cell transit overhead: median = "
            f"{per_cell_ov.median():.1f} s, P95 abs = "
            f"{per_cell_ov.abs().quantile(0.95):.1f} s")

    with step('mode=transit: save cell-baked gross ODM + geo pair index'):
        context.create_tiered_odm(
            costs_gross, network_name='transit', data_name='time_gross_npvm')
        # Geo pair index persisted separately — 09b / 10 need it (dest
        # ids); the cost ODM's arrays hold values, not ids.
        context.create_tiered_odm(
            transit_pairs, network_name='transit', data_name='geo_pairs')


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    with step('load cells + zones + snap + population/employment weights'):
        # `transit_access` needed by `per_cell_transit_overhead` —
        # `overheads_transit` coefs reference its `*_zone_dev` columns.
        prop_cols = ['population', 'snap', 'employment']
        transit_cols = ['transit_access'] if scenario.transit_source else []
        cells = context.get_properties('cells', prop_cols + transit_cols, add_shapes=True)
        zones = context.get_properties('zones', prop_cols, add_shapes=True)
        # Keep the full cells frame: origin nodes are those of is_active
        # cells (Swiss cells snapped to every active mode; 05's
        # `orig_cells`), but destinations remain per-mode-snappable
        # including buffer cells, so cross-border destinations reach the
        # geo_pairs.
        active_cell_ids = set(cells.loc[cells['is_active'] == 1].index)
        logging.info(f"  → {len(cells):,} cells ({len(active_cell_ids):,} active origins), "
                     f"{len(zones):,} zones")

    with step('load overheads_road coefs'):
        coefs = context.get_coefs('overheads_road')
        logging.info(f"  → profiles in coefs: {list(coefs.columns)}")

    cell_to_zone = cells['zone_id'].to_dict()

    for case in scenario.mode_configs.values():
        mode = case.mode
        node_col = f'node_id_{mode}'

        with step(f'mode={mode}: load node pairs + node density → cells'):
            pairs_node = cast(TieredODNodePairs, context.get_tiered_odm(
                network_name=mode, data_name='node_pairs'))
            node_props = context.get_properties('nodes', f'{mode}_extended')
            # Dest-side filter matches 05's: cells with valid snap for THIS
            # mode + valid zone_id + zone present in zones_m. Buffer cells
            # (`is_aoi=False`) that snap successfully stay in as destinations
            # — origin nodes are those of is_active cells (05's `orig_cells`).
            zones_m = zones[zones[node_col].notna()].copy()
            keep_dest = (
                cells[node_col].notna()
                & cells['zone_id'].notna()
                & cells['zone_id'].isin(zones_m.index)
            )
            cells_m = cells[keep_dest].copy()
            n_dest_buffer = int((~cells_m.index.isin(active_cell_ids)).sum())
            logging.info(f"  → dest-eligible: {len(cells_m):,} cells "
                         f"({n_dest_buffer:,} buffer-only), {len(zones_m):,} zones")
            density_col_local = f'_density_{mode}'
            cells_m = cells_m.join(
                node_props[[ROAD_OVERHEAD_DENSITY_COL]].rename(
                    columns={ROAD_OVERHEAD_DENSITY_COL: density_col_local}),
                on=node_col)
            cells_m[density_col_local] = cells_m[density_col_local].fillna(0.0)
            logging.info(
                f"  → density {density_col_local}: "
                f"median {cells_m[density_col_local].median():.3f}, "
                f"P95 {cells_m[density_col_local].quantile(0.95):.3f}")

        with step(f'mode={mode}: reindex pairs (node → geo)'):
            pairs_geo, _ = od_pairs.reindex_by_geo_unit(
                pairs_node, None, cells_m,
                cell_node_column=node_col,
                zones=zones_m,
                zone_node_column=node_col,
                r_cells=case.radii.r_cells,
                r_medium=case.radii.r_medium,
                r_zones=case.radii.r_zones,
            )
            n_c2c = sum(len(a) for a in (pairs_geo.cells_to_cells or {}).values())
            n_c2z = sum(len(a) for a in (pairs_geo.cells_to_zones or {}).values())
            n_z2z = sum(len(a) for a in (pairs_geo.zones_to_zones or {}).values())
            logging.info(f"  → geo pairs: c2c={n_c2c:,}, c2z={n_c2z:,}, z2z={n_z2z:,}")

        with step(f'mode={mode}: save geo pair index'):
            # Consumers of geo-keyed ODMs (09b, 10) need this pair
            # index — cost ODM's arrays hold values, not dest ids.
            context.create_tiered_odm(
                pairs_geo, network_name=mode, data_name='geo_pairs')

        same_mode_labels = [case.source_for(p).name for p in case.profiles]

        for profile in case.profiles:
            src = case.source_for(profile)
            label = src.name

            with step(f'profile={label}: load net-time node ODM + reindex'):
                costs_node = cast(TieredODNodePairs, context.get_tiered_odm(
                    network_name=mode, data_name=src.cost_data_name))
                _, costs_geo = od_pairs.reindex_by_geo_unit(
                    pairs_node, costs_node, cells_m,
                    cell_node_column=node_col,
                    zones=zones_m,
                    zone_node_column=node_col,
                    r_cells=case.radii.r_cells,
                    r_medium=case.radii.r_medium,
                    r_zones=case.radii.r_zones,
                )
                assert costs_geo is not None

            # The minimum trip time applies to the door-to-door cost (10_accessibilities), as
            # in the overhead calibration (07a) and the utility model (09a), not to the routed
            # time.
            with step(f'profile={label}: scale by α + bake full road overhead'):
                costs_net = cast(TieredODGeoPairs, costs_geo)

                # Same-mode profile labels are passed so derived profiles
                # (walk_prm → rwalk, uncalibrated ebikes → rbike) inherit
                # the parent profile's coefs.
                overhead_label = resolve_road_overhead_column(
                    label, coefs.columns,
                    same_mode_profile_labels=same_mode_labels)
                if overhead_label != label:
                    logging.info(f"  → falling back to {overhead_label!r} for overhead {label!r}")
                alpha = route_time_alpha(coefs[overhead_label])
                _scale_tiered_costs_in_place(costs_net, alpha)
                orig_ov, dest_ov = per_cell_road_overheads(
                    cells_m, coefs[overhead_label],
                    density_col=density_col_local,
                    snap_dist_col=f'distance_{mode}',
                )
                cb_weight = cells_m['combined_total']
                zone_ids = cells_m['zone_id']
                orig_zone_ov = weighted_group_mean(orig_ov, cb_weight, zone_ids)
                dest_zone_ov = weighted_group_mean(dest_ov, cb_weight, zone_ids)

                costs_gross = overhead.add_geo_overheads(
                    costs_net, pairs_geo,
                    origin_cell=orig_ov,
                    dest_cell=dest_ov,
                    origin_zone=orig_zone_ov,
                    dest_zone=dest_zone_ov,
                    cell_to_zone=cell_to_zone,
                )
                logging.info(
                    f"  → α = {alpha:.4f} scaled onto routed costs; "
                    f"per-cell full overhead: origin median "
                    f"{orig_ov.median():.1f} s (P95 "
                    f"{orig_ov.quantile(0.95):.1f} s); dest median "
                    f"{dest_ov.median():.1f} s")

            with step(f'profile={label}: save cell-baked gross ODM'):
                context.create_tiered_odm(
                    costs_gross,
                    network_name=mode,
                    data_name=f'time_gross_{src.name}',
                )

    # ---- Transit: cell-tier ODM built from 05's z2z NPVM lookup -----
    if scenario.transit_source is not None:
        _build_transit_gross(context, cells, zones, cell_to_zone)

    context.close()


if __name__ == '__main__':
    main()
