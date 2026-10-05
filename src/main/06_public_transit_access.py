"""
Per-cell public-transit access + zone-deviation bonus for the atlas.

Per-cell access metrics (routed travel time, seconds):
  - `t_walk_to_transit_nearest`  walk to nearest 1 transit stop
  - `t_walk_to_transit_mean3`    mean walk to nearest 3 transit stops
  - `t_bike_to_train_nearest`    bike to nearest 1 heavy-rail stop

Each metric also gets a `<metric>_zone_dev` deviation column (negative =
better than zone mean). These per-cell deviations are then applied as a
bonus / penalty on top of NPVM's zone-to-zone PT times to sharpen the
zone-level PT skim back to cell resolution for accessibility.

Pipeline per metric: load 05's pre-routed costs → `reindex_by_geo_unit`
to cell-keyed → `lookup_dest_column_geo` on the POI count column
(clipped at 1 per cell to avoid multi-direction bus-stop double-count)
→ `accessibility.nearest_k` → weighted zone-mean deviation.

Inputs (PUBLIC, under <scenario>/):
    properties/cells_pois.csv                              # `mobility_transit`, `mobility_transit_train`
    properties/cells_population.csv + cells_snap.csv + shapes/cells.gpkg
    properties/zones_population.csv + zones_snap.csv + shapes/zones.gpkg
    odm/walk_node_pairs.npz + odm/walk_time_net_rwalk.npz  # from 05
    odm/bike_node_pairs.npz + odm/bike_time_net_rbike.npz  # from 05 (rbike profile)

Output (PUBLIC, under <scenario>/):
    properties/cells_transit_access.csv

Run:
    python -m main.06_public_transit_access --scenario <name>
"""

import logging
from collections.abc import Sequence
from typing import cast

import pandas as pd

from aperta import accessibility, od_pairs
from aperta.od_pairs import TieredODNodePairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from aperta.data_processing import weighted_group_mean
from scenarios import get_scenario


# (mode, profile, transit-col, output-label, ks) per analysis. Column
# must appear in `scenario.transit_stops_cols`; analyses whose column
# is absent get skipped with a warning at runtime. Add entries to
# extend (e.g. walk→train separately).
_TRANSIT_ANALYSES: tuple[tuple[str, str | None, str, str, tuple[int, ...]], ...] = (
    # (mode, profile, transit-col, output-label, ks)
    ('walk', 'rwalk', 'mobility_transit',       'transit', (1, 3)),
    ('bike', 'rbike', 'mobility_transit_train', 'train',   (1,)),
)


def _compute_nearest(
    context, cells: pd.DataFrame, zones: pd.DataFrame, *,
    mode_config, profile: str | None, weight_attr: str, weight_label: str,
    ks: Sequence[int | float],
) -> pd.DataFrame:
    """One mode × one profile × one destination-type pipeline. Returns the
    `nearest_k` result DataFrame (origin-cell-indexed, MultiIndex cols
    `(k, label)`).

    Reads pre-routed costs from `odm/<mode>_<source.cost_data_name>.npz`
    (built by 05) — no routing happens in this script.
    """
    mode = mode_config.mode
    radii = mode_config.radii
    source = mode_config.source_for(profile)
    node_col = f'node_id_{mode}'

    with step(f'mode={mode} profile={profile!r}: load node pair index + pre-routed costs'):
        pairs = context.get_tiered_odm(network_name=mode, data_name='node_pairs')
        costs = cast(TieredODNodePairs, context.get_tiered_odm(
            network_name=mode, data_name=source.cost_data_name))
        logging.info(
            f"  → pairs c2c={len(pairs.cells_to_cells or {}):,} "
            f"c2z={len(pairs.cells_to_zones or {}):,} "
            f"z2z={len(pairs.zones_to_zones or {}):,}")

    # `reindex_by_geo_unit` expects the per-mode snap under the generic
    # `node_id` name — remap on local copies so other modes still see
    # their own `node_id_<mode>` column intact.
    cells_m = cells.copy()
    zones_m = zones.copy()
    cells_m['node_id'] = cells_m[node_col]
    zones_m['node_id'] = zones_m[node_col]

    with step(f'mode={mode}: reindex node-keyed → geo-keyed (cell IDs)'):
        pairs_geo, costs_geo = od_pairs.reindex_by_geo_unit(
            pairs, costs, cells_m,
            cell_node_column='node_id',
            zones=zones_m, zone_node_column='node_id',
            r_cells=radii.r_cells, r_medium=radii.r_medium, r_zones=radii.r_zones,
        )
        # We passed a non-None `odm`, so `costs_geo` is guaranteed non-None.
        assert costs_geo is not None

    with step(f'mode={mode}: build destination weights ({weight_attr})'):
        # `lookup_dest_column_geo` needs the column on BOTH cells and zones —
        # zones get the sum of cell-level counts per zone (the per-zone
        # opportunity count is what z2z + c2z tiers consume).
        weights = od_pairs.lookup_dest_column_geo(
            weight_attr, pairs_geo, cells_m, zones=zones_m,
        )

    with step(f'mode={mode}: nearest_k (ks={ks}, label={weight_label!r})'):
        cell_to_zone = cells_m['zone_id'].to_dict()
        result = accessibility.nearest_k(
            costs_geo,
            {weight_label: weights},
            cell_to_zone,
            ks=list(ks),
        )
        for k in ks:
            col = result[(k, weight_label)]
            n_finite = int(col.notna().sum())
            median_s = float(col.median())
            logging.info(
                f"  → k={k}: {n_finite:,}/{len(col):,} cells finite; "
                f"median t = {median_s:.1f} s ({median_s/60:.1f} min)")
    return result


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    # Resolve which analyses actually run — an entry in `_TRANSIT_ANALYSES`
    # only fires if its column is in `scenario.transit_stops_cols`
    # (otherwise the col isn't guaranteed to exist in cells_pois.csv).
    declared_transit = set(scenario.transit_stops_cols)
    analyses = [(mode, profile, col, label, ks)
                for mode, profile, col, label, ks in _TRANSIT_ANALYSES
                if col in declared_transit]
    skipped = [f'{m}→{c}' for m, _, c, *_ in _TRANSIT_ANALYSES
               if c not in declared_transit]
    if skipped:
        logging.warning(
            f"skipping transit analyses (col not in scenario.transit_stops_cols): "
            f"{skipped}")
    if not analyses:
        raise ValueError(
            f"No transit analyses can run — none of _TRANSIT_ANALYSES's cols "
            f"({[c for _, _, c, *_ in _TRANSIT_ANALYSES]}) are in "
            f"scenario.transit_stops_cols ({tuple(scenario.transit_stops_cols)}).")
    active_cols = sorted({col for _, _, col, *_ in analyses})

    with step('load cells + zones + POIs (clip per-cell stop counts at 1)'):
        cells = context.get_properties(
            'cells', ['population', 'snap', 'pois'], add_shapes=True)
        zones = context.get_properties(
            'zones', ['population', 'snap'], add_shapes=True)
        # Clip per-cell counts at 1: N bus stops in one cell are usually
        # the same physical location (one per direction); treating each
        # as a separate opportunity inflates the k-near means. Then
        # aggregate cell flags → zone-level count (required for the c2z
        # + z2z tiers of `lookup_dest_column_geo`).
        for col in active_cols:
            cells[col] = cells[col].clip(upper=1)
            zones[col] = (
                cells.groupby('zone_id')[col].sum()
                .reindex(zones.index, fill_value=0)
            )
        parts = [f"{col} ≥ 1: {int(cells[col].sum()):,}" for col in active_cols]
        logging.info(
            f"  → {len(cells):,} cells, {len(zones):,} zones; " + '; '.join(parts))

    # Output column naming: `t_{mode}_to_{label}_{suffix}` with
    # `suffix = 'nearest'` for k=1 and `f'mean{k}'` for k>1. Matches
    # what 07's transit-access overheads expect.
    out_data: dict[str, pd.Series] = {}
    for mode, profile, col, label, ks in analyses:
        result = _compute_nearest(
            context, cells, zones,
            mode_config=scenario.mode_configs[mode],
            profile=profile,
            weight_attr=col,
            weight_label=label,
            ks=list(ks),
        )
        for k in ks:
            suffix = 'nearest' if k == 1 else f'mean{k}'
            out_data[f't_{mode}_to_{label}_{suffix}'] = result[(k, label)]

    # ---- Assemble per-cell output + zone-mean deviations --------------------
    with step('assemble per-cell output + combined_total-weighted zone deviations'):
        out = pd.DataFrame(out_data, index=cells.index)

        zone_id = cells['zone_id']
        weights = cells['combined_total']
        base_cols = list(out.columns)
        for col in base_cols:
            zone_mean = zone_id.map(
                weighted_group_mean(out[col], weights, zone_id))
            out[f'{col}_zone_dev'] = out[col] - zone_mean

        # Diagnostic: surfaces issues like "most zones have a single
        # finite cell → mostly 0 deviations".
        for col in base_cols:
            base = out[col]
            dev = out[f'{col}_zone_dev']
            n_total = len(base)
            n_base_finite = int(base.notna().sum())
            n_dev_finite = int(dev.notna().sum())
            dev_finite = dev[dev.notna()]
            n_dev_zero = int((dev_finite == 0).sum()) if len(dev_finite) else 0
            logging.info(
                f"  → {col}: "
                f"base finite={n_base_finite:,}/{n_total:,}; "
                f"dev finite={n_dev_finite:,} (zero={n_dev_zero:,}); "
                f"dev IQR=[{dev_finite.quantile(0.25):+.1f}, "
                f"{dev_finite.quantile(0.75):+.1f}] s"
                if len(dev_finite) else f"  → {col}: ALL NaN")

    with step('save properties/cells_transit_access.csv'):
        context.create_properties(out, data_name='transit_access')

    context.close()


if __name__ == '__main__':
    main()
