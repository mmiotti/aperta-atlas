"""
Materialize a single, mode-agnostic straight-line distance ODM
(`dist_line`, metres) built purely from cell + zone centroids and the
tier radii — no dependency on any mode's network or 08a's per-mode
geo_pairs. Straight-line distance from cell A to cell B is the same
regardless of mode, so one file is enough.

Tier construction mirrors aperta's `od_pairs.get_pairs`: pairs are
classified by zone-pair distance into cells_to_cells / cells_to_zones /
zones_to_zones using `_UNIFIED_RADII` from mode_configs.py. Every pair
within its tier radius carries `hypot(x_o - x_d, y_o - y_d)` between
centroids in `scenario.crs_main`. "Line" (straight-line / crow-flies)
contrasts with network distance, which this pipeline doesn't compute.

Inputs (PUBLIC, under `<scenario>/`):
    shapes/cells.gpkg + properties/cells_snap.csv           # for zone_id
    shapes/zones.gpkg

Outputs (PUBLIC, under `<scenario>/`):
    odm/dist_line_geo_pairs.npz     # mode-agnostic pair index (dest ids per origin)
    odm/dist_line_dist_line.npz     # straight-line distances aligned to those pairs

Run:
    python -m main.08b_dist_line_od --scenario <name>
"""

import logging

import numpy as np
import pandas as pd

from aperta import od_pairs
from aperta.od_pairs import TieredODGeoPairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from mode_configs import _UNIFIED_RADII


# Column added to cells + zones so `get_pairs` keys pairs by the geo
# unit's own ID (rather than a snap-node ID). Same name in both frames
# — `get_pairs` takes a single `node_column` argument.
_UNIT_COL = 'unit_id'


def _as_geo_pairs(node_pairs) -> TieredODGeoPairs:
    """Rewrap a `TieredODNodePairs` (returned by `get_pairs`) as
    `TieredODGeoPairs`. Structurally identical; the distinction is
    semantic (keys are geo unit IDs here, not network nodes)."""
    return TieredODGeoPairs(
        cells_to_cells=node_pairs.cells_to_cells,
        cells_to_zones=node_pairs.cells_to_zones,
        zones_to_zones=node_pairs.zones_to_zones,
    )


def main():
    context = init_context()
    radii = _UNIFIED_RADII

    with step('load cells + zones (shapes + zone_id + is_active)'):
        cells = context.get_properties('cells', 'snap', add_shapes=True)
        zones = context.get_shapes('zones')
        # Keep the full cells frame so buffer (is_aoi=False) cells appear
        # as destinations — straight-line distance is defined for every
        # cell regardless of mode snap. Origins are restricted to
        # is_active via `orig_cells` mask on get_pairs below. 10 filters
        # output rows to is_active cells at its final reindex.
        cells[_UNIT_COL] = cells.index
        zones[_UNIT_COL] = zones.index
        n_active = int((cells['is_active'] == 1).sum())
        logging.info(f"  → {len(cells):,} cells ({n_active:,} active origins), "
                     f"{len(zones):,} zones")

    with step(f'build dist_line pair index (r_cells={radii.r_cells:.0f} m, '
              f'r_medium={radii.r_medium:.0f} m, r_zones={radii.r_zones:.0f} m)'):
        pairs = _as_geo_pairs(od_pairs.get_pairs(
            cells, r_cells=radii.r_cells, node_column=_UNIT_COL,
            zones=zones, r_zones=radii.r_zones, r_medium=radii.r_medium,
            orig_cells=(cells['is_active'] == 1),
        ))
        n_pairs = sum(
            sum(len(v) for v in (tier or {}).values())
            for tier in (pairs.cells_to_cells,
                         pairs.cells_to_zones,
                         pairs.zones_to_zones))
        logging.info(f"  → {n_pairs:,} pairs across three tiers")

    with step('compute straight-line distances'):
        # Combined GeoDataFrame covering every ID that can appear in
        # pairs (both cells and zones). `get_euclidean_dists` uses its
        # geometry column as the xy source.
        centroids = pd.concat([
            cells.assign(geometry=cells.geometry.centroid)[['geometry']],
            zones.assign(geometry=zones.geometry.centroid)[['geometry']],
        ])
        dists = _as_geo_pairs(od_pairs.get_euclidean_dists(centroids, pairs))
        median_m = float(np.nanmedian(np.concatenate([
            np.concatenate(list(t.values()))
            for t in (dists.cells_to_cells,
                      dists.cells_to_zones,
                      dists.zones_to_zones)
            if t
        ])))
        logging.info(f"  → median pair distance: {median_m:.0f} m")

    with step('save dist_line ODM'):
        context.create_tiered_odm(pairs, network_name='dist_line', data_name='geo_pairs')
        context.create_tiered_odm(dists, network_name='dist_line', data_name='dist_line')

    context.close()


if __name__ == '__main__':
    main()
