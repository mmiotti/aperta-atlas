"""
Consolidate per-mode snap outputs into the merged `cells_snap.csv` /
`zones_snap.csv` that the rest of the pipeline reads, and derive the
`is_active` flag (`is_aoi AND snapped-to-all-scenario-modes`).

Scenario-level (no variants — single execution per scenario). Runs once
after all `02a_networks_snap` per-mode invocations have completed.

`is_active` is the canonical "should this cell/zone get accessibility
computed?" predicate. Consumers filter with
`cells = cells[cells['is_active']]` near the top of `main()` — avoids
sparse-snap cells getting walk accessibility but no car accessibility
(a silent inconsistency in cross-mode comparisons).

Inputs:
    properties/cells_snap_<mode>.csv     # from 02a, per mode
    properties/zones_snap_<mode>.csv     # from 02a, per mode
    shapes/cells.gpkg                    # for is_aoi column
    shapes/zones.gpkg                    # for is_aoi column

Outputs (under <scenario>/, PUBLIC):
    properties/cells_snap.csv            # node_id_<mode> + distance_<mode>
                                         # per scenario.mode_configs + is_active
    properties/zones_snap.csv            # same shape, zones

Run:
    python -m main.02c_active_flags --scenario <name>
"""

import logging

import geopandas as gpd
import pandas as pd

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta.errors import DataError

from scenarios import get_scenario


def _build_merged_snap(
    context, geo_type: str, modes: tuple[str, ...], is_aoi: pd.Series,
) -> pd.DataFrame:
    """Stitch per-mode `<geo_type>_<mode>_snap.csv` files into one
    merged frame with `node_id_<mode>` column, then AND `is_aoi` with
    all-modes-snap-success to derive `is_active`."""
    merged: pd.DataFrame | None = None
    for mode in modes:
        per_mode = context.get_properties(geo_type, f'snap_{mode}')
        if merged is None:
            merged = pd.DataFrame(index=per_mode.index)
        elif not merged.index.equals(per_mode.index):
            raise DataError(f"{geo_type}_snap_{mode}.csv index doesn't match prior index.")
        merged[f'node_id_{mode}'] = per_mode['node_id']
        merged[f'distance_{mode}'] = per_mode['distance']
        snapped = per_mode['node_id'].notna()
        logging.info(
            f"  {geo_type}/{mode}: {int(snapped.sum()):,} of {len(per_mode):,} "
            f"snapped ({100*snapped.mean():.1f}%)")
    assert merged is not None

    # `is_active` = in AOI AND snapped to every mode the scenario uses.
    all_modes_snapped = pd.Series(True, index=merged.index)
    for mode in modes:
        all_modes_snapped &= merged[f'node_id_{mode}'].notna()
    in_aoi = is_aoi.reindex(merged.index).fillna(False).astype(bool)
    merged['is_active'] = (in_aoi & all_modes_snapped).astype(int)
    n_active = int(merged['is_active'].sum())
    n_aoi = int(in_aoi.sum())
    logging.info(
        f"  {geo_type}: is_active = {n_active:,} of {len(merged):,} "
        f"({100*n_active/len(merged):.1f}%); "
        f"AOI ∩ all-mode-snap; AOI alone = {n_aoi:,}, dropped "
        f"{n_aoi - n_active:,} AOI cells for missing a snap")
    return merged


def main() -> None:
    context = init_context()
    scenario = get_scenario(context.scenario)
    modes = tuple(scenario.mode_configs)

    with step(f'load is_aoi (cells + zones) — scenario modes: {modes}'):
        cells_aoi = gpd.read_file(
            context.path_for(context.default_storage, 'shapes/cells.gpkg'),
        ).set_index('cell_id')['is_aoi']
        zones_aoi = gpd.read_file(
            context.path_for(context.default_storage, 'shapes/zones.gpkg'),
        ).set_index('zone_id')['is_aoi']

    with step('build merged cells_snap.csv (per-mode snap + is_active)'):
        cells_snap = _build_merged_snap(context, 'cells', modes, cells_aoi)
        context.create_properties(cells_snap, data_name='snap')

    with step('build merged zones_snap.csv'):
        zones_snap = _build_merged_snap(context, 'zones', modes, zones_aoi)
        context.create_properties(zones_snap, data_name='snap')

    context.close()


if __name__ == '__main__':
    main()
