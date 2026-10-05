"""
Per-building population for any area, via dasymetric mapping of the
GHS-POP 100 m gridded raster (from `population_ghs.py`) onto OSM
building footprints.

Public, reproducible path for population — complements the restricted-
access Swiss STATPOP orchestrator (`preparation/switzerland/land_use/
population_statpop.py`). Lower fidelity than STATPOP (GHS is modelled,
STATPOP is observed; GHS lacks age breakdowns) but global coverage and
zero data-access restrictions. For Swiss-only analyses with STATPOP
access, prefer STATPOP. For cross-border buffer regions or non-Swiss
cases, this is the path that works.

Method:
  1. Load the area's clipped GHS-POP raster (Mollweide, 100 m).
  2. Convert every non-zero pixel into a 100 m square polygon in the
     raster CRS, carrying the pixel value as `population_total`.
  3. Reproject buildings to the raster CRS so overlap areas are
     metrically meaningful (Mollweide is equal-area).
  4. `dasymetric.learn_coefficients` to derive per-OSM-tag intensities
     from the GHS-POP cells, then `dasymetric.per_building` for the
     per-cell-rescaled allocation (with nearest-building fallback for
     rural pixels no building overlaps).

Single `population_total` output column — GHS-POP has no demographic
breakdown.

Inputs (PUBLIC, cross-source):
    preparation/world/osm/shapes/buildings_<area_name>.gpkg
        # from preparation/world/osm/buildings_from_pbf.py
    preparation/world/land_use/population_<area_name>.tif
        # from preparation/world/land_use/population_ghs.py

Outputs (PUBLIC, under preparation/world/land_use/):
    properties/buildings_population_<area_name>.csv
        # indexed by building_id (OSM way ID); single column
        # `population_total`. Join with shapes/buildings_<area_name>.gpkg.

Run all variants sequentially (default):
    python -m preparation.world.land_use.population_per_building_from_ghs
Single variant:
    python -m preparation.world.land_use.population_per_building_from_ghs \\
        --variant switzerland
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta_atlas import dasymetric
from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS


# Nearest-building fallback distance (m). 200 m is generous for sparse
# rural pixels (the only realistic area for unmatched GHS-POP cells at
# 100 m resolution); GHS-POP cells in dense areas always overlap at
# least one building.
_NEAREST_FALLBACK_MAX_M = 200.0


variants = Variants([('area_name', str)])
for area in AREAS.values():
    variants.add(
        name=area.name,
        area_name=area.name,
    )


def _raster_to_cells(raster_path: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Read a GHS-POP raster + return (cells_gdf, cell_totals).

    Each non-zero pixel becomes a single polygon (the pixel's footprint
    in the raster's CRS) with `cell_id` (sequential int) and `geometry`.
    `cell_totals` carries `population_total` indexed by `cell_id`.
    Zero-population pixels are dropped (no contribution to dasymetric).
    """
    import rasterio
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Mask: pixels with positive population (and not no-data).
    if nodata is not None:
        mask = (data != nodata) & (data > 0)
    else:
        mask = data > 0
    rows, cols = np.where(mask)
    values = data[rows, cols].astype(float)

    if len(rows) == 0:
        raise ValueError(
            f"No non-zero pixels in {raster_path}. Check the GHS-POP "
            f"raster was generated for this area.")

    # Pixel footprints in raster CRS. `transform * (col, row)` gives
    # the upper-left corner of pixel (row, col); the pixel extends one
    # cell-size in each direction.
    polys = []
    for r, c in zip(rows, cols):
        x_ul, y_ul = transform * (c, r)
        x_lr, y_lr = transform * (c + 1, r + 1)
        # min/max because transform may flip Y axis (y_ul > y_lr).
        polys.append(box(
            min(x_ul, x_lr), min(y_ul, y_lr),
            max(x_ul, x_lr), max(y_ul, y_lr),
        ))

    cells = gpd.GeoDataFrame(
        {'cell_id': np.arange(len(polys))},
        geometry=polys, crs=crs,
    )
    cell_totals = pd.DataFrame(
        {'population_total': values},
        index=pd.Index(np.arange(len(polys)), name='cell_id'),
    )
    return cells, cell_totals


def main(variant) -> None:
    context = init_context(variant)

    with step('load OSM building shapes'):
        buildings_ctx = context.source('preparation/world/osm')
        buildings = buildings_ctx.get_shapes('buildings', data_name=variant.area_name)
        logging.info(f"  → {len(buildings):,} buildings loaded")

    with step('load GHS-POP raster + extract non-zero pixels'):
        # `population_ghs.py` writes its output to
        # `preparation/world/land_use/population_<area_name>.tif`.
        raster_path = context.path_for(Storage.PUBLIC, f'population_{variant.area_name}.tif')
        cells, cell_totals = _raster_to_cells(raster_path)
        logging.info(
            f"  → {len(cells):,} non-zero pixels; "
            f"total population: {cell_totals['population_total'].sum():,.0f}")

    with step(f'reproject buildings to raster CRS ({cells.crs})'):
        # Mollweide is equal-area, so `_overlap_area` in the dasymetric
        # overlay is metrically meaningful for cell rescaling.
        buildings = buildings.to_crs(cells.crs)

    with step('learn NNLS intensities from GHS-POP cells'):
        learned = dasymetric.learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='population_total',
            relevant_tags=dasymetric.POPULATION_TAGS,
            nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
        )

    with step('dasymetric mapping (population_total)'):
        tag_intensities = {
            t: float(i) for t, i in
            learned['intensity_population_total'].dropna().items()
        }
        coeffs = {
            t: tag_intensities.get(t, 1.0) for t in dasymetric.POPULATION_TAGS
        }
        out = dasymetric.per_building(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='population_total',
            coeffs=coeffs,
            nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
        )
        logging.info(
            f"  → distributed {out['population_total'].sum():,.0f} total "
            f"residents across {len(out):,} buildings "
            f"({(out['population_total'] > 0).sum():,} with non-zero)")

    data_name = f'population_{variant.area_name}'
    props = out[['population_total']].copy()
    context.create_properties(props, data_name=data_name)
    context.create_coefs(learned, f'population_ghs_intensities_{variant.area_name}')
    context.close()


if __name__ == '__main__':
    variants.run(main)
