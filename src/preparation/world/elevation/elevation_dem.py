"""
Download a Copernicus GLO-30 DEM (30 m resolution) clipped to a
buffered AOI polygon.

Independent of the network preparation chain — produces a standalone
raster that downstream per-mode "attach elevation to network" steps
will sample at every node + edge midpoint. The heavy lifting (AWS tile
fetch + mosaic + clip + optional reproject) is
`aperta.geo_processing.fetch_copernicus_dem`.

The 1° × 1° AWS tiles are cached under `tiles/` next to the output so
re-runs are fast and adding a new area (whose buffered polygon overlaps
an existing area's coverage) reuses overlapping tiles automatically.
Re-runs are no-ops when `<out>.tif` already exists (delete it to force
re-fetch).

Output CRS is WGS84 (`EPSG:4326`), uniform across the
`preparation/world/` namespace. Per-project reprojection to a metric
CRS (e.g. LV95 for Swiss work) happens at project-stage attach.

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_dem` variant per area. Buffer is the widest per-mode
buffer for the area (covers wherever any per-mode network goes).

Inputs:
    (none — fetched from https://copernicus-dem-30m.s3.amazonaws.com/)

Outputs (PUBLIC, under preparation/world/elevation/):
    dem_<area_name>.tif                            # mosaicked + clipped GeoTIFF, WGS84
    tiles/Copernicus_DSM_COG_10_N..._E..._DEM.tif  # cached raw tiles

Requires the `topo` extras (rasterio + requests). Install via
`pip install -e '../aperta[topo]'` or `pip install 'aperta[topo]'`.

Run all variants sequentially (default):
    python -m preparation.world.elevation.elevation_dem
Single variant:
    python -m preparation.world.elevation.elevation_dem --variant switzerland_dem
    python -m preparation.world.elevation.elevation_dem --variant bern_dem
"""

import logging
import os

from aperta.geo_processing import fetch_copernicus_dem
from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.common import buffered_place_polygon


# Buffer = widest per-mode for the area; DEM covers wherever any
# per-mode network goes.
variants = Variants([('place', str), ('area_name', str), ('buffer', int)])
for area in AREAS.values():
    variants.add(
        name=f'{area.name}_dem',
        place=area.place, area_name=area.name,
        buffer=widest_buffer(area),
    )


def main(variant) -> None:
    context = init_context(variant)
    out_name = f'dem_{variant.area_name}'

    out_relative = f'{out_name}.tif'
    out_path = context.path_for(Storage.PUBLIC, out_relative)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Tile cache lives next to the output so re-runs (and any future
    # country variant whose buffered polygon overlaps Switzerland's)
    # reuse already-downloaded 1° tiles. `cleanup_tiles=False` keeps
    # them for reuse — the cache is ~few MB per tile and pays for
    # itself on the first re-run.
    cache_tile_dir = os.path.join(os.path.dirname(out_path), 'tiles')

    with step('buffered_place_polygon'):
        polygon_gdf = buffered_place_polygon(variant.place, variant.buffer)
        polygon = polygon_gdf.geometry.iloc[0]

    # `fetch_copernicus_dem` is a no-op when out_path already exists —
    # log that area explicitly so the run output is informative.
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logging.info(
            f"  → DEM already exists at {out_path} ({size_mb:,.1f} MB); "
            f"skipping fetch. Delete the file to force re-fetch.")
    else:
        with step('fetch_copernicus_dem (AWS tiles + mosaic + clip)'):
            fetch_copernicus_dem(
                polygon=polygon,
                out_path=out_path,
                polygon_crs='EPSG:4326',  # `buffered_place_polygon` returns WGS84
                target_crs=None,          # keep WGS84 (preparation/world/ convention)
                cache_tile_dir=cache_tile_dir,
                cleanup_tiles=False,
                verbose=True,
            )
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            logging.info(f"  → wrote DEM: {size_mb:,.1f} MB")

    # Register with the dependency tracker so downstream attach steps
    # see this as a real prepared dataset (size unknown for rasters —
    # the n field is None, matching `create_generic`'s .graphml area).
    context.register_created_data(out_relative, Storage.PUBLIC, None)
    context.close()


if __name__ == '__main__':
    variants.run(main)
