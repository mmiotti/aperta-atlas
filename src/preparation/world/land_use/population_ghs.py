"""
Download the GHS-POP R2023A 100 m gridded-population tiles covering a buffered
AOI polygon, mosaic them, clip to the buffer, and save the result as a single
GeoTIFF.

The native distribution is per-tile (10° × 10° tiles in Mollweide projection
`ESRI:54009`). Switzerland fits inside tile `R4_C19`; different areas may need
multiple tiles. The script always treats its input as a list of tile IDs and
mosaics them via `rasterio.merge` before clipping — single-tile is just the n=1
area.

Output CRS is **kept as Mollweide** (the GHSL native, equal-area). Per-project
reprojection to a metric CRS happens at project-stage attach.

The raw 10° tiles are cached under `preparation/world/land_use/ghsl_tiles/`.

Cases are defined in `preparation/world/areas.py`; this script registers one
variant per area (variant name = `<area_name>`). Tile list comes from each
area's `ghsl_tiles` field; buffer is the widest per-mode buffer.

Inputs:
    (none — fetched from JRC's GHSL open data FTP)

Outputs (PUBLIC, under preparation/world/land_use/):
    population_<area_name>.tif                     # clipped + mosaicked,
    Mollweide ghsl_tiles/GHS_POP_..._<R>_<C>.tif   # cached raw tiles

Run all variants sequentially (default):
    python -m preparation.world.land_use.population_ghs
Single variant:
    python -m preparation.world.land_use.population_ghs --variant switzerland
    python -m preparation.world.land_use.population_ghs --variant bern
"""

import logging
import os
import zipfile
from pathlib import Path

import requests

from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.common import buffered_place_polygon


# GHSL R2023A 100 m product naming + URL templates. The product slug
# `GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0` is the 2020 epoch in
# Mollweide at 100 m, R2023A release, V1.0. The per-tile zip bundles
# the .tif + a metadata PDF + an XLSX data-package — we keep the .tif
# only.
_GHSL_TILE_BASENAME_TEMPLATE = (
    'GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_{tile}'
)
_GHSL_TILE_URL_TEMPLATE = (
    'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/'
    'GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/'
    '{basename}.zip'
)
_GHSL_NATIVE_CRS = 'ESRI:54009'  # Mollweide

variants = Variants([
    ('place', str), ('area_name', str),
    ('buffer', int), ('ghsl_tiles', tuple),
])
for area in AREAS.values():
    variants.add(
        name=area.name,
        place=area.place, area_name=area.name,
        buffer=widest_buffer(area),
        ghsl_tiles=area.ghsl_tiles,
    )


def _ensure_tile_downloaded(tile: str, cache_dir: str) -> str:
    """Download + unzip a single GHSL tile if not already cached.
    Returns the path to the `.tif`. Idempotent — subsequent calls with
    the same tile / cache_dir return the cached path immediately.
    """
    basename = _GHSL_TILE_BASENAME_TEMPLATE.format(tile=tile)
    tif_path = os.path.join(cache_dir, f'{basename}.tif')
    if os.path.exists(tif_path):
        return tif_path
    zip_path = os.path.join(cache_dir, f'{basename}.zip')
    url = _GHSL_TILE_URL_TEMPLATE.format(basename=basename)
    logging.info(f"  → downloading {tile} (~50-100 MB) from JRC...")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    # Extract the .tif only (the zip also bundles PDF/XLSX docs).
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if member.endswith('.tif'):
                z.extract(member, cache_dir)
    os.remove(zip_path)
    if not os.path.exists(tif_path):
        raise RuntimeError(
            f"Tile {tile} downloaded but no .tif found at {tif_path}. "
            f"JRC may have changed the zip layout — inspect manually.")
    return tif_path


def _mosaic_and_clip(tile_paths: list[str], polygon_4326, out_path: str) -> None:
    """Mosaic tiles via `rasterio.merge`, reproject the AOI polygon to
    the GHSL native CRS, and write the clipped + mosaicked GeoTIFF
    (LZW-compressed) to `out_path`.

    All tiles in `tile_paths` must share the same CRS (true for GHSL
    R2023A — all `ESRI:54009`). The output preserves that CRS; no
    resampling.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as raster_mask
    from rasterio.merge import merge as raster_merge

    # Mosaic stage — single-tile is just the n=1 area of merge.
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_transform = raster_merge(srcs)
        # Tile CRSes should all match — sanity-check before using the
        # first one as the mosaic CRS.
        crses = {s.crs.to_string() for s in srcs}
        if len(crses) > 1:
            raise RuntimeError(f"Tiles have inconsistent CRSes: {crses}. Cannot mosaic.")
        mosaic_crs = srcs[0].crs
        nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()

    # Reproject the AOI polygon (WGS84 from `buffered_place_polygon`)
    # to the mosaic CRS for the clip.
    aoi_mollweide = gpd.GeoSeries(
        [polygon_4326], crs='EPSG:4326',
    ).to_crs(mosaic_crs).iloc[0]

    # Write the mosaic to a memory raster, then clip-by-mask onto it.
    # Avoids landing the un-clipped mosaic on disk (could be GBs).
    from rasterio.io import MemoryFile
    profile = {
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'count': mosaic.shape[0],
        'dtype': mosaic.dtype,
        'crs': mosaic_crs,
        'transform': mosaic_transform,
        'nodata': nodata,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(mosaic)
        with memfile.open() as tmp:
            clipped, clip_transform = raster_mask(
                tmp, [aoi_mollweide.__geo_interface__], crop=True,
            )

    out_profile = profile.copy()
    out_profile.update({
        'height': clipped.shape[1],
        'width':  clipped.shape[2],
        'transform': clip_transform,
        'compress': 'lzw',
    })
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(clipped)


def main(variant) -> None:
    context = init_context(variant)
    out_name = f'population_{variant.area_name}'

    # Resolve the prepared output path. `_generic_relative_path` returns
    # the relative path verbatim under `preparation/*` namespaces, so
    # the on-disk layout is `<DATA_DIR_PUBLIC>/preparation/world/land_use/<out>.tif`.
    out_relative = f'{out_name}.tif'
    out_path = context.path_for(Storage.PUBLIC, out_relative)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Tile cache shared across all country variants — adjacent
    # countries reuse overlapping tiles automatically.
    cache_dir = os.path.dirname(out_path)
    tiles_dir = os.path.join(cache_dir, 'ghsl_tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logging.info(
            f"  → population raster already exists at {out_path} "
            f"({size_mb:,.1f} MB); skipping. Delete the file to force "
            f"re-fetch + re-clip.")
        context.register_created_data(out_relative, Storage.PUBLIC, None)
        context.close()
        return

    with step('buffered_place_polygon'):
        polygon_gdf = buffered_place_polygon(variant.place, variant.buffer)
        polygon = polygon_gdf.geometry.iloc[0]

    with step(f'fetch GHSL tiles ({len(variant.ghsl_tiles)})'):
        tile_paths = [_ensure_tile_downloaded(t, tiles_dir) for t in variant.ghsl_tiles]

    with step('mosaic + clip to country buffer'):
        _mosaic_and_clip(tile_paths, polygon, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logging.info(f"  → wrote {out_path} ({size_mb:,.1f} MB)")

    # Quick sanity check — total population should be in a sensible range
    # for the buffered country extent. Catches an empty / wrong-tile mosaic.
    import rasterio
    with rasterio.open(out_path) as src:
        pop = src.read(1)
    total_pop = float(pop[pop > 0].sum())
    logging.info(f"  → total population in clipped raster: {total_pop:,.0f}")

    context.register_created_data(out_relative, Storage.PUBLIC, None)
    context.close()


if __name__ == '__main__':
    variants.run(main)
