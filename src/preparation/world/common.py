"""
Shared helpers for `preparation/world/` scripts.

Every per-variant script in `preparation/world/{osm,elevation,land_use}/`
needs to build a buffered AOI polygon from the case-level `place` name —
the OSM-clip steps write it to `.poly` format for `osmium extract`;
the elevation and population scripts use the polygon directly to clip
their source rasters. Centralised here so there's one place to evolve
the polygon-construction logic.
"""

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union


def buffered_place_polygon(
    place: str | tuple[str, ...], buffer_m: int,
) -> gpd.GeoDataFrame:
    """Geocode one or more OSM-recognized places (country, canton, city, ...)
    via OSMnx and return a 1-row GeoDataFrame of their (unioned) polygon
    buffered by `buffer_m` meters, in EPSG:4326.

    `place` follows OSM nominatim conventions:
        'Switzerland', 'Canton of Bern, Switzerland', 'Zürich, Switzerland'.

    A tuple of place names is geocoded piecewise and unioned before
    buffering — used by multi-canton Area entries (e.g. the language-
    region cross-validation scenarios). The unioned geometry may be a
    MultiPolygon if the constituent places are geographically disjoint
    (e.g. TI is disjoint from the FR-speaking cantons).

    The buffer happens in a local UTM (chosen by `ox.projection.project_gdf`)
    so the radius is metrically meaningful; the result is reprojected back
    to WGS84 because OSM tooling (osmium polygon filter, OSMnx graph)
    all expect lat/lon.
    """
    if isinstance(place, str):
        place_gdf = ox.geocode_to_gdf(place)
        label = place
    else:
        parts = [ox.geocode_to_gdf(p) for p in place]
        unioned = unary_union([g.geometry.iloc[0] for g in parts])
        place_gdf = gpd.GeoDataFrame(
            {'geometry': [unioned]}, crs=parts[0].crs)
        label = ' + '.join(place)
    proj = ox.projection.project_gdf(place_gdf)
    proj['geometry'] = proj.geometry.buffer(buffer_m)
    out = proj.to_crs('EPSG:4326')
    out['place'] = label
    out['buffer_m'] = buffer_m
    return out


def write_poly_file(polygon_gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Write a 1-row GeoDataFrame's polygon to OSM `.poly` format
    (consumed by `osmium extract --polygon=...`).

    Format spec — one or more numbered rings, lon/lat coordinates, with
    sentinel `END` lines:

        polygon
        1
            lon  lat
            lon  lat
            ...
        END
        2
            lon  lat
            ...
        END
        END

    A MultiPolygon geometry emits one numbered ring per part (needed for
    disjoint multi-place AOIs, e.g. TI + the FR-speaking cantons).

    See https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format
    """
    geom = polygon_gdf.geometry.iloc[0]
    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
    else:
        parts = [geom]
    with open(output_path, 'w') as f:
        f.write('polygon\n')
        for i, part in enumerate(parts, start=1):
            f.write(f'{i}\n')
            for lon, lat in part.exterior.coords:
                f.write(f'  {lon:.6f}  {lat:.6f}\n')
            f.write('END\n')
        f.write('END\n')
