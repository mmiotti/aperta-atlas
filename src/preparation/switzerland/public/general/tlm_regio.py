"""
Extract water-body polygons from the lower-resolution swissTLMRegio dataset.

Used as a mask for cutting out lake areas from other shapefiles (e.g. traffic zones)
so they're visually intuitive. Lakes ≥ 10,000 m² are present in the source; flowing
and stagnant water layers are lines rather than polygons and are not included.

Inputs (PUBLIC, under raw/switzerland/swisstlmregio/):
    swissTLMRegio_Product_LV95.gpkg  (layer tlmregio_hydrography_lake)

Outputs (PUBLIC, under preparation/switzerland/general/):
    water_bodies.gpkg

Run:
    python -m preparation.switzerland.public.general.tlm_regio
"""

import geopandas as gpd
import shapely

from aperta_atlas.context import Context, Storage, init_context

from preparation.switzerland.common import CRS_CH


# Simplification tolerance for lake polygons (m). Lakes are used as cut-masks for
# other geometries; small tolerances keep the masks visually faithful without
# inflating file sizes.
SIMPLIFY_TOLERANCE_M = 2


def get_water_bodies(context: Context) -> gpd.GeoDataFrame:
    """Load lake polygons from swissTLMRegio, simplified to `SIMPLIFY_TOLERANCE_M`."""
    gpkg_path = context.raw_path(
        Storage.PUBLIC, 'switzerland/swisstlmregio/swissTLMRegio_Product_LV95.gpkg',
    )
    water_bodies = gpd.read_file(gpkg_path, layer='tlmregio_hydrography_lake').to_crs(CRS_CH)
    water_bodies.geometry = shapely.force_2d(
        water_bodies.geometry.simplify(SIMPLIFY_TOLERANCE_M))
    water_bodies.index.name = 'water_body_id'
    return water_bodies


def main():
    context = init_context()
    context.create_generic(get_water_bodies(context), 'water_bodies.gpkg')
    context.close()


if __name__ == '__main__':
    main()
