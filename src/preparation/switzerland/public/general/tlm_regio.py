"""
Extract water-body polygons from the lower-resolution swissTLMRegio dataset.

Used as a mask for cutting out lake areas from other shapefiles (e.g. traffic zones)
so they're visually intuitive. Lakes ≥ 10,000 m² are present in the source; flowing
and stagnant water layers are lines rather than polygons and are not included.

Inputs (under <DATA_DIR_PUBLIC>/raw/switzerland/swisstlmregio/):
    swissTLMRegio_Produkt_LV95.gdb  (layer TLMRegio_Lake)

Outputs (PUBLIC):
    shapes/water_bodies.gpkg

Run:
    python -m preparation.switzerland.public.general.tlm_regio
"""

import geopandas as gpd

from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage

from preparation.switzerland.common import CRS_CH


# Simplification tolerance for lake polygons (m). Lakes are used as cut-masks for
# other geometries; small tolerances keep the masks visually faithful without
# inflating file sizes.
SIMPLIFY_TOLERANCE_M = 2


def get_water_bodies(context: Context) -> gpd.GeoDataFrame:
    """Load lake polygons from swissTLMRegio, simplified to `SIMPLIFY_TOLERANCE_M`."""
    gdb_path = context.raw_path(
        Storage.PUBLIC, 'switzerland/swisstlmregio/swissTLMRegio_Produkt_LV95.gdb',
    )
    water_bodies = gpd.read_file(gdb_path, layer='TLMRegio_Lake').to_crs(CRS_CH)
    water_bodies.geometry = water_bodies.geometry.simplify(SIMPLIFY_TOLERANCE_M)
    water_bodies.index.name = 'water_body_id'
    return water_bodies


def main():
    context = init_context()
    context.create_generic(get_water_bodies(context), 'water_bodies.gpkg')
    context.close()


if __name__ == '__main__':
    main()
