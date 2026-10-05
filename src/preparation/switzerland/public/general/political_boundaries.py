"""
Build country / canton / municipality polygons from the swissBOUNDARIES3D dataset.

Reads three layers from the swisstopo GDB, dissolves the per-feature rows into one
multipolygon per named entity, harmonizes column names, and writes a `.gpkg` +
`.csv` pair for each level via `context.create_generic`.

Inputs (under <DATA_DIR_PUBLIC>/raw/switzerland/swissboundaries3d/):
    swissBOUNDARIES3D_1_4_LV95_LN02.gdb (layers TLM_LANDESGEBIET, TLM_KANTONSGEBIET,
                                         TLM_HOHEITSGEBIET)

Outputs (PUBLIC, under preparation/switzerland/general/):
    {countries,cantons,municipalities}.gpkg
    {countries,cantons,municipalities}.csv

Run:
    python -m preparation.switzerland.public.general.political_boundaries
"""

import geopandas as gpd
import numpy as np
import shapely

from typing import Callable

from aperta_atlas import utils
from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage

from preparation.switzerland.common import CRS_CH


# Swiss cantons aren't tagged with their 2-letter abbreviation in the source data;
# this maps the German `NAME` column to the canonical canton_id.
CANTON_NAME_TO_ID: dict[str, str] = {
    'Graubünden':              'GR', 'Bern':                    'BE',
    'Valais':                  'VS', 'Vaud':                    'VD',
    'Ticino':                  'TI', 'St. Gallen':              'SG',
    'Zürich':                  'ZH', 'Fribourg':                'FR',
    'Luzern':                  'LU', 'Aargau':                  'AG',
    'Uri':                     'UR', 'Thurgau':                 'TG',
    'Schwyz':                  'SZ', 'Jura':                    'JU',
    'Neuchâtel':               'NE', 'Solothurn':               'SO',
    'Glarus':                  'GL', 'Basel-Landschaft':        'BL',
    'Obwalden':                'OW', 'Nidwalden':               'NW',
    'Genève':                  'GE', 'Schaffhausen':            'SH',
    'Appenzell Ausserrhoden':  'AR', 'Zug':                     'ZG',
    'Appenzell Innerrhoden':   'AI', 'Basel-Stadt':             'BS',
}

# Source column → output column. Per-level layers share most of these.
_RENAME_COLS: dict[str, str] = {
    'NAME':            'original_areas_count',
    'SEE_FLAECHE':     'area_ha_lake',
    'KANTONSFLAECHE':  'area_ha_official',
    'BEZIRKSFLAECHE':  'area_ha_official',
    'GEM_FLAECHE':     'area_ha_official',
    'EINWOHNERZAHL':   'population',
    'BFS_NUMMER':      'bfs_number',
}

# Aggregator per source column (only applies when the column is present in the layer).
_AGG_COLS: dict[str, str | Callable] = {
    'NAME':            'count',
    'BFS_NUMMER':      utils.most_common,
    'SEE_FLAECHE':     'sum',
    'KANTONSFLAECHE':  'sum',
    'BEZIRKSFLAECHE':  'sum',
    'GEM_FLAECHE':     'sum',
    'EINWOHNERZAHL':   'sum',
}

_ID_COL_TO_NAME: dict[str, str] = {
    'municipality_id': 'municipalities',
    'country_id': 'countries',
    'canton_id': 'cantons',
}

_GDB_PATH = 'switzerland/swissboundaries3d/swissBOUNDARIES3D_1_4_LV95_LN02.gdb'


def _dissolve_and_rename(gdf: gpd.GeoDataFrame, id_column: str) -> gpd.GeoDataFrame:
    """Merge rows that share `id_column` into one multipolygon and rename columns to
    the output schema. Adds `area_m2_geometry` derived from the dissolved geometry.
    """
    gdf.geometry = shapely.force_2d(gdf.geometry)
    agg = {k: v for k, v in _AGG_COLS.items() if k in gdf.columns}
    gdf = gdf.dissolve(by=id_column, aggfunc=agg)
    gdf = gdf.rename(columns={k: v for k, v in _RENAME_COLS.items() if k in gdf.columns})
    gdf['area_m2_geometry'] = np.round(gdf.geometry.area, 0)
    gdf.index.name = id_column
    return gdf


def load_swiss_boundaries(
    context: Context,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and dissolve country, canton, and municipality polygons."""
    gdb_path = context.raw_path(Storage.PUBLIC, _GDB_PATH)
    # CRS conversion isn't strictly necessary (these layers already carry LV95 x/y)
    # but it makes the assumption explicit and suppresses pyogrio warnings.
    crs = CRS_CH

    countries = gpd.read_file(gdb_path, layer='TLM_LANDESGEBIET').to_crs(crs)
    countries['country_id'] = countries['NAME']
    countries = _dissolve_and_rename(countries, 'country_id')

    cantons = gpd.read_file(gdb_path, layer='TLM_KANTONSGEBIET').to_crs(crs)
    cantons['canton_id'] = cantons['NAME'].map(CANTON_NAME_TO_ID)
    cantons = _dissolve_and_rename(cantons, 'canton_id')

    municipalities = gpd.read_file(gdb_path, layer='TLM_HOHEITSGEBIET').to_crs(crs)
    municipalities['municipality_id'] = municipalities['NAME']
    municipalities = _dissolve_and_rename(municipalities, 'municipality_id')

    return countries, cantons, municipalities


def main():
    context = init_context()
    countries, cantons, municipalities = load_swiss_boundaries(context)
    for gdf in (countries, cantons, municipalities):
        name = _ID_COL_TO_NAME[str(gdf.index.name)]
        context.create_generic(gdf, name + '.gpkg')
        context.create_generic(gdf.drop(columns='geometry'), name + '.csv')
    context.close()


if __name__ == '__main__':
    main()
