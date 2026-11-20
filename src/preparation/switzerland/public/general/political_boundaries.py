"""
Build country / canton / municipality polygons from the swissTLMRegio boundaries.

Reads three layers from the swisstopo GeoPackage, dissolves the per-feature rows into
one multipolygon per named entity, harmonizes column names, and writes a `.gpkg` +
`.csv` pair for each level via `context.create_generic`. The layers also cover the
neighbouring countries' regions and communes: countries are kept (the Swiss one is
`Schweiz`, the first part of its multilingual name), cantons are Swiss only, and
municipalities Swiss and Liechtenstein only.

Inputs (PUBLIC, under raw/switzerland/swisstlmregio/):
    swissTLMRegio_BOUNDARIES_LV95.gpkg (layers swisstlmregio_landesgebiet,
        swisstlmregio_kantonsgebiet, swisstlmregio_hoheitsgebiet)

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
from aperta_atlas.context import Context, Storage, init_context

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
    'name':            'original_areas_count',
    'see_flaeche':     'area_ha_lake',
    'kantonsflaeche':  'area_ha_official',
    'bezirksflaeche':  'area_ha_official',
    'gem_flaeche':     'area_ha_official',
    'einwohnerzahl':   'population',
    'bfs_nummer':      'bfs_number',
}

# Aggregator per source column (only applies when the column is present in the layer).
_AGG_COLS: dict[str, str | Callable] = {
    'name':            'count',
    'bfs_nummer':      utils.most_common,
    'see_flaeche':     'sum',
    'kantonsflaeche':  'sum',
    'bezirksflaeche':  'sum',
    'gem_flaeche':     'sum',
    # Repeated on every part of a multi-part entity (areas are per part).
    'einwohnerzahl':   'max',
}
# Numeric source columns in which swisstopo marks missing values (lakes, Kommunanz
# parts) with a negative sentinel (-32768).
_SENTINEL_COLS = ('see_flaeche', 'kantonsflaeche', 'bezirksflaeche', 'gem_flaeche',
                  'einwohnerzahl')

_ID_COL_TO_NAME: dict[str, str] = {
    'municipality_id': 'municipalities',
    'country_id': 'countries',
    'canton_id': 'cantons',
}

_GPKG_PATH = 'switzerland/swisstlmregio/swissTLMRegio_BOUNDARIES_LV95.gpkg'


def _dissolve_and_rename(gdf: gpd.GeoDataFrame, id_column: str) -> gpd.GeoDataFrame:
    """Merge rows that share `id_column` into one multipolygon and rename columns to
    the output schema. Adds `area_m2_geometry` derived from the dissolved geometry.
    """
    gdf.geometry = shapely.force_2d(gdf.geometry)
    for col in _SENTINEL_COLS:
        if col in gdf.columns:
            gdf[col] = gdf[col].where(gdf[col] >= 0)
    agg = {k: v for k, v in _AGG_COLS.items() if k in gdf.columns}
    gdf = gdf.dissolve(by=id_column, aggfunc=agg)
    gdf = gdf.rename(columns={k: v for k, v in _RENAME_COLS.items() if k in gdf.columns})
    if 'population' in gdf.columns:
        gdf['population'] = gdf['population'].fillna(0)   # lakes, Kommunanz parts
    gdf['area_m2_geometry'] = np.round(gdf.geometry.area, 0)
    gdf.index.name = id_column
    return gdf


def load_swiss_boundaries(
    context: Context,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and dissolve country, canton, and municipality polygons."""
    gpkg_path = context.raw_path(Storage.PUBLIC, _GPKG_PATH)

    def read(layer: str) -> gpd.GeoDataFrame:
        # CRS conversion isn't strictly necessary (the layers are LV95 already) but it
        # makes the assumption explicit.
        return gpd.read_file(gpkg_path, layer=layer).to_crs(CRS_CH)

    countries = read('swisstlmregio_landesgebiet')
    countries['country_id'] = countries['name'].str.split('#').str[0]
    countries = _dissolve_and_rename(countries, 'country_id')

    cantons = read('swisstlmregio_kantonsgebiet')
    cantons = cantons[cantons['icc'] == 'CH'].copy()
    cantons['canton_id'] = cantons['name'].map(CANTON_NAME_TO_ID)
    cantons = _dissolve_and_rename(cantons, 'canton_id')

    municipalities = read('swisstlmregio_hoheitsgebiet')
    municipalities = municipalities[municipalities['icc'].isin(['CH', 'LI'])].copy()
    municipalities['municipality_id'] = municipalities['name']
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
