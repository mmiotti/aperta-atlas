"""
Attach fixed areas and the density around them to the standardized MTMC households (home) and
trips (origin).

> **Private data.** Reads the standardized MZMV tables (BFS-restricted) and the LUMOS hectares
> (aperta-lumos; proprietary inputs).

Every year's points are joined to the same, current polygons, so a place gets the same areas
in every survey year: today's municipality (state 2023-01-01, as `municipalities.gpkg`; number
and name), postcode (swisstopo directory) and canton (number and abbreviation, e.g. `ZH`), the
BFS typology of areas with urban character 2020 (all seven classes), the size class of the
agglomeration (2020) and the language region. `pool` combines the language region and the urban
character (e.g. `fr_core_city`) for pooled time series by settlement type. Homes in 1994 and 2000
are BFS postcode points, so there the finer areas are only as good as a postcode. Points just
outside a polygon (borders, lake shores) take the nearest one within 1 km; trip origins abroad
get no areas. The log checks the join against the survey's own municipality numbers and counts
target persons per pool.

Density around each point, from the LUMOS hectares of the closest LUMOS year (1994 → 1995,
2021 → 2020): residents and FTE (full-time equivalent jobs) within 1 km and 3 km, per hectare
of
  occupied  hectares with residents or FTE in that year;
  mask      hectares of the LUMOS settlement mask (the same in every year);
  gross     all land (hectares inside a municipality polygon: no large lakes, nothing abroad).
A hectare counts when its centre is within the radius of the centre of the point's hectare.
Empty for points located only to a postcode or coarser (all of 1994 and 2000), as a circle
around such a point is not the home's surroundings. Densities of whole municipalities and
postcodes: `b_mzmv_area_densities.py`.

Inputs:
    PRIVATE preparation/switzerland/surveys/mzmv_<year>/standardized/
        {households,people,trips}.csv              # from a_mzmv_standardize.py
    PRIVATE lumos/<LUMOS year>/properties/cells_population.csv   # aperta-lumos
    PUBLIC  preparation/switzerland/general/municipalities.gpkg
    PUBLIC  raw/switzerland/bfs_spatial_levels/communes_levels_2023-01-01.csv
    PUBLIC  raw/switzerland/postcodes/ortschaftenverzeichnis_plz_2056/AMTOVZ_SHP_LV95/
        AMTOVZ_ZIP.shp

Outputs:
    mzmv_<year>/standardized/household_context.csv   # PRIVATE — one row per household (hh_id)
    mzmv_<year>/standardized/trip_orig_context.csv   # PRIVATE — one row per trip (trip_id)

Run all years sequentially (default):
    python -m preparation.switzerland.private.surveys.b_mzmv_context
Single year:
    python -m preparation.switzerland.private.surveys.b_mzmv_context --variant 2015
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.signal import fftconvolve

from aperta_atlas.context import Context, Storage, init_context

from preparation.switzerland.common import CRS_CH
from preparation.switzerland.private.surveys.a_mzmv_standardize import variants

BFS_LEVELS = 'switzerland/bfs_spatial_levels/communes_levels_2023-01-01.csv'
POSTCODES = ('switzerland/postcodes/ortschaftenverzeichnis_plz_2056/AMTOVZ_SHP_LV95/'
             'AMTOVZ_ZIP.shp')
SNAP_M = 1000
LUMOS_YEARS = tuple(range(1990, 2030, 5))
RADII_M = (1000, 3000)
MEASURES = ('residents', 'fte')
DENOMINATORS = ('occupied', 'mask', 'gross')
COARSE = ('postcode', 'municipality', 'region', 'country')     # geocode levels: no radius

# RSC2020: Raum mit städtischem Charakter 2020
URBAN_CHARACTER = {
    1: 'core_city',             # Agglomerationskerngemeinde (Kernstadt)
    2: 'main_core',             # Agglomerationskerngemeinde (Hauptkern)
    3: 'secondary_core',        # Agglomerationskerngemeinde (Nebenkern)
    4: 'belt',                  # Agglomerationsgürtelgemeinde
    5: 'multi_oriented',        # Mehrfach orientierte Gemeinde
    6: 'rural_centre',          # Kerngemeinde ausserhalb Agglomerationen
    0: 'rural',                 # Ländliche Gemeinde ohne städtischen Charakter
}
# AGGLGK2020: Agglomerationsgrössenklasse (inhabitants of the agglomeration)
AGGLOMERATION_SIZE = {
    1: '500k_plus',             # Zürich, Genève, Basel
    2: '250k_500k',             # Bern, Lausanne
    3: '100k_250k',
    4: '50k_100k',
    5: 'under_50k',
    0: 'outside',               # not in an agglomeration
}
# SPRGEB2020: Sprachgebiete
LANGUAGE_REGION = {1: 'de', 2: 'fr', 3: 'it', 4: 'rm'}
# BFS canton number → official abbreviation
CANTON = dict(enumerate(('ZH', 'BE', 'LU', 'UR', 'SZ', 'OW', 'NW', 'GL', 'ZG', 'FR', 'SO', 'BS',
                         'BL', 'SH', 'AR', 'AI', 'SG', 'GR', 'AG', 'TG', 'TI', 'VD', 'VS', 'NE',
                         'GE', 'JU'), start=1))


def load_areas(context: Context) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(BFS levels by municipality number, municipality polygons, postcode polygons)."""
    levels = pd.read_csv(context.raw_path(Storage.PUBLIC, BFS_LEVELS), index_col='BfsCode')
    munis = (context.source('preparation/switzerland/general', storage=Storage.PUBLIC)
             .get_generic('municipalities.gpkg'))
    # swissBOUNDARIES3D numbers Greifensee like its lake; lakes, Liechtenstein and the
    # foreign enclaves have no BFS row and are left out, so points there snap to the shore.
    munis['bfs_number'] = munis['bfs_number'].replace({9040: 194})
    munis = munis[munis['bfs_number'].isin(levels.index)][['bfs_number', 'geometry']]
    pc = gpd.read_file(context.raw_path(Storage.PUBLIC, POSTCODES))
    pc = pc.set_crs(CRS_CH) if pc.crs is None else pc.to_crs(CRS_CH)   # the shp has no .prj
    postcodes = pc.dissolve('ZIP4').reset_index()[['ZIP4', 'geometry']]
    return levels, munis, postcodes


def areas(x: pd.Series, y: pd.Series, levels: pd.DataFrame, munis: gpd.GeoDataFrame,
          postcodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Areas of LV95 points; empty where a point has no coordinates or lies abroad."""
    ok = x.notna() & y.notna()
    pts = gpd.GeoDataFrame(index=x.index[ok], geometry=gpd.points_from_xy(x[ok], y[ok]),
                           crs=CRS_CH)
    muni = assign(pts, munis).reindex(x.index).astype('Int64')
    canton_no = muni.map(levels['CantonId']).astype('Int64')
    out = pd.DataFrame({
        'municipality': muni,
        'municipality_name': muni.map(levels['Name']),
        'postcode': pd.to_numeric(assign(pts, postcodes)).reindex(x.index).astype('Int64'),
        'canton_no': canton_no,
        'canton': canton_no.map(CANTON),
        'urban_character': muni.map(levels['RSC2020']).map(URBAN_CHARACTER),
        'agglomeration_size': muni.map(levels['AGGLGK2020']).map(AGGLOMERATION_SIZE),
        'language_region': muni.map(levels['SPRGEB2020']).map(LANGUAGE_REGION),
    }, index=x.index)
    out['pool'] = out['language_region'] + '_' + out['urban_character']
    return out


def assign(points: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> pd.Series:
    """Polygon id per point: the one it lies in, else the nearest within `SNAP_M`."""
    key = polys.columns[0]
    hit = gpd.sjoin(points, polys, predicate='within', how='left')
    hit = hit[~hit.index.duplicated()][key]
    miss = hit.isna()
    if miss.any():
        near = gpd.sjoin_nearest(points[miss], polys, max_distance=SNAP_M, how='left')
        hit[miss] = near[~near.index.duplicated()][key]
    return hit


def lumos_year(year: int) -> int:
    """The LUMOS year closest to a survey year (1994 → 1995, 2021 → 2020)."""
    return min(LUMOS_YEARS, key=lambda y: abs(y - year))


def lumos_cells(context: Context, year: int) -> pd.DataFrame:
    """LUMOS hectares of `year` (the settlement mask: the same cells in every year), indexed by
    cell_id `<x>:<y>`: `residents`, `fte` (full-time equivalent jobs, all three sectors) and the
    hectare's centre `x`, `y` (LV95)."""
    cells = (context.source(f'lumos/{year}', storage=Storage.PRIVATE)
             .get_properties('cells', 'population', allow_cache=False))
    xy = cells.index.to_series().str.split(':', expand=True).astype(int)
    return pd.DataFrame({'residents': cells['population'],
                         'fte': cells['combined_total'] - cells['population'],
                         'x': xy[0], 'y': xy[1]})


class Hectares:
    """The 100 m grid over Switzerland (LV95) as arrays, row 0 at the north edge, with the
    municipality and postcode of every land hectare by its centre (0: no land, i.e. large lakes
    and abroad). A point belongs to the hectare it lies in."""

    def __init__(self, munis: gpd.GeoDataFrame, postcodes: gpd.GeoDataFrame):
        pad = max(RADII_M) // 100 + 1
        x0, y0, x1, y1 = munis.total_bounds
        self.col0, self.top = int(x0 // 100) - pad, int(y1 // 100) + pad
        self.shape = (self.top - int(y0 // 100) + pad + 1, int(x1 // 100) + pad - self.col0 + 1)
        transform = from_origin(self.col0 * 100, (self.top + 1) * 100, 100, 100)

        def burn(polys: gpd.GeoDataFrame, key: str) -> np.ndarray:
            return rasterize(zip(polys.geometry, polys[key].astype(int)), out_shape=self.shape,
                             transform=transform, dtype='int32')

        self.municipality = burn(munis, 'bfs_number')
        self.land = self.municipality > 0
        self.postcode = np.where(self.land, burn(postcodes, 'ZIP4'), 0)

    def index(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row and column of the hectares of LV95 points on the grid, and which points are."""
        r = self.top - np.floor(np.asarray(y, float) / 100)
        c = np.floor(np.asarray(x, float) / 100) - self.col0
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])     # NaN: False
        return r[ok].astype(int), c[ok].astype(int), ok

    def grid(self, cells: pd.DataFrame, values) -> np.ndarray:
        """`values` (per LUMOS cell, or one for all) on the grid; 0 elsewhere."""
        r, c, ok = self.index(cells['x'], cells['y'])
        out = np.zeros(self.shape)
        out[r, c] = np.broadcast_to(np.asarray(values, float), ok.shape)[ok]
        return out


def within(grid: np.ndarray, radius_m: int) -> np.ndarray:
    """Per hectare, the sum over the hectares whose centre is within `radius_m` of its centre."""
    r = radius_m // 100
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    out = fftconvolve(grid, (dx ** 2 + dy ** 2 <= r ** 2).astype(float), mode='same')
    return np.maximum(out, 0)                                       # FFT round-off around 0


def radius_densities(hect: Hectares, cells: pd.DataFrame, x: pd.Series, y: pd.Series
                     ) -> pd.DataFrame:
    """`<measure>_per_ha_<denominator>_<radius>km` around LV95 points (see the module
    docstring); empty where a point has no coordinates or the circle no such hectares."""
    r, c, ok = hect.index(x, y)
    grids = {'residents': hect.grid(cells, cells['residents']),
             'fte': hect.grid(cells, cells['fte']),
             'occupied': hect.grid(cells, cells['residents'] + cells['fte'] > 0),
             'mask': hect.grid(cells, 1.0),
             'gross': hect.land.astype(float)}
    out = pd.DataFrame(index=x.index)
    for radius in RADII_M:
        at = {k: within(g, radius)[r, c] for k, g in grids.items()}
        for m in MEASURES:
            for d in DENOMINATORS:
                den = np.rint(at[d])
                vals = np.where(den > 0, at[m] / np.where(den > 0, den, 1), np.nan)
                out[f'{m}_per_ha_{d}_{radius // 1000}km'] = (
                    pd.Series(vals, index=x.index[ok]).round(2))
    return out


def _log_join(name: str, out: pd.DataFrame, survey_municipality: pd.Series,
              levels: pd.DataFrame) -> None:
    """How many points got areas, and how often the joined municipality is the survey's own
    number (where that number still exists; mergers since then differ by design)."""
    still = survey_municipality.isin(levels.index) & out['municipality'].notna()
    same = (out.loc[still, 'municipality'] == survey_municipality[still]).mean()
    logging.info(f"  {name}: {out['municipality'].notna().mean():.1%} located; municipality = "
                 f"the survey's own for {same:.1%} of those whose number still exists")


def _log_densities(name: str, dens: pd.DataFrame) -> None:
    col = dens['residents_per_ha_occupied_1km']
    logging.info(f"  {name}: densities for {col.notna().mean():.1%}; residents per occupied "
                 f"hectare within 1 km, median {col.median():.1f}")


def main(variant) -> None:
    context = init_context(variant)
    year = variant.mzmv_year
    base = f'mzmv_{year}/standardized'
    levels, munis, postcodes = load_areas(context)
    hect = Hectares(munis, postcodes)
    cells = lumos_cells(context, lumos_year(year))
    logging.info(f"  Densities from LUMOS {lumos_year(year)}")

    hh = context.get_generic(f'{base}/households.csv')
    hh_context = areas(hh['home_x'], hh['home_y'], levels, munis, postcodes)
    _log_join('Homes', hh_context, hh['home_municipality'], levels)
    exact = ~hh['home_geocode_level'].isin(COARSE)
    dens = radius_densities(hect, cells, hh['home_x'].where(exact), hh['home_y'].where(exact))
    _log_densities('Homes', dens)
    people = context.get_generic(f'{base}/people.csv')
    per_pool = people['hh_id'].map(hh_context['pool']).value_counts()
    logging.info(f"  Target persons per pool (language × urban character): "
                 f"{per_pool.sort_index().to_dict()}")
    context.create_generic(hh_context.join(dens), f'{base}/household_context.csv')

    trips = context.get_generic(f'{base}/trips.csv')
    abroad = trips['orig_in_switzerland'] == 0
    x, y = trips['orig_x'].mask(abroad), trips['orig_y'].mask(abroad)
    trip_context = areas(x, y, levels, munis, postcodes)
    _log_join('Trip origins', trip_context, trips['orig_municipality'], levels)
    exact = ~trips['orig_geocode_level'].isin(COARSE)
    dens = radius_densities(hect, cells, x.where(exact), y.where(exact))
    _log_densities('Trip origins', dens)
    context.create_generic(trip_context.join(dens), f'{base}/trip_orig_context.csv')
    context.close()


if __name__ == '__main__':
    variants.run(main)
