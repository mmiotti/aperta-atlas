"""
Density of every municipality and postcode in every LUMOS year, for pooling MTMC persons by
area and for changes over time.

> **Private data.** Reads the LUMOS hectares (aperta-lumos; proprietary inputs). Only area
> totals leave this script, no hectares.

The areas are those of `b_mzmv_context.py` (municipalities as of 2023-01-01, today's
postcodes), so an area has the same extent in every year. A LUMOS hectare belongs to the area
containing its centre, else the nearest within 1 km (lake shores, borders). Per area and year:
`residents`, `fte` (full-time equivalent jobs, all three sectors), the hectares
`ha_<denominator>` and `<measure>_per_ha_<denominator>`, with as denominator
  occupied  hectares with residents or FTE in that year;
  mask      hectares of the LUMOS settlement mask (the same in every year);
  gross     all land of the area (hectares by centre; large lakes belong to no area).
Same definitions as the radius densities of homes and trip origins. LUMOS years 1990–2025 in
5-year steps; the FTE have a series break between 2005 and 2010 (business census → STATENT)
and 1990 reuses the 1995 FTE.

Inputs:
    PRIVATE lumos/<year>/properties/cells_population.csv     # aperta-lumos, 1990 ... 2025
    PUBLIC  preparation/switzerland/general/municipalities.gpkg
    PUBLIC  raw/switzerland/bfs_spatial_levels/communes_levels_2023-01-01.csv
    PUBLIC  raw/switzerland/postcodes/ortschaftenverzeichnis_plz_2056/AMTOVZ_SHP_LV95/
        AMTOVZ_ZIP.shp

Outputs:
    mzmv_areas/municipality_density.csv   # PRIVATE — one row per municipality and LUMOS year
    mzmv_areas/postcode_density.csv       # PRIVATE — one row per postcode and LUMOS year

Run:
    python -m preparation.switzerland.private.surveys.b_mzmv_area_densities
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta_atlas.context import init_context

from preparation.switzerland.common import CRS_CH
from preparation.switzerland.private.surveys.b_mzmv_context import (
    DENOMINATORS, LUMOS_YEARS, MEASURES, Hectares, assign, load_areas, lumos_cells)


def area_table(cells_by_year: dict[int, pd.DataFrame], area: pd.Series, gross: pd.Series,
               name: str) -> pd.DataFrame:
    """Totals and densities per area (`area`: per LUMOS cell; `gross`: land hectares per area,
    which also lists the areas) and year."""
    tables = []
    for year, cells in cells_by_year.items():
        t = pd.DataFrame({'residents': cells['residents'], 'fte': cells['fte'],
                          'ha_occupied': cells['residents'] + cells['fte'] > 0, 'ha_mask': 1})
        t = t.groupby(area).sum().reindex(gross.index, fill_value=0)
        t['ha_gross'] = gross
        for m in MEASURES:
            for d in DENOMINATORS:
                t[f'{m}_per_ha_{d}'] = (t[m] / t[f'ha_{d}'].where(t[f'ha_{d}'] > 0)).round(2)
        tables.append(t.assign(year=year))
    out = pd.concat(tables).rename_axis(name).set_index('year', append=True).sort_index()
    out[list(MEASURES)] = out[list(MEASURES)].round(1)
    return out


def main() -> None:
    context = init_context()
    levels, munis, postcodes = load_areas(context)
    hect = Hectares(munis, postcodes)
    cells_by_year = {y: lumos_cells(context, y) for y in LUMOS_YEARS}
    first = cells_by_year[LUMOS_YEARS[0]]
    for y, cells in cells_by_year.items():
        logging.info(f"  LUMOS {y}: {cells['residents'].sum() / 1e6:.2f} M residents, "
                     f"{cells['fte'].sum() / 1e6:.2f} M FTE, {len(cells):,} hectares")
        if not cells.index.equals(first.index):
            logging.warning(f"  LUMOS {y}: other cells than {LUMOS_YEARS[0]}; aligned to those")
            cells_by_year[y] = cells.reindex(first.index).fillna({'residents': 0, 'fte': 0})
    pts = gpd.GeoDataFrame(index=first.index, crs=CRS_CH,
                           geometry=gpd.points_from_xy(first['x'], first['y']))

    for name, polys, ids in (('municipality', munis, hect.municipality),
                             ('postcode', postcodes, hect.postcode)):
        area = pd.to_numeric(assign(pts, polys))
        gross = pd.Series(np.bincount(ids.ravel()))
        gross = gross.reindex(polys.iloc[:, 0].astype(int).unique(), fill_value=0)
        last = cells_by_year[LUMOS_YEARS[-1]]
        logging.info(f"  {name}: {gross.size:,} areas ({(gross == 0).sum()} without a land "
                     f"hectare); LUMOS hectares in one: {area.notna().mean():.2%}, with "
                     f"{last.loc[area.notna(), 'residents'].sum() / last['residents'].sum():.2%}"
                     f" of the residents ({LUMOS_YEARS[-1]})")
        out = area_table(cells_by_year, area, gross, name)
        if name == 'municipality':
            out.insert(0, 'municipality_name', out.index.get_level_values(0).map(levels['Name']))
        context.create_generic(out, f'mzmv_areas/{name}_density.csv')
    context.close()


if __name__ == '__main__':
    main()
