"""
Standardize the raw MTMC (MZMV) files of every survey year onto one cross-year schema.

> **Private data.** Requires the raw MZMV deliveries (Switzerland's federal mobility survey,
> BFS-restricted access); not redistributable. Published as documentation of the method.

Decodes one survey year's raw tables with that year's codebook (`mzmv/codebook_<year>.py`)
and writes households, household members, people (target persons), vehicles, trips, legs,
day and overnight journeys and the journeys' other places with the same English-coded
columns for every year (`mzmv/schema.py`); columns a year did not survey are empty. On the
way: LV95 coordinates for all years (WGS84 from 2010, LV03 in 2005, BFS postcode points in
1994/2000), person / trip / leg ids that hold for households with two target persons (up to
2010), and each trip's main mode recomputed from its legs, by the BFS ranking and by longest
leg. Raw codes the codebook does not list are logged and set to empty — check the log after a
year's first run.

Which year has what, and notes on the columns: `mzmv/README.md`.

`b_mzmv_process.py` turns the 2015 and 2021 legs into the atlas's survey legs.

Inputs (PRIVATE, under raw/switzerland/mzmv/):
    MZ1994/ASCII(csv)/Dateien_csv/{hh,hp,zp,au,mo,wg,et}.txt
    MZ2000/ASCII(csv)/Dateien/{hh,hp,zp,au,mo,wg,et}.txt
    MZ2000/Codes/Schweiz/CH_Ortschaften.xls                 # postcode points (1994, 2000)
    MZ2005_geo/4_DB_ASCII(Sep_TAB)/Dateien/
        {haushalte,haushaltspersonen,zielpersonen,autos_motos,wege,etappen,
         reisen_1a,reisen_1b}.dat
    MZMV2010_geo/4_DB_csv/db_csv/
        {haushalte,haushaltspersonen,zielpersonen,fahrzeuge,wege,etappen,reisen1a,reisen1b}.csv
    MZMV{2015,2021}_mit_Geo/4_DB_csv/
        {haushalte,haushaltspersonen,zielpersonen,fahrzeuge,wege,etappen,tagesreisen,
         reisenmueb}.csv

Outputs (PRIVATE, under preparation/switzerland/surveys/):
    mzmv_<year>/standardized/
        {households,household_members,people,vehicles,trips,legs,
         day_journeys,overnight_journeys,journey_places}.csv

Run all years sequentially (default):
    python -m preparation.switzerland.private.surveys.a_mzmv_standardize
Single year:
    python -m preparation.switzerland.private.surveys.a_mzmv_standardize --variant 1994
"""

import logging

from aperta_atlas.context import Context, init_context
from aperta_atlas.variant import Variants

from preparation.switzerland.private.surveys.mzmv import (
    codebook_1994, codebook_2000, codebook_2005, codebook_2010, codebook_2015, codebook_2021,
    engine, geo,
)

CODEBOOKS = {cb.YEAR: cb for cb in (codebook_1994, codebook_2000, codebook_2005,
                                    codebook_2010, codebook_2015, codebook_2021)}

variants = Variants([('mzmv_year', int)])
for _y in CODEBOOKS:
    variants.add(name=str(_y), mzmv_year=_y)


def standardize_year(context: Context, year: int) -> None:
    codebook = CODEBOOKS[year]
    logging.info(f"Standardizing MTMC {year}")
    raw = engine.load_raw(context, codebook)
    points = geo.load_points(context) if codebook.XY_CRS is None else None
    for name, table in engine.standardize(codebook, raw, points).items():
        context.create_generic(table, f'mzmv_{year}/standardized/{name}.csv')


def main(variant) -> None:
    context = init_context(variant)
    standardize_year(context, variant.mzmv_year)
    context.close()


if __name__ == '__main__':
    variants.run(main)
