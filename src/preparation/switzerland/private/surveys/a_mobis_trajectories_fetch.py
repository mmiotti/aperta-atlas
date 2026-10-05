"""
Download GPS trajectories from the MOBIS Postgres server.

> **Private data.** Requires MOBIS Postgres server credentials
> (Switzerland's continuous mobility tracking panel); not
> redistributable. Published as documentation of the method.

Fetches the `motion_tag_trips` table in 1M-row pages and writes each page as its own
GeoPackage under `mobis_all/postgres_dump_<i>.gpkg`. Subsequent runs of
`a_mobis_trajectories_process.py` concatenate and simplify them.

Requires `POSTGRES_PASSWORD` to be set in `.env` (read into `context.env`).

Inputs:
    (network — `motion_tag_trips` table on id-hdb-psgr-cp50.ethz.ch:5432)

Outputs:
    mobis_all/postgres_dump_<i>.gpkg    # PRIVATE — one file per 1M-row page

Run:
    python -m preparation.switzerland.private.surveys.a_mobis_trajectories_fetch

NOTE: not yet fully updated / tested
"""

import logging

import geopandas as gpd
import pandas as pd
import psycopg
import shapely

from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage

from preparation.switzerland.common import CRS_LATLON


PAGE_SIZE = 1_000_000

_QUERY = """
    SELECT mt_trip_id, geometry, length, merge_count, merged_into_id,
           misdetected_completely, excluded
    FROM motion_tag_trips
    LIMIT %s OFFSET %s
"""

_COLUMNS = ('id', 'geometry', 'length', 'merge_count', 'merged_into_id',
            'misdetected_completed', 'excluded')


def _record_to_row(record: tuple) -> tuple:
    """Normalize one DB row: parse hex-WKB geometry, coerce nullable numerics to -1."""
    return (
        record[0],
        shapely.wkb.loads(record[1], hex=True),
        float(record[2]) if record[2] is not None else -1.0,
        int(record[3])   if record[3] is not None else -1,
        int(record[4])   if record[4] is not None else -1,
        record[5],
        record[6],
    )


def collect(context: Context) -> None:
    connect_str = (
        f"host=id-hdb-psgr-cp50.ethz.ch port=5432 dbname=mobis_study user=guest "
        f"password={context.env['POSTGRES_PASSWORD']}"
    )
    with psycopg.connect(connect_str) as conn, conn.cursor() as cursor:
        page = 0
        while True:
            cursor.execute(_QUERY, (PAGE_SIZE, PAGE_SIZE * page))
            records = cursor.fetchall()
            if not records:
                break
            rows = [_record_to_row(r) for r in records]
            df = pd.DataFrame.from_records(rows, columns=_COLUMNS)
            gdf = gpd.GeoDataFrame(
                df, geometry='geometry', crs=CRS_LATLON,
            ).set_index('id')
            n_before = len(gdf)
            gdf = gdf[~gdf.index.duplicated(keep='first')]
            if len(gdf) < n_before:
                logging.info(f"Page {page}: dropped {n_before - len(gdf):,} duplicates.")
            context.create_generic(gdf, f'mobis_all/postgres_dump_{page}.gpkg')
            page += 1


def main() -> None:
    context = init_context()
    collect(context)
    context.close()


if __name__ == '__main__':
    main()
