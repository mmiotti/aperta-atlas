"""
Simplify and consolidate MOBIS trajectory dumps produced by
`a_mobis_trajectories_fetch.py`.

> **Private data.** Consumes MOBIS Postgres dumps (Switzerland's
> continuous mobility tracking panel); not redistributable. Published
> as documentation of the method.

Reads each `mobis_all/postgres_dump_<i>.gpkg`, drops non-LineString geometries
(many trips come back as Points without a measured length), simplifies each route to
a target vertex count, then concatenates all pages and writes a single trajectories
table.

Inputs (under <DATA_DIR_PRIVATE>/preparation/.../mobis_all/):
    postgres_dump_<i>.gpkg              # produced by a_mobis_trajectories_fetch.py

Outputs:
    mobis_all/trajectories.npy          # PRIVATE — dict of {leg_id: route ndarray}
    mobis_all/trajectories.csv          # PRIVATE — per-leg metadata (length, sizes)

Run:
    python -m preparation.switzerland.private.surveys.a_mobis_trajectories_process
"""

import logging

import numpy as np
import pandas as pd
import shapely

from aperta import surveys
from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage

from preparation.switzerland.private.surveys.common import (
    TARGET_GEOMETRY_SIZE,
    MAXIMUM_GEOMETRY_SIZE,
)


# Number of paged dump files written by a_mobis_trajectories_fetch.
N_PAGES = 12


def _process_route(geometry: shapely.geometry.LineString, target_size: int,
                   max_size: int) -> dict:
    """Simplify one LineString and return the coordinates + size info."""
    simplified, new_size, size_ratio = surveys.simplify_geometry(
        geometry, target_size, max_size,
    )
    x, y = simplified.xy
    coords = np.vstack([x, y]).T
    return {'route': coords, 'geometry_size': new_size, 'geometry_size_ratio': size_ratio}


def process_page(context: Context, page: int, target_size: int, max_size: int) -> pd.DataFrame:
    gdf = context.get_generic(f'mobis_all/postgres_dump_{page}.gpkg')
    gdf.index = gdf.index.astype(int)
    is_linestring = gdf.geometry.apply(lambda g: isinstance(g, shapely.geometry.LineString))
    gdf['is_linestring'] = is_linestring.astype(int)
    gdf = gdf.rename(columns={'length': 'length_geodb'})
    df_routes = gdf.loc[is_linestring, 'geometry'].apply(
        lambda g: _process_route(g, target_size, max_size)
    )
    df_routes = pd.DataFrame(df_routes.to_list(), index=df_routes.index)
    return gdf[['length_geodb']].join(df_routes, how='left')


def main() -> None:
    context = init_context()
    pages = [process_page(context, i, TARGET_GEOMETRY_SIZE, MAXIMUM_GEOMETRY_SIZE)
             for i in range(N_PAGES)]
    routes = pd.concat(pages, axis=0)

    n_before = len(routes)
    routes = routes[~routes.index.duplicated(keep='first')]
    if len(routes) < n_before:
        logging.warning(
            f"{n_before - len(routes):,} duplicate trajectories removed "
            f"(of {n_before:,}).")

    context.create_generic(routes['route'].to_dict(), 'mobis_all/trajectories.npy')
    context.create_generic(routes.drop(columns='route'), 'mobis_all/trajectories.csv')
    context.close()


if __name__ == '__main__':
    main()
