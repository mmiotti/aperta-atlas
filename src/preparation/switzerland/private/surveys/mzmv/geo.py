"""
Locations for the standardized MTMC tables: reprojection to LV95 and, for the years without
coordinates (1994, 2000), points from postcodes and municipalities.

The postcode points come with the MTMC 2000 delivery (`CH_Ortschaften.xls`, BFS state
1 January 2000, LV03): one reference point per postcode. A municipality's point is the mean of
the points of its postcodes. Both match the 1994/2000 survey era better than today's postcode
polygons, which lack retired postcodes. Note the BFS licence remark in the delivery: using these
coordinates needs a GEOSTAT licence beyond the MTMC one.
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer

from aperta.errors import DataError
from aperta_atlas.context import Context, Storage

from preparation.switzerland.common import CRS_CH

CRS_LV03 = 'EPSG:21781'
POSTCODE_FILE = 'switzerland/mzmv/MZ2000/Codes/Schweiz/CH_Ortschaften.xls'


def to_lv95(x: pd.Series, y: pd.Series, crs: str) -> tuple[pd.Series, pd.Series]:
    """Reproject `x`/`y` (lon/lat for WGS84, easting/northing otherwise) to LV95, rounded to
    metres. For LV03 the axis order is detected from the values, since the Swiss X/Y naming is
    the reverse of easting/northing and the deliveries are not consistent about it."""
    if crs == CRS_LV03:
        x, y = _lv03_easting_northing(x, y)
    tx = Transformer.from_crs(crs, CRS_CH, always_xy=True)
    e, n = tx.transform(x.to_numpy(dtype=float), y.to_numpy(dtype=float))
    bad = ~(np.isfinite(e) & np.isfinite(n))
    e, n = np.round(np.where(bad, np.nan, e)), np.round(np.where(bad, np.nan, n))
    return pd.Series(e, index=x.index), pd.Series(n, index=y.index)


def _lv03_easting_northing(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (easting, northing): LV03 eastings are ~480-840 km, northings ~60-300 km."""
    med_a, med_b = a.median(), b.median()
    if not (pd.notna(med_a) and pd.notna(med_b)):
        return a, b
    if med_a < med_b:
        logging.info(f"  LV03: {a.name}/{b.name} hold northing/easting — swapping")
        a, b = b, a
    if not (400_000 < a.median() < 900_000 and 50_000 < b.median() < 350_000):
        raise DataError(
            f"{a.name}/{b.name} do not look like LV03 metres "
            f"(medians {a.median():.0f}, {b.median():.0f}).")
    return a, b


def load_points(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(postcode points, municipality points), each with LV95 `x`, `y`, indexed by postcode
    resp. BFS municipality number (state 1 January 2000)."""
    path = context.raw_path(Storage.PRIVATE, POSTCODE_FILE)
    loc = pd.read_excel(path, engine='calamine')
    loc['x'], loc['y'] = to_lv95(loc['PlzXCoord'], loc['PlzYCoord'], CRS_LV03)
    postcodes = loc.drop_duplicates('Plz').set_index('Plz')[['x', 'y']]
    municipalities = (loc.drop_duplicates(['Gemeinde', 'Plz'])
                      .groupby('Gemeinde')[['x', 'y']].mean().round())
    logging.info(f"Postcode points: {len(postcodes):,} postcodes, "
                 f"{len(municipalities):,} municipalities")
    return postcodes, municipalities


def locate_from_points(df: pd.DataFrame, prefix: str,
                       points: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """In place: set `<prefix>_x/_y/_geocode_level` from `<prefix>_postcode`, falling back to
    `<prefix>_municipality` for postcodes not in the 2000 list. Logs the match rates."""
    postcodes, municipalities = points
    x = pd.Series(np.nan, index=df.index)
    y = pd.Series(np.nan, index=df.index)
    level = pd.Series(np.nan, index=df.index, dtype=object)
    for col, table, name in ((f'{prefix}_postcode', postcodes, 'postcode'),
                             (f'{prefix}_municipality', municipalities, 'municipality')):
        if col not in df:
            continue
        todo = x.isna() & df[col].notna()
        hit = todo & df[col].isin(table.index)
        x[hit] = df.loc[hit, col].map(table['x'])
        y[hit] = df.loc[hit, col].map(table['y'])
        level[hit] = name
        if todo.any():
            logging.info(f"  {prefix}: {int(hit.sum()):,} of {int(todo.sum()):,} located by {name}")
    df[f'{prefix}_x'], df[f'{prefix}_y'], df[f'{prefix}_geocode_level'] = x, y, level
