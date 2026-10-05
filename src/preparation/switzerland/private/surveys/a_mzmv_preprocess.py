"""
Preprocess raw MZMV (Mikrozensus Mobilität und Verkehr) survey CSVs.

> **Private data.** Requires raw MZMV vintage CSVs (Switzerland's
> federal mobility survey, BFS-restricted access); not redistributable.
> Published as documentation of the method.

Reads the raw per-year MZMV tables (trips / legs / people / households / multi-day
trips), reprojects all _X/_Y coordinate columns from WGS84 to LV95, applies survey-
year column harmonizations so 2015 and 2021 are downstream-comparable, and writes
the result under `preparation/.../mzmv_<year>/preprocessed/`.

The MZMV is released roughly every five years; this script supports the 2015 and
2021 vintages. CSV separator differs between vintages (',' before 2021, ';' from
2021) — autodetected.

Inputs (under <DATA_DIR_PRIVATE>/raw/switzerland/mzmv/):
    MZMV{year}_mit_Geo/4_DB_csv/{wege,etappen,zielpersonen,haushalte,tagesreisen,reisenmueb}.csv

Outputs:
    mzmv_<year>/preprocessed/{trips,legs,people,households,trips_day,trips_multiday}.csv
        # PRIVATE

Run all years sequentially (default):
    python -m preparation.switzerland.private.surveys.a_mzmv_preprocess
Single year:
    python -m preparation.switzerland.private.surveys.a_mzmv_preprocess --variant 2021
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer

from aperta_atlas.context import init_context, Context
from aperta_atlas.context import Storage
from aperta_atlas.variant import Variants

from preparation.switzerland.common import CRS_CH, CRS_LATLON


MZMV_YEARS = (2015, 2021)
INVALID_COORD_SENTINEL = -999

variants = Variants([('mzmv_year', int)])
for _y in MZMV_YEARS:
    variants.add(name=str(_y), mzmv_year=_y)


# Files to preprocess: (output_name, raw_filename_without_csv). One row per table.
_FILES: tuple[tuple[str, str], ...] = (
    ('trips',          'wege'),
    ('legs',           'etappen'),
    ('people',         'zielpersonen'),
    ('households',     'haushalte'),
    ('trips_day',      'tagesreisen'),
    ('trips_multiday', 'reisenmueb'),
)


# Pre-2021 → 2021 column code harmonizations. Each maps old-vintage code values to
# the corresponding 2021 codes so downstream year-aware logic doesn't have to branch.

# Main travel mode (households / legs).
_PRE2021_WMITTEL_REMAP: dict[int, int] = {
    3: 5, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    14: 15, 15: 16, 16: 17, 17: 18,
}

# Leg mode (etappen).
_PRE2021_LEG_MODE_REMAP: dict[int, int] = {
    3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 12,
    12: 13, 13: 14, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 3, 21: 4,
}

# Education level (HAUSB → f40120).
_PRE2021_HAUSB_REMAP: dict[int, int] = {
    -99: -99, -98: -98, -97: -97,
    1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 3, 8: 4, 9: 4,
    10: 5, 11: 5, 12: 5, 13: 6, 14: 6, 15: 7, 16: 7, 17: 8, 18: 8, 19: 8,
}


def _reproject_xy_columns(df: pd.DataFrame, transformer: Transformer) -> None:
    """In-place: for every `<col>_X` / `<col>_Y` coord pair, add `<col>_X_LV95` and
    `<col>_Y_LV95` columns containing the reprojected (and rounded) coordinates.
    Coordinates marked invalid (< -998) or non-finite after reprojection are written
    as `INVALID_COORD_SENTINEL`.
    """
    for x_col in [c for c in df.columns if c.endswith('_X')]:
        y_col = x_col[:-2] + '_Y'
        coords = df[[x_col, y_col]].to_numpy()
        # transformer.transform expects (lat, lon); raw cols are (X=lon, Y=lat).
        x_lv95, y_lv95 = transformer.transform(coords[:, 1], coords[:, 0])
        xy = np.column_stack([x_lv95, y_lv95])
        invalid = (coords < -998).any(axis=1) | ~np.isfinite(xy).all(axis=1)
        xy[invalid] = INVALID_COORD_SENTINEL
        xy = np.round(xy).astype(int)
        df[f'{x_col}_LV95'] = xy[:, 0]
        df[f'{y_col}_LV95'] = xy[:, 1]


def _apply_pre_2021_harmonizations(df: pd.DataFrame) -> None:
    """In-place: convert pre-2021 column codes to the 2021 convention where present.

    Each remap fires only when its trigger column exists, so the same helper is safe
    to call on any of the MZMV tables.
    """
    if 'wmittel' in df.columns:
        df['wmittel2'] = df['wmittel'].replace(_PRE2021_WMITTEL_REMAP)
    if 'f51300' in df.columns:
        df['f51300'] = df['f51300'].replace(_PRE2021_LEG_MODE_REMAP)
    if 'f41610a' in df.columns:
        # f41610{a..g} → f41600_01{a..g} (transit-pass columns).
        for i in range(7):
            suffix = chr(97 + i)
            df[f'f41600_01{suffix}'] = df[f'f41610{suffix}']
    if 'HAUSB' in df.columns:
        df['f40120'] = df['HAUSB'].replace(_PRE2021_HAUSB_REMAP)
    if 'F20601' in df.columns:
        # Lowercase rename (income); 2021 uses lowercase.
        df['f20601'] = df['F20601']


def preprocess_year(context: Context, mzmv_year: int) -> None:
    logging.info(f"Processing MZMV year {mzmv_year}")
    transformer = Transformer.from_crs(CRS_LATLON, CRS_CH)
    raw_dir = f'switzerland/mzmv/MZMV{mzmv_year}_mit_Geo/4_DB_csv'
    sep = ',' if mzmv_year < 2021 else ';'

    for out_name, raw_name in _FILES:
        raw_path = context.raw_path(Storage.PRIVATE, f'{raw_dir}/{raw_name}.csv')
        df = pd.read_csv(raw_path, sep=sep, encoding='ISO-8859-1')
        if len(df.columns) < 2:
            raise ValueError(
                f"{raw_name}.csv parsed to {len(df.columns)} columns — separator may "
                f"be wrong (expected {sep!r}).")
        _reproject_xy_columns(df, transformer)
        if mzmv_year < 2021:
            _apply_pre_2021_harmonizations(df)
        if mzmv_year >= 2021 and 'Tag' in df.columns:
            df['tag'] = df['Tag']
        context.create_generic(
            df, f'mzmv_{mzmv_year}/preprocessed/{out_name}.csv',
            kws={'encoding': 'utf-8'},
        )


def main(variant) -> None:
    context = init_context(variant)
    preprocess_year(context, variant.mzmv_year)
    context.close()


if __name__ == '__main__':
    variants.run(main)
