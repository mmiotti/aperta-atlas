"""
AOI-scoped filters for training and validation data.

Purpose: restrict survey legs and traffic counters used for calibration
(04, 07, 09a, 03a) and counter-based validation (flows_vs_counters) to
the scenario's AOI. Region-agnostic — the AOI polygon is derived from
the scenario's own cells layer (`is_aoi=True`), so any scenario works
without a scenario-name check.

For switzerland-h10 (AOI = Switzerland), the filter is effectively
a pass-through: nearly all Swiss survey legs and counters fall inside
the country polygon. For cross-validation scenarios (cv-de-train,
cv-fr-test, etc.) the filter scopes training / validation to the
scenario's language region, so calibration and validation are both
performed on the same spatial subset.

Leg filter is symmetric: BOTH origin AND destination must fall inside
the AOI. Requiring both endpoints excludes cross-region legs, which
would leak out-of-region behaviour into training.
"""

import logging

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def load_aoi_polygon(context) -> BaseGeometry:
    """Return the scenario's AOI polygon in `scenario.crs_main`.

    Derived as the union of cells with `is_aoi=True` from the scenario's
    own `shapes/cells.gpkg`. This is the same polygon 01 used to classify
    cells (subject to cell-boundary discretization, sub-cell resolution).
    """
    cells = context.get_shapes('cells')
    aoi_cells = cells.loc[cells['is_aoi'].astype(bool)]
    if aoi_cells.empty:
        raise ValueError(
            "AOI is empty (no cells with is_aoi=True). Run 01_cells_zones "
            "first, or check that scenario.area_name is set correctly.")
    return unary_union(aoi_cells.geometry.tolist())


def filter_legs_by_aoi(
    legs, aoi_polygon: BaseGeometry, crs: str,
    *, orig_x: str = 'orig_x', orig_y: str = 'orig_y',
    dest_x: str = 'dest_x', dest_y: str = 'dest_y',
    label: str = 'legs',
):
    """Return only rows where BOTH endpoints fall inside `aoi_polygon`.

    `legs` is a pandas DataFrame with the four xy columns in `crs`.
    `aoi_polygon` must be in the same `crs` (typically `scenario.crs_main`).
    Logs the count reduction under `label` (e.g. 'MTMC legs', 'MOBIS legs').
    """
    before = len(legs)
    orig = gpd.GeoSeries(
        gpd.points_from_xy(legs[orig_x], legs[orig_y]),
        crs=crs, index=legs.index)
    dest = gpd.GeoSeries(
        gpd.points_from_xy(legs[dest_x], legs[dest_y]),
        crs=crs, index=legs.index)
    mask = orig.within(aoi_polygon) & dest.within(aoi_polygon)
    kept = legs.loc[mask]
    dropped = before - len(kept)
    logging.info(
        f"  → AOI filter ({label}): kept {len(kept):,}/{before:,} "
        f"({dropped:,} dropped; {100*dropped/max(before,1):.1f} %)")
    return kept


def filter_points_by_aoi(
    gdf: gpd.GeoDataFrame, aoi_polygon: BaseGeometry,
    *, label: str = 'points',
) -> gpd.GeoDataFrame:
    """Return only rows whose `geometry` falls inside `aoi_polygon`.

    `gdf` must be in the same CRS as `aoi_polygon`. Logs the count
    reduction under `label` (e.g. 'ASTRA counters').
    """
    before = len(gdf)
    kept = gdf.loc[gdf.geometry.within(aoi_polygon)].copy()
    dropped = before - len(kept)
    logging.info(
        f"  → AOI filter ({label}): kept {len(kept):,}/{before:,} "
        f"({dropped:,} dropped; {100*dropped/max(before,1):.1f} %)")
    return kept
