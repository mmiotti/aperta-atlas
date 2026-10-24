"""
AOI-scoped filters for training and validation data.

Purpose: restrict survey legs and traffic counters used for calibration
(04, 07, 09a, 03a) and counter-based validation (flows_vs_counters) to
the scenario's AOI. Region-agnostic — the AOI polygon is the scenario
area's place polygon buffered by `aoi_buffer_m`, the geometry 01 uses to
set `is_aoi`, so any scenario works without a scenario-name check.

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
from functools import lru_cache

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from preparation.world.areas import AREAS
from preparation.world.common import buffered_place_polygon
from scenarios import get_scenario


def load_aoi_polygon(context) -> BaseGeometry:
    """Return the scenario's AOI polygon in `scenario.crs_main`: the area's
    place polygon buffered by `aoi_buffer_m`, as in 01. Not the union of the
    `is_aoi` cells: 01 drops cells without residents, jobs or POIs, which
    would leave holes where survey legs start or end (car parks, parks).
    """
    scenario = get_scenario(context.scenario)
    return _aoi_polygon(scenario.area_name, scenario.crs_main)


@lru_cache(maxsize=None)
def _aoi_polygon(area_name: str, crs: str) -> BaseGeometry:
    area = AREAS[area_name]
    return (buffered_place_polygon(area.place, area.aoi_buffer_m)
            .to_crs(crs).geometry.union_all())


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
    reduction under `label` (e.g. 'traffic counters').
    """
    before = len(gdf)
    kept = gdf.loc[gdf.geometry.within(aoi_polygon)].copy()
    dropped = before - len(kept)
    logging.info(
        f"  → AOI filter ({label}): kept {len(kept):,}/{before:,} "
        f"({dropped:,} dropped; {100*dropped/max(before,1):.1f} %)")
    return kept
