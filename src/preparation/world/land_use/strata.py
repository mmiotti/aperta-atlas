"""
Density strata of buildings for the per-OSM-tag intensity coefficients: rural / suburban /
urban by the residents within `STRATUM_RADIUS_M` of the building, from GHS-POP.

Generic tags (`yes`, `office`, …; `_SPLIT_TAGS` in the STATPOP / STATENT calibrations)
hold very different numbers of residents or jobs per m² in a village and a city centre, so
their intensities are fitted per stratum. One register-free density defines the strata for
population and employment alike, in the calibration (Swiss buildings, STATPOP / STATENT) and
wherever the coefficients are applied (any area with a GHS-POP raster), so a stratum means
the same everywhere. (A density from the Swiss registers would leave every building outside
Switzerland at density 0, i.e. rural.)

Inputs (PUBLIC, under preparation/world/land_use/):
    population_<area_name>.tif      # population_ghs.py
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta import geo_processing
from aperta_atlas.context import Storage

from preparation.switzerland.common import classify_density_stratum

STRATUM_RADIUS_M = 1_000.0
# Residents per km² within `STRATUM_RADIUS_M`: about a scattered village / a dense town.
STRATUM_THRESHOLDS: tuple[float, float] = (500.0, 5_000.0)


def _ghs_points(raster_path: str) -> gpd.GeoDataFrame:
    """Centres of the non-zero GHS-POP pixels, with their `population_total`."""
    import rasterio
    from rasterio.transform import xy

    with rasterio.open(raster_path) as src:
        data, transform, crs, nodata = src.read(1), src.transform, src.crs, src.nodata
    mask = data > 0 if nodata is None else (data != nodata) & (data > 0)
    rows, cols = np.nonzero(mask)
    xs, ys = xy(transform, rows, cols, offset='center')
    return gpd.GeoDataFrame({'population_total': data[rows, cols].astype(float)},
                            geometry=gpd.points_from_xy(xs, ys), crs=crs)


def ghs_population_density(context, buildings: gpd.GeoDataFrame, area_name: str,
                           radius_m: float = STRATUM_RADIUS_M) -> pd.Series:
    """GHS-POP residents per m² within `radius_m` of each building's centroid (in a
    metric CRS: the buildings' own if projected, else their UTM zone)."""
    raster = context.source('preparation/world/land_use').path_for(
        Storage.PUBLIC, f'population_{area_name}.tif')
    crs = buildings.crs if buildings.crs.is_projected else buildings.estimate_utm_crs()
    targets = buildings.to_crs(crs)
    targets = targets.set_geometry(targets.geometry.centroid)
    return geo_processing.cross_sum_within_radius(
        targets=targets, sources=_ghs_points(raster).to_crs(crs), radius=radius_m,
        weight_column='population_total', return_density=True,
        name=f'ghs_population_density_{int(radius_m)}m')


def building_strata(context, buildings: gpd.GeoDataFrame, area_name: str) -> pd.Series:
    """Stratum label (`rural` / `suburban` / `urban`) of each building."""
    return classify_density_stratum(ghs_population_density(context, buildings, area_name),
                                    STRATUM_THRESHOLDS)


def warn_on_unmatched_tags(buildings, coef_tags: set[str]) -> None:
    """Log a WARNING listing OSM tag values present in `buildings` but
    absent from the calibrated coefs — those buildings will be silently
    filtered out by `per_building_from_coefficients`. Post-stratification,
    a healthy run should show zero missing."""
    building_tags = set(buildings['building'].dropna().astype(str).unique())
    missing = building_tags - coef_tags
    if not missing:
        return
    n_affected = int(buildings['building'].isin(missing).sum())
    frac = 100.0 * n_affected / max(len(buildings), 1)
    preview = sorted(missing)[:20]
    more = f' (+{len(missing) - 20} more)' if len(missing) > 20 else ''
    logging.warning(
        f"  ⚠ {n_affected:,} buildings ({frac:.1f}%) have tags not in "
        f"calibrated coefs — will be dropped from allocation. "
        f"Missing tags: {preview!r}{more}")
