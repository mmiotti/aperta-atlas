"""
Per-building population for any area, derived from OSM buildings × per-OSM-tag
intensity coefficients calibrated on Swiss STATPOP. No ground-truth cell totals
required — the calibrated intensities carry the absolute scale.

Consumes the calibrated per-(OSM tag, category) intensities produced by
`preparation/switzerland/public/land_use/population_statpop.py` (safe to
share even when the per-building STATPOP output is restricted) and
produces per-building population for buildings anywhere.

If the calibrated coefs are stratum-expanded (see `_SPLIT_TAGS` in the
STATPOP script), this script:

  1. Detects the stratification from the coefs' tag names (via
     `derive_stratification_from_tags`).
  2. Loads STATPOP hectare artifacts for the pop-density source.
  3. Classifies each building into a stratum by local pop density
     (1 km ring), matching how the calibration did.
  4. Rewrites building tags to `<tag>__<stratum>` for split tags.
  5. Applies coefs; warns loudly if any tag remains unmatched.

For non-Swiss areas the density source doesn't exist yet — raises
NotImplementedError if the coefs are stratified. Options going forward:
GHS-POP-derived density fallback, or re-calibrate without stratification.

Uses `dasymetric.per_building_from_coefficients`:
  - Per (OSM tag, category), reads `intensity_<category>` from the calibrated
    table and assigns each building `area_m2 × intensity`.
  - Tags not in the calibrated table at all are dropped from output.
  - Tags present but with NaN intensity for some category contribute 0
    for that category.

Inputs (PUBLIC, cross-source):
    preparation/world/osm/shapes/buildings_<area_name>.gpkg
        # from preparation/world/osm/buildings_from_pbf.py
    preparation/switzerland/land_use/coefs/population_<year>.csv
        # from preparation/switzerland/public/land_use/population_statpop.py
    preparation/switzerland/land_use/hectares_statpop_<year>.{gpkg,csv}
        # (Swiss areas only, when coefs are stratified)

Outputs (PUBLIC, under preparation/world/land_use/):
    properties/buildings_population_coef_<area_name>.csv
        # indexed by building_id (OSM way ID); columns
        # population_{total, 19minus, 20to34, 35to49, 50to64, 65to79, 80plus}.
        # Join with shapes/buildings_<area_name>.gpkg. Output is prefixed
        # `population_coef_` to distinguish from the GHS-based per-building
        # population (`population_<area_name>`) that
        # `population_per_building_from_ghs.py` produces.

Run all variants sequentially (default):
    python -m preparation.world.land_use.population_per_building_from_coef
Single variant:
    python -m preparation.world.land_use.population_per_building_from_coef \\
        --variant switzerland
"""

import logging

import pandas as pd

from aperta_atlas import dasymetric
from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.switzerland.common import (
    classify_density_stratum, compute_hectare_density,
    derive_stratification_from_tags, stratify_building_tags,
)
from preparation.switzerland.public.land_use.population_statpop import (
    STRATUM_THRESHOLDS as _STATPOP_STRATUM_THRESHOLDS,
)
from preparation.world.areas import AREAS


_AGE_BANDS: tuple[str, ...] = (
    '19minus', '20to34', '35to49', '50to64', '65to79', '80plus',
)
_OUTPUT_COLUMNS: tuple[str, ...] = ('population_total',) + tuple(
    f'population_{b}' for b in _AGE_BANDS)


# `calibration_year` selects which Swiss STATPOP calibration release the
# coefficients come from. Held per-variant so multiple variants can pin
# to different vintages (e.g. for sensitivity analysis).
variants = Variants([('area_name', str), ('calibration_year', str)])
for area in AREAS.values():
    variants.add(
        name=area.name,
        area_name=area.name,
        calibration_year='2025',
    )


def _load_hectare_density_source(context, calibration_year: str):
    """Load STATPOP hectare cells + values for use as a pop-density
    source. Swiss-only for now."""
    lu_ch = context.source('preparation/switzerland/land_use')
    cells = lu_ch.get_generic(f'hectares_statpop_{calibration_year}.gpkg')
    totals = lu_ch.get_generic(
        f'hectares_statpop_{calibration_year}.csv',
        kws={'index_col': 'cell_id'},
    )
    return cells, totals


def _warn_on_unmatched_tags(buildings, coef_tags: set[str]) -> None:
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


def main(variant) -> None:
    context = init_context(variant)

    with step('load OSM building shapes'):
        buildings_ctx = context.source('preparation/world/osm')
        buildings = buildings_ctx.get_shapes('buildings', data_name=variant.area_name)
        logging.info(f"  → {len(buildings):,} buildings; "
                     f"{buildings['building'].nunique():,} unique OSM tags")

    with step(f'load calibrated coefficients ({variant.calibration_year})'):
        cal_ctx = context.source('preparation/switzerland/land_use')
        calibrated = cal_ctx.get_coefs(f'population_{variant.calibration_year}')
        logging.info(f"  → {len(calibrated):,} calibrated tags loaded")

    # Detect stratification from the coefs' tag names — no shared state
    # with the producer script needed.
    split_tags, strata = derive_stratification_from_tags(calibrated.index)

    if split_tags:
        logging.info(
            f"  → coefs are stratified: split_tags={sorted(split_tags)!r}, "
            f"strata={strata!r}")
        if not AREAS[variant.area_name].is_swiss:
            raise NotImplementedError(
                f"Calibrated coefs are stratum-expanded but area "
                f"{variant.area_name!r} is not Swiss; STATPOP hectare "
                f"density is only available for Swiss areas. Either add "
                f"a GHS-POP-based density fallback for non-Swiss areas, "
                f"or re-calibrate STATPOP with `_SPLIT_TAGS = set()` for "
                f"stratification-free coefs.")

        with step('classify buildings by local pop density (1 km ring)'):
            hex_cells, hex_totals = _load_hectare_density_source(
                context, variant.calibration_year)
            # OSM buildings ship in EPSG:4326 (degrees); hectares are LV95
            # (metres). Reproject so centroid + radius are metric.
            assert hex_cells.crs is not None
            if buildings.crs != hex_cells.crs:
                buildings = buildings.to_crs(hex_cells.crs)
                logging.info(f"  → reprojected buildings to {buildings.crs}")
            buildings['pop_density_1km'] = compute_hectare_density(
                buildings, hex_cells, hex_totals,
                column='population_total', radius_m=1_000.0,
            )
            buildings['stratum'] = classify_density_stratum(
                buildings['pop_density_1km'], _STATPOP_STRATUM_THRESHOLDS)
            counts = buildings['stratum'].value_counts()
            logging.info(
                f"  → stratum counts: " +
                ", ".join(f"{s}={counts.get(s, 0):,}"
                          for s in ('rural', 'suburban', 'urban')))

        with step(f'stratify building tags (split_tags={sorted(split_tags)!r})'):
            stratify_building_tags(buildings, split_tags)

    _warn_on_unmatched_tags(buildings, set(str(t) for t in calibrated.index))

    with step('apply calibrated intensities × area'):
        out = dasymetric.per_building_from_coefficients(
            buildings,
            columns=_OUTPUT_COLUMNS,
            calibrated_coeffs=calibrated,
        )
        band_sum = sum(out[f'population_{b}'].sum() for b in _AGE_BANDS)
        logging.info(
            f"  → assigned {out['population_total'].sum():,.0f} total "
            f"population across {len(out):,} buildings | " +
            ", ".join(f"{b}={out[f'population_{b}'].sum():,.0f}"
                      for b in _AGE_BANDS) +
            f" | Σbands={band_sum:,.0f}")

    # Save: per-building population columns as a property file (no
    # geometry — `buildings_from_pbf.py`'s shapes file is the geometry
    # source of truth, joined back via the `building_id` index). Prefixed
    # `population_coef_` to distinguish from GHS-based per-building
    # population which uses `population_<area>` (unprefixed).
    data_name = f'population_coef_{variant.area_name}'
    props = out[list(_OUTPUT_COLUMNS)].copy()
    context.create_properties(props, data_name=data_name)
    context.close()


if __name__ == '__main__':
    variants.run(main)
