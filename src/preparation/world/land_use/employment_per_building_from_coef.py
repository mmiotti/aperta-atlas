"""
Per-building employment for any area, derived from OSM buildings × per-OSM-tag
intensity coefficients calibrated on Swiss STATENT. No ground-truth cell totals
required — the calibrated intensities carry the absolute scale.

Consumes the calibrated per-(OSM tag, sector) intensities produced by
`preparation/switzerland/public/land_use/employment_statent.py` (safe to
share even when the per-building STATENT output is restricted) and
produces per-building employment for buildings anywhere.

If the calibrated coefs are stratum-expanded (see `_SPLIT_TAGS` in the
STATENT script), this script:

  1. Detects the stratification from the coefs' tag names (via
     `derive_stratification_from_tags`).
  2. Classifies each building into a stratum by GHS-POP residents
     within 1 km (`preparation.world.land_use.strata`), as the
     calibration did; works for any area with a GHS-POP raster.
  3. Rewrites building tags to `<tag>__<stratum>` for split tags.
  4. Applies coefs; warns loudly if any tag remains unmatched.

Uses `dasymetric.per_building_from_coefficients`:
  - Per (OSM tag, sector), reads `intensity_<sector>` from the calibrated table
    and assigns each building `area_m2 × intensity`.
  - Tags not in the calibrated table at all are dropped from output.
  - Tags present but with NaN intensity for some sector contribute 0 for that sector.

Inputs (PUBLIC, cross-source):
    preparation/world/osm/shapes/buildings_<area_name>.gpkg
        # from preparation/world/osm/buildings_from_pbf.py
    preparation/switzerland/land_use/coefs/employment_statent_<year>.csv
        # from preparation/switzerland/public/land_use/employment_statent.py
    preparation/world/land_use/population_<area_name>.tif
        # GHS-POP (population_ghs.py), for the density strata

Outputs (PUBLIC, under preparation/world/land_use/):
    properties/buildings_employment_<area_name>.csv
        # indexed by building_id (OSM way ID); columns
        # employment_{total, primary, secondary, tertiary}.
        # Join with shapes/buildings_<area_name>.gpkg.

Run all variants sequentially (default):
    python -m preparation.world.land_use.employment_per_building_from_coef
Single variant:
    python -m preparation.world.land_use.employment_per_building_from_coef \\
        --variant switzerland
"""

import logging

from aperta_atlas import dasymetric
from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.switzerland.common import (
    derive_stratification_from_tags, stratify_building_tags,
)
from preparation.world.areas import AREAS
from preparation.world.land_use.strata import building_strata, warn_on_unmatched_tags


_SECTORS: tuple[str, ...] = ('primary', 'secondary', 'tertiary')
_OUTPUT_COLUMNS: tuple[str, ...] = ('employment_total',) + tuple(
    f'employment_{s}' for s in _SECTORS)


# `calibration_year` selects which Swiss STATENT calibration release the
# coefficients come from. Held per-variant so multiple variants can pin
# to different vintages (e.g. for sensitivity analysis).
variants = Variants([('area_name', str), ('calibration_year', str)])
for area in AREAS.values():
    variants.add(
        name=area.name,
        area_name=area.name,
        calibration_year='2024',
    )


def main(variant) -> None:
    context = init_context(variant)

    with step('load OSM building shapes'):
        buildings_ctx = context.source('preparation/world/osm')
        buildings = buildings_ctx.get_shapes('buildings', data_name=variant.area_name)
        logging.info(f"  → {len(buildings):,} buildings; "
                     f"{buildings['building'].nunique():,} unique OSM tags")

    with step(f'load calibrated coefficients ({variant.calibration_year})'):
        cal_ctx = context.source('preparation/switzerland/land_use')
        calibrated = cal_ctx.get_coefs(f'employment_statent_{variant.calibration_year}')
        logging.info(f"  → {len(calibrated):,} calibrated tags loaded")

    # Detect stratification from the coefs' tag names — no shared state
    # with the producer script needed.
    split_tags, strata = derive_stratification_from_tags(calibrated.index)

    if split_tags:
        logging.info(
            f"  → coefs are stratified: split_tags={sorted(split_tags)!r}, "
            f"strata={strata!r}")
        with step('classify buildings by GHS-POP density (1 km ring)'):
            buildings['stratum'] = building_strata(context, buildings, variant.area_name)
            counts = buildings['stratum'].value_counts()
            logging.info(
                "  → stratum counts: " +
                ", ".join(f"{s}={counts.get(s, 0):,}"
                          for s in ('rural', 'suburban', 'urban')))

        with step(f'stratify building tags (split_tags={sorted(split_tags)!r})'):
            stratify_building_tags(buildings, split_tags)

    warn_on_unmatched_tags(buildings, {str(t) for t in calibrated.index})

    with step('apply calibrated intensities × area'):
        out = dasymetric.per_building_from_coefficients(
            buildings,
            columns=_OUTPUT_COLUMNS,
            calibrated_coeffs=calibrated,
        )
        sector_sum = sum(out[f'employment_{s}'].sum() for s in _SECTORS)
        logging.info(
            f"  → assigned {out['employment_total'].sum():,.0f} total employees "
            f"across {len(out):,} buildings | " +
            ", ".join(f"{s}={out[f'employment_{s}'].sum():,.0f}"
                      for s in _SECTORS) +
            f" | Σsectors={sector_sum:,.0f}")

    context.create_properties(out[list(_OUTPUT_COLUMNS)].copy(),
                              data_name=f'employment_{variant.area_name}')
    context.close()


if __name__ == '__main__':
    variants.run(main)
