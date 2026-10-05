"""
Distribute Swiss hectare-level employment totals (BFS STATENT) across OSM
building footprints via dasymetric mapping.

Pipeline: load STATENT CSV → uncap privacy-suppressed values → read
sector totals directly (B08EMPTS1/2/3) + per-industry FTE from NOGA-08
codes → build 100 m hectare cell polygons → classify buildings by local
emp density (1 km ring) → stratify OSM tag values (`_SPLIT_TAGS` ×
rural/suburban/urban) → one NNLS fit per category (`employment_total`
+ each sector) with per-cell rescaling to match STATENT totals exactly
→ per-cell proportional split of sector into 11 industries.

**Privacy uncapping.** Values ≤ cap are redrawn via `replace_smaller_equal`
with weights `[0.45, 0.25, 0.20, 0.10]` to approximate the true tail.
Fixed RNG seed for reproducibility. Cap is auto-detected from the total
column.

**Density-based stratification** (see `_SPLIT_TAGS`,
`STRATUM_THRESHOLDS`). Tag values in `_SPLIT_TAGS` get separate NNLS
coefficients per stratum, because FTE intensity (jobs/m²) for the same
tag varies with local employment density. Non-split tags share one
intensity across strata. Density source is STATENT (independent of
STATPOP-side pop-density stratification).

**Sector totals read directly** from `B08EMPTS1/2/3` (primary /
secondary / tertiary), not aggregated from per-industry codes.
Aggregating per-industry values accumulates capping error additively;
sector totals are single privacy-capped values.

**11-industry detail via per-cell proportional split.** After the
NNLS fit for `employment_total` + each sector, per-building sector
employment is split across 11 industries (agriculture, mining,
manufacturing, construction_utilities, retail, transport_logistics,
hospitality_recreation, services, research_technology,
healthcare_veterinary, education_welfare) by the cell-level industry
share within sector. Avoids needing per-(OSM tag × industry) priors;
assumes industry mix within each 100 m cell is roughly uniform across
its buildings — defensible at this scale.

`employment_total` and each sector are fit independently — each
category gets its own NNLS regression and per-cell rescaling. Building
totals from `employment_total` are not constrained to equal the sum of
per-sector values (both derive from independent fits); log line reports
both for sanity.

Inputs (PUBLIC, under raw/switzerland/statent/):
    betriebszaehlungen_<year>_ha_2056.csv   # BFS release
Cross-source (via `context.source(...)`):
    preparation/world/osm/shapes/buildings_<country>.gpkg

Outputs (PUBLIC, under preparation/switzerland/land_use/):
    properties/buildings_employment_statent_<year>.csv   # per-building
        # columns: employment_{total, primary, secondary, tertiary,
        # agriculture, mining, manufacturing, construction_utilities,
        # retail, transport_logistics, hospitality_recreation, services,
        # research_technology, healthcare_veterinary, education_welfare}.
    coefs/employment_statent_<year>.csv                  # per-(tag, category)
        # intensities (FTE/m²). Consumed by
        # preparation/world/land_use/employment_per_building_from_coef.py
        # via context.get_coefs(...).
    hectares_statent_<year>.{gpkg,csv}                   # hectare-cell
        # polygons + per-hectare employment values (used by atlas
        # cell_source='hectares' in `main/01_cells_zones.py`).

Run all variants sequentially (default):
    python -m preparation.switzerland.public.land_use.employment_statent
Single variant:
    python -m preparation.switzerland.public.land_use.employment_statent --variant statent_2024
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta_atlas import dasymetric
from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.switzerland.common import (
    build_hectare_cells, classify_density_stratum, combine_learned_per_category,
    compute_hectare_density, expand_relevant_tags, filter_to_inside_ch,
    load_swiss_country, merge_per_category_outputs, replace_smaller_equal,
    stratify_building_tags,
)


# Max distance (in metres) for the nearest-building fallback for cells
# with no building overlap (only for those cells). Set conservatively.
_NEAREST_FALLBACK_MAX_M = 200.0

# OSM tag values whose FTE intensity varies with local emp density.
# Each split tag gets separate NNLS coefficients per stratum
# (rural / suburban / urban). Non-split tags share one intensity.
_SPLIT_TAGS: set[str] = {
    'yes',
    'commercial',
    'industrial',
    'retail',
    'office',
    'warehouse',
    'apartments',
    'hospital',
    'residential',
    'school',
}

# Per-km² breakpoints for the 3-way rural / suburban / urban split.
# Can be different from population's, but in practice, similar cutoffs made
# sense (there is less employment overall than population, but employment is
# more concentrated).
STRATUM_THRESHOLDS: tuple[float, float] = (500.0, 5_000.0)

# STATENT filename pattern (BFS public release; both .csv and .gpkg exist,
# we read the .csv).
_STATENT_CSV_PATTERN = 'switzerland/statent/betriebszaehlungen_{year}_ha_2056.csv'


def _detect_privacy_cap(values: np.ndarray) -> int:
    """Return BFS's privacy-cap value inferred from `values`.

    BFS caps true counts 1..n by replacing them with the literal cap
    value n (any cell that would report 1, 2, …, n publishes n instead).
    The smallest strictly-positive integer in the array IS the cap.
    Assumes at least one capped cell exists — safe at national hectare
    scale."""
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("No positive values in the array — cannot detect cap.")
    return int(positive.min())

# NOGA-08 / NACE Rev. 2 industry codes published by STATENT under
# `B08<code>EMP` (B08 = data series; <code> = NOGA-08 2-digit industry;
# EMP = FTE count).
#
# Sector totals (`employment_primary`, `_secondary`, `_tertiary`) are
# read DIRECTLY from BFS-published `B08EMPTS{1,2,3}` columns
# (see `_SECTOR_COLS`), not aggregated from these industry codes —
# per-industry values carry higher relative capping error.
# `_INDUSTRY_CODES` + `_INDUSTRY_TO_SECTOR` remain the source of truth
# for the per-industry breakdown columns.
#
# Sector boundaries follow the standard BFS / Eurostat Fisher-Clark
# rollup of NACE Rev. 2:
#   - Primary   (Section A):     codes 01–03  (Agriculture, forestry, fishing)
#   - Secondary (Sections B–F):  codes 05–43  (Mining, Manufacturing,
#                                              Utilities, Construction)
#   - Tertiary  (Sections G–U):  codes 45–99  (Trade, transport, services,
#                                              admin, …)
# Codes 04, 40, 44 are intentional gaps in NOGA-08.
_SECTORS: tuple[str, ...] = ('primary', 'secondary', 'tertiary')

# Per-industry NOGA-08 codes. Codes determine which raw STATENT
# `B08<code>EMP` columns roll up into each industry. Insertion order is
# the canonical industry ordering (used wherever `tuple(_INDUSTRY_CODES)`
# is iterated).
_INDUSTRY_CODES: dict[str, list[str]] = {
    'agriculture':           ['01', '02', '03'],
    'mining':                ['05', '06', '07', '08', '09'],
    'manufacturing':         [str(i) for i in range(10, 34)],
    'construction_utilities':['35', '36', '37', '38', '39', '41', '42', '43'],
    'retail':                ['45', '46', '47'],
    'transport_logistics':   ['49', '50', '51', '52', '53'],
    'hospitality_recreation':['55', '56', '58', '59', '79', '90', '91', '92', '93', '94'],
    'services':              ['60', '61', '62', '63', '64', '65', '66', '68',
                              '69', '70', '73', '77', '78', '80', '81', '82',
                              '84', '95', '96'],
    'research_technology':   ['71', '72', '74'],
    'healthcare_veterinary': ['75', '86', '87'],
    'education_welfare':     ['85', '88'],
}
_INDUSTRY_TO_SECTOR: dict[str, str] = {
    'agriculture':           'primary',
    'mining':                'secondary',
    'manufacturing':         'secondary',
    'construction_utilities':'secondary',
    'retail':                'tertiary',
    'transport_logistics':   'tertiary',
    'hospitality_recreation':'tertiary',
    'services':              'tertiary',
    'research_technology':   'tertiary',
    'healthcare_veterinary': 'tertiary',
    'education_welfare':     'tertiary',
}

# Per-sector STATENT column names — BFS publishes sector totals directly
# (`B08EMPTS1` = primary, `B08EMPTS2` = secondary, `B08EMPTS3` = tertiary).
# Reading these DIRECTLY is strictly better than aggregating per-industry
# codes: sector totals are single privacy-capped values, whereas summing
# 10-40 per-industry cells accumulates the capping error additively.
_SECTOR_COLS: dict[str, str] = {
    'primary':   'B08EMPTS1',
    'secondary': 'B08EMPTS2',
    'tertiary':  'B08EMPTS3',
}


variants = Variants([('year', str), ('buildings_country', str)])
variants.add(name='statent_2024', year='2024', buildings_country='switzerland')


def _load_and_aggregate_statent(
    csv_path: str, rng: np.random.Generator,
) -> pd.DataFrame:
    """Load the STATENT CSV, apply privacy uncapping, expose the
    3-sector + 11-industry totals plus an `employment_total` column.

    Returns a DataFrame indexed by `(E_KOORD, N_KOORD)` (hectare lower-
    left corner) with columns `employment_<sector>` for the 3 sectors,
    one `employment_<industry>` per entry in `_INDUSTRY_CODES`, and
    `employment_total`. Cells outside Switzerland are not in the CSV
    by design. Privacy cap auto-detected from the total column.

    Sector totals are read DIRECTLY from `B08EMPTS{1,2,3}` — not
    aggregated from per-industry codes — because per-industry values
    carry higher relative capping error than the sector totals.
    Per-industry columns remain useful for finer-grained analysis and
    are exposed as-is (noisy for small industries in sparse cells).
    """
    coords = ['E_KOORD', 'N_KOORD']
    df = pd.read_csv(csv_path, sep=';').set_index(coords)
    logging.info(f"  → loaded {len(df):,} STATENT hectare rows, {len(df.columns)} columns")

    # `B08EMPT` is the published per-cell FTE total (privacy-capped).
    total_arr = df['B08EMPT'].to_numpy(copy=True)
    cap = _detect_privacy_cap(total_arr)
    logging.info(f"  → detected STATENT privacy cap = {cap}")
    logging.info(
        f"  → sum of raw B08EMPT (with caps in place): {total_arr.sum():,.0f}")

    total = replace_smaller_equal(rng, total_arr, cap)

    def _uncap_col(col: str) -> np.ndarray:
        """Read + uncap a single STATENT column; return zeros if absent."""
        if col not in df.columns:
            return np.zeros(len(df), dtype=float)
        v = df[col].to_numpy(copy=True)
        v = replace_smaller_equal(rng, v, cap)
        return v.astype(float)

    def _aggregate_industry(codes: list[str]) -> np.ndarray:
        """Sum of per-industry STATENT columns after privacy uncapping.
        Note: cap error accumulates additively with the number of codes
        summed — sector totals should be read directly, not aggregated
        this way (see `_SECTOR_COLS`)."""
        col_names = [f'B08{c}EMP' for c in codes if f'B08{c}EMP' in df.columns]
        if not col_names:
            return np.zeros(len(df), dtype=float)
        v = df[col_names].to_numpy(copy=True)
        v = replace_smaller_equal(rng, v, cap)
        return v.sum(axis=1).astype(float)

    # Build all output columns in a dict, then concat once. STATENT has
    # ~200 raw columns; inserting 14 new columns one-by-one triggers
    # pandas PerformanceWarning about fragmentation.
    new_cols: dict[str, np.ndarray] = {'employment_total': total.astype(float)}
    for sector, col in _SECTOR_COLS.items():
        new_cols[f'employment_{sector}'] = _uncap_col(col)
    for industry, codes in _INDUSTRY_CODES.items():
        new_cols[f'employment_{industry}'] = _aggregate_industry(codes)
    out = pd.DataFrame(new_cols, index=df.index)
    logging.info(
        f"  → aggregated; national total FTE: {out['employment_total'].sum():,.0f}; "
        f"by sector: " +
        ", ".join(f"{s}={out[f'employment_{s}'].sum():,.0f}" for s in _SECTORS))
    return out


# Industry → sector mapping resolved to column names for the per-cell
# proportional split (industry employment = sector employment ×
# cell-level (industry / sector) share at building's centroid cell).
_INDUSTRY_SPLITS: dict[str, str] = {
    f'employment_{ind}': f'employment_{_INDUSTRY_TO_SECTOR[ind]}'
    for ind in _INDUSTRY_CODES
}


def main(variant) -> None:
    context = init_context(variant)
    rng = np.random.default_rng(42)

    csv_path = context.raw_path(
        Storage.PUBLIC, _STATENT_CSV_PATTERN.format(year=variant.year))

    with step(f'parse + uncap + aggregate STATENT ({variant.year})'):
        stat_df = _load_and_aggregate_statent(csv_path, rng)

    with step('build hectare cell polygons (100 m squares, LV95)'):
        cells = build_hectare_cells(stat_df)
        cell_totals = stat_df.copy()
        cell_totals.index = cells['cell_id'].values
        cell_totals.index.name = 'cell_id'

    with step('persist STATENT hectare cells + per-hectare values as generic artifacts'):
        # Re-key cell_id from positional int to coordinate-stable string
        # ('E_N' from LV95 lower-left corners) so downstream consumers
        # (e.g. atlas 01, cell_source='hectares') can safely union
        # STATPOP and STATENT hectares — both share the same BFS 100 m
        # grid, so coordinates are the natural cross-source key.
        coord_ids = [f'{int(e)}_{int(n)}' for e, n in zip(
            stat_df.index.get_level_values(0),
            stat_df.index.get_level_values(1))]
        out_cells = cells.copy()
        out_cells['cell_id'] = coord_ids
        out_totals = cell_totals.copy()
        out_totals.index = pd.Index(coord_ids, name='cell_id')
        # Both files anchored on the source name (statent) for symmetry
        # and provenance; the CSV's column names carry the semantic info.
        context.create_generic(out_cells, f'hectares_statent_{variant.year}.gpkg')
        context.create_generic(
            out_totals, f'hectares_statent_{variant.year}.csv',
            kws={'float_format': '%.4f'},
        )

    with step('load OSM building shapes (buildings_download.py output)'):
        buildings_ctx = context.source('preparation/world/osm')
        buildings = buildings_ctx.get_shapes('buildings', data_name=variant.buildings_country)
        # `build_hectare_cells` always sets crs=LV95 — assertion narrows
        # the type checker's `CRS | None` to `CRS`.
        assert cells.crs is not None
        if buildings.crs != cells.crs:
            buildings = buildings.to_crs(cells.crs)
        logging.info(f"  → {len(buildings):,} buildings loaded; reprojected to {buildings.crs}")

    with step('filter buildings to inside CH borders (calibration area)'):
        # `learn_coefficients`'s bias correction assumes `buildings` IS
        # the calibration area's building set. The OSM input includes
        # the country + buffer ring; we must trim to CH so the published
        # β values reflect "average intensity across all CH buildings of
        # this tag" — invariant to buffer presence and consistent under
        # bottom-up application.
        n_before = len(buildings)
        country = load_swiss_country(context)
        buildings = filter_to_inside_ch(buildings, country)
        logging.info(
            f"  → {len(buildings):,} of {n_before:,} buildings inside CH "
            f"({len(buildings)/n_before*100:.1f} %)")

    with step('classify buildings by local employment density (1 km ring)'):
        buildings['emp_density_1km'] = compute_hectare_density(
            buildings, cells, cell_totals,
            column='employment_total', radius_m=1_000.0,
        )
        buildings['stratum'] = classify_density_stratum(
            buildings['emp_density_1km'], STRATUM_THRESHOLDS)
        counts = buildings['stratum'].value_counts()
        logging.info(
            f"  → stratum counts: " +
            ", ".join(f"{s}={counts.get(s, 0):,}"
                      for s in ('rural', 'suburban', 'urban')))

    with step(f'stratify building tags (split_tags={sorted(_SPLIT_TAGS)!r})'):
        stratify_building_tags(buildings, _SPLIT_TAGS)

    # One NNLS fit + allocation per category. Total's tag set is the
    # union of per-sector tag sets; each is stratum-expanded per
    # `_SPLIT_TAGS`.
    _base_tags_per_category: dict[str, frozenset[str]] = {
        'total': frozenset().union(*dasymetric.EMPLOYMENT_TAGS_PER_SECTOR.values()),
        **dasymetric.EMPLOYMENT_TAGS_PER_SECTOR,
    }
    tags_per_category = {
        cat: expand_relevant_tags(base, _SPLIT_TAGS)
        for cat, base in _base_tags_per_category.items()
    }

    learned_per_category: dict[str, 'pd.DataFrame'] = {}
    per_building_outputs: dict[str, 'gpd.GeoDataFrame'] = {}
    for cat, relevant_tags in tags_per_category.items():
        col = f'employment_{cat}'
        with step(f'{col}: learn NNLS coefficients'):
            learned_per_category[col] = dasymetric.learn_coefficients(
                buildings, cells, cell_totals,
                cell_id_col='cell_id', column=col,
                relevant_tags=relevant_tags,
                nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
            )
        with step(f'{col}: allocate to buildings'):
            tag_intensities = {
                t: float(i) for t, i in
                learned_per_category[col][f'intensity_{col}'].dropna().items()
            }
            coeffs = {t: tag_intensities.get(t, 1.0) for t in relevant_tags}
            per_building_outputs[col] = dasymetric.per_building(
                buildings, cells, cell_totals,
                cell_id_col='cell_id', column=col,
                coeffs=coeffs,
                nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
            )

    with step('merge per-category outputs'):
        out = merge_per_category_outputs(buildings, per_building_outputs)
        sector_sum = sum(out[f'employment_{s}'].sum() for s in _SECTORS)
        logging.info(
            f"  → distributed {out['employment_total'].sum():,.0f} total FTE "
            f"across {len(out):,} buildings | " +
            ", ".join(f"{s}={out[f'employment_{s}'].sum():,.0f}"
                      for s in _SECTORS) +
            f" | Σsectors={sector_sum:,.0f}")

    with step('combine per-category learned coefficients into one wide-form table'):
        learned = combine_learned_per_category(learned_per_category)

    # 10-industry breakdown via per-cell proportional split within sector.
    with step('industry proportional split (within sector, per cell)'):
        out = dasymetric.apply_proportional_split(
            out, cells, cell_totals,
            cell_id_col='cell_id', splits=_INDUSTRY_SPLITS,
        )
        for industry in _INDUSTRY_CODES:
            tot = out[f'employment_{industry}'].sum()
            logging.info(f"  → {industry}: {tot:,.0f} FTE")

    data_name = f'employment_statent_{variant.year}'
    keep_cols = (
        [f'employment_{s}' for s in _SECTORS]
        + [f'employment_{i}' for i in _INDUSTRY_CODES]
        + ['employment_total'])
    props = out[keep_cols].copy()

    with step('save per-building employment + calibrated coefficients'):
        context.create_properties(props, data_name=data_name)
        # Calibrated per-(OSM tag, sector) intensities — namespace-scoped
        # coef. The downstream public-data consumer
        # (`preparation/world/land_use/employment_per_building_from_coef`)
        # reads it back via `get_coefs`. Lives at
        # `preparation/switzerland/land_use/coefs/<data_name>.csv`.
        context.create_coefs(learned, data_name)

    context.close()


if __name__ == '__main__':
    variants.run(main)
