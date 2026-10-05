"""
Distribute Swiss hectare-level population totals (BFS STATPOP) across OSM
building footprints via dasymetric mapping.

Higher fidelity than the GHS-POP-based alternative in
`preparation/world/land_use/population_per_building_from_ghs.py`, limited
to Swiss borders.

Pipeline: load STATPOP CSV → uncap privacy-suppressed values (BBTOT ≤ cap
is redrawn from a weighted distribution) → aggregate to
`population_total` + 6 age bands → build 100 m hectare cell polygons →
classify buildings by local pop density (1 km ring) → stratify OSM tag
values (`_SPLIT_TAGS` × rural/suburban/urban) → one NNLS fit per
category (`population_total` + each age band) with per-cell rescaling
to match STATPOP totals exactly.

**Privacy uncapping.** Values ≤ cap are redrawn via
`replace_smaller_equal` with weights `[0.65, 0.25, 0.10]` (single
households most common, then pairs, then triples). Fixed RNG seed for
reproducibility. Cap is auto-detected from the total column.

**Density-based stratification** (see `_SPLIT_TAGS`,
`STRATUM_THRESHOLDS`). Tag values in `_SPLIT_TAGS` get separate NNLS
coefficients per stratum, because residential intensity (people/m²) for
the same tag varies systematically with local density. Non-split tags
share one intensity across strata.

**Age bands.** STATPOP publishes finer age breakdowns (split further by
gender). The 6 bands here are the standard aggregation:
`19minus / 20to34 / 35to49 / 50to64 / 65to79 / 80plus`. Gender is
collapsed (M + W summed within band) to keep the output schema compact.

`population_total` and each age band are fit independently — each
category gets its own NNLS regression and per-cell rescaling. Building
totals from `population_total` are not constrained to equal the sum of
building band values (both derive from independent fits with capping
noise); log line reports both for sanity.

Inputs (PUBLIC, under raw/switzerland/statpop/):
    statistik-bevoelkerung_haushalte_<year>_ha_2056.csv   # BFS release
Cross-source (via `context.source(...)`):
    preparation/world/osm/shapes/buildings_<country>.gpkg

Outputs (PUBLIC, under preparation/switzerland/land_use/):
    properties/buildings_population_<year>.csv       # per-building
        # columns: population_{total, 19minus, 20to34, 35to49,
        # 50to64, 65to79, 80plus}. Join with buildings_<country>.gpkg.
    coefs/population_<year>.csv                      # per-(tag, category)
        # intensities (residents/m²). Consume via context.get_coefs(...).
    hectares_statpop_<year>.{gpkg,csv}               # hectare-cell
        # polygons + per-hectare population values (used by atlas
        # cell_source='hectares' in `main/01_cells_zones.py`).

Run all variants sequentially (default):
    python -m preparation.switzerland.public.land_use.population_statpop
Single variant:
    python -m preparation.switzerland.public.land_use.population_statpop --variant statpop_2025
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

# OSM tag values whose residential intensity varies with local pop
# density. Each split tag gets separate NNLS coefficients per stratum
# (rural / suburban / urban). Non-split tags share one intensity.
_SPLIT_TAGS: set[str] = {
    'yes',
    'residential',
    'apartments',
    # 'house',
    # 'detached',
    # 'semidetached_house',
    # 'terrace',
}

# Per-km² breakpoints for the 3-way rural / suburban / urban split.
# Tuned to Swiss settlement patterns: 500 ≈ scattered village,
# 5000 ≈ dense town / small-city outskirt.
STRATUM_THRESHOLDS: tuple[float, float] = (500.0, 5_000.0)

# STATPOP filename pattern (BFS public release; both .csv and .gpkg exist,
# we read the .csv).
_STATPOP_CSV_PATTERN = 'switzerland/statpop/statistik-bevoelkerung_haushalte_{year}_ha_2056.csv'


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

# 6 age bands. Each entry maps to the STATPOP sub-band codes (2-digit
# suffix) that compose it; M and W gender columns are summed within band.
_AGE_BANDS: dict[str, list[str]] = {
    '19minus': ['01', '02', '03', '04'],   # 0-4, 5-9, 10-14, 15-19
    '20to34':  ['05', '06', '07'],         # 20-24, 25-29, 30-34
    '35to49':  ['08', '09', '10'],
    '50to64':  ['11', '12', '13'],
    '65to79':  ['14', '15', '16'],
    '80plus':  ['17', '18', '19'],
}

variants = Variants([('year', str), ('buildings_country', str)])
variants.add(name='statpop_2025', year='2025', buildings_country='switzerland')


def _load_and_aggregate_statpop(
    csv_path: str, rng: np.random.Generator,
) -> pd.DataFrame:
    """Load the STATPOP CSV, apply privacy uncapping, aggregate to
    `population_total` + the 6 age bands.

    Returns a DataFrame indexed by `(E_KOORD, N_KOORD)` with the 7
    population columns. Cells outside Switzerland are not in the CSV
    by design. The privacy cap is auto-detected from the total column
    (BFS has changed it between releases).
    """
    coords = ['E_KOORD', 'N_KOORD']
    df = pd.read_csv(csv_path, sep=';').set_index(coords)
    logging.info(f"  → loaded {len(df):,} STATPOP hectare rows, {len(df.columns)} columns")

    # BFS columns: `BBTOT` (Bevölkerung Total), `BBM<code>` (Männer,
    # 2-digit age band), `BBW<code>` (Weiblich, 2-digit age band).
    total_col = 'BBTOT'
    if total_col not in df.columns:
        raise KeyError(
            f"Expected STATPOP total column {total_col!r} not found; "
            f"available columns (first 15): "
            f"{sorted(df.columns)[:15]!r}. Column naming may have changed "
            f"— verify BFS's schema.")

    total_arr = df[total_col].to_numpy(copy=True)
    cap = _detect_privacy_cap(total_arr)
    logging.info(f"  → detected STATPOP privacy cap = {cap}")
    logging.info(
        f"  → sum of raw BBTOT (with caps in place): {total_arr.sum():,.0f}")

    df['population_total'] = replace_smaller_equal(
        rng, total_arr, cap).astype(float)

    # Each age band: sum M + W sub-band columns after privacy uncapping.
    # Per-sub-band values are much smaller than the total, so the
    # cap-fraction is proportionally larger — treat age-band figures as
    # noisier than `population_total`.
    for band, codes in _AGE_BANDS.items():
        col_names = [f'BBM{c}' for c in codes] + [f'BBW{c}' for c in codes]
        col_names = [c for c in col_names if c in df.columns]
        if not col_names:
            df[f'population_{band}'] = 0.0
            continue
        v = df[col_names].to_numpy(copy=True)
        v = replace_smaller_equal(rng, v, cap)
        df[f'population_{band}'] = v.sum(axis=1).astype(float)

    keep = ['population_total'] + [f'population_{b}' for b in _AGE_BANDS]
    out = df[keep].copy()
    logging.info(
        f"  → aggregated; national total population: "
        f"{out['population_total'].sum():,.0f}; by age band: " +
        ", ".join(f"{b}={out[f'population_{b}'].sum():,.0f}"
                  for b in _AGE_BANDS))
    return out


def main(variant) -> None:
    context = init_context(variant)
    rng = np.random.default_rng(42)

    csv_path = context.raw_path(
        Storage.PUBLIC, _STATPOP_CSV_PATTERN.format(year=variant.year))

    with step(f'parse + uncap + aggregate STATPOP ({variant.year})'):
        stat_df = _load_and_aggregate_statpop(csv_path, rng)

    with step('build hectare cell polygons (100 m squares, LV95)'):
        cells = build_hectare_cells(stat_df)
        cell_totals = stat_df.copy()
        cell_totals.index = cells['cell_id'].values
        cell_totals.index.name = 'cell_id'

    with step('persist STATPOP hectare cells + per-hectare values as generic artifacts'):
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
        # Both files anchored on the source name (statpop) for symmetry
        # and provenance; the CSV's column names carry the semantic info.
        context.create_generic(out_cells, f'hectares_statpop_{variant.year}.gpkg')
        context.create_generic(
            out_totals, f'hectares_statpop_{variant.year}.csv',
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

    with step('classify buildings by local population density (1 km ring)'):
        buildings['pop_density_1km'] = compute_hectare_density(
            buildings, cells, cell_totals,
            column='population_total', radius_m=1_000.0,
        )
        buildings['stratum'] = classify_density_stratum(
            buildings['pop_density_1km'], STRATUM_THRESHOLDS)
        counts = buildings['stratum'].value_counts()
        logging.info(
            f"  → stratum counts: " +
            ", ".join(f"{s}={counts.get(s, 0):,}"
                      for s in ('rural', 'suburban', 'urban')))

    with step(f'stratify building tags (split_tags={sorted(_SPLIT_TAGS)!r})'):
        stratify_building_tags(buildings, _SPLIT_TAGS)
        effective_tags = expand_relevant_tags(
            dasymetric.POPULATION_TAGS, _SPLIT_TAGS)
        logging.info(
            f"  → {len(dasymetric.POPULATION_TAGS)} base tags "
            f"→ {len(effective_tags)} effective tags (split × strata)")

    # One NNLS fit + allocation per category. Total + per-age-band all
    # share `effective_tags` (POPULATION_TAGS with stratum expansion).
    categories = ['total'] + list(_AGE_BANDS)
    learned_per_category: dict[str, 'pd.DataFrame'] = {}
    per_building_outputs: dict[str, 'gpd.GeoDataFrame'] = {}
    for cat in categories:
        col = f'population_{cat}'
        with step(f'{col}: learn NNLS coefficients'):
            learned_per_category[col] = dasymetric.learn_coefficients(
                buildings, cells, cell_totals,
                cell_id_col='cell_id', column=col,
                relevant_tags=effective_tags,
                nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
            )
        with step(f'{col}: allocate to buildings'):
            tag_intensities = {
                t: float(i) for t, i in
                learned_per_category[col][f'intensity_{col}'].dropna().items()
            }
            coeffs = {t: tag_intensities.get(t, 1.0) for t in effective_tags}
            per_building_outputs[col] = dasymetric.per_building(
                buildings, cells, cell_totals,
                cell_id_col='cell_id', column=col,
                coeffs=coeffs,
                nearest_fallback_max_m=_NEAREST_FALLBACK_MAX_M,
            )

    with step('merge per-category outputs'):
        out = merge_per_category_outputs(buildings, per_building_outputs)
        band_sum = sum(out[f'population_{b}'].sum() for b in _AGE_BANDS)
        logging.info(
            f"  → distributed {out['population_total'].sum():,.0f} residents "
            f"across {len(out):,} buildings | " +
            ", ".join(f"{b}={out[f'population_{b}'].sum():,.0f}"
                      for b in _AGE_BANDS) +
            f" | Σbands={band_sum:,.0f}")

    with step('combine per-category learned coefficients into one wide-form table'):
        learned = combine_learned_per_category(learned_per_category)

    data_name = f'population_{variant.year}'
    keep_cols = (['population_total'] + [f'population_{b}' for b in _AGE_BANDS])
    props = out[keep_cols].copy()

    with step('save per-building population + calibrated coefficients'):
        context.create_properties(props, data_name=data_name)
        # Calibrated per-(OSM tag, age band) intensities — namespace-scoped
        # coef. Symmetric with STATENT's `create_coefs` output; consumed
        # via `get_coefs` if any downstream ever needs it. Lives at
        # `preparation/switzerland/land_use/coefs/<data_name>.csv`.
        context.create_coefs(learned, data_name)

    context.close()


if __name__ == '__main__':
    variants.run(main)
