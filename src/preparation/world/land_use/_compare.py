"""
Sanity-check the alternative population/employment sources (GHS-POP for
population, coefficient-based for employment) against the direct hectare-
based Swiss ground truth (STATPOP / STATENT).

Builds an 8-column per-building DataFrame for buildings inside CH:

    pop_statpop             pop_ghs
    emp_primary_statent     emp_primary_coef
    emp_secondary_statent   emp_secondary_coef
    emp_tertiary_statent    emp_tertiary_coef

Reports per-column fraction-non-zero, then per-pair correlation + bias
at three spatial scales: per-building, H3 res 10 (avg edge 66 m, area
0.015 km²), H3 res 8 (avg edge 461 m, area 0.74 km²). Correlation
typically improves with aggregation (smoothing out per-building noise
from the dasymetric assumption).

Requires `h3` (`pip install h3`). Reads property CSVs and the buildings
GPKG directly to avoid the dependency-tracking machinery.

Run:
    cd src && python -m preparation.world.land_use._compare
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta_atlas.context import init_context, Storage
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.switzerland.common import filter_to_inside_ch, load_swiss_country


logging.basicConfig(level=logging.INFO, format='%(message)s')


variants = Variants([
    ('area_name', str),
    ('statpop_year', str),
    ('statent_year', str),
])
variants.add(
    name='switzerland_compare',
    area_name='switzerland',
    statpop_year='2021',
    statent_year='2020',
)


def _read_csv(context, namespace: str, rel_path: str) -> pd.DataFrame:
    src = context.source(namespace)
    path = src.path_for(Storage.PUBLIC, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)


def _read_gpkg(context, namespace: str, rel_path: str) -> gpd.GeoDataFrame:
    src = context.source(namespace)
    path = src.path_for(Storage.PUBLIC, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return gpd.read_file(path)


def main(variant) -> None:
    context = init_context(variant)

    with step('load OSM buildings + filter to inside CH borders'):
        # Universe = ALL CH buildings. Each source's per-building values
        # get reindexed onto this universe; missing entries → 0. This is
        # the right comparison: both methods must produce sensible
        # totals/values across the same set of buildings.
        buildings = _read_gpkg(
            context, 'preparation/world/osm',
            f'shapes/buildings_{variant.area_name}.gpkg')
        if 'building_id' in buildings.columns:
            buildings = buildings.set_index('building_id')
        country = load_swiss_country(context)
        buildings = filter_to_inside_ch(buildings, country)
        ch_idx = buildings.index
        logging.info(f"  → {len(buildings):,} CH buildings (universe)")

    with step('load STATPOP per-building population'):
        statpop = _read_csv(
            context, 'preparation/switzerland/land_use',
            f'properties/buildings_population_{variant.statpop_year}.csv')

    with step('load GHS-POP per-building population'):
        ghs = _read_csv(
            context, 'preparation/world/land_use',
            f'properties/buildings_population_{variant.area_name}.csv')

    with step('load STATENT per-building employment'):
        statent = _read_csv(
            context, 'preparation/switzerland/land_use',
            f'properties/buildings_employment_statent_{variant.statent_year}.csv')

    with step('load coef-based per-building employment'):
        coef = _read_csv(
            context, 'preparation/world/land_use',
            f'properties/buildings_employment_{variant.area_name}.csv')

    with step('build 8-column comparison frame (universe = all CH buildings)'):
        df = pd.DataFrame(index=ch_idx)
        df['pop_statpop'] = (
            statpop['population_total'].reindex(ch_idx).fillna(0.0).astype(float))
        df['pop_ghs'] = (
            ghs['population_total'].reindex(ch_idx).fillna(0.0).astype(float))
        for sector in ('primary', 'secondary', 'tertiary'):
            df[f'emp_{sector}_statent'] = (
                statent[f'employment_{sector}']
                .reindex(ch_idx).fillna(0.0).astype(float))
            df[f'emp_{sector}_coef'] = (
                coef[f'employment_{sector}']
                .reindex(ch_idx).fillna(0.0).astype(float))
        logging.info(f"  → {len(df):,} buildings in comparison frame")

    # ---- Per-column non-zero fractions + sums ----
    print()
    print("=" * 90)
    print(f"Per-column non-zero fraction & sum (n = {len(df):,} buildings)")
    print("=" * 90)
    for col in df.columns:
        n_nz = int((df[col] > 0).sum())
        frac = 100.0 * n_nz / len(df) if len(df) else 0.0
        tot = float(df[col].sum())
        print(f"  {col:>26}: {n_nz:>9,} non-zero ({frac:5.1f}%) | sum = {tot:>15,.0f}")

    pairs = [
        ('population',           'pop_statpop',           'pop_ghs'),
        ('primary employment',   'emp_primary_statent',   'emp_primary_coef'),
        ('secondary employment', 'emp_secondary_statent', 'emp_secondary_coef'),
        ('tertiary employment',  'emp_tertiary_statent',  'emp_tertiary_coef'),
    ]

    def _report_pair_stats(frame: pd.DataFrame, scale_label: str) -> None:
        print()
        print("-" * 90)
        print(f"{scale_label}  (n = {len(frame):,})")
        print("-" * 90)
        print(f"  {'pair':>22}: {'corr':>8}  | {'sum_truth':>15}  {'sum_alt':>15}  {'bias':>8}")
        for name, col_truth, col_alt in pairs:
            corr = float(frame[col_truth].corr(frame[col_alt]))
            sum_t = float(frame[col_truth].sum())
            sum_a = float(frame[col_alt].sum())
            bias_pct = 100 * (sum_a - sum_t) / sum_t if sum_t > 0 else float('nan')
            print(
                f"  {name:>22}: {corr:>+8.4f}  | "
                f"{sum_t:>15,.0f}  {sum_a:>15,.0f}  {bias_pct:>+7.1f}%")

    _report_pair_stats(df, "Per-building correlation + bias")

    # ---- H3 aggregation ----
    try:
        import h3
    except ImportError:
        print()
        print("WARNING: `h3` not installed; skipping H3 aggregation.")
        print("         Install with: pip install h3")
        context.close()
        return

    with step('attach H3 indexes (centroids → lat/lon)'):
        # Reuse the already-loaded CH-filtered buildings; reproject
        # centroids to WGS84 for h3.latlng_to_cell.
        b_ch = buildings.to_crs('EPSG:4326')
        cents = b_ch.geometry.centroid
        lats = cents.y.to_numpy()
        lons = cents.x.to_numpy()
        logging.info(
            f"  → computing H3 cells for {len(b_ch):,} CH buildings...")
        h3_r10 = np.array(
            [h3.latlng_to_cell(lat, lon, 10) for lat, lon in zip(lats, lons)])
        h3_r8 = np.array(
            [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(lats, lons)])
        df_with_h3 = df.copy()
        df_with_h3['h3_r10'] = pd.Series(h3_r10, index=b_ch.index).reindex(df.index)
        df_with_h3['h3_r8'] = pd.Series(h3_r8, index=b_ch.index).reindex(df.index)
        df_with_h3 = df_with_h3.dropna(subset=['h3_r10', 'h3_r8'])

    value_cols = [c for c in df.columns]
    agg10 = df_with_h3.groupby('h3_r10')[value_cols].sum()
    agg8 = df_with_h3.groupby('h3_r8')[value_cols].sum()

    _report_pair_stats(agg10, "Aggregated to H3 res 10 (avg 66 m edge, 0.015 km²)")
    _report_pair_stats(agg8, "Aggregated to H3 res 8 (avg 461 m edge, 0.74 km²)")

    print()
    context.close()


if __name__ == '__main__':
    variants.run(main)
