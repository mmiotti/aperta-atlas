"""
Shared helpers for `preparation/switzerland/` scripts (analogous to
`preparation/world/common.py`).

Covers BFS-specific conventions shared across STATENT (employment) and
STATPOP (population) — and any future BFS-derived orchestrators:

  - **Hectare-cell polygons** (`build_hectare_cells`): both releases
    publish per-100m-cell totals indexed by `(E_KOORD, N_KOORD)`
    lower-left-corner LV95 coordinates; downstream dasymetric mapping
    consumes these as a polygon GeoDataFrame.

  - **Privacy-suppression uncap** (`replace_smaller_equal`): BFS caps
    small per-cell counts (≤ 3 for STATPOP, ≤ 4 for STATENT) as the
    literal cap value. The helper redraws the capped values from a
    weighted distribution that approximates the true tail.

  - **Per-category output merge helpers** (`merge_per_category_outputs`,
    `combine_learned_per_category`): glue for orchestrators that run
    one `dasymetric.per_building` / `learn_coefficients` pass per
    category (sector, age band, …) and need to stitch the per-category
    results back into a single per-building GeoDataFrame and a single
    wide-form calibrated-coefficients table.

  - **Swiss country polygon** (`load_swiss_country`, `filter_to_inside_ch`):
    load the country polygon from political_boundaries.py's output and
    filter a buildings GeoDataFrame to those inside CH borders. Used by
    dasymetric orchestrators so the calibration area is the country
    (not the cells subset), which is required for β to extrapolate
    correctly when applied bottom-up.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from aperta import geo_processing
from aperta_atlas.context import Storage


CRS_CH = 'EPSG:2056'           # Swiss metric CRS (LV95 / CH1903+) —
                               # what BFS publishes in for STATPOP /
                               # STATENT / NPVM / etc. The canonical
                               # in-Switzerland CRS; doesn't change.
CRS_LATLON = 'EPSG:4326'       # WGS84 lat/lon — for external APIs +
                               # OSM ingest before reprojection.
_CRS_LV95 = CRS_CH             # Internal alias kept for one in-file
                               # reference. Swiss metric CRS — what BFS publishes in
_HECTARE_SIZE_M = 100          # BFS-standard cell size (100m × 100m)


def replace_smaller_equal(
    rng: np.random.Generator, a: np.ndarray, n: int,
) -> np.ndarray:
    """Redraw privacy-capped values in `a` (integers 1..n) from a
    weighted distribution to approximate the unobserved true
    distribution.

    BFS caps any per-cell count <= n as the literal cap value (n=3 for
    STATPOP, n=4 for STATENT) to prevent re-identification of small
    populations. The weights below are tuned so that redrawn column
    sums roughly match uncapped national totals reported elsewhere by
    BFS. Same RNG seed across calls → deterministic output for
    reproducibility.

    Mutates `a` in place AND returns it (convenience for chaining).

    Args:
        rng: a numpy `Generator` seeded by the caller (use a fixed seed
            in orchestrators for reproducibility).
        a: integer-valued numpy array. Entries with `1 <= a <= n` are
            redrawn; other entries are left alone.
        n: cap value. Supported: 3 (STATPOP) and 4 (STATENT). For
            other `n`, redraws uniformly (no informative weights).
    """
    f = (1 <= a) & (a <= n)
    if n == 3:
        p = [0.65, 0.25, 0.10]
    elif n == 4:
        p = [0.45, 0.25, 0.20, 0.10]
    else:
        p = None
    to_replace = rng.choice(list(range(1, n + 1)), f.sum(), p=p)
    a[f] = to_replace
    return a


def build_hectare_cells(stat_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Construct 100m × 100m LV95 hectare-cell polygons from a BFS-style
    DataFrame indexed on `(E_KOORD, N_KOORD)` lower-left-corner
    coordinates.

    Both STATENT (employment) and STATPOP (population) publish their
    hectare-level data with this index convention. `cell_id` is a
    positional integer assigned at construction — gives a compact,
    fast-to-join key for the downstream dasymetric step.

    Args:
        stat_df: DataFrame whose index is a 2-level MultiIndex of LV95
            (E_KOORD, N_KOORD) coordinates (lower-left corner per BFS
            convention). Levels can be int or float.

    Returns:
        GeoDataFrame with `cell_id` (0..N-1) + `geometry` (square Polygon
        in LV95), one row per `stat_df` row in input order.
    """
    e = stat_df.index.get_level_values(0).to_numpy(dtype=float)
    n = stat_df.index.get_level_values(1).to_numpy(dtype=float)
    polys = [box(ex, ny, ex + _HECTARE_SIZE_M, ny + _HECTARE_SIZE_M)
             for ex, ny in zip(e, n)]
    return gpd.GeoDataFrame(
        {'cell_id': np.arange(len(stat_df))},
        geometry=polys,
        crs=_CRS_LV95,
    )


def merge_per_category_outputs(
    buildings: gpd.GeoDataFrame,
    per_category_outputs: dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Merge per-category `dasymetric.per_building` outputs into one
    GeoDataFrame.

    `per_category_outputs` is `{column_name: per_building_output_for_that_column}`.
    The merged output has one row per building that's in coverage in AT
    LEAST ONE category (union of in-coverage indices) and one numeric
    column per entry in `per_category_outputs`. Buildings missing from
    a category's output (not in its `relevant_tags` or not in coverage)
    get 0.0 for that column.

    Buildings' original order is preserved.
    """
    in_coverage: set = set()
    for out_col in per_category_outputs.values():
        in_coverage.update(out_col.index)
    merged_idx = [i for i in buildings.index if i in in_coverage]
    out = buildings.loc[merged_idx].copy()
    for col, out_col in per_category_outputs.items():
        if col not in out_col.columns:
            raise ValueError(
                f"per-category output for {col!r} is missing the column "
                f"{col!r}. Did you pass the right (key, value) pairing?")
        out[col] = out_col[col].reindex(out.index).fillna(0.0)
    return out


def combine_learned_per_category(
    learned_per_category: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-category `dasymetric.learn_coefficients` outputs into
    one wide-form DataFrame for the published calibrated-coefficients CSV.

    Outer-join on OSM tag index. Per-category columns
    (`intensity_<col>`, `_p05`, `_p95`, `contribution_<col>`) are preserved
    with their suffixes — they naturally disambiguate. Tags appear with
    NaN in categories they weren't part of. `n_observations` is taken as
    the max across categories (a tag's building count is invariant; if
    it differs it's because some categories filtered area=0 differently).

    Args:
        learned_per_category: `{column_name: learn_coefficients_output_for_that_column}`.

    Returns:
        Wide-form DataFrame indexed by OSM tag with all per-category
        columns + a single `n_observations` column.
    """
    if not learned_per_category:
        return pd.DataFrame()
    merged: pd.DataFrame | None = None
    n_obs_series: pd.Series | None = None
    for col, df in learned_per_category.items():
        if df.empty:
            continue
        non_obs_cols = [c for c in df.columns if c != 'n_observations']
        slice_df = df[non_obs_cols]
        merged = slice_df if merged is None else merged.join(slice_df, how='outer')
        obs_here = df['n_observations'] if 'n_observations' in df.columns else None
        if obs_here is not None:
            n_obs_series = (
                obs_here if n_obs_series is None
                else n_obs_series.combine(obs_here, max, fill_value=0))
    if merged is None:
        return pd.DataFrame()
    if n_obs_series is not None:
        merged = merged.join(n_obs_series.rename('n_observations'), how='left')
        merged['n_observations'] = merged['n_observations'].fillna(0).astype(int)
    return merged.sort_index()


def load_swiss_country(context) -> gpd.GeoDataFrame:
    """Load the Swiss country polygon (single row) from
    `political_boundaries.py`'s output, filtered to country_id='Schweiz'.

    The source file `countries.gpkg` contains polygons for Switzerland
    PLUS slivers of the neighbouring countries that appear in the
    TLM_LANDESGEBIET layer at the borders. Without the filter, the
    centroid-within join in `filter_to_inside_ch` would match against
    any of those polygons — wrong, and empirically buggy.

    Preparation-namespace generic files live at the namespace root
    (no `generic/` subfolder).
    """
    gen_ctx = context.source('preparation/switzerland/general')
    path = gen_ctx.path_for(Storage.PUBLIC, 'countries.gpkg')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Swiss country polygon not found at {path}. Run "
            f"`preparation/switzerland/general/political_boundaries.py` first.")
    countries = gpd.read_file(path)
    if 'country_id' not in countries.columns:
        raise ValueError(
            f"`countries.gpkg` at {path} is missing the `country_id` "
            f"column. Got: {list(countries.columns)}.")
    schweiz = countries[countries['country_id'] == 'Schweiz']
    if not len(schweiz):
        raise ValueError(
            f"No row with country_id='Schweiz' in {path}. Available: "
            f"{countries['country_id'].tolist()}")
    return schweiz.reset_index(drop=True)


def filter_to_inside_ch(
    buildings: gpd.GeoDataFrame,
    country: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter `buildings` to those whose centroid is inside the Swiss
    country polygon. Preserves the input's CRS and index.

    Centroid-within is fast (O(n log n) via R-tree) and accurate enough
    for building footprints: only buildings exactly straddling the CH
    border are misclassified, which is negligible at country scale.
    """
    import logging

    if country.crs is None or buildings.crs is None:
        raise ValueError("Both `country` and `buildings` must have a CRS.")
    # Defensive: log country-polygon stats. Surfaces issues like wrong
    # geometry type (LineString instead of Polygon), tiny test polygon,
    # invalid geometries, etc. — all of which cause the sjoin to silently
    # return ~0 matches.
    geom_types = country.geometry.geom_type.unique().tolist()
    total_area_km2 = float(country.geometry.to_crs('EPSG:2056').area.sum()) / 1e6
    n_invalid = int((~country.geometry.is_valid).sum())
    logging.info(
        f"  → country polygon: {len(country)} row(s), CRS={country.crs}, "
        f"types={geom_types}, area={total_area_km2:,.0f} km², "
        f"invalid={n_invalid}")

    b_for_join = buildings if buildings.crs == country.crs else buildings.to_crs(country.crs)
    # `.values` strips the buildings' index from the centroid GeoSeries
    # so it aligns positionally with `_b_idx` (a fresh RangeIndex).
    # Without this, the GeoDataFrame constructor tries to align the
    # named-index centroids against the dict's RangeIndex and almost
    # all rows end up as NaN — sjoin then matches almost nothing.
    centroids = gpd.GeoDataFrame(
        {'_b_idx': np.arange(len(b_for_join))},
        geometry=b_for_join.geometry.centroid.values,
        crs=b_for_join.crs,
    )
    joined = gpd.sjoin(
        centroids, country[['geometry']],
        how='inner', predicate='within',
    )
    in_ch_idx = joined['_b_idx'].drop_duplicates().values
    return buildings.iloc[in_ch_idx].copy()


# ---------------------------------------------------------------------------
# Density-based building stratification (used by STATPOP + STATENT dasymetric
# calibration when a tag's intensity varies meaningfully with local density).
# ---------------------------------------------------------------------------


def compute_hectare_density(
    buildings: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    cell_totals: pd.DataFrame,
    column: str,
    radius_m: float = 1_000.0,
) -> pd.Series:
    """Per-building density of `column` in a `radius_m` ring, sourced
    from the hectare grid.

    Wraps `aperta.geo_processing.cross_sum_within_radius` with the
    conventions the two dasymetric-calibration scripts share:
    building centroids are the targets, hectare centroids are the
    sources, and the return is density (sum / π·r²) — i.e. the source
    unit per m². Multiply by 1e6 for the more intuitive per-km² units.

    Args:
        buildings: polygon GDF with a metric CRS.
        cells: hectare polygons (from `build_hectare_cells`), with
            `cell_id`; must share `buildings.crs`.
        cell_totals: DataFrame indexed by cell_id including `column`.
        column: hectare value column to sum (e.g. `'population_total'`).
        radius_m: ring radius in metres. Default 1 km (Swiss walkability
            + local-context envelope).

    Returns:
        pd.Series of density values (per m²), indexed by
        `buildings.index`, aligned name = `<column>_density_<radius>m`.
    """
    hex_pts = cells.copy()
    hex_pts.geometry = cells.geometry.centroid
    hex_pts = hex_pts.join(cell_totals[column], on='cell_id')
    return geo_processing.cross_sum_within_radius(
        targets=buildings.set_geometry(buildings.geometry.centroid),
        sources=hex_pts,
        radius=radius_m,
        weight_column=column,
        return_density=True,
        name=f'{column}_density_{int(radius_m)}m',
    )


def classify_density_stratum(
    density_per_m2: pd.Series,
    thresholds_per_km2: tuple[float, float] = (500.0, 5_000.0),
    labels: tuple[str, str, str] = ('rural', 'suburban', 'urban'),
) -> pd.Series:
    """Bucket per-m² densities into three ordered strata (default:
    rural / suburban / urban) using per-km² breakpoints.

    Default thresholds (500 / 5000 per km²) are calibrated to Swiss
    settlement patterns — 500 ≈ scattered village, 5000 ≈ dense town /
    small-city outskirt. Tune for other geographies.

    Returns a Series of stratum labels (dtype str) indexed by the
    input's index. NaN-safe: NaN densities → labels[0].
    """
    v_per_km2 = density_per_m2.fillna(0.0) * 1e6
    bins = [-np.inf, *thresholds_per_km2, np.inf]
    return pd.cut(v_per_km2, bins=bins, labels=list(labels)).astype(str)


def stratify_building_tags(
    buildings: gpd.GeoDataFrame,
    split_tags: set[str],
    stratum_col: str = 'stratum',
    tag_col: str = 'building',
) -> gpd.GeoDataFrame:
    """Rewrite `buildings[tag_col]` in-place for rows whose tag value
    is in `split_tags`: append `__<stratum>` from `buildings[stratum_col]`.
    Non-matching rows are untouched.

    Call ONCE per pipeline (typically right after
    `classify_density_stratum`). Downstream callers can compose the
    effective tag set for their own `relevant_tags` argument via
    `expand_relevant_tags`.

    Returns the same GeoDataFrame (mutated) for chaining.
    """
    if not split_tags:
        return buildings
    if stratum_col not in buildings.columns:
        raise KeyError(
            f"Building stratification needs a {stratum_col!r} column "
            f"on `buildings` — call `classify_density_stratum` first.")
    is_split = buildings[tag_col].isin(split_tags)
    buildings.loc[is_split, tag_col] = (
        buildings.loc[is_split, tag_col]
        + '__' + buildings.loc[is_split, stratum_col])
    return buildings


def derive_stratification_from_tags(
    tag_set,
    separator: str = '__',
) -> tuple[set[str], list[str]]:
    """Inverse of `expand_relevant_tags`: parse `<base><sep><stratum>`
    entries from a calibrated coefs table's tag set and return
    `(split_tag_bases, strata_labels)`. Non-suffixed entries are
    ignored (they were non-split tags).

    Lets downstream consumers detect whether a coefs file was
    stratum-expanded without needing shared state with the producer.

    Args:
        tag_set: any iterable of str (frozenset, set, pd.Index).
        separator: the stratum separator used by
            `stratify_building_tags` / `expand_relevant_tags`. Default
            `'__'` matches those helpers' convention.
    """
    split = set()
    strata = set()
    for tag in tag_set:
        s = str(tag)
        if separator in s:
            base, _, stratum = s.rpartition(separator)
            split.add(base)
            strata.add(stratum)
    return split, sorted(strata)


def expand_relevant_tags(
    base_tags: frozenset[str] | set[str],
    split_tags: set[str],
    strata: list[str] | tuple[str, ...] = ('rural', 'suburban', 'urban'),
) -> frozenset[str]:
    """Return the effective relevant-tag set for
    `dasymetric.learn_coefficients` / `.per_building`, replacing each
    tag in `base_tags ∩ split_tags` with its per-stratum variants
    (`<tag>__<stratum>` for each entry in `strata`).

    Symmetric with `stratify_building_tags` — call this to build the
    `relevant_tags` argument; call `stratify_building_tags` (once)
    to prepare the matching building tag values. Together they let
    NNLS learn a separate intensity per stratum per split tag without
    any library-side changes.
    """
    effective = set(base_tags)
    for tag in split_tags:
        if tag in effective:
            effective.remove(tag)
            for s in strata:
                effective.add(f'{tag}__{s}')
    return frozenset(effective)
