"""
Dasymetric mapping for distributing per-cell totals (population, jobs,
…) across building footprints. Pure, country-agnostic algorithms;
shape-agnostic over cells (hectare squares, municipalities, H3, …).
Country-specific orchestrators in `preparation/<country>/land_use/`
handle data acquisition + schema cleanup, then call these.

**Three primitives**:

  - `learn_coefficients` — NNLS regression on cell-level totals →
    per-OSM-tag per-m² intensities (`β`). One regression per category
    (sector, age band, …). Cells with no overlapping relevant-tag
    building can be rescued via nearest-fallback synthetic rows so the
    learned β matches what allocation does downstream. Logs a 3-tier
    coverage diagnostic (overlay / fallback / unreachable).

  - `per_building` — top-down distribution of per-cell totals to
    buildings via overlay + per-cell rescale (cell totals matched
    exactly) + nearest-building fallback. The hectare/admin case is
    the same code path: km-sized cells degenerate to "each building
    belongs to one cell" and rescale matches per-cell totals exactly;
    100 m cells get buildings straddling multiple cells split by
    overlap area.

  - `per_building_from_coefficients` — bottom-up `β × area` for the
    world-side path (any region with OSM but no cell-level data).
    No per-cell rescaling — there's nothing to anchor to.

Plus `apply_proportional_split` for sub-category expansion (e.g.
10-industry detail from 3-sector dasymetric output).

The membership sets `EMPLOYMENT_TAGS_PER_SECTOR` + `POPULATION_TAGS`
live here as defaults tuned for Swiss STATENT / STATPOP — override in
your orchestrator if your country tags differently. Per-category
output merging / CSV layout lives outside this module in
`preparation/switzerland/common.py`.
"""

import logging
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from aperta_atlas.stats_helpers import warn_high_collinearity


# Per-category MEMBERSHIP sets — default OSM tags eligible per category.
# Within a category, intensities are LEARNED from the data via
# `learn_coefficients`. Mixed-use tags (e.g. `commercial`, `service`)
# appear in multiple categories' sets and get a separate learned
# intensity per category. Buildings with tags outside ALL relevant sets
# are dropped entirely (no fallback to silos / garages / sheds).
#
# **`yes` trade-off (primary/secondary)**. OSM tags farms / industrial
# buildings inconsistently — many real farms are tagged `yes` rather
# than `farm`. Including `yes` in primary/secondary memberships:
#   + STATENT-direct can allocate ~all of the national target (cells
#     with yes-tagged farms get reached).
#   + Aggregate totals match between STATENT-direct and coef-based.
#   - Per-building signal dilutes: β × area applies to ALL yes buildings
#     (residential included), so coef-based spreads employment thinly
#     across the city — poor correlation with STATENT-direct's
#     concentrated per-cell-rescaled output.
# Excluding `yes` (current choice):
#   + Strong per-building signal: only tags that *typically* host
#     primary/secondary employment carry β. Coef-based and STATENT-direct
#     concentrate on the same buildings → higher correlation.
#   - STATENT-direct loses ~60% of primary / ~30% of secondary target
#     (cells with only yes-tagged farms can't be reached within fallback
#     range) — bias-vs-direct looks artificially huge but actually
#     reflects STATENT-direct's under-attribution, not coef over-prediction.
# Bias rescaling in `learn_coefficients` makes coef-based reproduce
# `cal_target_sum` either way; the choice affects per-building precision
# vs Swiss-side allocation completeness.

EMPLOYMENT_TAGS_PER_SECTOR: dict[str, frozenset[str]] = {
    # Primary sector — agriculture & forestry buildings.
    # NOTE: without including 'yes' (generic buildings), we miss quite a lot of
    # primary sector in the allocation. WITH including it, the coefficients
    # will allocate a small amount of primary sector employment
    # to each building
    'primary': frozenset({
        'farm', 'farm_auxiliary', 'barn', 'stable', 'cowshed', 'sty',
        'greenhouse', 'yes',
    }),
    # Secondary sector — industry, manufacturing, construction; plus
    # mixed-use tags that often host workshops / light manufacturing.
    'secondary': frozenset({
        'industrial', 'factory', 'manufacture', 'warehouse',
        # `commercial` and `service` are mixed-use → also in tertiary.
        'commercial', 'service', 'yes',
    }),
    # Tertiary sector — offices, retail, hospitality, education, health,
    # civic/public, plus mixed-use residential. Excluded:
    # `clinic` and `religious` (NNLS gave negative/near-zero β on Swiss
    # STATENT — clinics are typically tagged `hospital`, religious sites
    # rarely employ); `house`, `detached`, `terrace`, `semidetached_house`,
    # `bungalow` (NNLS β was negative — pure residential, no home-business
    # signal above noise).
    'tertiary': frozenset({
        'office', 'retail', 'supermarket', 'kiosk', 'hospital', 'school',
        'university', 'college', 'hotel', 'civic', 'public', 'government',
        'museum', 'chapel', 'church', 'commercial', 'service',
        # Mixed-use residential — apartments + dormitory show enough
        # home-business / mixed-use employment signal to keep; the
        # detached-family-home tags above don't.
        'apartments', 'residential', 'yes',
    }),
}

POPULATION_TAGS: frozenset[str] = frozenset({
    'apartments', 'residential', 'dormitory', 'house', 'detached',
    'terrace', 'semidetached_house', 'farm', 'yes',
})
# Excluded from POPULATION_TAGS: `bungalow`, `cabin`, `hut` — NNLS gave
# negative / near-zero β on Swiss STATPOP (bungalow rare in CH; cabin
# and hut are typically seasonal/recreational with no permanent residents).


def _nearest_relevant_building(
    cells: gpd.GeoDataFrame,
    unmatched_cell_ids: Sequence | np.ndarray | set,
    relevant_buildings: gpd.GeoDataFrame,
    *,
    cell_id_col: str,
    max_distance: float,
    bring: Sequence[str],
) -> pd.DataFrame:
    """For each cell in `unmatched_cell_ids`, find the nearest building
    in `relevant_buildings` within `max_distance` of the cell centroid.
    Returns one row per (cell, nearest building) pair; cells with no
    in-range neighbor are dropped. `bring` lists which building columns
    to carry through (e.g. `['_b_idx']` for per_building's position
    lookup, `[tag_col, area_col]` for learn_coefficients's synthetic-row
    construction). Shared between per_building (allocation) and
    learn_coefficients (synthetic-row construction) so learning + allocation
    use the same fallback model.
    """
    bring = list(bring)
    unmatched = cells[cells[cell_id_col].isin(set(unmatched_cell_ids))]
    if not len(unmatched):
        return pd.DataFrame(columns=[cell_id_col, *bring])
    centroids = gpd.GeoDataFrame(
        {cell_id_col: unmatched[cell_id_col].values},
        geometry=unmatched.geometry.centroid.values,
        crs=cells.crs,
    )
    builds_lean = gpd.GeoDataFrame(
        {c: relevant_buildings[c].values for c in bring},
        geometry=relevant_buildings.geometry.values,
        crs=relevant_buildings.crs,
    )
    nearest = gpd.sjoin_nearest(
        centroids, builds_lean,
        how='left', max_distance=max_distance,
    ).drop(columns=['index_right'], errors='ignore')
    nearest = nearest.dropna(subset=bring).copy()
    return nearest[[cell_id_col, *bring]]


def per_building(
    buildings: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    cell_totals: pd.DataFrame,
    *,
    cell_id_col: str,
    column: str,
    coeffs: dict[str, float],
    building_tag_col: str = 'building',
    nearest_fallback_max_m: float | None = 200.0,
) -> gpd.GeoDataFrame:
    """Distribute per-cell totals (one column) across building footprints
    via overlay-based dasymetric mapping + nearest-building fallback.

    See module docstring for the 4-step method. Orchestrators call this
    once per category (sector, age band, …) with that category's
    per-tag learned intensities. Buildings outside `coeffs.keys()` are
    filtered out before any spatial work.

    Args:
        buildings: polygon GeoDataFrame with `building_tag_col`. Same CRS
            as `cells`. Must be metric so overlap areas are in m².
        cells: polygon GeoDataFrame with `cell_id_col`. Same CRS as
            `buildings`. Cells can be any polygon (hectare squares,
            municipalities, H3 indexes, …) — the algorithm is shape-agnostic.
        cell_totals: DataFrame indexed by cell_id values; must include
            `column` as a column. Values are per-cell totals to disaggregate.
        cell_id_col: cell identifier column.
        column: single output column name. Output GeoDataFrame carries
            one new column named identically.
        coeffs: `{tag: intensity}` per-tag per-m² coefficients (typically
            from `learn_coefficients`). Buildings whose tag isn't a key
            here are filtered out before overlay + fallback.
        building_tag_col: OSM tag column in `buildings`. Default `'building'`.
        nearest_fallback_max_m: cells with zero building overlap get their
            full count assigned to the nearest in-tag building within this
            distance (CRS units, expected m). `None` disables fallback.

    Returns:
        Copy of `buildings` filtered to in-coverage rows (= those that
        either overlapped a cell or were a nearest-fallback target),
        plus one new `column` holding the assigned value. Out-of-coverage
        buildings are dropped — they'd otherwise distort downstream
        calibration with value=0 + full area.
    """
    if buildings.crs is None or cells.crs is None:
        raise ValueError("Both `buildings` and `cells` must have a CRS set.")
    if buildings.crs != cells.crs:
        raise ValueError(
            f"CRS mismatch: buildings={buildings.crs}, cells={cells.crs}. "
            f"Reproject one before calling.")
    if column not in cell_totals.columns:
        raise ValueError(
            f"`cell_totals` is missing required column {column!r}. "
            f"Got {list(cell_totals.columns)}.")

    # Filter to relevant-tag buildings (tags the coefficient table covers).
    # Done before any spatial work so out-of-tag buildings can't sneak in
    # via the nearest-fallback path either.
    relevant_tags = set(coeffs.keys())
    n_before = len(buildings)
    buildings = buildings[buildings[building_tag_col].isin(relevant_tags)].copy()
    n_tag_matched = len(buildings)

    cell_targets = cell_totals[column].astype(float)

    # Positional building index — overlay drops the original index, so
    # carry `_b_idx` through to re-assemble.
    n_buildings = len(buildings)
    b_in = gpd.GeoDataFrame(
        {'_b_idx': np.arange(n_buildings),
         building_tag_col: buildings[building_tag_col].values},
        geometry=buildings.geometry.values,
        crs=buildings.crs,
    )
    c_in = cells[[cell_id_col, 'geometry']].copy()

    # 1. Overlay buildings × cells -> per-(building, cell) intersection.
    overlays = gpd.overlay(
        b_in, c_in, how='intersection', keep_geom_type=False,
    )
    if len(overlays):
        overlays['_overlap_area'] = overlays.geometry.area

    contribs = pd.Series(0.0, index=np.arange(n_buildings))

    if len(overlays):
        # 2. Per-overlay prediction = coef[tag] × overlap_area.
        coef_per_row = (
            overlays[building_tag_col].map(coeffs)
            .fillna(0.0).astype(float))   # .fillna defensive: filter above is authoritative
        overlays['_pred'] = coef_per_row * overlays['_overlap_area']

        # 3. Per-cell rescale: target / predicted_sum per cell.
        pred_sum = overlays.groupby(cell_id_col)['_pred'].transform('sum')
        target_per_row = (
            overlays[cell_id_col].map(cell_targets).astype(float))
        scale = (target_per_row / pred_sum).replace([np.inf, -np.inf], 0).fillna(0)

        # 4. Per-(building, cell) contribution → per-building aggregate.
        overlays['_contrib'] = overlays['_pred'] * scale
        per_b = overlays.groupby('_b_idx')['_contrib'].sum()
        contribs = contribs.add(per_b, fill_value=0.0)

    in_coverage_b_idx: set[int] = set()
    if len(overlays):
        in_coverage_b_idx.update(int(i) for i in overlays['_b_idx'].unique())

    # 5. Nearest-building fallback for cells with zero overlap → assign
    #    full target to the nearest in-tag building within max_m.
    rescued_cell_ids: set = set()
    overlapped_cell_ids = (
        set(overlays[cell_id_col].unique()) if len(overlays) else set())
    if nearest_fallback_max_m is not None and nearest_fallback_max_m > 0:
        unmatched_cell_ids = cells.loc[
            ~cells[cell_id_col].isin(overlapped_cell_ids), cell_id_col].values
        nearest = _nearest_relevant_building(
            cells, unmatched_cell_ids, b_in,
            cell_id_col=cell_id_col,
            max_distance=nearest_fallback_max_m,
            bring=['_b_idx'],
        )
        if len(nearest):
            nearest['_b_idx'] = nearest['_b_idx'].astype(int)
            nearest['_t'] = nearest[cell_id_col].map(cell_targets).astype(float)
            fallback_per_b = nearest.groupby('_b_idx')['_t'].sum()
            contribs = contribs.add(fallback_per_b, fill_value=0.0)
            in_coverage_b_idx.update(int(i) for i in nearest['_b_idx'].unique())
            rescued_cell_ids = set(nearest[cell_id_col].unique())

    # 6. In-coverage filter (= overlapped a cell OR rescued by fallback).
    # Out-of-coverage buildings are dropped (would distort downstream
    # calibration with value=0 + full area). Typical case: OSM buildings
    # in the buffer outside a country-scoped STATPOP/STATENT run.
    # `dtype=bool` is load-bearing: with `n_buildings == 0` the list
    # comprehension is empty and `np.array([])` defaults to float64,
    # which then raises `IndexError: arrays used as indices must be of
    # integer (or boolean) type` on line 324's fancy-indexing.
    in_coverage_mask = np.array(
        [i in in_coverage_b_idx for i in range(n_buildings)], dtype=bool)
    n_in_coverage = int(in_coverage_mask.sum())

    logging.info(
        f"  → alloc [{column}]: {n_before:,} buildings → "
        f"{n_tag_matched:,} with relevant tag → "
        f"{n_in_coverage:,} received value")

    # "% of target unassigned" — cells with target > 0, no overlap, no
    # fallback. For the typical case (same `nearest_fallback_max_m` as
    # learn) this matches `learn_coefficients`'s "% of target lost".
    total_t = float(cell_targets.sum())
    if total_t > 0:
        unassigned_cell_ids = (
            set(cell_targets.index[cell_targets > 0])
            - overlapped_cell_ids - rescued_cell_ids)
        if unassigned_cell_ids:
            lost_t = float(cell_targets.loc[list(unassigned_cell_ids)].sum())
            pct_lost = 100.0 * lost_t / total_t
            fallback_label = (
                f"or within {nearest_fallback_max_m:g}m"
                if nearest_fallback_max_m else "(fallback disabled)")
            logging.info(
                f"  → unassigned [{column}]: {pct_lost:.1f}% of total target "
                f"landed in cells with no building inside {fallback_label}")

    out = buildings.iloc[in_coverage_mask].copy()
    out[column] = contribs.reindex(np.arange(n_buildings)[in_coverage_mask]).fillna(0.0).values
    return out


# =====================================================================
# Proportional sub-category split — expand a coarse per-building
# dasymetric output (e.g. per-sector employment, total population) into
# finer sub-categories (industries within sectors, age bands within
# total population) by applying per-cell (sub / parent) ratios at each
# building's centroid cell. Cell-level sub-category structure is assumed
# to apply uniformly to the buildings within each cell.
# =====================================================================


def apply_proportional_split(
    out: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    cell_totals: pd.DataFrame,
    *,
    cell_id_col: str,
    splits: dict[str, str],
) -> gpd.GeoDataFrame:
    """For each `(sub_col, parent_col)` in `splits`, add a `sub_col`
    column to `out` whose per-building value is
    `out[parent_col] × cell_share` where `cell_share =
    cell_totals[sub_col] / cell_totals[parent_col]` at the building's
    centroid cell (division by zero → 0).

    Use after `per_building` to expand a coarse output into finer
    sub-categories that share the same per-cell structure. Canonical
    use: industry-within-sector for employment (10-industry detail
    from 3-sector dasymetric output). Buildings outside all cells
    get 0 for every sub column.

    Args:
        out: GeoDataFrame from `per_building`. Must contain every value
            in `splits.values()` as a column.
        cells: GeoDataFrame with `cell_id_col` + `geometry`.
        cell_totals: DataFrame indexed by `cell_id_col` values; must
            contain every key AND value in `splits` as a column.
        cell_id_col: cell identifier column / index name.
        splits: `{sub_col: parent_col}` mapping.

    Returns:
        Copy of `out` with one new column per key in `splits`.
    """
    # 1. Per-cell sub-category shares = sub / parent (clip inf/NaN to 0).
    cell_shares = pd.DataFrame(index=cell_totals.index)
    for sub_col, parent_col in splits.items():
        share = cell_totals[sub_col] / cell_totals[parent_col]
        cell_shares[sub_col] = share.replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Building -> centroid cell (single-cell assignment). `.values`
    # on the geometry strips its index — necessary because `out` may
    # carry a non-sequential index (e.g. osm_id after per_building's
    # in-coverage filter), which would otherwise misalign against the
    # dict-derived RangeIndex and produce all-NaN geometries.
    b_centroids = gpd.GeoDataFrame(
        {'_b_idx': np.arange(len(out))},
        geometry=out.geometry.centroid.values, crs=out.crs,
    )
    joined = gpd.sjoin(
        b_centroids, cells[[cell_id_col, 'geometry']],
        how='left', predicate='within',
    ).drop(columns=['index_right'], errors='ignore')

    # 3. Apply per-(sub, building) value = parent × cell_share. Build
    # the new column as a numpy array, then assign — avoids
    # `out.loc[bool_array, str_col]` which is valid pandas but trips
    # type-checkers' tuple-indexer overload resolution.
    matched = joined[cell_id_col].notna()
    matched_arr = matched.to_numpy()
    # Surface silent-zero cases: with buildings + cells both present but
    # zero centroid matches, every split column would silently come out
    # 0. Almost always indicates a CRS mismatch or an index-alignment
    # bug in the b_centroids construction.
    if len(out) and len(cells) and not matched_arr.any():
        logging.warning(
            f"  → apply_proportional_split: 0 of {len(out):,} buildings "
            f"matched any of {len(cells):,} cells via centroid-within. "
            f"Every split column will be 0. Check buildings/cells CRS "
            f"match and that `out` has the expected geometry.")
    out = out.copy()
    for sub_col, parent_col in splits.items():
        sub_values = np.zeros(len(out), dtype=float)
        if matched_arr.any():
            cell_ids = joined.loc[matched, cell_id_col].astype(int)
            share = cell_ids.map(cell_shares[sub_col]).fillna(0).to_numpy()
            parent_arr = out[parent_col].to_numpy()
            sub_values[matched_arr] = parent_arr[matched_arr] * share
        out[sub_col] = sub_values
    return out


# =====================================================================
# Coefficient learning — derive per-OSM-tag per-m² intensities directly
# from cell-level totals via non-negative least squares regression.
#
# For each category column independently, solves:
#     T_c ≈ Σ_tag (A_{c,tag} × β_tag)   subject to β ≥ 0
# where T_c is the cell total for the column, A_{c,tag} is the total
# overlay area of buildings tagged `tag` in cell c, and β_tag is the
# learned per-m² intensity for that tag.
#
# No prior coefficients are needed — intensities are inferred entirely
# from the spatial pattern of cell totals vs. building tag mix. The
# learned table is the canonical "calibrated coefficients" output:
# safe to publish alongside the per-building output (aggregated
# statistics only — no individual record exposure).
# =====================================================================


def learn_coefficients(
    buildings: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    cell_totals: pd.DataFrame,
    *,
    cell_id_col: str,
    column: str,
    relevant_tags: frozenset[str] | set[str],
    building_tag_col: str = 'building',
    area_col: str = 'area_m2',
    nearest_fallback_max_m: float | None = 200.0,
    collinearity_warn_threshold: float = 0.99,
) -> pd.DataFrame:
    """Learn per-OSM-tag per-m² intensities for a single `column` via
    NNLS regression on per-cell totals.

    Linear system `y = X β` where:
      - y = per-cell totals for this column.
      - X = augmented area matrix:
          * Overlay rows: `X[c, tag] = sum overlap area of relevant-tag
            buildings in cell c`.
          * Fallback rows: for cells with NO relevant-tag overlap but a
            nearest relevant-tag building B within
            `nearest_fallback_max_m`, a synthetic row
            `X[c_syn, tag(B)] = B.area` is added. Matches the model
            `per_building` uses for allocation → learned β values are
            consistent with downstream use.
      - β = per-tag intensities (FTE/m² or residents/m²), constrained
        to ≥ 0 via `scipy.optimize.nnls`.

    Logs a 3-tier coverage diagnostic per target-bearing cell: fittable
    via overlay, fittable via fallback, or truly unreachable (no
    relevant building nearby — excluded from the regression; sets the
    lower bound on the residual but cannot bias β).

    Standard errors via the OLS-style formula
    `SE(β_j) = √(σ² × (X'X)⁻¹_jj)` with `σ² = RSS / max(n - k, 1)`.
    Two-sided p-value (H0: β = 0) via `p = 2 × Φ(-|β| / SE)`. Slightly
    overestimates SE for tags where the non-negativity constraint binds
    (β = 0, where the convention yields p = 1); accurate for non-binding
    tags.

    **Bias correction**: NNLS β minimizes per-cell squared error, not
    total preservation. β is rescaled at the end so that applying it
    to ALL `buildings` in the input reproduces `cal_target_sum`:
    `Σ_T (β_eff × Σ_area_T_in_input) = cal_target_sum`.

    **CRITICAL caller contract**: `buildings` MUST be pre-filtered to
    the calibration area's buildings (e.g. country borders, where the
    ground truth applies). If it includes buffer-ring buildings outside
    that area, β is under-scaled and applying it to a country-only
    subset under-predicts. With proper pre-filtering, β can be applied
    to any region for extrapolation and the total scales proportionally
    with that region's relevant-tag area (the right behavior absent any
    per-cell anchor outside the calibration region).

    Per-cell rescaling in `per_building` is invariant to this scaling
    (only relative β within a cell matters there). The p-value is also
    scale-invariant.

    Args:
        buildings: polygon GeoDataFrame with `building_tag_col` +
            `area_col`. Same CRS as `cells`. Must be metric.
        cells: polygon GeoDataFrame with `cell_id_col`. Same CRS as
            `buildings`.
        cell_totals: DataFrame indexed by cell_id values; must include
            `column`.
        cell_id_col, column: as named.
        relevant_tags: eligible OSM tags for this column. Other-tag
            buildings are filtered out entirely.
        building_tag_col, area_col: column names. Defaults `'building'`,
            `'area_m2'`.
        nearest_fallback_max_m: max fallback distance. `None` disables.
        collinearity_warn_threshold: WARN if any pair of tag columns
            has |corr| > this in X.

    Returns:
        DataFrame indexed by OSM tag (sorted), with columns:
          - `intensity_<column>` — bias-corrected per-m² intensity
          - `p_value_<column>` — two-sided p-value for H0: β = 0
          - `contribution_<column>` — β × overlay area for this tag,
            i.e. share of `cal_target_sum` explained by this tag. Σ
            across tags ≈ `cal_target_sum` by construction.
          - `n_observations` — count of unique buildings of this tag
            that overlapped any calibration cell (in-calibration count;
            input buildings outside all cells, e.g. buffer ring, are
            excluded).
    """
    from scipy.optimize import nnls
    from scipy.stats import norm

    if buildings.crs is None or cells.crs is None:
        raise ValueError("Both `buildings` and `cells` must have a CRS set.")
    if buildings.crs != cells.crs:
        raise ValueError(
            f"CRS mismatch: buildings={buildings.crs}, cells={cells.crs}. "
            f"Reproject one before calling.")
    if column not in cell_totals.columns:
        raise ValueError(f"`cell_totals` is missing column {column!r}.")
    if building_tag_col not in buildings.columns:
        raise ValueError(
            f"`buildings` is missing the tag column {building_tag_col!r}.")
    if area_col not in buildings.columns:
        raise ValueError(
            f"`buildings` is missing the area column {area_col!r}.")

    int_col = f'intensity_{column}'
    p_col = f'p_value_{column}'
    contrib_col = f'contribution_{column}'
    out_cols = [int_col, p_col, contrib_col, 'n_observations']

    def _empty_output() -> pd.DataFrame:
        empty = pd.DataFrame(columns=out_cols)
        empty.index.name = building_tag_col
        return empty

    # Drop area=0 buildings (shouldn't appear; guarded defensively).
    buildings = buildings[buildings[area_col] > 0]

    # Filter to relevant-tag buildings.
    relevant_tags_set = set(relevant_tags)
    b_in = buildings[buildings[building_tag_col].isin(relevant_tags_set)]
    if not len(b_in):
        logging.warning(
            f"  → learn_coefficients [{column}]: no buildings match any tag "
            f"in `relevant_tags`; returning empty result.")
        return _empty_output()

    # Per-tag full-area sum and observation count, across the WHOLE
    # `buildings` input. The caller's responsibility is to pre-filter
    # `buildings` to the calibration area (e.g. country borders); this
    # function takes the input at face value as the calibration scope.
    # Used as the denominator for bias rescaling (β reflects average
    # intensity across all input buildings) and for the contribution
    # column. Critical: this is what makes `β × area` give consistent
    # totals when later applied to all buildings of the same area.
    area_sum_per_tag = b_in.groupby(building_tag_col)[area_col].sum()
    n_obs_per_tag = b_in.groupby(building_tag_col).size().rename('n_observations')

    # Overlay buildings × cells → per-(cell, tag) area matrix.
    b_overlay = gpd.GeoDataFrame(
        {building_tag_col: b_in[building_tag_col].values},
        geometry=b_in.geometry.values, crs=b_in.crs,
    )
    c_in = cells[[cell_id_col, 'geometry']].copy()
    overlays = gpd.overlay(b_overlay, c_in, how='intersection', keep_geom_type=False)
    cell_targets = cell_totals[column].astype(float)

    if len(overlays):
        overlays['_area'] = overlays.geometry.area
        area_matrix = (
            overlays.groupby([cell_id_col, building_tag_col])['_area']
            .sum().unstack(building_tag_col, fill_value=0.0))
    else:
        area_matrix = pd.DataFrame(
            index=pd.Index([], name=cell_id_col),
            columns=pd.Index([], name=building_tag_col), dtype=float)

    area_matrix = area_matrix.reindex(cell_targets.index, fill_value=0.0)

    # ---- Coverage diagnostic (tier 1: overlap) ----
    row_sums = area_matrix.sum(axis=1).to_numpy()
    target_pos_mask = (cell_targets.to_numpy() > 0)
    overlap_match_mask = target_pos_mask & (row_sums > 0)
    no_overlap_mask = target_pos_mask & (row_sums == 0)
    n_target_pos = int(target_pos_mask.sum())
    n_overlap_match = int(overlap_match_mask.sum())
    n_no_overlap = int(no_overlap_mask.sum())
    total_target_sum = float(cell_targets.sum())

    # ---- Tier 2: nearest-fallback synthetic rows ----
    # For cells with no overlap, find the nearest relevant-tag building
    # within max_m. Add a synthetic regression row: full B.area attributed
    # to cell C's target via tag(B). Matches per_building's allocation
    # behavior so learning and allocation see the same model.
    n_fallback_rescued = 0
    fallback_rescued_target_sum = 0.0
    fallback_extra_rows: list[tuple] = []  # (cell_id, tag, B_area)
    if nearest_fallback_max_m is not None and nearest_fallback_max_m > 0 and n_no_overlap > 0:
        no_overlap_cell_ids = cell_targets.index.to_numpy()[no_overlap_mask]
        nearest = _nearest_relevant_building(
            cells, no_overlap_cell_ids, b_in,
            cell_id_col=cell_id_col,
            max_distance=nearest_fallback_max_m,
            bring=[building_tag_col, area_col],
        )
        n_fallback_rescued = len(nearest)
        if n_fallback_rescued:
            rescued_targets = cell_targets.loc[nearest[cell_id_col].values]
            fallback_rescued_target_sum = float(rescued_targets.sum())
            fallback_extra_rows = list(zip(
                nearest[cell_id_col].values,
                nearest[building_tag_col].values,
                nearest[area_col].astype(float).values,
            ))

    n_unreachable = n_no_overlap - n_fallback_rescued
    unreachable_target_sum = (
        float(cell_targets.to_numpy()[no_overlap_mask].sum()) - fallback_rescued_target_sum)

    # Coverage % computed here, logged below alongside NNLS fit summary.
    if n_target_pos > 0:
        pct_overlap = 100.0 * n_overlap_match / n_target_pos
        pct_fallback = 100.0 * n_fallback_rescued / n_target_pos
        pct_unreachable = 100.0 * n_unreachable / n_target_pos
        pct_unreachable_target = (
            100.0 * unreachable_target_sum / total_target_sum
            if total_target_sum > 0 else 0.0)
    else:
        pct_overlap = pct_fallback = pct_unreachable = pct_unreachable_target = 0.0

    # ---- Build augmented (X, y) for NNLS ----
    # Drop cells with zero row-sum (no overlap) that weren't rescued by
    # fallback — they have nothing to fit. This includes truly unreachable
    # cells AND zero-target cells with no relevant buildings (which contribute
    # nothing to the loss either way).
    # If we have fallback rows, build them now and add them.
    if fallback_extra_rows:
        # Pivot fallback rows into a (cell, tag) area frame.
        fb_df = pd.DataFrame(
            fallback_extra_rows, columns=[cell_id_col, building_tag_col, '_area'])
        fb_matrix = (
            fb_df.groupby([cell_id_col, building_tag_col])['_area']
            .sum().unstack(building_tag_col, fill_value=0.0))
        # Align columns to area_matrix's column set.
        all_tags = sorted(set(area_matrix.columns) | set(fb_matrix.columns))
        area_matrix = area_matrix.reindex(columns=all_tags, fill_value=0.0)
        fb_matrix = fb_matrix.reindex(columns=all_tags, fill_value=0.0)
        # Use cell_id values directly (the fallback rows replace zero-row
        # entries in area_matrix for those cell_ids).
        for cid in fb_matrix.index:
            if cid in area_matrix.index:
                area_matrix.loc[cid] = (
                    area_matrix.loc[cid].to_numpy() + fb_matrix.loc[cid].to_numpy())

    # Build y from cell_targets aligned to area_matrix's index.
    y = cell_targets.reindex(area_matrix.index).fillna(0.0).to_numpy()

    # Drop rows that have zero row-sum (no overlay + no fallback) AND
    # zero target — pure no-ops.
    # Also drop zero-row + positive-target (truly unreachable): they
    # contribute target² to RSS but can't influence β.
    row_sums_aug = area_matrix.sum(axis=1).to_numpy()
    keep_row_mask = row_sums_aug > 0
    if not keep_row_mask.any():
        logging.warning(
            f"  → learn_coefficients [{column}]: no fittable rows after "
            f"fallback augmentation; returning empty result.")
        return _empty_output()
    X_full = area_matrix.iloc[keep_row_mask].to_numpy()
    y = y[keep_row_mask]
    tag_names_full = area_matrix.columns.tolist()

    # Drop tag columns with zero column-sum (no signal for that tag).
    col_sums = X_full.sum(axis=0)
    nonzero_col_mask = col_sums > 0
    if not nonzero_col_mask.any():
        logging.warning(
            f"  → learn_coefficients [{column}]: all tag columns have "
            f"zero area; returning empty result.")
        return _empty_output()
    X = X_full[:, nonzero_col_mask]
    tag_names = [t for t, keep in zip(tag_names_full, nonzero_col_mask) if keep]

    warn_high_collinearity(
        X, tag_names,
        threshold=collinearity_warn_threshold,
        context_label=f"learn_coefficients [{column}]",
    )

    # NNLS.
    beta, _residual_norm = nnls(X, y)

    # OLS-style standard errors.
    y_pred = X @ beta
    residuals = y - y_pred
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float((residuals ** 2).sum() / dof)
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    # Two-sided p-value for H0: β = 0. For NNLS-clipped β = 0 (constraint
    # binds), z = 0 → p = 1 (cannot reject null). For β >> 0 with small
    # SE, p → 0. Where SE = 0 (e.g. degenerate column), p = NaN.
    # Computed BEFORE the bias rescaling below — under rescaling both β
    # and SE would scale by the same factor, so z = β/SE is invariant
    # and the p-value is unchanged either way; computing here just makes
    # that explicit.
    with np.errstate(invalid='ignore', divide='ignore'):
        z_stat = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * norm.sf(np.abs(z_stat))

    # Bias correction: NNLS β minimizes squared error per cell, NOT
    # total preservation. Applied bottom-up as `β × area` to all
    # relevant-tag buildings in the calibration area, NNLS gives a
    # total ≠ cal_target_sum. Rescale β so the SUM across all input
    # buildings reproduces cal_target_sum:
    #
    #     Σ_T (β_eff × area_sum_per_tag[T]) = cal_target_sum
    #
    # CRITICAL: this requires `buildings` (the input) to BE the
    # calibration area's buildings — i.e. the caller has already
    # filtered to inside the country/region where `cell_totals` applies.
    # If the caller passes a wider set (e.g. CH + buffer), β gets
    # under-scaled and applying it to CH alone under-predicts.
    #
    # When the published β is later applied to a different region for
    # extrapolation, it scales proportionally with that region's
    # relevant-tag area — which is the right behavior, since we have
    # no per-cell anchor outside the calibration region.
    #
    # Per-cell rescaling in `per_building` is invariant to this scaling
    # (only relative β within a cell matters there), so Swiss-side output
    # is unchanged.
    predicted_in_calibration = sum(
        beta[i] * float(area_sum_per_tag.get(tag_names[i], 0.0))
        for i in range(len(tag_names)))
    if predicted_in_calibration > 0:
        bias_scale = total_target_sum / predicted_in_calibration
        beta = beta * bias_scale
        logging.info(
            f"  → rescale [{column}]: β × {bias_scale:.4f} so "
            f"Σ (β × area) over all calibration-area buildings = "
            f"target sum ({predicted_in_calibration:,.0f} → "
            f"{total_target_sum:,.0f})")

    # Contribution per tag = β_eff × full area of that tag's buildings
    # in the calibration area. Σ contributions = cal_target_sum BY
    # CONSTRUCTION (same scaling), so each tag's contribution = its
    # share of the national target.
    contributions = np.array([
        beta[i] * float(area_sum_per_tag.get(tag_names[i], 0.0))
        for i in range(len(tag_names))
    ])

    total_target = float(y.sum())
    total_predicted = float(y_pred.sum())
    residual_pct = (
        100.0 * (total_predicted - total_target) / total_target
        if total_target != 0 else 0.0)
    logging.info(
        f"  → coverage [{column}]: {n_target_pos:,} cells with target > 0 → "
        f"{pct_overlap:.1f}% via overlay + {pct_fallback:.1f}% via fallback + "
        f"{pct_unreachable:.1f}% unreachable "
        f"({pct_unreachable_target:.1f}% of target lost)")
    logging.info(
        f"  → NNLS [{column}]: {len(tag_names)} tags, "
        f"β∈[{beta.min():.4g}, {beta.max():.4g}], fit {residual_pct:+.2f}%")

    rows: dict[str, dict[str, float]] = {}
    for i, tag in enumerate(tag_names):
        rows[tag] = {
            int_col: float(beta[i]),
            p_col: float(p_values[i]),
            contrib_col: float(contributions[i]),
        }
    out = pd.DataFrame.from_dict(rows, orient='index')
    out = out.sort_index()
    out.index.name = building_tag_col
    out = out.join(n_obs_per_tag, how='left')
    out['n_observations'] = out['n_observations'].fillna(0).astype(int)
    return out[out_cols]


def per_building_from_coefficients(
    buildings: gpd.GeoDataFrame,
    columns: Sequence[str],
    *,
    calibrated_coeffs: pd.DataFrame,
    building_tag_col: str = 'building',
    area_col: str = 'area_m2',
    precision: int | None = 2,
) -> gpd.GeoDataFrame:
    """Compute per-building values from learned per-tag intensity
    coefficients × building area. No per-cell rescaling — there are no
    cell totals in this path (the world-side use case where ground-truth
    cell data doesn't exist).

    Each output column reads its intensity from `intensity_<col>` in
    `calibrated_coeffs`. Per-(tag, col) NaN entries (= tag not in that
    column's membership during learning) result in 0 for that
    building's column. Buildings whose tag isn't in
    `calibrated_coeffs.index` at all are dropped entirely.

    Output values are rounded to `precision` decimals (default 2 →
    quantized to 0.01). With the default, very small estimates (β ×
    area < 5e-3) round to 0 — appropriate for bottom-up extrapolation
    where a per-building fraction below ~0.005 FTE / resident is below
    the meaningful precision of the underlying calibration. Pass
    `precision=None` to get raw float output (no rounding).

    Args:
        buildings: GeoDataFrame with `building_tag_col` (OSM tag) and
            `area_col` (m²). Index preserved on output.
        columns: output column names (e.g.
            `('employment_primary', 'employment_secondary',
            'employment_tertiary')`). Each column must be reachable
            via `intensity_<col>` in `calibrated_coeffs`.
        calibrated_coeffs: output of `learn_coefficients`, indexed by
            OSM building tag, with `intensity_<col>` columns. Required.
        building_tag_col: tag column name in `buildings`.
        area_col: area column name in `buildings`.
        precision: round each output column to this many decimals.
            Default 2 → quantize to 0.01. `None` → no rounding.

    Returns:
        Copy of the (tag-filtered) `buildings` with one column per
        entry in `columns`, holding (intensity × area_m2) per building.
        Index preserved.
    """
    columns = tuple(columns)
    if building_tag_col not in buildings.columns:
        raise ValueError(
            f"`buildings` is missing the tag column {building_tag_col!r}.")
    if area_col not in buildings.columns:
        raise ValueError(
            f"`buildings` is missing the area column {area_col!r}.")
    if calibrated_coeffs is None or not len(calibrated_coeffs):
        raise ValueError(
            "`calibrated_coeffs` is required and must be non-empty. "
            "Run `learn_coefficients` first.")
    for col in columns:
        intensity_col = f'intensity_{col}'
        if intensity_col not in calibrated_coeffs.columns:
            raise ValueError(
                f"`calibrated_coeffs` is missing column {intensity_col!r}. "
                f"Did you pass the right `columns`?")

    # Build per-(tag, col) intensity lookup. NaN values in the
    # calibrated table mean "tag wasn't in this column's relevant set
    # during learning" → contribute 0 for that column.
    relevant_tags = set(str(t) for t in calibrated_coeffs.index)
    n_before = len(buildings)
    buildings = buildings[buildings[building_tag_col].isin(relevant_tags)].copy()
    n_dropped = n_before - len(buildings)
    if n_dropped:
        logging.info(
            f"  → filtered out {n_dropped:,} buildings with tags outside "
            f"the calibrated set ({len(buildings):,} / {n_before:,} kept = "
            f"{len(buildings) / n_before * 100:.1f} %)")

    out = buildings.copy()
    tags = out[building_tag_col].astype(str).values
    areas = out[area_col].astype(float).values
    for col in columns:
        intensity_series = calibrated_coeffs[f'intensity_{col}']
        intensities = np.array(
            [float(intensity_series.get(t, 0.0)) for t in tags])
        intensities = np.where(np.isnan(intensities), 0.0, intensities)
        values = intensities * areas
        if precision is not None:
            values = np.round(values, precision)
        out[col] = values
    return out
