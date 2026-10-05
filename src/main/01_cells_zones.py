"""
Build the cells + zones layers for one atlas scenario.

1. Define cell layer based on `scenario.cell_source`:
     - `'h3'`         → H3 grid at `scenario.cell_h3_resolution` over
                        `area.place` + widest mode-buffer. Polygon cells.
                        Per-cell pop/emp: aggregate from per-building
                        values (which come via dasymetric mapping in the
                        Swiss preparation scripts).
     - `'hectares'`   → 100 m squares loaded from persisted STATPOP +
                        STATENT hectare artifacts (Swiss only). Polygon
                        cells. Per-cell pop/emp: attached directly from
                        the artifacts (no dasymetric — hectare data IS
                        the ground truth at that resolution).
     - `'buildings'`  → OSM building centroids as POINT cells. Per-cell
                        pop/emp: attached from `_load_per_building_values`
                        (dasymetric-derived per-building values).
2. Define zone layer: either NPVM Swiss traffic zones
   (`scenario.zone_h3_resolution is None`, the original Swiss path) or
   a coarser H3 grid at `scenario.zone_h3_resolution` over the same
   buffered area polygon (Cambridge UK + future non-Swiss cases).
3. Load POIs (categories + weights). Allocate to cells: centroid-in-
   polygon for polygon cells, nearest-cell-centroid for point cells.
4. Allocate cells to zones (cell centroid → zone polygon).
5. Drop cells with no population AND no employment AND no POIs.
6. Drop zones with no cells.
7. Save cells.gpkg + cells_centroids.gpkg + zones.gpkg + per-property CSVs.

Outputs (per `Storage`, under <scenario>/):
    shapes/cells.gpkg                  # polygon OR point per cell_source
    shapes/cells_centroids.gpkg        # point-representation companion
    shapes/zones.gpkg                  # NPVM zones OR H3 hex polygons
    properties/cells_population.csv    # per-cell population_total
    properties/cells_employment.csv    # per-cell employment_total
    properties/cells_pois.csv          # per-cell n_pois + per-category counts
    properties/zones_population.csv    # per-zone aggregates
    properties/zones_employment.csv
    properties/zones_pois.csv
"""

import logging

import geopandas as gpd
import pandas as pd

from aperta import geo_mapping, geo_processing
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from preparation.switzerland.common import filter_to_inside_ch, load_swiss_country
from preparation.world.areas import AREAS, widest_buffer
from preparation.world.common import buffered_place_polygon
from scenarios import get_scenario


# =====================================================================
# Per-cell-source cell builders. Each returns a GeoDataFrame with
# `cell_id` as the index and `geometry` as either polygons (h3,
# hectares) or points (buildings). For hectares/buildings, pop/emp
# value columns are attached; for h3 they are attached later by
# aggregating per-building values.
# =====================================================================

def _build_cells_h3(scenario, area_buffered, crs_main):
    """H3 hex grid over the buffered area. Polygon cells; values
    attached later via per-building aggregation."""
    cells = geo_processing.build_h3_grid(
        area_buffered.geometry.union_all(),
        scenario.cell_h3_resolution,
        polygon_crs=area_buffered.crs.to_string(),
        target_crs=crs_main,
        id_column='cell_id',
    )
    logging.info(f"  → {len(cells):,} H3 cells at resolution {scenario.cell_h3_resolution}")
    return cells


def _build_cells_hectares(context, scenario, area_buffered, crs_main):
    """100 m hectare cells loaded from persisted STATPOP + STATENT
    artifacts (Swiss only). Union of both sources' hectares; values
    attached directly (fillna=0 where a hectare is present in one
    source but not the other)."""
    lu_ch = context.source('preparation/switzerland/land_use')

    # Load hectare shapes + values from both Swiss ground-truth sources.
    # File names are source-anchored (statpop / statent); semantic content
    # (population vs employment columns) is self-evident from the CSVs.
    pop_shape = lu_ch.get_generic(f'hectares_statpop_{scenario.statpop_year}.gpkg')
    emp_shape = lu_ch.get_generic(f'hectares_statent_{scenario.statent_year}.gpkg')
    pop_vals = lu_ch.get_generic(
        f'hectares_statpop_{scenario.statpop_year}.csv',
        kws={'index_col': 'cell_id'},
    )
    emp_vals = lu_ch.get_generic(
        f'hectares_statent_{scenario.statent_year}.csv',
        kws={'index_col': 'cell_id'},
    )

    # Union both hectare grids (STATPOP-residential ∪ STATENT-commercial).
    # Coordinate-based cell_ids from the preparation-side re-keying make
    # this a safe cross-source join.
    cells = (
        pd.concat([pop_shape, emp_shape])
        .drop_duplicates(subset='cell_id')
        .set_index('cell_id')
    )
    if cells.crs is None or cells.crs.to_string() != crs_main:
        cells = cells.to_crs(crs_main)

    # Attach pop + emp values; fillna=0 for hectares covered by only
    # one source.
    cells = cells.join(pop_vals[list(scenario.population_cols)], how='left')
    cells = cells.join(emp_vals[list(scenario.employment_cols)], how='left')
    value_cols = [*scenario.population_cols, *scenario.employment_cols]
    cells[value_cols] = cells[value_cols].fillna(0.0)

    # Clip to the buffered analysis area (drop far-outside hectares).
    envelope = area_buffered.to_crs(crs_main).geometry.union_all()
    cells = cells.loc[cells.geometry.centroid.within(envelope)]

    logging.info(
        f"  → {len(cells):,} hectare cells "
        f"(STATPOP {scenario.statpop_year} ∪ STATENT {scenario.statent_year}, "
        f"clipped to area+buffer); pop={cells['population_total'].sum():,.0f}, "
        f"emp={cells['employment_total'].sum():,.0f}")
    return cells


def _build_cells_buildings(context, scenario, area_buffered, crs_main,
                            country, ids_in_ch):
    """OSM building centroids as POINT cells; per-building pop/emp
    attached from `_load_per_building_values`. Building index becomes
    the cell_id — stable across OSM refreshes for the same building
    (OSM way ID)."""
    osm_ctx = context.source('preparation/world/osm')
    buildings = osm_ctx.get_shapes('buildings', data_name=scenario.area_name)
    buildings = buildings.to_crs(crs_main)

    pop, emp = _load_per_building_values(
        context, scenario, ids_in_ch, buildings.index)
    buildings = buildings.join(pop).join(emp)

    # Cells are points (building centroids). Index is building_id.
    cells = gpd.GeoDataFrame(
        buildings[
            [*scenario.population_cols, *scenario.employment_cols]
        ].copy(),
        geometry=buildings.geometry.centroid,
        crs=crs_main,
    )
    cells.index.name = 'cell_id'

    logging.info(
        f"  → {len(cells):,} building cells (OSM centroids); "
        f"pop={cells['population_total'].sum():,.0f}, "
        f"emp={cells['employment_total'].sum():,.0f}")
    return cells


# =====================================================================
# Per-building pop/emp loader — used by h3 (aggregated) and buildings
# (attached directly) paths. Not called for hectares.
# =====================================================================

def _combine_buildings(
    public: pd.DataFrame,
    private: pd.DataFrame | None,
    cols: tuple[str, ...],
    building_ids_in_ch: pd.Index,
    all_building_ids: pd.Index,
) -> pd.DataFrame:
    """Build a per-building DataFrame for the requested columns. Start
    with the public source (covers CH + buffer), then overwrite the
    CH-inside subset with private values where available. `private=None`
    skips the overwrite step (= pure-public scenario).
    """
    result = public.reindex(all_building_ids)[list(cols)].fillna(0.0)
    if private is not None:
        ch_idx = all_building_ids.intersection(building_ids_in_ch)
        result.loc[ch_idx] = (private.reindex(ch_idx)[list(cols)].fillna(0.0))
    return result.astype(float)


def _load_per_building_values(
    context, scenario, building_ids_ch: pd.Index, all_building_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    lu_ch = context.source('preparation/switzerland/land_use')
    lu_world = context.source('preparation/world/land_use')

    # Population "public" source depends on scenario.population_source:
    #   - 'coef' → dasymetric per-OSM-tag intensities calibrated on STATPOP,
    #             produced by preparation/world/land_use/population_per_building_from_coef.py
    #             (`population_coef_<area>`).
    #   - 'ghs' or 'statpop' → GHS-POP dasymetric-per-building
    #             (`population_<area>`). For 'statpop', GHS also serves as
    #             the outside-CH fallback when private STATPOP overrides
    #             CH-inside buildings.
    if scenario.population_source == 'coef':
        pop_public = lu_world.get_properties(
            'buildings', f'population_coef_{scenario.ghs_pop_area_name}')
    else:
        pop_public = lu_world.get_properties(
            'buildings', f'population_{scenario.ghs_pop_area_name}')
    # Employment "public" source: only one option — dasymetric per-OSM-tag
    # intensities calibrated on STATENT (`employment_<area>`).
    emp_public = lu_world.get_properties('buildings', f'employment_{scenario.ghs_pop_area_name}')

    if scenario.population_source == 'statpop':
        pop_private = lu_ch.get_properties('buildings', f'population_{scenario.statpop_year}')
    else:
        pop_private = None
    if scenario.employment_source == 'statent':
        emp_private = lu_ch.get_properties('buildings', f'employment_statent_{scenario.statent_year}')
    else:
        emp_private = None

    pop_df = _combine_buildings(pop_public, pop_private, scenario.population_cols, building_ids_ch, all_building_ids)
    emp_df = _combine_buildings(emp_public, emp_private, scenario.employment_cols, building_ids_ch, all_building_ids)
    return pop_df, emp_df


# =====================================================================
# Small helpers to keep main() readable across the polygon/point cell
# distinction.
# =====================================================================

def _cell_points(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of cell points (centroid for polygon cells;
    the point itself for point cells). Geopandas `.centroid` returns the
    input for Point geometries, so this is uniform across cell sources —
    the helper just exists to make intent explicit at call sites."""
    return gpd.GeoDataFrame(
        geometry=cells.geometry.centroid, index=cells.index, crs=cells.crs,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    context = init_context()
    scenario = get_scenario(context.scenario)
    area = AREAS[scenario.area_name]
    crs_main = scenario.crs_main
    aoi_polygon = buffered_place_polygon(area.place, area.aoi_buffer_m).to_crs(crs_main)
    aoi_geom = aoi_polygon.geometry.union_all()
    area_buffered = buffered_place_polygon(area.place, widest_buffer(area))

    # --- CH classification is needed early for the buildings path -----
    country = load_swiss_country(context)
    ch_polygon = country.geometry.union_all()

    # For the buildings path we need CH-inside building IDs before the
    # cell-builder can attach values. Compute here once; also reused in
    # h3 path below.
    needs_ch_filter = (
        scenario.population_source == 'statpop'
        or scenario.employment_source == 'statent'
    )
    ids_in_ch = pd.Index([])
    buildings = None  # populated in h3 + buildings paths
    if scenario.cell_source in ('h3', 'buildings'):
        with step('load buildings + classify CH-inside'):
            osm_ctx = context.source('preparation/world/osm')
            buildings = osm_ctx.get_shapes('buildings', data_name=scenario.area_name)
            buildings = buildings.to_crs(crs_main)
            if needs_ch_filter:
                buildings_ch = filter_to_inside_ch(buildings, country)
                ids_in_ch = buildings_ch.index
                logging.info(
                    f"  → {len(buildings):,} total buildings "
                    f"({len(ids_in_ch):,} in CH, "
                    f"{len(buildings) - len(ids_in_ch):,} in buffer)")
            else:
                logging.info(f"  → {len(buildings):,} total buildings")

    # --- Build cell layer, dispatching on cell_source -----------------
    with step(f"build cells (source={scenario.cell_source!r})"):
        if scenario.cell_source == 'h3':
            cells = _build_cells_h3(scenario, area_buffered, crs_main)
            values_attached_at_build_time = False
        elif scenario.cell_source == 'hectares':
            cells = _build_cells_hectares(context, scenario, area_buffered, crs_main)
            values_attached_at_build_time = True
        elif scenario.cell_source == 'buildings':
            cells = _build_cells_buildings(
                context, scenario, area_buffered, crs_main,
                country=country, ids_in_ch=ids_in_ch)
            values_attached_at_build_time = True
        else:
            raise ValueError(
                f"Unknown scenario.cell_source={scenario.cell_source!r}")

    # --- Classify cells as AOI + CH-inside -----------------------------
    with step('classify cells as AOI + CH-inside'):
        cell_points = _cell_points(cells)
        cells['is_aoi'] = cell_points.geometry.within(aoi_geom)
        cells['is_in_ch'] = cell_points.geometry.within(ch_polygon)
        n_aoi = int(cells['is_aoi'].sum())
        n_ch = int(cells['is_in_ch'].sum())
        logging.info(
            f"  → {n_aoi:,} AOI cells ({100*n_aoi/len(cells):.1f} %); "
            f"{n_ch:,} CH-inside cells")

    # --- Load zones: NPVM (Swiss) OR H3 grid (any area) ---------------
    if scenario.zone_h3_resolution is None:
        with step('load NPVM traffic zones (Swiss path)'):
            general_ctx = context.source('preparation/switzerland/general')
            zones = general_ctx.get_generic('traffic_zones_without_lakes.gpkg').set_index('zone_id')
            zones = zones.to_crs(crs_main)
            zones['is_aoi'] = zones.geometry.centroid.within(aoi_geom).astype(int)
            logging.info(f"  → {len(zones):,} NPVM zones ({int(zones['is_aoi'].sum()):,} in AOI)")
    else:
        with step(f'build H3 zone grid (res {scenario.zone_h3_resolution})'):
            zones = geo_processing.build_h3_grid(
                area_buffered.geometry.union_all(),
                scenario.zone_h3_resolution,
                polygon_crs=area_buffered.crs.to_string(),
                target_crs=crs_main,
                id_column='zone_id',
            )
            zones = zones.set_index('zone_id')
            zones['is_aoi'] = zones.geometry.centroid.within(aoi_geom).astype(int)
            logging.info(f"  → {len(zones):,} H3 zones ({int(zones['is_aoi'].sum()):,} in AOI)")

    with step('classify zones as CH-inside'):
        zones['is_in_ch'] = zones.geometry.centroid.within(ch_polygon)
        logging.info(f"  → {int(zones['is_in_ch'].sum()):,} of {len(zones):,} zones in CH")

    # --- Attach per-cell pop/emp for the h3 path ----------------------
    if not values_attached_at_build_time:
        # h3 only: use loaded buildings + per-building values, aggregate
        # into cells via containment.
        with step('attach per-building population + employment'):
            assert buildings is not None
            pop, emp = _load_per_building_values(
                context, scenario, ids_in_ch, buildings.index)
            buildings = buildings.join(pop).join(emp)
            logging.info(
                f"  → totals: population={buildings['population_total'].sum():,.0f}, "
                f"employment_total={buildings['employment_total'].sum():,.0f}")

        with step('allocate buildings → cells (centroid-in-polygon)'):
            _b_centroids = gpd.GeoDataFrame(
                geometry=buildings.geometry.centroid, crs=buildings.crs)
            _ids, _ = geo_mapping.map_points_to_polygons(
                _b_centroids, cells, allow_nearest=False)
            buildings['cell_id'] = _ids
            n_inside_cell = int(buildings['cell_id'].notna().sum())
            logging.info(
                f"  → {n_inside_cell:,} of {len(buildings):,} buildings allocated "
                f"({100*n_inside_cell/len(buildings):.1f} %)")

        with step('aggregate buildings → per-cell population / employment'):
            agg_cols = [*scenario.population_cols, *scenario.employment_cols]
            cell_aggs = (
                buildings.dropna(subset=['cell_id'])
                .groupby('cell_id')[agg_cols].sum()
                .reindex(cells.index, fill_value=0.0))
            cells = cells.join(cell_aggs)

    # --- Load POIs ----------------------------------------------------
    with step('load POIs'):
        osm_ctx = context.source('preparation/world/osm')
        pois = osm_ctx.get_generic(f'pois_{scenario.area_name}.gpkg')
        pois = pois.to_crs(crs_main)
        logging.info(f"  → {len(pois):,} POIs; {pois['category'].nunique():,} categories")

    # --- Allocate POIs → cells ----------------------------------------
    # Branch on cell geometry type: containment for polygon cells,
    # nearest-cell-centroid for point cells (buildings).
    if scenario.cell_source in ('h3', 'hectares'):
        with step('allocate POIs → cells (point-in-polygon)'):
            _p_centroids = gpd.GeoDataFrame(
                geometry=pois.geometry, crs=pois.crs)
            _ids, _ = geo_mapping.map_points_to_polygons(
                _p_centroids, cells, allow_nearest=False)
            pois['cell_id'] = _ids
    else:  # buildings — cells are points
        with step('allocate POIs → cells (nearest-cell-centroid)'):
            _p_centroids = gpd.GeoDataFrame(
                geometry=pois.geometry, crs=pois.crs)
            _ids, _ = geo_mapping.map_points_to_points(
                _p_centroids, _cell_points(cells))
            pois['cell_id'] = _ids
    n_pois_in_cell = int(pois['cell_id'].notna().sum())
    logging.info(f"  → {n_pois_in_cell:,} of {len(pois):,} POIs allocated to a cell")

    # --- Allocate cells → zones (cell point → zone polygon) -----------
    with step('allocate cells → zones'):
        _ids, _ = geo_mapping.map_points_to_polygons(
            _cell_points(cells), zones, allow_nearest=False)
        cells['zone_id'] = _ids
        n_cells_in_zone = int(cells['zone_id'].notna().sum())
        logging.info(
            f"  → {n_cells_in_zone:,} of {len(cells):,} cells allocated "
            f"to a zone ({100*n_cells_in_zone/len(cells):.1f} %)")

    # --- Aggregate per-cell POI counts --------------------------------
    with step('per-cell POI counts + per-category counts'):
        cells_n_pois = (
            pois.dropna(subset=['cell_id'])
            .groupby('cell_id').size().rename('n_pois')
            .reindex(cells.index, fill_value=0).astype(int))
        cells['n_pois'] = cells_n_pois
        cells_poi_per_cat = (
            pois.dropna(subset=['cell_id'])
            .groupby(['cell_id', 'category']).size()
            .unstack('category', fill_value=0))
        logging.info(
            f"  → cell totals: pop={cells['population_total'].sum():,.0f}, "
            f"emp_total={cells['employment_total'].sum():,.0f}, "
            f"n_pois={cells['n_pois'].sum():,}")

    # --- Drop empty cells and cells without a zone --------------------
    with step('drop empty cells and cells without zone ID'):
        n_before = len(cells)
        keep = ((cells['population_total'] > 0) |
                (cells['employment_total'] > 0) |
                (cells['n_pois'] > 0))
        cells = cells.loc[keep]
        cells_poi_per_cat = cells_poi_per_cat.reindex(cells.index, fill_value=0).astype(int)
        cells_poi_per_cat['n_pois'] = cells['n_pois']
        logging.info(f"  → kept {len(cells):,} non-empty cells ({100*len(cells)/n_before:.1f} %)")
        n_before = len(cells)
        cells = cells[cells['zone_id'].notna()]
        cells_poi_per_cat = cells_poi_per_cat.reindex(cells.index, fill_value=0).astype(int)
        logging.info(f"  → kept {len(cells):,} cells with Zone ID ({100*len(cells)/n_before:.1f} %)")

    # --- Aggregate per-zone sums --------------------------------------
    with step('aggregate per-zone population / employment / POIs'):
        agg_cols = [*scenario.population_cols, *scenario.employment_cols, 'n_pois']
        zone_aggs = (
            cells.dropna(subset=['zone_id'])
            .groupby('zone_id')[agg_cols].sum()
            .reindex(zones.index, fill_value=0))
        zones = zones.join(zone_aggs)
        zones['n_pois'] = zones['n_pois'].astype(int)
        logging.info(
            f"  → zone totals: pop={zones['population_total'].sum():,.0f}, "
            f"emp_total={zones['employment_total'].sum():,.0f}, "
            f"n_pois={zones['n_pois'].sum():,}")

    with step('drop empty zones'):
        n_before = len(zones)
        keep = ((zones['population_total'] > 0)
                | (zones['employment_total'] > 0)
                | (zones['n_pois'] > 0))
        zones = zones.loc[keep]
        n_foreign = int((~zones['is_in_ch']).sum())
        logging.info(
            f"  → kept {len(zones):,} of {n_before:,} zones ({n_foreign:,} foreign zones)"
            f"({100*len(zones)/n_before:.1f} %)")

    with step('aggregate per-category POI counts per zone'):
        zones_poi_per_cat = (
            cells_poi_per_cat.join(cells[['zone_id']])
            .dropna(subset=['zone_id'])
            .groupby('zone_id').sum()
            .reindex(zones.index, fill_value=0)
            .astype(int))

    if 'population_total' in scenario.population_cols and 'employment_total' in scenario.employment_cols:
        cells['combined_total'] = cells['population_total'] + cells['employment_total']
        zones['combined_total'] = zones['population_total'] + zones['employment_total']
        _save_pop_cols = [*scenario.population_cols, 'combined_total']

    # --- Save ---------------------------------------------------------
    with step('save shapes + centroids + properties'):
        context.create_shapes(cells, extra_columns=['zone_id', 'is_aoi', 'is_in_ch'])
        # Point-representation companion (matches the zones_centroids
        # convention). For point cells, this is a copy of shapes/cells.
        context.create_shapes(_cell_points(cells), data_name='centroids')

        context.create_shapes(zones, extra_columns=['is_aoi', 'is_in_ch'])

        # AOI subset — convenience for GIS / visualisation.
        context.create_shapes(cells.loc[cells['is_aoi']], data_name='aoi')

        context.create_properties(cells[['zone_id', 'is_aoi']], data_name='zone_ids')
        context.create_properties(cells[_save_pop_cols], data_name='population')
        context.create_properties(cells[list(scenario.employment_cols)], data_name='employment')
        context.create_properties(cells_poi_per_cat, data_name='pois')

        context.create_properties(zones[_save_pop_cols], data_name='population')
        context.create_properties(zones[list(scenario.employment_cols)], data_name='employment')
        context.create_properties(zones_poi_per_cat, data_name='pois')

    context.close()


if __name__ == '__main__':
    main()
