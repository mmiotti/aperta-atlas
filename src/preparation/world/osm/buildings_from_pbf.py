"""
Stream OSM building footprints from a locally clipped PBF.

Independent of the network preparation chain; only depends on
`clip_pbf.py`'s output. Mirrors `obstacles_from_pbf.py` and
`pois_from_pbf.py` in structure (pyosmium streaming handler,
per-area variants, GeoDataFrame output).

Used as the building-side input for `preparation/switzerland/land_use/
employment_dasymetric.py` (and future EU equivalents). The OSM `building`
tag value is preserved verbatim so the dasymetric mapping can apply its
per-tag × per-sector coefficient priors (`aperta_atlas.dasymetric.
DEFAULT_COEFFS`).

What's emitted:

  - **Closed ways with `building=*`**: footprint Polygon. The handler
    skips point-tagged buildings (`node` with `building=*`) and
    open / non-closed ways (no valid polygon). Very small footprints
    (`< _MIN_FOOTPRINT_M2`) are filtered out — these are typically
    tagging noise (1-2 m² roofs, sheds with no real employment role).

  - **Relations are not handled** (multipolygon buildings with holes /
    multiple outer rings). At Swiss scope these are <0.5 % of all
    `building=*` features and don't materially affect the dasymetric
    mapping. Revisit if a category looks short.

The `area_m2` column is computed in a local UTM (auto-picked) so the
downstream dasymetric mapping has a metric area per building without
needing to reproject the whole GeoDataFrame again. Polygons are
Douglas-Peucker-simplified at `_SIMPLIFY_TOLERANCE_M` (0.5 m) in the
same metric CRS to shrink disk size + RAM without measurably changing
area.

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_buildings` variant per area.

Inputs (PUBLIC, under raw/global/osm/):
    <pbf_name>-latest.osm.pbf       # output of `clip_pbf.py`

Outputs (PUBLIC, under preparation/world/osm/):
    shapes/buildings_<area_name>.gpkg     # polygon features, WGS84
        # index: building_id (OSM way ID values)
        # columns: building (OSM tag), area_m2, geometry
        # downstream property files are written by orchestrators to
        # properties/buildings_<data_name>.csv and join on building_id.

Requires `osmium-tool` (CLI) AND `pyosmium` (Python).

Run after `clip_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.buildings_from_pbf
Single variant:
    python -m preparation.world.osm.buildings_from_pbf --variant switzerland_buildings
    python -m preparation.world.osm.buildings_from_pbf --variant bern_buildings
"""

import logging

import geopandas as gpd
import osmium
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.osm.common import clipped_pbf


# Buildings smaller than this (in m²) are filtered out — typically
# tagging noise (1-2 m² roof points, sheds with no real role).
_MIN_FOOTPRINT_M2 = 25.0

# Douglas-Peucker simplification tolerance (in metres, applied in a
# local UTM where the area computation happens). 0.5 m matches the
# tolerance used in `networks_consolidate.py` — preserves real
# corners + removes sub-meter vectorisation noise. Saves ~30-50 %
# vertex count + ~30 % disk size on typical OSM building polygons
# without measurably changing area_m2 (well below 1 % for any
# building >= the `_MIN_FOOTPRINT_M2` threshold).
_SIMPLIFY_TOLERANCE_M = 0.5


# Buffer is the widest per-mode buffer for the area — covers any
# network's accessibility radius without proliferating per-mode building
# files. Output: `<area_name>_buildings.gpkg` per area.
variants = Variants([
    ('place', str), ('area_name', str), ('pbf_name', str), ('buffer', int),
])
for area in AREAS.values():
    variants.add(
        name=f'{area.name}_buildings',
        place=area.place, area_name=area.name,
        pbf_name=f'{area.name}_buffered',
        buffer=widest_buffer(area),
    )


class BuildingHandler(osmium.SimpleHandler):
    """Streaming pyosmium handler that collects OSM building footprints.

    Only closed ways carrying the `building=*` tag are emitted (as
    Polygons). Point-tagged buildings (uncommon, no footprint geometry)
    and multipolygon relations (rare, would require relation handling)
    are skipped.

    `locations=True` (passed to `apply_file`) is required so way nodes
    have resolved coordinates for polygon construction.
    """

    def __init__(self) -> None:
        super().__init__()
        # (osm_id, building_tag, coords_list). Coords are (lon, lat)
        # tuples in WGS84.
        self.features: list[tuple[int, str, list[tuple[float, float]]]] = []

    def way(self, w) -> None:
        building = w.tags.get('building')
        if building is None:
            return
        coords = [(n.location.lon, n.location.lat)
                  for n in w.nodes if n.location.valid()]
        # Need >=4 points (3 distinct + the duplicate closing point) to
        # form a valid polygon ring. Skip degenerate cases.
        if len(coords) < 4 or coords[0] != coords[-1]:
            return
        self.features.append((w.id, building, coords))


def features_to_gdf(features: list[tuple[int, str, list[tuple[float, float]]]]) -> gpd.GeoDataFrame:
    """Build a WGS84 GeoDataFrame of building polygons + their tags.

    Index is `building_id` (OSM way ID values — globally unique +
    stable across PBF refreshes, so property files written by
    downstream orchestrators can be joined back without going stale on
    re-extraction).

    Pipeline per building: validate → metric-CRS project → drop tiny
    footprints → Douglas-Peucker simplify (`_SIMPLIFY_TOLERANCE_M`) →
    back to WGS84. The simplification runs IN the metric CRS so
    `tolerance` is in meters; once-each round-trip projection
    amortises across all later steps.
    """
    polygons = [Polygon(coords) for _, _, coords in features]
    osm_ids = [osm_id for osm_id, _, _ in features]
    tags = [tag for _, tag, _ in features]
    gdf = gpd.GeoDataFrame(
        {'building': tags},
        geometry=polygons,
        index=pd.Index(osm_ids, name='building_id'),
        crs='EPSG:4326',
    )

    # Drop self-intersecting / invalid polygons before any other op
    # (they'd give meaningless areas + crash the simplify pass). Tiny
    # minority (<0.05 % at Swiss scope) from mis-tagged corner-cutting.
    valid_mask = gdf.geometry.is_valid
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logging.info(
            f"  → dropped {n_invalid:,} invalid polygon(s) "
            f"({n_invalid/len(gdf)*100:.3f} %)")
        gdf = gdf[valid_mask].copy()

    # Project to local UTM for area + simplification (both need metric
    # CRS for sensible m² and tolerance-in-m semantics). One round-trip;
    # restore WGS84 at the end.
    metric_gdf = ox.projection.project_gdf(gdf)
    gdf['area_m2'] = metric_gdf.geometry.area.values

    # Filter tiny polygons BEFORE simplification (simplifying a 2 m²
    # noise polygon could collapse it to a degenerate line).
    before = len(gdf)
    keep = gdf['area_m2'] >= _MIN_FOOTPRINT_M2
    gdf = gdf[keep].copy()
    metric_gdf = metric_gdf[keep].copy()
    dropped = before - len(gdf)
    if dropped:
        logging.info(
            f"  → dropped {dropped:,} tiny polygons (<{_MIN_FOOTPRINT_M2:.0f} m²); "
            f"{len(gdf):,} buildings kept")

    # Douglas-Peucker simplify in metric, project back to WGS84.
    simplified_metric = metric_gdf.geometry.simplify(
        tolerance=_SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    n_before_simplify = int(sum(
        len(g.exterior.coords) if hasattr(g, 'exterior') else 0
        for g in metric_gdf.geometry))
    n_after_simplify = int(sum(
        len(g.exterior.coords) if hasattr(g, 'exterior') else 0
        for g in simplified_metric))
    logging.info(
        f"  → DP simplify (tol={_SIMPLIFY_TOLERANCE_M} m): "
        f"{n_before_simplify:,} → {n_after_simplify:,} exterior vertices "
        f"({(n_before_simplify-n_after_simplify)/n_before_simplify*100:.0f} % reduction)")
    simplified_wgs84 = gpd.GeoSeries(simplified_metric, crs=metric_gdf.crs).to_crs('EPSG:4326')
    gdf['geometry'] = simplified_wgs84.values

    return gdf[['building', 'area_m2', 'geometry']]


def main(variant) -> None:
    context = init_context(variant)
    out_name = f'{variant.area_name}_buildings'

    with clipped_pbf(context, variant, out_name) as pbf_path:
        with step('stream parse (pyosmium handler)'):
            handler = BuildingHandler()
            handler.apply_file(pbf_path, locations=True)
        # Top-10 building tag summary so the downstream dasymetric step
        # has a known baseline to compare against.
        from collections import Counter
        tag_counts = Counter(b for _, b, _ in handler.features)
        n_features = len(handler.features)
        logging.info(
            f"  → collected {n_features:,} closed-way buildings "
            f"across {len(tag_counts)} distinct `building=*` tags")
        for tag, count in tag_counts.most_common(10):
            logging.info(f"      {tag}: {count:,}")

    with step('build GeoDataFrame (WGS84) + simplify + compute area_m2'):
        gdf = features_to_gdf(handler.features)

    # Save via `create_shapes` — buildings is a canonical (scaffolding-
    # only) unit, so the file lands at `shapes/buildings_<area_name>.gpkg`
    # alongside cells/zones/etc. `extra_columns=['building']` keeps the
    # OSM `building=*` tag in the shape file (identity-level attribute)
    context.create_shapes(gdf, data_name=variant.area_name, extra_columns=['building'])
    context.close()


if __name__ == '__main__':
    variants.run(main)
