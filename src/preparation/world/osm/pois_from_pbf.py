"""
Stream OSM POI features from a locally clipped PBF — destinations
(shops, schools, leisure, gastronomy, …), mobility infrastructure
(car/bike parking, car-sharing), and public-transit stops.

Independent of the network preparation chain; only depends on
`clip_pbf.py`'s output. Mirrors `obstacles_from_pbf.py` in structure
(pyosmium streaming handler, per-area variants, GeoDataFrame output).
The categorisation here is broader: each POI category bundles a set of
OSM `key:value` tags with per-tag weights that downstream accessibility
code can sum / aggregate.

POI categories cover three families:

  - `poi_*` — destination types for accessibility (errands, education,
    leisure). Most are nodes (shops, kindergartens); some are commonly
    tagged on ways (parks, schools as building polygons).
  - `mobility_*` — car parking, bicycle parking, car-sharing stations.
    Often tagged on ways (parking lots as polygons).
  - `mobility_transit*` — bus stops, train / tram stations, rail halts.
    Almost exclusively nodes.

For each matching OSM feature the handler emits ONE row per matching
category. A `shop:supermarket` that belongs to BOTH
`poi_errands_supermarkets` AND `poi_errands_groceries` produces two
rows (with category-specific weights), so downstream `groupby('category')`
gives a per-category POI set without further filtering. A feature with
multiple matching tags (rare — e.g. an `amenity:restaurant` also tagged
`amenity:cafe`) similarly emits one row per matching tag.

Way features become point features: the polygon centroid for closed
rings (school buildings, parks), the length-weighted LineString
centroid for open ways (hiking trails). Relations (multipolygon
campuses, complex park boundaries) are not yet handled — V1 covers
node + way only; revisit if specific category coverage looks short.

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_pois` variant per area.

Inputs (PUBLIC, under raw/global/osm/):
    <pbf_name>-latest.osm.pbf       # output of `clip_pbf.py`

Outputs (PUBLIC, under preparation/world/osm/):
    pois_<area_name>.gpkg           # point features, WGS84

Requires `osmium-tool` (CLI) AND `pyosmium` (Python).

Run after `clip_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.pois_from_pbf
Single variant:
    python -m preparation.world.osm.pois_from_pbf --variant switzerland_pois
    python -m preparation.world.osm.pois_from_pbf --variant bern_pois
"""

import logging

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString, Polygon

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.osm.common import clipped_pbf


# Buffer = widest per-mode for the area; POI set spans the widest
# routing network's extent.
variants = Variants([
    ('place', str), ('area_name', str), ('pbf_name', str), ('buffer', int),
])
for area in AREAS.values():
    variants.add(
        name=f'{area.name}_pois',
        place=area.place, area_name=area.name,
        pbf_name=f'{area.name}_buffered',
        buffer=widest_buffer(area),
    )


# Category -> [(tag, weight), ...]. Each tag is an OSM `key:value`
# string. Weights are subjective per-tag importance within the
# category; downstream accessibility code can sum them, gravity-weight
# them, or ignore them and treat each row as count=1. Categories are
# adapted from the historical urbanmobilityatlas POI map (the LUMOS /
# Atlas predecessor) and grouped by life-domain (errands / education /
# leisure / mobility / transit).
POI_CATEGORIES: dict[str, list[tuple[str, float]]] = {
    # ---- Errands ---------------------------------------------------
    # Groceries (broader — includes bakery / farm + speciality shops).
    'poi_errands_groceries': [
        ('shop:convenience', 0.5),
        ('shop:supermarket', 1.5),
        ('shop:bakery',      0.5),
        ('shop:butcher',     0.5),
        ('shop:farm',        0.5),
        ('shop:greengrocer', 0.5),
        ('shop:cheese',      0.5),  # Swiss-relevant
        ('shop:deli',        0.5),
        ('shop:beverages',   0.5),
    ],
    # Services: banking, post, retail services, DIY, personal care.
    # DIY and garden centres land here (rather than in a separate
    # category) because the Swiss MTMC travel survey lumps them under
    # "errands". Healthcare (incl. pharmacy) is in `poi_healthcare`;
    # childcare is in `poi_education_preschool` alongside kindergarten.
    'poi_errands_services': [
        ('amenity:post_office',  1.0),
        ('amenity:bank',         1.0),
        ('shop:electronics',     0.5),
        ('shop:doityourself',    1.0),
        ('shop:garden_centre',   1.0),
        ('shop:dry_cleaning',    0.5),
        ('shop:hairdresser',     1.0),
        ('shop:kiosk',           1.0),
        ('shop:wine',            0.5),
        ('shop:optician',        0.5),
        ('shop:books',           1.0),
        ('shop:newsagent',       0.5),
        ('shop:florist',         0.5),
    ],
    # ---- Healthcare ------------------------------------------------
    # Split out of errands_services so "services" doesn't conflate
    # admin/retail with medical. Pharmacy stays in scope for routine
    # access; clinic weighted highest because clinics are
    # multi-physician hubs (fewer features, more visits per feature).
    'poi_healthcare': [
        ('amenity:pharmacy', 1.5),
        ('amenity:doctors',  1.5),  # plural in OSM
        ('amenity:dentist',  1.0),
        ('amenity:clinic',   2.0),
    ],
    # ---- Education -------------------------------------------------
    # Both `kindergarten` and `childcare` here — Swiss "Kita"
    # facilities are often tagged with both, which double-counts
    # within this category. Acceptable: the category is meant to
    # capture "place a small child goes during the day" and either
    # tag signals that.
    'poi_education_preschool': [
        ('amenity:kindergarten', 1.0),
        ('amenity:childcare',    1.0),
    ],
    'poi_education_school': [
        ('amenity:school', 1.0),
    ],
    # Libraries land in `_higher` rather than their own slot because
    # they're explicitly educational institutions that serve all ages
    # (children's section through to academic / university libraries
    # at one end of the spectrum). If "primary education" matters as
    # a separate signal later, move out into a dedicated slot.
    'poi_education_higher': [
        ('amenity:university', 1.0),
        ('amenity:college',    1.0),
        ('amenity:library',    1.5),
    ],
    # ---- Leisure ---------------------------------------------------
    'poi_leisure_gastronomy': [
        ('amenity:restaurant', 1.0),
        ('amenity:fast_food',  0.5),  # McDonald's, kebab, döner — common
        ('amenity:cafe',       1.0),
        ('amenity:bar',        1.0),
        ('amenity:pub',        1.0),
        ('amenity:biergarten', 1.0),  # ~10-20 features in CH; near-noise
        ('amenity:ice_cream',  1.0),
    ],
    # Amenities (cultural / entertainment / shopping malls + some leisure
    # venues). Tourism destinations (museums, resorts, beach_resort)
    # live in `poi_tourism` so this category stays focused on local /
    # everyday cultural amenities.
    'poi_leisure_amenity': [
        ('amenity:theatre',          2.0),
        ('amenity:cinema',           2.0),
        ('amenity:concert_hall',     3.0),  # only ~11 in Switzerland
        ('amenity:community_centre', 1.0),
        ('amenity:arts_centre',      1.0),
        ('amenity:nightclub',        1.5),
        ('shop:gift',                1.0),
        ('shop:clothes',             0.5),
        ('shop:beauty',              0.5),
        ('shop:mall',                2.0),
        ('leisure:amusement_arcade', 1.0),
        ('leisure:bowling_alley',    1.0),
        ('leisure:dance',            1.0),
        ('leisure:escape_game',      1.0),
        ('leisure:bathing_place',    0.5),
        ('leisure:stadium',          3.0),
        ('leisure:water_park',       0.5),
        ('leisure:playground',       0.5),
        ('leisure:dog_park',         1.0),
    ],
    'poi_leisure_hiking': [
        # ~50k+ guideposts on Swiss hiking trails; low weight prevents
        # them from dominating the category by count.
        ('information:guidepost',   0.2),
        ('highway:trailhead',       1.0),
        ('leisure:park',            0.5),
        ('leisure:nature_reserve',  5.0),
    ],
    # Active leisure (sports + outdoor). `leisure:swimming_pool` excluded
    # because OSM mixes private + public (~45k pools in CH). `leisure:pitch`
    # weighted low because every tennis / ping-pong pitch is tagged.
    'poi_leisure_sports': [
        ('leisure:fitness_centre', 2.0),
        ('leisure:fitness_station',0.5),
        ('leisure:golf_course',    2.0),  # numerous in Switzerland
        ('leisure:horse_riding',   1.0),
        ('leisure:ice_rink',       1.0),
        ('leisure:pitch',          0.2),
        ('leisure:miniature_golf', 1.0),
        ('leisure:sports_centre',  2.0),
        ('leisure:swimming_area',  1.0),
        ('leisure:track',          1.0),
    ],
    # ---- Tourism --------------------------------------------------
    # Tourism destinations carved out of `poi_leisure_amenity` so the
    # amenity slot stays focused on local/everyday culture and this
    # slot captures the kinds of POIs that draw cross-region trips.
    # Resorts moved here from amenity for the same reason.
    'poi_tourism': [
        ('tourism:museum',       2.0),
        ('tourism:gallery',      0.5),
        ('tourism:zoo',          3.0),
        ('tourism:theme_park',   3.0),
        ('leisure:beach_resort', 0.5),
        ('leisure:resort',       0.5),
    ],
    # ---- Mobility infrastructure -----------------------------------
    # `amenity:parking_space` is OSM's per-stall tag, often nested inside
    # `amenity:parking` polygons; low weight avoids double-counting where
    # both are mapped.
    'mobility_parking_cars': [
        ('amenity:parking',       1.0),
        ('amenity:parking_space', 0.2),
    ],
    'mobility_parking_bicycles': [
        ('amenity:bicycle_parking', 1.0),
    ],
    'mobility_service_sharing': [
        ('amenity:car_sharing',     1.0),
        ('amenity:bicycle_rental',  1.0),
    ],
    # ---- Public transit -------------------------------------------
    # `mobility_transit` is the union of bus + rail + tram (anything a
    # commuter steps onto). `mobility_transit_rail` excludes buses;
    # `mobility_transit_train` excludes tram (heavy rail only).
    #
    # `public_transport:platform` covers the modern OSM PT schema that
    # increasingly replaces `highway:bus_stop` + `railway:tram_stop` on
    # re-mapped Swiss city stops. Only included in the union since
    # generic platforms don't disambiguate bus / tram / rail without
    # extra `bus=yes` / `train=yes` sub-tags we don't read here.
    'mobility_transit': [
        ('highway:bus_stop',           0.5),
        ('railway:station',            3.0),
        ('railway:halt',               1.0),
        ('railway:tram_stop',          1.0),
        ('public_transport:platform',  0.5),
    ],
    'mobility_transit_rail': [
        ('railway:station',    3.0),
        ('railway:halt',       1.0),
        ('railway:tram_stop',  1.0),
    ],
    'mobility_transit_train': [
        ('railway:station',    3.0),
        ('railway:halt',       1.0),
    ],
}


def _build_tag_index(
    categories: dict[str, list[tuple[str, float]]],
) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Invert the category -> tags map to a (osm_key, osm_value) ->
    [(category, weight), ...] index for O(1) lookup in the handler.
    """
    out: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for cat, specs in categories.items():
        for tag, weight in specs:
            key, value = tag.split(':', 1)
            out.setdefault((key, value), []).append((cat, weight))
    return out


_TAG_INDEX: dict[tuple[str, str], list[tuple[str, float]]] = _build_tag_index(POI_CATEGORIES)


def _way_centroid(coords: list[tuple[float, float]]):
    """Point centroid of a way. Closed rings -> Polygon (area) centroid;
    open ways -> LineString (length-weighted) centroid. `coords` must
    have >= 2 points; closed rings need >= 4 (including the duplicate
    closing point) to form a valid polygon.
    """
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return Polygon(coords).centroid
    return LineString(coords).centroid


class POIHandler(osmium.SimpleHandler):
    """Streaming pyosmium handler that collects OSM POI features.

    For each node or way, iterates its tags and consults `_TAG_INDEX`
    to find matching POI categories. Emits ONE row per (feature,
    category) match — a feature in multiple categories yields multiple
    rows, which simplifies downstream `groupby('category')`.

    `locations=True` (passed to `apply_file`) is required so way nodes
    have resolved coordinates for centroid computation.
    """

    def __init__(self) -> None:
        super().__init__()
        # (category, osm_key, osm_value, weight, osm_type, osm_id, lon, lat).
        # `osm_type` is 'n' or 'w' — OSM ids are unique only within a type.
        self.features: list[
            tuple[str, str, str, float, str, int, float, float]
        ] = []

    def node(self, n) -> None:
        for tag in n.tags:
            matches = _TAG_INDEX.get((tag.k, tag.v))
            if matches is None:
                continue
            lon, lat = n.location.lon, n.location.lat
            for cat, weight in matches:
                self.features.append(
                    (cat, tag.k, tag.v, weight, 'n', n.id, lon, lat))

    def way(self, w) -> None:
        # Tag check first — most ways have no POI tag and can skip the
        # expensive coordinate build entirely.
        tag_matches: list[tuple[str, str, list[tuple[str, float]]]] = []
        for tag in w.tags:
            matches = _TAG_INDEX.get((tag.k, tag.v))
            if matches is not None:
                tag_matches.append((tag.k, tag.v, matches))
        if not tag_matches:
            return
        coords = [(n.location.lon, n.location.lat)
                  for n in w.nodes if n.location.valid()]
        if len(coords) < 2:
            return
        centroid = _way_centroid(coords)
        for k, v, cats_weights in tag_matches:
            for cat, weight in cats_weights:
                self.features.append(
                    (cat, k, v, weight, 'w', w.id, centroid.x, centroid.y))


def features_to_gdf(
    features: list[tuple[str, str, str, float, str, int, float, float]],
) -> gpd.GeoDataFrame:
    """Convert a list of POI tuples to a WGS84 GeoDataFrame with a
    default integer index named `poi_id`.

    Columns: `category`, `osm_key`, `osm_value`, `weight`, `osm_type`,
    `osm_id`, `geometry`. Multiple rows for the same OSM feature are
    allowed (one per matching category); `(osm_type, osm_id)` alone is
    NOT unique. The fresh integer index gives `Context.create_generic`'s
    uniqueness check something it accepts.
    """
    df = pd.DataFrame(features, columns=[
        'category', 'osm_key', 'osm_value', 'weight',
        'osm_type', 'osm_id', 'lon', 'lat',
    ])
    df.index.name = 'poi_id'
    gdf = gpd.GeoDataFrame(
        df[['category', 'osm_key', 'osm_value', 'weight',
            'osm_type', 'osm_id']],
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )
    return gdf


def main(variant) -> None:
    context = init_context(variant)
    out_name = f'pois_{variant.area_name}'

    with clipped_pbf(context, variant, out_name) as pbf_path:
        with step('stream parse (pyosmium handler)'):
            handler = POIHandler()
            handler.apply_file(pbf_path, locations=True)
        # By-category summary so the downstream step has a baseline to
        # compare against after snap-to-network-node.
        from collections import Counter
        cat_counts = Counter(f[0] for f in handler.features)
        n_features = len(handler.features)
        logging.info(
            f"  → collected {n_features:,} POI feature-category rows "
            f"across {len(cat_counts)} categories")
        for cat in sorted(cat_counts):
            logging.info(f"      {cat}: {cat_counts[cat]:,}")

    with step('build GeoDataFrame'):
        gdf = features_to_gdf(handler.features)

    with step('save pois.gpkg'):
        context.create_generic(gdf, f'{out_name}.gpkg')

    context.close()


if __name__ == '__main__':
    variants.run(main)
