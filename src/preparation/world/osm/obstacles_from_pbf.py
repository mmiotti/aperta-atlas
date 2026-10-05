"""
Stream OSM obstacle features from a locally clipped PBF — traffic signals,
stops, give-ways, crossings, mini-roundabouts, and roundabout-way centroids.

Independent of the network preparation chain; only depends on
`clip_pbf.py`'s output. Consumed by `networks_decorate.py`, which
snaps each obstacle to the nearest consolidated-network node and writes
a per-node boolean attribute (`is_traffic_signal`, etc.).

Output is a single per-area GeoDataFrame of point features; obstacles
are mode-agnostic (a roundabout is a roundabout regardless of routing
mode), so the same file feeds all per-mode decoration runs.

Obstacle kinds collected:

  Node features (`highway=...`):
    - traffic_signals — signalized intersection
    - stop            — stop sign
    - give_way        — yield sign
    - crossing        — pedestrian / cyclist crossing
    - mini_roundabout — small single-node roundabout

  Way features (`junction=roundabout`):
    - roundabout      — full roundabout loop; output as the way's
                        geometric centroid

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_obstacles` variant per area.

Inputs (PUBLIC, under raw/global/osm/):
    <pbf_name>-latest.osm.pbf       # output of `clip_pbf.py`

Outputs (PUBLIC, under preparation/world/osm/):
    obstacles_<area_name>.gpkg      # point features, WGS84

Requires `osmium-tool` (CLI) AND `pyosmium` (Python).

Run after `clip_pbf.py`.

Run all variants sequentially (default):
    python -m preparation.world.osm.obstacles_from_pbf
Single variant:
    python -m preparation.world.osm.obstacles_from_pbf --variant switzerland_obstacles
    python -m preparation.world.osm.obstacles_from_pbf --variant bern_obstacles
"""

import logging

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString

from aperta_atlas.context import init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.osm.common import clipped_pbf


# Buffer = widest per-mode for the area; obstacles align with the
# widest network's spatial extent so cross-border highway interchanges
# + traffic signals near the border are captured.
variants = Variants([
    ('place', str), ('area_name', str), ('pbf_name', str), ('buffer', int),
])
for area in AREAS.values():
    variants.add(
        name=f'{area.name}_obstacles',
        place=area.place, area_name=area.name,
        pbf_name=f'{area.name}_buffered',
        buffer=widest_buffer(area),
    )


# Node `highway=...` values treated as obstacles. Each becomes a Point
# feature with `kind` set to the matched value.
_OBSTACLE_NODE_HIGHWAYS: frozenset[str] = frozenset({
    'traffic_signals',
    'stop',
    'give_way',
    'crossing',
    'mini_roundabout',
})


class ObstacleHandler(osmium.SimpleHandler):
    """Streaming pyosmium handler that collects OSM obstacle features.

    Nodes whose `highway` tag is in `_OBSTACLE_NODE_HIGHWAYS` are emitted
    as Point features with `kind = highway-tag-value`. Ways with
    `junction=roundabout` are emitted as Point features with
    `kind='roundabout'`, located at the way's geometric centroid.

    `locations=True` (passed to `apply_file`) is required so way nodes
    have resolved coordinates for centroid computation.
    """

    def __init__(self) -> None:
        super().__init__()
        # (kind, osm_type, osm_id, lon, lat). `osm_type` is 'n' (node)
        # or 'w' (way) since OSM ids are unique only within a type — a
        # node 42 and a way 42 are unrelated.
        self.features: list[tuple[str, str, int, float, float]] = []

    def node(self, n) -> None:
        if 'highway' not in n.tags:
            return
        kind = n.tags['highway']
        if kind in _OBSTACLE_NODE_HIGHWAYS:
            self.features.append((kind, 'n', n.id, n.location.lon, n.location.lat))

    def way(self, w) -> None:
        if w.tags.get('junction') != 'roundabout':
            return
        coords = [(n.location.lon, n.location.lat)
                  for n in w.nodes if n.location.valid()]
        if len(coords) < 2:
            return
        centroid = LineString(coords).centroid
        self.features.append(('roundabout', 'w', w.id, centroid.x, centroid.y))


def features_to_gdf(features: list[tuple[str, str, int, float, float]]) -> gpd.GeoDataFrame:
    """Convert a list of `(kind, osm_type, osm_id, lon, lat)` tuples to
    a GeoDataFrame of WGS84 points with a default integer index.

    `osm_id` alone isn't unique across element types (a node and a way
    can share the same numeric id), so we keep `osm_type` + `osm_id` as
    columns for traceability and use a fresh integer index that
    `Context.create_generic`'s uniqueness check accepts.
    """
    df = pd.DataFrame(features, columns=['kind', 'osm_type', 'osm_id', 'lon', 'lat'])
    df.index.name = 'obstacle_id'
    gdf = gpd.GeoDataFrame(
        df[['kind', 'osm_type', 'osm_id']],
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )
    return gdf


def main(variant) -> None:
    context = init_context(variant)
    out_name = f'obstacles_{variant.area_name}'

    with clipped_pbf(context, variant, out_name) as pbf_path:
        with step('stream parse (pyosmium handler)'):
            handler = ObstacleHandler()
            handler.apply_file(pbf_path, locations=True)
        # By-kind summary so the downstream decoration step has a known
        # baseline to compare against after snap-to-node.
        from collections import Counter
        kind_counts = Counter(f[0] for f in handler.features)
        logging.info(
            f"  → collected {len(handler.features):,} obstacles "
            f"({dict(sorted(kind_counts.items()))})")

    with step('build GeoDataFrame'):
        gdf = features_to_gdf(handler.features)

    context.create_generic(gdf, f'{out_name}.gpkg')
    context.close()


if __name__ == '__main__':
    variants.run(main)
