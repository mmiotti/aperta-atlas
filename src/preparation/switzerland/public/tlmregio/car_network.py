"""
Simplified car network from swissTLMRegio, in the layout of the OSM car network.

swissTLMRegio (swisstopo, 1:200,000) holds the main road network of
Switzerland and its neighbouring countries: motorways, expressways, main,
connecting and minor roads, without most of the local street network. Its
road lines end at the junctions, so they form a routable network as they are.
This script turns it into a car network that the atlas pipeline can use in
place of the OSM one (simplified-network test of the Technical Validation;
also a template for regions with only a coarse road dataset):

  1. Roads: the `tlmregio_transportation_road` layer without footpaths, farm
     and forest tracks (`Fahrstraes`: narrow rural access roads, a quarter
     unpaved, often closed to general traffic), roads banned for cars and roads
     under construction, kept where they intersect the area's buffered polygon
     (`<area>_buffered.poly`, the car buffer of the OSM extracts). This leaves
     about 25,700 km inside Switzerland (OSM car network: 70,900 km; LUMOS 2020
     network: 28,500 km). At 1:200,000, settlements are areas: their internal
     streets are not part of the dataset.
  2. Classes (2 × 2, as in LUMOS): motorway (motorways, directional motorway
     carriageways and expressways; ramps count as other roads) or other, and
     urban (at least half of the length inside settlement or city-centre
     land cover) or rural. The class sets the speed prior (`_SPEED_PRIOR_KPH`;
     ramps get their own) that the edge-weight calibration (04) scales; the
     `objval` class maps to an OSM-like `highway` value, so the OSM-based
     stages read it unchanged.
  3. Graph: nodes at the line ends (rounded to 0.1 m), both directions per
     road (swissTLMRegio has no one-way information), largest connected
     component, WGS84 like the OSM graphs.
  4. Decoration as in `preparation/world/osm/networks_decorate.py`: junction
     topology, highway ranks, lanes per direction, `speed_kph` / `travel_time`;
     the OSM node-obstacle flags (signals, stops, …) are 0, as swissTLMRegio
     has no such objects. In their place, `is_4way_urban` marks 4-way junctions
     whose roads are mostly urban (where most signals are), so that 04 can fit
     a separate urban junction delay.

Inputs:
    raw/switzerland/swisstlmregio/swissTLMRegio_Product_LV95.gpkg
        tlmregio_transportation_road, tlmregio_landcover_landcover
    raw/global/osm/<area>_buffered.poly

Outputs (PUBLIC, under preparation/switzerland/tlmregio/):
    nw/<area>_car_consolidated.graphml
    properties/edges_<area>_car_consolidated.csv                    # base attributes
    properties/{nodes,edges}_<area>_car_consolidated_decorated.csv  # as networks_decorate

Run all areas (default):
    python -m preparation.switzerland.public.tlmregio.car_network
Single area:
    python -m preparation.switzerland.public.tlmregio.car_network --variant bern-metro
"""

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import Polygon

from aperta.network_processing import flag_node_intersection_topology
from aperta_atlas.context import Storage, init_context
from aperta_atlas.graph_simplification import round_coords
from aperta_atlas.osm import flag_node_osm_classification, lanes_per_direction
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

_GPKG = 'switzerland/swisstlmregio/swissTLMRegio_Product_LV95.gpkg'
_CRS = 'EPSG:2056'
_URBAN_LANDCOVER = ('Siedl', 'Stadtzentr')
_URBAN_MIN_SHARE = 0.5
_NODE_ROUND_M = 0.1

# swissTLMRegio road class → OSM-like `highway` value and total lane count.
_CLASSES: dict[str, tuple[str, int]] = {
    # `Autob_Ri`: one carriageway of a split motorway, drawn per direction; as both
    # directions are kept (no one-way information), 2 lanes give 1 lane per direction.
    'Autobahn': ('motorway', 4), 'Autob_Ri': ('motorway', 2),
    'Autobahn_Rampe': ('motorway_link', 2), 'Autob_Ri_Rampe': ('motorway_link', 2),
    'Autostr': ('trunk', 2), 'Autostr_Rampe': ('trunk_link', 2),
    'HauptStrAB6': ('primary', 2), 'HauptStrAB4': ('primary', 2),
    'HauptStrAB_Rampe': ('primary_link', 2),
    'VerbindStr6': ('secondary', 2), 'VerbindStr4': ('secondary', 2),
    'VerbindStr_Rampe': ('secondary_link', 2),
    'NebenStr6': ('tertiary', 2), 'NebenStr3': ('unclassified', 2),
    'NebenStr_Rampe': ('tertiary_link', 2),
}
_MOTORWAY = ('Autobahn', 'Autob_Ri', 'Autostr')
# Speed prior (km/h) per 2 × 2 class; motorways by their legal limit.
_SPEED_PRIOR_KPH = {'motorway': {'Autobahn': 120, 'Autob_Ri': 120, 'Autostr': 100},
                    'urban': 50, 'rural': 80, 'ramp': 60}

# Same decoration columns as networks_decorate.py (obstacle flags are all 0 here).
_OBSTACLE_FLAGS = ('traffic_signal', 'stop', 'give_way', 'crossing', 'mini_roundabout',
                   'roundabout')
_BASE_EDGE_COLUMNS = ['highway', 'maxspeed', 'lanes', 'bridge', 'tunnel', 'objval',
                      'is_motorway', 'is_urban', 'is_motorway_urban', 'urban_share']
_DECORATED_COLUMNS = [
    'highway', 'lanes', 'lanes_per_direction', 'speed_kph', 'travel_time',
    'n_streets', 'is_t_junction', 'is_4way', 'is_4way_urban', 'max_highway_rank',
    'min_highway_rank',
    'is_t_junction_major', 'is_4way_major', 'is_t_junction_anchor', 'is_4way_anchor',
    *(f'is_{flag}' for flag in _OBSTACLE_FLAGS),
]

variants = Variants([('area_name', str)])
for _area in ('switzerland', 'bern-metro'):
    variants.add(name=_area, area_name=_area)


def _read_poly(path: str) -> Polygon:
    """Outer ring of an OSM `.poly` file (lon / lat), as written by
    `preparation.world.common.write_poly_file`."""
    coords = []
    with open(path) as f:
        for line in f.read().splitlines()[2:]:
            if line.strip() == 'END':
                break
            lon, lat = map(float, line.split())
            coords.append((lon, lat))
    return Polygon(coords)


def _speed_prior(objval: str, highway: str, is_urban: bool) -> int:
    if objval in _MOTORWAY:
        return _SPEED_PRIOR_KPH['motorway'][objval]
    if highway.endswith('_link'):
        return _SPEED_PRIOR_KPH['ramp']
    return _SPEED_PRIOR_KPH['urban' if is_urban else 'rural']


def _load_roads(context, area_name: str) -> gpd.GeoDataFrame:
    gpkg = context.raw_path(Storage.PUBLIC, _GPKG)
    roads = gpd.read_file(gpkg, layer='tlmregio_transportation_road',
                          columns=['objval', 'banned', 'underconst', 'construct'])
    roads = roads.set_crs(_CRS, allow_override=True)
    roads['geometry'] = shapely.force_2d(roads.geometry)
    keep = (roads['objval'].isin(_CLASSES) & (roads['banned'] != 'True')
            & (roads['underconst'] == 0))
    roads = roads[keep].reset_index(drop=True)
    area = gpd.GeoSeries([_read_poly(context.raw_path(
        Storage.PUBLIC, f'global/osm/{area_name}_buffered.poly'))], crs='EPSG:4326').to_crs(_CRS)
    roads = roads[roads.intersects(area.iloc[0])].reset_index(drop=True)

    landcover = gpd.read_file(gpkg, layer='tlmregio_landcover_landcover', columns=['objval'])
    settled = landcover[landcover['objval'].isin(_URBAN_LANDCOVER)].set_crs(
        _CRS, allow_override=True)
    # Disjoint settlement pieces; each road is intersected only with the pieces it
    # touches.
    parts = shapely.get_parts(shapely.force_2d(settled.union_all()))
    road_idx, part_idx = shapely.STRtree(parts).query(roads.geometry.values,
                                                      predicate='intersects')
    inside = shapely.length(shapely.intersection(roads.geometry.values[road_idx],
                                                 parts[part_idx]))
    share = pd.Series(inside).groupby(road_idx).sum() / roads.length
    roads['urban_share'] = share.reindex(roads.index).fillna(0.0).clip(0, 1)
    roads['is_urban'] = (roads['urban_share'] >= _URBAN_MIN_SHARE).astype(int)
    roads['highway'] = roads['objval'].map(lambda v: _CLASSES[v][0])
    roads['lanes'] = roads['objval'].map(lambda v: _CLASSES[v][1])
    roads['is_motorway'] = roads['objval'].isin(_MOTORWAY).astype(int)
    roads['is_motorway_urban'] = roads['is_motorway'] * roads['is_urban']
    roads['maxspeed'] = [_speed_prior(o, h, u) for o, h, u
                         in zip(roads['objval'], roads['highway'], roads['is_urban'])]
    roads['bridge'] = (roads['construct'] == 'Brücke').astype(float)
    roads['tunnel'] = (roads['construct'] == 'Tunnel').astype(float)
    return roads


def _build_graph(roads: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Nodes at rounded line ends; one edge per direction; WGS84."""
    starts = shapely.get_point(roads.geometry, 0)
    ends = shapely.get_point(roads.geometry, -1)
    xy = np.round(np.column_stack([
        np.concatenate([shapely.get_x(starts), shapely.get_x(ends)]),
        np.concatenate([shapely.get_y(starts), shapely.get_y(ends)])]) / _NODE_ROUND_M)
    keys, node_of = np.unique(xy, axis=0, return_inverse=True)
    node_of = node_of.reshape(-1)
    u_of, v_of = node_of[:len(roads)], node_of[len(roads):]

    lonlat = gpd.GeoSeries(gpd.points_from_xy(keys[:, 0] * _NODE_ROUND_M,
                                              keys[:, 1] * _NODE_ROUND_M), crs=_CRS).to_crs(4326)
    geoms_wgs = roads.geometry.to_crs(4326)
    # `simplified`: edges carry their full geometry, so `ox.project_graph` projects it.
    g = nx.MultiDiGraph(crs='EPSG:4326', simplified=True)
    for n, p in enumerate(lonlat):
        g.add_node(n, x=p.x, y=p.y)
    attrs = roads[_BASE_EDGE_COLUMNS].to_dict('records')
    for u, v, a, length, geom in zip(u_of, v_of, attrs, roads.length, geoms_wgs):
        if u == v:
            continue   # loops (e.g. roundabouts drawn as one line) carry no route
        g.add_edge(int(u), int(v), length=float(length), geometry=geom, **a)
        g.add_edge(int(v), int(u), length=float(length), geometry=shapely.reverse(geom), **a)
    largest = max(nx.weakly_connected_components(g), key=len)
    return g.subgraph(largest).copy()


def main(variant):
    context = init_context(variant)
    name = f'{variant.area_name}_car_consolidated'

    with step('load + classify swissTLMRegio roads'):
        roads = _load_roads(context, variant.area_name)
        by_class = roads.length.groupby([roads['is_motorway'], roads['is_urban']]).sum() / 1_000
        logging.info(f"  → {len(roads):,} roads, {roads.length.sum() / 1_000:,.0f} km; km by "
                     f"(motorway, urban): {by_class.round(0).to_dict()}")

    with step('build graph'):
        g = _build_graph(roads)
        logging.info(f"  → {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges "
                     f"(largest connected component)")

    with step('decorate (as networks_decorate)'):
        flag_node_intersection_topology(g)
        flag_node_osm_classification(g)
        for _, _, d in g.edges(data=True):
            d['lanes_per_direction'] = lanes_per_direction({**d, 'oneway': False})
        ox.add_edge_speeds(g)        # every edge carries its class prior as `maxspeed`
        ox.add_edge_travel_times(g)
        for n, d in g.nodes(data=True):
            for flag in _OBSTACLE_FLAGS:
                d[f'is_{flag}'] = 0
            # Both directions of every road are edges, so the out-edges cover all roads.
            urban = np.mean([e['is_urban'] for _, _, e in g.out_edges(n, data=True)])
            d['is_4way_urban'] = int(d.get('is_4way', 0) == 1 and urban >= _URBAN_MIN_SHARE)
        n_4way = sum(d.get('is_4way', 0) == 1 for _, d in g.nodes(data=True))
        n_4way_urban = sum(d['is_4way_urban'] for _, d in g.nodes(data=True))
        logging.info(f"  → {n_4way:,} 4-way junctions, {n_4way_urban:,} of them urban")
        round_coords(g, decimals=7)

    with step('save graph + base and decorated properties'):
        context.create_nw(g, data_name=name, save_properties='edges',
                          properties_columns=_BASE_EDGE_COLUMNS)
        context.create_nw(g, data_name=name, save_skeleton=False, save_properties=True,
                          properties_name='decorated', properties_columns=_DECORATED_COLUMNS)
    context.close()


if __name__ == '__main__':
    variants.run(main)
