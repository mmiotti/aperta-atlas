"""
Prepare Swiss traffic counter measurements (NPVM zaehldaten).

> **Private data.** Requires the NPVM zaehldaten gpkg (Swiss traffic
> counts pooled from several sources, restricted-access); not redistributable. Published
> as documentation of the method.

Reads the NPVM zaehldaten gpkg (per-direction line geometries with annual
average weekday traffic), takes `traffic_cars` from `dwv_pw` (passenger cars;
0.9 × `dwv_allefz` where cars are not counted separately), flags the motorway stations
(`is_highway`: Federal Roads Office stations with two or more lanes; by the
station alone, never by its count, so the validation is not selected on the
observed value), computes the line-midpoint bearing for later matching
against directed routing-graph edges, and writes one point per counter
midpoint.

Inputs (PRIVATE, under raw/switzerland/traffic_counters/ch_npvm/):
    npvm_zaehldaten_2023.gpkg

Outputs (PRIVATE, under preparation/switzerland/traffic_counters/):
    traffic_counters.gpkg          # point geometries (line midpoints)
                                   # cols: traffic_cars, bearing_deg,
                                   #       is_highway

Run:
    python -m preparation.switzerland.private.traffic_counters.prepare_counters
"""

import geopandas as gpd

from aperta import geo_processing
from aperta_atlas.context import Storage, init_context


def main() -> None:
    context = init_context()
    folder = context.raw_path(Storage.PRIVATE, 'switzerland/traffic_counters/ch_npvm')
    gdf = gpd.read_file(f'{folder}/npvm_zaehldaten_2023.gpkg')
    # The file has only DWV (average weekday traffic), no average over all days. `dwv_pw` is 0
    # where cars are not counted separately (over half the stations): 90 % of all vehicles.
    gdf['traffic_cars'] = gdf['dwv_pw']
    no_cars = gdf['traffic_cars'].isna() | (gdf['traffic_cars'] < 1)
    gdf.loc[no_cars, 'traffic_cars'] = gdf['dwv_allefz'] * 0.9
    gdf['is_highway'] = ((gdf['NUMLANES'] >= 2) & (gdf['zw_quelle'] == 'ASTRA')).astype(int)
    # Midpoint compass bearing, for matching to directed edges; counters are directional.
    gdf['bearing_deg'] = geo_processing.line_bearings_deg(gdf.geometry)
    geom = [line.interpolate(0.5, normalized=True) for line in gdf.geometry]
    points = gpd.GeoDataFrame(gdf[['traffic_cars', 'bearing_deg', 'is_highway']],
                              geometry=geom, crs=gdf.crs)
    points = points[points.traffic_cars > 0]
    context.create_generic(points, 'traffic_counters.gpkg')
    context.close()


if __name__ == '__main__':
    main()
