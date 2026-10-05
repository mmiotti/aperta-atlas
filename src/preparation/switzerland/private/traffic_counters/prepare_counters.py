"""
Prepare Swiss traffic counter measurements (NPVM zaehldaten).

> **Private data.** Requires the NPVM zaehldaten gpkg (Swiss traffic
> counts pooled from several sources, restricted-access); not redistributable. Published
> as documentation of the method.

Reads the NPVM zaehldaten gpkg (per-direction line geometries with annual
average daily traffic), derives `traffic_cars` from the dwv_pw fallback,
classifies each counter as highway / main / local by lane count + flow
threshold, computes the line-midpoint bearing for later matching against
directed routing-graph edges, and writes one point per counter midpoint.

Inputs (PRIVATE):
    <DATA_DIR_PRIVATE>/raw/switzerland/traffic_counters/ch_npvm/npvm_zaehldaten_2023.gpkg

Outputs (PRIVATE):
    traffic_counters.gpkg          # point geometries (line midpoints)
                                   # cols: traffic_cars, bearing_deg,
                                   #       is_highway, is_main, is_local

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
    gdf['traffic_cars'] = gdf['dwv_pw']
    gdf.loc[gdf['traffic_cars'] < 1, 'traffic_cars'] = gdf['dwv_allefz'] * 0.9
    gdf['is_highway'] = (
        ((gdf['NUMLANES'] >= 2) & (gdf['zw_quelle'] == 'ASTRA')) &
        (gdf['traffic_cars'] > 3_000)
    ).astype(int)
    gdf['is_main'] = ((gdf['traffic_cars'] > 1_000) & (gdf['is_highway'] == 0)).astype(int)
    gdf['is_local'] = ((gdf['is_highway'] == 0) & (gdf['is_main'] == 0)).astype(int)
    road_types = ['is_highway', 'is_main', 'is_local']
    # Compass bearing at each counter's midpoint, for matching to directed edges
    # in the routing graph later. Counters are directional (each row reports
    # one direction of travel), so the bearing is meaningful as-is — no
    # modulo-180° reduction.
    gdf['bearing_deg'] = geo_processing.line_bearings_deg(gdf.geometry)
    geom = [line.interpolate(0.5, normalized=True) for line in gdf.geometry]
    points = gpd.GeoDataFrame(gdf[['traffic_cars', 'bearing_deg'] + road_types],
                              geometry=geom, crs=gdf.crs)
    points = points[points.traffic_cars > 0]
    context.create_generic(points, 'traffic_counters.gpkg')
    context.close()


if __name__ == '__main__':
    main()
