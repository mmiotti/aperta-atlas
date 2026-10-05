"""
Build the Swiss NPVM-2023 traffic-zones layer (domestic + foreign), with weighted
centroids, land-area (lakes removed), and internal-distance.

Source files:

  - **All zones** (`Verkehrszonen_Ausland_NPVM_2023.gpkg`, EPSG:4326) — despite
    the name ("Ausland" = foreign), this file ships BOTH the Swiss-domestic
    NPVM zones AND the border-region foreign zones together. Reprojected to LV95.
  - **Custom centroids** (`1_Verkehrszonen_Schweiz_Zentroide_NPVM_2023.gpkg`,
    LV03 → reprojected) — weighted representative points for Swiss-domestic
    zones only (e.g. population- or OD-weighted). Better routing-analysis
    anchor than `geometry.centroid` for the zones that have them.

`is_domestic` is derived from the centroid join: a zone is domestic iff a
custom centroid is shipped for it. Foreign zones fall back to
`geometry.centroid` (TODO once networks are prepared: replace with
`aperta.network_snap.transport_centroid` for snap-quality anchors).

Zone IDs are prefixed `ZD<n>` for domestic / `ZF<n>` for foreign to keep
the two populations visually distinguishable in joins and plots.

Lake removal only applies to domestic zones — no equivalent water-body
dataset is available abroad, so foreign rows pass through unchanged into
`traffic_zones_without_lakes.gpkg`.

Inputs:
    Verkehrszonen_Ausland_NPVM_2023.gpkg            # ALL zones (mis-named)
    1_Verkehrszonen_Schweiz_Zentroide_NPVM_2023.gpkg # Swiss-only custom centroids
Inputs (same-namespace; prior stage):
    water_bodies.gpkg  (from tlm_regio.py)

Outputs (PUBLIC, under preparation/switzerland/general/):
    traffic_zones.gpkg
    traffic_zones.csv
    traffic_zones_without_lakes.gpkg
    traffic_zones_centroids.gpkg

Run:
    python -m preparation.switzerland.public.general.traffic_zones
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from aperta import data_processing
from aperta.errors import DataError
from aperta_atlas.context import Context, Storage, init_context

from preparation.switzerland.common import CRS_CH


# Common columns of the harmonised output (`metro` may be NaN — only
# Swiss-domestic agglomerations have it).
_HARMONISED_COLS = ['zone_id', 'is_domestic', 'nuts2', 'nuts3', 'metro',
                    'municipality', 'centroid_x', 'centroid_y', 'geometry']


def load_zones(context: Context) -> gpd.GeoDataFrame:
    """Load all NPVM zones from the (mis-named) 'Ausland' file and attach
    custom Swiss centroids. Returns a GeoDataFrame indexed by `zone_id`
    (`ZD<n>` for domestic / `ZF<n>` for foreign) in LV95.

    `is_domestic` is derived from the centroid merge: True iff a custom
    centroid is shipped for the zone.
    """

    # Despite its name, the Ausland file contains BOTH Swiss-domestic and
    # foreign zones — there's no need to also load the separate
    # `Verkehrszonen_Schweiz_NPVM_2023.gpkg` (which would duplicate the
    # Swiss zones).
    input_dir = context.raw_path(Storage.PUBLIC, 'switzerland/npvm/2023/traffic_zones')
    zones = gpd.read_file(f'{input_dir}/Verkehrszonen_Ausland_NPVM_2023.gpkg').to_crs(CRS_CH)
    zones['ID_Zone'] = zones['ID_Zone'].astype(int)
    zones = zones.set_index('ID_Zone')

    # Custom centroids (Swiss-only shipment; LV03 → LV95).
    centroids = gpd.read_file(f'{input_dir}/1_Verkehrszonen_Schweiz_Zentroide_NPVM_2023.gpkg').to_crs(CRS_CH)
    centroids['ID_Zone'] = centroids['No'].astype(int)
    centroids = centroids.set_index('ID_Zone')
    centroids['centroid_x'] = centroids.geometry.x
    centroids['centroid_y'] = centroids.geometry.y

    # Attach custom centroids (Swiss only — foreign rows get NaN here,
    # filled with geometric centroid below). The merge key is the
    # zone number column: `ID_Zone` in the combined file vs `No` in the
    # centroids file (BFS keeps these consistent for Swiss zones).
    zones = zones.join(centroids[['centroid_x', 'centroid_y']])

    # Domestic ⇔ a custom centroid was joined.
    zones['is_domestic'] = zones['centroid_x'].notna()

    # Foreign zones: fall back to geometric centroid.
    foreign_mask = ~zones['is_domestic']
    foreign_centroids = zones.loc[foreign_mask].geometry.centroid
    zones.loc[foreign_mask, 'centroid_x'] = foreign_centroids.x.to_numpy()
    zones.loc[foreign_mask, 'centroid_y'] = foreign_centroids.y.to_numpy()

    # zone_id with disambiguating prefix.
    prefix = zones['is_domestic'].map({True: 'ZD', False: 'ZF'})
    zones['zone_id'] = prefix + zones.index.astype(int).astype(str)

    # Metro name only exists in domestic zone shapes (why...?)
    zd = gpd.read_file(f'{input_dir}/Verkehrszonen_Schweiz_NPVM_2023.gpkg').to_crs(CRS_CH)
    zd['ID_Zone'] = zd['No'].astype(int)
    zd = zd.set_index('ID_Zone')    
    zd = zd.rename(columns={'N_Agglo': 'metro', 'N_Gem': 'municipality'})
    zones = zones.rename(columns={'NUTS2': 'nuts2', 'NUTS3': 'nuts3'})
    zones = zones.join(zd[['metro', 'municipality']])
    zones = gpd.GeoDataFrame(zones[_HARMONISED_COLS], crs=CRS_CH)
    return zones.set_index('zone_id')


def _add_centroid_lat_lon(zones: gpd.GeoDataFrame, from_crs: str,
                          to_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """Add `centroid_lat` / `centroid_lon` derived from `centroid_x` / `centroid_y`."""
    transformer = Transformer.from_crs(from_crs, to_crs)
    lat, lon = transformer.transform(zones['centroid_x'].to_numpy(),
                                     zones['centroid_y'].to_numpy())
    zones['centroid_lat'] = lat
    zones['centroid_lon'] = lon
    return zones


def remove_lakes_from_domestic(
    zones: gpd.GeoDataFrame, lakes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Cut lake polygons out of domestic zone polygons; foreign rows pass through
    unchanged (no foreign water-body dataset available)."""
    domestic_mask = zones['is_domestic']
    domestic = zones.loc[domestic_mask]
    foreign = zones.loc[~domestic_mask]

    domestic_cut = gpd.overlay(domestic.reset_index(), lakes,
                               how='difference', keep_geom_type=False)
    domestic_cut = domestic_cut.set_index('zone_id')

    out = gpd.GeoDataFrame(pd.concat([domestic_cut, foreign]), crs=zones.crs)
    if set(out.index) != set(zones.index):
        raise DataError("Zones without lakes do not have same index as full zones.")
    return out.loc[zones.index]  # preserve original row order


def main():
    context = init_context()

    zones = load_zones(context)
    zones = _add_centroid_lat_lon(zones, from_crs=CRS_CH)

    lakes = context.get_generic('water_bodies.gpkg')
    zones_without_lakes = remove_lakes_from_domestic(zones, lakes)
    zones_without_lakes['area_m2_geometry'] = np.round(
        zones_without_lakes.geometry.area, 0,
    )
    zones = zones.join(zones_without_lakes[['area_m2_geometry']])
    zones = zones.rename(columns={'area_m2_geometry': 'land_area_m2'})

    context.create_generic(zones, 'traffic_zones.gpkg')
    context.create_generic(zones.drop(columns='geometry'), 'traffic_zones.csv')
    context.create_generic(zones_without_lakes, 'traffic_zones_without_lakes.gpkg')

    # Centroid-as-geometry shape — easier to plot, snap-to-network, etc.
    zones_centroids = zones.copy()
    zones_centroids.geometry = gpd.points_from_xy(
        zones['centroid_x'], zones['centroid_y'], crs=zones.crs,
    )
    context.create_generic(zones_centroids, 'traffic_zones_centroids.gpkg')
    context.close()


if __name__ == '__main__':
    main()
