"""
Condensed image story for the Swiss Urban Mobility Atlas methodology.

A trimmed version of `visualization/story.py` — same visual grammar (locked
plot rectangle, right decoration panel, per-scenario focal point) but only
the ~10-14 frames that carry the paper's methods narrative:

  1. Network preprocessing — raw OSM edges
  2. Network preprocessing — consolidated edges (colored by OSM speed_kph)
     with reattached traffic signals highlighted
  3. Cell layer
  4. Zone layer
  5. Virtual-node insertion (snap of cells to network)
  6. Traffic-flow estimation
  7. Calibrated edge speeds — bike (rbike)
  8. Calibrated edge speeds — car_peak
  9. Calibrated edge speeds — car_night
 10. Cell-level transit-access overhead (walk-to-transit)
 11. Cell-level car overhead (from 07a's fit)
 12. Accessibility — car
 13. Accessibility — walk
 14. Accessibility — cross-modal fastest

All shared infrastructure (StoryConfig, styles, helpers, frames 3-14) is
imported from `visualization.story`. This module only defines the two
genuinely new frames (raw OSM + consolidated OSM with speeds/signals,
transit-access overhead) and a zones-focused frame; everything else is
delegated.

Run:
    python -m visualization.story_condensed --scenario <name>
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from aperta_atlas.context import init_context
from aperta_atlas.utils import step


# Reuse EVERYTHING from the full story: config, styles, helpers, existing
# frame functions. This module only adds new frames on top.
import visualization.story as story
from visualization.story import (
    StoryConfig,
    _BASEMAP,
    _CELL_EDGE,
    _CELL_LW,
    _LABEL_FONTSIZE,
    _LABEL_PAD,
    _NATIVE_NODE_COLOR,
    _NATIVE_NODE_SIZE,
    _NETWORK_ALPHA,
    _NETWORK_COLOR,
    _NETWORK_WIDTH,
    _TICK_FONTSIZE,
    _TITLE_COLOR,
    _add_basemap,
    _clip_to_extent,
    _load_cells,
    _load_zones,
    add_colorbar,
    add_legend,
    make_fig,
    # Reused frames (imported for the _FRAMES list below).
    frame_11_calibrated_speeds,       # car_peak
    frame_11b_calibrated_speeds_night,
)

from scenarios import get_scenario


# --- Story-wide overrides for the condensed narrative -----------------
# Titles are added in the presentation slides, not the frames, and the
# red focal-point marker isn't useful without a story about that
# specific location. Kill both without editing story.py's frame bodies:
# `_FOCAL_POINT_SIZE` is looked up on `story.` at call time, and `set_title`
# is defined locally so frames in THIS module skip it too.
story._FOCAL_POINT_SIZE = 0
# Slightly thicker calibrated-speed edges so frames 8/9 read as a heat
# map of the network, not a thin scribble.
story._SPEED_LW = 1.3
def set_title(*_args, **_kwargs) -> None:  # noqa: E305
    """No-op — condensed story adds titles in the presentation layer."""
    return None


# =====================================================================
# New helpers — raw OSM network + consolidated network + transit access
# =====================================================================


def _load_raw_osm_edges(context, scenario, mode: str) -> gpd.GeoDataFrame:
    """Load the raw pre-consolidation OSM network for `mode` in the
    scenario's area, project to `story._CRS_MAIN`, return an edges
    GeoDataFrame with the base edge properties attached (from
    `networks_from_pbf.py`)."""
    prep_ctx = context.source('preparation/world/osm')
    nw_name = f'{scenario.area_name}_{mode}'
    graph = prep_ctx.get_nw(
        data_name=nw_name,
        # Passing the nw_name itself asks get_nw to attach the network's
        # own base property file (edges_<nw_name>.csv from
        # networks_from_pbf's save_properties=True). See
        # `_expand_nw_property_names` in aperta_atlas.context.
        add_edge_properties=nw_name,
        allow_cache=True,
    )
    edges = _edges_gdf(graph, crs='EPSG:4326').to_crs(story._CRS_MAIN)
    return edges


def _load_consolidated_osm_edges(
    context, scenario, mode: str,
) -> gpd.GeoDataFrame:
    """Post-consolidation + post-decoration OSM network. Graphml
    topology is the consolidated one; edge properties come from the
    decorated overlay (`edges_<name>_consolidated_decorated.csv`, from
    networks_decorate.py) — has speed_kph among other filled-in tags."""
    prep_ctx = context.source('preparation/world/osm')
    graph = prep_ctx.get_nw(
        data_name=f'{scenario.area_name}_{mode}_consolidated',
        # `'decorated'` expands to `<data_name>_decorated`.
        add_edge_properties='decorated',
        allow_cache=True,
    )
    edges = _edges_gdf(graph, crs='EPSG:4326').to_crs(story._CRS_MAIN)
    return edges


def _load_consolidated_osm_nodes(
    context, scenario, mode: str,
) -> gpd.GeoDataFrame:
    """Nodes from the consolidated graph with the decorated node overlay
    (`nodes_<name>_consolidated_decorated.csv`), which has
    `is_traffic_signal` and the intersection-classification flags."""
    prep_ctx = context.source('preparation/world/osm')
    graph = prep_ctx.get_nw(
        data_name=f'{scenario.area_name}_{mode}_consolidated',
        add_node_properties='decorated',
        allow_cache=True,
    )
    nodes = _nodes_gdf(graph, crs='EPSG:4326').to_crs(story._CRS_MAIN)
    return nodes


def _edges_gdf(graph: nx.MultiDiGraph, crs: str) -> gpd.GeoDataFrame:
    """Convert a MultiDiGraph's edges to a GeoDataFrame. Preserves
    all edge attributes; geometry from the `geometry` attr if present,
    else straight line between endpoint (x, y) node attrs."""
    from shapely.geometry import LineString
    rows = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        geom = d.get('geometry')
        if geom is None:
            nu, nv = graph.nodes[u], graph.nodes[v]
            geom = LineString([(nu['x'], nu['y']), (nv['x'], nv['y'])])
        row = {'u': u, 'v': v, 'k': k, 'geometry': geom, **{
            key: val for key, val in d.items() if key != 'geometry'
        }}
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs=crs)


def _nodes_gdf(graph: nx.MultiDiGraph, crs: str) -> gpd.GeoDataFrame:
    """Convert a MultiDiGraph's nodes to a GeoDataFrame."""
    from shapely.geometry import Point
    rows = []
    for n, d in graph.nodes(data=True):
        x = d.get('x'); y = d.get('y')
        if x is None or y is None:
            continue
        rows.append({'node_id': n, 'geometry': Point(x, y), **{
            k: v for k, v in d.items() if k not in ('x', 'y')
        }})
    return gpd.GeoDataFrame(rows, crs=crs).set_index('node_id')


# --- Transit-access loader --------------------------------------------

def _load_cell_transit_access_combined(
    context, cells: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach a combined ABSOLUTE transit-access quality per cell:
    the β-weighted sum of the raw `t_walk_to_transit_nearest` and
    `t_bike_to_train_nearest` (both in seconds) from stage 06, using
    07b's fitted coefficients. Semantically this is the expected
    transit-access time contribution per cell — 0 = best (right on top
    of a transit stop AND a train station), larger = worse."""
    access = context.get_properties('cells', 'transit_access')
    # Drop overlapping columns (e.g. `cell_id`) so `.join` doesn't error.
    access = access[[c for c in access.columns if c not in cells.columns]]
    cells = cells.join(access, how='left')
    coefs = context.get_coefs('overheads_transit')

    # The production 'transit' column carries the fitted β for each
    # ZONE-DEV feature. The RAW absolute feature has the same β as its
    # deviation (deviation is a linear shift), so we can reuse those
    # β's against the raw values to get an absolute-time indicator.
    beta_walk = float(coefs.loc['t_walk_to_transit_nearest_zone_dev', 'transit']) \
        if 't_walk_to_transit_nearest_zone_dev' in coefs.index else 0.0
    beta_bike = float(coefs.loc['t_bike_to_train_nearest_zone_dev', 'transit']) \
        if 't_bike_to_train_nearest_zone_dev' in coefs.index else 0.0

    # Preserve NaN: cells missing either raw feature (e.g. no transit
    # or no train station within the search radius set by stage 06)
    # should render as no-data, NOT as 0-seconds = "best-possible access".
    t_walk = cells.get('t_walk_to_transit_nearest', pd.Series(float('nan'),
                                                              index=cells.index))
    t_bike = cells.get('t_bike_to_train_nearest', pd.Series(float('nan'),
                                                            index=cells.index))
    cells['transit_access_score_s'] = beta_walk * t_walk + beta_bike * t_bike
    return cells


# =====================================================================
# New frames
# =====================================================================

# Speed limit color scale for the consolidated-network frame. Chosen
# so urban (30-50 km/h) is mid-cmap and motorway (~80-120) is at top.
_SPEED_LIMIT_VMIN = 20.0
_SPEED_LIMIT_VMAX = 120.0

# Highlight color for reattached traffic signals.
_SIGNAL_COLOR = '#e15759'
_SIGNAL_EDGE = '#7a1f22'
_SIGNAL_SIZE = 26

# Small black points for 4-way intersections (either _major or _anchor
# flag). Smaller than the signal glyph — signals are the headline
# reattachment, 4-ways are context.
_FOURWAY_COLOR = '#111111'
_FOURWAY_SIZE = 10


def _plot_buildings(ax, context, scenario) -> None:
    """Paint OSM building footprints underneath the map subject (used
    as context in frames 1-4). No-op if the buildings shapefile isn't
    available for this scenario's area."""
    try:
        buildings = _clip_to_extent(story._load_buildings(context, scenario), ax)
    except (FileNotFoundError, KeyError) as e:
        logging.warning(f"  ⚠ buildings unavailable: {e}")
        return
    if buildings.empty:
        return
    buildings.plot(
        ax=ax, color=story._BLDG_FILL, edgecolor=story._BLDG_EDGE,
        linewidth=story._BLDG_LW, alpha=0.9, zorder=2,
    )


def frame_01_nw_raw(context, scenario, cfg: StoryConfig) -> Figure:
    """Raw pre-consolidation OSM car network at focal zoom, over
    building footprints. Shows the density of degree-2 nodes and edge
    segments that consolidation collapses in the next frame."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    _plot_buildings(ax, context, scenario)
    edges = _load_raw_osm_edges(context, scenario, mode='car')
    edges = _clip_to_extent(edges, ax)
    edges.plot(ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH,
               alpha=_NETWORK_ALPHA, zorder=3)
    return fig


def frame_02_nw_consolidated_speeds(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Consolidated OSM car network (degree-2 chains collapsed) colored
    by `speed_kph` attribute — an OSM-tag-derived speed limit filled in
    by `networks_from_pbf.py`. Traffic signals that survived
    consolidation (or were reattached to the surviving nodes of a
    collapsed chain) are drawn as highlighted red points; 4-way
    intersections as small black dots for context."""
    from dataclasses import replace
    # Wider right panel so the two-item legend + the colorbar labels
    # (`OSM-tag speed_kph (km/h)`) fit without clipping.
    frame_cfg = replace(cfg, right_panel_width=0.20)
    fig, ax = make_fig(frame_cfg, extent_key='med')
    _add_basemap(ax)
    _plot_buildings(ax, context, scenario)

    edges = _load_consolidated_osm_edges(context, scenario, mode='car')
    edges = _clip_to_extent(edges, ax)
    if 'speed_kph' not in edges.columns:
        logging.warning(
            "  ⚠ 'speed_kph' missing on consolidated edges — plotting flat.")
        edges.plot(ax=ax, color=_NETWORK_COLOR,
                   linewidth=_NETWORK_WIDTH * 2, alpha=_NETWORK_ALPHA)
    else:
        norm = Normalize(vmin=_SPEED_LIMIT_VMIN, vmax=_SPEED_LIMIT_VMAX)
        edges.plot(
            ax=ax, column='speed_kph', cmap='viridis', norm=norm,
            linewidth=_NETWORK_WIDTH * 2, alpha=0.9,
        )
        sm = ScalarMappable(cmap='viridis', norm=norm)
        add_colorbar(fig, frame_cfg, sm,
                     label='OSM-tag speed_kph (km/h)')

    nodes = _load_consolidated_osm_nodes(context, scenario, mode='car')
    nodes = _clip_to_extent(nodes, ax)

    legend_handles: list = []
    legend_labels: list = []

    # 4-way intersections (either _major or _anchor flag from
    # networks_decorate). Drawn first so signal glyphs land on top.
    def _bool_col(df, col):
        return (df[col].fillna(0).astype(int) == 1) if col in df.columns \
            else pd.Series(False, index=df.index)

    fourway_mask = _bool_col(nodes, 'is_4way_major') | _bool_col(nodes, 'is_4way_anchor')
    fourways = nodes[fourway_mask]
    if not fourways.empty:
        fourways.plot(
            ax=ax, color=_FOURWAY_COLOR, markersize=_FOURWAY_SIZE,
            linewidth=0, alpha=0.85, zorder=4,
        )
        legend_handles.append(Line2D(
            [0], [0], marker='o', color='none',
            markerfacecolor=_FOURWAY_COLOR, markersize=7, linestyle='None',
        ))
        legend_labels.append('4-way intersection')

    if 'is_traffic_signal' in nodes.columns:
        signals = nodes[_bool_col(nodes, 'is_traffic_signal')]
        if not signals.empty:
            signals.plot(
                ax=ax, color=_SIGNAL_COLOR, edgecolor=_SIGNAL_EDGE,
                markersize=_SIGNAL_SIZE, linewidth=0.6, zorder=5,
            )
            legend_handles.append(Line2D(
                [0], [0], marker='o', color='none',
                markerfacecolor=_SIGNAL_COLOR, markeredgecolor=_SIGNAL_EDGE,
                markersize=8, linestyle='None',
            ))
            legend_labels.append('Traffic signal')

    if legend_handles:
        add_legend(fig, frame_cfg, (legend_handles, legend_labels))

    set_title(fig, cfg, 'Consolidated network + OSM speed_kph + traffic signals')
    return fig


_ZONE_COLOR = '#e15759'          # red — distinct from cells (blue) and network (dark grey)
_ZONE_LW = 1.6
_LIGHT_NETWORK_ALPHA = 0.25       # frames 3/4/5: network is context, not the subject
_LIGHT_NETWORK_WIDTH = _NETWORK_WIDTH * 0.6


def _load_car_edges_local(context) -> gpd.GeoDataFrame:
    """Car-network edges geometry — thin duplicate of `story._load_car_edges`
    to sidestep re-importing a private helper for the local frames."""
    edges = context.get_shapes('edges', data_name='car', allow_cache=True)
    if edges.crs is not None and edges.crs.to_string() != story._CRS_MAIN:
        edges = edges.to_crs(story._CRS_MAIN)
    return edges


def frame_cell_grid(context, scenario, cfg: StoryConfig) -> Figure:
    """Cell layer (H3 res 10) at focal zoom on top of building
    footprints (light grey) + a faded car network. Cells sit at the
    top of the z-order so the hex raster reads as the subject.
    Local override of `story.frame_06_cell_grid`."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    _plot_buildings(ax, context, scenario)
    edges = _clip_to_extent(_load_car_edges_local(context), ax)
    edges.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_LIGHT_NETWORK_WIDTH,
        alpha=_LIGHT_NETWORK_ALPHA, zorder=3,
    )
    cells = _clip_to_extent(_load_cells(context), ax)
    cells.boundary.plot(
        ax=ax, color=_CELL_EDGE, linewidth=_CELL_LW,
        alpha=0.85, zorder=4,
    )
    return fig


_CELL_CENTROID_COLOR = '#1f77b4'   # blue — same family as cell edges
_CELL_CENTROID_SIZE = 6
_CELL_CENTROID_ALPHA = 0.85


def _virtual_nodes_frame(
    context, scenario, cfg: StoryConfig, *, show_virtuals: bool,
) -> Figure:
    """Shared body for frames 5a / 5b. `show_virtuals=False` renders the
    "before" state — network + native nodes + cell/zone centroids as
    the raw O/D points. `show_virtuals=True` overlays the resulting
    virtual snap nodes on top, so the transition 5a → 5b makes the
    insertion step legible."""
    from dataclasses import replace
    # Give the legend a wider right panel so the 4-item legend fits
    # without either being clipped or overlapping the map.
    frame_cfg = replace(cfg, right_panel_width=0.20)
    fig, ax = make_fig(frame_cfg, extent_key='med')
    _add_basemap(ax)

    edges = _clip_to_extent(_load_car_edges_local(context), ax)
    edges.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH,
        alpha=_NETWORK_ALPHA, zorder=3,
    )
    nodes = story._load_car_nodes_with_virtual_flag(context)
    native = _clip_to_extent(nodes[~nodes['is_virtual']], ax)
    native.plot(
        ax=ax, color=_NATIVE_NODE_COLOR, markersize=_NATIVE_NODE_SIZE,
        linewidth=0, alpha=_NETWORK_ALPHA, zorder=4,
    )
    cell_centroids = _clip_to_extent(_load_cell_centroids(context), ax)
    if not cell_centroids.empty:
        cell_centroids.plot(
            ax=ax, color=_CELL_CENTROID_COLOR, markersize=_CELL_CENTROID_SIZE,
            alpha=_CELL_CENTROID_ALPHA, zorder=5,
        )
    zone_centroids = _clip_to_extent(_load_zone_centroids(context), ax)
    if not zone_centroids.empty:
        zone_centroids.plot(
            ax=ax, color=_ZONE_COLOR, markersize=_ZONE_CENTROID_SIZE,
            edgecolor='white', linewidth=0.5, zorder=6,
        )
    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=_NATIVE_NODE_COLOR, markersize=6),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=_CELL_CENTROID_COLOR, markersize=6),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=_ZONE_COLOR,
               markeredgecolor='white', markersize=7),
    ]
    legend_labels = ['network node', 'cell centroid', 'zone centroid']

    if show_virtuals:
        virt = _clip_to_extent(nodes[nodes['is_virtual']], ax)
        virt.plot(
            ax=ax, color=story._VIRTUAL_NODE_COLOR,
            markersize=story._VIRTUAL_NODE_SIZE,
            edgecolor=story._VIRTUAL_NODE_EDGE,
            linewidth=story._VIRTUAL_NODE_EDGE_WIDTH,
            zorder=7,
        )
        legend_handles.append(Line2D(
            [0], [0], marker='o', color='none',
            markerfacecolor=story._VIRTUAL_NODE_COLOR,
            markeredgecolor=story._VIRTUAL_NODE_EDGE, markersize=8,
        ))
        legend_labels.append('virtual snap node')

    add_legend(fig, frame_cfg, (legend_handles, legend_labels))
    return fig


def frame_virtual_nodes_pre(context, scenario, cfg: StoryConfig) -> Figure:
    """5a — the state BEFORE virtual-node insertion: network with only
    its native nodes, plus cell + zone centroids as the raw O/D points
    that motivate the insertion step."""
    return _virtual_nodes_frame(context, scenario, cfg, show_virtuals=False)


def frame_virtual_nodes(context, scenario, cfg: StoryConfig) -> Figure:
    """5b — AFTER virtual-node insertion: same view as 5a with the
    resulting virtual snap nodes overlaid, showing how the centroids
    from 5a get grafted onto the network."""
    return _virtual_nodes_frame(context, scenario, cfg, show_virtuals=True)


_ZONE_CENTROID_SIZE = 22


def _load_zone_centroids(context) -> gpd.GeoDataFrame:
    """Zone centroid POINTS from `shapes/zones_centroids.gpkg` — the
    canonical O/D-snap points written by 02a (custom-for-CH origin,
    transport_centroid for foreign zones). NOT the geometric centroid
    of `zones.gpkg`, which can fall outside the polygon or on the wrong
    side of a network barrier."""
    zc = context.get_shapes('zones', data_name='centroids', allow_cache=True)
    if zc.crs is not None and zc.crs.to_string() != story._CRS_MAIN:
        zc = zc.to_crs(story._CRS_MAIN)
    return zc


def _load_cell_centroids(context) -> gpd.GeoDataFrame:
    """Cell centroid POINTS from `shapes/cells_centroids.gpkg` — the
    per-cell point representation written by 01, used as the O/D basis
    for cell-tier routing."""
    cc = context.get_shapes('cells', data_name='centroids', allow_cache=True)
    if cc.crs is not None and cc.crs.to_string() != story._CRS_MAIN:
        cc = cc.to_crs(story._CRS_MAIN)
    return cc


def frame_zones(context, scenario, cfg: StoryConfig) -> Figure:
    """Zone layer (NPVM) at focal zoom, matching `frame_cell_grid`'s
    extent so 3 → 4 read as the same view with a coarser tessellation.
    Buildings + faded network for context; zone polygons AND zone
    centroids in red — centroids are the O/D snap-target for zone-tier
    routing, loaded from `zones_centroids.gpkg` (NPVM's custom points,
    not geometric)."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    _plot_buildings(ax, context, scenario)
    edges = _clip_to_extent(_load_car_edges_local(context), ax)
    edges.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_LIGHT_NETWORK_WIDTH,
        alpha=_LIGHT_NETWORK_ALPHA, zorder=3,
    )
    zones = _clip_to_extent(_load_zones(context), ax)
    zones.plot(
        ax=ax, facecolor='none', edgecolor=_ZONE_COLOR,
        linewidth=_ZONE_LW, alpha=0.9, zorder=4,
    )
    zone_centroids = _clip_to_extent(_load_zone_centroids(context), ax)
    if not zone_centroids.empty:
        zone_centroids.plot(
            ax=ax, color=_ZONE_COLOR, markersize=_ZONE_CENTROID_SIZE,
            edgecolor='white', linewidth=0.5, zorder=5,
        )
    return fig


# White edge on hex cells leaves a visible gap between adjacent cells
# so the raster reads as a discretisation of space rather than a solid
# blob. Applied by all hex-based frames (10-15).
_HEX_EDGECOLOR = 'white'
_HEX_LINEWIDTH = 0.85


_FLOW_LOG_VMIN = 10.0
_FLOW_LOG_VMAX = 50_000.0


def frame_traffic_flows_log(context, scenario, cfg: StoryConfig) -> Figure:
    """Traffic flows on the car network with a LOG color scale — the
    full range spans 4-5 orders of magnitude, so a linear scale
    (`story.frame_10_traffic_flows`) collapses everything below the
    top decile into one dark band. LogNorm vmin=10, vmax=50 000 keeps
    both quiet side streets and motorway trunks legible."""
    from matplotlib.colors import LogNorm

    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    base = _clip_to_extent(story._load_car_edges(context), ax)
    if not base.empty:
        base.plot(
            ax=ax, color=story._NETWORK_FADED_COLOR,
            linewidth=_NETWORK_WIDTH, alpha=story._NETWORK_FADED_ALPHA,
            zorder=2,
        )
    edges = story._load_car_edges_with_attr(context, 'car_flows', 'flow_estimate')
    edges = _clip_to_extent(edges, ax)
    if edges.empty:
        return fig

    flows = edges['flow_estimate'].fillna(0.0).clip(lower=_FLOW_LOG_VMIN)
    norm = LogNorm(vmin=_FLOW_LOG_VMIN, vmax=_FLOW_LOG_VMAX)
    # Line width scales with log-flow so high-flow corridors read as
    # thicker bands, matching the color emphasis.
    lw_min, lw_max = story._FLOW_LW_MIN, story._FLOW_LW_MAX * 1.3
    # Log-spaced bin edges: 10, 100, 1_000, 10_000, +∞ — one decade per bin.
    bin_edges = [_FLOW_LOG_VMIN, 100.0, 1_000.0, 10_000.0, float('inf')]
    bin_widths = np.linspace(lw_min, lw_max, len(bin_edges) - 1)
    for i, lw in enumerate(bin_widths):
        mask = (flows >= bin_edges[i]) & (flows < bin_edges[i + 1])
        subset = edges[mask]
        if subset.empty:
            continue
        subset.plot(
            ax=ax, column='flow_estimate', cmap=story._FLOW_CMAP,
            norm=norm, linewidth=lw, alpha=0.95, zorder=3 + i * 0.1,
        )
    sm = ScalarMappable(cmap=story._FLOW_CMAP, norm=norm)
    add_colorbar(fig, cfg, sm, label='Flow estimate (veh/day, log scale)')
    return fig


_BIKE_SPEED_VMIN = 5.0
_BIKE_SPEED_VMAX = 25.0
_BIKE_SPEED_CMAP = 'viridis'
_BIKE_EDGE_LW = 1.4


def frame_calibrated_bike(context, scenario, cfg: StoryConfig) -> Figure:
    """Effective-speed heatmap on the bike (rbike) network from 04's
    edge-weight calibration. Complements the car peak/night frames but
    uses its own kph range (bikes ≈ 5-25 km/h, not 30-80)."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)

    col = 'effective_speed_kph_rbike'
    edges = context.get_shapes('edges', data_name='bike', allow_cache=True)
    if edges.crs is not None and edges.crs.to_string() != story._CRS_MAIN:
        edges = edges.to_crs(story._CRS_MAIN)
    props = context.get_properties('edges', 'bike_calibrated', allow_cache=True)
    if col not in props.columns:
        raise KeyError(f"{col!r} not in edges_bike_calibrated.csv "
                       f"(available: {list(props.columns)})")
    edges = edges.join(props[[col]], how='left')
    edges = _clip_to_extent(edges, ax)

    norm = Normalize(vmin=_BIKE_SPEED_VMIN, vmax=_BIKE_SPEED_VMAX)
    if not edges.empty:
        edges.plot(
            ax=ax, column=col, cmap=_BIKE_SPEED_CMAP, norm=norm,
            linewidth=_BIKE_EDGE_LW, alpha=0.95, zorder=3,
        )
    sm = ScalarMappable(cmap=_BIKE_SPEED_CMAP, norm=norm)
    add_colorbar(fig, cfg, sm, label='Effective bike speed (km/h)')
    set_title(fig, cfg, 'Calibrated bike (rbike) effective speeds')
    return fig


def frame_transit_access(context, scenario, cfg: StoryConfig) -> Figure:
    """Per-cell ABSOLUTE transit-access score: β-weighted sum of raw
    walk-to-nearest-stop and bike-to-nearest-station times from stage
    06, weighted by 07b's fitted coefficients. 0 = best possible access
    (sitting on top of both a transit stop and a train station);
    larger = worse. Sequential colormap (viridis-r: bright = good,
    dark = bad) so the ordering is intuitive."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)

    cells = _load_cells(context)
    try:
        cells = _load_cell_transit_access_combined(context, cells)
    except (ValueError, KeyError) as e:
        logging.warning(f"  ⚠ transit-access score unavailable: {e}")
        cells = _clip_to_extent(cells, ax)
        cells.plot(ax=ax, facecolor='none', edgecolor=_CELL_EDGE,
                   linewidth=_CELL_LW * 0.3)
        return fig
    cells = _clip_to_extent(cells, ax)
    # Drop cells with no walk/bike-to-transit data (stage 06 didn't
    # find a stop/station within the search radius) so they don't
    # render at all — otherwise they'd falsely read as "0 = best".
    cells = cells[cells['transit_access_score_s'].notna()]
    vals = cells['transit_access_score_s']
    if vals.empty:
        return fig
    # Colour range: 0 to the 98th percentile within the view (min-based
    # anchor at 0 matches the "0 = best" semantic).
    vmax = max(60.0, float(vals.quantile(0.98)))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = 'viridis_r'
    cells.plot(
        ax=ax, column='transit_access_score_s', cmap=cmap, norm=norm,
        edgecolor=_HEX_EDGECOLOR, linewidth=_HEX_LINEWIDTH, alpha=0.9,
    )
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = add_colorbar(
        fig, cfg, sm,
        label='Transit-access score (min · 0 = best)',
    )
    from matplotlib.ticker import FuncFormatter
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/60:.1f}'))
    return fig


def frame_cell_overheads_hex(context, scenario, cfg: StoryConfig) -> Figure:
    """Per-cell car-peak overhead (const_s/2 + density_coef · density
    + snap_dist_coef · snap_dist) on the hex raster. Local hex version
    of `story.frame_13_cell_overheads`."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    cells = _load_cells(context)
    overheads = story._compute_cell_overheads(
        context, profile='car_peak', side='origin')
    cells = cells.join(overheads.rename('overhead_s'), how='inner')
    cells = _clip_to_extent(cells, ax)
    if cells.empty:
        return fig
    oh = cells['overhead_s']
    vmin = float(oh.quantile(0.02))
    vmax = float(oh.quantile(0.98))
    if vmax <= vmin:
        vmin, vmax = 0.0, max(vmax, 1.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cells.plot(
        ax=ax, column='overhead_s', cmap=story._OVERHEAD_CMAP,
        norm=norm, edgecolor=_HEX_EDGECOLOR, linewidth=_HEX_LINEWIDTH,
        alpha=0.85, zorder=3,
    )
    sm = ScalarMappable(cmap=story._OVERHEAD_CMAP, norm=norm)
    add_colorbar(fig, cfg, sm, label='Origin overhead, car peak (s)')
    return fig


def _plot_access_hex(
    context, cfg: StoryConfig, profile: str, label_suffix: str,
) -> Figure:
    """Shared body for the per-mode accessibility hex frames."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    cells = story._load_cells_with_access(
        context, profile, 'nearest_k', story._ACCESS_COLUMN)
    cells = _clip_to_extent(cells, ax)
    if not cells.empty:
        norm = Normalize(vmin=story._ACCESS_VMIN, vmax=story._ACCESS_VMAX)
        cells.plot(
            ax=ax, column=story._ACCESS_COLUMN,
            cmap=story._ACCESS_CMAP, norm=norm,
            edgecolor=_HEX_EDGECOLOR, linewidth=_HEX_LINEWIDTH,
            alpha=story._ACCESS_CELL_ALPHA, zorder=3,
        )
        sm = ScalarMappable(cmap=story._ACCESS_CMAP, norm=norm)
        add_colorbar(fig, cfg, sm,
                     label=story._ACCESS_LABEL + '  · ' + label_suffix)
    return fig


def frame_access_car_hex(context, scenario, cfg: StoryConfig) -> Figure:
    """Access-by-car heatmap on the hex raster (car_peak profile)."""
    return _plot_access_hex(context, cfg, 'car_peak', 'car peak')


def frame_access_walk_hex(context, scenario, cfg: StoryConfig) -> Figure:
    """Access-by-walk heatmap on the hex raster (rwalk profile)."""
    return _plot_access_hex(context, cfg, 'rwalk', 'walk')


def frame_access_fastest_hex(context, scenario, cfg: StoryConfig) -> Figure:
    """Cross-modal min(walk, car) accessibility on the hex raster."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)
    cells = story._load_cells_with_min_access(
        context, ['rwalk', 'car_peak'], 'nearest_k', story._ACCESS_COLUMN)
    cells = _clip_to_extent(cells, ax)
    if not cells.empty:
        norm = Normalize(vmin=story._ACCESS_VMIN, vmax=story._ACCESS_VMAX)
        cells.plot(
            ax=ax, column=story._ACCESS_COLUMN,
            cmap=story._ACCESS_CMAP, norm=norm,
            edgecolor=_HEX_EDGECOLOR, linewidth=_HEX_LINEWIDTH,
            alpha=story._ACCESS_CELL_ALPHA, zorder=3,
        )
        sm = ScalarMappable(cmap=story._ACCESS_CMAP, norm=norm)
        add_colorbar(fig, cfg, sm,
                     label=story._ACCESS_LABEL + '  · min(walk, car)')
    return fig


# Utility-based grocery gravity access. Column in
# `cells_access_util_gravity_rwalk.csv` — Σ w · exp(-β · disutility)
# with β from `AccessibilityGrid.gravity_util_betas`. β=1.0 is the
# middle setting the util grid ships with.
_GRAVITY_GROCERIES_COL = 'poi_errands_groceries_exp1.0'
_GRAVITY_CMAP = 'viridis'


def frame_access_walk_gravity_groceries(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Utility-based gravity walk access to grocery stores, on the hex
    raster. Reads `cells_access_util_gravity_rwalk.csv` (grid_key='util',
    travel_cost='util' in the scenario's `accessibility_grids`) — that
    file's costs come from 09b's walk disutility (utility-scale, not
    time-scale), so 10 stores `logsum = ln(Σ w·exp(-β·disutility))`,
    which is on a log-utility scale (typically negative, larger = better).
    β=1.0 (mid-range from the grid's `gravity_util_betas`)."""
    fig, ax = make_fig(cfg, extent_key='med')
    _add_basemap(ax)

    # NOTE: bypass `_load_cells_with_access` because it filters `col < 0`
    # as an AOI sentinel — legitimate for time-domain grids where -1 is
    # unreachable, but wrong for utility logsums where values ARE
    # negative by construction.
    cells = _load_cells(context)
    access = context.get_properties(
        'cells', f'access_util_gravity_rwalk', allow_cache=True)
    if _GRAVITY_GROCERIES_COL not in access.columns:
        raise KeyError(f"{_GRAVITY_GROCERIES_COL!r} not in "
                       f"cells_access_util_gravity_rwalk.csv")
    cells = cells.join(access[[_GRAVITY_GROCERIES_COL]], how='left')
    cells = _clip_to_extent(cells, ax)
    cells = cells[cells[_GRAVITY_GROCERIES_COL].notna()]
    vals = cells[_GRAVITY_GROCERIES_COL]
    if vals.empty:
        return fig
    # Range on the actual distribution (2nd–98th percentile) — logsums
    # here are ≈ −16 to −1, so a hard 0 anchor would collapse contrast.
    vmin = float(vals.quantile(0.02))
    vmax = float(vals.quantile(0.98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cells.plot(
        ax=ax, column=_GRAVITY_GROCERIES_COL, cmap=_GRAVITY_CMAP,
        norm=norm, edgecolor=_HEX_EDGECOLOR, linewidth=_HEX_LINEWIDTH,
        alpha=story._ACCESS_CELL_ALPHA, zorder=3,
    )
    sm = ScalarMappable(cmap=_GRAVITY_CMAP, norm=norm)
    add_colorbar(fig, cfg, sm,
                 label='Grocery access · walk logsum (utility, β=1.0)')
    return fig


# =====================================================================
# Frame sequence + main
# =====================================================================

_FRAMES = [
    # Network preprocessing (2)
    frame_01_nw_raw,
    frame_02_nw_consolidated_speeds,
    # Geospatial mapping: cells + zones (2)
    frame_cell_grid,                   # local: faded network + hex cells + buildings
    frame_zones,                       # local: same extent, cells+zones layer
    # Network mapping: virtual nodes — 5a (before) then 5b (after) so the
    # insertion step reads as a transition rather than a single jump.
    frame_virtual_nodes_pre,
    frame_virtual_nodes,
    # Traffic-flow estimation — log color scale for the wide dynamic range
    frame_traffic_flows_log,
    # Calibrated edges — car peak + night (bike removed for now)
    frame_11_calibrated_speeds,        # car_peak
    frame_11b_calibrated_speeds_night,
    # Overhead estimation (2) — car (per-cell) first, then transit
    frame_cell_overheads_hex,          # local: hex cells (not circles)
    frame_transit_access,
    # Accessibility (4) — all on hex raster; nearest-K per mode + gravity
    frame_access_car_hex,
    frame_access_walk_hex,
    frame_access_fastest_hex,
    frame_access_walk_gravity_groceries,
]


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    # Set the per-scenario metric CRS on the story module BEFORE any
    # frame runs — imported frames + helpers read `story._CRS_MAIN`
    # for their reprojection targets.
    story._CRS_MAIN = scenario.crs_main

    defaults = story._DEFAULTS.get(scenario.name)
    if defaults is None:
        logging.warning(
            f"no StoryDefaults entry for scenario {scenario.name!r}; "
            f"falling back to Bern Hauptgebäude (frames will likely "
            f"render in the wrong place). Add a `_DEFAULTS` entry in "
            f"`visualization/story.py`.")
        defaults = story._FALLBACK_DEFAULTS

    overrides: dict = {}
    if defaults.extents is not None:
        overrides['extents'] = defaults.extents
    if defaults.dpi is not None:
        overrides['dpi'] = defaults.dpi
    # Custom output subdir so this story's PNGs don't overwrite the
    # full story's frame_NN.png files.
    cfg = StoryConfig(
        center_xy=story._latlon_to_metric(*defaults.focal_latlon),
        destinations_latlon=list(defaults.destinations_latlon),
        output_subdir='story_condensed',
        **overrides,
    )
    logging.info(
        f"  scenario={scenario.name!r}, CRS={story._CRS_MAIN!r}; "
        f"focal point (lat, lon) {defaults.focal_latlon} → metric "
        f"{cfg.center_xy[0]:.1f}, {cfg.center_xy[1]:.1f}")

    import re
    for i, fn in enumerate(_FRAMES, start=1):
        # Strip both the `frame_` prefix AND any per-function numeric
        # prefix (e.g. `11b_` from `frame_11b_calibrated_speeds_night`)
        # so the sequence position is the only number in the filename.
        short_name = re.sub(r'^\d+[a-z]?_', '',
                            fn.__name__.removeprefix('frame_'))
        out_name = f'{i:02d}_{short_name}.png'
        with step(f'frame {i:02d}/{len(_FRAMES)}: {fn.__name__}'):
            fig = fn(context, scenario, cfg)
            context.create_results(
                fig, f'{cfg.output_subdir}/{out_name}',
                kws={'bbox_inches': None, 'pad_inches': 0,
                     'dpi': cfg.dpi},
            )
            plt.close(fig)

    context.close()


if __name__ == '__main__':
    main()
