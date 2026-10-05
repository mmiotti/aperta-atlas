"""
Sequential `image story` for the Swiss Urban Mobility Atlas methodology.

Produces a numbered series of PNGs (`results/story/frame_NN.png`) that walks
through the atlas pipeline at a single focal location, at successively wider
zoom levels. Frame 1 is a 1×1 km zoom on buildings; the story moves through
cell allocation, network insertion, tier circles, traffic-flow estimation,
calibrated edge weights, overheads, accessibility, multi-modal comparisons,
and bike-specific deep-dives.

Design constraints:
  - The main plot rectangle is locked to a fixed figure-relative position
    (`StoryConfig.plot_rect`) via `fig.add_axes`. Colorbars, legends, and
    titles get their own pre-positioned axes / text calls OUTSIDE the
    locked rectangle. `tight_layout` is never used. Net effect: the data
    area sits at exactly the same pixel coordinates in every frame,
    making the sequence read as a smooth zoom/overlay story.
  - Each frame function is self-contained — loads its own data, draws
    explicitly. No shared mutable state between frames; easy to reorder,
    disable, or re-run a single frame.
  - All `context.get_*` calls pass `allow_cache=True` so re-used layers
    (cells, buildings, networks) load once per process.

Re-configuration:
  - Change `StoryConfig.center_xy` (the `_LV95_TRANSFORMER.transform`
    return value of the desired lat/lon point) to relocate the focal
    point. The current default is the main building of the University
    of Bern (46.950137, 7.436966).
  - Run with `--scenario <name>` to switch the source project scenario
    (default: `bern-default`).

Inputs vary per frame; the most-used layers are:
    - Buildings (PUBLIC, preparation/world/osm/shapes/buildings_<area>.gpkg)
    - Cells + zones (PUBLIC, <scenario>/shapes/...)
    - Networks (walk / bike / car .graphml + edge / node properties)
    - Accessibility outputs (cells_access_*_<profile>.csv)
    - Survey routes (if used) (PRIVATE, generic/survey_leg_times.csv)

Outputs:
    RESULTS, under <scenario>/story/frame_NN.png

Run:
    python -m visualization.story --scenario <name>
"""

import logging
from dataclasses import dataclass, field

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from pyproj import Transformer
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import Point

from aperta.network_processing import attach_edge_properties
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from scenarios import get_scenario


# Per-scenario CRS — module-level mutable, reassigned in `main()` from
# `scenario.crs_main` before any frame runs. Helpers and frames read
# this rather than threading `cfg.crs_main` through every loader. Bern
# / Switzerland default is EPSG:2056 (LV95); Cambridge UK overrides to
# EPSG:27700 (British National Grid).
_CRS_MAIN: str = 'EPSG:2056'


# CartoDB Positron (no labels, retina). Public tiles now require an
# `?key=` parameter; read the key from `.env` (`CARTO_API_KEY=...`).
# NOTE: assign via `_BASEMAP['url']` (dict item), not `.url` (attribute).
# `xyzservices.TileProvider.build_url` reads the URL template from the
# dict; setting the attribute silently no-ops in `build_url`.
from aperta_atlas.context import env_values
_CARTO_KEY = env_values.get('CARTO_API_KEY')
_BASEMAP = cx.providers.CartoDB.PositronNoLabels(r='@2x')  # type: ignore[attr-defined]
if _CARTO_KEY:
    _BASEMAP['url'] = _BASEMAP['url'] + f'?key={_CARTO_KEY}'
else:
    logging.warning(
        "CARTO_API_KEY not set in .env — CartoDB tiles will render "
        "with an 'API key required' watermark.")


# WGS84 lat/lon → `_CRS_MAIN`. Module-level + lru_cache so the
# transformer initialisation cost is paid once per target CRS (cheap
# even if it switches mid-run, which it doesn't in practice).
from functools import lru_cache


@lru_cache(maxsize=8)
def _metric_transformer(target_crs: str) -> Transformer:
    return Transformer.from_crs('EPSG:4326', target_crs, always_xy=True)


def _latlon_to_metric(lat: float, lon: float) -> tuple[float, float]:
    """Project (lat, lon) → (x, y) in the current `_CRS_MAIN`. Note
    `always_xy=True` on the transformer expects (lon, lat) input order."""
    x, y = _metric_transformer(_CRS_MAIN).transform(lon, lat)
    return float(x), float(y)


@dataclass(frozen=True)
class StoryDefaults:
    """Per-scenario defaults for `main()` — focal point + example
    destinations. `extents` and `dpi` override the `StoryConfig`
    defaults when set; leave as `None` to inherit the StoryConfig
    default (currently `dpi=128` and the bern-scale extents).
    """
    focal_latlon: tuple[float, float]
    destinations_latlon: list[tuple[float, float]]
    extents: dict[str, float] | None = None
    dpi: int | None = None


# University of Bern — Hauptgebäude. Default focal point + a few
# Bern-area destinations for frame 12 example routes.
_DEFAULTS: dict[str, StoryDefaults] = {
    'bern-default': StoryDefaults(
        focal_latlon=(46.950137, 7.436966),
        destinations_latlon=[
            (46.943528, 7.408003),
            (46.957286, 7.452039),
            (46.940121, 7.460528),
        ],
    ),
    'bern-public': StoryDefaults(
        focal_latlon=(46.950137, 7.436966),
        destinations_latlon=[
            (46.943528, 7.408003),
            (46.957286, 7.452039),
            (46.940121, 7.460528),
        ],
    ),
    'switzerland-h10': StoryDefaults(
        # ETH Zurich Hönggerberg campus centre.
        focal_latlon=(47.408552, 8.507549),
        destinations_latlon=[],
        # Wider than the Bern default so the metro-Zürich context (Uetliberg,
        # Limmattal, ETH main campus) fits into the frame.
        extents={
            'in':     660.0,     # ~1.3 km wide
            'med':   4_400.0,    # ~8.8 km wide
            'far':  22_000.0,    # ~44 km wide
            'tier': 66_000.0,    # ~132 km wide
        },
    ),
    'switzerland-public': StoryDefaults(
        focal_latlon=(46.950137, 7.436966),
        destinations_latlon=[],
    ),
    # Trinity College — Porter's Lodge on Trinity Street.
    'cambridgeuk-public': StoryDefaults(
        focal_latlon=(52.205891, 0.117931),
        destinations_latlon=[
            (52.196876, 0.122216),
            (52.194240, 0.136962),
            (52.210943, 0.114697),
        ],
        dpi=100,
        extents={
            'in':     500.0,    # (frames 1-3)
            'med':   4_000.0,   # (frames 11-18)
            'far':   4_000.0,   # (frames 4, 19)
            'tier': 25_000.0,   # (frame 8)
        },
    ),
}


# Fallback for scenarios with no entry in `_DEFAULTS` — `main()` will
# warn and use these (Bern Hauptgebäude). Add a `StoryDefaults` entry
# above instead of editing this fallback.
_FALLBACK_DEFAULTS = StoryDefaults(
    focal_latlon=(46.950137, 7.436966),
    destinations_latlon=[],
)


@dataclass(frozen=True)
class StoryConfig:
    """One image story's configuration.

    Layout: the plot rectangle fills the ENTIRE 16:9 figure by default.
    No margins on any side. When a frame needs a colorbar / legend /
    title, the decoration is drawn as an OVERLAY (typically over the
    right portion of the plot) — the plot extent and data center do
    NOT shift. Across frames the focal point therefore lands at exactly
    the same pixel position regardless of whether decorations are
    shown. The "off-centered" appearance of the data center when a
    colorbar is visible is intentional: it preserves cross-frame
    positional continuity.

    `center_xy`         — focal point in _CRS_MAIN coordinates.
    `extents`           — per-zoom-level data x half-width (meters).
                          The y half-width is derived in `make_fig`
                          from the plot area's aspect ratio so the
                          displayed distances stay equal in x and y.
                          Frames pick a key via `extent_key`.
    `fig_size_in`       — figure size in inches. 16:9 default.
    `plot_rect`         — [left, bottom, width, height] in figure-rel
                          coords for the locked main plot axes. Default
                          fills the whole figure.
    `colorbar_rect`     — [left, bottom, width, height] in figure-rel
                          coords for a colorbar axes overlay. Used by
                          `add_colorbar`. Picked once and re-used so
                          successive frames anchor their colorbars at
                          identical pixel positions.
    `legend_anchor`     — figure-rel (x, y) for a legend; passed as
                          `bbox_to_anchor=`. `loc='upper right'` so the
                          legend's top-right corner lands at this point.
    `title_anchor`      — figure-rel (x, y) for an optional title text;
                          `ha='right'`, `va='top'`.
    `right_panel_width` — when any decoration (colorbar / legend /
                          title) is drawn, a single translucent panel
                          spanning the FULL HEIGHT of the figure and
                          the right `right_panel_width` of its width
                          is rendered as the shared backdrop. Default
                          0.15 → right 15 % of the figure, flush to
                          top / bottom / right edges, no margin.
    `decoration_bg`     — when truthy, the right-panel backdrop is
                          drawn behind colorbar / legend overlays so
                          their labels remain readable against the
                          basemap underneath.
    `dpi`               — DPI used for both figure rendering and PNG
                          save. At fig_size_in=(16, 9) + dpi=128, output
                          is 2048×1152 px.
    `output_subdir`     — subfolder under `results/` for the saved PNGs.
    """
    center_xy: tuple[float, float]
    destinations_latlon: list[tuple[float, float]] = field(default_factory=list)
    extents: dict[str, float] = field(default_factory=lambda: {
        'in':     500.0,   # ~1 km wide
        'med':   2_500.0,  # ~5 km wide
        'far':  15_000.0,  # ~30 km wide
        'tier': 50_000.0,  # ~100 km wide — needed for r_zones to fit
    })
    fig_size_in: tuple[float, float] = (16.0, 9.0)
    plot_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # Decoration overlay positions — used when explicitly invoked.
    colorbar_rect: tuple[float, float, float, float] = (0.90, 0.18, 0.018, 0.55)
    legend_anchor: tuple[float, float] = (0.97, 0.95)
    title_anchor: tuple[float, float] = (0.97, 0.96)
    right_panel_width: float = 0.15
    decoration_bg: bool = True
    dpi: int = 128
    output_subdir: str = 'story'


# ---------------------------------------------------------------------
# Style — single source of truth so frames stay visually consistent.
# ---------------------------------------------------------------------

# Buildings
_BLDG_FILL = '#d8d4cd'          # warm light grey
_BLDG_EDGE = '#a59f95'
_BLDG_LW = 0.3
_FOCAL_BLDG_FILL = '#ff7f0e'    # accent orange
_FOCAL_BLDG_EDGE = '#b25400'
_FOCAL_BLDG_LW = 0.8

# Cells
_CELL_EDGE = '#1f77b4'          # blue
_CELL_LW = 0.6
_FOCAL_CELL_FILL = '#1f77b4'
_FOCAL_CELL_ALPHA = 0.18
_FOCAL_CELL_EDGE_LW = 1.4

# Focal point marker
_FOCAL_POINT_COLOR = '#d62728'  # red
_FOCAL_POINT_SIZE = 65

# Background
_BG_COLOR = '#ffffff'

# Title / colorbar / legend font sizes — uniformly bumped so they read
# clearly against the high-DPI figures (and at smaller embed sizes).
_TITLE_FONTSIZE = 17
_LABEL_FONTSIZE = 16
_TICK_FONTSIZE = 14
_LEGEND_FONTSIZE = 14
# Padding (points) between colorbar tick labels and the colorbar's
# rotated axis label. mpl default is ~4; bumped for breathing room.
_LABEL_PAD = 20
_TITLE_COLOR = '#222222'

# Effective-speed scale (frames 11 + 12) — shared so colorbars in both
# frames index the same colour to the same speed. vmax chosen so urban
# (30-50 km/h) sits in the middle of the cmap and motorway (~80 km/h)
# lands at the top.
_SPEED_VMIN = 10.0
_SPEED_VMAX = 80.0

# Network (car / walk / bike base style)
_NETWORK_COLOR = '#3d3d3d'
_NETWORK_WIDTH = 0.5
_NETWORK_ALPHA = 0.75
# Faded network — when network is shown as context behind another layer.
_NETWORK_FADED_COLOR = '#888888'
_NETWORK_FADED_ALPHA = 0.35

# Native network nodes (junctions of the un-densified graph, frames 5-7).
_NATIVE_NODE_COLOR = _NETWORK_COLOR
_NATIVE_NODE_SIZE = 7

# Virtual nodes (cell-snap insertions, frame 7)
_VIRTUAL_NODE_COLOR = '#ff7f0e'    # orange — pops against grey network
_VIRTUAL_NODE_SIZE = 34            # was 28 (+20 % for readability)
_VIRTUAL_NODE_EDGE = '#b22222'     # firebrick — thin red border for contrast
_VIRTUAL_NODE_EDGE_WIDTH = 0.7

# Cell grid (outlines only, frames 6 / 8)
_CELL_OUTLINE_COLOR = '#1f77b4'
_CELL_OUTLINE_WIDTH = 1.5
_CELL_OUTLINE_ALPHA = 0.35
# Inner shrink applied to cell polygons before drawing them as filled
# fills (frames 9 close-cells, 13, 14, 16, 17) so adjacent cells have
# a visible gap rather than reading as a solid mass. ~15 m on a 250 m
# hex = ~12 % linear shrink; visually clean without obscuring the hex.
_CELL_FILL_SHRINK_M = 7.0

# Edge color-by-attribute (frames 10 + 11)
_FLOW_CMAP = 'plasma'              # traffic flow (sequential, dark→bright)
_FLOW_LW_MIN = 0.3                 # linewidth at low flow
_FLOW_LW_MAX = 2.5                 # linewidth at high flow
_SPEED_CMAP = 'RdYlGn'             # effective speed (slow=red, fast=green)
_SPEED_LW = 0.9                    # uniform linewidth for speed frame

# Example routes (frame 12) — coloured by edge `effective_speed_kph`
# using the shared `_SPEED_CMAP / _SPEED_VMIN / _SPEED_VMAX` so the
# routes index the same scale as frame 11's colorbar.
_ROUTE_LW = 2.6
_ROUTE_ALPHA = 0.95
_FADED_NETWORK_COLOR = '#bbbbbb'
_FADED_NETWORK_LW = 0.35
_FADED_NETWORK_ALPHA = 0.7
_DEST_MARKER_SIZE = 80
_DEST_MARKER_EDGE = 'white'
# Intersection-node hierarchy (frame 12). 4-way + t-junction share
# the smaller size to read as "junction types"; traffic signals get
# a larger marker so they pop visually as the higher-friction class.
_INT_4WAY_COLOR = '#404040'        # dark grey
_INT_4WAY_SIZE = 16
_INT_T_JUNCTION_COLOR = '#6aa7e0'  # lighter blue — distinct from grey + red
_INT_T_JUNCTION_SIZE = 16          # same as 4-way (intentional)
_INT_SIGNAL_COLOR = '#d62728'      # red
_INT_SIGNAL_SIZE = 42

# Cell overheads (frame 13). Distinct from `_ACCESS_CMAP` so the two
# heatmap families read as different metrics at a glance — yellow→
# purple here vs yellow→red for accessibility.
_OVERHEAD_CMAP = 'YlGnBu'          # low=yellow → mid=green → high=blue

# Accessibility — nearest-k mean travel time per cell (frames 14, 16, 17).
# Same scale across all three so visual comparison is direct: cell at
# the same colour means same access time, regardless of mode.
_ACCESS_CMAP = 'YlOrRd'            # low=yellow (good), high=red (bad)
_ACCESS_VMIN = 60.0                # 1 min — practical floor
_ACCESS_VMAX = 900.0               # 15 min — beyond this is "far"
_ACCESS_CELL_ALPHA = 0.72
# The destination + k pair used for the access frames. Change to swap
# in a different destination type or k value.
_ACCESS_COLUMN = 'poi_errands_groceries_k3'
_ACCESS_LABEL = 'Mean time to nearest 3 groceries (s)'

# Three-network overlay (frame 15) — each network at the same alpha so
# overlap visually combines colours; widths vary slightly so layered
# edges stay distinguishable.
_NW_WALK_COLOR = '#446644'         # dark green
_NW_BIKE_COLOR = '#1f5fa0'         # blue
_NW_CAR_COLOR  = '#b04a3a'         # muted red
_NW_OVERLAY_ALPHA = 0.6
_NW_WALK_LW = 0.45
_NW_BIKE_LW = 0.55
_NW_CAR_LW  = 0.75

# Bike-score detail (frames 18-20). `bike_infra_score` in
# `edges_bike_extended.csv` is 0-5 (median 3) — set vmax accordingly.
_BIKE_SCORE_CMAP = 'viridis'
_BIKE_SCORE_VMIN = 0.0
_BIKE_SCORE_VMAX = 2.0   # 02b's edge_bike_infra_score buckets {0, 1, 2}
_BIKE_NW_LW = 0.85
_BIKE_NW_ALPHA = 0.95
_BIKE_ROUTE_LW = 3.0
# Background colour for car + walk networks when bike is the focus.
_BIKE_BG_COLOR = '#b8b8b8'
_BIKE_BG_LW = 0.4
_BIKE_BG_ALPHA = 0.5

# Tier circles + zone fills (frame 9)
_TIER_CIRCLE_COLOR = '#d62728'    # red
_TIER_CIRCLE_WIDTH = 1.4
_TIER_CIRCLE_ALPHA = 0.9
_TIER_CLOSE_FILL = '#a1d99b'      # light green (cells_to_cells band)
_ZONE_MEDIUM_FILL = '#ffbb78'     # warm peach (cells_to_zones band)
_ZONE_FAR_FILL = '#aec7e8'        # light blue (zones_to_zones band)
_ZONE_FILL_ALPHA = 0.42
# Per-tier OUTLINE colors — slightly darker than the fill so the
# outline reads at full alpha against the translucent fill underneath.
_TIER_CLOSE_EDGE = '#41ab5d'      # darker green
_ZONE_MEDIUM_EDGE = '#cc7733'     # darker peach / amber
_ZONE_FAR_EDGE = '#3a6a92'        # darker blue
_ZONE_EDGE_WIDTH = 0.4
_ZONE_EDGE_ALPHA = 0.9


# ---------------------------------------------------------------------
# Figure factory + helpers
# ---------------------------------------------------------------------

def make_fig(
    cfg: StoryConfig, extent_key: str, *, basemap: bool = True,
) -> tuple[Figure, Axes]:
    """Build a fresh figure with the main plot axes positioned EXACTLY
    at `cfg.plot_rect`, regardless of any decorations added later.

    Returns `(figure, main_axes)`. The main axes is given a rectangular
    extent (xlim / ylim) centred on `cfg.center_xy`. The x half-width
    comes from `cfg.extents[extent_key]`; the y half-width is derived
    from the plot rectangle's aspect ratio so that data distances are
    equal in both directions (`aspect='equal'`).

    Add data layers onto `ax`; add titles, colorbars, legends via the
    dedicated helpers (`set_title`, `add_colorbar`, etc.) which use
    figure-relative positioning to stay OUTSIDE the locked rectangle.

    `basemap=True` (default) overlays Positron tiles (no labels,
    retina) as the bottom layer.
    """
    fig = plt.figure(
        figsize=cfg.fig_size_in, dpi=cfg.dpi, facecolor=_BG_COLOR)
    ax = fig.add_axes(cfg.plot_rect)
    half_w = cfg.extents[extent_key]
    # Derive y half-width from the plot area's aspect ratio so
    # `aspect='equal'` doesn't waste any of the locked rectangle.
    plot_w_in = cfg.fig_size_in[0] * cfg.plot_rect[2]
    plot_h_in = cfg.fig_size_in[1] * cfg.plot_rect[3]
    half_h = half_w * (plot_h_in / plot_w_in)
    x0, y0 = cfg.center_xy
    ax.set_xlim(x0 - half_w, x0 + half_w)
    ax.set_ylim(y0 - half_h, y0 + half_h)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(_BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if basemap:
        _add_basemap(ax)
    _draw_scale_bar(fig, ax)
    return fig, ax


# Round-number lengths the scale bar snaps to. Picked to give pleasant
# readouts ("200 m", "2 km") at the typical zoom envelopes.
_SCALE_BAR_NICE_M: tuple[int, ...] = (
    50, 100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000, 100_000,
)


def _draw_scale_bar(fig: Figure, ax: Axes) -> None:
    """Draw a horizontal scale bar in the bottom-LEFT corner of `ax`.

    Left-corner placement leaves the bottom-right free for slide
    numbers / page footers when frames are embedded.

    Drawn as a FIGURE-level artist (via `fig.add_artist`) rather than
    an axes-level one, so its zorder lives in the figure's z-stack
    (alongside `_draw_right_decoration_panel`'s `zorder=4.9` patch) —
    otherwise the panel would overpaint the scale bar regardless of
    any in-axes zorder. Position still anchored to `ax.transAxes` so
    the bar tracks the axes (not absolute figure coords)."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    xlim = ax.get_xlim()
    visible_m = abs(xlim[1] - xlim[0])
    if visible_m <= 0:
        return
    target = visible_m * 0.11
    bar_length_m = min(_SCALE_BAR_NICE_M, key=lambda x: abs(x - target))
    bar_frac = bar_length_m / visible_m
    margin = 0.015
    bar_left = margin
    bar_right = bar_left + bar_frac
    if bar_right > 1.0 - margin:  # too wide for the plot — skip
        return
    bar_y = 0.030

    # zorder=6 sits above the right-panel decoration (zorder=4.9) and
    # above colorbar / legend axes (zorder=5).
    z = 6.0

    fig.add_artist(Line2D(
        [bar_left, bar_right], [bar_y, bar_y],
        transform=ax.transAxes, color='black',
        linewidth=2.5, solid_capstyle='butt', zorder=z,
    ))
    tick_h = 0.012
    for x in (bar_left, bar_right):
        fig.add_artist(Line2D(
            [x, x], [bar_y - tick_h, bar_y + tick_h],
            transform=ax.transAxes, color='black',
            linewidth=2.5, zorder=z,
        ))
    label = (
        f'{bar_length_m:,} m' if bar_length_m < 1_000
        else f'{bar_length_m / 1_000:g} km'
    )
    fig.add_artist(Text(
        (bar_left + bar_right) / 2, bar_y + 0.016,
        label, transform=ax.transAxes,
        ha='center', va='bottom', fontsize=14, fontweight='bold',
        color='black', zorder=z,
    ))


def _add_basemap(ax: Axes) -> None:
    """Overlay the Positron-NoLabels-Retina basemap on `ax` at the
    current extent. Tile zoom is picked by contextily from the extent
    size. Attribution is suppressed since the right margin handles all
    annotations."""
    try:
        cx.add_basemap(
            ax,
            source=_BASEMAP,
            crs=_CRS_MAIN,
            zoom='auto',
            attribution=False,
        )
    except Exception as e:  # network failure, tile-server hiccup, etc.
        logging.warning(f"basemap failed: {e!r}; continuing without tiles")


def set_title(fig: Figure, cfg: StoryConfig, text: str) -> None:
    """Per-frame title text overlaid in the top-right of the figure,
    right-aligned at `cfg.title_anchor`. Use this instead of
    `fig.suptitle` so the title never shifts the locked plot rectangle."""
    fig.text(
        cfg.title_anchor[0], cfg.title_anchor[1], text,
        ha='right', va='top',
        fontsize=_TITLE_FONTSIZE, color=_TITLE_COLOR,
        wrap=True,
    )


def _draw_right_decoration_panel(fig: Figure, cfg: StoryConfig) -> None:
    """Draw a single translucent panel covering the right
    `cfg.right_panel_width` of the figure, full height, flush to the
    top / bottom / right edges. Acts as the shared backdrop for any
    colorbar / legend / title in the right zone.

    Safe to call multiple times — only the first call adds the patch
    (idempotent via a sentinel attribute on `fig`). Allows `add_colorbar`
    and `add_legend` to invoke it independently without double-rendering
    in frames that use both."""
    if getattr(fig, '_story_right_panel_drawn', False):
        return
    w = cfg.right_panel_width
    fig.patches.append(Rectangle(
        (1.0 - w, 0.0), w, 1.0,
        transform=fig.transFigure,
        facecolor='white', alpha=0.85,
        edgecolor='none', zorder=4.9,
    ))
    setattr(fig, '_story_right_panel_drawn', True)


def add_colorbar(
    fig: Figure, cfg: StoryConfig, mappable: ScalarMappable,
    label: str | None = None,
) -> 'plt.Colorbar':
    """Draw a colorbar at `cfg.colorbar_rect` overlaid on the plot.
    The plot extent does NOT change; the colorbar sits on top.
    If `cfg.decoration_bg`, the shared right-panel backdrop is drawn
    behind it (full-height, flush-to-edge translucent rectangle)."""
    if cfg.decoration_bg:
        _draw_right_decoration_panel(fig, cfg)
    cbar_ax = fig.add_axes(cfg.colorbar_rect, zorder=5)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    if label is not None:
        cbar.set_label(
            label, fontsize=_LABEL_FONTSIZE, labelpad=_LABEL_PAD)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
    return cbar


def add_legend(
    fig: Figure, cfg: StoryConfig, handles_labels=None, **legend_kws,
) -> 'plt.Legend':
    """Figure-level legend anchored at `cfg.legend_anchor` with
    `loc='upper right'`. Pass explicit `(handles, labels)` via
    `handles_labels=(handles, labels)`, or rely on matplotlib's
    auto-collection from the locked plot axes.

    Shares the right-panel backdrop with any colorbar already drawn
    on this figure (or draws the panel itself if no colorbar is present).
    Legend itself uses `frameon=False` to avoid double-framing."""
    legend_kws.setdefault('fontsize', _LEGEND_FONTSIZE)
    if handles_labels is not None:
        handles, labels = handles_labels
        leg = fig.legend(
            handles, labels,
            loc='upper right',
            bbox_to_anchor=cfg.legend_anchor,
            frameon=False,
            **legend_kws,
        )
    else:
        leg = fig.legend(
            loc='upper right',
            bbox_to_anchor=cfg.legend_anchor,
            frameon=False,
            **legend_kws,
        )
    if cfg.decoration_bg:
        _draw_right_decoration_panel(fig, cfg)
    return leg


def _clip_to_extent(gdf: gpd.GeoDataFrame, ax: Axes) -> gpd.GeoDataFrame:
    """Return only rows whose geometry intersects the current axes
    extent. Cheap bounding-box filter — no projection, just a query
    on the GeoDataFrame's spatial index where present."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # `cx` uses sindex; falls back to brute-force if absent.
    return gdf.cx[xmin:xmax, ymin:ymax]


# ---------------------------------------------------------------------
# Data loaders (cached via context.allow_cache=True)
# ---------------------------------------------------------------------

def _load_buildings(context, scenario) -> gpd.GeoDataFrame:
    """Buildings polygons for the scenario's area, reprojected to _CRS_MAIN."""
    osm = context.source('preparation/world/osm')
    buildings = osm.get_shapes(
        'buildings', data_name=scenario.area_name, allow_cache=True)
    if buildings.crs is not None and buildings.crs.to_string() != _CRS_MAIN:
        buildings = buildings.to_crs(_CRS_MAIN)
    return buildings


def _load_cells(context) -> gpd.GeoDataFrame:
    """Atlas cells (250 m hex)."""
    cells = context.get_shapes('cells', allow_cache=True)
    if cells.crs is not None and cells.crs.to_string() != _CRS_MAIN:
        cells = cells.to_crs(_CRS_MAIN)
    return cells


def _focal_point_geom(cfg: StoryConfig) -> Point:
    """The focal point as a shapely Point in _CRS_MAIN."""
    return Point(*cfg.center_xy)


def _find_focal_building(
    buildings: gpd.GeoDataFrame, cfg: StoryConfig,
) -> gpd.GeoDataFrame:
    """Building polygon containing the focal point, if any. Returns a
    1-row GeoDataFrame (or empty if the focal point lands outside all
    buildings — e.g., on a courtyard or street)."""
    pt = _focal_point_geom(cfg)
    hit = buildings[buildings.geometry.intersects(pt)]
    return hit


def _find_focal_cell(
    cells: gpd.GeoDataFrame, cfg: StoryConfig,
) -> gpd.GeoDataFrame:
    """Cell containing the focal point. Returns a 1-row GeoDataFrame."""
    pt = _focal_point_geom(cfg)
    hit = cells[cells.geometry.intersects(pt)]
    return hit


def _load_car_edges(context) -> gpd.GeoDataFrame:
    """Car-network edges geometry (all edges, incl. virtuals)."""
    edges = context.get_shapes('edges', data_name='car', allow_cache=True)
    if edges.crs is not None and edges.crs.to_string() != _CRS_MAIN:
        edges = edges.to_crs(_CRS_MAIN)
    return edges


def _load_mode_edges(context, mode: str) -> gpd.GeoDataFrame:
    """Edges geometry for any mode (walk / bike / car)."""
    edges = context.get_shapes('edges', data_name=mode, allow_cache=True)
    if edges.crs is not None and edges.crs.to_string() != _CRS_MAIN:
        edges = edges.to_crs(_CRS_MAIN)
    return edges


def _load_cells_with_access(
    context, profile: str, kind: str, col: str, metric: str = 'time',
) -> gpd.GeoDataFrame:
    """Load cells + a single access column from
    `cells_access_<metric>_<kind>_<profile>.csv` (10's current schema).
    Use this for the per-mode accessibility heatmaps (frames 14, 16).
    Negative sentinel values (the -1 fill from 10) are converted to NaN
    so they don't pollute the colormap range."""
    cells = _load_cells(context)
    name = f'access_{metric}_{kind}_{profile}'
    access = context.get_properties('cells', name, allow_cache=True)
    if col not in access.columns:
        raise KeyError(
            f"{col!r} not in cells_{name}.csv "
            f"(available examples: {list(access.columns)[:6]})")
    joined = cells.join(access[[col]], how='left')
    # `-1` is the AOI sentinel that 10 writes for cells where no mode
    # was reachable. Treat as NaN for plotting / colour-scaling.
    joined[col] = joined[col].where(joined[col] >= 0)
    return joined


def _load_bike_edges_with_score(context) -> gpd.GeoDataFrame:
    """Bike-network edges + the `bike_infra_score` column from
    `edges_bike_extended.csv` for color-by-score plotting (frames 18-20)."""
    edges = _load_mode_edges(context, 'bike')
    ext = context.get_properties(
        'edges', 'bike_extended', allow_cache=True)
    if 'bike_infra_score' not in ext.columns:
        raise KeyError(
            f"bike_infra_score not in edges_bike_extended.csv "
            f"(available: {list(ext.columns)[:8]})")
    return edges.join(ext[['bike_infra_score']], how='left')


def _load_bike_graph(
    context, weight_col: str = 'duration_calibrated_rbike',
) -> nx.MultiDiGraph:
    """Bike graph + the calibrated-duration weight attached as an edge
    attribute for routing (frame 19)."""
    graph = context.get_nw(data_name='bike', allow_cache=True)
    cal = context.get_properties(
        'edges', 'bike_calibrated', allow_cache=True)
    if weight_col not in cal.columns:
        raise KeyError(
            f"{weight_col!r} not in edges_bike_calibrated.csv "
            f"(available: {list(cal.columns)})")
    attach_edge_properties(graph, cal[[weight_col]])
    return graph


def _compute_bike_infra_score_to_nearest_k_groceries(
    context, k: int = 3, cutoff_s: float = 3600.0,
) -> pd.Series:
    """Per-cell mean `bike_infra_score` along bike routes to the K nearest
    grocery cells. Computed without the tier-radius restriction so
    every reachable origin gets a value (aperta's
    `tiered_path_aggregate` would otherwise restrict to the
    `cells_to_cells` tier — `r_cells = 1.5 km` for bike — and leave
    every cell outside that radius from groceries without a value).

    Implementation: scipy multi-source `dijkstra` rooted at every
    grocery bike-snap node with `cutoff_s` (default 60 min, ≈ 20 km
    bike-route radius). For each origin snap node we then pick the
    K nearest grocery sources reachable, walk the predecessor chain
    back from origin to each, accumulate length-weighted bike_infra_score
    along the recovered edges, and average across whatever was found
    (graceful fallback — uses fewer than K if fewer were reachable).
    Per-snap-node results are mapped back to cells via
    `cells_snap.node_id_bike` (multiple cells can share a snap node)."""
    graph = context.get_nw(data_name='bike', allow_cache=True)
    cal = context.get_properties(
        'edges', 'bike_calibrated', allow_cache=True)
    ext = context.get_properties(
        'edges', 'bike_extended', allow_cache=True)
    attach_edge_properties(
        graph, cal[['duration_calibrated_rbike']])
    attach_edge_properties(graph, ext[['bike_infra_score']])

    with step('build CSR + per-directed-edge (length, score) lookup'):
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)
        # MultiGraph: keep the min-duration edge per (u_idx, v_idx) —
        # matches what scipy's CSR Dijkstra will pick.
        edge_lookup: dict[tuple[int, int], tuple[float, float, float]] = {}
        for u, v, _kk, d in graph.edges(keys=True, data=True):
            dur = float(d.get('duration_calibrated_rbike', np.inf))
            if not np.isfinite(dur):
                continue
            u_i = node_to_idx[u]
            v_i = node_to_idx[v]
            length = float(d.get('length', 0.0))
            score = float(d.get('bike_infra_score', 0.0))
            cur = edge_lookup.get((u_i, v_i))
            if cur is None or dur < cur[0]:
                edge_lookup[(u_i, v_i)] = (dur, length, score)
        rows = np.fromiter((k_[0] for k_ in edge_lookup),
                           dtype=np.int64, count=len(edge_lookup))
        cols = np.fromiter((k_[1] for k_ in edge_lookup),
                           dtype=np.int64, count=len(edge_lookup))
        weights = np.fromiter((v[0] for v in edge_lookup.values()),
                              dtype=np.float64, count=len(edge_lookup))
        csr = sp.csr_matrix(
            (weights, (rows, cols)), shape=(n_nodes, n_nodes))
        logging.info(
            f"  → {n_nodes:,} nodes, {len(edge_lookup):,} directed edges")

    with step('identify grocery + AOI bike-snap nodes'):
        cells_snap = context.get_properties(
            'cells', 'snap', allow_cache=True)
        pois = context.get_properties(
            'cells', 'pois', allow_cache=True)
        if 'poi_errands_groceries' not in pois.columns:
            raise KeyError(
                "poi_errands_groceries not in cells_pois.csv")
        snap_lookup = cells_snap['node_id_bike'].dropna().astype(int)
        grocery_cell_ids = pois.index[pois['poi_errands_groceries'] > 0]
        grocery_node_set = {
            snap_lookup[c] for c in grocery_cell_ids
            if c in snap_lookup.index
        }
        grocery_indices = [
            node_to_idx[n] for n in grocery_node_set if n in node_to_idx]
        # All unique AOI bike-snap nodes — aggregate per-node, then map
        # back to cells (cells sharing a snap node inherit the value).
        aoi_node_set = set(snap_lookup.unique())
        aoi_node_to_idx = {
            n: node_to_idx[n] for n in aoi_node_set if n in node_to_idx}
        logging.info(
            f"  → {len(grocery_indices):,} unique grocery snap nodes, "
            f"{len(aoi_node_to_idx):,} unique AOI snap nodes")

    with step(
        f'scipy multi-source Dijkstra '
        f'({len(grocery_indices)} sources, cutoff {cutoff_s:.0f}s)'
    ):
        dist, pred = sp_dijkstra(
            csr, directed=True, indices=grocery_indices,
            return_predecessors=True, limit=cutoff_s,
        )

    with step(
        f'per-snap-node nearest-{k} weighted bike-score '
        f'(graceful fallback)'
    ):
        per_node: dict = {}
        short_count = 0
        unreach_count = 0
        for node, node_idx in aoi_node_to_idx.items():
            times = dist[:, node_idx]
            finite = np.isfinite(times)
            if not finite.any():
                unreach_count += 1
                continue
            order = np.argsort(times)
            reachable = [int(o) for o in order if finite[o]][:k]
            scores: list[float] = []
            for gi in reachable:
                source_idx = grocery_indices[gi]
                pred_arr = pred[gi]
                # Walk back from node to source via predecessors.
                path: list[int] = [node_idx]
                curr = node_idx
                while curr != source_idx:
                    prev = int(pred_arr[curr])
                    if prev < 0:   # disconnected sentinel
                        path = []
                        break
                    path.append(prev)
                    curr = prev
                if len(path) < 2:
                    continue
                # Reverse to source→target ordering.
                path.reverse()
                tot_len = 0.0
                tot_sxl = 0.0
                for j in range(len(path) - 1):
                    attrs = edge_lookup.get((path[j], path[j + 1]))
                    if attrs is None:
                        continue
                    _, length, score = attrs
                    tot_len += length
                    tot_sxl += score * length
                if tot_len > 0:
                    scores.append(tot_sxl / tot_len)
            # Graceful: use whatever was reachable / reconstructed.
            if scores:
                per_node[node] = sum(scores) / len(scores)
                if len(scores) < k:
                    short_count += 1
        logging.info(
            f"  → scored {len(per_node):,} snap nodes "
            f"(short={short_count:,}, unreachable={unreach_count:,})")

    with step('map per-snap-node value back to cells'):
        per_cell = cells_snap['node_id_bike'].map(per_node)
        per_cell.name = 'bike_infra_score_route'
        logging.info(
            f"  → scored {int(per_cell.notna().sum()):,} cells "
            f"(of {len(per_cell):,} total)")
    return per_cell


def _load_or_compute_bike_infra_score_route(
    context, k: int = 3,
) -> 'pd.Series':
    """Cached wrapper around `_compute_bike_infra_score_to_nearest_k_groceries`.
    First run computes the metric (~30-60 s for Bern; minutes for
    Switzerland) and writes
    `properties/cells_bike_infra_score_route_groceries_k{k}.csv`. Subsequent
    runs load instantly via `context.get_properties`."""
    import pandas as pd
    data_name = f'bike_infra_score_route_groceries_k{k}'
    try:
        df = context.get_properties(
            'cells', data_name, allow_cache=True)
        if 'bike_infra_score_route' in df.columns:
            return df['bike_infra_score_route']
    except Exception:
        pass
    logging.info(
        f"  → no cached cells_{data_name}.csv; computing the "
        "path-aggregated bike-score metric")
    series = _compute_bike_infra_score_to_nearest_k_groceries(context, k=k)
    df = pd.DataFrame({'bike_infra_score_route': series})
    df.index.name = 'cell_id'
    context.create_properties(df, data_name=data_name)
    logging.info(f"  → cached at properties/cells_{data_name}.csv")
    return df['bike_infra_score_route']


def _load_cells_with_min_access(
    context, profiles: list[str], kind: str, col: str, metric: str = 'time',
) -> gpd.GeoDataFrame:
    """Load cells + per-cell MIN across `profiles` for one access
    column. Reads `cells_access_<metric>_<kind>_<profile>.csv` per profile.
    Used for cross-modal "fastest mode" maps (frame 17). Profiles where
    the column is missing entirely are skipped silently.

    `skipna=False` on the per-cell `min` is load-bearing: a cell
    reachable by walk but NOT by car would otherwise inherit walk's
    (often slow) value as its "fastest", which reads as a bug — a
    cell shown as having BAD multi-modal access despite walking being
    the only option. Requiring non-NaN across all profiles excludes
    such cells from the map, matching frame 14's behaviour for cells
    with missing car access. Quick fix pending a deeper look at why
    10 leaves some cells with no car-access value."""
    import pandas as pd
    cells = _load_cells(context)
    per_profile = pd.DataFrame(index=cells.index)
    for profile in profiles:
        name = f'access_{metric}_{kind}_{profile}'
        access = context.get_properties('cells', name, allow_cache=True)
        if col not in access.columns:
            logging.warning(
                f"  → {col!r} missing from {name}; "
                "skipping that mode in the min.")
            continue
        s = access[col]
        # Treat the -1 AOI sentinel as NaN so it doesn't dominate min.
        per_profile[profile] = s.where(s >= 0)
    min_col = per_profile.min(axis=1, skipna=False)
    return cells.join(min_col.rename(col), how='left')


def _load_car_nodes_with_virtual_flag(context) -> gpd.GeoDataFrame:
    """Car-network nodes geometry joined with the `is_virtual` flag
    from `nodes_car_core.csv`. Uses `.join` (index-aligned) to sidestep
    the `node_id`-as-both-column-and-index ambiguity in `get_properties`
    output."""
    nodes = context.get_shapes('nodes', data_name='car', allow_cache=True)
    core = context.get_properties('nodes', 'car_core', allow_cache=True)
    flag_col = 'is_virtual'
    if flag_col in core.columns:
        # `nodes` and `core` are both indexed by `node_id`; `.join`
        # aligns on the index only.
        joined = nodes.join(core[[flag_col]], how='left')
        joined[flag_col] = joined[flag_col].fillna(False).astype(bool)
    else:
        joined = nodes.copy()
        joined[flag_col] = False
    if joined.crs is not None and joined.crs.to_string() != _CRS_MAIN:
        joined = joined.to_crs(_CRS_MAIN)
    return joined


def _shrink_cells_for_fill(
    cells: gpd.GeoDataFrame, shrink_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Apply a small negative buffer to each cell so adjacent cells
    have a visible gap when plotted as filled polygons. Used by frames
    where the cell SHAPE / tessellation is part of the visual story
    (frame 9 close-cells, where the H3 grid structure is the point).
    All non-geometry columns are preserved.

    For pure result-display frames (13, 14, 16, 17, 20 — colored
    metric per cell), use `_cells_as_circles` instead: circles render
    isotropically regardless of projection, sidestepping the slight
    anisotropic skew of H3 hexes in projected CRSs (more pronounced at
    higher latitudes / outside the CRS's central meridian)."""
    if shrink_m is None:
        shrink_m = _CELL_FILL_SHRINK_M
    out = cells.copy()
    out.geometry = out.geometry.buffer(-shrink_m)
    return out


def _cells_as_circles(
    cells: gpd.GeoDataFrame,
    radius_m: float | None = None,
    shrink_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Replace each cell's hex geometry with a circle centred on the
    cell's centroid. `radius_m` defaults to the area-equivalent
    radius of the median cell (`sqrt(median_area / π)`); `shrink_m`
    (default `_CELL_FILL_SHRINK_M`) shrinks each circle so adjacent
    cells have a visible gap.

    Used by result-display frames where the colored metric is the
    point and a clean, isotropic render is preferred over honest H3
    tessellation. Hex frames (3, 6, 9) keep `_shrink_cells_for_fill`
    so the grid structure stays visible."""
    import math
    if cells.empty:
        return cells.copy()
    if shrink_m is None:
        shrink_m = _CELL_FILL_SHRINK_M
    if radius_m is None:
        median_area = float(cells.geometry.area.median())
        radius_m = math.sqrt(median_area / math.pi)
    out = cells.copy()
    out.geometry = cells.geometry.centroid.buffer(max(radius_m - shrink_m, 1.0))
    return out


def _load_cells_with_pois(context, cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subset of `cells` whose `n_pois > 0` in `cells_pois.csv` —
    "cells with at least one location of interest"."""
    pois = context.get_properties('cells', 'pois', allow_cache=True)
    if 'n_pois' not in pois.columns:
        logging.warning("cells_pois.csv missing `n_pois`; using full cells")
        return cells
    has_poi = pois.loc[pois['n_pois'].fillna(0) > 0].index
    return cells[cells.index.isin(has_poi)]


def _load_zones(context) -> gpd.GeoDataFrame:
    """Atlas zones (NPVM traffic zones) with centroid columns."""
    zones = context.get_shapes('zones', allow_cache=True)
    if zones.crs is not None and zones.crs.to_string() != _CRS_MAIN:
        zones = zones.to_crs(_CRS_MAIN)
    return zones


def _load_car_edges_with_attr(
    context, attr_data_name: str, attr_col: str,
) -> gpd.GeoDataFrame:
    """Load car edges + join a single attribute column from
    `edges_<attr_data_name>.csv`. Returns a GeoDataFrame with the
    geometry and `attr_col`."""
    edges = _load_car_edges(context)
    props = context.get_properties('edges', attr_data_name, allow_cache=True)
    if attr_col not in props.columns:
        raise KeyError(f"{attr_col!r} not in edges_{attr_data_name}.csv "
                       f"(available: {list(props.columns)})")
    joined = edges.join(props[[attr_col]], how='left')
    return joined


def _load_car_graph_with_weight(context, weight_col: str) -> nx.MultiDiGraph:
    """Car graph (post-insertion) with `weight_col` attached as an
    edge attribute. Used for routing in frame 12."""
    graph = context.get_nw(data_name='car', allow_cache=True)
    # Calibrated columns live in `edges_car_calibrated.csv`.
    calibrated = context.get_properties(
        'edges', 'car_calibrated', allow_cache=True)
    if weight_col not in calibrated.columns:
        raise KeyError(
            f"{weight_col!r} not in edges_car_calibrated.csv "
            f"(available: {list(calibrated.columns)})")
    attach_edge_properties(graph, calibrated[[weight_col]])
    return graph


def _load_car_nodes_with_core_flags(context) -> gpd.GeoDataFrame:
    """Car nodes geometry joined with the junction-type flags from
    `nodes_car_core.csv` (is_virtual, is_4way / _major / _anchor,
    is_traffic_signal, is_t_junction / _major / _anchor)."""
    nodes = context.get_shapes('nodes', data_name='car', allow_cache=True)
    core = context.get_properties('nodes', 'car_core', allow_cache=True)
    flag_cols = [
        'is_virtual',
        'is_4way', 'is_4way_major', 'is_4way_anchor',
        'is_t_junction', 'is_t_junction_major', 'is_t_junction_anchor',
        'is_traffic_signal',
    ]
    present = [c for c in flag_cols if c in core.columns]
    if present:
        joined = nodes.join(core[present], how='left')
        for c in present:
            joined[c] = joined[c].fillna(False).astype(bool)
    else:
        joined = nodes.copy()
    if joined.crs is not None and joined.crs.to_string() != _CRS_MAIN:
        joined = joined.to_crs(_CRS_MAIN)
    return joined


def _snap_xy_to_nearest_node(
    xy: tuple[float, float], nodes_gdf: gpd.GeoDataFrame,
) -> int:
    """Return the `node_id` of the nodes-GDF row closest to `xy`
    (LV95 coords). Uses a KDTree over node geometries."""
    coords = np.column_stack([nodes_gdf.geometry.x, nodes_gdf.geometry.y])
    tree = cKDTree(coords)
    _, idx = tree.query([xy], k=1)
    return int(nodes_gdf.iloc[int(idx[0])]['node_id'])


def _route_path_edges(
    graph: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame,
    orig_node: int, dest_node: int, weight_col: str,
) -> tuple[gpd.GeoDataFrame, float]:
    """Return `(route_edges_gdf, total_cost)` — a GeoDataFrame containing
    the edges along the shortest path (preserving all columns from
    `edges_gdf`, so callers can color by any attribute), plus the path
    cost in `weight_col` units. Returns `(empty_gdf, 0.0)` if no path."""
    try:
        path_nodes = nx.shortest_path(
            graph, source=orig_node, target=dest_node, weight=weight_col)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return gpd.GeoDataFrame(
            columns=edges_gdf.columns, crs=edges_gdf.crs), 0.0
    edges_indexed = (edges_gdf.set_index('edge_id')
                     if 'edge_id' in edges_gdf.columns
                     else edges_gdf)
    eids: list[str] = []
    total_cost = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_dict = graph.get_edge_data(u, v) or {}
        if not edge_dict:
            continue
        best_k = min(edge_dict,
                     key=lambda k: float(edge_dict[k].get(weight_col, np.inf)))
        eid = f"{u}:{v}:{best_k}"
        if eid in edges_indexed.index:
            eids.append(eid)
        total_cost += float(edge_dict[best_k].get(weight_col, 0.0))
    if not eids:
        return gpd.GeoDataFrame(
            columns=edges_gdf.columns, crs=edges_gdf.crs), 0.0
    return edges_indexed.loc[eids].reset_index(), total_cost


def _compute_cell_overheads(
    context, profile: str = 'car_peak', side: str = 'origin',
) -> 'pd.Series':
    """Per-cell trip overhead (seconds) for a given profile via
    `main.common.per_cell_road_overheads` — supports 07a's current
    shared endpoint coefs (`density`, `snap_dist`) as well as the older
    split (`orig_density`/`dest_density`, `orig_snap_dist`/`dest_snap_dist`)
    convention. `side ∈ {'origin', 'destination'}`. `is_multi_leg_trip`
    is treated as an ODM leg (=0) so its coef doesn't contribute here."""
    import pandas as pd
    from main.common import per_cell_road_overheads

    cells_snap = context.get_properties('cells', 'snap', allow_cache=True)
    node_props = context.get_properties(
        'nodes', 'car_extended', allow_cache=True)
    coefs = context.get_coefs('overheads_road')

    snap = cells_snap['node_id_car'].dropna().astype(int)
    density = snap.map(node_props['density_r1000_norm']).fillna(0.0)
    cells_df = pd.DataFrame({'density': density}, index=snap.index)
    orig, dest = per_cell_road_overheads(
        cells_df, coefs[profile],
        density_col='density', snap_dist_col='snap_dist',
    )
    return orig if side == 'origin' else dest


# ---------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------

def frame_01_buildings_zoom_in(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 1 — zoomed-in view (1×1 km), buildings visible on basemap."""
    fig, ax = make_fig(cfg, extent_key='in')
    buildings = _load_buildings(context, scenario)
    bldg_in_view = _clip_to_extent(buildings, ax)
    bldg_in_view.plot(
        ax=ax, color=_BLDG_FILL, edgecolor=_BLDG_EDGE,
        linewidth=_BLDG_LW, zorder=2,
    )
    return fig


def frame_02_pick_location(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 2 — same view as frame 1, but with the focal point marked
    and the building containing it highlighted."""
    fig, ax = make_fig(cfg, extent_key='in')
    buildings = _load_buildings(context, scenario)
    bldg_in_view = _clip_to_extent(buildings, ax)
    focal_bldg = _find_focal_building(buildings, cfg)
    # Other buildings first (faded), then the focal building on top.
    others = bldg_in_view[~bldg_in_view.index.isin(focal_bldg.index)]
    others.plot(
        ax=ax, color=_BLDG_FILL, edgecolor=_BLDG_EDGE,
        linewidth=_BLDG_LW, zorder=2,
    )
    if not focal_bldg.empty:
        focal_bldg.plot(
            ax=ax, color=_FOCAL_BLDG_FILL, edgecolor=_FOCAL_BLDG_EDGE,
            linewidth=_FOCAL_BLDG_LW, zorder=3,
        )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=4,
    )
    return fig


def frame_03_assign_to_cell(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 3 — overlay the cell containing the focal point on the
    same zoomed-in building view. Buildings sit underneath the
    semi-transparent cell fill."""
    fig, ax = make_fig(cfg, extent_key='in')
    buildings = _load_buildings(context, scenario)
    bldg_in_view = _clip_to_extent(buildings, ax)
    bldg_in_view.plot(
        ax=ax, color=_BLDG_FILL, edgecolor=_BLDG_EDGE,
        linewidth=_BLDG_LW, zorder=2,
    )
    # Focal cell — fill semi-transparent + bold outline.
    cells = _load_cells(context)
    focal_cell = _find_focal_cell(cells, cfg)
    if not focal_cell.empty:
        focal_cell.plot(
            ax=ax, color=_FOCAL_CELL_FILL, alpha=_FOCAL_CELL_ALPHA,
            edgecolor=_CELL_EDGE, linewidth=_FOCAL_CELL_EDGE_LW,
            zorder=3,
        )
    # Focal building + point still visible.
    focal_bldg = _find_focal_building(buildings, cfg)
    if not focal_bldg.empty:
        focal_bldg.plot(
            ax=ax, color=_FOCAL_BLDG_FILL, edgecolor=_FOCAL_BLDG_EDGE,
            linewidth=_FOCAL_BLDG_LW, zorder=4,
        )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_04_zoom_out(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 4 — zoom out to 5×5 km. Buildings recede into background
    texture; the focal cell stays highlighted as the visual anchor."""
    fig, ax = make_fig(cfg, extent_key='med')
    buildings = _load_buildings(context, scenario)
    bldg_in_view = _clip_to_extent(buildings, ax)
    # At this zoom, building outlines disappear visually if drawn —
    # render as fill-only with a very thin edge.
    bldg_in_view.plot(
        ax=ax, color=_BLDG_FILL, edgecolor='none',
        linewidth=0, zorder=2,
    )
    cells = _load_cells(context)
    focal_cell = _find_focal_cell(cells, cfg)
    if not focal_cell.empty:
        focal_cell.plot(
            ax=ax, color=_FOCAL_CELL_FILL, alpha=_FOCAL_CELL_ALPHA,
            edgecolor=_CELL_EDGE, linewidth=_FOCAL_CELL_EDGE_LW,
            zorder=3,
        )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=4,
    )
    return fig


# ---------------------------------------------------------------------
# Phase 2 frames — network + cells + tiers
# ---------------------------------------------------------------------

def frame_05_network_pre_insert(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 5 — show the full car network at medium zoom, with ONLY
    the native (un-densified) nodes drawn. Edges are shown in full
    (cells will be added in frame 6; virtual nodes appear in frame 7)."""
    fig, ax = make_fig(cfg, extent_key='med')
    edges = _load_car_edges(context)
    edges_in_view = _clip_to_extent(edges, ax)
    edges_in_view.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH,
        alpha=_NETWORK_ALPHA, zorder=3,
    )
    # Native nodes only (i.e. is_virtual=False) — same colour as the
    # network so they read as nodes of the un-densified graph.
    nodes = _load_car_nodes_with_virtual_flag(context)
    native = nodes[~nodes['is_virtual']]
    native_in_view = _clip_to_extent(native, ax)
    native_in_view.plot(
        ax=ax, color=_NATIVE_NODE_COLOR, markersize=_NATIVE_NODE_SIZE,
        linewidth=0, alpha=_NETWORK_ALPHA, zorder=4,
    )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=6,
    )
    return fig


def frame_06_cell_grid(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 6 — overlay the cell outlines (filtered to cells with
    ≥ 1 POI) on top of the frame-5 layer (full edges + native nodes)."""
    fig, ax = make_fig(cfg, extent_key='med')
    edges = _load_car_edges(context)
    edges_in_view = _clip_to_extent(edges, ax)
    edges_in_view.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH,
        alpha=_NETWORK_ALPHA, zorder=3,
    )
    nodes = _load_car_nodes_with_virtual_flag(context)
    native = nodes[~nodes['is_virtual']]
    native_in_view = _clip_to_extent(native, ax)
    native_in_view.plot(
        ax=ax, color=_NATIVE_NODE_COLOR, markersize=_NATIVE_NODE_SIZE,
        linewidth=0, alpha=_NETWORK_ALPHA, zorder=4,
    )
    cells = _load_cells(context)
    cells_with_pois = _load_cells_with_pois(context, cells)
    cells_in_view = _clip_to_extent(cells_with_pois, ax)
    cells_in_view.boundary.plot(
        ax=ax, color=_CELL_OUTLINE_COLOR, linewidth=_CELL_OUTLINE_WIDTH,
        alpha=_CELL_OUTLINE_ALPHA, zorder=5,
    )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=6,
    )
    return fig


def frame_07_virtual_nodes(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 7 — add the virtual (cell-snap) nodes in purple on top of
    the frame-6 composition (edges + native nodes + POI-filtered cell
    outlines). The 'native vs virtual' contrast is the headline."""
    fig, ax = make_fig(cfg, extent_key='med')
    edges = _load_car_edges(context)
    edges_in_view = _clip_to_extent(edges, ax)
    edges_in_view.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH,
        alpha=_NETWORK_ALPHA, zorder=3,
    )
    nodes = _load_car_nodes_with_virtual_flag(context)
    native = nodes[~nodes['is_virtual']]
    native_in_view = _clip_to_extent(native, ax)
    native_in_view.plot(
        ax=ax, color=_NATIVE_NODE_COLOR, markersize=_NATIVE_NODE_SIZE,
        linewidth=0, alpha=_NETWORK_ALPHA, zorder=4,
    )
    cells = _load_cells(context)
    cells_with_pois = _load_cells_with_pois(context, cells)
    cells_in_view = _clip_to_extent(cells_with_pois, ax)
    cells_in_view.boundary.plot(
        ax=ax, color=_CELL_OUTLINE_COLOR, linewidth=_CELL_OUTLINE_WIDTH,
        alpha=_CELL_OUTLINE_ALPHA, zorder=5,
    )
    # Virtual nodes (cell-snap insertions) — larger, purple.
    virt_nodes = nodes[nodes['is_virtual']]
    virt_in_view = _clip_to_extent(virt_nodes, ax)
    virt_in_view.plot(
        ax=ax, color=_VIRTUAL_NODE_COLOR, markersize=_VIRTUAL_NODE_SIZE,
        edgecolor=_VIRTUAL_NODE_EDGE, linewidth=_VIRTUAL_NODE_EDGE_WIDTH,
        zorder=6,
    )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=7,
    )
    return fig


def frame_08_zoom_out_far(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 8 — zoom out to the tier-system scale (~100 km wide),
    showing the POI-filtered cell layer and the car network at
    regional scale. Same extent as frame 9, so frames 8 → 9 read as a
    continuous transition."""
    fig, ax = make_fig(cfg, extent_key='tier')
    edges = _load_car_edges(context)
    edges_in_view = _clip_to_extent(edges, ax)
    edges_in_view.plot(
        ax=ax, color=_NETWORK_COLOR, linewidth=_NETWORK_WIDTH * 0.5,
        alpha=_NETWORK_ALPHA * 0.5, zorder=3,
    )
    cells = _load_cells(context)
    cells_with_pois = _load_cells_with_pois(context, cells)
    cells_in_view = _clip_to_extent(cells_with_pois, ax)
    cells_in_view.boundary.plot(
        ax=ax, color=_CELL_OUTLINE_COLOR, linewidth=_CELL_OUTLINE_WIDTH * 0.5,
        alpha=_CELL_OUTLINE_ALPHA, zorder=4,
    )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_09_tiers(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 9 — introduce the three-tier system. Concentric circles at
    `r_cells / r_medium / r_zones` (bike case) around the focal point;
    zones whose centroid falls in the cells-to-zones band shown in
    peach, zones in the zones-to-zones band shown in light blue.
    Each individual zone is outlined in its tier's colour so the
    discrete-zone structure stays readable through the translucent fill."""
    fig, ax = make_fig(cfg, extent_key='tier')
    radii = scenario.mode_configs['bike'].radii
    x0, y0 = cfg.center_xy

    # Zone fills by tier band — uses zone centroid distance to focal.
    zones = _load_zones(context)
    if 'centroid_x' in zones.columns and 'centroid_y' in zones.columns:
        zx, zy = zones['centroid_x'], zones['centroid_y']
    else:
        cs = zones.geometry.centroid
        zx, zy = cs.x, cs.y
    dist = ((zx - x0) ** 2 + (zy - y0) ** 2) ** 0.5
    medium = zones[(dist > radii.r_cells) & (dist <= radii.r_medium)]
    far = zones[(dist > radii.r_medium) & (dist <= radii.r_zones)]
    medium_in_view = _clip_to_extent(medium, ax)
    far_in_view = _clip_to_extent(far, ax)

    # Fill (translucent) + boundary (opaque) plotted separately per
    # tier so the per-zone outlines read at full alpha against the
    # softer fill underneath.
    far_in_view.plot(
        ax=ax, color=_ZONE_FAR_FILL, alpha=_ZONE_FILL_ALPHA,
        edgecolor='none', zorder=3,
    )
    far_in_view.boundary.plot(
        ax=ax, color=_ZONE_FAR_EDGE, linewidth=_ZONE_EDGE_WIDTH,
        alpha=_ZONE_EDGE_ALPHA, zorder=3.5,
    )
    medium_in_view.plot(
        ax=ax, color=_ZONE_MEDIUM_FILL, alpha=_ZONE_FILL_ALPHA,
        edgecolor='none', zorder=4,
    )
    medium_in_view.boundary.plot(
        ax=ax, color=_ZONE_MEDIUM_EDGE, linewidth=_ZONE_EDGE_WIDTH,
        alpha=_ZONE_EDGE_ALPHA, zorder=4.5,
    )

    # Cells in the cells_to_cells (close) band — actual hex cells whose
    # centroid is within `r_cells` of the focal point. Shown in green
    # to distinguish from the (zone-resolution) medium / far bands.
    cells = _load_cells(context)
    cells_centroids = cells.geometry.centroid
    dist_cells = ((cells_centroids.x - x0) ** 2
                  + (cells_centroids.y - y0) ** 2) ** 0.5
    close_cells = cells[dist_cells <= radii.r_cells]
    close_in_view = _shrink_cells_for_fill(_clip_to_extent(close_cells, ax))
    close_in_view.plot(
        ax=ax, color=_TIER_CLOSE_FILL, alpha=_ZONE_FILL_ALPHA,
        edgecolor='none', zorder=4.7,
    )
    close_in_view.boundary.plot(
        ax=ax, color=_TIER_CLOSE_EDGE, linewidth=_ZONE_EDGE_WIDTH,
        alpha=_ZONE_EDGE_ALPHA, zorder=4.8,
    )

    # Three concentric tier circles.
    for r in (radii.r_cells, radii.r_medium, radii.r_zones):
        ax.add_patch(Circle(
            (x0, y0), r,
            fill=False, edgecolor=_TIER_CIRCLE_COLOR,
            linewidth=_TIER_CIRCLE_WIDTH, alpha=_TIER_CIRCLE_ALPHA,
            zorder=5,
        ))

    # Focal point on top.
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=6,
    )
    return fig


# ---------------------------------------------------------------------
# Phase 3 frames — flows → calibrated speeds → example routes → overheads
# ---------------------------------------------------------------------

def frame_10_traffic_flows(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 10 — show estimated traffic flows on the car network.
    Edge color scaled by `flow_estimate`; edge width also grows with
    flow so the high-flow corridors pop visually."""
    fig, ax = make_fig(cfg, extent_key='med')
    # Baseline: every car edge in a faint gray underneath, so edges
    # with zero / missing `flow_estimate` (which would otherwise land
    # at the bottom of the YlOrRd colormap and read as near-invisible
    # against the basemap) still show as part of the network.
    base_edges = _load_car_edges(context)
    base_in_view = _clip_to_extent(base_edges, ax)
    if not base_in_view.empty:
        base_in_view.plot(
            ax=ax, color=_NETWORK_FADED_COLOR,
            linewidth=_NETWORK_WIDTH, alpha=_NETWORK_FADED_ALPHA,
            zorder=2,
        )
    edges = _load_car_edges_with_attr(
        context, 'car_flows', 'flow_estimate')
    edges_in_view = _clip_to_extent(edges, ax)
    if not edges_in_view.empty:
        flows = edges_in_view['flow_estimate'].fillna(0.0).clip(lower=0)
        vmax = float(flows.quantile(0.98)) if len(flows) else 1.0
        vmax = max(vmax, 1.0)
        norm = Normalize(vmin=0.0, vmax=vmax)
        # geopandas .plot accepts only a scalar linewidth; bin the
        # edges by flow quartile and plot each bin with a progressively
        # thicker line so high-flow corridors pop. The last bin's
        # upper bound is `+inf` rather than `vmax * 1.0001` — `vmax`
        # is the 98th percentile, so a finite ceiling would silently
        # drop the top 2 % of edges (motorways / trunk roads).
        bin_edges = [0.0, vmax * 0.25, vmax * 0.5, vmax * 0.75,
                     float('inf')]
        bin_widths = [_FLOW_LW_MIN,
                      _FLOW_LW_MIN + 0.33 * (_FLOW_LW_MAX - _FLOW_LW_MIN),
                      _FLOW_LW_MIN + 0.67 * (_FLOW_LW_MAX - _FLOW_LW_MIN),
                      _FLOW_LW_MAX]
        for i, lw in enumerate(bin_widths):
            mask = (flows >= bin_edges[i]) & (flows < bin_edges[i + 1])
            bin_gdf = edges_in_view[mask]
            if bin_gdf.empty:
                continue
            bin_gdf.plot(
                ax=ax, column='flow_estimate', cmap=_FLOW_CMAP,
                norm=norm, linewidth=lw, alpha=0.95, zorder=3 + i * 0.1,
            )
        sm = ScalarMappable(cmap=_FLOW_CMAP, norm=norm)
        sm.set_array([])
        add_colorbar(fig, cfg, sm, label='Flow estimate (veh/day)')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_11_calibrated_speeds(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 11 — show the calibrated effective speed per edge for the
    car-peak profile. Red = slow, green = fast (the RdYlGn cmap).
    Uses the shared `_SPEED_VMIN / _SPEED_VMAX` range so frame 12's
    route coloring keys to the exact same scale."""
    return _calibrated_speeds_frame(cfg, context, profile='car_peak')


def frame_11b_calibrated_speeds_night(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Frame 11b — same as frame 11 but for the `car_night` profile.
    Shares the colour scale with frame 11 so the two are directly
    comparable: same hue at the same speed."""
    return _calibrated_speeds_frame(cfg, context, profile='car_night')


def _calibrated_speeds_frame(
    cfg: StoryConfig, context, *, profile: str,
) -> Figure:
    """Body shared by frame 11 (`profile='car_peak'`) and frame 11b
    (`profile='car_night'`). Plot the per-edge calibrated effective
    speed on the `_SPEED_CMAP` / `_SPEED_VMIN-_SPEED_VMAX` scale."""
    col = f'effective_speed_kph_{profile}'
    fig, ax = make_fig(cfg, extent_key='med')
    edges = _load_car_edges_with_attr(context, 'car_calibrated', col)
    edges_in_view = _clip_to_extent(edges, ax)
    norm = Normalize(vmin=_SPEED_VMIN, vmax=_SPEED_VMAX)
    if not edges_in_view.empty:
        edges_in_view.plot(
            ax=ax, column=col, cmap=_SPEED_CMAP, norm=norm,
            linewidth=_SPEED_LW, alpha=0.95, zorder=3,
        )
    sm = ScalarMappable(cmap=_SPEED_CMAP, norm=norm)
    sm.set_array([])
    add_colorbar(fig, cfg, sm, label='Effective speed (km/h)')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_12_routes_intersections(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Frame 12 — faded car network + intersection-node overlay
    classifying every junction as traffic signal / 4-way / t-junction.
    No routes (was previously shortest-path overlays from focal point
    to `cfg.destinations_latlon`)."""
    from matplotlib.lines import Line2D
    import pandas as pd

    fig, ax = make_fig(cfg, extent_key='med')

    # Faded car network as a backdrop for the intersection points.
    edges = _load_car_edges_with_attr(
        context, 'car_calibrated', 'effective_speed_kph_car_peak')
    edges_in_view = _clip_to_extent(edges, ax)
    edges_in_view.plot(
        ax=ax, color=_FADED_NETWORK_COLOR, linewidth=_FADED_NETWORK_LW,
        alpha=_FADED_NETWORK_ALPHA, zorder=2,
    )

    # Intersection classification. A node can be tagged as more than
    # one type (e.g. a 4-way intersection that ALSO has a signal); the
    # render priority is signal > 4-way > t-junction so the highest-
    # friction class wins visually.
    nodes = _load_car_nodes_with_core_flags(context)
    nodes_in_view = _clip_to_extent(nodes, ax)

    def _col_mask(df, col):
        """Boolean Series for `col` if present, else all-False."""
        if col in df.columns:
            return df[col].astype(bool)
        return pd.Series(False, index=df.index)

    signal_mask = _col_mask(nodes_in_view, 'is_traffic_signal')
    fourway_mask = (
        _col_mask(nodes_in_view, 'is_4way_major')
        | _col_mask(nodes_in_view, 'is_4way_anchor')
    )
    tjunction_mask = (
        _col_mask(nodes_in_view, 'is_t_junction_major')
        | _col_mask(nodes_in_view, 'is_t_junction_anchor')
    )
    signals = nodes_in_view[signal_mask]
    fourways = nodes_in_view[fourway_mask & ~signal_mask]
    tjuncs = nodes_in_view[tjunction_mask & ~fourway_mask & ~signal_mask]

    if not tjuncs.empty:
        tjuncs.plot(
            ax=ax, color=_INT_T_JUNCTION_COLOR,
            markersize=_INT_T_JUNCTION_SIZE,
            edgecolor='white', linewidth=0.4, zorder=6,
        )
    if not fourways.empty:
        fourways.plot(
            ax=ax, color=_INT_4WAY_COLOR, markersize=_INT_4WAY_SIZE,
            edgecolor='white', linewidth=0.4, zorder=7,
        )
    if not signals.empty:
        signals.plot(
            ax=ax, color=_INT_SIGNAL_COLOR, markersize=_INT_SIGNAL_SIZE,
            edgecolor='white', linewidth=0.5, zorder=8,
        )

    # Focal point on top.
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=9,
    )

    # Intersection-type legend.
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_INT_SIGNAL_COLOR, markeredgecolor='white',
               markersize=14, label='Traffic signal'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_INT_4WAY_COLOR, markeredgecolor='white',
               markersize=9, label='4-way'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_INT_T_JUNCTION_COLOR,
               markeredgecolor='white',
               markersize=9, label='T-junction'),
    ]
    add_legend(
        fig, cfg,
        handles_labels=(
            legend_handles, [h.get_label() for h in legend_handles]),
    )
    return fig


def frame_13_cell_overheads(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 13 — per-cell trip overhead for the car-peak profile
    (constant + density-based, from `coefs/overheads_road.csv`).
    Color map shows where parking-search / out-of-vehicle time
    dominates: warmer = larger overhead."""
    fig, ax = make_fig(cfg, extent_key='med')
    cells = _load_cells(context)
    overheads = _compute_cell_overheads(
        context, profile='car_peak', side='origin')
    cells_with_oh = cells.join(overheads.rename('overhead_s'), how='inner')
    cells_in_view = _cells_as_circles(
        _clip_to_extent(cells_with_oh, ax))
    if not cells_in_view.empty:
        oh = cells_in_view['overhead_s']
        vmin = float(oh.quantile(0.02))
        vmax = float(oh.quantile(0.98))
        if vmax <= vmin:
            vmin, vmax = 0.0, max(vmax, 1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cells_in_view.plot(
            ax=ax, column='overhead_s', cmap=_OVERHEAD_CMAP,
            norm=norm, edgecolor='none', alpha=0.7, zorder=3,
        )
        sm = ScalarMappable(cmap=_OVERHEAD_CMAP, norm=norm)
        sm.set_array([])
        add_colorbar(
            fig, cfg, sm,
            label='Origin overhead, car peak (s)')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


# ---------------------------------------------------------------------
# Phase 4 frames — accessibility + multi-modal comparison
# ---------------------------------------------------------------------

def _plot_access_heatmap(
    ax: Axes, cells_in_view: gpd.GeoDataFrame, col: str,
) -> None:
    """Shared painter for frames 14/16/17 — colour cells by `col`
    using the shared `_ACCESS_CMAP / _ACCESS_VMIN / _ACCESS_VMAX`."""
    norm = Normalize(vmin=_ACCESS_VMIN, vmax=_ACCESS_VMAX)
    cells_in_view.plot(
        ax=ax, column=col, cmap=_ACCESS_CMAP, norm=norm,
        edgecolor='none', alpha=_ACCESS_CELL_ALPHA, zorder=3,
    )


def frame_14_access_car(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 14 — zoom back in to medium; show per-cell mean travel
    time to the nearest 3 groceries by CAR (peak profile)."""
    fig, ax = make_fig(cfg, extent_key='med')
    cells = _load_cells_with_access(
        context, 'car_peak', 'nearest_k', _ACCESS_COLUMN)
    cells_in_view = _cells_as_circles(_clip_to_extent(cells, ax))
    if not cells_in_view.empty:
        _plot_access_heatmap(ax, cells_in_view, _ACCESS_COLUMN)
        sm = ScalarMappable(
            cmap=_ACCESS_CMAP,
            norm=Normalize(vmin=_ACCESS_VMIN, vmax=_ACCESS_VMAX))
        sm.set_array([])
        add_colorbar(fig, cfg, sm, label=_ACCESS_LABEL + '  · car peak')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_15_walking_network(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 15 — the walking network."""
    fig, ax = make_fig(cfg, extent_key='med')
    walk_edges = _load_mode_edges(context, 'walk')
    in_view = _clip_to_extent(walk_edges, ax)
    if not in_view.empty:
        in_view.plot(
            ax=ax, color=_NW_WALK_COLOR, linewidth=_NW_WALK_LW,
            alpha=_NW_OVERLAY_ALPHA, zorder=5,
        )
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=7,
    )
    return fig


def frame_16_access_walk(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 16 — same access heatmap as frame 14 but for the WALK
    profile. Same colour scale so frames 14 and 16 are directly
    comparable side-by-side."""
    fig, ax = make_fig(cfg, extent_key='med')
    cells = _load_cells_with_access(
        context, 'rwalk', 'nearest_k', _ACCESS_COLUMN)
    cells_in_view = _cells_as_circles(_clip_to_extent(cells, ax))
    if not cells_in_view.empty:
        _plot_access_heatmap(ax, cells_in_view, _ACCESS_COLUMN)
        sm = ScalarMappable(
            cmap=_ACCESS_CMAP,
            norm=Normalize(vmin=_ACCESS_VMIN, vmax=_ACCESS_VMAX))
        sm.set_array([])
        add_colorbar(fig, cfg, sm, label=_ACCESS_LABEL + '  · walk')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


def frame_17_access_fastest(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 17 — cross-modal "fastest of walk vs car" map. Per cell,
    take the MIN across walk / car_peak access times. Same colour
    scale as frames 14/16 so the colour intensity is directly
    comparable across all three frames."""
    fig, ax = make_fig(cfg, extent_key='med')
    cells = _load_cells_with_min_access(
        context, ['rwalk', 'car_peak'],
        'nearest_k', _ACCESS_COLUMN,
    )
    cells_in_view = _cells_as_circles(_clip_to_extent(cells, ax))
    if not cells_in_view.empty:
        _plot_access_heatmap(ax, cells_in_view, _ACCESS_COLUMN)
        sm = ScalarMappable(
            cmap=_ACCESS_CMAP,
            norm=Normalize(vmin=_ACCESS_VMIN, vmax=_ACCESS_VMAX))
        sm.set_array([])
        add_colorbar(
            fig, cfg, sm,
            label=_ACCESS_LABEL + '  · min(walk, car)')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


# ---------------------------------------------------------------------
# Phase 5 frames — bike-specific deep-dive (bike_infra_score, utility cost)
# ---------------------------------------------------------------------

def _plot_bike_bg_networks(context, ax: Axes) -> None:
    """Faded car + walk networks shown as context behind the bike
    network (frames 18-20). Single light grey at low alpha."""
    for mode in ('car', 'walk'):
        edges = _load_mode_edges(context, mode)
        in_view = _clip_to_extent(edges, ax)
        if not in_view.empty:
            in_view.plot(
                ax=ax, color=_BIKE_BG_COLOR, linewidth=_BIKE_BG_LW,
                alpha=_BIKE_BG_ALPHA, zorder=2,
            )


def frame_18_bike_infra_score(context, scenario, cfg: StoryConfig) -> Figure:
    """Frame 18 — fade the car + walk networks, then show the bike
    network coloured by `bike_infra_score` (0-2). The visual point:
    bike-friendliness varies edge-by-edge, and aperta carries
    arbitrary per-edge feature attributes (here `bike_infra_score`) that
    can flow into routing or utility computation."""
    fig, ax = make_fig(cfg, extent_key='med')
    _plot_bike_bg_networks(context, ax)
    bike_edges = _load_bike_edges_with_score(context)
    in_view = _clip_to_extent(bike_edges, ax)
    norm = Normalize(vmin=_BIKE_SCORE_VMIN, vmax=_BIKE_SCORE_VMAX)
    if not in_view.empty:
        in_view.plot(
            ax=ax, column='bike_infra_score', cmap=_BIKE_SCORE_CMAP, norm=norm,
            linewidth=_BIKE_NW_LW, alpha=_BIKE_NW_ALPHA, zorder=3,
        )
    sm = ScalarMappable(cmap=_BIKE_SCORE_CMAP, norm=norm)
    sm.set_array([])
    add_colorbar(fig, cfg, sm, label='Bike score')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=6,
    )
    return fig


def frame_19_bike_routes_time(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Frame 19 — bike routes from the focal point to each destination,
    routed for shortest TIME (`duration_calibrated_rbike`). Each route
    is then colored edge-by-edge by `bike_infra_score` so the reader sees
    the bike-friendliness profile of the time-optimal path."""
    fig, ax = make_fig(cfg, extent_key='med')
    _plot_bike_bg_networks(context, ax)
    # Bike network faded so routes stand out.
    bike_edges = _load_bike_edges_with_score(context)
    bike_in_view = _clip_to_extent(bike_edges, ax)
    bike_in_view.plot(
        ax=ax, color=_BIKE_BG_COLOR, linewidth=_BIKE_BG_LW,
        alpha=_BIKE_BG_ALPHA, zorder=3,
    )

    weight_col = 'duration_calibrated_rbike'
    graph = _load_bike_graph(context, weight_col=weight_col)
    nodes_for_snap = context.get_shapes(
        'nodes', data_name='bike', allow_cache=True)
    if nodes_for_snap.crs is not None and \
            nodes_for_snap.crs.to_string() != _CRS_MAIN:
        nodes_for_snap = nodes_for_snap.to_crs(_CRS_MAIN)
    norm = Normalize(vmin=_BIKE_SCORE_VMIN, vmax=_BIKE_SCORE_VMAX)
    x0, y0 = cfg.center_xy
    orig_node = _snap_xy_to_nearest_node((x0, y0), nodes_for_snap)
    for i, (lat, lon) in enumerate(cfg.destinations_latlon):
        dx, dy = _latlon_to_lv95(lat, lon)
        dest_node = _snap_xy_to_nearest_node((dx, dy), nodes_for_snap)
        route_edges, total_cost = _route_path_edges(
            graph, bike_edges, orig_node, dest_node, weight_col)
        if not route_edges.empty:
            route_edges.plot(
                ax=ax, column='bike_infra_score',
                cmap=_BIKE_SCORE_CMAP, norm=norm,
                linewidth=_BIKE_ROUTE_LW, alpha=0.95, zorder=5,
            )
        ax.scatter(
            [dx], [dy], s=_DEST_MARKER_SIZE,
            color=_FOCAL_POINT_COLOR, edgecolor=_DEST_MARKER_EDGE,
            linewidth=1.4, zorder=6,
        )
        logging.info(
            f"  → bike route {i+1} (time-opt): {total_cost/60:.1f} min")
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=7,
    )
    sm = ScalarMappable(cmap=_BIKE_SCORE_CMAP, norm=norm)
    sm.set_array([])
    add_colorbar(fig, cfg, sm, label='Bike score')
    return fig


def frame_19_bike_infra_score_to_groceries(
    context, scenario, cfg: StoryConfig,
) -> Figure:
    """Frame 20 — per-AOI-cell, mean `bike_infra_score` along the bike routes
    to the cell's `k` nearest grocery cells, averaged across those `k`
    routes (default k=3). Computed via scipy multi-source Dijkstra
    rooted at every grocery bike-snap node + per-cell path
    reconstruction + length-weighted bike-score along each path.

    First-time compute is cached at
    `properties/cells_bike_infra_score_route_groceries_k3.csv`; subsequent
    re-runs of the story load that file instantly. To recompute
    (e.g. after a different cells / network upstream), delete the
    cache file and re-run."""
    fig, ax = make_fig(cfg, extent_key='med')
    cells = _load_cells(context)
    score = _load_or_compute_bike_infra_score_route(context, k=3)
    cells_with_score = cells.join(
        score.rename('bike_infra_score_route'), how='left')
    cells_in_view = _cells_as_circles(
        _clip_to_extent(cells_with_score, ax))
    norm = Normalize(vmin=_BIKE_SCORE_VMIN, vmax=_BIKE_SCORE_VMAX)
    if not cells_in_view.empty:
        cells_in_view.plot(
            ax=ax, column='bike_infra_score_route',
            cmap=_BIKE_SCORE_CMAP, norm=norm,
            edgecolor='none', alpha=_ACCESS_CELL_ALPHA, zorder=3,
        )
    sm = ScalarMappable(cmap=_BIKE_SCORE_CMAP, norm=norm)
    sm.set_array([])
    add_colorbar(
        fig, cfg, sm,
        label='Mean bike score, route to nearest 3 groceries')
    x0, y0 = cfg.center_xy
    ax.scatter(
        [x0], [y0], s=_FOCAL_POINT_SIZE, color=_FOCAL_POINT_COLOR,
        edgecolor='white', linewidth=1.2, zorder=5,
    )
    return fig


# ---------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------

# Order matters: frame N is saved as `frame_NN.png` in this order.
_FRAMES = [
    frame_01_buildings_zoom_in,
    frame_02_pick_location,
    frame_03_assign_to_cell,
    frame_04_zoom_out,
    frame_05_network_pre_insert,
    frame_06_cell_grid,
    frame_07_virtual_nodes,
    frame_08_zoom_out_far,
    frame_09_tiers,
    frame_10_traffic_flows,
    frame_11_calibrated_speeds,
    frame_11b_calibrated_speeds_night,
    frame_12_routes_intersections,
    frame_13_cell_overheads,
    frame_14_access_car,
    frame_15_walking_network,
    frame_16_access_walk,
    frame_17_access_fastest,
    frame_18_bike_infra_score,
    frame_19_bike_infra_score_to_groceries,
    # frame_XX_bike_routes_time,
]


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    # Set the per-scenario metric CRS BEFORE any frame runs — frame
    # helpers (`_load_buildings`, `_load_cells`, …) read this module-
    # level variable to know which CRS to reproject into.
    global _CRS_MAIN
    _CRS_MAIN = scenario.crs_main

    defaults = _DEFAULTS.get(scenario.name)
    if defaults is None:
        logging.warning(
            f"no StoryDefaults entry for scenario {scenario.name!r}; "
            f"falling back to Bern Hauptgebäude (frames will likely "
            f"render in the wrong place). Add a `_DEFAULTS` entry in "
            f"this file.")
        defaults = _FALLBACK_DEFAULTS

    overrides: dict = {}
    if defaults.extents is not None:
        overrides['extents'] = defaults.extents
    if defaults.dpi is not None:
        overrides['dpi'] = defaults.dpi
    cfg = StoryConfig(
        center_xy=_latlon_to_metric(*defaults.focal_latlon),
        destinations_latlon=list(defaults.destinations_latlon),
        **overrides,
    )
    logging.info(
        f"  scenario={scenario.name!r}, CRS={_CRS_MAIN!r}; "
        f"focal point (lat, lon) {defaults.focal_latlon} → metric "
        f"{cfg.center_xy[0]:.1f}, {cfg.center_xy[1]:.1f}")

    for i, fn in enumerate(_FRAMES, start=1):
        frame_name = fn.__name__
        with step(f'frame {i:02d}/{len(_FRAMES)}: {frame_name}'):
            fig = fn(context, scenario, cfg)
            # `bbox_inches=None` disables `create_results`'s default
            # `tight` cropping — keeps the full 16:9 figure incl. the
            # empty right margin. `pad_inches=0` is belt-and-braces.
            context.create_results(
                fig, f'{cfg.output_subdir}/{frame_name}.png',
                kws={'bbox_inches': None, 'pad_inches': 0,
                     'dpi': cfg.dpi},
            )
            plt.close(fig)

    context.close()


if __name__ == '__main__':
    main()
