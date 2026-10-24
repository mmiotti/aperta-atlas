"""
Network maps for the atlas data descriptor (Data Records and Technical Validation).

Figure `network_speeds.png` (2 × 2): the calibrated walking, cycling and car
(base hours) networks in the city of Bern, each street coloured by its effective
speed (length over calibrated duration, both directions together: a street's two
directions differ only on slopes and by congestion, and drawing both would hide
one), and the car network's speed loss in peak hours relative to base hours. The
durations include the calibrated junction delays, so slow streets also mark busy
junctions. Own colour scale per panel, over the 2nd-98th percentile of the drawn
streets (car speeds: 2nd-99.5th, to include the motorways; log).

Figure `network_tlmregio.png`: the car network of the release (OpenStreetMap) next
to the swissTLMRegio main-road network of the simplified-network test
(`network-tlmregio`), around Bern; motorways and expressways highlighted.

Scale bars, bold lowercase panel labels and a white background, as in
`access_maps.py`.

Inputs (scenario storage, under `<scenario>/`):
    shapes/edges_{walk,bike,car}.gpkg
    properties/edges_{walk,bike,car}_calibrated.csv         # 04
    properties/edges_car_core.csv                           # 02a (highway)
and (PUBLIC, preparation/switzerland/tlmregio/):
    nw/<area>_car_consolidated.graphml + properties/edges_<area>_car_consolidated.csv

Outputs (RESULTS, under `<scenario>/`):
    network_speeds.png
    network_tlmregio.png

Run:
    python -m data_paper.network_maps --scenario switzerland-h10
"""

import logging

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from scenarios import get_scenario
from data_paper.access_maps import (
    _FIG_WIDTH_IN, _FONT_SIZE, _colorbar, _panel_label, _scale_bar)

# (xmin, ymin, xmax, ymax), LV95. Speeds: the city of Bern with the motorways on its
# northern, eastern and western edges. swissTLMRegio: Bern and its surroundings.
_EXTENT_SPEEDS = (2_596_000, 1_197_000, 2_604_000, 1_203_000)
_EXTENT_TLMREGIO = (2_588_000, 1_189_500, 2_613_000, 1_209_500)
# (title, mode, profile) of the speed panels; the loss panel compares the car
# profiles in `_LOSS_PROFILES` (reference, compared).
_SPEED_PANELS = [('Walking', 'walk', 'rwalk'), ('Bicycle', 'bike', 'rbike'),
                 ('Car (base hours)', 'car', 'car_base')]
_LOSS_PROFILES = ('car_base', 'car_peak')
# Colour-scale limits (percentiles of the drawn streets); car speeds up to the
# motorways, which are few in the frame.
_PCT_RANGE = {'walk': (2, 98), 'bike': (2, 98), 'car': (2, 99.5), 'loss': (2, 98)}
_LINE_WIDTH = {'walk': 0.45, 'bike': 0.45, 'car': 0.6}
_MOTORWAY = ('motorway', 'motorway_link', 'trunk', 'trunk_link')


def _cmap(name: str = 'viridis', lo: float = 0.0, hi: float = 0.88):
    # without the palest end, which vanishes on white
    return mpl.colors.ListedColormap(mpl.colormaps[name](np.linspace(lo, hi, 256)))


def _frame(ax, extent) -> None:
    x0, y0, x1, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(color='black', linewidth=0.6)


def _segments(geoms: gpd.GeoSeries) -> list[np.ndarray]:
    return [np.asarray(g.coords)[:, :2] for g in geoms]


def _street_speeds(context, mode: str, profiles: tuple[str, ...],
                   extent) -> gpd.GeoDataFrame:
    """Streets in `extent` with the round-trip effective speed (km/h) of each profile,
    in columns named after the profiles."""
    x0, y0, x1, y1 = extent
    edges = gpd.read_file(context.path_for(context.default_storage, f'shapes/edges_{mode}.gpkg'),
                          bbox=(x0, y0, x1, y1)).set_index('edge_id')
    cols = [f'duration_calibrated_{p}' for p in profiles]
    cal = pd.read_csv(context.path_for(context.default_storage,
                                       f'properties/edges_{mode}_calibrated.csv'),
                      usecols=['edge_id', *cols], index_col='edge_id')
    edges = edges.join(cal, how='inner')
    edges = edges[np.isfinite(edges[cols]).all(axis=1)]
    edges['length'] = edges.geometry.length
    uvk = edges.index.str.split(':', expand=True).to_frame(index=False).to_numpy()
    lo, hi = np.minimum(uvk[:, 0], uvk[:, 1]), np.maximum(uvk[:, 0], uvk[:, 1])
    edges['street'] = [f'{a}:{b}:{k}' for a, b, k in zip(lo, hi, uvk[:, 2])]
    grouped = edges.groupby('street')
    streets = grouped.first()[['geometry']]
    length = grouped['length'].sum()
    for p, col in zip(profiles, cols):
        streets[p] = 3.6 * length / grouped[col].sum()
    return gpd.GeoDataFrame(streets, geometry='geometry', crs=edges.crs)


def _norm(values: np.ndarray, key: str, log: bool) -> Normalize:
    lo, hi = np.percentile(values, _PCT_RANGE[key])
    return LogNorm(vmin=lo, vmax=hi) if log and hi >= 5 * lo else Normalize(vmin=lo, vmax=hi)


def _draw_streets(ax, streets: gpd.GeoDataFrame, values: np.ndarray, norm, cmap,
                  width: float) -> None:
    order = np.argsort(values)   # largest values on top
    ax.add_collection(LineCollection(
        _segments(streets.geometry.iloc[order]), colors=cmap(norm(values[order])),
        linewidths=width, rasterized=True))
    _frame(ax, _EXTENT_SPEEDS)


def _figure_speeds(context) -> plt.Figure:
    """2 × 2 panels in inches: title, map, colour bar (70 % of the map width) and its
    labels per row."""
    x0, y0, x1, y1 = _EXTENT_SPEEDS
    margin, gap = 0.03, 0.14
    panel_w = (_FIG_WIDTH_IN - 2 * margin - gap) / 2
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    title_h, cbar_gap, cbar_h, cbar_lab = 0.22, 0.07, 0.08, 0.42
    row_h = title_h + panel_h + cbar_gap + cbar_h + cbar_lab
    fig_w, fig_h = _FIG_WIDTH_IN, 2 * row_h
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes(i: int):
        row, col = divmod(i, 2)
        left = margin + col * (panel_w + gap)
        top = fig_h - row * row_h - title_h
        ax = fig.add_axes((left / fig_w, (top - panel_h) / fig_h, panel_w / fig_w,
                           panel_h / fig_h))
        cbar_bottom = top - panel_h - cbar_gap - cbar_h
        cax = fig.add_axes(((left + 0.15 * panel_w) / fig_w, cbar_bottom / fig_h,
                            0.7 * panel_w / fig_w, cbar_h / fig_h))
        return ax, cax

    cmap = _cmap()
    car = None
    for i, (title, mode, profile) in enumerate(_SPEED_PANELS):
        with step(f'{mode}: streets + speeds'):
            profiles = (tuple(dict.fromkeys((profile, *_LOSS_PROFILES))) if mode == 'car'
                        else (profile,))
            streets = _street_speeds(context, mode, profiles, _EXTENT_SPEEDS)
            if mode == 'car':
                car = streets
            speeds = streets[profile].to_numpy()
            logging.info(f"  → {len(streets):,} streets; speed (km/h) percentiles 2/50/98: "
                         + ', '.join(f'{v:.1f}' for v in np.percentile(speeds, (2, 50, 98))))
        norm = _norm(speeds, mode, log=True)
        ax, cax = _axes(i)
        _draw_streets(ax, streets, speeds, norm, cmap, _LINE_WIDTH[mode])
        ax.set_title(title, fontsize=_FONT_SIZE, pad=3)
        _panel_label(ax, 'abcd'[i])
        _scale_bar(ax, 2_000)
        _colorbar(fig, norm, cmap, 'Effective speed (km/h)', cax=cax)

    ref, cmp = _LOSS_PROFILES
    loss = 100 * (1 - car[cmp] / car[ref]).to_numpy()
    logging.info("  → speed loss (%), percentiles 2/50/98: "
                 + ', '.join(f'{v:.1f}' for v in np.percentile(loss, (2, 50, 98))))
    loss_cmap = _cmap('YlOrRd', 0.2, 1.0)
    norm = _norm(loss, 'loss', log=False)
    ax, cax = _axes(3)
    _draw_streets(ax, car, loss, norm, loss_cmap, _LINE_WIDTH['car'])
    ax.set_title('Car: peak vs base hours', fontsize=_FONT_SIZE, pad=3)
    _panel_label(ax, 'd')
    _scale_bar(ax, 2_000)
    _colorbar(fig, norm, loss_cmap, 'Speed loss in peak hours (%)', cax=cax)
    return fig


def _tlmregio_edges(context, area: str, extent) -> gpd.GeoDataFrame:
    name = f'{area}_car_consolidated'
    g = context.source('preparation/switzerland/tlmregio', storage=Storage.PUBLIC).get_nw(
        name, add_edge_properties=name)
    edges = ox.graph_to_gdfs(g, nodes=False).to_crs(2056)
    edges = edges.cx[extent[0]:extent[2], extent[1]:extent[3]]
    return edges.drop_duplicates(subset=['geometry'])


def _osm_car_edges(context, extent) -> gpd.GeoDataFrame:
    x0, y0, x1, y1 = extent
    edges = gpd.read_file(context.path_for(context.default_storage, 'shapes/edges_car.gpkg'),
                          bbox=(x0, y0, x1, y1)).set_index('edge_id')
    core = pd.read_csv(context.path_for(context.default_storage, 'properties/edges_car_core.csv'),
                       usecols=['edge_id', 'highway'], index_col='edge_id')
    edges = edges.join(core, how='left')
    edges['highway'] = edges['highway'].astype(str).str.strip("[]'").str.split("'").str[0]
    return edges


def _draw_roads(ax, edges: gpd.GeoDataFrame, extent, km_label: str) -> None:
    motorway = edges['highway'].isin(_MOTORWAY)
    ax.add_collection(LineCollection(_segments(edges.geometry[~motorway]), colors='0.25',
                                     linewidths=0.25, rasterized=True))
    ax.add_collection(LineCollection(_segments(edges.geometry[motorway]),
                                     colors=mpl.colormaps['viridis'](0.25),
                                     linewidths=0.9, rasterized=True))
    _frame(ax, extent)
    ax.text(0.99, 0.99, km_label, transform=ax.transAxes, ha='right', va='top',
            fontsize=_FONT_SIZE - 2, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                                               pad=1.5))


def _clipped_km(edges: gpd.GeoDataFrame, extent) -> float:
    """Road length (km) inside `extent`, counting each road once (the two directions
    of a road share their geometry)."""
    from shapely.geometry import box
    once = edges.drop_duplicates(subset=['geometry'])
    reverse = once.geometry.apply(lambda g: g.reverse().wkb)
    seen, keep = set(), []
    for wkb, rev in zip(once.geometry.apply(lambda g: g.wkb), reverse):
        keep.append(rev not in seen)
        seen.add(wkb)
    return float(once[keep].geometry.intersection(box(*extent)).length.sum() / 1_000)


def _figure_tlmregio(context) -> plt.Figure:
    scenario = get_scenario(context.scenario)
    x0, y0, x1, y1 = _EXTENT_TLMREGIO
    panel_w = _FIG_WIDTH_IN / 2
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    fig, axes = plt.subplots(1, 2, figsize=(_FIG_WIDTH_IN, panel_h + 0.3),
                             gridspec_kw=dict(wspace=0.03, left=0.005, right=0.995,
                                              top=1 - 0.25 / (panel_h + 0.3), bottom=0.005))
    with step('OpenStreetMap car network'):
        osm = _osm_car_edges(context, _EXTENT_TLMREGIO)
    with step(f'swissTLMRegio ({scenario.area_name})'):
        tlm = _tlmregio_edges(context, scenario.area_name, _EXTENT_TLMREGIO)
    for ax, (title, edges), letter in zip(
            axes, [('OpenStreetMap (released car network)', osm),
                   ('swissTLMRegio (main roads only)', tlm)], 'ab'):
        km = _clipped_km(edges, _EXTENT_TLMREGIO)
        logging.info(f"  → {title}: {len(edges):,} edges, {km:,.0f} km of road in the frame")
        _draw_roads(ax, edges, _EXTENT_TLMREGIO, f'{km:,.0f} km of road')
        ax.set_title(title, fontsize=_FONT_SIZE, pad=3)
        _panel_label(ax, letter)
        _scale_bar(ax, 5_000)
    return fig


def main():
    context = init_context()
    plt.rcParams.update({'font.size': _FONT_SIZE, 'font.family': 'sans-serif'})
    for name, build in (('network_speeds.png', _figure_speeds),
                        ('network_tlmregio.png', _figure_tlmregio)):
        with step(name):
            fig = build(context)
            context.create_results(fig, name, kws={'dpi': 300})
            plt.close(fig)
    context.close()


if __name__ == '__main__':
    main()
