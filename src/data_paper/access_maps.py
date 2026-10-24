"""
Accessibility maps for the atlas data descriptor (Data Overview section).

Figure 1 (`access_maps_cost_types.png`), cost types: groceries by walking in the
city of Bern, drawn as H3 cells. Columns: straight-line distance, walking time,
walking utility; rows: nearest-k (mean cost to the `_K` nearest), cumulative
count (within `_COUNT_DIST_M` / `_COUNT_TIME_MIN`, sums of the first released
bins) and gravity (half-decays `_GRAVITY_DIST_M` / `_GRAVITY_TIME_MIN`, utility
β = 1). The settings reach about equally far: half of their weight lies within
about 0.2-0.35 km for groceries in the city. The utility count cell stays empty: counts are not computed for
utility. Own colour scale per panel.

Figure 2 (`access_maps_modes.png`), modes: tertiary-sector jobs, gravity, per
mode (walking, bicycle, public transport, car peak). Rows: Switzerland by time
(cells binned to `_NATIONAL_PIXEL_M` pixels, no frame: the whole country is
shown; the box marks the zoom), Bern by time and Bern by utility (H3 cells).
One colour scale per cost type, shared across modes and, for time, across the
two scales; utilities are in one currency across modes, so their shared scale
compares the modes directly.

Colour scales span the 1st-99th percentile of the plotted values; log for
time / distance gravity spanning a factor of 10 or more, square root for counts
(skewed, with zeros), else linear;
brighter = better access throughout (scales of cost-based panels reversed).
Utility gravity is a logsum (10 stores the log of the sum), hence linear.
Scale bars, bold lowercase panel labels, viridis and a white background, as
Scientific Data asks for figures. Cells are drawn from their H3 ids, so no
geometry file is read.

Inputs (scenario storage, under `<scenario>/`):
    properties/cells_access_{time,util}_{nearest_k,counts,gravity}_<profile>.csv   # 10
    properties/cells_access_dist_{nearest_k,counts,gravity}.csv

Outputs (RESULTS, under `<scenario>/`):
    access_maps_cost_types.png
    access_maps_modes.png

Run:
    python -m data_paper.access_maps --scenario switzerland-h10
"""

import logging

import h3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import Transformer

from aperta_atlas.context import init_context
from aperta_atlas.utils import step

# Grid keys of the release scenario's `accessibility_grids` (scenarios.py).
_GRIDS = {'time': 'time', 'util': 'util', 'dist': 'dist'}
_JOBS = 'employment_tertiary'
_GROCERIES = 'poi_errands_groceries'
_K = 3
# Figure 1 (groceries, walking): settings that reach about equally far, rather than
# equal numbers (a 300-m half-decay reaches much further than a 300-m bin).
_COUNT_DIST_M = 300
_COUNT_TIME_MIN = 10
_COUNT_TIME_BINS = ((0, 5), (5, 10))     # released bins summed to 0-10 min
_GRAVITY_DIST_M = 100
_GRAVITY_TIME_MIN = 2
# Figure 2 (jobs): time half-decay reaching about as far as the logsum (β = 1) for
# cycling, car and public transport (walking utility decays faster).
_JOBS_HALF_DECAY_MIN = 5
_UTIL_DECAY = 'exp1.0'
_WALK = 'rwalk'
_MODES = [('Walking', 'rwalk'), ('Bicycle', 'rbike'),
          ('Public transport', 'transit'), ('Car (peak)', 'car_peak')]

# (xmin, ymin, xmax, ymax), LV95. Figure 2's zoom frames Bern and its
# agglomeration (inside Bern-Mittelland, the TV scenarios' area); Figure 1 the city.
_EXTENT_BERN = (2_588_000, 1_189_500, 2_613_000, 1_209_500)
_EXTENT_BERN_CITY = (2_594_000, 1_194_500, 2_607_000, 1_204_500)
_NATIONAL_PIXEL_M = 500
_PCT_RANGE = (1, 99)          # colour-scale limits (percentiles of plotted values)

_FIG_WIDTH_IN = 6.3          # the article's text width, so fonts print at their nominal size
_ROW_LABEL_IN = 0.35          # left margin taken by the row labels
_FONT_SIZE = 9                # titles and row labels; colour-bar labels −1, ticks and
                              # scale bars −2
_TO_LV95 = Transformer.from_crs('EPSG:4326', 'EPSG:2056', always_xy=True)


def _cmap(reverse: bool = False):
    cmap = mpl.colormaps['viridis_r' if reverse else 'viridis'].copy()
    cmap.set_bad('white')
    return cmap


def _load(context, grid: str, metric: str, profile: str | None,
          cols: str | tuple[str, ...]) -> pd.Series:
    """One accessibility column (or the sum of several, e.g. adjacent count
    bins), indexed by H3 cell id; non-finite dropped."""
    cols = (cols,) if isinstance(cols, str) else cols
    suffix = f'_{profile}' if profile else ''
    rel = f'properties/cells_access_{_GRIDS[grid]}_{metric}{suffix}.csv'
    path = context.path_for(context.default_storage, rel)
    s = pd.read_csv(path, usecols=['cell_id', *cols], index_col='cell_id')[list(cols)].sum(
        axis=1, min_count=len(cols))
    return s[np.isfinite(s)]


def _cell_xy(cell_ids: pd.Index) -> pd.DataFrame:
    lat, lng = zip(*(h3.cell_to_latlng(c) for c in cell_ids))
    x, y = _TO_LV95.transform(lng, lat)
    return pd.DataFrame({'x': x, 'y': y}, index=cell_ids)


def _hexagons(cell_ids: pd.Index) -> list[np.ndarray]:
    polys = []
    for c in cell_ids:
        lat, lng = zip(*h3.cell_to_boundary(c))
        x, y = _TO_LV95.transform(lng, lat)
        polys.append(np.column_stack([x, y]))
    return polys


def _in_extent(xy: pd.DataFrame, extent, margin: float = 500.0) -> pd.Series:
    x0, y0, x1, y1 = extent
    return (xy['x'].between(x0 - margin, x1 + margin)
            & xy['y'].between(y0 - margin, y1 + margin))


def _draw(ax, values: pd.Series, xy: pd.DataFrame, extent, norm, cmap,
          pixel_m: float | None = None, frame: bool = True) -> None:
    """Cells in `extent`: H3 hexagons, or (with `pixel_m`) the mean per pixel;
    a thin black frame unless `frame=False`."""
    xy = xy.loc[values.index]
    keep = _in_extent(xy, extent)
    values, xy = values[keep], xy[keep]
    x0, y0, x1, y1 = extent
    if pixel_m is not None:
        nx, ny = int(np.ceil((x1 - x0) / pixel_m)), int(np.ceil((y1 - y0) / pixel_m))
        ix = np.clip(((xy['x'] - x0) // pixel_m).astype(int), 0, nx - 1)
        iy = np.clip(((xy['y'] - y0) // pixel_m).astype(int), 0, ny - 1)
        flat = (iy * nx + ix).to_numpy()
        sums = np.bincount(flat, weights=values.to_numpy(), minlength=nx * ny)
        counts = np.bincount(flat, minlength=nx * ny)
        with np.errstate(invalid='ignore', divide='ignore'):
            grid = np.where(counts > 0, sums / counts, np.nan).reshape(ny, nx)
        ax.imshow(grid, origin='lower', extent=(x0, x0 + nx * pixel_m, y0, y0 + ny * pixel_m),
                  cmap=cmap, norm=norm, interpolation='nearest')
    else:
        # Outlined in their own colour: closes the anti-aliasing seams (moiré) between
        # hexagons only a few pixels wide.
        colours = cmap(norm(values.to_numpy()))
        ax.add_collection(PolyCollection(_hexagons(values.index), facecolors=colours,
                                         edgecolors=colours, linewidths=0.25, rasterized=True))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(visible=frame, color='black', linewidth=0.6)


def _scale_bar(ax, length_m: float) -> None:
    y0, y1 = ax.get_ylim()
    label = f'{length_m / 1_000:g} km'
    ax.add_artist(AnchoredSizeBar(
        ax.transData, length_m, label, loc='lower right', frameon=True,
        size_vertical=(y1 - y0) * 0.008, pad=0.2, borderpad=0.2, sep=2,
        fontproperties=FontProperties(size=_FONT_SIZE - 2)))
    bar = ax.artists[-1]
    bar.patch.set(facecolor='white', alpha=0.7, edgecolor='none')


def _panel_label(ax, letter: str) -> None:
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha='left', va='top',
            fontweight='bold', fontsize=_FONT_SIZE + 1)


def _row_label(ax, text: str) -> None:
    ax.text(-0.04, 0.5, text, transform=ax.transAxes, ha='right', va='center',
            rotation=90, fontsize=_FONT_SIZE)


def _colorbar(fig, norm, cmap, label: str, **kws) -> None:
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), orientation='horizontal', **kws)
    cb.set_label(label, fontsize=_FONT_SIZE - 1)
    cb.ax.tick_params(labelsize=_FONT_SIZE - 2)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(_tick_label))
    if isinstance(norm, LogNorm):
        wide = norm.vmax > 100 * norm.vmin          # > 2 decades: powers of ten only
        cb.ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,) if wide else (1, 2, 5)))
        cb.ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        cb.ax.xaxis.set_major_locator(MaxNLocator(4))


def _tick_label(v: float, _pos=None) -> str:
    if abs(v) >= 10_000:
        return f'{v / 1_000:,.0f}k'
    return f'{v:,.0f}' if abs(v) >= 1_000 else f'{v:g}'


def _log_norm(values: pd.Series) -> Normalize:
    """Log scale over the plotted range; linear if it spans less than a
    factor of 10 (too few log ticks to read)."""
    pos = values[values > 0]
    lo, hi = np.percentile(pos, _PCT_RANGE)
    return LogNorm(vmin=lo, vmax=hi) if hi >= 10 * lo else Normalize(vmin=lo, vmax=hi)


def _norm(values: pd.Series, scale: str) -> Normalize:
    """`log` (falls back to linear below a factor of 10), `sqrt` (skewed values
    with zeros, e.g. counts) or `linear`, over the 1st-99th percentile."""
    if scale == 'log':
        return _log_norm(values)
    lo, hi = np.percentile(values, _PCT_RANGE)
    return PowerNorm(0.5, vmin=max(lo, 0.0), vmax=hi) if scale == 'sqrt' else Normalize(lo, hi)


def _figure_cost_types(context) -> plt.Figure:
    """3 × 3: rows nearest-k / count / gravity, columns distance / walking time /
    walking utility, groceries in the city of Bern."""
    g = _GROCERIES
    # (grid, metric, profile, columns, factor, colour-bar label, colour scale, reverse?);
    # None = empty
    specs = [
        [('dist', 'nearest_k', None, f'{g}_k{_K}', 1e-3, f'Mean distance to {_K} nearest [km]',
          'linear', True),
         ('time', 'nearest_k', _WALK, f'{g}_k{_K}', 1 / 60, f'Mean time to {_K} nearest [min]',
          'linear', True),
         ('util', 'nearest_k', _WALK, f'{g}_k{_K}', 1.0, f'Mean disutility to {_K} nearest',
          'linear', True)],
        [('dist', 'counts', None, f'{g}_0_{_COUNT_DIST_M}m', 1.0,
          f'Groceries within {_COUNT_DIST_M} m', 'sqrt', False),
         ('time', 'counts', _WALK, tuple(f'{g}_{a}_{b}min' for a, b in _COUNT_TIME_BINS),
          1.0, f'Groceries within {_COUNT_TIME_MIN} min', 'sqrt', False),
         None],
        [('dist', 'gravity', None, f'{g}_exp{_GRAVITY_DIST_M}m', 1.0,
          f'Groceries ({_GRAVITY_DIST_M}-m half-decay)', 'log', False),
         ('time', 'gravity', _WALK, f'{g}_exp{_GRAVITY_TIME_MIN}min', 1.0,
          f'Groceries ({_GRAVITY_TIME_MIN}-min half-decay)', 'log', False),
         ('util', 'gravity', _WALK, f'{g}_{_UTIL_DECAY}', 1.0,
          'Groceries (logsum, β = 1)', 'linear', False)],
    ]
    columns = ['Straight-line distance', 'Time (walking)', 'Utility (walking)']
    rows = [f'Nearest {_K}', 'Cumulative count', 'Gravity']

    loaded = [[_load(context, *sp[:4]) * sp[4] if sp else None for sp in r] for r in specs]
    cells = pd.Index(sorted(set().union(*(s.index for r in loaded for s in r if s is not None))))
    xy = _cell_xy(cells)
    xy = xy[_in_extent(xy, _EXTENT_BERN_CITY)]

    x0, y0, x1, y1 = _EXTENT_BERN_CITY
    panel_w = (_FIG_WIDTH_IN - _ROW_LABEL_IN) / len(columns)
    row_h = panel_w * (y1 - y0) / (x1 - x0) + 0.68   # + title / colour-bar overhead
    fig = plt.figure(figsize=(_FIG_WIDTH_IN, len(rows) * row_h), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.04)
    letters = iter('abcdefghi')
    for i, row in enumerate(specs):
        for j, spec in enumerate(row):
            ax = fig.add_subplot(len(rows), len(columns), i * len(columns) + j + 1)
            if i == 0:
                ax.set_title(columns[j], fontsize=_FONT_SIZE)
            if j == 0:
                _row_label(ax, rows[i])
            _panel_label(ax, next(letters))
            if spec is None:
                _empty_panel(fig, ax, 'Not computed\nfor utility')
                continue
            *_, cb_label, scale, reverse = spec
            values = loaded[i][j]
            values = values[values.index.isin(xy.index)]
            norm, cmap = _norm(values, scale), _cmap(reverse)
            _draw(ax, values, xy, _EXTENT_BERN_CITY, norm, cmap)
            _scale_bar(ax, 2_000)
            _colorbar(fig, norm, cmap, cb_label, ax=ax, fraction=0.05, pad=0.02)
    return fig


def _empty_panel(fig, ax, text: str) -> None:
    """Framed panel with a note; an invisible colour bar keeps it aligned with its row."""
    x0, y0, x1, y1 = _EXTENT_BERN_CITY
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(color='black', linewidth=0.6)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            color='0.45', fontsize=_FONT_SIZE)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=_cmap()), ax=ax,
                      orientation='horizontal', fraction=0.05, pad=0.02)
    cb.ax.set_visible(False)


def _figure_modes(context) -> plt.Figure:
    """3 × 4: rows Switzerland (time), Bern (time), Bern (utility); columns modes;
    tertiary-sector jobs, gravity."""
    time = {label: _load(context, 'time', 'gravity', p, f'{_JOBS}_exp{_JOBS_HALF_DECAY_MIN}min')
            for label, p in _MODES}
    util = {label: _load(context, 'util', 'gravity', p, f'{_JOBS}_{_UTIL_DECAY}')
            for label, p in _MODES}
    cells = pd.Index(sorted(set().union(*(s.index for s in [*time.values(), *util.values()]))))
    xy = _cell_xy(cells)
    bern = _in_extent(xy, _EXTENT_BERN)
    time_norm, time_cmap = _log_norm(pd.concat(time.values())), _cmap()
    util_in_bern = pd.concat(s[s.index.isin(xy.index[bern])] for s in util.values())
    util_norm, util_cmap = _norm(util_in_bern, 'linear'), _cmap()

    pad = 5_000
    national = (xy['x'].min() - pad, xy['y'].min() - pad, xy['x'].max() + pad, xy['y'].max() + pad)
    panel_w = (_FIG_WIDTH_IN - _ROW_LABEL_IN) / len(_MODES)
    h_nat = panel_w * (national[3] - national[1]) / (national[2] - national[0])
    h_bern = panel_w * (_EXTENT_BERN[3] - _EXTENT_BERN[1]) / (_EXTENT_BERN[2] - _EXTENT_BERN[0])
    cb_h = 0.08
    fig = plt.figure(figsize=(_FIG_WIDTH_IN, h_nat + 2 * h_bern + 2 * cb_h + 1.5),
                     layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.03)
    gs = fig.add_gridspec(5, len(_MODES), height_ratios=[h_nat, h_bern, cb_h, h_bern, cb_h])
    letters = iter('abcdefghijkl')
    panel_rows = [
        (0, time, national, _NATIONAL_PIXEL_M, 50_000, 'Switzerland',
         time_norm, time_cmap, False),
        (1, time, _EXTENT_BERN, None, 5_000, 'Bern, time', time_norm, time_cmap, True),
        (3, util, _EXTENT_BERN, None, 5_000, 'Bern, utility', util_norm, util_cmap, True),
    ]
    for gs_row, series, extent, pixel_m, bar_m, row_text, norm, cmap, frame in panel_rows:
        for j, (label, _p) in enumerate(_MODES):
            ax = fig.add_subplot(gs[gs_row, j])
            _draw(ax, series[label], xy, extent, norm, cmap, pixel_m=pixel_m, frame=frame)
            _scale_bar(ax, bar_m)
            _panel_label(ax, next(letters))
            if gs_row == 0:
                ax.set_title(label, fontsize=_FONT_SIZE)
                bx0, by0, bx1, by1 = _EXTENT_BERN
                ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                       edgecolor='#d62728', linewidth=0.8))
            if j == 0:
                _row_label(ax, row_text)
    _colorbar(fig, time_norm, time_cmap, 'Tertiary-sector jobs, weighted by travel time '
              f'({_JOBS_HALF_DECAY_MIN}-min half-decay)', cax=fig.add_subplot(gs[2, 1:3]))
    _colorbar(fig, util_norm, util_cmap, 'Tertiary-sector jobs, logsum (β = 1)',
              cax=fig.add_subplot(gs[4, 1:3]))
    return fig


def main():
    context = init_context()
    plt.rcParams.update({'font.size': _FONT_SIZE, 'font.family': 'sans-serif'})

    with step('figure 1: cost types (groceries, walking, city of Bern)'):
        fig = _figure_cost_types(context)
        context.create_results(fig, 'access_maps_cost_types.png', kws={'dpi': 300})
        plt.close(fig)

    with step('figure 2: modes (tertiary jobs; Switzerland, Bern by time and utility)'):
        fig = _figure_modes(context)
        context.create_results(fig, 'access_maps_modes.png', kws={'dpi': 300})
        plt.close(fig)

    logging.info("  → done")
    context.close()


if __name__ == '__main__':
    main()
