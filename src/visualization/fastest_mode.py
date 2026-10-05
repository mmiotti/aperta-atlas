"""
Per-cell fastest-mode comparison plot for the Swiss Urban Mobility Atlas.

For each of 8 destination types, ranks the four modes (walk, bike, car,
transit) by which one "wins" on accessibility at each AOI cell, then
plots a stacked 100 % bar chart showing the mode-share of winners
per destination type.

Two figures are produced, both for the "regular bicycle / peak hours"
combination (= car uses `car_peak` profile, bike uses `bike` profile):

  1. **Nearest-k** — for each destination, pick the per-destination
     `k` from `_DEST_TYPES` (cf. project-paper guidance: e.g. groceries
     k=3, schools k=1, jobs k=100 with ×0.01 weight scaling). Winner =
     mode with the LOWEST nearest-k mean travel time. NaN (k
     unreachable) doesn't vote.

  2. **Gravity** — mode-specific exponential decay matched to a typical
     median trip duration:
       walk    → exp10min    (β = ln 2 / 08 min)
       bike    → exp20min
       car     → exp20min
       transit → exp30min
     Winner = mode with the HIGHEST gravity (Σ_j w_j · exp(-β·t_ij)).
     Cells with zero across all modes (rare; nothing reachable from
     this cell) don't vote.

The denominator at each destination is "AOI cells with at least one
mode reachable" — percentages sum to 100 %. The all-modes-unreachable
share is logged but not plotted.

Inputs (PUBLIC, under `<scenario>/`):
    properties/cells_access_<grid_key>_nearest_k_rwalk.csv        # from 10
    properties/cells_access_<grid_key>_nearest_k_rbike.csv        # from 10
    properties/cells_access_<grid_key>_nearest_k_car_peak.csv     # from 10
    properties/cells_access_<grid_key>_nearest_k_transit.csv      # from 10
    properties/cells_access_<grid_key>_gravity_rwalk.csv          # from 10
    properties/cells_access_<grid_key>_gravity_rbike.csv          # from 10
    properties/cells_access_<grid_key>_gravity_car_peak.csv       # from 10
    properties/cells_access_<grid_key>_gravity_transit.csv        # from 10

    `<grid_key>` = the value of `--variant` (default: `'default'`).

Outputs (RESULTS, under `<scenario>/`):
    fastest_mode_nearest_k.png
    fastest_mode_gravity.png

Run:
    python -m visualization.fastest_mode --scenario <name>
"""

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from aperta_atlas.context import init_context
from aperta_atlas.utils import step


# Stacked from bottom to top in the bar chart; ordering matches the
# reference figure (walking at the bottom, transit at the top).
@dataclass(frozen=True)
class ModeConfig:
    label: str            # display label
    profile: str          # data_name suffix used in access_*_<profile>.csv
    color: str
    gravity_decay: str    # decay-name suffix in gravity output (exp10min etc.)


# The three non-bike modes are constant across the figure's three
# subplots — only the bike-family entry varies (regular / ebike25 /
# ebike45). All three bike variants share the same display label
# 'Bicycle' + colour so the legend stays a single 4-mode block; the
# subplot title is what tells the reader which bike variant is in play.
_WALK = ModeConfig('Walking', 'walk',     '#1a2540', 'exp10min')
_CAR  = ModeConfig('Car',     'car_peak', '#e02b3a', 'exp20min')
_TX   = ModeConfig('Transit', 'transit',  '#7fc23a', 'exp30min')
_BIKE_COLOR = '#4a7be0'


# Per-subplot bike-family variant: (subplot title, profile, gravity decay).
# `gravity_decay` could diverge per ebike variant in principle (e.g. a
# faster mode has a longer median trip and might warrant a slower
# decay), but the three bike variants all do similar 08-25 min trips
# in practice, so we keep exp20min across the family.
_BIKE_VARIANTS: list[tuple[str, str, str]] = [
    ('Reg. bicycle / peak hours', 'bike',    'exp20min'),
    ('E-bike 25 / peak hours',    'ebike25', 'exp20min'),
    ('E-bike 45 / peak hours',    'ebike45', 'exp20min'),
]


def _modes_for_variant(
    bike_profile: str, bike_gravity_decay: str,
) -> list[ModeConfig]:
    """Four-mode stack for one subplot: walking, the chosen bike
    variant, car, transit."""
    return [
        _WALK,
        ModeConfig('Bicycle', bike_profile, _BIKE_COLOR, bike_gravity_decay),
        _CAR,
        _TX,
    ]


@dataclass(frozen=True)
class DestType:
    """One destination type for the figure.

    `label`       — display label on the x-axis.
    `column_root` — column-name root in the access CSVs (without the
                    `_kN` / `_<decay>` suffix). For POIs this is
                    e.g. `poi_errands_groceries`; for employment
                    `employment_secondary` / `employment_tertiary`.
    `k`           — nearest-k value to use for this destination.
                    Pop/emp values are interpreted as "×0.01 scaled" —
                    i.e. `employment_tertiary` at k=100 means "average
                    time to nearest 08,000 jobs".
    """
    label: str
    column_root: str
    k: int


_DEST_TYPES: list[DestType] = [
    DestType('Errands: groceries',   'poi_errands_groceries',   k=3),
    DestType('Errands: services',    'poi_errands_services',    k=30),
    DestType('Leisure: gastronomy',  'poi_leisure_gastronomy',  k=10),
    DestType('Leisure: hiking',      'poi_leisure_hiking',      k=30),
    DestType('Education: school',    'poi_education_school',    k=1),
    DestType('Education: higher',    'poi_education_higher',    k=1),
    DestType('Employment: secondary', 'employment_secondary',   k=100),
    DestType('Employment: tertiary',  'employment_tertiary',    k=100),
]


def _load_per_mode(
    context, grid_key: str, metric: str, modes: list[ModeConfig],
) -> dict[str, pd.DataFrame]:
    """Load `cells_access_<grid_key>_<metric>_<profile>.csv` for each
    mode. `metric` is `'nearest_k'` or `'gravity'`. Returns
    `{mode_label: DataFrame indexed by cell_id}`. Indexes are aligned
    by 10 (both reindex to the canonical AOI cell set)."""
    out: dict[str, pd.DataFrame] = {}
    for mode in modes:
        df = context.get_properties(
            'cells', f'access_{grid_key}_{metric}_{mode.profile}')
        out[mode.label] = df
    return out


def _stack_mode_values(
    per_mode: dict[str, pd.DataFrame], column_per_mode: dict[str, str],
) -> pd.DataFrame:
    """Pull `column_per_mode[m]` from each mode's frame; stack as a
    DataFrame indexed by cell_id with one column per mode."""
    out = pd.DataFrame({
        m: per_mode[m][col] for m, col in column_per_mode.items()
    })
    return out


def _winners_share(
    winners: pd.Series, mode_labels: list[str],
    weights: pd.Series | None = None,
) -> pd.Series:
    """% share of each mode among non-NaN winners. Reindexes to
    `mode_labels` (fills 0 for modes that never won).

    `weights` (optional): per-cell weight (e.g. `population_total`).
    When provided, the share is weighted — the answer becomes "share
    of the population whose cell's fastest mode is X" instead of
    "share of cells where mode is X". `weights` is aligned to
    `winners.index`; cells with NaN winner don't contribute.
    """
    voting = winners.dropna()
    if weights is None:
        counts = voting.value_counts(normalize=True) * 100
    else:
        w = weights.reindex(voting.index).fillna(0.0)
        per_mode = w.groupby(voting).sum()
        total = float(w.sum())
        counts = (per_mode / total * 100) if total > 0 else per_mode * 0.0
    return counts.reindex(mode_labels, fill_value=0.0)


def _safe_idxmin(values: pd.DataFrame) -> pd.Series:
    """`idxmin(axis=1)` with all-NaN rows mapped to NaN. Newer pandas
    raises 'Encountered all NA values' under `skipna=True` instead of
    returning NaN for those rows; drop them first and re-pad."""
    valid = ~values.isna().all(axis=1)
    if not valid.any():
        return pd.Series(pd.NA, index=values.index, dtype=object)
    return values.loc[valid].idxmin(axis=1).reindex(values.index)


def _safe_idxmax(values: pd.DataFrame) -> pd.Series:
    """Mirror of `_safe_idxmin` for the max-wins case."""
    valid = ~values.isna().all(axis=1)
    if not valid.any():
        return pd.Series(pd.NA, index=values.index, dtype=object)
    return values.loc[valid].idxmax(axis=1).reindex(values.index)


def compute_nearest_k_winners(
    context, grid_key: str, modes: list[ModeConfig], dests: list[DestType],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Winner = mode with MIN nearest-k mean time per cell.
    Returns DataFrame[destination, mode] = % winners.

    `weights`: optional per-cell weight series (e.g. `population_total`).
    Without weights, each cell votes once. With weights, each cell's
    vote is weighted (e.g. by residents)."""
    weight_tag = ' (pop-weighted)' if weights is not None else ''
    with step(f'nearest-k: load per-mode CSVs (grid={grid_key}){weight_tag}'):
        per_mode = _load_per_mode(context, grid_key, 'nearest_k', modes)

    rows: dict[str, pd.Series] = {}
    mode_labels = [m.label for m in modes]
    for dest in dests:
        column_per_mode = {m.label: f'{dest.column_root}_k{dest.k}'
                           for m in modes}
        values = _stack_mode_values(per_mode, column_per_mode)
        # All-NaN rows simply don't vote (no reachable mode); newer
        # pandas raises on those, hence the `_safe_idxmin` wrapper.
        winners = _safe_idxmin(values)
        n_total = len(values)
        n_voting = int(winners.notna().sum())
        rows[dest.label] = _winners_share(winners, mode_labels, weights)
        logging.info(
            f"  → {dest.label:<22s} (k={dest.k:>3d}): "
            f"{n_voting:>6,d}/{n_total:>6,d} cells voted "
            f"({100*(n_total-n_voting)/max(n_total,1):.1f} % "
            "all-modes-unreachable)")
    return pd.DataFrame(rows).T  # rows = dest label, cols = mode label


def compute_gravity_winners(
    context, grid_key: str, modes: list[ModeConfig], dests: list[DestType],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Winner = mode with MAX gravity (mode-specific decay).
    `weights`: see `compute_nearest_k_winners`."""
    weight_tag = ' (pop-weighted)' if weights is not None else ''
    with step(f'gravity: load per-mode CSVs (grid={grid_key}){weight_tag}'):
        per_mode = _load_per_mode(context, grid_key, 'gravity', modes)

    rows: dict[str, pd.Series] = {}
    mode_labels = [m.label for m in modes]
    for dest in dests:
        column_per_mode = {
            m.label: f'{dest.column_root}_{m.gravity_decay}'
            for m in modes
        }
        values = _stack_mode_values(per_mode, column_per_mode)
        # NaN-fill values from the 10 sentinel (-1) and 0 are both
        # "unreachable". Treat any non-positive max as "no winner" so
        # ties at 0 don't always credit the first mode in the columns.
        max_val = values.max(axis=1)
        winners = _safe_idxmax(values).where(max_val > 0)
        n_total = len(values)
        n_voting = int(winners.notna().sum())
        rows[dest.label] = _winners_share(winners, mode_labels, weights)
        logging.info(
            f"  → {dest.label:<22s}: "
            f"{n_voting:>6,d}/{n_total:>6,d} cells voted "
            f"({100*(n_total-n_voting)/max(n_total,1):.1f} % "
            "no-mode-reachable)")
    return pd.DataFrame(rows).T


def plot_winners_on_ax(
    ax: Axes, df: pd.DataFrame, modes: list[ModeConfig], subtitle: str,
) -> None:
    """Draw one stacked-bar subplot on `ax`. The figure-level legend +
    suptitle are added by the caller."""
    bottom = pd.Series(0.0, index=df.index)
    for mode in modes:
        vals = df[mode.label]
        ax.bar(df.index, vals, bottom=bottom, color=mode.color,
               label=mode.label, edgecolor='black', linewidth=0.3)
        bottom = bottom + vals
    ax.set_ylim(0, 100)
    ax.set_title(subtitle, fontsize=10)
    ax.tick_params(axis='x', rotation=90)
    ax.spines[['top', 'right']].set_visible(False)


def _compute_winners_for_variant(
    context, grid_key: str, bike_profile: str, bike_decay: str,
    metric: str, dests: list[DestType],
    weights: pd.Series | None,
) -> tuple[pd.DataFrame, list[ModeConfig]]:
    """Compute the winners DataFrame for one subplot (one bike variant
    × one metric type). Returns `(df, modes_used)`."""
    modes = _modes_for_variant(bike_profile, bike_decay)
    if metric == 'nearest_k':
        df = compute_nearest_k_winners(
            context, grid_key, modes, dests, weights=weights)
    elif metric == 'gravity':
        df = compute_gravity_winners(
            context, grid_key, modes, dests, weights=weights)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return df, modes


def plot_three_variants_figure(
    context, grid_key: str, metric: str, weights: pd.Series | None,
    weighting_qualifier: str,
) -> plt.Figure:
    """Build a 1×3 figure: three subplots, one per bike-family variant.
    Shares y-axis; figure-level legend on the right; suptitle indicates
    the metric + weighting."""
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.2), sharey=True)
    last_modes: list[ModeConfig] = []
    for ax, (subtitle, bike_profile, bike_decay) in zip(axes, _BIKE_VARIANTS):
        logging.info(f"  → subplot: {subtitle} ({metric}, grid={grid_key})")
        df, modes = _compute_winners_for_variant(
            context, grid_key, bike_profile, bike_decay, metric, _DEST_TYPES,
            weights)
        plot_winners_on_ax(ax, df, modes, subtitle)
        last_modes = modes
    axes[0].set_ylabel('% of cases where mode is fastest')

    # Shared legend on the right — reverse so top-of-stack
    # (Transit) is at the top of the legend.
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1],
               loc='center left', bbox_to_anchor=(0.99, 0.5),
               frameon=False)

    metric_title = ('nearest-k' if metric == 'nearest_k'
                    else 'gravity (mode-specific β)')
    suptitle = f'Fastest mode — {metric_title}{weighting_qualifier}'
    fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    # Avoid the "tight_layout: variable `last_modes` not used" hint
    # — kept for callers that want to introspect what was plotted.
    del last_modes
    return fig


def main():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--scenario', default=None,
                   help='(consumed by init_context)')
    p.add_argument('--variant', default='default',
                   help="Grid key from `scenario.accessibility_grids` — "
                        "picks which grid's `cells_access_<grid>_*` "
                        "files to read (default: 'default').")
    args, _ = p.parse_known_args()
    grid_key = args.variant

    context = init_context()

    with step('load residential population per cell (for pop-weighted variant)'):
        pop = context.get_properties('cells', 'population')['population_total']
        logging.info(
            f"  → loaded population_total for {len(pop):,} cells; "
            f"sum = {pop.sum():,.0f}")

    # Cell-equally-weighted + population-weighted, both metrics.
    weighting_variants: list[tuple[str, pd.Series | None, str]] = [
        ('per_cell', None, ''),
        ('pop',      pop,  ' (population-weighted)'),
    ]

    for variant_tag, weights, qualifier in weighting_variants:
        suffix = '' if variant_tag == 'per_cell' else '_pop'

        for metric in ('nearest_k', 'gravity'):
            with step(f'{metric} / {variant_tag} / grid={grid_key}: compute 3 subplots + save'):
                fig = plot_three_variants_figure(
                    context, grid_key, metric, weights, qualifier)
                context.create_results(
                    fig, f'fastest_mode_{grid_key}_{metric}{suffix}.png')
                plt.close(fig)

    context.close()


if __name__ == '__main__':
    main()
