"""
Calibrate `overheads_transit` at TRIP level (α=1 constrained fits).

Every spec in `_TRANSIT_SPECS` is fit as an independent constrained OLS
that regresses `(trip_time_measured − raw_z2z)` on Σ(orig + dest) of
its endpoint `*_zone_dev` features + intercept. Trip set: survey trips
containing ≥ 1 transit leg (aggregated via
`main.common.aggregate_transit_trips`). α is fixed at 1 (no z2z
scaling) because the applied model (`main.common.per_cell_transit_overhead`)
adds per-cell overheads to raw z2z without scaling.

Output `overheads_transit.csv` has:
  * rows: `cap_s`, `const_s`, and one row per endpoint feature used in
    any spec (NaN in spec columns that don't use that feature).
  * columns: one per spec (e.g. `only_walk`, `only_bike`, `both`) plus
    `transit` — the production column, aliased from `_PRODUCTION_SPEC`.
    `per_cell_transit_overhead` reads the `transit` column; the other
    columns are for inspection / to swap the production choice.

Fit is INDEPENDENT of the current `overheads_transit` values and of
`survey/08b_trip_transit_times.py`'s per-trip predictions (uses raw
NPVM z2z at trip endpoints).

**Trip-level** (not leg-level) because NPVM z2z is door-to-door while
a transit LEG's `time_measured` is only the stop-to-stop ride segment.
Fitting leg-level was flawed (2026-09 finding) — the y-side was
stop-to-stop but the X-side (baseline + endpoint zone-dev at leg
endpoints) was door-to-door and wrong-cell for overhead purposes.
This script aggregates legs to trips itself (via
`main.common.aggregate_transit_trips`, shared with survey/08b) and
does its own NPVM z2z lookup at trip endpoints.

OLS shape per spec::

    trip_time_measured = const + α · raw_z2z(trip_orig_zone, trip_dest_zone)
                              + Σ β_i · (feature_i(trip_orig_cell)
                                       + feature_i(trip_dest_cell))

with `share_endpoint_coefs=True` — one β per feature, applied to the
sum of orig + dest cell values. Physically: the marginal cost of a
walk-to-transit deviation doesn't depend on which end you're on.

Inputs (PRIVATE, under <scenario>/):
    generic/survey_legs.csv               # from survey/02d (needs trip_id)

Inputs (PUBLIC, under <scenario>/):
    properties/cells_transit_access.csv   # 06 — *_zone_dev columns

Inputs (external): NPVM z2z table via `main.common.npvm_transit_z2z_lookup`.

Outputs (PUBLIC, under <scenario>/):
    coefs/<kind>/overheads_transit.csv                    # calibrated coefs
    results/transit/scatter_z2z_vs_measured.png           # 1:1 vs measured

Run:
    python -m main.07b_transit_overhead_coefs --scenario <name>
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from aperta_atlas import coefs
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from main.common import aggregate_transit_trips, npvm_transit_z2z_lookup
from scenarios import get_scenario, scenario_needs_survey_prep


# Trip-line-distance filter (metres). Implausibly short or long trips
# get excluded from the fit.
_TRIP_DIST_LINE_MIN = 500.0
_TRIP_DIST_LINE_MAX = 200_000.0

# Endpoint features probed at trip level. Each `_TRANSIT_SPECS` entry is
# a distinct α=1 constrained OLS on (measured − raw_z2z) with Σ(orig +
# dest) of the listed features + intercept, written as its own column in
# `overheads_transit.csv`. `_PRODUCTION_SPEC` names the column whose
# coefs the consumer (`per_cell_transit_overhead`) reads via the `transit`
# column (aliased into `transit` for backward compatibility).
_TRANSIT_SPECS: dict[str, tuple[str, ...]] = {
    'only_walk': ('t_walk_to_transit_nearest_zone_dev',),
    'only_bike': ('t_bike_to_train_nearest_zone_dev',),
    'both':      ('t_walk_to_transit_nearest_zone_dev',
                  't_bike_to_train_nearest_zone_dev'),
}
_PRODUCTION_SPEC: str = 'both__uncon'

# Symmetric clipping bound (seconds) applied to each cell's overhead
# (± cap_s). Chosen to accommodate typical fitted contributions
# (const_s/2 ~ 50-100 s + endpoint β × dev ~ few-hundred s) plus a
# margin. Trip-level total clipping envelope is ±2 × cap_s.
_TRANSIT_OVERHEAD_CAP_S: float = 900.0


def _fit_constrained(
    df: pd.DataFrame, feature_cols: tuple[str, ...], spec_name: str,
) -> pd.Series:
    """α=1 constrained OLS: regress (measured − raw_z2z) on shared endpoint
    features + intercept. Returns coef Series (const + one β per feature)."""
    df = df.dropna(subset=['target', 'z2z', *feature_cols])
    if len(df) < 100:
        raise RuntimeError(
            f"spec={spec_name!r} (constrained): only {len(df):,} rows after "
            f"dropna — insufficient for calibration.")
    X = sm.add_constant(df[list(feature_cols)], has_constant='add')
    y = df['target']
    result = sm.OLS(y, X).fit()
    logging.info(
        f"  → spec={spec_name!r} (α=1 constrained): "
        f"R² = {result.rsquared:.3f}, n = {int(result.nobs):,}, "
        f"resid std = {result.resid.std():.1f} s; "
        f"mean(y−z2z) = {y.mean():.1f} s")
    for name, coef in result.params.items():
        logging.info(f"     {str(name):>34s}: {coef:+.4f}")
    return result.params


def _fit_unconstrained(
    df: pd.DataFrame, feature_cols: tuple[str, ...], spec_name: str,
) -> pd.Series:
    """Unconstrained OLS: regress measured trip time on raw z2z + shared
    endpoint features + intercept. Returns coef Series (const + t_routed +
    one β per feature). `t_routed` is the α (z2z slope) — informational
    only; the applied model uses α = 1."""
    df = df.dropna(subset=['y', 'z2z', *feature_cols])
    if len(df) < 100:
        raise RuntimeError(
            f"spec={spec_name!r} (unconstrained): only {len(df):,} rows after "
            f"dropna — insufficient for calibration.")
    X = df[['z2z', *feature_cols]].rename(columns={'z2z': 't_routed'})
    X = sm.add_constant(X, has_constant='add')
    y = df['y']
    result = sm.OLS(y, X).fit()
    logging.info(
        f"  → spec={spec_name!r} (unconstrained, informational): "
        f"R² = {result.rsquared:.3f}, n = {int(result.nobs):,}, "
        f"resid std = {result.resid.std():.1f} s; "
        f"mean y = {y.mean():.1f} s, mean z2z = {df['z2z'].mean():.1f} s")
    for name, coef in result.params.items():
        logging.info(f"     {str(name):>34s}: {coef:+.4f}")
    return result.params


def _fit_calibrated_transit_overhead(
    trips: pd.DataFrame, cells_transit: pd.DataFrame,
) -> pd.DataFrame:
    """Fit every spec in `_TRANSIT_SPECS` in BOTH the α=1 constrained
    (production) and unconstrained (informational) form, and pack all
    coefs into one DataFrame:

        rows    : cap_s, const_s, t_routed (α), and one row per endpoint
                  feature. NaN where a coef doesn't apply to that column.
        columns : `<spec>` (constrained fit, applied schema) and
                  `<spec>__uncon` (unconstrained fit with α, informational)
                  for each spec in `_TRANSIT_SPECS`, plus `transit` —
                  the production column, aliased from `_PRODUCTION_SPEC`.

    `per_cell_transit_overhead` reads only the `transit` column and skips
    the `t_routed` row (α=1 in the applied model). The `_uncon` columns
    are there to inspect the fitted slope + how much the constraint costs
    us in R² / bias.
    """
    all_feature_cols = sorted({c for cols in _TRANSIT_SPECS.values() for c in cols})
    for col in all_feature_cols:
        if col not in cells_transit.columns:
            raise ValueError(
                f"Transit overhead spec references {col!r}, missing from "
                f"cells_transit_access.csv (available: "
                f"{sorted(cells_transit.columns)})")

    df = pd.DataFrame({
        'y':      trips['trip_time_measured'].astype(float),
        'z2z':    trips['raw_z2z'].astype(float),
        'target': (trips['trip_time_measured'].astype(float)
                   - trips['raw_z2z'].astype(float)),
    }, index=trips.index)
    for col in all_feature_cols:
        col_map = cells_transit[col]
        orig = trips['trip_orig_cell'].map(col_map).astype(float).fillna(0.0)
        dest = trips['trip_dest_cell'].map(col_map).astype(float).fillna(0.0)
        df[col] = (orig + dest).to_numpy()

    row_index = ['cap_s', 'const_s', 't_routed', *all_feature_cols]
    columns = [c for spec in _TRANSIT_SPECS for c in (spec, f'{spec}__uncon')]
    out = pd.DataFrame(index=row_index, columns=columns, dtype=float)
    out.index.name = 'param'

    for spec_name, feature_cols in _TRANSIT_SPECS.items():
        # Constrained (applied) fit.
        params_c = _fit_constrained(df, feature_cols, spec_name)
        out.loc['cap_s', spec_name] = _TRANSIT_OVERHEAD_CAP_S
        out.loc['const_s', spec_name] = float(params_c['const'])
        # t_routed row stays NaN for constrained columns (α = 1 by construction).
        for c in feature_cols:
            out.loc[c, spec_name] = float(params_c[c])

        # Unconstrained (informational) fit.
        uncon_name = f'{spec_name}__uncon'
        params_u = _fit_unconstrained(df, feature_cols, spec_name)
        out.loc['cap_s', uncon_name] = _TRANSIT_OVERHEAD_CAP_S
        out.loc['const_s', uncon_name] = float(params_u['const'])
        out.loc['t_routed', uncon_name] = float(params_u['t_routed'])
        for c in feature_cols:
            out.loc[c, uncon_name] = float(params_u[c])

    # `transit` column is what per_cell_transit_overhead reads. Alias
    # from the production spec (constrained).
    if _PRODUCTION_SPEC not in out.columns:
        raise RuntimeError(
            f"_PRODUCTION_SPEC={_PRODUCTION_SPEC!r} not in fitted specs "
            f"({list(_TRANSIT_SPECS)}).")
    out['transit'] = out[_PRODUCTION_SPEC]
    return out


def _scatter_z2z_vs_measured(
    context, trips: pd.DataFrame, calibrated_coefs: pd.DataFrame,
) -> None:
    """Hexbin scatter of measured trip time vs raw z2z (both in minutes).
    Overlays 1:1 line and the calibrated α=1 constant offset
    (`y = z2z + const_s`) for reference — endpoint terms vary per trip
    and can't be drawn as a single line."""
    sub = trips[['trip_time_measured', 'raw_z2z']].dropna()
    x = sub['raw_z2z'].to_numpy() / 60.0
    y = sub['trip_time_measured'].to_numpy() / 60.0
    upper = float(np.percentile(np.concatenate([x, y]), 99))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    hb = ax.hexbin(x, y, gridsize=80, cmap='viridis', mincnt=1, bins='log',
                   extent=(0, upper, 0, upper))
    fig.colorbar(hb, ax=ax).set_label('count (log scale)')
    ax.plot([0, upper], [0, upper], 'r--', lw=1.5, label='1:1 (NPVM = survey)')

    const_s = float(calibrated_coefs.loc['const_s', 'transit'])
    x_line = np.linspace(0, upper, 50)
    ax.plot(x_line, const_s / 60.0 + x_line, color='orange', lw=2,
            label=f'calibrated (α=1, endpoints=0): y = {const_s/60:+.1f} + x  (min)')

    ax.set(xlim=(0, upper), ylim=(0, upper),
           xlabel='NPVM z2z transit time at trip endpoints (min)',
           ylabel='Survey measured TRIP time, door-to-door (min)',
           title=(f'NPVM PT z2z vs. survey TRIP time (door-to-door)\n'
                  f'n = {len(sub):,} trips '
                  f'(axes capped at 99th percentile = {upper:.0f} min)'))
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')

    context.create_results(fig, 'transit/scatter_z2z_vs_measured.png')
    plt.close(fig)


def _prep_trip_data(context, scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load legs, aggregate to transit-containing trips, quality/AOI filter,
    attach raw NPVM z2z per trip. Returns `(trips, cells_transit)` ready
    for the constrained OLS fit."""
    with step('load survey_legs.csv'):
        legs = context.get_generic('survey_legs.csv', storage=Storage.PRIVATE)
        if 'trip_id' not in legs.columns:
            raise KeyError(
                "Column 'trip_id' not in survey_legs.csv. "
                "Verify it's in survey/02d's `_KEEP_COLS` and rerun 02d.")
        logging.info(f"  → {len(legs):,} legs")

    with step('aggregate legs → transit-containing trips'):
        trips = aggregate_transit_trips(legs)
        n_by_pattern = trips['access_pattern'].value_counts().to_dict()
        logging.info(
            f"  → {len(trips):,} transit trips; access patterns: {n_by_pattern}")

    with step('quality filter (both trip-level envelope flags == 1)'):
        n_before = len(trips)
        for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
            if flag in trips.columns:
                trips = trips[trips[flag] == 1]
        logging.info(f"  → {len(trips):,} / {n_before:,} trips kept")

    with step(f'trip_dist_line ∈ [{_TRIP_DIST_LINE_MIN:.0f}, '
              f'{_TRIP_DIST_LINE_MAX:.0f}] m'):
        n_before = len(trips)
        trips = trips[
            (trips['trip_dist_line'] >= _TRIP_DIST_LINE_MIN)
            & (trips['trip_dist_line'] <= _TRIP_DIST_LINE_MAX)]
        logging.info(f"  → {len(trips):,} / {n_before:,} trips kept")

    with step('AOI filter (both trip endpoints inside scenario AOI)'):
        aoi_polygon = load_aoi_polygon(context)
        trips = filter_legs_by_aoi(
            trips, aoi_polygon, crs=scenario.crs_main,
            orig_x='trip_orig_x', orig_y='trip_orig_y',
            dest_x='trip_dest_x', dest_y='trip_dest_y',
            label='transit trips')

    with step('NPVM z2z lookup at trip endpoints (raw, no overhead)'):
        lookup = npvm_transit_z2z_lookup(context)
        keys = list(zip(trips['trip_orig_zone'], trips['trip_dest_zone']))
        trips['raw_z2z'] = pd.Series(
            [lookup.get(k, float('nan')) for k in keys],
            index=trips.index, dtype=float)
        n_hit = int(trips['raw_z2z'].notna().sum())
        logging.info(
            f"  → {n_hit:,}/{len(trips):,} trips with NPVM z2z entry "
            f"({100*n_hit/max(len(trips),1):.1f} %); "
            f"median raw_z2z = {trips['raw_z2z'].median():.0f} s")

    with step('load cells_transit_access (endpoint zone-dev columns)'):
        cells_transit = context.get_properties('cells', 'transit_access')
        logging.info(f"  → {len(cells_transit):,} cells")

    return trips, cells_transit


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    # Prep survey data only when this scenario actually calibrates
    # overheads_transit. ImportFrom / HandWritten scenarios skip the
    # heavy lifting; `coefs.resolve` copies / verifies without calling
    # the calibrate_fn.
    trips_state: tuple[pd.DataFrame, pd.DataFrame] | None = None
    if scenario_needs_survey_prep(scenario):
        trips_state = _prep_trip_data(context, scenario)

    def _calibrate() -> pd.DataFrame:
        if trips_state is None:
            raise RuntimeError(
                "scenario declares overheads_transit=Calibrate() but survey "
                "prep was skipped — this should not happen. Report as a bug.")
        trips, cells_transit = trips_state
        params_df = _fit_calibrated_transit_overhead(trips, cells_transit)
        with step('save scatter (measured vs NPVM z2z)'):
            _scatter_z2z_vs_measured(context, trips, params_df)
        return params_df

    coefs.resolve(
        context, name='overheads_transit', calibrate_fn=_calibrate,
    )
    context.close()


if __name__ == '__main__':
    main()
