"""
Validate atlas's per-edge AADT flow estimates (from 03b_traffic_flows.py)
against Swiss traffic counters (NPVM zaehldaten).

Loads the saved `edges_car_flows.csv` + the car graph from atlas,
re-tags each edge with a tier bucket (`highway` / `main` / `local`)
derived from `OSM_HIGHWAY_RANKS`, snaps each counter to the right
edge using `calibration.snap_counters_to_edges` (with per-counter
search radius + bearing tolerance + tier-matched eligibility), then
runs `calibration.evaluate_against_counters` to pair each matched
counter with its edge's modeled AADT. Agreement stats (`technical_validation.common.
agreement_stats`, same columns as `times_vs_survey*.py`) are reported for
all counters and per tier.

Out-of-sample: 03a / 03b never see the counters (flow coefficients come
from the survey; absolute volumes from the per-person trip rate).

Also draws a log-log scatter of modeled vs observed AADT per matched
counter, coloured by road class, with the overall R²_fit in its title.

Inputs:
    <scenario>/  (PUBLIC, via context)
        nw/car.graphml + properties/{nodes,edges}_car_core.csv
        properties/edges_car_flows.csv      # from 03b_traffic_flows.py
    PRIVATE/preparation/switzerland/traffic_counters/traffic_counters.gpkg
        # directional point counters with `traffic_cars`, `bearing_deg`,
        # `is_highway` / `is_main` / `is_local` flags.

Outputs (RESULTS, under <scenario>/):
    technical_validation/flows_vs_counters.csv   # per-tier stats
    technical_validation/flows_vs_counters.png   # modeled vs observed AADT scatter

Run:
    python -m technical_validation.flows_vs_counters --scenario <name>
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from aperta import calibration
from aperta.network_processing import parse_edge_id
from aperta_atlas.context import Storage, init_context
from aperta_atlas.osm import OSM_HIGHWAY_RANKS
from aperta_atlas.utils import step

from preparation.world.areas import AREAS
from preparation.world.common import buffered_place_polygon
from scenarios import get_scenario
from technical_validation.common import agreement_stats


# Counter-to-edge snap. Highway counters get a wider radius (sparser
# layout, lower risk of catching a parallel local road); non-highway
# counters get a tight radius. Bearing tolerance prevents
# opposite-direction counters from cross-snapping on two-way roads.
_HIGHWAY_RADIUS_M = 50.0
_NON_HIGHWAY_RADIUS_M = 10.0
_BEARING_TOL_DEG = 10.0

# Tier-classification cutoffs over `OSM_HIGHWAY_RANKS`. Matches the
# extended notebook's bucketing exactly.
_TIER_HIGHWAY_MIN_RANK = 6   # motorway / trunk
_TIER_MAIN_MIN_RANK = 3      # primary / secondary / tertiary
# Anything below tertiary → 'local' (residential / service / unknown).

_COUNTERS_NAMESPACE = 'preparation/switzerland/traffic_counters'
_COUNTERS_FILE = 'traffic_counters.gpkg'

_AOI_BUFFER = -10_000


def _edge_tier(d: dict) -> str:
    """Map an edge attribute dict to a tier bucket via OSM highway rank."""
    hwy = d.get('highway')
    if isinstance(hwy, list):
        hwy = hwy[0] if hwy else None
    rank = OSM_HIGHWAY_RANKS.get(hwy, -1)
    if rank >= _TIER_HIGHWAY_MIN_RANK:
        return 'highway'
    if rank >= _TIER_MAIN_MIN_RANK:
        return 'main'
    return 'local'


_TIERS: list[tuple[str, str | None]] = [
    ('ALL', None), ('highway', 'is_highway'), ('main', 'is_main'), ('local', 'is_local'),
]

# Scatter: road class → (legend label, colour). Fixed order, first three slots of a
# colour-blind-checked categorical palette (valid for all pairs of three series).
_SCATTER_STYLE: dict[str, tuple[str, str]] = {
    'highway': ('Motorway / trunk', '#2a78d6'),
    'main':    ('Main road', '#eb6834'),
    'local':   ('Local road', '#1baf7a'),
}


def _plot_scatter(context, merged: pd.DataFrame, counters: pd.DataFrame,
                  rows: list[dict]) -> None:
    """Log-log scatter of modeled vs observed AADT per matched counter, one colour
    per road class, with a 1:1 line and the overall R²_fit as title. Zero values cannot be drawn on log axes and are
    left out (logged)."""
    drawable = (merged['observed'] > 0) & (merged['modeled'] > 0)
    if not drawable.all():
        logging.info(f"  → {int((~drawable).sum()):,} counters with zero AADT not drawn")
    pts = merged[drawable]
    tier_of = pd.Series('local', index=pts.index)
    tier_of[counters.loc[pts.index, 'is_main'] == 1] = 'main'
    tier_of[counters.loc[pts.index, 'is_highway'] == 1] = 'highway'

    fig, ax = plt.subplots(figsize=(3.6, 3.6), layout='constrained')
    handles = {}
    # Larger classes first, so the sparse ones stay visible on top.
    for tier in tier_of.value_counts().index:
        label, color = _SCATTER_STYLE[tier]
        sub = pts[tier_of == tier]
        handles[tier] = ax.scatter(
            sub['observed'], sub['modeled'], s=8, color=color, alpha=0.6,
            edgecolors='white', linewidths=0.3, zorder=2,
            label=f'{label} (n = {len(sub):,})')
    lo = 0.8 * min(pts['observed'].min(), pts['modeled'].min())
    hi = 1.25 * max(pts['observed'].max(), pts['modeled'].max())
    (one_to_one,) = ax.plot([lo, hi], [lo, hi], color='0.45', lw=0.8, ls='--', zorder=1,
                            label='1:1')
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
    ax.set_xlabel('Observed AADT (vehicles/day)', fontsize=8)
    ax.set_ylabel('Modeled AADT (vehicles/day)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which='major', color='0.9', lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[handles[t] for t in _SCATTER_STYLE if t in handles] + [one_to_one],
              fontsize=7, frameon=False, loc='upper left')
    overall = next(r for r in rows if r['tier'] == 'ALL')
    ax.set_title(rf"$R^2_\mathrm{{fit}}$ = {overall['r2_fit']:.2f}", fontsize=8)
    context.create_results(fig, 'technical_validation/flows_vs_counters.png',
                           kws={'dpi': 300})
    plt.close(fig)


def _log_and_save(context, rows: list[dict]) -> None:
    df = pd.DataFrame(rows).rename(columns={
        'n': 'n_counters', 'bias_pct': 'bias%', 'mae': 'MAE', 'rmse': 'RMSE',
        'r2': 'R²_1to1', 'r2_fit': 'R²_fit', 'mae_pct': 'MAE%',
    })
    display_cols = ['tier', 'n_counters', 'n_in_area',
                    'bias', 'bias%', 'MAE', 'RMSE',
                    'slope', 'R²_1to1', 'R²_fit', 'MAE%']
    context.create_results(
        df[display_cols], 'technical_validation/flows_vs_counters.csv',
        kws={'index': False, 'float_format': '%.4f'},
    )
    logging.info('\n' + df[display_cols].to_string(index=False, float_format='{:.3f}'.format))


def main():
    context = init_context()
    assert context.scenario is not None, \
        ('atlas requires a scenario (set --scenario or '
         "DEFAULT_SCENARIO in src/scenarios.py)")
    scenario = get_scenario(context.scenario)
    crs_main = scenario.crs_main

    # ---------- Load graph + flow estimates ------------------------------
    with step('load car graph (+ core overlay for highway tags)'):
        graph = context.get_nw(
            data_name='car',
            add_node_properties='core',
            add_edge_properties=['core', 'from_nodes'],
        )
        logging.info(
            f"  → {graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges")

    with step('tag every edge with tier (highway / main / local)'):
        n_per_tier = {'highway': 0, 'main': 0, 'local': 0}
        for _, _, _, d in graph.edges(keys=True, data=True):
            d['_tier'] = _edge_tier(d)
            n_per_tier[d['_tier']] += 1
        logging.info(f"  → tier distribution: {n_per_tier}")

    with step('load edge flow estimates (from 03b_traffic_flows.py)'):
        flows_df = context.get_properties('edges', 'car_flows')
        # Flow CSV is indexed by `edge_id` (string `'u:v:k'`); convert to
        # tuple index so `evaluate_against_counters` can do its (u,v,k)
        # lookup against the graph's edges.
        flows = pd.Series(
            flows_df['flow_estimate'].to_numpy(),
            index=pd.MultiIndex.from_tuples(
                [parse_edge_id(s) for s in flows_df.index],
                names=['u', 'v', 'k'],
            ),
            name='flow_estimate',
        )
        logging.info(
            f"  → {len(flows):,} edges with flow; "
            f"median {flows.median():,.0f}, P95 {flows.quantile(0.95):,.0f} veh/day")

    # ---------- Load + snap counters --------------------------------------
    with step('load counters (PRIVATE Swiss traffic counts)'):
        ctr_ctx = context.source(_COUNTERS_NAMESPACE, storage=Storage.PRIVATE)
        counters_path = ctr_ctx.path_for(ctr_ctx.default_storage, _COUNTERS_FILE)
        counters = gpd.read_file(counters_path).to_crs(crs_main)
        logging.info(
            f"  → {len(counters):,} counters loaded "
            f"(highway: {int(counters['is_highway'].sum()):,}, "
            f"main: {int(counters['is_main'].sum()):,}, "
            f"local: {int(counters['is_local'].sum()):,})")

    with step('filter counters → area polygon (inner core, no buffer)'):
        # The area polygon (e.g. Canton of Bern boundary) — counters
        # outside it see traffic going to/from places the simulation
        # can't model (because they're outside the cell+zone layer),
        # which inflates observed AADT relative to modeled. Filter to
        # in-area-polygon counters for a fair comparison.
        area = AREAS[scenario.area_name]
        # `union_all()` on a 1-row GeoSeries returns that single geometry;
        # cleaner type-wise than `.geometry.iloc[0]` (whose pandas-side
        # return type isn't narrowed to BaseGeometry by geopandas stubs).
        core_polygon = (
            buffered_place_polygon(area.place, _AOI_BUFFER)
            .to_crs(crs_main).geometry.union_all())
        n_before = len(counters)
        in_core = counters.geometry.within(core_polygon)
        counters = counters[in_core].copy()
        logging.info(
            f"  → kept {len(counters):,} of {n_before:,} counters "
            f"({100 * len(counters) / n_before:.1f} %) inside the "
            f"{area.place!r} polygon")

    with step(
        f'snap counters → edges (bearing_tol={_BEARING_TOL_DEG}°, '
        f'radius {_HIGHWAY_RADIUS_M:.0f}/{_NON_HIGHWAY_RADIUS_M:.0f} m for '
        f'highway/other)'
    ):
        # Per-counter radius: wider for highway counters.
        max_distance = counters['is_highway'].map(
            {1: _HIGHWAY_RADIUS_M, 0: _NON_HIGHWAY_RADIUS_M})

        # Per-counter eligibility: highway counters only snap to highway
        # edges, main to main, local to local. Prevents catching the wrong
        # road class on parallel infrastructure (e.g. a highway counter
        # snapping to a frontage road).
        def _eligible_for_counter(counter_row, candidate_edges):
            if counter_row['is_highway']:
                wanted = 'highway'
            elif counter_row['is_main']:
                wanted = 'main'
            else:
                wanted = 'local'
            return candidate_edges[candidate_edges['_tier'] == wanted]

        snapped = calibration.snap_counters_to_edges(
            counters, graph,
            max_distance=max_distance,
            bearing_tol_deg=_BEARING_TOL_DEG,
            eligible_edges=_eligible_for_counter,
        )
        counters = counters.join(snapped)
        n_matched = int(counters['u'].notna().sum())
        logging.info(
            f"  → snapped {n_matched:,} of {len(counters):,} counters "
            f"({100 * n_matched / len(counters):.1f} %)")
        for label, col in [('highway', 'is_highway'), ('main', 'is_main'),
                           ('local', 'is_local')]:
            mask = counters[col] == 1
            n_tot = int(mask.sum())
            n_ma = int((mask & counters['u'].notna()).sum())
            if n_tot:
                logging.info(
                    f"     {label:7s}: {n_ma:,} of {n_tot:,} matched "
                    f"({100 * n_ma / n_tot:.1f} %)")

    # ---------- Evaluate: overall + per-tier ------------------------------
    with step('evaluate modeled vs observed AADT (overall + per-tier)'):
        # `merged` pairs every matched counter (index preserved) with its
        # edge's modeled AADT; tier filtering happens on that frame.
        merged = calibration.evaluate_against_counters(flows, counters)['merged']
        rows: list[dict] = []
        for tier, col in _TIERS:
            in_tier = (counters[col] == 1) if col else pd.Series(True, index=counters.index)
            sub = merged[merged.index.isin(counters.index[in_tier])]
            rows.append({'tier': tier, 'n_in_area': int(in_tier.sum()),
                         **agreement_stats(sub['observed'].to_numpy(dtype=float),
                                           sub['modeled'].to_numpy(dtype=float))})

    with step('summary'):
        _log_and_save(context, rows)

    with step('scatter plot: modeled vs observed AADT'):
        _plot_scatter(context, merged, counters, rows)

    context.close()


if __name__ == '__main__':
    main()
