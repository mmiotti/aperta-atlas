"""
Route every survey trip per (mode, calibrated profile) for the atlas.

For each mode in `scenario.mode_configs`, loads the graph + 04's
calibrated per-edge durations, then routes every survey leg (orig node
→ dest node) for every profile under that mode. Per-mode `dist_line`
masks drop trips beyond what's plausible for that mode.

One variant per leg set (see survey/02d). The `mtmc` set is routed on
every mode (09a needs every alternative); the validation-only MOBIS sets
are routed only on each leg's chosen mode.

Uses `aperta_atlas.routing_mp.shortest_path_metrics_one_to_one_mp` —
fork-multiprocessing wrapper around nx bidirectional Dijkstra (chosen
over igraph: ~6× faster on unique-source per-trip queries at country
scale). Most memory-intensive script in the pipeline; the lean graph
is shared across workers via fork-COW.

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_legs[_<leg_set>].csv                  # from survey/02d

Inputs (PUBLIC, under `<scenario>/`):
    nw/<mode>.graphml
    properties/edges_<mode>_core.csv
    properties/edges_<mode>_calibrated.csv               # from 04
    properties/edges_<mode>_extended.csv                 # from 02b — delta_elevation
    properties/edges_<mode>_from_nodes.csv               # from 02b — density, bike_infra_score, etc.

Output (PRIVATE, under `<scenario>/`):
    generic/survey_leg_times[_<leg_set>].csv
                                 # per road profile <p>: t_routed_<p>, length_<p>,
                                 # elev_gain_<p>, elev_loss_<p>, density_r500_norm_<p>,
                                 # mean_abs_slope_r250_<p>, n_traffic_signals_<p>;
                                 # bike-mode extras: bike_infra_score_avg_r{250,500}_<p>;
                                 # car-mode extras: speed_limit_avg_r{250,500}_<p>, vc_beta_2.0_<p>;
                                 # naive-baseline routing (04's
                                 # `duration_baseline_<p>` weights):
                                 # t_baseline_<p>, length_baseline_<p>
                                 # per road profile; t_baseline_car +
                                 # length_baseline_car aggregated by
                                 # peak_str;
                                 # transit (NPVM z2z lookup): t_routed_transit_z2z

Run (default variant `mtmc`):
    python -m survey.05_leg_times --scenario <name>
MOBIS validation leg set:
    python -m survey.05_leg_times --scenario <name> --variant mobis_precovid
"""

import logging

import numpy as np
import pandas as pd

from aperta.network_processing import attach_edge_properties
from aperta_atlas import routing_mp
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from main.common import SURVEY_LEG_SETS, npvm_transit_z2z_lookup, survey_file
from scenarios import get_scenario, scenario_needs_survey_prep


# `mode_simplified` values routed on each network, for leg sets routed on the chosen mode only.
# `anybike` / `anyebike` are MOBIS bike legs of unknown sub-type (before its 2020-07 split); routed
# like any bike leg, so they get every bike profile's times.
_CHOSEN_MODE_VALUES: dict[str, set[str]] = {
    'walk': {'walk'},
    'bike': {'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike'},
    'car':  {'car'},
}


# Set to e.g. 10_000 to route only the first N legs (timing tests). None = all.
_LIMIT_TRIPS: int | None = None


# Per-mode `dist_line` (Euclidean OD distance) cap in meters. Trips beyond
# the cap for a mode are NOT routed for any of that mode's profiles —
# downstream code reads NaN in the corresponding output columns.
_DIST_LINE_MAX: dict[str, float | None] = {
    'walk':  5_000.0,
    'bike': 30_000.0,
    'car': 150_000.0,
}

# Per-edge attributes aggregated along each routed path. `'sum'` =
# cumulative total; `'length_weighted'` = mean weighted by per-edge
# `length`. Per-profile columns land in `survey_leg_times.csv` as
# `<attr>_<profile>` (except `is_traffic_signal` → `n_traffic_signals_<profile>`).
# ⚠ Mirror kept in `main.common._ROUTE_EDGE_FEATURES_*` (used by
# `build_disutility_spec`'s route classifier) — keep them in sync.
_PATH_EDGE_FEATURES_UNIVERSAL: dict[str, str] = {
    'length':              'sum',              # metres — routed distance
    'elev_gain':           'sum',              # metres of climb
    'elev_loss':           'sum',              # metres of descent
    'density_r500_norm':   'length_weighted',  # avg local density along route
    'mean_abs_slope_r250': 'length_weighted',  # avg local hilliness
    'is_traffic_signal':   'sum',              # count of signals traversed
}
# Per-mode extras — only aggregated on the graph where the feature is
# natural (bike_infra_score on bike, speed_limit + traffic pressure on
# car).
_PATH_EDGE_FEATURES_MODE: dict[str, dict[str, str]] = {
    'walk': {},
    'bike': {
        'bike_infra_score_avg_r250': 'length_weighted',
    },
    'car': {
        'speed_limit_avg_r250':      'length_weighted',
        'vc_beta_2.0':               'length_weighted',   # BPR-style traffic pressure (03b)
    },
}

# Per-mode extra edge-property files beyond `<mode>_extended` +
# `<mode>_from_nodes`. Only used to source path-aggregation features
# that don't live in those two. Attach-time logic picks each path
# feature from whichever loaded frame has it, so new entries here just
# need a matching column in `_PATH_EDGE_FEATURES_MODE`.
_EXTRA_EDGE_PROPERTIES: dict[str, tuple[str, ...]] = {
    'car': ('car_flows',),   # vc + vc_beta_* + flow_estimate (from 03b)
}


def _path_edge_features_for(mode: str) -> dict[str, str]:
    return {**_PATH_EDGE_FEATURES_UNIVERSAL,
            **_PATH_EDGE_FEATURES_MODE.get(mode, {})}


def _routable_mask(legs: pd.DataFrame, mode: str) -> pd.Series:
    """True where the trip's `dist_line` is within the mode's cap. 05a's
    completeness filter guarantees every leg has snapped to every mode,
    so no snap-presence check is needed here."""
    cap = _DIST_LINE_MAX[mode]
    if cap is None:
        return pd.Series(True, index=legs.index)
    return legs['dist_line'] < cap


def main(variant):
    context = init_context(variant)
    scenario = get_scenario(context.scenario)
    if not scenario_needs_survey_prep(scenario):
        logging.info(
            f"Scenario {scenario.name!r} has no survey-driven calibrators — "
            f"skipping (survey prep outputs would have no consumer).")
        context.close()
        return
    leg_set = variant.leg_set
    chosen_mode_only = leg_set != 'mtmc'

    with step(f'load survey legs ({leg_set})'):
        legs = context.get_generic(
            survey_file('survey_legs.csv', leg_set), storage=Storage.PRIVATE)
        if _LIMIT_TRIPS is not None:
            legs = legs.iloc[:_LIMIT_TRIPS].copy()
            logging.info(f"  → restricted to first {_LIMIT_TRIPS:,} legs")
        logging.info(f"  → {len(legs):,} legs")

    # Output frame: NaN-initialised, one column per profile filled in below.
    out = pd.DataFrame(index=legs.index)

    for mode_config in scenario.mode_configs.values():
        # All profiles within a mode share `edges_data_name`; the
        # `edge_column` differs per profile.
        sources = [mode_config.source_for(p) for p in mode_config.profiles]
        weight_cols = [src.edge_column for src in sources]
        # Baseline edge-weight columns for the naive-routing comparison.
        # Named `duration_baseline_<profile>` in `edges_<mode>_calibrated.csv`
        # (04 writes them alongside `duration_calibrated_<profile>`).
        baseline_cols = [c.replace('duration_calibrated_', 'duration_baseline_')
                         for c in weight_cols]
        edges_data_name = sources[0].edges_data_name

        with step(f'mode={mode_config.mode}: load graph skeleton'):
            # Attach weight columns below (post-lean); keeps worker
            # inherited memory minimal via fork-COW.
            graph = context.get_nw(data_name=mode_config.mode, allow_cache=False)
            logging.info(f"  → graph {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

        with step(f'mode={mode_config.mode}: load calibrated edge durations table'):
            # `allow_cache=False` throughout this per-mode block: the
            # frames are only used to attach columns to the graph;
            # keeping them cached would bloat every fork-multiprocessing
            # worker's inherited memory via COW pages.
            calibrated = context.get_properties(
                'edges', edges_data_name, allow_cache=False)

        with step(f'mode={mode_config.mode}: attach path-aggregation edge features'):
            extended = context.get_properties(
                'edges', f'{mode_config.mode}_extended', allow_cache=False)
            elev_gain = extended['elevation_gain'].fillna(0.0).rename('elev_gain')
            elev_loss = extended['elevation_loss'].fillna(0.0).rename('elev_loss')
            # Feature sources: `<mode>_from_nodes` (always) + any files in
            # `_EXTRA_EDGE_PROPERTIES`. `_pick()` finds each feature in
            # the first frame that has it.
            feature_sources = [
                context.get_properties(
                    'edges', f'{mode_config.mode}_from_nodes',
                    allow_cache=False)
            ]
            for extra_name in _EXTRA_EDGE_PROPERTIES.get(mode_config.mode, ()):
                feature_sources.append(
                    context.get_properties(
                        'edges', extra_name, allow_cache=False))
            path_features = _path_edge_features_for(mode_config.mode)
            # `length` is native; `elev_gain` / `elev_loss` attached above.
            _ALREADY_ATTACHED = ('length', 'elev_gain', 'elev_loss')
            path_feat_cols = [c for c in path_features
                              if c not in _ALREADY_ATTACHED]

            def _pick(col: str) -> pd.Series:
                for src in feature_sources:
                    if col in src.columns:
                        return src[col].fillna(0.0)
                raise KeyError(
                    f"path feature {col!r} not found in any edge-property "
                    f"file loaded for mode={mode_config.mode!r}. "
                    f"Available: {[list(s.columns) for s in feature_sources]}")
            attach_edge_properties(
                graph,
                pd.concat(
                    [elev_gain, elev_loss] + [_pick(c) for c in path_feat_cols],
                    axis=1,
                ),
            )
            logging.info(
                f"  → elev + node-derived feats attached; "
                f"median elev_gain = {elev_gain.median():.2f} m, "
                f"loss = {elev_loss.median():.2f} m per edge")

        with step(f'mode={mode_config.mode}: attach weights + lean graph'):
            # Baseline weights only attached if the columns are present
            # in the calibrated CSV (older 04 outputs won't have them —
            # in that case the naive-routing comparison is skipped).
            baseline_available = [c for c in baseline_cols
                                  if c in calibrated.columns]
            weights_to_attach = list(weight_cols) + baseline_available
            attach_edge_properties(graph, calibrated[weights_to_attach])
            # Lean keeps: routing weights (calibrated + baseline) + `length`
            # (needed by the metrics variant to sum path distance) + all
            # attrs aggregated along the path via `path_features`.
            lean_attrs = weight_cols + baseline_available + ['length'] + list(path_features)
            lean = routing_mp.lean_graph(graph, lean_attrs)
            logging.info(f"  → lean graph kept {lean.number_of_edges():,} edges with attrs {lean_attrs}")

        # Pre-fork cleanup: everything above is now baked into `lean`
        # (edge attrs copied in; routing needs only `lean`). Freeing
        # the raw property frames + full-fat graph BEFORE
        # `shortest_path_metrics_one_to_one_mp` forks workers means
        # each of the 6 fork-COW children starts with a smaller
        # inherited page set. Effect scales linearly with n_workers.
        del graph, calibrated, extended
        del elev_gain, elev_loss, feature_sources

        mask = _routable_mask(legs, mode_config.mode)
        if chosen_mode_only:
            mask &= legs['mode_simplified'].isin(_CHOSEN_MODE_VALUES[mode_config.mode])
        orig_col = f'orig_node_id_{mode_config.mode}'
        dest_col = f'dest_node_id_{mode_config.mode}'
        routable = legs.loc[mask, [orig_col, dest_col]].copy()
        # Snap columns round-trip through CSV as float (NaN-friendly). Cast
        # back to the graph's int node-id type after dropping NaNs above.
        routable[orig_col] = routable[orig_col].astype(int)
        routable[dest_col] = routable[dest_col].astype(int)
        logging.info(
            f"  → {len(routable):,}/{len(legs):,} legs routable for mode={mode_config.mode} "
            f"(dist_line cap = {_DIST_LINE_MAX[mode_config.mode]})")
        if routable.empty:
            logging.warning(f"  ⚠ no legs to route for mode={mode_config.mode} — skipped.")
            del lean, routable
            continue

        for src in sources:
            with step(f'profile={src.name}: route {len(routable):,} legs (mp, metrics)', 1):
                # Metrics variant: aggregates `path_features` along each
                # routed path. Add entries to the module-top dicts to
                # collect more features without touching this call site.
                metrics = routing_mp.shortest_path_metrics_one_to_one_mp(
                    lean,
                    trip_ids=routable.index.values,
                    origins=routable[orig_col].values,
                    destinations=routable[dest_col].values,
                    weight=src.edge_column,
                    edge_features=dict(path_features),
                    n_workers=6,
                )
                # `metrics` is indexed by trip_id; unroutable trips dropped.
                # Reindex to legs.index → NaN where missing.
                costs = metrics['cost']
                out[f't_routed_{src.name}'] = costs.reindex(legs.index)
                for attr in path_features:
                    out_col = ('n_traffic_signals' if attr == 'is_traffic_signal'
                               else attr)
                    out[f'{out_col}_{src.name}'] = metrics[attr].reindex(legs.index)
                n_routed = costs.notna().sum()
                logging.info(
                    f"  → {n_routed:,}/{len(routable):,} paths found "
                    f"({100*n_routed/max(len(routable),1):.1f} %); "
                    f"median t_routed = {costs.median():.1f} s; "
                    f"median elev gain = {metrics['elev_gain'].median():.1f} m, "
                    f"loss = {metrics['elev_loss'].median():.1f} m; "
                    f"median density_r500_norm = "
                    f"{metrics['density_r500_norm'].median():.3f}; "
                    f"median n_traffic_signals = "
                    f"{metrics['is_traffic_signal'].median():.1f}")

            # Naive-baseline routing pass — same origins/destinations but
            # weight = `duration_baseline_<profile>` (per-edge naive
            # duration from 04). Only cost + length are aggregated; other
            # path features aren't needed for the validation comparison.
            baseline_col = src.edge_column.replace(
                'duration_calibrated_', 'duration_baseline_')
            if baseline_col not in lean.edges[next(iter(lean.edges))]:
                logging.info(
                    f"  → baseline column {baseline_col!r} not in graph; "
                    f"naive-routing comparison skipped for profile={src.name!r}.")
                continue
            with step(f'profile={src.name}: baseline route ({baseline_col})', 1):
                metrics_b = routing_mp.shortest_path_metrics_one_to_one_mp(
                    lean,
                    trip_ids=routable.index.values,
                    origins=routable[orig_col].values,
                    destinations=routable[dest_col].values,
                    weight=baseline_col,
                    edge_features={'length': 'sum'},
                    n_workers=6,
                )
                costs_b = metrics_b['cost']
                out[f't_baseline_{src.name}'] = costs_b.reindex(legs.index)
                out[f'length_baseline_{src.name}'] = metrics_b['length'].reindex(legs.index)
                n_routed_b = costs_b.notna().sum()
                logging.info(
                    f"  → {n_routed_b:,}/{len(routable):,} baseline paths; "
                    f"median t_baseline = {costs_b.median():.1f} s")

        # End of this mode's routing — free `lean` (potentially a few
        # hundred MB for car) before the next mode's graph load starts
        # accumulating. Prevents brief double-holding at the mode boundary.
        del lean, routable

    # Aggregate car profiles → `t_routed_car` picked by each leg's
    # `peak_str`. Only fires if all 3 car profiles were routed. Mirrors
    # the same aggregation for the naive baseline (`t_baseline_car`)
    # when the three baseline columns are present.
    car_cols = {'t_routed_car_peak', 't_routed_car_base', 't_routed_car_night'}
    if car_cols.issubset(out.columns):
        with step('aggregate car profiles → t_routed_car by peak_str'):
            peak_str = legs['peak_str']
            out['t_routed_car'] = np.select(
                [peak_str == 'peak', peak_str == 'base', peak_str == 'night'],
                [out['t_routed_car_peak'],
                 out['t_routed_car_base'],
                 out['t_routed_car_night']],
                default=np.nan,
            )
            n_aggregated = int(out['t_routed_car'].notna().sum())
            n_unknown_peak = int((~peak_str.isin(['peak', 'base', 'night'])).sum())
            logging.info(
                f"  → {n_aggregated:,}/{len(out):,} aggregated "
                f"({n_unknown_peak:,} legs with unknown peak_str → NaN)")
    car_baseline_cols = {'t_baseline_car_peak', 't_baseline_car_base', 't_baseline_car_night'}
    if car_baseline_cols.issubset(out.columns):
        with step('aggregate car baseline profiles → t_baseline_car by peak_str'):
            peak_str = legs['peak_str']
            out['t_baseline_car'] = np.select(
                [peak_str == 'peak', peak_str == 'base', peak_str == 'night'],
                [out['t_baseline_car_peak'],
                 out['t_baseline_car_base'],
                 out['t_baseline_car_night']],
                default=np.nan,
            )
            # `length_baseline_car` — same aggregation on the routed
            # baseline length (naive-path length can differ from calibrated-
            # path length when baseline speeds vary per edge).
            out['length_baseline_car'] = np.select(
                [peak_str == 'peak', peak_str == 'base', peak_str == 'night'],
                [out['length_baseline_car_peak'],
                 out['length_baseline_car_base'],
                 out['length_baseline_car_night']],
                default=np.nan,
            )

    with step('transit: NPVM z2z lookup per leg → t_routed_transit_z2z'):
        # Aperta has no native transit router — "routing" here is a
        # zone-to-zone lookup into NPVM's PT travel-time matrix keyed on
        # (orig_zone_id, dest_zone_id). Legs whose zone-pair is missing
        # from NPVM get NaN.
        lookup = npvm_transit_z2z_lookup(context)
        keys = list(zip(legs['orig_zone_id'], legs['dest_zone_id']))
        out['t_routed_transit_z2z'] = pd.Series(
            [lookup.get(k, float('nan')) for k in keys], index=legs.index)
        n_with = int(out['t_routed_transit_z2z'].notna().sum())
        logging.info(
            f"  → {n_with:,}/{len(legs):,} legs with NPVM z2z entry "
            f"({100*n_with/max(len(legs),1):.1f} %); "
            f"median t_routed_transit_z2z = "
            f"{out['t_routed_transit_z2z'].median():.1f} s")

    context.create_generic(out, survey_file('survey_leg_times.csv', leg_set),
                           storage=Storage.PRIVATE, kws={'float_format': '%.3f'})
    logging.info(f"  → saved {len(out):,} rows × {len(out.columns)} profile cols")

    context.close()


variants = Variants([('leg_set', str)])
for _leg_set in SURVEY_LEG_SETS:
    variants.add(name=_leg_set, leg_set=_leg_set)


if __name__ == '__main__':
    variants.run(main, default='mtmc')
