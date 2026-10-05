"""
Materialize per-mode / per-profile cell-baked DISUTILITY ODMs.

Reads 08's cell-baked gross-time ODMs (road + transit) and applies
09a's calibrated utility spec to produce disutility-domain ODMs —
smaller = better, same semantics as time. `build_disutility_spec`
sign-flips at load time so downstream (10) can consume these as if
they were time costs (nearest picks smallest, gravity uses `exp(-D)`,
etc.).

One ODM per (mode, profile, utility-variant) combination is written.
Utility variants are discovered from the scenario's `coefs` dict —
every key of the form `utility_<variant>` triggers one utility ODM
build per profile.

Formula per OD pair (gross_sec = the corresponding cell in the
underlying gross-time ODM):

    D(i, j) = ASC_effective                    (built into `spec.disutility.constant`)
            + β_time_effective · gross_sec     (element-wise in this fn)
            + Σ_o β_o · feat_o(i) + Σ_d β_d · feat_d(j)   (aperta's add_endpoint_utility)
            + β_time_log · log(gross_sec)      (apply_log_time_utility)

Every β above is sign-flipped from 09a's raw fit — see
`build_disutility_spec`.

Inputs (PUBLIC, under `<scenario>/`):
    odm/<mode>_time_gross_<profile>.npz     # from 08a (road)
    odm/<mode>_geo_pairs.npz                # from 08a (road; shared across profiles)
    odm/transit_time_gross_npvm.npz         # from 08a (transit)
    odm/transit_geo_pairs.npz               # from 08a (transit)
    coefs/<kind>/utility_<variant>.csv        # β table from 09a (rows: b_*, asc_*;
                                              # cols: value, rob_std_err,
                                              # rob_t_test, rob_p_value)
    coefs/<kind>/utility_<variant>_stats.csv  # companion sidecar: sd_avg_<sd_col>
                                              # + t_cut_min_<mode> rows (two-col
                                              # `key, value`)
    properties/nodes_walk_extended.csv,
      nodes_bike_extended.csv,
      nodes_car_extended.csv,
      nodes_car_flows_avg.csv               # per-node endpoint features

Output (PUBLIC, under `<scenario>/`):
    odm/<mode>_util_<profile>_<variant>.npz     # per-profile disutility ODM (road)
    odm/transit_util_npvm_<variant>.npz         # per-variant disutility ODM (transit)

Run:
    python -m main.09b_od_utilities --scenario <name>
"""

import logging
from typing import cast

from aperta import NOTE as _NOTE_LEVEL
from aperta.od_pairs import TieredODGeoPairs
from aperta_atlas.context import init_context
from aperta_atlas.utils import step

from main.common import (
    build_disutility_odm,
    build_disutility_spec,
    check_util_matches_cost_shape,
    coefs_for_unrun_profiles,
    extract_sd_averages_from_coefs,
    extract_t_cut_min_from_coefs,
    join_util_node_features,
    profile_to_util_mode,
)
from scenarios import get_scenario


def _utility_variants_from_scenario(scenario) -> list[str]:
    """Discover utility variants from the scenario's coefs dict —
    every `utility_<name>` key is one variant name to build. Excludes
    `utility_<name>_stats` companion sidecars (loaded separately)."""
    return sorted(
        k[len('utility_'):]
        for k in scenario.coefs.keys()
        if k.startswith('utility_') and not k.endswith('_stats')
    )


def main():
    context = init_context()
    scenario = get_scenario(context.scenario)

    utility_variants = _utility_variants_from_scenario(scenario)
    if not utility_variants:
        logging.warning(
            "no `utility_*` coefs declared in scenario.coefs — nothing "
            "for 09b to do. Add e.g. `'utility_default': Calibrate()` "
            "to build the default variant.")
        context.close()
        return
    logging.info(f"utility variants to build: {utility_variants}")

    # Cache per-variant coefs once — otherwise the inner loop reloads
    # each CSV per (mode, profile) combination.
    util_coefs_by_variant = {
        v: context.get_coefs(f'utility_{v}') for v in utility_variants
    }
    # Track every coef key consumed by SOME spec build (across all road
    # profiles + transit) per variant. Checked at the end — anything
    # residual is a coef 09a fit that 09b silently ignored (naming-scheme
    # drift, disabled mode, or feature not wired into build_disutility_spec).
    consumed_by_variant: dict[str, set[str]] = {v: set() for v in utility_variants}

    # ---- Prepare cells + zones with endpoint feature columns --------
    with step('load cells + zones + endpoint features'):
        prop_cols = ['population', 'snap', 'employment']
        cells = context.get_properties('cells', prop_cols, add_shapes=True)
        zones = context.get_properties('zones', prop_cols, add_shapes=True)
        # No `is_active` filter — utility ODMs must mirror the gross-
        # time ODM shape, which covers all cells that appear as
        # destinations (not just active origins). Filtering here would
        # NaN-poison every inactive-dest utility.
        cells_m = cells.copy()
        zones_m = zones.copy()
        # Endpoint features are joined per-feature from each feature's
        # source graph (density from walk, bike_infra from bike, etc.).
        # Cells missing the source-mode snap → NaN feature → NaN utility
        # on any pair involving them.
        join_util_node_features(cells_m, zones_m, context)
        cells_m['unit_id'] = cells_m.index  # `add_endpoint_utility` uses this
        zones_m['unit_id'] = zones_m.index
        n_active = int((cells['is_active'] == 1).sum())
        logging.info(
            f"  → {len(cells_m):,} cells ({n_active:,} active) "
            f"+ {len(zones_m):,} zones")

    with step('load utility stats sidecars (SD averages + t_cut_min)'):
        # `utility_<v>_stats` companions travel with each `utility_<v>`
        # under the same CoefSource declaration (validated in
        # Scenario.__post_init__). ImportFrom copies both files
        # automatically via `coefs.resolve` in 09a.
        stats_by_variant = {
            v: context.get_coefs(f'utility_{v}_stats') for v in utility_variants
        }
        sd_averages_by_variant = {
            v: extract_sd_averages_from_coefs(stats_by_variant[v])
            for v in utility_variants
        }
        t_cut_min_by_variant = {
            v: extract_t_cut_min_from_coefs(stats_by_variant[v]) or None
            for v in utility_variants
        }
        v0 = utility_variants[0]
        n_sd = len(sd_averages_by_variant[v0])
        t_cut = t_cut_min_by_variant[v0]
        logging.info(f"  → utility_{v0}_stats: {n_sd} sd_avg row(s); "
                     f"t_cut_min p95 (min) = "
                     f"{ {m: round(v, 1) for m, v in (t_cut or {}).items()} }")
        if n_sd == 0:
            logging.warning(
                "  ⚠ utility stats have no sd_avg_* rows — SD correction "
                "disabled (ASC represents reference-category person, NOT "
                "population-average). Re-run 09a to bake sd_avg in.")
        if t_cut is None:
            logging.warning(
                "  ⚠ utility stats have no t_cut_min_* rows — log-time "
                "extrapolation falls back to pure log. Re-run 09a with "
                "survey_summary.csv present to bake t_cut_min in.")

    # ---- Road modes: build disutility ODM per profile per variant ---
    for case in scenario.mode_configs.values():
        mode = case.mode
        for profile in case.profiles:
            src = case.source_for(profile)
            label = src.name
            util_mode = profile_to_util_mode(label)
            if util_mode is None:
                logging.info(
                    f"  → mode={mode} profile={label}: no util spec "
                    f"(profile has no 09a utility mode). Skipping.")
                continue

            # Gross cost ODM + geo pair index (dest ids). Both shared
            # across variants. Aperta's endpoint helpers need the pair
            # index (dest ids), not the cost ODM (values).
            gross_name = f'time_gross_{src.name}'
            with step(f'mode={mode} profile={label}: load gross ODM + geo pairs'):
                costs_gross = cast(TieredODGeoPairs, context.get_tiered_odm(
                    network_name=mode, data_name=gross_name))
                pairs_geo = cast(TieredODGeoPairs, context.get_tiered_odm(
                    network_name=mode, data_name='geo_pairs'))

            for variant in utility_variants:
                spec = build_disutility_spec(
                    util_coefs_by_variant[variant], label,
                    sd_averages=sd_averages_by_variant[variant],
                    t_cut_min_per_mode=t_cut_min_by_variant[variant],
                    consumed_out=consumed_by_variant[variant],
                )
                if spec is None:
                    logging.info(f"  → mode={mode} profile={label} variant={variant!r}: spec unavailable (skipping)")
                    continue

                with step(f'mode={mode} profile={label} variant={variant!r}: build disutility'):
                    util_geo = build_disutility_odm(
                        cost_odm_sec=costs_gross,
                        pairs_geo=pairs_geo,
                        spec=spec,
                        cells_m=cells_m, zones_m=zones_m,
                    )
                    check_util_matches_cost_shape(
                        cost_odm=costs_gross, util_odm=util_geo,
                        label=f'{mode}/{src.name}/{variant}',
                    )
                    context.create_tiered_odm(
                        util_geo,
                        network_name=mode,
                        data_name=f'util_{src.name}_{variant}',
                    )
                    logging.info(
                        f"  → disutility ODM saved "
                        f"({mode}_util_{src.name}_{variant}); "
                        f"β_time_eff={spec.disutility.cost_coefficient:+.5f}, "
                        f"β_time_log={spec.b_time_log:+.3f}, "
                        f"constant={spec.disutility.constant:+.3f}")

    # ---- Transit: same treatment as road ----------------------------
    with step('mode=transit: load gross ODM + geo pairs'):
        transit_costs = cast(TieredODGeoPairs, context.get_tiered_odm(
            network_name='transit', data_name='time_gross_npvm'))
        transit_pairs = cast(TieredODGeoPairs, context.get_tiered_odm(
            network_name='transit', data_name='geo_pairs'))

    for variant in utility_variants:
        spec = build_disutility_spec(
            util_coefs_by_variant[variant], 'transit',
            sd_averages=sd_averages_by_variant[variant],
            t_cut_min_per_mode=t_cut_min_by_variant[variant],
            consumed_out=consumed_by_variant[variant],
        )
        if spec is None:
            logging.warning(
                f"  ⚠ transit variant={variant!r}: no utility spec — "
                f"09a coefs don't include transit column. Skipping.")
            continue

        with step(f'mode=transit variant={variant!r}: build disutility'):
            transit_util = build_disutility_odm(
                cost_odm_sec=transit_costs,
                pairs_geo=transit_pairs,
                spec=spec,
                cells_m=cells_m, zones_m=zones_m,
            )
            check_util_matches_cost_shape(
                cost_odm=transit_costs, util_odm=transit_util,
                label=f'transit/{variant}',
            )
            context.create_tiered_odm(
                transit_util,
                network_name='transit',
                data_name=f'util_npvm_{variant}',
            )
            logging.info(
                f"  → transit disutility ODM saved "
                f"(transit_util_npvm_{variant}); "
                f"β_time_eff={spec.disutility.cost_coefficient:+.5f}, "
                f"β_time_log={spec.b_time_log:+.3f}, "
                f"constant={spec.disutility.constant:+.3f}")

    # ---- Per-variant residual check: any 09a coef no spec consumed? -
    with step('audit: 09a coefs not consumed by any 09b spec'):
        run_profile_labels = {
            case.source_for(p).name
            for case in scenario.mode_configs.values() for p in case.profiles}
        for variant in utility_variants:
            values = util_coefs_by_variant[variant]['value']
            all_beta_keys = {
                k for k in values.index
                if isinstance(k, str) and (k.startswith('b_') or k.startswith('asc_'))
            }
            residual = all_beta_keys - consumed_by_variant[variant]
            unrun = coefs_for_unrun_profiles(residual, run_profile_labels)
            drift = sorted(residual - unrun)
            if unrun:
                logging.log(
                    _NOTE_LEVEL,
                    f"  utility_{variant}: {len(unrun)} coef(s) belong to car profiles "
                    f"this scenario doesn't run — unused by design: {sorted(unrun)}")
            if drift:
                logging.warning(
                    f"  ⚠ utility_{variant}: {len(drift)} coef(s) NOT "
                    f"consumed by any spec — likely a naming-scheme drift, "
                    f"a disabled mode, or a feature 09b doesn't wire yet: "
                    f"{drift}")
            if not residual:
                logging.info(f"  → utility_{variant}: all "
                             f"{len(all_beta_keys)} b_*/asc_* coefs consumed.")

    context.close()


if __name__ == '__main__':
    main()
