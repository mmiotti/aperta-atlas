"""
Per-leg origin + destination overheads for every survey leg, per profile.

Sidecar to survey/05's `survey_leg_times.csv` — writes per-leg columns:

    t_overhead_orig_<profile>  # seconds, at leg's origin cell
    t_overhead_dest_<profile>  # seconds, at leg's destination cell
    t_routed_bias_<profile>    # seconds, α-scale-bias correction of routed time

Downstream 09a reconstructs gross per-leg time as `t_routed_<p> +
t_overhead_orig_<p> + t_overhead_dest_<p> − t_routed_bias_<p>` — matches
what main/08a bakes into the gross ODM at c2c/c2z tiers. Uses the same
helpers (`main.common.per_cell_{road,transit}_overheads` +
`route_time_alpha`) so numbers align.

`t_routed_bias_<p> = (1 − α) × t_routed_<p>` where α comes from the
`t_routed` row in `overheads_<road,transit>` for that profile.
Positive bias means the raw routed time overestimates (subtract to
correct); α = 1 → zero bias (no correction). For transit, α ≈ 0.86
in the unconstrained fit corrects long z2z trips that would otherwise
over-predict.

Legs with NaN `orig_cell_id` / `dest_cell_id` (foreign endpoints) get
NaN overhead on that side. NaN `t_routed_<p>` → NaN bias.

One variant per leg set (see survey/02d); file names carry the leg-set
suffix for the validation-only MOBIS sets.

Inputs (PRIVATE, under `<scenario>/`):
    generic/survey_legs[_<leg_set>].csv          # from survey/02d_prepare_survey_legs
    generic/survey_leg_times[_<leg_set>].csv     # from survey/05 (per-leg t_routed_<p>)

Inputs (PUBLIC, under `<scenario>/`):
    properties/cells_{population,employment,snap,pois,transit_access}.csv
    properties/nodes_<mode>_extended.csv         # per-mode density join
    shapes/cells.gpkg
    coefs/<kind>/overheads_road.csv              # from 07a
    coefs/<kind>/overheads_transit.csv           # from 07b

Output (PRIVATE, under `<scenario>/`):
    generic/survey_leg_overheads[_<leg_set>].csv # per-leg orig + dest overheads + bias

Run (default variant `mtmc`):
    python -m survey.08a_add_leg_overheads --scenario <name>
MOBIS validation leg set:
    python -m survey.08a_add_leg_overheads --scenario <name> --variant mobis_precovid
"""

import logging

import pandas as pd

from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants

from main.common import (
    SURVEY_LEG_SETS,
    per_cell_road_overheads,
    per_cell_transit_overhead,
    resolve_road_overhead_column,
    route_time_alpha,
    survey_file,
)
from mode_configs import TRANSIT_MODE_CONFIG
from scenarios import get_scenario, scenario_needs_survey_prep


# Per-node density column — kept in sync with 09a's `_DENSITY_COL`.
_DENSITY_COL = 'density_r1000_norm'


def _road_overhead_columns_for_leg(
    legs: pd.DataFrame,
    cells_m: pd.DataFrame,
    coefs: pd.DataFrame,
    profile_label: str,
    density_col_local: str,
    snap_dist_col: str,
    same_mode_profile_labels: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute per-cell (orig, dest) overhead for one road profile and
    map onto legs via `orig_cell_id` / `dest_cell_id`. NaN cell → NaN
    overhead. Mirrors what 08a bakes into the gross ODM so 09a fits the
    same decomposition 10 reads back. `same_mode_profile_labels` lets
    derived profiles (e.g. walk_prm) inherit the parent profile's
    overhead — see `resolve_road_overhead_column`.
    """
    overhead_label = resolve_road_overhead_column(
        profile_label, coefs.columns,
        same_mode_profile_labels=same_mode_profile_labels)
    if overhead_label != profile_label:
        logging.info(
            f"  → overhead for {profile_label!r} not in coefs; "
            f"falling back to {overhead_label!r}")
    orig_ov_per_cell, dest_ov_per_cell = per_cell_road_overheads(
        cells_m, coefs[overhead_label],
        density_col=density_col_local, snap_dist_col=snap_dist_col)
    orig_ov_leg = legs['orig_cell_id'].map(orig_ov_per_cell)
    dest_ov_leg = legs['dest_cell_id'].map(dest_ov_per_cell)
    return orig_ov_leg, dest_ov_leg


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
    legs = context.get_generic(survey_file('survey_legs.csv', leg_set), storage=Storage.PRIVATE)

    times_name = survey_file('survey_leg_times.csv', leg_set)
    with step(f'load {times_name} (for per-leg t_routed_<profile>)'):
        leg_times = context.get_generic(times_name, storage=Storage.PRIVATE)
        # Align to legs' index — legs share leg_id with routed after 02d.
        leg_times = leg_times.reindex(legs.index)
        logging.info(f"  → {len(leg_times):,} rows joined")

    with step('load cells (population, employment, snap, pois, transit_access)'):
        prop_cols = ['population', 'employment', 'snap', 'pois', 'transit_access']
        cells = context.get_properties('cells', prop_cols, add_shapes=True)
        logging.info(f"  → {len(cells):,} cells loaded")

    with step('load overheads_road coefs'):
        road_coefs = context.get_coefs('overheads_road')
        logging.info(f"  → profiles in overheads_road: {list(road_coefs.columns)}")

    with step('load overheads_transit coefs'):
        transit_coefs = context.get_coefs('overheads_transit')
        logging.info(f"  → transit coef rows: {list(transit_coefs.index)}")

    # Output frame: NaN-initialised, one column pair per profile filled in below.
    out = pd.DataFrame(index=legs.index)

    # ---- Road modes -------------------------------------------------------
    for mode_config in scenario.mode_configs.values():
        mode = mode_config.mode
        node_col = f'node_id_{mode}'

        with step(f'mode={mode}: join per-cell density (nodes_{mode}_extended)'):
            node_props = context.get_properties('nodes', f'{mode}_extended')
            cells_m = cells.copy()
            density_col_local = f'_density_{mode}'
            cells_m = cells_m.join(
                node_props[[_DENSITY_COL]].rename(
                    columns={_DENSITY_COL: density_col_local}),
                on=node_col)
            cells_m[density_col_local] = cells_m[density_col_local].fillna(0.0)

        same_mode = [mode_config.source_for(p).name for p in mode_config.profiles]
        for profile in mode_config.profiles:
            src = mode_config.source_for(profile)
            label = src.name
            with step(f'profile={label}: per-leg orig + dest overhead + routed bias'):
                orig_ov, dest_ov = _road_overhead_columns_for_leg(
                    legs, cells_m, road_coefs,
                    profile_label=label,
                    density_col_local=density_col_local,
                    snap_dist_col=f'distance_{mode}',
                    same_mode_profile_labels=same_mode,
                )
                out[f't_overhead_orig_{label}'] = orig_ov
                out[f't_overhead_dest_{label}'] = dest_ov
                # `t_routed_bias_<label>` = (1 − α) × t_routed. Uses the
                # same coefs column resolution as the overhead so derived
                # profiles inherit the parent's α.
                overhead_label = resolve_road_overhead_column(
                    label, road_coefs.columns,
                    same_mode_profile_labels=same_mode)
                alpha = route_time_alpha(road_coefs[overhead_label])
                routed_col = f't_routed_{label}'
                if routed_col in leg_times.columns:
                    out[f't_routed_bias_{label}'] = (
                        (1.0 - alpha) * leg_times[routed_col])
                else:
                    logging.warning(
                        f"  ⚠ {routed_col!r} missing from survey_leg_times.csv; "
                        f"t_routed_bias_{label} will be NaN.")
                    out[f't_routed_bias_{label}'] = float('nan')
                n_finite = int((orig_ov.notna() & dest_ov.notna()).sum())
                logging.info(
                    f"  → α = {alpha:.4f} (bias = (1−α) × t_routed); "
                    f"{n_finite:,}/{len(legs):,} legs with finite orig+dest "
                    f"overhead; median orig = {orig_ov.median():.1f} s, "
                    f"median dest = {dest_ov.median():.1f} s, "
                    f"median bias = {out[f't_routed_bias_{label}'].median():.1f} s")

    # ---- Transit ----------------------------------------------------------
    with step(f'profile={TRANSIT_MODE_CONFIG.mode}: per-cell + per-leg overhead + routed bias'):
        per_cell_ov = per_cell_transit_overhead(cells, transit_coefs)
        # Symmetric: same per-cell value at origin and destination sides.
        orig_ov = legs['orig_cell_id'].map(per_cell_ov)
        dest_ov = legs['dest_cell_id'].map(per_cell_ov)
        out[f't_overhead_orig_{TRANSIT_MODE_CONFIG.mode}'] = orig_ov
        out[f't_overhead_dest_{TRANSIT_MODE_CONFIG.mode}'] = dest_ov
        # α scaling: (1 − α) × t_routed_transit_z2z. α comes from the
        # `transit` column's `t_routed` row (unconstrained fit) — 1.0 if
        # only a constrained-α fit is available.
        alpha_transit = route_time_alpha(transit_coefs['transit'])
        routed_col_transit = 't_routed_transit_z2z'
        if routed_col_transit in leg_times.columns:
            out[f't_routed_bias_{TRANSIT_MODE_CONFIG.mode}'] = (
                (1.0 - alpha_transit) * leg_times[routed_col_transit])
        else:
            logging.warning(
                f"  ⚠ {routed_col_transit!r} missing from survey_leg_times.csv; "
                f"t_routed_bias_{TRANSIT_MODE_CONFIG.mode} will be NaN.")
            out[f't_routed_bias_{TRANSIT_MODE_CONFIG.mode}'] = float('nan')
        n_finite = int((orig_ov.notna() & dest_ov.notna()).sum())
        logging.info(
            f"  → α = {alpha_transit:.4f} (bias = (1−α) × t_routed_transit_z2z); "
            f"{n_finite:,}/{len(legs):,} legs with finite orig+dest transit "
            f"overhead; per-cell median = {per_cell_ov.median():.1f} s, "
            f"median bias = "
            f"{out[f't_routed_bias_{TRANSIT_MODE_CONFIG.mode}'].median():.1f} s")

    context.create_generic(
        out, survey_file('survey_leg_overheads.csv', leg_set), storage=Storage.PRIVATE)
    logging.info(f"  → wrote {len(out):,} legs × {len(out.columns)} overhead columns")
    context.close()


variants = Variants([('leg_set', str)])
for _leg_set in SURVEY_LEG_SETS:
    variants.add(name=_leg_set, leg_set=_leg_set)


if __name__ == '__main__':
    variants.run(main, default='mtmc')
