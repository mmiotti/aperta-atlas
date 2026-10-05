"""Shared helpers for the overhead + disutility + accessibility stages.

Families:
  - Accessibility grid: `bins_from_edges_min`, `gravity_decays_from_half_decay_min`.
  - Flatten / scale: `flatten_*_dest_columns`, `scaled_weights_for_nearest_k`.
  - Overhead: `resolve_road_overhead_column`, `per_cell_road_overheads`,
    `per_cell_transit_overhead` — baked into 08's gross ODMs.
  - Disutility spec: `profile_to_util_mode`, `build_disutility_spec`,
    `build_disutility_odm`, `apply_log_time_utility` — consumed by 09b/10.
  - Survey-leg helpers: `SURVEY_LEG_SETS`, `survey_file`, `npvm_transit_z2z_lookup`,
    `car_overhead_by_peak`.
"""
import logging
import math
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

from aperta import utility as _utility
from aperta.accessibility import Bin, Decay, exp_decay
from aperta.od_pairs import TieredODGeoPairs, TieredODPairs
from aperta.utility import Utility

# Preserves the concrete subclass (GeoPairs / NodePairs) in helpers'
# return annotations. Mirrors aperta's `floor_intrazonal_costs`.
_TOD = TypeVar('_TOD', bound=TieredODPairs)


# Leg sets the survey stages (02d → 05 → 08a) can prepare, each mapped to its source folders under
# `preparation/switzerland/surveys/`. 'mtmc' is the canonical set that 03a, 04, 07a/b, 08b/c and 09a
# read; the MOBIS sets are validation-only and get leg-set-suffixed files (`survey_file`). Keep the
# MOBIS cohorts apart: 'mobis_precovid' is normal traffic (and car's edge-weight training cohort,
# so car comparisons on it are in-sample); 'mobis_covid' (2020-wk13 to 2022-03) is an anomalous
# traffic regime, but the only cohort with confirmed regular-bike ('rbike') legs.
SURVEY_LEG_SETS: dict[str, tuple[str, ...]] = {
    'mtmc':           ('mzmv_2015', 'mzmv_2021'),
    'mobis_precovid': ('mobis_precovid',),
    'mobis_covid':    ('mobis_covid',),
}


def survey_file(name: str, leg_set: str) -> str:
    """Leg-set-specific survey file name: unchanged for 'mtmc' (`survey_legs.csv`), suffixed for
    the MOBIS sets (`survey_legs_mobis_precovid.csv`)."""
    if leg_set not in SURVEY_LEG_SETS:
        raise ValueError(f"Unknown leg set {leg_set!r}; expected one of {list(SURVEY_LEG_SETS)}.")
    if leg_set == 'mtmc':
        return name
    stem, ext = name.rsplit('.', 1)
    return f'{stem}_{leg_set}.{ext}'


# Cells-frame columns that are NOT accessibility destinations.
NON_DESTINATION_COLS: frozenset[str] = frozenset({
    'n_pois', 'cell_id', 'zone_id', 'node_id', 'geometry', 'combined_total',
})


# Pop + employment counts overwhelm POI counts at a shared `k` grid; scale by
# 0.01 so `k=8` for `population_total` means "nearest 1,000 people" instead
# of "nearest 8". POI destinations stay unscaled.
NEAREST_K_SCALE_FACTOR: float = 0.01


def bins_from_edges_min(bin_edges_min: tuple[int, ...]) -> list[Bin]:
    """Convert an integer minute-edges tuple into half-open `Bin(lo_s, hi_s)`
    records. Empty tuple → []."""
    return [
        Bin(name=f'{lo}_{hi}min', lo=float(lo * 60), hi=float(hi * 60))
        for lo, hi in zip(bin_edges_min[:-1], bin_edges_min[1:])
    ]


def gravity_decays_from_half_decay_min(
    gravity_half_decay_min: tuple[int, ...],
) -> list[Decay]:
    """Convert half-decay minutes → `Decay` records with `β = ln(2)/(m·60)`.
    Empty tuple → []."""
    return [
        exp_decay(f'exp{m}min', beta=math.log(2) / (m * 60.0))
        for m in gravity_half_decay_min
    ]


def bins_from_edges_m(bin_edges_m: tuple[float, ...]) -> list[Bin]:
    """Convert a metres-edges tuple into half-open `Bin(lo_m, hi_m)` records.
    Empty tuple → []."""
    return [
        Bin(name=f'{int(lo)}_{int(hi)}m', lo=float(lo), hi=float(hi))
        for lo, hi in zip(bin_edges_m[:-1], bin_edges_m[1:])
    ]


def gravity_decays_from_half_decay_m(
    gravity_half_decay_m: tuple[float, ...],
) -> list[Decay]:
    """Convert half-decay metres → `Decay` records with `β = ln(2)/m`.
    Empty tuple → []."""
    return [
        exp_decay(f'exp{int(m)}m', beta=math.log(2) / m)
        for m in gravity_half_decay_m
    ]


# Max destination types materialised simultaneously as weight ODMs. Caps
# the weight-array memory footprint; per-profile cost ODMs are cached
# across batches so batching costs no redundant work.
DEST_BATCH_SIZE = 5


def flatten_bin_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(bin_name, property)` MultiIndex → `<property>_<bin_name>` flat.
    Also renames the index to `cell_id` for `create_properties`.
    """
    new_cols = [f'{prop}_{bin_name}' for bin_name, prop in result.columns]
    flat = pd.DataFrame(
        result.to_numpy(), index=result.index, columns=new_cols)
    flat.index.name = 'cell_id'
    return flat


def flatten_k_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(k, property)` MultiIndex → `<property>_k<int(k)>` flat columns."""
    new_cols = [f'{prop}_k{int(k)}' for k, prop in result.columns]
    flat = pd.DataFrame(
        result.to_numpy(), index=result.index, columns=new_cols)
    flat.index.name = 'cell_id'
    return flat


def flatten_decay_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(decay_name, property)` MultiIndex → `<property>_<decay_name>`."""
    new_cols = [f'{prop}_{decay}' for decay, prop in result.columns]
    flat = pd.DataFrame(
        result.to_numpy(), index=result.index, columns=new_cols)
    flat.index.name = 'cell_id'
    return flat


def scaled_weights_for_nearest_k(
    weights: dict[str, TieredODPairs],
    scaled_dests: set[str] | frozenset[str],
) -> dict[str, TieredODPairs]:
    """Return `weights` with `scaled_dests` per-tier arrays multiplied by
    `NEAREST_K_SCALE_FACTOR`. Unscaled destinations pass by reference."""
    def _scale_tier(tier):
        if tier is None:
            return None
        return {k: v * NEAREST_K_SCALE_FACTOR for k, v in tier.items()}

    out: dict[str, TieredODPairs] = {}
    for d, w in weights.items():
        if d in scaled_dests:
            out[d] = TieredODGeoPairs(
                cells_to_cells=_scale_tier(w.cells_to_cells),
                cells_to_zones=_scale_tier(w.cells_to_zones),
                zones_to_zones=_scale_tier(w.zones_to_zones),
            )
        else:
            out[d] = w
    return out


# ---------------------------------------------------------------------------
# Overhead helpers
# ---------------------------------------------------------------------------


def resolve_road_overhead_column(
    profile_label: str, coefs_columns,
    same_mode_profile_labels: list[str] | tuple[str, ...] = (),
) -> str:
    """Match a profile → column in `overheads_road` coefs. Falls back to the
    first `same_mode_profile_labels` sibling in the coefs (typically the
    parent calibrated profile). Raises `KeyError` on total miss.
    """
    if profile_label in coefs_columns:
        return profile_label
    for sibling in same_mode_profile_labels:
        if sibling != profile_label and sibling in coefs_columns:
            return sibling
    raise KeyError(
        f"profile {profile_label!r} missing from overheads_road coefs "
        f"and no same-mode sibling in {list(same_mode_profile_labels)!r} "
        f"matched either (available: {list(coefs_columns)})")


def route_time_alpha(coefs_for_profile: pd.Series) -> float:
    """Extract the fitted `t_routed` slope (α) from an overhead-coef series.
    Defaults to 1.0 when the row is absent (constrained-α fits, or older
    coefs). α ≠ 1 encodes a multiplicative bias correction on the routed
    time; downstream code applies it as `gross = t_routed + orig_ov +
    dest_ov − (1 − α) × t_routed` (or `gross = α × t_routed + …` for
    ODM-level assemblies). See `main/07a_road_overhead_coefs.py` and
    `main/07b_transit_overhead_coefs.py` for the fitting side.
    """
    return float(coefs_for_profile.get('t_routed', 1.0))


def _shared_or_split_coef(
    coefs_for_profile: pd.Series,
    shared_key: str, orig_key: str, dest_key: str,
) -> tuple[float, float]:
    """Prefer shared coef (`share_endpoint_coefs=True`); fall back to
    split orig/dest. Missing everywhere → (0, 0)."""
    if shared_key in coefs_for_profile.index:
        shared = float(coefs_for_profile[shared_key])
        return shared, shared
    return (
        float(coefs_for_profile.get(orig_key, 0.0)),
        float(coefs_for_profile.get(dest_key, 0.0)),
    )


def per_cell_road_overheads(
    cells: pd.DataFrame, coefs_for_profile: pd.Series,
    density_col: str, snap_dist_col: str,
) -> tuple[pd.Series, pd.Series]:
    """Full per-cell road overhead (`orig`, `dest`) from 07's OLS fit,
    per side: `const/2 + density_coef · density + snap_dist_coef · snap_dist`.
    Baked into 08's gross ODM.
    """
    half_const = float(coefs_for_profile.get('const', 0.0)) / 2.0
    density_orig_coef, density_dest_coef = _shared_or_split_coef(
        coefs_for_profile, 'density', 'orig_density', 'dest_density')
    density = cells[density_col].fillna(0.0)

    if snap_dist_col in cells.columns:
        snap_orig_coef, snap_dest_coef = _shared_or_split_coef(
            coefs_for_profile, 'snap_dist',
            'orig_snap_dist', 'dest_snap_dist')
        snap_dist = cells[snap_dist_col].fillna(0.0)
        snap_orig = snap_orig_coef * snap_dist
        snap_dest = snap_dest_coef * snap_dist
    else:
        snap_orig = pd.Series(0.0, index=cells.index)
        snap_dest = pd.Series(0.0, index=cells.index)

    return (
        half_const + density_orig_coef * density + snap_orig,
        half_const + density_dest_coef * density + snap_dest,
    )


def per_cell_transit_overhead(
    cells: pd.DataFrame, transit_coefs: pd.DataFrame,
) -> pd.Series:
    """Per-cell transit overhead (seconds) = `const_s/2 + Σ coef × cells[column]`
    clipped to ±`cap_s`. Applied at both endpoints by 08a — the two half-consts
    sum to the fitted trip-level intercept.

    Reserved rows in `transit_coefs.index`:
      - `cap_s`: symmetric clipping bound (required).
      - `const_s`: trip-level intercept from 07b's constrained OLS
        (optional; default 0). Split in half so orig + dest sum to
        `const_s` at trip level.
    All other rows name columns in `cells` (typically stage 06's
    `*_zone_dev` features); NaN column values contribute 0.
    """
    if 'cap_s' not in transit_coefs.index:
        raise ValueError("overheads_transit coef missing required `cap_s` row.")
    cap_s = float(transit_coefs.loc['cap_s', 'transit'])
    if 'const_s' in transit_coefs.index:
        const_half = float(transit_coefs.loc['const_s', 'transit']) / 2.0
    else:
        const_half = 0.0
    # `t_routed` (z2z scaling) can appear in informational columns from
    # 07b's unconstrained fits, but the applied model uses α=1 (adds
    # overheads to raw z2z without scaling), so the row is skipped here.
    _RESERVED = ('cap_s', 'const_s', 't_routed')
    combined = pd.Series(const_half, index=cells.index)
    for param, row in transit_coefs.iterrows():
        if param in _RESERVED:
            continue
        if param not in cells.columns:
            raise ValueError(f"overheads_transit references column {param!r} not in "
                             f"cells (available zone-dev: "
                             f"{sorted(c for c in cells.columns if 'zone_dev' in c)})")
        combined = combined + float(row['transit']) * cells[param].fillna(0.0)
    return combined.clip(lower=-cap_s, upper=cap_s)


# ---------------------------------------------------------------------------
# Disutility spec helpers
# ---------------------------------------------------------------------------

# Routing profile → 09a utility-mode label. `None` → no 09a spec exists;
# 09b skips and 10 falls back to the time grid.
_PROFILE_TO_UTIL_MODE: dict[str, str | None] = {
    'rwalk':     'walk',
    'walk_prm':  None,          # not modelled in 09a
    'rbike':     'rbike',
    'ebike25':   'ebike25',     # only if 09a ran with ebikes not excluded
    'ebike45':   None,          # not modelled in 09a (too few observations)
    'car_base':  'car',
    'car_peak':  'car',
    'car_night': 'car',
    'transit':   'transit',
}

# profile_label → (asc_modifier_coef, b_time_modifier_coef). `build_disutility_spec`
# applies whichever coefs 09a's `car_peak_night` variant produced.
_CAR_HOUR_MODIFIER: dict[str, tuple[str, str]] = {
    'car_peak':  ('asc_car_peak',  'b_time_car_peak'),
    'car_night': ('asc_car_night', 'b_time_car_night'),
}


def coefs_for_unrun_profiles(keys, run_profile_labels) -> set[str]:
    """Those of `keys` that belong to a car hour profile not in `run_profile_labels`: its
    asc / b_time modifiers and `b_car_<hour>_x_*` interactions (same rule as
    `build_disutility_spec`). Unused by design when a scenario drops that profile."""
    keys = {k for k in keys if isinstance(k, str)}
    out: set[str] = set()
    for label, modifier_keys in _CAR_HOUR_MODIFIER.items():
        if label in run_profile_labels:
            continue
        hour = label.split('_')[-1]
        out |= set(modifier_keys) | {k for k in keys if k.startswith(f'b_car_{hour}_x_')}
    return out & keys

# Endpoint point-feature sources (single source of truth in main.utility_config).
from main.utility_config import FEATURE_SCALE, POINT_FEATURE_SOURCES


@dataclass(frozen=True)
class DisutilitySpec:
    """Parsed disutility spec for one routing profile. All β and the ASC
    are SIGN-FLIPPED (09a fits utility; disutility = −utility) so that
    downstream smaller = better, matching time semantics — `nearest_k`,
    `floor_intrazonal_costs`, and `exp(-β·D)` gravity all work directly.

    `b_time_log` sits outside aperta's `Utility` because `log(cost)` isn't
    per-edge aggregatable; applied as an OD post-step by
    `apply_log_time_utility`. `t_cut_min` (optional, minutes) triggers a
    C¹ linear extrapolation of the log term beyond the survey's data-rich
    region — see `apply_log_time_utility` for the piecewise formula.
    `sd_correction` is exposed for audit; already folded into
    `disutility.constant`.
    """
    disutility: Utility
    b_time_log: float
    profile_label: str
    util_mode: str
    sd_correction: float = 0.0
    t_cut_min: float | None = None


def profile_to_util_mode(profile_label: str) -> str | None:
    """09a utility mode for this routing profile; `None` if not modelled."""
    if profile_label not in _PROFILE_TO_UTIL_MODE:
        raise KeyError(f"Unknown routing profile {profile_label!r} — add an entry "
                       f"to `_PROFILE_TO_UTIL_MODE`.")
    return _PROFILE_TO_UTIL_MODE[profile_label]


def compute_sd_averages(survey_legs: pd.DataFrame) -> dict[str, float]:
    """Person-weight-weighted `{sd_col: P(sd = 1)}` for every `sd_*` column
    in `survey_legs`. Shifts each mode's ASC to represent a
    "population-average person" rather than the arbitrary withheld base
    category. Requires `weight_person` (from survey/02d).
    """
    if 'weight_person' not in survey_legs.columns:
        raise ValueError("compute_sd_averages requires weight_person in survey_legs.")
    w = survey_legs['weight_person'].astype(float)
    w_total = float(w.sum())
    if w_total <= 0:
        raise ValueError("compute_sd_averages: Σ(weight_person) is not positive.")
    averages: dict[str, float] = {}
    for sd in [c for c in survey_legs.columns if c.startswith('sd_')]:
        try:
            col = survey_legs[sd].astype(float)
        except (ValueError, TypeError):
            continue
        averages[sd] = float((w * col).sum() / w_total)
    return averages


def load_util_cutoffs_from_summary(
    summary: pd.DataFrame,
    metric: str = 't_gross_min',
    percentile: str = 'p95',
) -> dict[str, float]:
    """Per-mode `{mode: t_cut_min}` from survey/08b's `survey_summary.csv`,
    for `build_disutility_spec(..., t_cut_min_per_mode=)`. Beyond `t_cut_min`
    the log-time term switches to linear extrapolation.
    """
    sub = summary[summary['metric'] == metric]
    if percentile not in sub.columns:
        raise KeyError(f"percentile {percentile!r} not in survey_summary "
                       f"(have: {list(sub.columns)}).")
    return dict(zip(sub.index, sub[percentile].astype(float)))


# ---------------------------------------------------------------------------
# Extractors — parse the companion sidecar (`utility_<variant>_stats.csv`)
# into the dicts `build_disutility_spec` consumes. The sidecar is produced
# by 09a's `_compute_utility_stats` and copied on ImportFrom via
# `coefs.resolve`. Prefixes are the source of truth.

def extract_sd_averages_from_coefs(stats: pd.DataFrame) -> dict[str, float]:
    """Pull `sd_avg_<sd_col>` rows from the utility stats sidecar into a
    `{sd_col: mean}` dict. Consumed by `build_disutility_spec` to fold
    the population-reference correction into the ASC."""
    values = stats['value']
    prefix = 'sd_avg_'
    return {
        k[len(prefix):]: float(values[k])
        for k in values.index
        if isinstance(k, str) and k.startswith(prefix)
    }


def extract_t_cut_min_from_coefs(stats: pd.DataFrame) -> dict[str, float]:
    """Pull `t_cut_min_<mode>` rows from the utility stats sidecar into
    a `{mode: t_cut_min}` dict. Empty dict if none present (09a wrote
    without a `survey_summary.csv`); downstream should treat that as
    "no log-time cutoff — pure log"."""
    values = stats['value']
    prefix = 't_cut_min_'
    return {
        k[len(prefix):]: float(values[k])
        for k in values.index
        if isinstance(k, str) and k.startswith(prefix)
    }


def build_disutility_spec(
    coefs: pd.DataFrame, profile_label: str,
    sd_averages: dict[str, float] | None = None,
    t_cut_min_per_mode: dict[str, float] | None = None,
    consumed_out: set[str] | None = None,
) -> DisutilitySpec | None:
    """`DisutilitySpec` for one routing profile from 09a's coefs.

    Steps:
      1. ASC / constant (walk is base at 0).
      2. Linear + log-time coefs (per-minute in 09a's calibration).
      3. Car peak/night modifier: apply whichever of `asc_car_{peak,night}`
         (ASC shift) or `b_time_car_{peak,night}` (time-coef additive) 09a
         fit. Feature-interaction peak/night coefs are flagged but ignored
         (not yet wired downstream).
      4. Endpoint-feature coefs classified by `_orig`/`_dest`/`_od` suffix
         + SD correction folded into ASC (via `sd_averages` dict, read
         from the companion `utility_<variant>_stats` sidecar).
      5. Unit scaling min → sec (absorbs a `-b_time_log · log(60)` offset
         into the ASC).
      6. Sign-flip utility → disutility (see class docstring).

    Route features (`b_<feat>_route_<mode>`) are ignored — the cell-baked
    ODM has no per-path aggregation surface.

    Returns `None` if the profile has no 09a util mode, or if the
    referenced mode's ASC is missing from the coefs.

    `consumed_out`: caller-supplied set that receives every coef key the
    builder touched (accepted, unclassified-but-warned, or explicitly
    flagged-and-ignored). Lets the caller detect coefs that no spec
    consumed — see 09b's per-variant residual check.
    """
    util_mode = profile_to_util_mode(profile_label)
    if util_mode is None:
        return None

    values = coefs['value']

    if util_mode == 'walk':
        constant = 0.0
    else:
        asc_key = f'asc_{util_mode}'
        if asc_key not in values.index:
            logging.warning(f"  → build_disutility_spec({profile_label!r}): "
                            f"{asc_key!r} not in coefs — skipping.")
            return None
        constant = float(values[asc_key])
        if consumed_out is not None:
            consumed_out.add(asc_key)

    b_time_key = f'b_time_{util_mode}'
    if b_time_key not in values.index:
        logging.warning(f"  → build_disutility_spec({profile_label!r}): required "
                        f"{b_time_key!r} not in coefs — skipping.")
        return None
    cost_coefficient = float(values[b_time_key])
    if consumed_out is not None:
        consumed_out.add(b_time_key)

    # Apply whichever asc/b_time car peak/night modifiers 09a fit; flag
    # feature-interaction coefs (not yet wired downstream).
    if profile_label in _CAR_HOUR_MODIFIER:
        asc_mod_key, b_time_mod_key = _CAR_HOUR_MODIFIER[profile_label]
        applied = []
        if asc_mod_key in values.index:
            constant += float(values[asc_mod_key])
            applied.append(asc_mod_key)
        if b_time_mod_key in values.index:
            cost_coefficient += float(values[b_time_mod_key])
            applied.append(b_time_mod_key)
        hour = profile_label.split('_')[-1]  # 'peak' or 'night'
        feature_keys = [k for k in values.index
                        if k.startswith(f'b_car_{hour}_x_')]
        if feature_keys:
            logging.warning(f"  → build_disutility_spec({profile_label!r}): "
                            f"feature-interaction peak/night coefs {feature_keys} "
                            f"IGNORED (not yet supported downstream).")
        if not applied and not feature_keys:
            logging.warning(f"  → build_disutility_spec({profile_label!r}): no "
                            f"peak/night modifier coefs. Skipping.")
            return None
        if not applied and feature_keys:
            return None
        # Record feature-interaction keys so the tail-classifier doesn't
        # re-warn them as unclassified.
        consumed_carhour_keys = set(applied) | set(feature_keys)
        if consumed_out is not None:
            consumed_out.update(consumed_carhour_keys)
    else:
        consumed_carhour_keys = set()

    b_time_log_key = f'b_time_log_{util_mode}'
    if b_time_log_key not in values.index:
        logging.warning(f"  → build_disutility_spec({profile_label!r}): "
                        f"{b_time_log_key!r} not in coefs — assuming β_time_log = 0.")
    b_time_log = float(values.get(b_time_log_key, 0.0))
    if consumed_out is not None and b_time_log_key in values.index:
        consumed_out.add(b_time_log_key)

    # Classify remaining `b_*_<util_mode>` coefs by suffix:
    #   `_orig` / `_dest` → per-side; `_od` → both sides (mirrors 09a's
    #   `endpoints_combined`); `b_sd_*` → SD correction. Unrecognised
    #   patterns land in `unclassified` and warn.
    origin_features: dict[str, float] = {}
    destination_features: dict[str, float] = {}
    sd_correction = 0.0
    unclassified: list[str] = []
    consumed_time_keys = {b_time_key}
    if b_time_log_key in values.index:
        consumed_time_keys.add(b_time_log_key)

    mode_suffix = f'_{util_mode}'
    for key in values.index:
        if not isinstance(key, str) or not key.startswith('b_'):
            continue
        if not key.endswith(mode_suffix):
            continue
        if key in consumed_time_keys or key in consumed_carhour_keys:
            continue
        # Strip 'b_' prefix and '_<util_mode>' suffix → the "core" part.
        core = key[2:-len(mode_suffix)]
        if core.startswith('sd_'):
            if sd_averages is not None:
                sd_correction += float(values[key]) * sd_averages.get(core, 0.0)
        elif core.endswith('_orig'):
            origin_features[core[:-len('_orig')]] = float(values[key])
        elif core.endswith('_dest'):
            destination_features[core[:-len('_dest')]] = float(values[key])
        elif core.endswith('_od'):
            # Both sides get β so `add_endpoint_utility`'s sum reproduces
            # `β · (orig + dest)`.
            feat = core[:-len('_od')]
            beta = float(values[key])
            origin_features[feat] = beta
            destination_features[feat] = beta
        else:
            unclassified.append(key)
        if consumed_out is not None:
            consumed_out.add(key)

    if unclassified:
        logging.warning(f"  → build_disutility_spec({profile_label!r}): "
                        f"{len(unclassified)} unconsumed coef(s) for {util_mode!r}: "
                        f"{sorted(unclassified)}.")

    constant += sd_correction

    # 09a fit on minutes, aperta routes on seconds:
    #   β_time_eff = β_time / 60;   ASC_eff = ASC − β_time_log · log(60).
    cost_coefficient = cost_coefficient / 60.0
    constant = constant - b_time_log * math.log(60.0)

    # utility → disutility: negate all coefs so smaller = better and
    # `nearest_k` / `floor_intrazonal_costs` / `exp(-β·D)` apply directly.
    constant = -constant
    cost_coefficient = -cost_coefficient
    origin_features = {k: -v for k, v in origin_features.items()}
    destination_features = {k: -v for k, v in destination_features.items()}
    b_time_log = -b_time_log

    disutility = Utility(
        constant=constant,
        cost_coefficient=cost_coefficient,
        origin_features=origin_features,
        destination_features=destination_features,
    )
    t_cut_min: float | None = None
    if t_cut_min_per_mode is not None:
        t_cut_min = t_cut_min_per_mode.get(util_mode)
        if t_cut_min is None:
            logging.warning(f"  → build_disutility_spec({profile_label!r}): no "
                            f"t_cut_min entry for {util_mode!r} — using pure log.")

    return DisutilitySpec(
        disutility=disutility,
        b_time_log=b_time_log,
        profile_label=profile_label,
        util_mode=util_mode,
        sd_correction=sd_correction,
        t_cut_min=t_cut_min,
    )


def apply_log_time_utility(
    full_u: _TOD,
    cost_odm: TieredODPairs,
    b_time_log: float,
    t_cut_sec: float | None = None,
) -> _TOD:
    """Add the log-time utility contribution per OD pair.

    Standard: `contribution = b_time_log × log(cost)`.
    With `t_cut_sec` set: piecewise (log below, linear above) preserving
    value + slope at the switchover — combined with `β_time · t` the full
    utility is C¹-continuous. Purpose: keep the log term from dominating
    in the survey's data-thin long-time tail.

    `cost_odm` is in SECONDS; the log(60) offset from 09a's minute→sec
    conversion is already absorbed into `constant` in `build_disutility_spec`.
    """
    if b_time_log == 0.0:
        return full_u

    log_t_cut = math.log(t_cut_sec) if t_cut_sec is not None else 0.0

    def _log_contrib(c_arr: np.ndarray) -> np.ndarray:
        """Piecewise log contribution. Non-finite inputs (incl. `cost = 0` →
        `-∞`) clamp to 0, matching `route_utility`'s empty-path handling."""
        with np.errstate(divide='ignore', invalid='ignore'):
            log_c = np.log(c_arr)
        log_c = np.where(np.isfinite(log_c), log_c, 0.0)
        if t_cut_sec is not None and t_cut_sec > 0:
            above = c_arr > t_cut_sec
            log_c = np.where(
                above,
                log_t_cut + (c_arr - t_cut_sec) / t_cut_sec,
                log_c,
            )
        return log_c

    def _add_tier(u_tier: dict | None, c_tier: dict | None) -> dict | None:
        if u_tier is None or c_tier is None:
            return None
        out = {}
        for origin, u_arr in u_tier.items():
            c_arr = c_tier[origin]
            log_c = _log_contrib(c_arr)
            out[origin] = (u_arr + b_time_log * log_c).astype(
                u_arr.dtype, copy=False)
        return out

    return type(full_u)(
        cells_to_cells=_add_tier(
            full_u.cells_to_cells, cost_odm.cells_to_cells),
        cells_to_zones=_add_tier(
            full_u.cells_to_zones, cost_odm.cells_to_zones),
        zones_to_zones=_add_tier(
            full_u.zones_to_zones, cost_odm.zones_to_zones),
    )


def join_util_node_features(
    cells_m: pd.DataFrame, zones_m: pd.DataFrame, context,
) -> None:
    """Join per-node point features onto `cells_m`/`zones_m` in place.
    Each feature from its source graph (per `POINT_FEATURE_SOURCES`), via
    `node_id_<source_mode>`; missing features silently skipped. Applies
    `FEATURE_SCALE` matching 09a's fit-time scaling.
    """
    files_cache: dict[str, pd.DataFrame] = {}
    for feat, (source_mode, data_name) in POINT_FEATURE_SOURCES.items():
        if data_name not in files_cache:
            files_cache[data_name] = context.get_properties('nodes', data_name)
        node_props = files_cache[data_name]
        if feat not in node_props.columns:
            continue
        scale = FEATURE_SCALE.get(feat, 1.0)
        s = node_props[feat] / scale
        node_col = f'node_id_{source_mode}'
        cells_m[feat] = cells_m[node_col].map(s)
        zones_m[feat] = zones_m[node_col].map(s)


def build_disutility_odm(
    cost_odm_sec: _TOD,
    pairs_geo: TieredODPairs,
    spec: 'DisutilitySpec',
    cells_m: pd.DataFrame, zones_m: pd.DataFrame,
    unit_col: str = 'unit_id',
) -> _TOD:
    """Assemble one profile's disutility ODM from a pre-computed time-cost
    ODM. No re-routing; the cell-baked regime has no per-path aggregation
    surface. Per OD pair:

        U(i, j) = ASC + β_time · cost + Σ β_endpoint · feature
                + β_time_log · log(cost)

    `unit_col` (default `'unit_id'`) is the cells/zones column matching
    the geo-keyed pairs' origin/dest IDs — used as `node_column` for
    aperta's endpoint lookup.
    """
    # Base: cost_coefficient × cost. Non-finite costs propagate as NaN.
    coef = spec.disutility.cost_coefficient
    def _scale_tier(tier: dict | None) -> dict | None:
        if tier is None:
            return None
        out = {}
        for origin, arr in tier.items():
            cost_finite = np.where(np.isfinite(arr), arr, np.nan)
            out[origin] = (coef * cost_finite).astype(arr.dtype, copy=False)
        return out
    base_u = type(cost_odm_sec)(
        cells_to_cells=_scale_tier(cost_odm_sec.cells_to_cells),
        cells_to_zones=_scale_tier(cost_odm_sec.cells_to_zones),
        zones_to_zones=_scale_tier(cost_odm_sec.zones_to_zones),
    )
    full_u = _utility.add_endpoint_utility(
        base_u, pairs_geo, spec.disutility,
        cells=cells_m, zones=zones_m, node_column=unit_col,
    )
    if spec.b_time_log != 0.0:
        t_cut_sec = spec.t_cut_min * 60.0 if spec.t_cut_min is not None else None
        full_u = apply_log_time_utility(
            full_u, cost_odm_sec, spec.b_time_log, t_cut_sec=t_cut_sec,
        )
    return full_u


def check_util_matches_cost_shape(
    cost_odm: 'TieredODPairs',
    util_odm: 'TieredODPairs',
    label: str,
) -> None:
    """Warn per tier when `util_odm` has more NaNs than `cost_odm` —
    typically a network-buffer / feature-source-mode mismatch (a feature
    isn't populated for cells outside its source graph's buffer).
    """
    for tier_name in ('cells_to_cells', 'cells_to_zones', 'zones_to_zones'):
        cost_tier = getattr(cost_odm, tier_name)
        util_tier = getattr(util_odm, tier_name)
        if cost_tier is None and util_tier is None:
            continue
        if cost_tier is None or util_tier is None:
            logging.warning(f"  ⚠ {label}[{tier_name}]: cost/util tier presence mismatch "
                            f"(cost={'set' if cost_tier else 'None'}, "
                            f"util={'set' if util_tier else 'None'}).")
            continue
        cost_nans = 0
        util_extra_nans = 0
        cost_finite_total = 0
        for origin, cost_arr in cost_tier.items():
            util_arr = util_tier.get(origin)
            cost_nan_mask = ~np.isfinite(np.asarray(cost_arr))
            if util_arr is None:
                util_nan_mask = np.ones_like(cost_nan_mask)
            else:
                util_nan_mask = ~np.isfinite(np.asarray(util_arr))
            cost_nans += int(cost_nan_mask.sum())
            cost_finite_total += int((~cost_nan_mask).sum())
            util_extra_nans += int((util_nan_mask & ~cost_nan_mask).sum())
        if util_extra_nans > 0:
            pct = 100 * util_extra_nans / max(cost_finite_total, 1)
            logging.warning(f"  ⚠ {label}[{tier_name}]: {util_extra_nans:,} of "
                            f"{cost_finite_total:,} time-finite pairs ({pct:.1f}%) "
                            f"came out NaN in utility. Likely feature-source / "
                            f"network-buffer mismatch (walk ⊂ bike ⊂ car buffer).")


# ---------------------------------------------------------------------------
# Survey-leg helpers
# ---------------------------------------------------------------------------


def npvm_transit_z2z_lookup(context) -> dict[tuple, float]:
    """Flatten NPVM z2z travel-time ODM to `{(orig_zone, dest_zone): sec}`."""
    npvm = context.source('preparation/switzerland/npvm')
    idx = npvm.get_tiered_odm('npvm_2023_transit', 'idx')
    times = npvm.get_tiered_odm('npvm_2023_transit', 'travel_time')
    z2z_idx = idx.zones_to_zones or {}
    z2z_times = times.zones_to_zones or {}
    lookup: dict[tuple, float] = {}
    for orig, dests in z2z_idx.items():
        vals = z2z_times[orig]
        for dest, val in zip(dests, vals):
            lookup[(orig, dest)] = float(val)
    return lookup


def car_overhead_by_peak(df: pd.DataFrame, side: str) -> pd.Series:
    """Per-leg car overhead (seconds) at `side` ∈ {`orig`, `dest`}, picking
    `t_overhead_{side}_car_{peak,base,night}` by `peak_str`. Unrecognised
    `peak_str` → NaN."""
    col_map = {
        'peak':  f't_overhead_{side}_car_peak',
        'base':  f't_overhead_{side}_car_base',
        'night': f't_overhead_{side}_car_night',
    }
    ov = np.select(
        [df['peak_str'] == k for k in col_map],
        [df[c] for c in col_map.values()],
        default=np.nan,
    )
    return pd.Series(ov, index=df.index)


def car_routed_bias_by_peak(df: pd.DataFrame) -> pd.Series:
    """Per-leg car `t_routed_bias` (seconds), picking
    `t_routed_bias_car_{peak,base,night}` by `peak_str`. Parallel to
    `car_overhead_by_peak` — unrecognised `peak_str` → NaN. Consumers
    SUBTRACT this from gross: `gross = t_routed + orig_ov + dest_ov − bias`."""
    col_map = {
        'peak':  't_routed_bias_car_peak',
        'base':  't_routed_bias_car_base',
        'night': 't_routed_bias_car_night',
    }
    bias = np.select(
        [df['peak_str'] == k for k in col_map],
        [df[c] for c in col_map.values()],
        default=np.nan,
    )
    return pd.Series(bias, index=df.index)


# ---------------------------------------------------------------------------
# Trip-level survey aggregation (transit)
# ---------------------------------------------------------------------------

_TRANSIT_BIKE_MODES: frozenset[str] = frozenset({'rbike', 'ebike25', 'ebike45'})


def _classify_transit_access(mode_set: frozenset[str]) -> str:
    """Categorical access-mode label for a transit-containing trip.
    Priority: park_and_ride (has car) > bike_and_ride (has bike, no car)
    > walk_only (walk or nothing else) > mixed (rare — car + bike, etc.)."""
    non_transit = mode_set - {'transit'}
    if 'car' in non_transit:
        return 'park_and_ride'
    if non_transit & _TRANSIT_BIKE_MODES:
        return 'bike_and_ride'
    if non_transit <= {'walk'}:
        return 'walk_only'
    return 'mixed'


def aggregate_transit_trips(legs: pd.DataFrame) -> pd.DataFrame:
    """Filter `legs` to trips containing ≥1 transit leg, aggregate legs
    → one row per trip. Ordering within a trip is by `leg_id` (the
    DataFrame index); first leg's orig / last leg's dest define the
    trip's door-to-door endpoints.

    Shared by `survey/08b_trip_transit_times.py` (production per-trip
    predictions) and `main/07b_transit_overhead_coefs.py` (trip-
    level overhead fit) so both operate on identical trip aggregates.

    Requires `legs` to have `trip_id`, `time_measured`, `dist_measured`,
    `orig_x/y`, `dest_x/y`, `orig_cell_id`, `dest_cell_id`,
    `orig_zone_id`, `dest_zone_id`, `mode_simplified`, `peak_str`,
    `weight_person`. Envelope flags (`is_within_speed_envelope`,
    `is_within_detour_envelope`) are aggregated as `min` (0 if ANY
    leg fails) when present.

    Output columns:
        (index) trip_id
        trip_time_measured, trip_dist_measured, trip_dist_line
        trip_orig_{x,y,cell,zone}, trip_dest_{x,y,cell,zone}
        weight_person, peak_str
        n_legs_in_trip, n_transit_legs
        has_walk_access, has_car_access, has_bike_access (0/1)
        access_pattern (walk_only|park_and_ride|bike_and_ride|mixed)
        is_within_speed_envelope, is_within_detour_envelope (0/1, if present)
    """
    trip_has_transit = legs.groupby('trip_id')['mode_simplified'].apply(
        lambda s: 'transit' in set(s))
    keep_trips = trip_has_transit[trip_has_transit].index

    # Sort by (trip_id, leg_id) so first/last leg per trip is well-defined.
    sub = legs.loc[legs['trip_id'].isin(keep_trips)].reset_index()
    sub = sub.sort_values(['trip_id', 'leg_id']).set_index('leg_id')

    first_leg = sub.groupby('trip_id').head(1).set_index('trip_id')
    last_leg = sub.groupby('trip_id').tail(1).set_index('trip_id')

    agg = pd.DataFrame(index=first_leg.index)
    agg.index.name = 'trip_id'

    agg['trip_time_measured'] = sub.groupby('trip_id')['time_measured'].sum()
    agg['trip_dist_measured'] = sub.groupby('trip_id')['dist_measured'].sum()
    agg['trip_orig_x'] = first_leg['orig_x']
    agg['trip_orig_y'] = first_leg['orig_y']
    agg['trip_dest_x'] = last_leg['dest_x']
    agg['trip_dest_y'] = last_leg['dest_y']
    agg['trip_orig_cell'] = first_leg['orig_cell_id']
    agg['trip_dest_cell'] = last_leg['dest_cell_id']
    agg['trip_orig_zone'] = first_leg['orig_zone_id']
    agg['trip_dest_zone'] = last_leg['dest_zone_id']
    agg['weight_person'] = first_leg['weight_person']
    agg['peak_str'] = first_leg['peak_str']
    agg['n_legs_in_trip'] = sub.groupby('trip_id').size()
    agg['n_transit_legs'] = sub.groupby('trip_id').apply(
        lambda g: int((g['mode_simplified'] == 'transit').sum()),
        include_groups=False)

    mode_sets = sub.groupby('trip_id')['mode_simplified'].apply(
        lambda s: frozenset(s.dropna()))
    agg['has_walk_access'] = mode_sets.apply(lambda s: 'walk' in s).astype(int)
    agg['has_car_access'] = mode_sets.apply(lambda s: 'car' in s).astype(int)
    agg['has_bike_access'] = mode_sets.apply(
        lambda s: bool(s & _TRANSIT_BIKE_MODES)).astype(int)
    agg['access_pattern'] = mode_sets.apply(_classify_transit_access)

    agg['trip_dist_line'] = np.sqrt(
        (agg['trip_dest_x'] - agg['trip_orig_x']) ** 2
        + (agg['trip_dest_y'] - agg['trip_orig_y']) ** 2)

    for flag in ('is_within_speed_envelope', 'is_within_detour_envelope'):
        if flag in sub.columns:
            agg[flag] = sub.groupby('trip_id')[flag].min().astype('Int64')

    return agg


