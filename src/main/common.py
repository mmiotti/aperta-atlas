"""Shared helpers for the overhead + disutility + accessibility stages.

Families:
  - Accessibility grid: `bins_from_edges_{min,m}`, `gravity_decays_from_half_decay_{min,m}`,
    `gravity_decays_util`, `util_logsums`.
  - Flatten / scale: `flatten_*_dest_columns`, `scaled_weights_for_nearest_k`.
  - Overhead: `resolve_road_overhead_column`, `route_time_alpha`, `zone_weights`,
    `per_cell_road_overheads`, `per_cell_transit_overhead` — baked into 08a's
    gross ODMs.
  - Disutility spec: `profile_to_util_mode`, `build_disutility_spec`,
    `build_disutility_odm`, the piecewise-linear time term
    (`piecewise_time_coefs`, `piecewise_time_utility`, `time_utility`), the cross-modal
    hour shifts (`crossmodal_hour_shift*`) — consumed by 09b/10.
  - Survey-leg helpers: `SURVEY_LEG_SETS`, `survey_file`, `npvm_transit_z2z_lookup`,
    `car_overhead_by_peak`, `car_routed_bias_by_peak`.
  - Trip set: `survey_trips`, `trip_drop_cause`, `*_QUALITY_FLAGS`, the `TRIP_*` rules.
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
from main.utility_config import FEATURE_SCALE, POINT_FEATURE_SOURCES
from preparation.switzerland.private.surveys.common import (
    MIN_TRIP_GAP_S, SPEED_ENVELOPES_KMH, in_trip_order, report_intervals,
)
from preparation.switzerland.private.surveys.standardize import speed_envelope_sides

# Preserves the concrete subclass (GeoPairs / NodePairs) in helpers'
# return annotations. Mirrors aperta's `floor_intrazonal_costs`.
_TOD = TypeVar('_TOD', bound=TieredODPairs)


# Leg sets the survey stages (02d → 05 → 08a) can prepare, each mapped to its source folders under
# `preparation/switzerland/surveys/`. 'mtmc' holds the MTMC legs, read by 03a.
# `TRIP_SET` holds the same legs as door-to-door trips
# (`survey_trips`), read by 07a/b, 09a and the validation of the times, so that the overheads
# and the mode-choice model use the same trips. The MOBIS sets are validation-only and get leg-set-suffixed files
# (`survey_file`). Keep the MOBIS cohorts apart: 'mobis_precovid' is normal traffic (and car's
# edge-weight training cohort, so car comparisons on it are in-sample); 'mobis_covid' (2020-wk13 to
# 2022-03) is an anomalous traffic regime, but the only cohort with confirmed regular-bike
# ('rbike') legs.
SURVEY_LEG_SETS: dict[str, tuple[str, ...]] = {
    'mtmc':           ('mzmv_2015', 'mzmv_2021'),
    'mtmc_trips':     ('mzmv_2015', 'mzmv_2021'),
    'mobis_precovid': ('mobis_precovid',),
    'mobis_covid':    ('mobis_covid',),
}
TRIP_SET = 'mtmc_trips'
MOBIS_LEG_SETS: tuple[str, ...] = ('mobis_precovid', 'mobis_covid')


def survey_file(name: str, leg_set: str) -> str:
    """Leg-set-specific survey file name: unchanged for 'mtmc' (`survey_legs.csv`), with
    `survey_leg` → `survey_trip` for `TRIP_SET` (`survey_trips.csv`, `survey_trip_times.csv`),
    suffixed for the MOBIS sets (`survey_legs_mobis_precovid.csv`)."""
    if leg_set not in SURVEY_LEG_SETS:
        raise ValueError(f"Unknown leg set {leg_set!r}; expected one of {list(SURVEY_LEG_SETS)}.")
    if leg_set == 'mtmc':
        return name
    if leg_set == TRIP_SET:
        if 'survey_leg' not in name:
            raise ValueError(f"No trip-set name for {name!r}.")
        return name.replace('survey_leg', 'survey_trip')
    stem, ext = name.rsplit('.', 1)
    return f'{stem}_{leg_set}.{ext}'


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


def gravity_decays_util(betas: tuple[float, ...]) -> list[Decay]:
    """`exp(-β·D)` decays on disutility, named `exp<β>`; pair with `util_logsums`."""
    return [exp_decay(f'exp{b}', beta=b) for b in betas]


def util_logsums(raw: pd.DataFrame) -> pd.DataFrame:
    """Gravity on disutility → logsums `(1/β)·ln Σ w·exp(-β·D)`, in utility units.

    `raw` is `accessibility.gravity` output for `gravity_decays_util` (columns
    `(exp<β>, property)`). The 1/β makes each value the expected maximum utility of a logit
    destination choice with sensitivity β relative to mode choice, so all β share the units
    of the mode-choice utilities. A sum of 0 (nothing reachable) → NaN.
    """
    beta = raw.columns.get_level_values(0).str.removeprefix('exp').astype(float)
    return np.log(raw.where(raw > 0)) / beta.to_numpy()


# Max destination types materialised simultaneously as weight ODMs. Caps
# the weight-array memory footprint; per-profile cost ODMs are cached
# across batches so batching costs no redundant work.
DEST_BATCH_SIZE = 5


def _flat(result: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`result`'s values under flat `columns`, index named `cell_id` (`create_properties`)."""
    flat = pd.DataFrame(result.to_numpy(), index=result.index, columns=columns)
    flat.index.name = 'cell_id'
    return flat


def flatten_bin_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(bin_name, property)` MultiIndex → `<property>_<bin_name>` flat columns."""
    return _flat(result, [f'{prop}_{bin_name}' for bin_name, prop in result.columns])


def flatten_k_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(k, property)` MultiIndex → `<property>_k<int(k)>` flat columns."""
    return _flat(result, [f'{prop}_k{int(k)}' for k, prop in result.columns])


def flatten_decay_dest_columns(result: pd.DataFrame) -> pd.DataFrame:
    """`(decay_name, property)` MultiIndex → `<property>_<decay_name>` flat columns."""
    return _flat(result, [f'{prop}_{decay}' for decay, prop in result.columns])


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
    Defaults to 1.0 when the row is absent or NaN (α fixed at 1, e.g. 07b's
    constrained specs). α ≠ 1 encodes a multiplicative bias correction on the
    routed time; downstream code applies it as `gross = t_routed + orig_ov +
    dest_ov − (1 − α) × t_routed` (or `gross = α × t_routed + …` for
    ODM-level assemblies). See `main/07a_road_overhead_coefs.py` and
    `main/07b_transit_overhead_coefs.py` for the fitting side.
    """
    alpha = coefs_for_profile.get('t_routed')
    return 1.0 if pd.isna(alpha) else float(alpha)


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


# Per-node density column of the road-overhead model. 07a fits on it, and every place that
# applies the fitted coefficients (08a, survey/08a, validation, figures) must use the same one.
ROAD_OVERHEAD_DENSITY_COL = 'density_r500_norm'

# Floor on a cell's residents + jobs weight in zone means (zone overheads in 08a, zone-mean
# access deviations in 06). Without it, a zone whose cells are all empty gets a NaN mean,
# which made all its zone-tier costs NaN; empty cells (which can hold POIs) count a little.
ZONE_WEIGHT_FLOOR = 1.0


def zone_weights(cells: pd.DataFrame) -> pd.Series:
    """Cell weights for zone means: `combined_total`, floored at `ZONE_WEIGHT_FLOOR`."""
    return cells['combined_total'].fillna(0.0).clip(lower=ZONE_WEIGHT_FLOOR)


def per_cell_road_overheads(
    cells: pd.DataFrame, coefs_for_profile: pd.Series,
    density_col: str, snap_dist_col: str,
) -> tuple[pd.Series, pd.Series]:
    """Full per-cell road overhead (`orig`, `dest`) from 07a's fit,
    per side: `const/2 + density_coef · density + snap_dist_coef · snap_dist`.
    Baked into 08a's gross ODMs.
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
      - `const_s`: trip-level intercept from 07b's fit (optional;
        default 0). Split in half so orig + dest sum to `const_s` at
        trip level.
      - `t_routed`: α, skipped here (see `route_time_alpha`).
    All other rows name columns in `cells` (stage 06's
    `cells_transit_access` features); NaN coefficients (features the spec
    doesn't use) and NaN column values contribute 0.
    """
    if 'cap_s' not in transit_coefs.index:
        raise ValueError("overheads_transit coef missing required `cap_s` row.")
    cap_s = float(transit_coefs.loc['cap_s', 'transit'])
    const_half = float(transit_coefs['transit'].get('const_s', 0.0)) / 2.0
    _RESERVED = ('cap_s', 'const_s', 't_routed')
    combined = pd.Series(const_half, index=cells.index)
    for param, row in transit_coefs.iterrows():
        if param in _RESERVED or pd.isna(row['transit']):   # NaN: feature not in the spec
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
# 09b writes no util ODM, so 10 skips the profile in util grids.
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
# applies whichever coefs 09a's `car_peak_night` variant produced (none by default).
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


def crossmodal_hour_shift(coefs: pd.DataFrame, profile_label: str, hour: str) -> float:
    """Utility shift of the mode of `profile_label` in the cross-modal profile of car hour
    `hour` ('night', 'base', 'peak'): 09a's `asc_<mode>_<hour>` (`peak_night_asc_modes`) for
    the single-profile modes, whose ODMs are at base hours; 0 for walking (the base), at base
    hours, and where 09a fit no shift. Car: 0, since its hour profiles carry their own."""
    util_mode = profile_to_util_mode(profile_label)
    key = f'asc_{util_mode}_{hour}'
    if util_mode in (None, 'car') or key not in coefs.index:
        return 0.0
    return float(coefs.loc[key, 'value'])


def crossmodal_hour_shift_keys(keys) -> set[str]:
    """Those of `keys` that are hour shifts of single-profile modes (`asc_<mode>_peak`,
    `asc_<mode>_night`, not car): applied by 10 in the cross-modal profiles, not by a 09b
    spec."""
    return {k for k in keys if isinstance(k, str) and k.startswith('asc_')
            and not k.startswith('asc_car_') and k.endswith(('_peak', '_night'))}


# Reference door-to-door time (minutes) at which 09a centers the time term,
# `f(t) − f(T)` (`f` piecewise linear, zero at t = 0): the constants are then the
# utility of a trip of T minutes rather than of 0 minutes, which lies far outside
# the data for transit and leaves its constant poorly determined. Common to all
# modes, so that constants compare modes at equal travel time; fixed, so that they
# compare across fits. 09a stores it in the coef file (`t_ref_min`).
UTILITY_T_REF_MIN = 20.0

# 09a's piecewise-linear time utility: one slope (per minute) per time band, named
# `b_time_pw_<mode>_<lo>_<hi>` (minutes; the last band open, `hi` = `inf`). The names
# carry the bands, so released coefficient files describe the time term completely.
_PIECEWISE_TIME_PREFIX = 'b_time_pw_'


def piecewise_time_coef_name(mode: str, lo: float, hi: float) -> str:
    """Coef name of `mode`'s time slope on the band [lo, hi) minutes (`hi` = inf: open)."""
    hi_str = 'inf' if math.isinf(hi) else f'{hi:g}'
    return f'{_PIECEWISE_TIME_PREFIX}{mode}_{lo:g}_{hi_str}'


def piecewise_time_coefs(values: pd.Series, mode: str) -> tuple[list[float], list[float],
                                                                 list[str]]:
    """Band starts (minutes), slopes (per minute) and coef names of `mode`'s piecewise-linear
    time utility in 09a's coefs (`values`: coef → value), in band order. `KeyError` if the
    mode has none; `ValueError` if the bands don't join up or the last one isn't open."""
    prefix = f'{_PIECEWISE_TIME_PREFIX}{mode}_'
    bands = sorted(
        (*(float(x) for x in key[len(prefix):].split('_')), float(values[key]), key)
        for key in values.index if isinstance(key, str) and key.startswith(prefix))
    if not bands:
        raise KeyError(f"no piecewise time coefs ({prefix}*) for {mode!r}.")
    los, his = [b[0] for b in bands], [b[1] for b in bands]
    if his[:-1] != los[1:] or not math.isinf(his[-1]):
        raise ValueError(f"{mode!r}: piecewise time bands {list(zip(los, his))} don't join "
                         f"up or don't end open.")
    return los, [b[2] for b in bands], [b[3] for b in bands]


def piecewise_time_utility(t, knots, slopes) -> np.ndarray:
    """`Σ_k slope_k · clip(t − knot_k, 0, knot_{k+1} − knot_k)`, the last band open: the
    piecewise-linear time utility (zero at t = 0, flat below the first knot), in the units
    of `t` and `slopes`. Non-finite `t` → NaN."""
    t = np.asarray(t, dtype=float)
    t = np.where(np.isfinite(t), t, np.nan)
    out = np.zeros_like(t)
    for lo, hi, slope in zip(knots, [*knots[1:], np.inf], slopes):
        out = out + slope * np.clip(t - lo, 0.0, hi - lo)
    return out


def time_utility(values: pd.Series, mode: str, t_min) -> np.ndarray:
    """`mode`'s time utility at `t_min` (minutes) from 09a's coefs, uncentered (zero at
    t = 0): add the constant and subtract the value at `t_ref_min` for 09a's utility."""
    knots, slopes, _ = piecewise_time_coefs(values, mode)
    return piecewise_time_utility(t_min, knots, slopes)


@dataclass(frozen=True)
class DisutilitySpec:
    """Parsed disutility spec for one routing profile. All β and the ASC
    are SIGN-FLIPPED (09a fits utility; disutility = −utility) so that
    downstream smaller = better, matching time semantics — `nearest_k`,
    `floor_intrazonal_costs`, and `exp(-β·D)` gravity all work directly.

    `disutility` carries the constant and the endpoint features; the time
    term is piecewise linear in the door-to-door time, so it sits outside
    aperta's `Utility` (whose `cost_coefficient` stays 0): `time_disutility`
    evaluates it from `time_knots_sec` (band starts) and `time_slopes`
    (per second). `sd_correction` is exposed for audit; already folded
    into `disutility.constant`.
    """
    disutility: Utility
    time_knots_sec: tuple[float, ...]
    time_slopes: tuple[float, ...]
    profile_label: str
    util_mode: str
    sd_correction: float = 0.0

    def time_disutility(self, t_sec) -> np.ndarray:
        """Time part of the disutility at `t_sec` (seconds); non-finite → NaN."""
        return piecewise_time_utility(t_sec, self.time_knots_sec, self.time_slopes)


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


def build_disutility_spec(
    coefs: pd.DataFrame, profile_label: str,
    sd_averages: dict[str, float] | None = None,
    consumed_out: set[str] | None = None,
) -> DisutilitySpec | None:
    """`DisutilitySpec` for one routing profile from 09a's coefs.

    Steps:
      1. ASC / constant (walk is base at 0).
      2. Piecewise-linear time slopes (per minute in 09a's calibration, bands
         read from the coef names; `piecewise_time_coefs`), with the
         centering offset at `t_ref_min` moved into the ASC.
      3. Car peak/night modifier: apply whichever of `asc_car_{peak,night}`
         (ASC shift) or `b_time_car_{peak,night}` (added to every band's
         slope, centered as in 09a) 09a fit; with neither, the hour profiles
         take the car's coefs. Feature-interaction peak/night coefs are not
         supported downstream: ignored with a warning next to an ASC / time
         modifier, and without one the hour profile gets no spec.
      4. Endpoint-feature coefs classified by `_orig`/`_dest`/`_od` suffix
         + SD correction folded into ASC (via `sd_averages` dict, read
         from the companion `utility_<variant>_stats` sidecar).
      5. Unit scaling min → sec (knots × 60, slopes / 60).
      6. Sign-flip utility → disutility (see class docstring).

    Route features (`b_<feat>_route_<mode>`) are ignored — the cell-baked
    ODM has no per-path aggregation surface.

    Returns `None` if the profile has no 09a util mode, if the mode's ASC or
    piecewise time coefs are missing, or (car hours) if 09a fit only
    feature-interaction modifiers.

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

    try:
        knots_min, slopes, time_keys = piecewise_time_coefs(values, util_mode)
    except KeyError:
        if f'b_time_{util_mode}' in values.index:
            raise ValueError(f"utility coefs have linear + log time terms for {util_mode!r} "
                             f"(fitted before the piecewise-linear form) — re-run main/09a "
                             f"for this scenario's source.") from None
        logging.warning(f"  → build_disutility_spec({profile_label!r}): no piecewise time "
                        f"coefs for {util_mode!r} — skipping.")
        return None
    if consumed_out is not None:
        consumed_out.update(time_keys)

    # 09a centers the time term at `t_ref_min` (see `UTILITY_T_REF_MIN`): `f(t) − f(T)`.
    # The released utilities keep the zero point of the uncentered form, a zero-length
    # walk, rather than a T-minute walk: adding walking's centering offset to every mode
    # leaves the choice probabilities unchanged.
    if 't_ref_min' not in values.index:
        raise ValueError("utility coefs lack 't_ref_min' (fitted before the time terms were "
                         "centered) — re-run main/09a for this scenario's source.")
    t_ref = float(values['t_ref_min'])
    constant += (float(time_utility(values, 'walk', t_ref))
                 - float(piecewise_time_utility(t_ref, knots_min, slopes)))

    # Apply whichever asc/b_time car peak/night modifiers 09a fit; flag
    # feature-interaction coefs (not yet wired downstream).
    if profile_label in _CAR_HOUR_MODIFIER:
        asc_mod_key, b_time_mod_key = _CAR_HOUR_MODIFIER[profile_label]
        applied = []
        if asc_mod_key in values.index:
            constant += float(values[asc_mod_key])
            applied.append(asc_mod_key)
        if b_time_mod_key in values.index:
            # 09a: `b · (t − T)` on top of the base time term.
            b_mod = float(values[b_time_mod_key])
            slopes = [s + b_mod for s in slopes]
            constant -= b_mod * t_ref
            applied.append(b_time_mod_key)
        hour = profile_label.split('_')[-1]  # 'peak' or 'night'
        feature_keys = [k for k in values.index
                        if k.startswith(f'b_car_{hour}_x_')]
        if feature_keys:
            logging.warning(f"  → build_disutility_spec({profile_label!r}): "
                            f"feature-interaction peak/night coefs {feature_keys} "
                            f"IGNORED (not yet supported downstream).")
        # Only feature interactions: no spec. (No modifiers at all, 09a's default: the
        # hour profiles share the car's coefs and differ only in their congested times.)
        if not applied and feature_keys:
            return None
        # Record feature-interaction keys so the tail-classifier doesn't
        # re-warn them as unclassified.
        consumed_carhour_keys = set(applied) | set(feature_keys)
        if consumed_out is not None:
            consumed_out.update(consumed_carhour_keys)
    else:
        consumed_carhour_keys = set()

    # Classify remaining `b_*_<util_mode>` coefs by suffix:
    #   `_orig` / `_dest` → per-side; `_od` → both sides (mirrors 09a's
    #   `endpoints_combined`); `b_sd_*` → SD correction. Unrecognised
    #   patterns land in `unclassified` and warn. (The time slopes end in
    #   their band, not the mode, so they never match.)
    origin_features: dict[str, float] = {}
    destination_features: dict[str, float] = {}
    sd_correction = 0.0
    unclassified: list[str] = []

    mode_suffix = f'_{util_mode}'
    for key in values.index:
        if not isinstance(key, str) or not key.startswith('b_'):
            continue
        if not key.endswith(mode_suffix):
            continue
        if key in consumed_carhour_keys:
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

    # utility → disutility: negate all coefs so smaller = better and
    # `nearest_k` / `floor_intrazonal_costs` / `exp(-β·D)` apply directly.
    # 09a fit on minutes, aperta routes on seconds: knots × 60, slopes / 60.
    disutility = Utility(
        constant=-constant,
        cost_coefficient=0.0,
        origin_features={k: -v for k, v in origin_features.items()},
        destination_features={k: -v for k, v in destination_features.items()},
    )
    return DisutilitySpec(
        disutility=disutility,
        time_knots_sec=tuple(k * 60.0 for k in knots_min),
        time_slopes=tuple(-s / 60.0 for s in slopes),
        profile_label=profile_label,
        util_mode=util_mode,
        sd_correction=sd_correction,
    )


def join_util_node_features(
    cells_m: pd.DataFrame, zones_m: pd.DataFrame, context,
) -> None:
    """Join per-node point features onto `cells_m`/`zones_m` in place.
    Each feature from its source graph (per `POINT_FEATURE_SOURCES`), via
    `node_id_<source_mode>`; missing features silently skipped. Applies
    `FEATURE_SCALE` matching 09a's fit-time scaling. If the source mode isn't
    in the scenario, a point feature is read from a configured mode instead:
    02b computes the same point features for every network.
    """
    from scenarios import get_scenario
    modes = list(get_scenario(context.scenario).mode_configs)
    files_cache: dict[str, pd.DataFrame] = {}
    for feat, (source_mode, data_name) in POINT_FEATURE_SOURCES.items():
        if source_mode not in modes:
            if not data_name.endswith('_extended'):
                continue
            source_mode = modes[0]
            data_name = f'{source_mode}_extended'
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

        U(i, j) = ASC + f(cost) + Σ β_endpoint · feature

    with `f` the piecewise-linear time term (`spec.time_disutility`, cost in
    seconds). `unit_col` (default `'unit_id'`) is the cells/zones column matching
    the geo-keyed pairs' origin/dest IDs — used as `node_column` for
    aperta's endpoint lookup.
    """
    # Time term per pair. Non-finite costs propagate as NaN.
    def _time_tier(tier: dict | None) -> dict | None:
        if tier is None:
            return None
        return {origin: spec.time_disutility(arr).astype(arr.dtype, copy=False)
                for origin, arr in tier.items()}
    time_u = type(cost_odm_sec)(
        cells_to_cells=_time_tier(cost_odm_sec.cells_to_cells),
        cells_to_zones=_time_tier(cost_odm_sec.cells_to_zones),
        zones_to_zones=_time_tier(cost_odm_sec.zones_to_zones),
    )
    return _utility.add_endpoint_utility(
        time_u, pairs_geo, spec.disutility,
        cells=cells_m, zones=zones_m, node_column=unit_col,
    )


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
        util_extra_nans = 0
        cost_finite_total = 0
        for origin, cost_arr in cost_tier.items():
            util_arr = util_tier.get(origin)
            cost_nan_mask = ~np.isfinite(np.asarray(cost_arr))
            if util_arr is None:
                util_nan_mask = np.ones_like(cost_nan_mask)
            else:
                util_nan_mask = ~np.isfinite(np.asarray(util_arr))
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
    z2z_times = times.zones_to_zones or {}
    return {(orig, dest): float(val) for orig, dests in (idx.zones_to_zones or {}).items()
            for dest, val in zip(dests, z2z_times[orig])}


def _pick_by_peak(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Per leg, column `<prefix>_<peak_str>` (peak / base / night); other `peak_str` → NaN."""
    hours = ('peak', 'base', 'night')
    return pd.Series(np.select([df['peak_str'] == h for h in hours],
                               [df[f'{prefix}_{h}'] for h in hours], default=np.nan),
                     index=df.index)


def car_overhead_by_peak(df: pd.DataFrame, side: str) -> pd.Series:
    """Per-leg car overhead (seconds) at `side` ∈ {`orig`, `dest`}, from
    `t_overhead_{side}_car_{peak,base,night}` by `peak_str`."""
    return _pick_by_peak(df, f't_overhead_{side}_car')


def car_routed_bias_by_peak(df: pd.DataFrame) -> pd.Series:
    """Per-leg car `t_routed_bias` (seconds), from `t_routed_bias_car_{peak,base,night}`
    by `peak_str`. Consumers SUBTRACT it: `gross = t_routed + orig_ov + dest_ov − bias`."""
    return _pick_by_peak(df, 't_routed_bias_car')


# ---------------------------------------------------------------------------
# Trip-level survey aggregation
# ---------------------------------------------------------------------------

_TRANSIT_BIKE_MODES: frozenset[str] = frozenset({'rbike', 'ebike25', 'ebike45'})

# Plausibility flags (0/1); consumers keep the rows with all flags == 1.
# Of legs (survey prep, `standardize`; `is_within_gap_envelope` is trip-level already); left
# out of the trip set's rows, which carry `TRIP_QUALITY_FLAGS` instead.
LEG_QUALITY_FLAGS: tuple[str, ...] = (
    'is_within_speed_envelope', 'is_within_detour_envelope', 'is_within_gap_envelope',
)
# Of the trips of the trip set (`survey_trips`, rules at `TRIP_MIN_DIST_LINE_M`), applied by
# 07a and 07b, which fit the trips' times, and by the validation of those times.
TRIP_QUALITY_FLAGS: tuple[str, ...] = (
    'is_valid_time', 'is_valid_main_legs', 'is_main_above_min_speed', 'is_main_below_max_speed',
    'is_above_min_dist_line', 'is_within_detour_ratio', 'is_within_unexplained_time',
    'is_within_other_legs_time',
)
# Applied by 09a, which asks whether a trip and its mode are real, not whether its time is:
# a slow main leg (stops, strolling) or long legs besides it do not count against it.
CHOICE_QUALITY_FLAGS: tuple[str, ...] = tuple(
    f for f in TRIP_QUALITY_FLAGS
    if f not in ('is_main_above_min_speed', 'is_within_other_legs_time'))


# Plausibility of a trip of the trip set (`survey_trips`, `TRIP_QUALITY_FLAGS`). The main-mode
# legs are checked on their own, as 04 checks legs (survey prep's `is_valid_dist_time` and speed
# envelope, its two sides apart; too fast only from `TRIP_FAST_CHECK_MIN_TIME_S` reported, as
# shorter reports are too coarse to tell a runner or a wrong mode from a reporting slip); the
# walking and feeder legs only through the whole trip:
#   - door-to-door time (first departure → last arrival) above 0;
#   - ends at least `TRIP_MIN_DIST_LINE_M` apart: closer ends mostly share a cell or a network
#     node, so the modeled times are only their floors;
#   - summed leg distances at most `TRIP_MAX_DETOUR_RATIO` × the straight line, the latter
#     floored at `TRIP_DETOUR_FLOOR_M` since short trips' distances are rounded estimates
#     (loops, errands on the way);
#   - unexplained time (stops on the way; `_trip_quality_flags`) at most
#     min(`TRIP_MAX_UNEXPLAINED_S`, max(`TRIP_UNEXPLAINED_FLOOR_S`, `TRIP_MAX_UNEXPLAINED_RATIO`
#     × the explained time)), and the time between legs at least survey prep's
#     `MIN_TRIP_GAP_S` (clock noise);
#   - each leg besides the main mode's (reported) at most `TRIP_MAX_OTHER_LEG_S` (cycling and
#     car trips: their walking legs) or `TRIP_MAX_TRANSIT_OTHER_LEG_S` (transit: walking,
#     cycling or car legs): longer walks are mostly activities of their own (a stroll from
#     the parked car) rather than the way to or from the vehicle, and take far longer than
#     the overheads allow for. Looser for transit, whose walks to, from and between stops
#     and drives to park-and-ride are part of the trip, as NPVM's times and 07b's access
#     terms represent.
TRIP_FAST_CHECK_MIN_TIME_S = 10 * 60
TRIP_MIN_DIST_LINE_M = 100.0
TRIP_MAX_DETOUR_RATIO = 3.0                 # as survey prep's `MAX_DETOUR_RATIO` for legs
TRIP_DETOUR_FLOOR_M = 500.0
TRIP_MAX_UNEXPLAINED_S = 30 * 60
TRIP_UNEXPLAINED_FLOOR_S = 10 * 60
TRIP_MAX_UNEXPLAINED_RATIO = 1.0
TRIP_MAX_OTHER_LEG_S = 15 * 60
TRIP_MAX_TRANSIT_OTHER_LEG_S = 30 * 60


def _trip_quality_flags(sub: pd.DataFrame, main_mode: pd.Series, span: pd.Series,
                        dist_line: pd.Series) -> pd.DataFrame:
    """`TRIP_QUALITY_FLAGS` (0/1) per trip, and its `time_gap` (between legs) and
    `time_unexplained` (s). `sub`: the trips' legs in trip order; `main_mode`, `span`
    (door-to-door time, s) and `dist_line` per trip. A trip's explained time is the
    time of its main-mode legs plus, for each other leg, at most the time it takes at its
    mode's lowest plausible speed (survey prep's speed envelope; reports read as rounded, as
    there); the rest of the door-to-door time is unexplained: the time between legs, and what
    a walking or feeder leg takes beyond that speed (e.g. 30 min reported for 200 m on foot)."""
    trip = sub['trip_id']
    mode = sub['mode_simplified']
    is_main = mode == trip.map(main_mode)

    def all_main(holds: pd.Series) -> pd.Series:
        return (holds | ~is_main).groupby(trip).all()

    too_slow, too_fast = speed_envelope_sides(sub, rounded_reports=True)
    checked = sub['time_measured'] >= TRIP_FAST_CHECK_MIN_TIME_S
    n_spared = int((~all_main(~too_fast) & all_main(~(too_fast & checked))).sum())
    logging.info(f"  → {n_spared:,} trips with a too fast main-mode leg reported under "
                 f"{TRIP_FAST_CHECK_MIN_TIME_S / 60:.0f} min, not flagged")

    exact_lo, _, rounded_lo, _, _ = report_intervals(sub['time_measured'].to_numpy(dtype=float))
    time_lo = pd.Series(np.fmin(exact_lo, rounded_lo), index=sub.index)
    min_speed = mode.map({m: lo / 3.6 for m, (lo, _) in SPEED_ENVELOPES_KMH.items()})
    excess = (time_lo - sub['dist_measured'] / min_speed).clip(lower=0.0)
    excess = excess.where(~is_main).fillna(0.0)
    gap = span - sub['time_measured'].groupby(trip).sum()
    longest_other = sub['time_measured'].where(~is_main, 0.0).groupby(trip).max()
    other_limit = np.where(main_mode.reindex(longest_other.index).eq('transit'),
                           TRIP_MAX_TRANSIT_OTHER_LEG_S, TRIP_MAX_OTHER_LEG_S)
    unexplained = gap + excess.groupby(trip).sum()
    limit = np.minimum(TRIP_MAX_UNEXPLAINED_S, np.maximum(
        TRIP_UNEXPLAINED_FLOOR_S, TRIP_MAX_UNEXPLAINED_RATIO * (span - unexplained)))
    detour = sub['dist_measured'].groupby(trip).sum() / np.maximum(dist_line, TRIP_DETOUR_FLOOR_M)
    flags = pd.DataFrame({
        'is_valid_time': span > 0,
        'is_valid_main_legs': all_main(sub['is_valid_dist_time'] == 1),
        'is_main_above_min_speed': all_main(~too_slow),
        'is_main_below_max_speed': all_main(~(too_fast & checked)),
        'is_above_min_dist_line': dist_line >= TRIP_MIN_DIST_LINE_M,
        'is_within_detour_ratio': ~(detour > TRIP_MAX_DETOUR_RATIO),
        # NaN times pass here; `is_valid_time` drops them.
        'is_within_unexplained_time': ~((gap < MIN_TRIP_GAP_S) | (unexplained > limit)),
        'is_within_other_legs_time': longest_other <= other_limit,
    })
    flags = flags[list(TRIP_QUALITY_FLAGS)].astype(int)
    return flags.assign(time_gap=gap, time_unexplained=unexplained)


def trip_drop_cause(legs: pd.DataFrame) -> pd.Series:
    """Per trip (`trip_id`): the first reason `survey_trips` drops it, NaN if it keeps it
    (categorical, the causes in the order they are checked). `legs` as for `survey_trips`."""
    sub = in_trip_order(legs[legs['trip_id'].notna()])
    trip = sub['trip_id']
    mode = sub['mode_simplified']
    pos = sub.groupby('trip_id', sort=False).cumcount()
    is_walk, is_transit = mode == 'walk', mode == 'transit'
    is_feeder = ~is_walk & ~is_transit        # vehicle legs other than transit
    transit_pos = pos.where(is_transit).groupby(trip)
    first_t, last_t = transit_pos.transform('min'), transit_pos.transform('max')

    def per_trip(mask: pd.Series) -> pd.Series:
        return mask.groupby(trip).sum()
    info = pd.DataFrame({
        'n_present': trip.groupby(trip).size(),
        'n_legs': sub.groupby('trip_id')['n_legs_in_trip'].first(),
        'n_walk': per_trip(is_walk),
        'n_transit': per_trip(is_transit),
        'n_feeder': per_trip(is_feeder),
        'n_feeder_between': per_trip(is_feeder & (pos > first_t) & (pos < last_t)),
        'n_feeder_before': per_trip(is_feeder & (pos < first_t)),
        'n_feeder_after': per_trip(is_feeder & (pos > last_t)),
    })
    on_foot = (info['n_present'] == 1) & (info['n_walk'] == 1)
    road = (info['n_transit'] == 0) & (info['n_feeder'] == 1)
    transit = ((info['n_transit'] >= 1) & (info['n_feeder_between'] == 0)
               & (info['n_feeder_before'] <= 1) & (info['n_feeder_after'] <= 1))
    # Causes in order; a trip counts under the first that applies.
    causes = {
        'legs missing': info['n_present'] != info['n_legs'],
        'other leg combination': ~(on_foot | road | transit),
    }
    cause = pd.Series(pd.Categorical([None] * len(info), categories=list(causes)),
                      index=info.index)
    for name, mask in causes.items():
        cause[mask & cause.isna()] = name
    return cause


def survey_trips(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per door-to-door trip of the kinds the overhead fits (07a, 07b) and the
    mode-choice model (09a) use, from the trips' legs (`TRIP_SET`, built in survey/02d).

    Main mode, and the legs a trip may have besides it:
      - walk: none (a single walking leg);
      - rbike / ebike25 / ebike45 / car: exactly one leg of that mode, plus walking legs; car
        as the driver or as a passenger (incl. taxi);
      - transit: one or more transit legs, with only walking between them; before the first
        and after the last, walking plus at most one cycling or car leg per end
        (park-and-ride, drop-off), whichever leg is longest (as the MTMC's main mode).
    A trip needs all its legs (`n_legs_in_trip`): legs that failed survey/02d's filters (abroad,
    other modes) drop it. Dropped trips are logged by cause. Plausibility is flagged, not
    filtered (`TRIP_QUALITY_FLAGS`, rules at `TRIP_MIN_DIST_LINE_M`).

    Columns as survey/02d's legs, at trip level, indexed by `trip_id`: the ends `orig_x/y`
    (first leg) and `dest_x/y` (last leg) and their `dist_line`; `time_measured` door to door
    (first departure → last arrival, incl. waiting between legs; NaN without both);
    `dist_measured` summed; `mode_simplified` the main mode, with the main leg's
    `mode_is_passenger`, `peak_str` and `hour_*` (transit: the first transit leg's); person
    and departure columns (`weight_person`, `sd_*`, `datetime_departure`, `is_daylight`, ...)
    from the first leg; `TRIP_QUALITY_FLAGS` (the legs' own flags are left out) and
    `time_gap`, `time_unexplained` (s) behind them; `n_legs_in_trip`, `n_walk_legs` (besides
    the main legs), `n_transit_legs`, and `transit_feeder` (`walk`, `bike`, `car` or
    `bike_car`; transit only).
    """
    sub = in_trip_order(legs[legs['trip_id'].notna()])
    cause = trip_drop_cause(sub)
    logging.info(f"  → {int(cause.isna().sum()):,} of {len(cause):,} trips kept; dropped: "
                 + ', '.join(f'{c} {n:,}' for c, n in cause.value_counts(sort=False).items()))

    sub = sub[sub['trip_id'].isin(cause.index[cause.isna()])]
    trip = sub['trip_id']
    mode = sub['mode_simplified']
    pos = sub.groupby('trip_id', sort=False).cumcount()
    is_walk, is_transit = mode == 'walk', mode == 'transit'
    is_bike, is_car = mode.isin(_TRANSIT_BIKE_MODES), mode == 'car'
    g = sub.groupby('trip_id')
    first = g.head(1).set_index('trip_id')
    last = g.tail(1).set_index('trip_id')
    # Main leg: the first transit leg, else the vehicle leg, else the walking leg.
    rank = np.select([is_transit, ~is_walk], [0, 1], 2)
    main = (sub.assign(_rank=rank, _pos=pos.to_numpy()).sort_values(['_rank', '_pos'])
            .groupby('trip_id').head(1).set_index('trip_id'))

    out = first.drop(columns=[c for c in (*LEG_QUALITY_FLAGS, 'is_valid_dist_time')
                              if c in first.columns])
    out[['dest_x', 'dest_y', 'datetime_arrival']] = last[['dest_x', 'dest_y', 'datetime_arrival']]
    for col in ('mode_simplified', 'mode_is_passenger', 'peak_str', 'hour_peak', 'hour_night',
                'hour_base'):
        out[col] = main[col]
    out['dist_line'] = np.hypot(out['dest_x'] - out['orig_x'], out['dest_y'] - out['orig_y'])
    out['dist_measured'] = g['dist_measured'].sum()
    span = (pd.to_datetime(out['datetime_arrival'], errors='coerce')
            - pd.to_datetime(out['datetime_departure'], errors='coerce')).dt.total_seconds()
    span = span.where(span >= 0, span + 86_400)            # trips across midnight
    out['time_measured'] = span
    flags = _trip_quality_flags(sub, out['mode_simplified'], span, out['dist_line'])
    out = out.join(flags)
    logging.info("  → share passing each plausibility check: "
                 + ', '.join(f'{f} {flags[f].mean():.1%}' for f in TRIP_QUALITY_FLAGS)
                 + f"; all {flags[list(TRIP_QUALITY_FLAGS)].all(axis=1).mean():.1%}")
    out['n_legs_in_trip'] = g.size()
    out['n_transit_legs'] = is_transit.groupby(trip).sum()
    out['n_walk_legs'] = (is_walk.groupby(trip).sum()
                          - (out['mode_simplified'] == 'walk').astype(int))
    bike, car = is_bike.groupby(trip).any(), is_car.groupby(trip).any()
    feeder = np.select([bike & car, bike, car], ['bike_car', 'bike', 'car'], 'walk')
    out['transit_feeder'] = pd.Series(feeder, index=bike.index).where(
        out['mode_simplified'] == 'transit')
    out.index.name = 'trip_id'
    logging.info(f"  → main modes: {out['mode_simplified'].value_counts().to_dict()}; "
                 f"transit feeders: {out['transit_feeder'].value_counts().to_dict()}")
    return out
