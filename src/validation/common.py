"""Shared helpers for the validation scripts.

Currently: physical-plausibility filter used to symmetrically exclude
legs where either the ground truth or the model output is implausible
(GPS glitches, graph pathologies). Applied by `times_vs_survey.py` on
the same subset of legs on both sides of each comparison.
"""

import logging

import numpy as np
import pandas as pd


# Per-mode physical-plausibility bounds for `dist / time` (m/s).
# Loose enough to keep legitimate legs (including e-bikes on the
# bike side), tight enough to catch API errors, GPS glitches, and
# routing pathologies where one side goes through a barrier.
PLAUSIBLE_SPEED_RANGE_M_S: dict[str, tuple[float, float]] = {
    'walk': (0.5, 2.5),    # 1.8 - 9 km/h
    'bike': (1.5, 10.0),   # 5.4 - 36 km/h  (ebikes fit under the ceiling)
    'car':  (2.0, 35.0),   # 7 - 126 km/h
}

# Per-mode plausible route-to-line detour ratio.
# `route_dist / dist_line` — typical urban routes are ~1.2-1.6; ratios
# above MAX flag "the router took a huge detour" (often across a barrier:
# lake / rail / motorway). Set uniformly conservative across modes so a
# legitimate detour around a lake at the AOI boundary can still pass —
# per-mode differentiation isn't warranted since the ordering of routing
# constraints (walk < bike < car) would push in the opposite direction
# from what a purpose-of-filter argument would suggest, and the
# variation is smaller than the safety margin we want anyway.
MAX_DETOUR_RATIO: dict[str, float] = {
    'walk': 3.0,
    'bike': 3.0,
    'car':  3.0,
}

# Survey `mode_simplified` → physical mode for the plausibility bounds
# above. Transit is excluded (z2z lookup, not routed; no per-leg
# distance to check). ebike variants and MOBIS's unknown-sub-type bike
# modes (`anybike` / `anyebike`) map to bike.
SURVEY_MODE_TO_PHYSICAL: dict[str, str] = {
    'walk':     'walk',
    'rbike':    'bike',
    'ebike25':  'bike',
    'ebike45':  'bike',
    'anybike':  'bike',
    'anyebike': 'bike',
    'car':      'car',
}


def is_plausible_leg(
    physical_mode: str,
    dist_m: pd.Series | np.ndarray,
    time_s: pd.Series | np.ndarray,
    dist_line_m: pd.Series | np.ndarray,
) -> pd.Series:
    """Boolean mask: True for legs that pass the per-mode plausibility check.

    Two failing modes:
      - Implied speed (`dist / time`) outside `PLAUSIBLE_SPEED_RANGE_M_S`
        → API error, GPS glitch, or mislabelled mode.
      - Detour ratio (`dist / dist_line`) exceeds `MAX_DETOUR_RATIO`
        → routed through a huge, physically implausible detour.

    Unknown `physical_mode` → conservative True (no filtering).
    NaN in any input → False (fail-safe: excluded).
    """
    dist_m = pd.Series(dist_m).astype(float)
    time_s = pd.Series(time_s).astype(float)
    dist_line_m = pd.Series(dist_line_m).astype(float)
    if physical_mode not in PLAUSIBLE_SPEED_RANGE_M_S:
        return pd.Series(True, index=dist_m.index)
    lo, hi = PLAUSIBLE_SPEED_RANGE_M_S[physical_mode]
    max_detour = MAX_DETOUR_RATIO[physical_mode]
    speed = dist_m / time_s.where(time_s > 0)
    detour = dist_m / dist_line_m.where(dist_line_m > 0)
    return speed.between(lo, hi) & (detour <= max_detour)


def plausibility_mask_bulk(
    survey_mode: pd.Series,
    dist_m: pd.Series,
    time_s: pd.Series,
    dist_line_m: pd.Series,
) -> pd.Series:
    """Boolean mask per leg for a mixed-mode DataFrame (like `survey_legs`).
    Dispatches to `is_plausible_leg` per unique physical mode. Legs whose
    `survey_mode` isn't in `SURVEY_MODE_TO_PHYSICAL` (e.g. transit) pass
    through with True (no check applicable)."""
    keep = pd.Series(True, index=survey_mode.index)
    for smode in survey_mode.dropna().unique():
        physical = SURVEY_MODE_TO_PHYSICAL.get(smode)
        if physical is None:
            continue
        mask = survey_mode == smode
        keep.loc[mask] = is_plausible_leg(
            physical,
            dist_m.loc[mask], time_s.loc[mask], dist_line_m.loc[mask],
        ).values
    return keep


def log_exclusion(label: str, keep_gt: pd.Series, keep_model: pd.Series) -> pd.Series:
    """Log per-source exclusion counts + return the combined AND mask.
    `keep_gt` / `keep_model` = plausibility mask on ground truth / model side.
    """
    n = len(keep_gt)
    n_gt = int((~keep_gt).sum())
    n_model = int((~keep_model).sum())
    combined = keep_gt & keep_model
    n_kept = int(combined.sum())
    logging.info(
        f"  → {label}: ground truth excludes {n_gt:,}, model excludes "
        f"{n_model:,}; {n_kept:,} / {n:,} kept after both filters "
        f"({100*n_kept/max(n,1):.1f}%)")
    return combined
