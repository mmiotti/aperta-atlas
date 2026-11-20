"""
Switzerland-specific helpers shared across the survey-prep scripts in this folder.

Main groups:

  1. **Time-of-day / plausibility** (`is_peak`, `is_night`, `report_intervals`)
     plus their Swiss-tuned defaults (`PEAK_HOURS_WEEKDAY`, `SPEED_ENVELOPES_KMH`,
     `MAX_DETOUR_RATIO`, `MAX_TRIP_GAP_S`, ...). The defaults assume Swiss
     commuting patterns and the MZMV/MOBIS mode taxonomy.

  2. **Mode simplification** (`MODE_TO_SIMPLIFIED`): standardized MTMC leg
     modes → the mode groups the atlas models.

These live here (not in `aperta/`) because they depend on Swiss commuting /
mode-name conventions. The MTMC code books themselves are in `mzmv/`.

Also here: small generic helpers (`add_lat_lon`, `add_group_id`, `bin_labels`,
`in_trip_order`) and the survey column allowlist (`ALLOWED_SURVEY_PREFIXES`).
"""

from typing import Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer


def add_lat_lon(
    df: pd.DataFrame,
    prefix: str,
    from_crs: str,
    to_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """Add `<prefix>_lat` and `<prefix>_lon` columns by transforming `<prefix>_x` / `<prefix>_y`."""
    transformer = Transformer.from_crs(from_crs, to_crs)
    lat, lon = transformer.transform(df[f"{prefix}_x"].to_numpy(), df[f"{prefix}_y"].to_numpy())
    df[f"{prefix}_lat"] = lat
    df[f"{prefix}_lon"] = lon
    return df


def add_group_id(df: pd.DataFrame, id_col: str, out_col: str = "group_id") -> pd.DataFrame:
    """Add `out_col` with sequential integer IDs based on unique values of `id_col`
    (in order of first appearance; NaN → -1)."""
    codes, _ = pd.factorize(df[id_col], sort=False)
    df[out_col] = codes
    return df


def bin_labels(bins: Sequence[int | float]) -> list[str]:
    """Human-readable labels for a `pd.cut` binning.

    Given right-closed bins like `(0, 6, 13, 18, 25, …, 120)` produces
    `['leq6', '7to13', '14to18', '19to25', …, '76+']` — one label per
    interval `(bins[i], bins[i+1]]`. Convention:

    - First bin: `'leq{bins[1]}'` (or literal `'0'` if `bins[1] == 0`).
    - Last bin:  `'{bins[-2]+1}+'` (open-ended, "and above").
    - Single-value bins (where `bins[i]+1 == bins[i+1]`) collapse to
      just the value, so `(-1, 0, 1, 2, 99)` becomes `['0', '1', '2', '3+']`.
    """
    out: list[str] = []
    for i in range(len(bins) - 1):
        if i == 0:
            out.append('0' if bins[i + 1] == 0 else f'leq{bins[i + 1]}')
        elif i == len(bins) - 2:
            out.append(f'{bins[i] + 1}+')
        elif bins[i] + 1 == bins[i + 1]:
            out.append(f'{bins[i] + 1}')
        else:
            out.append(f'{bins[i] + 1}to{bins[i + 1]}')
    return out


# ---------------------------------------------------------------------------
# Time-of-day / plausibility constants (Swiss commuting + mode taxonomy).
# Used by the survey-prep scripts (`b_mzmv_process.py`, `b_mobis_process.py`)
# to flag rows by peak/night/weekday and implausible speeds / detours / gaps.

# The slowest part of peak hours in the morning is quite short; higher impact
# in the afternoon.
PEAK_HOURS_WEEKDAY: tuple[int, ...] = (8, 16, 17, 18)
NIGHT_HOURS_WEEKDAY: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23)
NIGHT_HOURS_WEEKEND: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23)

# Light, service and season flags (`standardize.attach_daylight_and_season`), separate from
# the traffic profiles above:
# - Sun altitude (degrees) below which it counts as dark: civil twilight, when outdoor light
#   no longer suffices (civil dusk / dawn ≈ 30-40 min after sunset / before sunrise).
CIVIL_TWILIGHT_DEG: float = -6.0
# - Departure hours (00:00-04:59) in which public transport mostly doesn't run; the night
#   networks run on weekend nights only.
LATE_NIGHT_HOURS: tuple[int, ...] = (0, 1, 2, 3, 4)
# - Meteorological seasons, by month.
SEASON_BY_MONTH: dict[int, str] = {
    12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall',
}

# Per-mode speed envelopes for outlier filtering on survey legs.
# Keys align with the post-classification `mode_simplified` values.
# Bike family:
#   'rbike'    — confirmed mechanical (MTMC F51300=2; MOBIS post-2020-07 Bicycle)
#   'ebike25'  — confirmed 25 km/h ebike (MTMC F51300=3)
#   'ebike45'  — confirmed 45 km/h ebike (MTMC F51300=4)
#   'anybike'  — unknown-subtype bike (MOBIS pre-2020-07 Bicycle; Bikesharing)
#   'anyebike' — ebike, assist limit unknown (MOBIS Mode::Ebicycle)
# Deliberately STRICT — false positives (dropping legitimate trips) are
# cheaper than false negatives (mode-misclassified trips silently
# poisoning per-mode edge-weight calibration). Loosen only if the logged
# out-of-envelope rate suggests real trips are being lost.
SPEED_ENVELOPES_KMH: dict[str, tuple[float, float]] = {
    'walk':     (2.0,   8.0),  # 3-5 km/h typical; upper generous to admit GPS/rounding noise on brisk walks
    'rbike':    (5.0,  30.0),  # Mechanical bike; lower relaxed for Alpine climbs / urban stops
    'ebike25':  (7.0,  32.0),  # 25 km/h assist limit + margin; motor kicks in from a stop
    'ebike45':  (7.0,  47.0),  # 45 km/h assist limit + margin
    'anybike':  (5.0,  32.0),  # Unknown mechanical / ebike25 mix — lower matches rbike
    'anyebike': (7.0,  47.0),  # Unknown ebike25 / ebike45 mix — lower matches other ebikes
    'car':      (5.0, 130.0),  # 130 km/h Swiss highway top; slow urban floor
}


# Per-mode max plausible route detour ratio `dist_measured / dist_line`. Typical urban routes
# are ~1.2-1.6; above the max, the leg involved a huge detour (errands along the way, or a path
# so far from the straight line that it isn't representative for edge-weight fitting). Uniform
# 3.0: the variation between modes is smaller than the safety margin.
MAX_DETOUR_RATIO: dict[str, float] = dict.fromkeys(SPEED_ENVELOPES_KMH, 3.0)


# Trip-level time-gap envelope: gap = (last leg's arrival − first leg's
# departure) − Σ leg `time_measured`, i.e. time between legs (waiting at
# stops, transfers, parking). Gaps beyond MAX_TRIP_GAP_S, or beyond both
# MAX_TRIP_GAP_RATIO × Σ legs and TRIP_GAP_RATIO_FLOOR_S, flag "did something
# else along the way" (short stops / activities inside one reported trip);
# gaps below MIN_TRIP_GAP_S flag inconsistent reports. The ratio term catches
# stops inside short trips; its floor keeps ordinary waits at stops and
# transfers in.
MAX_TRIP_GAP_S: float = 30 * 60
MAX_TRIP_GAP_RATIO: float = 1 / 2
TRIP_GAP_RATIO_FLOOR_S: float = 5 * 60
MIN_TRIP_GAP_S: float = -5 * 60


def in_trip_order(legs: pd.DataFrame) -> pd.DataFrame:
    """`legs` sorted by `trip_id`, then by the number that ends each leg id: the leg number
    for MTMC ids (`<year>_<HHNR>_<person no.>_<leg no.>`, numbered over the survey day), the
    row number for MOBIS legs in survey prep. Sorting the ids as text would put leg 10 before
    leg 9."""
    seq = legs.index.astype(str).str.extract(r'(\d+)$', expand=False).astype(int)
    return (legs.assign(_leg_seq=seq.to_numpy())
            .sort_values(['trip_id', '_leg_seq']).drop(columns='_leg_seq'))


def report_intervals(time_s: np.ndarray) -> tuple[np.ndarray, ...]:
    """Bounds (s) of the true times a report can stand for. Exact reading:
    report ± 0.5 min. Rounded reading (reports on 5-min steps only):
    report ± 2.5 min, except "5 min" → 0–7.5 min, since short trips are
    hardly ever reported as 0. Returns `(exact_lo, exact_hi, rounded_lo,
    rounded_hi, on_step)`; rounded bounds are NaN off the 5-min steps."""
    m = np.round(time_s / 60.0)
    on_step = (m % 5 == 0)
    exact_lo = np.maximum(m - 0.5, 0.0) * 60.0
    exact_hi = (m + 0.5) * 60.0
    rounded_lo = np.where(m == 5, 0.0, np.maximum(m - 2.5, 0.0)) * 60.0
    rounded_hi = (m + 2.5) * 60.0
    rounded_lo = np.where(on_step, rounded_lo, np.nan)
    rounded_hi = np.where(on_step, rounded_hi, np.nan)
    return exact_lo, exact_hi, rounded_lo, rounded_hi, on_step


def is_peak(df: pd.DataFrame,
            peak_hours: Sequence[int] = PEAK_HOURS_WEEKDAY) -> pd.Series:
    """1 if `hour_of_day` is in `peak_hours` AND `sd_bool_weekday == 1`, else 0."""
    return (df['hour_of_day'].isin(peak_hours) & (df['sd_bool_weekday'] == 1)).astype(int)


def is_night(df: pd.DataFrame,
             weekday_hours: Sequence[int] = NIGHT_HOURS_WEEKDAY,
             weekend_hours: Sequence[int] = NIGHT_HOURS_WEEKEND) -> pd.Series:
    """1 if `hour_of_day` is in the relevant night window for that day-type, else 0."""
    weekday = df['sd_bool_weekday'] == 1
    in_weekday_window = df['hour_of_day'].isin(weekday_hours)
    in_weekend_window = df['hour_of_day'].isin(weekend_hours)
    return ((weekday & in_weekday_window) | (~weekday & in_weekend_window)).astype(int)


# ---------------------------------------------------------------------------
# Modes and reference data
# ---------------------------------------------------------------------------

# Standardized MTMC leg modes (`mzmv/schema.py`) → the mode groups the atlas models.
MODE_TO_SIMPLIFIED: dict[str, str] = {
    'transit_rail': 'transit', 'transit_bus': 'transit', 'transit_tram': 'transit',
    'motorbike_small': 'motorbike',
    'taxi': 'car',
}


# ---------------------------------------------------------------------------
# Survey config (region-agnostic but accessed alongside the Swiss helpers)
# ---------------------------------------------------------------------------

ALLOWED_SURVEY_PREFIXES: tuple[str, ...] = (
    'time_', 'dist_', 'speed_', 'timestamp_', 'datetime_', 'hour_', 'mode_', 'peak_',
    'is_', 'season', 'purpose', 'sd_',
    'orig_', 'dest_', 'n_', 'weight_', 'route_', 'include_', 'group_', 'subset_',
    'elev_',  # endpoint elevation (`elev_orig`, `elev_dest` from DEM sample)
    'municipality_id', 'canton_id', 'zone_id', 'longest_leg_id', 'hh_',
    # Cross-survey leg / trip / person identifiers.
    'trip_id', 'leg_id', 'participant_id',
)

# Bike family: see the `SPEED_ENVELOPES_KMH` notes.
MODES_MODE_CHOICE_MODEL = (
    'walk', 'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike', 'car', 'transit')

# If max geometry size is 500 and min precision is 250 m, trajectories longer than
# ~125 km will be excluded due to too many points after simplification.
TARGET_GEOMETRY_SIZE = 100
MAXIMUM_GEOMETRY_SIZE = 500
