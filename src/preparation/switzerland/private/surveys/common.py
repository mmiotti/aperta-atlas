"""
Switzerland-specific helpers shared across the survey-prep scripts in this folder.

Four groups:

  1. **Time-of-day / plausibility checks** (`is_peak`, `is_night`,
     `is_realistic_speed`, `is_not_too_long`) plus their Swiss-tuned defaults
     (`PEAK_HOURS_WEEKDAY`, `SPEED_ENVELOPES_KMH`, ...). The defaults assume
     Swiss commuting patterns and the MZMV/MOBIS mode taxonomy.

  2. **MZMV code → string mappings** (`wmittel2_to_str`, `f51300_to_str`,
     `f52900a_to_str`, `f51700w_to_str`). These translate the integer column
     codes of the Swiss Mikrozensus Mobilität und Verkehr survey into
     human-readable mode / purpose strings.

  3. **Mode simplification chain** (`MODE_TO_SIMPLIFIED`).

  4. **Emissions and externalities** (`get_gco2eq_per_km`, `get_gco2eq_per_km_car`,
     `get_gco2eq_per_km_not_car`, `get_ext_rp_per_km`). Switzerland-calibrated
     gCO2eq/km figures by mode, plus external cost (Rp/km) by mode.

These live here (not in `aperta/`) because they depend on Swiss-specific data: MZMV
column codes, Swiss electricity mix, Swiss external-cost calibrations, and Swiss
commuting / mode-name conventions.

Generic helpers that *used* to live in `aperta.surveys` (now removed) —
bin labels, group-ID assignment, lat/lon transforms, straight-line
distance, geometry simplification, time-zone attachment — landed in
`aperta_atlas.utils` (timing + small numeric helpers),
`aperta.data_processing` (`add_straight_line_dist` — still library-tier
since aperta.calibration uses it), and here (`add_lat_lon`,
`add_group_id`, `attach_time_zone` — atlas-local since surveys are
their only caller).

For Switzerland-wide constants used across multiple prep scripts (cities, coverage
fractions), see `SWISS_CITIES_100K_POP` and `FRACTION_COVERED`.
"""

import logging
import zoneinfo
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
    """Add `out_col` with sequential integer IDs based on unique values of `id_col`.

    Uses `pd.factorize` — O(n) and avoids a materialised replacement dict.
    """
    codes, _ = pd.factorize(df[id_col], sort=False)
    df[out_col] = codes
    return df


def attach_time_zone(dt: pd.Series, timezone_name: str) -> pd.Series:
    """Stamp every datetime in `dt` with `timezone_name`, returning a
    Series of ISO-format strings.

    Each value is taken to be a naive (tz-less) timestamp in the given
    zone (so '2021-04-15 14:30:00' + 'Europe/Zurich' becomes
    `'2021-04-15T14:30:00+02:00'`). Returns strings rather than
    tz-aware Timestamps so the column round-trips losslessly through
    CSV without pandas inferring a different dtype on reload.

    Was previously in the deleted `aperta.surveys` (as
    `attach_time_zone_to_pd_datetime`); lifted from
    `_archive/uma_surveys/common/survey_processing.py` and parked here
    because the only caller is `b_mzmv_process.py`.
    """
    tz = zoneinfo.ZoneInfo(timezone_name)
    return dt.apply(
        lambda v: v.to_pydatetime().replace(tzinfo=tz).isoformat()
    )


def bin_labels(bins: Sequence[int | float]) -> list[str]:
    """Human-readable labels for a `pd.cut` binning.

    Given right-closed bins like `(0, 6, 13, 18, 25, …, 120)` produces
    `['leq6', '7to13', '14to18', '19to25', …, '76+']` — one label per
    interval `(bins[i], bins[i+1]]`. Convention:

    - First bin: `'leq{bins[1]}'` (or literal `'0'` if `bins[1] == 0`).
    - Last bin:  `'{bins[-2]+1}+'` (open-ended, "and above").
    - Single-value bins (where `bins[i]+1 == bins[i+1]`) collapse to
      just the value, so `(0, 1, 5)` becomes `['leq1', '2to5']` and
      `(-1, 0, 1)` becomes `['0', '1+']`.

    Was previously in the deleted `aperta.surveys`; lifted from
    `_archive/uma_surveys/common/survey_processing.py` and parked here
    because the only callers live under `preparation/switzerland/surveys/`.
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
# to flag rows by peak/night/weekday and to drop implausible speeds.

# The slowest part of peak hours in the morning is quite short; higher impact
# in the afternoon.
PEAK_HOURS_WEEKDAY: tuple[int, ...] = (8, 16, 17, 18)
NIGHT_HOURS_WEEKDAY: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23)
NIGHT_HOURS_WEEKEND: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23)

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
# poisoning per-mode edge-weight calibration). Loosen only if a WARNING
# on drop rate suggests real trips are being lost.
SPEED_ENVELOPES_KMH: dict[str, tuple[float, float]] = {
    'walk':     (2.0,   8.0),  # 3-5 km/h typical; upper generous to admit GPS/rounding noise on brisk walks
    'rbike':    (5.0,  30.0),  # Mechanical bike; lower relaxed for Alpine climbs / urban stops
    'ebike25':  (7.0,  32.0),  # 25 km/h assist limit + margin; motor kicks in from a stop
    'ebike45':  (7.0,  47.0),  # 45 km/h assist limit + margin
    'anybike':  (5.0,  32.0),  # Unknown mechanical / ebike25 mix — lower matches rbike
    'anyebike': (7.0,  47.0),  # Unknown ebike25 / ebike45 mix — lower matches other ebikes
    'car':      (5.0, 130.0),  # 130 km/h Swiss highway top; slow urban floor
}


# Per-mode max plausible route detour ratio `dist_measured / dist_line`.
# Typical urban routes are ~1.2-1.6; ratios above the max flag "the trip
# involved a huge detour" (either the user made errands along the way
# and reported the full walked distance, or the router / survey took a
# path so far from the OD straight-line that the leg isn't representative
# for edge-weight fitting). Uniform 3.0 across modes — routing-constraint
# ordering (walk < bike < car) would push in the opposite direction from
# what a filter argument would want, and the variation is smaller than
# the safety margin we want anyway.
MAX_DETOUR_RATIO: dict[str, float] = {
    'walk':     3.0,
    'rbike':    3.0,
    'ebike25':  3.0,
    'ebike45':  3.0,
    'anybike':  3.0,
    'anyebike': 3.0,
    'car':      3.0,
}


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


def is_realistic_speed(mode: str,
                       speed_kmh: float,
                       envelopes: dict[str, tuple[float, float]] = SPEED_ENVELOPES_KMH) -> int:
    """1 if `speed_kmh` is in the realistic envelope for `mode`, else 0."""
    if mode not in envelopes:
        raise ValueError(
            f"Unknown mode {mode!r}; known: {sorted(envelopes)}. "
            f"Pass a custom `envelopes` dict to add modes.")
    lo, hi = envelopes[mode]
    return 1 if lo <= speed_kmh <= hi else 0


def is_not_too_long(duration_s: float, max_seconds: int = 7200) -> int:
    """1 if `duration_s` is at or below `max_seconds` (default 2 h), else 0."""
    return 1 if duration_s <= max_seconds else 0


# ---------------------------------------------------------------------------
# Population coverage and reference data
# ---------------------------------------------------------------------------

# MZMV only covers people age 6+, and age 6 is undersampled. Each age under 20
# contributes ~1% of the total population; the effective coverage is ~94%.
FRACTION_COVERED = 0.94

# Top 6 Swiss cities by population (>100k). Used to flag origin/destination subsets.
SWISS_CITIES_100K_POP = ('Lausanne', 'Zürich', 'Basel', 'Bern', 'Genève', 'Winterthur')


# ---------------------------------------------------------------------------
# MZMV code → string mappings
# ---------------------------------------------------------------------------

# Note: "Kleinmotorrad" (small motorbike) is combined with "Motorrad" here (both → 12,
# 'motorbike'). In `F51300_TO_STR` below, they're separately classified.
WMITTEL2_TO_STR: dict[int, str] = {
    -99: 'other',
    1: 'plane', 2: 'transit_rail', 3: 'boat', 4: 'transit_tram', 5: 'transit_bus',
    6: 'transit_other', 7: 'coach', 8: 'car', 9: 'truck', 10: 'taxi', 11: 'taxi',
    12: 'motorbike', 13: 'motorbike_small', 14: 'ebike25', 15: 'rbike', 16: 'walk',
    17: 'micro', 18: 'other',
}

# Note: "Kleinmotorrad" is separately classified as 'motorbike_small' here.
F51300_TO_STR: dict[int, str] = {
    -99: 'other',   # Pseudoetappe
    -98: 'other',   # Keine Antwort
    -97: 'other',   # Weiss Nicht
    95: 'other',
    1: 'walk', 2: 'rbike', 3: 'ebike25', 4: 'ebike45',
    5: 'motorbike_small', 6: 'motorbike_small', 7: 'motorbike', 8: 'motorbike',
    9: 'car', 10: 'car',
    11: 'transit_rail', 12: 'transit_bus', 13: 'transit_tram',
    14: 'taxi', 15: 'taxi', 16: 'coach', 17: 'truck', 18: 'boat', 19: 'plane',
    20: 'other', 21: 'micro',
}

F52900A_TO_STR: dict[int, str] = {
    -99: 'na', -98: 'na', -97: 'na',
    1: 'mode_transfer',       # Umsteigen / Verkehrsmittelwechsel / Auto abstellen
    2: 'work_commute',        # Arbeiten
    3: 'education',           # Ausbildung / Schule
    4: 'errands_groceries',   # Einkaufen
    5: 'errands_services',    # Besorgungen / Dienstleistungen
    6: 'work_business',       # Geschäftliche Tätigkeit
    7: 'work_business',       # Dienstfahrt
    8: 'leisure',             # Freizeitaktivität
    9: 'accompany',           # Begleitweg (Kinder)
    10: 'accompany',          # Begleitweg / Serviceweg (Andere)
    11: 'return',             # Rückkehr nach Hause
    12: 'other', 13: 'other',
}

F51700W_TO_STR: dict[int, str | float] = {
    -99: np.nan, -98: np.nan, -97: np.nan,
    1: 'leisure_visit',         # Besuche
    2: 'leisure_gastronomy',    # Restaurant / Bar / Café
    3: 'leisure_active',        # Aktiver Sport
    4: 'leisure_active',        # Wanderung
    5: 'leisure_other',         # Velofahrt
    6: 'leisure_amenity',       # Passiver Sport
    7: 'leisure_active',        # Nicht-sportliche Aussenaktivität (Spaziergang)
    8: 'leisure_amenity',       # Medizin / Wellness / Fitness
    9: 'leisure_amenity',       # Kultur / Freizeitanlagen
    10: 'leisure_other',        # Unbezahlte Arbeit
    11: 'leisure_other',        # Vereinstätigkeit
    12: 'leisure_other',        # Ausflug / Ferien
    13: 'leisure_amenity',      # Religion
    14: 'leisure_other',        # Häusliche Freizeitaktivitäten auswärts
    15: 'leisure_other',        # Essen ohne Gastronomiebesuch
    16: 'leisure_amenity',      # Einkaufsbummel / Shopping
    17: 'leisure_other',        # Rundreise
    18: 'leisure_other',        # Anderes
    22: 'leisure_other',        # mehrere Aktivitäten
    90: 'leisure_other',        # Anderes (legs)
}

MODE_TO_SIMPLIFIED: dict[str, str] = {
    'transit_rail': 'transit', 'transit_bus': 'transit', 'transit_tram': 'transit',
    'transit_other': 'transit',
    'motorbike_small': 'motorbike',
    'taxi': 'car',
}

# ---------------------------------------------------------------------------
# Plane-distance binning
# ---------------------------------------------------------------------------

def get_detailed_plane_mode(distance_m: float) -> str:
    """Map a single distance to a short/medium/long plane category."""
    if distance_m < 800_000:
        return 'plane_short'
    if distance_m < 3_000_000:
        return 'plane_medium'
    if distance_m < 99_000_000:
        return 'plane_long'
    return 'plane_na'


def vectorized_plane_mode(dist_m: np.ndarray) -> np.ndarray:
    """Vectorized counterpart to `get_detailed_plane_mode` for whole columns."""
    out = np.full(len(dist_m), 'plane_na', dtype=object)
    out[dist_m < 99_000_000] = 'plane_long'
    out[dist_m < 3_000_000] = 'plane_medium'
    out[dist_m < 800_000] = 'plane_short'
    return out


# ---------------------------------------------------------------------------
# Emissions (gCO2eq / km), Swiss-calibrated
# ---------------------------------------------------------------------------

def get_gco2eq_per_km(mode: str, speed_kmh: float, which: str, car_type: str,
                      region: str | None) -> float:
    """gCO2eq/km for a trip in `mode` at `speed_kmh`. `which` is 'operating', 'embodied',
    or 'total'. `car_type` and `region` only matter when `mode in ('car', 'taxi')`.
    """
    out = 0.0
    for w in ('operating', 'embodied'):
        if w == which or which == 'total':
            if mode in ('car', 'taxi'):
                out += get_gco2eq_per_km_car(speed_kmh, w, car_type, region)
            else:
                out += get_gco2eq_per_km_not_car(mode, speed_kmh, w)
    return out


# Per-mode emissions tables for non-car modes. Sources noted inline below.
_GCO2EQ_PER_KM_OPERATING: dict[str, float] = {
    'transit_rail': 8,           # Regionalverkehr incl. S-Bahn
    'transit_tram': 42,
    'transit_bus': 100,          # MOBI-tool quotes 171 for Diesel-Kleinbus
    'transit_other': 100,        # ~1% of transit; approximated
    'ebike25': 0, 'ebike45': 0, 'rbike': 0, 'walk': 0,
    'motorbike': 100, 'motorbike_small': 100,
    'micro': 30,                 # Estimate
    'coach': 50,                 # Estimate
    # Plane: Kerosene 2.63 kgCO2eq/L (NZ Min. Env.), 2.9 kgCO2eq/kg including aviation
    # radiative forcing (~+50%). bdl.aero baseline per pax-km × 0.9 (efficiency gain).
    'plane_short': 215,          # 5.5 L / 100 pax-km, <800 km
    'plane_medium': 133,         # 3.4 L / 100 pax-km, 800-3000 km
    'plane_long': 125,           # 3.2 L / 100 pax-km, >3000 km
    'truck': 500,                # Estimate
    'other': 0,
    'boat': 50,                  # Estimate
}

_GCO2EQ_PER_KM_EMBODIED: dict[str, float] = {
    'transit_rail': 0, 'transit_tram': 0, 'transit_bus': 0, 'transit_other': 0,
    'ebike25': 9, 'ebike45': 11, 'rbike': 6,    # MOBI-tool v3
    'walk': 0,
    'motorbike': 10, 'motorbike_small': 10,
    'micro': 20,                  # Estimate, low lifetime km
    'coach': 0,
    'plane_short': 0, 'plane_medium': 0, 'plane_long': 0,
    'truck': 0, 'other': 0, 'boat': 0,
}


def get_gco2eq_per_km_not_car(mode: str, speed_kmh: float, which: str) -> float:
    table = _GCO2EQ_PER_KM_OPERATING if which == 'operating' else _GCO2EQ_PER_KM_EMBODIED
    co2 = table[mode]
    if co2 < 0:
        # MOBI-tool: 186 g per pax-km avg; Carboncounter: ~300 per v-km (~215 per
        # pax-km). Anchor on ~200 baseline, with deviation from 80 km/h sweet spot.
        co2 = 150 + abs(80 - speed_kmh) * 1.5
    return co2


def get_gco2eq_per_km_car(speed_kmh: float, which: str, car_type: str,
                          region: str | None) -> float:
    """
    Mobi Tool ~186 g/pax-km avg; Carboncounter ~300 g/v-km (~215 g/pax-km).
    Carculator (tailpipe + rest operating including road + vehicle production):
      ICEV Medium Gasoline:  169 + 62 + 53 = 284
      ICEV Medium Diesel:    155 + 58 + 54 = 267
      BEV Medium:             0 + 49 + 87  = 136
    Top-down 2024 numbers: ~10 t tailpipe / passenger-car sector; ~4.8 M cars × ~10.3k
    km/yr ≈ 49.9 Mvkm/yr domestic → ~200 g/v-km (tailpipe). Newly-sold 2023 Gas-ICE
    ≈ 6.8 L/100 km; Diesel ≈ 6.2; Diesel share <15% ⇒ mix ≈ 6.6 L/100 km ≈ 160 g/km.
    Gap to 200 reflects WLTP-vs-real consumption. Target tailpipe: 190 g/km.
    """
    if which == 'embodied':
        lifetime_km = 200_000
        return {'icev_mix': 10.6, 'bev_mix': 17.4}[car_type] / lifetime_km * 1e6
    if which == 'operating':
        if car_type.startswith('icev'):
            tailpipe_to_operating = 1.37
            ref = {'mix': 1.0}[car_type[car_type.index('_') + 1:]]
            # +60% consumption at 16 km/h, +30% at 130 km/h, ~min at 80 km/h.
            factor = (80 - speed_kmh) ** 2 / 7_000
            avg_factor = (80 - 44) ** 2 / 7_000          # Avg travel speed ≈ 44 km/h
            return 190 / (1 + avg_factor) * (1 + factor) * tailpipe_to_operating * ref
        if car_type.startswith('bev'):
            ref = {'mix': 49}[car_type[car_type.index('_') + 1:]]
            factor = {
                'ch': 1.1,                     # ~75 g/km Carculator + 10% aux/charging
                'global': (280 / 75) * 1.1,    # ~280 g/km global mix + 10% aux/charging
            }[region]
            return ref * factor
    return np.nan


# ---------------------------------------------------------------------------
# Externalities (Rp / km), Swiss-calibrated
# ---------------------------------------------------------------------------

_EXT_RP_PER_KM: dict[str, float] = {
    'transit_rail': 1, 'transit_tram': 3, 'transit_bus': 20, 'transit_other': 20,
    'car': 12, 'taxi': 12,
    'ebike25': -28, 'ebike45': -28, 'rbike': -36, 'walk': -95,
    'motorbike': 40, 'motorbike_small': 105,
    'micro': 17, 'coach': 15,
    'plane_short': -999_999, 'plane_medium': -999_999, 'plane_long': -999_999,
    'truck': 50, 'other': 0, 'boat': 109,
}


def get_ext_rp_per_km(mode: str, speed_kmh: float) -> float:
    """External cost in Swiss Rappen / km for `mode`. `speed_kmh` unused but kept for
    API symmetry with the emissions functions (some external costs could be
    speed-dependent in future).
    """
    return _EXT_RP_PER_KM[mode]


# ---------------------------------------------------------------------------
# Survey config (region-agnostic but accessed alongside the Swiss helpers)
# ---------------------------------------------------------------------------

ALLOWED_SURVEY_PREFIXES: tuple[str, ...] = (
    'time_', 'dist_', 'speed_', 'timestamp_', 'datetime_', 'hour_', 'mode_', 'peak_',
    'is_', 'metric_', 'gco2eq_', 'kgco2eq_', 'externalities_', 'purpose', 'sd_',
    'orig_', 'dest_', 'n_', 'weight_', 'route_', 'include_', 'group_', 'subset_',
    'elev_',  # endpoint elevation (`elev_orig`, `elev_dest` from DEM sample)
    'municipality_id', 'canton_id', 'zone_id', 'longest_leg_id', 'hh_',
    # Cross-survey leg / trip / person identifiers (replace MOBIS's
    # former `'_id' in c` substring catch-all — see `standardize.py`
    # docstring for the intentional unification).
    'trip_id', 'leg_id', 'participant_id',
)

# Bike family: rbike / ebike25 / ebike45 = confirmed sub-types.
# anybike = mixed/unknown (MOBIS pre-split, Bikesharing).
# anyebike = ebike with unknown assist limit (MOBIS Ebicycle).
# See `mode_simplified` docstring in `standardize.py` and the mapping
# notes at the top of `b_mobis_process.py`.
MODES_MODE_CHOICE_MODEL = (
    'walk', 'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike', 'car', 'transit')
MODES_TRAVEL_TIME_MODEL = (
    'walk', 'rbike', 'ebike25', 'ebike45', 'anybike', 'anyebike', 'car')

# If max geometry size is 500 and min precision is 250 m, trajectories longer than
# ~125 km will be excluded due to too many points after simplification.
TARGET_GEOMETRY_SIZE = 100
MAXIMUM_GEOMETRY_SIZE = 500


# ---------------------------------------------------------------------------
# Swiss-specific spatial-subset post-processing
# (run after the trip endpoints have been spatially joined against zone geometries)
# ---------------------------------------------------------------------------

def add_swiss_city_subsets_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `subset_*_within` / `subset_*_border` columns for trips whose
    origin AND destination are inside (within) or have at least one endpoint inside
    (border) one of the top-6 Swiss cities.

    Requires endpoint-suffixed columns `municipality_id_orig`, `municipality_id_dest`,
    `metro_orig`, `metro_dest` (set upstream by a per-endpoint spatial join).
    """
    for level, label in (('municipality_id', 'municipalities_100k'), ('metro', 'metros_100k')):
        orig_col = f'{level}_orig'
        dest_col = f'{level}_dest'
        is_orig_city = trips[orig_col].isin(SWISS_CITIES_100K_POP)
        is_dest_city = trips[dest_col].isin(SWISS_CITIES_100K_POP)
        trips[f'subset_{label}_within'] = (is_orig_city & is_dest_city).astype(int)
        trips[f'subset_{label}_border'] = (is_orig_city | is_dest_city).astype(int)
        logging.info(
            f"...trips within top-6 cities ({level}): "
            f"{trips[f'subset_{label}_within'].sum():,}; "
            f"border: {trips[f'subset_{label}_border'].sum():,}")
    return trips


def add_swiss_location_subsets(df: pd.DataFrame, prefix: str = 'hh') -> pd.DataFrame:
    """Add Swiss-specific location-subset columns prefixed by `<prefix>_`:
    `is_in_metro`, `is_in_100k_metro`, `is_in_plateau`, `is_in_extended_plateau`.

    Requires `<prefix>_metro`, `<prefix>_canton_id`, and a top-level `lfi_region_id`
    column (set upstream by a per-location spatial join with the appropriate
    extra-zone columns).
    """
    metro_col = f'{prefix}_metro'
    canton_col = f'{prefix}_canton_id'
    df[f'{prefix}_is_in_metro'] = df[metro_col].notna().astype(int)
    df[f'{prefix}_is_in_100k_metro'] = df[metro_col].isin(SWISS_CITIES_100K_POP).astype(int)
    df[f'{prefix}_is_in_plateau'] = (
        (df['lfi_region_id'] == 'Plateau') & (df[canton_col] != 'GE')
    ).astype(int)
    df[f'{prefix}_is_in_extended_plateau'] = (
        df['lfi_region_id'].isin(['Plateau', 'Pre-Alps']) & (df[canton_col] != 'GE')
    ).astype(int)
    n_metro = df[f'{prefix}_is_in_metro'].sum()
    logging.info(f"Households in any metro area: {n_metro:,} of {len(df):,}")
    return df
