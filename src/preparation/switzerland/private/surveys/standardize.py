"""
Survey-standardization helpers shared by `b_mzmv_process.py` and
`b_mobis_process.py`. Each helper acts on an already-normalized legs
DataFrame with the canonical column set — coordinates as
`orig_x/orig_y/dest_x/dest_y` (LV95), `hour_of_day`, `sd_bool_weekday`,
`mode_simplified`, `time_measured`, `dist_measured`, `speed_kmh`.
Survey-specific adapters run first to bring raw source columns onto
this schema; then these helpers run identically on both.

Design principles:

  - `peak_str` middle bucket is `'base'` (not `'offpeak'`), and the
    corresponding column is `hour_base` (not `hour_offpeak`). Matches
    `04_edge_weights.py`'s car-bucket naming.

  - Column filter is prefix-only (`ALLOWED_SURVEY_PREFIXES`). No
    `'_id' in c` substring catch-all. Explicit ID columns downstream
    consumers need must be in the prefix list.

  - Flag names describe PROPERTIES of the trip (e.g.
    `is_within_speed_envelope`), never a consumer (avoid names like
    `include_in_<downstream>_model`). Consumers compose the flag set
    they need at the call site, with an inline mode-set filter.

Row-level flag inventory (all 0/1, all present on every leg in
legs.csv — no row removals from these flags):

    is_valid_dist_time           — dist / time / dist_line (survey-specific
                                    illegal-value checks; MOBIS also
                                    requires a non-null trip_id)
    is_in_switzerland            — origin AND destination in CH
    is_plausible                 — MOBIS only: `implausible` field False
    is_valid_domestic_trip       — composite AND of all per-reason
                                    validity flags above (the "domestic"
                                    qualifier makes explicit that scope
                                    is Swiss-only)
    is_within_speed_envelope     — implied speed within per-mode envelope
                                    (mode-misclassification detector; trip
                                    length is a consumer concern, not
                                    baked into this flag)
    is_within_detour_envelope    — `dist_measured / dist_line` within
                                    per-mode cap (catches errand-stops
                                    where the reported distance reflects
                                    the full walked path rather than the
                                    OD-relevant path)
    elev_orig, elev_dest         — endpoint elevation in metres ASL,
                                    sampled from the Copernicus DEM at
                                    each leg's orig / dest centroids
                                    (NaN outside DEM extent)
    is_within_elevation_band     — composite gate: both endpoints have
                                    valid DEM samples AND max endpoint
                                    is below the alpine threshold
                                    (default 1000 m). Drops DEM-boundary
                                    artifacts + alpine trips in one flag
    is_land_based                — mode ∉ {plane, boat, aerialway}
                                    (both surveys)
    include_in_route_matching    — has usable route geometry

Person-level cascade (`is_person_diary_clean`) is INTENTIONALLY NOT
pre-computed. Cascade rules depend on the analysis question (a trip
outside Switzerland is out-of-scope, not "bad data"), so it's a
downstream operation. Call `person_has_any_bad_trip` to compute it
against whichever combination of flags matches your analysis.

Row REMOVALS (before flagging) — separate concern, applied by the two
`b_*_process.py` scripts:

    MOBIS: cohort filter (`--variant pre_covid` OR `--variant covid`)
    MZMV:  age filter (drops `sd_ordinal_age == 'leq6'` respondents)

Downstream compositions are documented at the consumer, not enforced
here. Typical shapes:

    # Edge-weight calibration training:
    is_valid_domestic_trip & is_within_speed_envelope
                           & is_within_detour_envelope
                           & mode ∈ {training modes}

    # Mode-choice model training:
    is_valid_domestic_trip & mode ∈ {mode-choice alternatives incl. transit}

    # Person-level statistics (drop persons with any invalid trip):
    bad = person_has_any_bad_trip(legs, person_col='participant_id',
        exclude_masks={'invalid': legs['is_valid_dist_time'] == 0})
    person_stats = legs[~bad].groupby(...).agg(...)

The 04-side `_apply_survey_gates` in `main/04_edge_weights.py` is the
canonical example of composing trip-quality gates.
"""

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

import geopandas as gpd

from aperta import data_processing, geo_processing

from preparation.switzerland.common import CRS_CH, CRS_LATLON
from preparation.switzerland.private.surveys import common
from preparation.switzerland.private.surveys.common import (
    ALLOWED_SURVEY_PREFIXES, MAXIMUM_GEOMETRY_SIZE,
    MAX_DETOUR_RATIO,
    MODES_MODE_CHOICE_MODEL, MODES_TRAVEL_TIME_MODEL,
    SPEED_ENVELOPES_KMH,
)


def attach_hour_flags(legs: pd.DataFrame) -> None:
    """Set `hour_peak`, `hour_night`, `hour_base`, `peak_str` in place.

    Requires `hour_of_day` (int, 0-23 or -1) and `sd_bool_weekday`
    (0/1) already on `legs`. Peak/night windows come from
    `common.is_peak` / `common.is_night` (Swiss-tuned defaults).
    `peak_str ∈ {'peak', 'base', 'night'}` — matches
    `04_edge_weights.py`'s car-bucket names.
    """
    legs['hour_peak'] = common.is_peak(legs)
    legs['hour_night'] = common.is_night(legs)
    legs['hour_base'] = (
        (legs['hour_peak'] == 0) & (legs['hour_night'] == 0)
    ).astype(int)
    legs['peak_str'] = np.select(
        [legs['hour_peak'] == 1, legs['hour_base'] == 1],
        ['peak', 'base'],
        default='night',
    )


def attach_lat_lon_and_dist_line(
    legs: pd.DataFrame,
    crs_main: str = CRS_CH,
    crs_latlon: str = CRS_LATLON,
) -> pd.DataFrame:
    """Add `orig_lat/orig_lon`, `dest_lat/dest_lon`, and `dist_line`
    (straight-line orig→dest in metres) to `legs`. Requires
    `orig_x/orig_y/dest_x/dest_y` in `crs_main`."""
    legs = common.add_lat_lon(legs, 'orig', crs_main, crs_latlon)
    legs = common.add_lat_lon(legs, 'dest', crs_main, crs_latlon)
    legs = data_processing.add_straight_line_dist(legs)
    return legs


def attach_endpoint_elevation(
    legs: pd.DataFrame,
    *,
    dem_path: str,
) -> None:
    """Sample the DEM raster at each leg's origin + destination centroids;
    write `elev_orig` and `elev_dest` (metres above sea level) in place.
    Endpoints outside the DEM extent get NaN.

    Requires `orig_lat`, `orig_lon`, `dest_lat`, `dest_lon` on `legs`
    (as produced by `attach_lat_lon_and_dist_line`). The DEM at
    `dem_path` must be a single-band raster in WGS84 (matches the
    convention set by `preparation/world/elevation/elevation_dem.py`).

    Downstream can compose (e.g. `elev_net = elev_dest - elev_orig`
    for net climb/descent). The `is_within_elevation_band` gate for
    quality + alpine exclusion is written separately by
    `attach_within_elevation_band_flag`.
    """
    for side in ('orig', 'dest'):
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                legs[f'{side}_lon'], legs[f'{side}_lat']),
            index=legs.index,
            crs='EPSG:4326',
        )
        legs[f'elev_{side}'] = geo_processing.sample_raster_at_points(
            points, dem_path, name=f'elev_{side}')
    n_orig_ok = int(legs['elev_orig'].notna().sum())
    n_dest_ok = int(legs['elev_dest'].notna().sum())
    logging.info(
        f"  Endpoint elevation from DEM: "
        f"{n_orig_ok:,}/{len(legs):,} orig, "
        f"{n_dest_ok:,}/{len(legs):,} dest have valid elev "
        f"(NaN = outside DEM extent)")


def attach_within_elevation_band_flag(
    legs: pd.DataFrame,
    *,
    min_m: float = 0.0,
    max_m: float = 1_000.0,
) -> None:
    """Write `is_within_elevation_band` (0/1) onto `legs` in place. A
    trip is flagged 1 iff BOTH endpoints have valid elevation samples
    satisfying `min_m < elev < max_m`. Composite of two concerns:

      - DATA QUALITY: drops elev == 0 (DEM boundary artifacts) and NaN
        (endpoint outside DEM extent). Non-negotiable — invalid data.
      - SCOPE: drops alpine trips (max endpoint above `max_m`). Reason:
        alpine trips are heavily biased by fit self-selected cyclists /
        hikers (person-fitness confound) — leaving them out gives
        cleaner edge-weight, overhead, and utility coefficients.

    Consumed by `main/04_edge_weights.py`, `main/07a_road_overhead_coefs.py`,
    and `survey/02d_prepare_survey_legs.py` (which propagates the
    filter to any downstream reading `survey_legs.csv`, e.g. 09a/09b).

    Requires `elev_orig` and `elev_dest` on `legs` (written by
    `attach_endpoint_elevation`). Defaults calibrated for Swiss data
    (max_m=1000 covers the mid-plateau + foothills but drops the Alps).

    NaN handling: `skipna=False` on min/max so NaN endpoints propagate
    to False (drop the trip). A single missing endpoint disqualifies.
    """
    if not {'elev_orig', 'elev_dest'}.issubset(legs.columns):
        raise ValueError(
            "attach_within_elevation_band_flag requires 'elev_orig' and "
            "'elev_dest' columns — call attach_endpoint_elevation first.")
    elev_min = legs[['elev_orig', 'elev_dest']].min(axis=1, skipna=False)
    elev_max = legs[['elev_orig', 'elev_dest']].max(axis=1, skipna=False)
    keep = (elev_min > min_m) & (elev_max < max_m)
    legs['is_within_elevation_band'] = keep.astype(int)
    n_kept = int(keep.sum())
    logging.info(
        f"  Elevation band [{min_m:.0f}, {max_m:.0f}] m: "
        f"{n_kept:,}/{len(legs):,} legs in-band "
        f"(dropped: NaN, elev<={min_m:.0f}, elev>={max_m:.0f})")


def attach_validity_flags(
    legs: pd.DataFrame,
    *,
    validity_checks: dict[str, pd.Series],
) -> None:
    """Write a per-reason `is_<name>` flag column PER `validity_checks`
    entry AND the composite `is_valid_domestic_trip` (AND of all),
    all in place.

    `validity_checks`: dict of `{flag_name: EXCLUDE_mask}`. Each entry
    becomes `legs[flag_name] = (~mask).astype(int)`. Conventional names:

        'is_valid_dist_time'   — dist / time / dist_line-in-country OK
        'is_in_switzerland'    — origin AND destination in CH
        'is_plausible'         — (MOBIS only) `implausible` field False

    The composite `is_valid_domestic_trip` is the AND of every
    per-reason flag. "Domestic" makes the scope explicit — the flag
    fires only for trips within Switzerland with no data-quality
    issues. Downstream analyses that want a broader scope (e.g.
    including foreign trips for person-level counts) compose the
    per-reason flags directly.

    Mode filtering is INTENTIONALLY not part of any of these flags —
    consumers filter `mode_simplified` inline against whatever mode set
    they care about (edge-weight training, mode-choice training, person
    stats).

    Log line per reason: CUMULATIVE excluded-row count (only rows this
    filter contributes on top of previous filters — order-dependent, so
    the sum matches the total excluded).
    """
    exclude = pd.Series(False, index=legs.index)
    for flag_name, mask in validity_checks.items():
        legs[flag_name] = (~mask).astype(int)
        n_new = int((~exclude & mask).sum())
        logging.info(f"  Filter {flag_name}: excluding {n_new:,} new rows")
        exclude = exclude | mask
    legs['is_valid_domestic_trip'] = (~exclude).astype(int)
    logging.info(
        f"  → is_valid_domestic_trip == 1: {(~exclude).sum():,} "
        f"of {len(legs):,} trip(s)")


def attach_include_in_route_matching(
    legs: pd.DataFrame,
    max_size: int = MAXIMUM_GEOMETRY_SIZE,
) -> None:
    """Set `include_in_route_matching` from `route_geo_size` in place.
    Requires `route_geo_size` (int) already on `legs` — either mapped
    from a per-leg geometry-size lookup (MZMV) or renamed from a
    trajectories join (MOBIS)."""
    legs['include_in_route_matching'] = (
        (legs['route_geo_size'] > 0) & (legs['route_geo_size'] <= max_size)
    ).astype(int)


def filter_columns_by_prefix(
    legs: pd.DataFrame,
    prefixes: tuple[str, ...] = ALLOWED_SURVEY_PREFIXES,
) -> pd.DataFrame:
    """Return `legs` with only columns whose name starts with any
    entry in `prefixes`. Index is preserved."""
    cols = [c for c in legs.columns if c.startswith(prefixes)]
    return legs[cols]


def attach_speed_envelope_flag(
    legs: pd.DataFrame,
    *,
    envelopes: dict[str, tuple[float, float]] = SPEED_ENVELOPES_KMH,
    short_trip_dist_m: float | None = None,
) -> pd.Series:
    """Write `is_within_speed_envelope` (0/1) onto `legs` in place; also
    return the exclude mask (== `is_within_speed_envelope == 0`) so
    callers can compose it into a person-level cascade or other filter.

    The flag captures "implied speed is within the mode's envelope".
    A leg is flagged in-envelope iff `speed_kmh` is within the
    `(min, max)` envelope for its `mode_simplified`. Catches mode
    misclassifications (walk at 20 km/h → actually bike; bike at 60
    km/h → actually car). Modes not in `envelopes` (transit,
    motorbike, etc.) are treated as in-envelope (flag stays 1).

    Requires `mode_simplified`, `speed_kmh`, `dist_measured` on `legs`.

    `short_trip_dist_m`: for surveys with self-reported times rounded
    to a coarse grid (e.g. MTMC's 5-min steps + trip-overhead
    inclusion), short trips can produce artificially LOW implied
    speeds that hit the LOWER envelope bound spuriously. When set,
    trips shorter than this threshold skip the lower bound; only the
    upper bound applies. Pass `None` (default) for precise
    GPS-tracked surveys like MOBIS.

    Absolute trip-length filtering is intentionally NOT part of this
    flag — it's the CONSUMER's per-mode concern. `04_edge_weights.py`
    sets `min_trip_distance` per profile (100 m walk, 250 m bike,
    500 m car), which acts as the effective minimum for training.

    Logs per-mode: pre-envelope n, min/max/mean/median observed
    speed, envelope bounds, and out-of-envelope count + rate.

    Downstream: leaves `is_valid_domestic_trip` unchanged — consumers filter
    on this flag independently (see `04_edge_weights.py`).
    """
    exclude = pd.Series(False, index=legs.index)

    is_short_leniency = (
        legs['dist_measured'] < short_trip_dist_m
        if short_trip_dist_m is not None
        else pd.Series(False, index=legs.index)
    )
    for mode, (lo, hi) in envelopes.items():
        in_mode = legs['mode_simplified'] == mode
        n_mode = int(in_mode.sum())
        if n_mode == 0:
            continue
        speeds = legs.loc[in_mode, 'speed_kmh']
        s_min, s_max = float(speeds.min()), float(speeds.max())
        s_mean, s_med = float(speeds.mean()), float(speeds.median())

        over_upper = in_mode & (legs['speed_kmh'] > hi)
        under_lower = in_mode & ~is_short_leniency & (legs['speed_kmh'] < lo)
        bad_env = over_upper | under_lower
        n_over = int(over_upper.sum())
        n_under = int(under_lower.sum())
        n_bad = int(bad_env.sum())
        pct = 100.0 * n_bad / n_mode
        short_note = (
            f" (short-trip leniency: lower bound skipped below "
            f"{short_trip_dist_m:.0f} m)" if short_trip_dist_m is not None else ""
        )
        logging.info(
            f"  Speed envelope [{mode}]: n={n_mode:,}  "
            f"observed {s_min:.1f}-{s_max:.1f} km/h "
            f"(mean {s_mean:.1f}, median {s_med:.1f})  "
            f"envelope ({lo}, {hi}) km/h  → out-of-envelope {n_bad:,} "
            f"({pct:.1f}%): over_upper={n_over:,}, under_lower={n_under:,}"
            f"{short_note}")
        exclude = exclude | bad_env
    legs['is_within_speed_envelope'] = (~exclude).astype(int)
    logging.info(
        f"  → is_within_speed_envelope == 1: {(~exclude).sum():,} "
        f"of {len(legs):,} trip(s)")
    return exclude


def attach_within_detour_envelope_flag(
    legs: pd.DataFrame,
    *,
    max_ratios: dict[str, float] = MAX_DETOUR_RATIO,
) -> pd.Series:
    """Write `is_within_detour_envelope` (0/1) onto `legs` in place;
    also return the exclude mask (`is_within_detour_envelope == 0`).

    The flag captures "route length is within a plausible detour ratio
    of the OD straight-line". A leg passes iff
    `dist_measured / dist_line <= max_ratios[mode_simplified]`.

    Catches legs where the person made stops / errands along the way
    (self-reported `dist_measured` reflects the full walked/biked/driven
    path, not the OD-relevant distance) OR where the survey routed the
    leg via an implausibly long detour. Both cases would poison
    per-mode edge-weight calibration in `04_edge_weights.py` by
    inflating the observed time-per-metre ratio.

    Requires `mode_simplified`, `dist_measured`, `dist_line` on `legs`.
    Modes not in `max_ratios` (transit, etc.) are treated as
    in-envelope. Legs with NaN in `dist_measured` or `dist_line` are
    also treated as in-envelope (defer to other filters).

    Logs per-mode: pre-envelope n, mean/median observed detour ratio,
    max cap, and out-of-envelope count + rate.
    """
    exclude = pd.Series(False, index=legs.index)
    dist_line = legs['dist_line'].astype(float)
    dist_measured = legs['dist_measured'].astype(float)
    detour = dist_measured / dist_line.where(dist_line > 0)
    for mode, max_ratio in max_ratios.items():
        in_mode = legs['mode_simplified'] == mode
        n_mode = int(in_mode.sum())
        if n_mode == 0:
            continue
        mode_detour = detour.loc[in_mode]
        d_mean = float(mode_detour.mean())
        d_med = float(mode_detour.median())
        # NaN detour → not excluded (upstream/other filters handle those)
        bad_env = in_mode & (detour > max_ratio).fillna(False)
        n_bad = int(bad_env.sum())
        pct = 100.0 * n_bad / n_mode
        logging.info(
            f"  Detour envelope [{mode}]: n={n_mode:,}  "
            f"observed ratio mean {d_mean:.2f}, median {d_med:.2f}  "
            f"cap {max_ratio:.1f}  → out-of-envelope {n_bad:,} ({pct:.1f}%)")
        exclude = exclude | bad_env
    legs['is_within_detour_envelope'] = (~exclude).astype(int)
    logging.info(
        f"  → is_within_detour_envelope == 1: {(~exclude).sum():,} "
        f"of {len(legs):,} trip(s)")
    return exclude


def person_has_any_bad_trip(
    legs: pd.DataFrame,
    *,
    person_col: str,
    exclude_masks: dict[str, pd.Series],
) -> pd.Series:
    """Return a per-trip bool `pd.Series`: True for trips of persons who
    have ANY trip matching any of `exclude_masks`. Downstream analyses
    that need person-diary integrity call this to filter out biased
    diaries. Definition of "diary integrity" is the caller's — pass
    whichever exclusion masks match the analysis's concept of "bad".

    This is a DOWNSTREAM helper, not called during prep — the survey
    scripts don't bake any specific cascade rule into legs.csv. A
    person analyst using `is_valid_dist_time`, `is_within_speed_envelope`,
    etc. composes their own cascade at analysis time.

    Example (person-level trip counts, excluding respondents with any
    data-quality problem):

        bad = standardize.person_has_any_bad_trip(
            legs, person_col='participant_id',
            exclude_masks={'invalid_dist_time': legs['is_valid_dist_time'] == 0},
        )
        trip_counts = legs[~bad].groupby('participant_id').size()

    Logs per mask: bad persons + cascaded-trip counts.
    """
    person = legs[person_col]
    combined_bad = pd.Series(False, index=legs.index)
    for name, mask in exclude_masks.items():
        bad_persons_mask = person.isin(person[mask].unique())
        n_pers = int(person[mask].nunique())
        n_trips_cascaded = int(bad_persons_mask.sum())
        n_trips_direct = int(mask.sum())
        n_trips_extra = n_trips_cascaded - n_trips_direct
        logging.info(
            f"  Person cascade [{name}]: {n_pers:,} bad person(s) → "
            f"{n_trips_cascaded:,} trips ({n_trips_direct:,} direct + "
            f"{n_trips_extra:,} cascaded)")
        combined_bad = combined_bad | bad_persons_mask
    n_bad_persons = int(person[combined_bad].nunique())
    n_total_persons = int(person.nunique())
    logging.info(
        f"  → {n_bad_persons:,} of {n_total_persons:,} person(s) marked bad; "
        f"{int(combined_bad.sum()):,} of {len(legs):,} trip(s) cascaded")
    return combined_bad
