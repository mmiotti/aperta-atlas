"""
Calibrate per-profile per-edge weights (durations) against per-mode
ground-truth trip times for the Swiss Urban Mobility Atlas.

Ground truth per mode is set by `scenario.edge_weights_sources[mode]`:
  'mtmc'            MZMV 2015 + 2021 legs (self-reported durations).
  'mobis_precovid'  MOBIS pre-2020 GPS-tracked legs (leg-suitability filter applied).
  'mobis_covid'     MOBIS 2020-wk13 through 2022-03 (COVID-era; anomalous traffic regime).

Blending sources is intentionally NOT supported — MOBIS otherwise
dominates by sample count. Pick one source per mode.

For walk (rwalk) + car (peak / base / night), each fitted profile is fit
independently. The three car variants share the network + prior
coefficients — only the time-of-day bucket filtering the legs differs
(survey prep's `peak_str`, the same label every downstream car consumer
uses).

For bike, only the regular bike (rbike) is fit. Ebike profiles
(ebike25 / ebike45) are DERIVED from rbike's fit by:
  1. Replacing rbike's uniform baseline speed with a per-edge baseline
     of `min(target_speed_kph, OSM_speed_kph)` — e.g. 22 km/h (ebike25)
     or 40 km/h (ebike45), each capped by the OSM-derived speed limit.
  2. Scaling rbike's fitted feature coefficients by per-feature
     multipliers (see `_EBIKE_DERIVATIONS`). Examples:
       elevation_gain × 0.33  (motor assist eases climbs)
       elevation_loss × 0.50 (ebike25) / 0 (ebike45) (downhill help shrinks)
       is_4way / is_traffic_signal × 1.30 (ebike25) / 1.80 (ebike45)
                                          (faster baseline → bigger relative
                                           cost of stop/start at intersections)
  3. Capping each edge's duration at the slower bicycle profiles'
     (`not_slower_than`): ebike25 ≤ rbike, ebike45 ≤ ebike25 and rbike.
All three bike columns are written to a single `edges_bike_calibrated.csv`
(mirroring car's multi-profile-in-one-file pattern). Edit
`_EBIKE_DERIVATIONS` to tune the ebike rules. Walk's `walk_prm` (persons
with reduced mobility) is derived from rwalk the same way
(`_WALK_DERIVATIONS`: 3.5 km/h, elevation coefs × 3, steps and
slopes > 20 % impassable).

Inputs (PRIVATE) — subset actually read depends on the scenario's
`edge_weights_sources` mapping:
    preparation/switzerland/surveys/mzmv_2015/legs.csv         # 'mtmc'
    preparation/switzerland/surveys/mzmv_2021/legs.csv         # 'mtmc'
    preparation/switzerland/surveys/mobis_precovid/legs.csv    # 'mobis_precovid'
    preparation/switzerland/surveys/mobis_covid/legs.csv       # 'mobis_covid'
Inputs (scenario storage, under <scenario>/):
    nw/<mode>.graphml + properties/nodes_<mode>_core.csv       # from 02a
    properties/edges_<mode>_{core,from_nodes,extended}.csv     # from 02a / 02b
    properties/edges_car_flows.csv                             # from 03b (car only)

Outputs (scenario storage, under <scenario>/):
    properties/edges_<mode>_calibrated.csv
        # duration_calibrated_<profile> + effective_speed_kph_<profile>
        # + duration_baseline_<profile> for every profile under <mode>.
        # Bike's file has 3 profile columns each (rbike, ebike25, ebike45);
        # walk's 2 (rwalk, walk_prm).
        # `duration_baseline_*` is the per-edge naive baseline duration
        # (length / baseline_speed): OSM `speed_kph` for car; uniform
        # `cfg.baseline_speed_kph` for walk / bike. Consumed by
        # survey/05 for the naive-routing comparison in validation.
    coefs/<kind>/edge_weights_<mode>.csv
        # <kind> ∈ {calibrated, transferred, manual} per the scenario's
        # `coefs={'edge_weights_<mode>': ...}` declaration (see
        # README "Coefficients" + aperta_atlas.coefs).
        # One column per profile; bike's / walk's file shows the fitted
        # parent's coefs alongside the rule-derived ones.

Run all variants (walk / bike / car) sequentially (default):
    python -m main.04_edge_weights --scenario <name>
Single mode:
    python -m main.04_edge_weights --scenario <name> --variant car
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from aperta import calibration, routing_prep
from aperta_atlas import coefs
from aperta_atlas.context import Storage, _edge_id, init_context
from aperta_atlas.utils import step
from aperta_atlas.variant import Variants
from aoi_filter import filter_legs_by_aoi, load_aoi_polygon
from scenarios import get_scenario


_N_ITERATIONS = 5
# Minimum R² improvement per iteration to accept the update and continue;
# fits usually stop before `_N_ITERATIONS`. Lower to keep iterating past
# marginal gains; raise to stop earlier.
_R2_TOLERANCE = 1e-5


@dataclass(frozen=True)
class DerivedFromCalibrated:
    """Rules to derive a variant profile's per-edge durations from a fitted
    parent (rbike → ebike25/ebike45; rwalk → walk_prm).

    Per-edge effective speed = clip(min(`target_speed_kph`, OSM speed_kph),
    [min_speed_kph, max_speed_kph]).

    `coef_multipliers` — `{feature: factor}`. Each parent-profile
    coefficient is multiplied by `factor` (1.0 for any feature not listed).
    Applies to both multiplier features and additive route features.
    `baseline_time` is NOT scaled here — the speed change lives in
    `target_speed_kph`.

    `edge_exclusions` — list of `edges_df → bool Series` predicates. Any
    edge matching any predicate gets duration = inf (impassable). Each
    callable receives the edges GeoDataFrame with post-calibration
    attributes attached; returns an index-aligned bool Series.

    `not_slower_than` — profiles derived or fitted earlier whose per-edge
    durations cap this profile's (an e-bike can always ride like a regular
    bicycle), e.g. where the OSM speed cap or the larger stop penalties
    would otherwise make it slower.
    """
    profile: str
    target_speed_kph: float
    min_speed_kph: float
    max_speed_kph: float
    coef_multipliers: dict[str, float] = field(default_factory=dict)
    edge_exclusions: list[Callable[[pd.DataFrame], pd.Series]] = field(
        default_factory=list)
    not_slower_than: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileCalibration:
    """Per-profile calibration setup. `profile` is the output identifier
    (e.g. 'car_peak'); `mode` is the graph mode ('walk' / 'bike' / 'car'),
    so the three car variants all share one graph. Coefficient dicts are
    starting points for the OLS iteration. `derived_profiles` runs after
    the OLS fit — used by bike to produce ebike25 / ebike45 alongside
    rbike, and by walk to produce walk_prm.
    """
    profile: str | list[str]
    mode: str
    constant: float | None
    baseline_speed_kph: float | None  # None ⇒ use graph's per-edge speed_kph as-is (car)
    multiplier_features: dict = field(default_factory=dict)
    additive_route_features: dict = field(default_factory=dict)
    additive_endpoint_features: dict = field(default_factory=dict)
    min_speed_kph: float = 1.0
    max_speed_kph: float = 120.0
    min_trip_distance: float = 1_000.0
    max_trip_distance: float = 50_000.0
    max_snap_distance: float = 300.0  # legs with an endpoint farther from the network are dropped
    derived_profiles: list[DerivedFromCalibrated] = field(default_factory=list)


# Predicates for walk_prm edge exclusions.
def _is_steps(edges: pd.DataFrame) -> pd.Series:
    """OSM `highway=steps` edges."""
    def _first_if_list(v):
        if isinstance(v, list):
            return v[0] if v else None
        return v
    return edges['highway'].map(_first_if_list) == 'steps'


def _too_steep(edges: pd.DataFrame) -> pd.Series:
    """Edges with |slope| > 20%."""
    dz = edges['elevation_delta'].astype(float)
    length = edges['length'].astype(float).clip(lower=0.1)
    return (dz.abs() / length) > 0.20


_WALK_DERIVATIONS: list[DerivedFromCalibrated] = [
    DerivedFromCalibrated(
        profile='walk_prm',
        target_speed_kph=3.5,
        min_speed_kph=0.5,
        max_speed_kph=5.0,
        coef_multipliers={
            'elevation_gain':       3.0,
            'elevation_loss':       3.0,
        },
        edge_exclusions=[_is_steps, _too_steep],
    ),
]


_EBIKE_DERIVATIONS: list[DerivedFromCalibrated] = [
    DerivedFromCalibrated(
        profile='ebike25',
        target_speed_kph=22.0,
        min_speed_kph=3.0,
        max_speed_kph=50.0,
        coef_multipliers={
            'elevation_gain':    0.33,  # motor assist eases climbs
            'elevation_loss':    0.50,  # downhill saves less time (still pedalling)
            'is_4way':           1.30,  # faster baseline → bigger relative stop cost
            'is_traffic_signal': 1.30,
        },
        not_slower_than=('rbike',),
    ),
    DerivedFromCalibrated(
        profile='ebike45',
        target_speed_kph=40.0,
        min_speed_kph=5.0,
        max_speed_kph=50.0,
        coef_multipliers={
            'elevation_gain':    0.33,
            'elevation_loss':    0.00,  # at 40 km/h, downhill barely helps
            'is_4way':           1.80,
            'is_traffic_signal': 1.80,
        },
        not_slower_than=('rbike', 'ebike25'),
    ),
]


# Per-profile configurations.
_PROFILE_CONFIG: list[ProfileCalibration] = [
    ProfileCalibration(
        profile='rwalk',
        mode='walk',
        constant=None,
        baseline_speed_kph=5.0,             # flat-ground pedestrian average
        multiplier_features={
            # 'is_traffic_signal': 0.0,     # Near-zero coefficient
            'density_r500_norm': -0.1,
        },
        additive_route_features={
            'elevation_gain':    2.5,
            'elevation_loss':    1.0,
            'is_4way':           15.0,
            # 'is_traffic_signal': 0.0,
        },
        additive_endpoint_features={
            'snap_dist':          0.0,
        },
        min_trip_distance=250.0,            
        max_trip_distance=2_500.0,       
        derived_profiles=_WALK_DERIVATIONS,  # walk_prm from rwalk fit
    ),
    ProfileCalibration(
        profile='rbike',
        mode='bike',
        constant=None,
        baseline_speed_kph=17.5,
        multiplier_features={
            # 'slope_uphill':         0.0,
            # 'slope_downhill':       0.0,
            # 'bike_infra_score_avg_r250': 0.0, # Has effect, but can be confounded by other things (road type). Coefficient quite large, probably non-linear. Need to re-think bike infrastructure quantification.
            # 'speed_limit_avg_r250': 0.0, # Some interesting interaction with gain/loss, runs away with it (no convergence).
            'density_r500_norm': -0.1,
        },
        additive_route_features={
            # 'is_traffic_signal': 3.0, # Coefficient negative, but also few features per route; hard to fit
            'is_4way':           5.0,
            'elevation_gain':    2.8,
            'elevation_loss':    0.0,
        },
        additive_endpoint_features={
            'snap_dist':          0.0,
        },
        min_trip_distance=500.0,
        max_trip_distance=20_000.0,
        derived_profiles=_EBIKE_DERIVATIONS,  # ebike25 / ebike45 from rbike fit
    ),
    ProfileCalibration(
        profile=['car_night', 'car_base', 'car_peak'],
        mode='car',
        constant=None,
        baseline_speed_kph=None,
        multiplier_features={
            'density_r500_norm':  0.2,
            'vc_beta_2.0':        0.2,
        },
        additive_route_features={
            'is_4way':            10.0,
            'is_traffic_signal':  10.0,
        },
        additive_endpoint_features={
            'snap_dist':          0.0,
        },
        min_trip_distance=500.0,
        max_trip_distance=100_000.0,
    ),
]


# Car on the swissTLMRegio network (`scenario.car_network_source == 'tlmregio'`;
# simplified-network test). The 2 × 2 classes as in LUMOS: the baseline is the
# class speed prior from preparation (motorway by legal limit, otherwise 50 km/h
# urban / 80 km/h rural, ramps 60 km/h); the fit scales motorway, urban and urban
# motorway against rural roads, plus congestion. No density term (overlaps the
# urban class) and no signals (swissTLMRegio has none); instead, urban 4-way
# junctions get an extra delay on top of `is_4way` (most signals are in towns).
# Endpoints snap up to the scenario's 2 km, so access to the main roads is part of
# `snap_dist`.
_TLMREGIO_CAR_CONFIG = replace(
    next(c for c in _PROFILE_CONFIG if c.mode == 'car'),
    multiplier_features={
        'is_motorway':        0.0,
        'is_urban':           0.2,
        'is_motorway_urban':  0.0,
        'vc_beta_2.0':        0.2,
    },
    additive_route_features={
        'is_4way':            10.0,
        'is_4way_urban':      5.0,
    },
    max_snap_distance=2_000.0,
)


def _profile_config(mode: str, scenario) -> ProfileCalibration:
    """Calibration config of `mode` for `scenario`; the swissTLMRegio car network has its own."""
    if mode == 'car' and scenario.car_network_source == 'tlmregio':
        return _TLMREGIO_CAR_CONFIG
    return next(c for c in _PROFILE_CONFIG if c.mode == mode)


# One variant per mode (walk / bike / car). Each runs the corresponding
# entry in `_PROFILE_CONFIG` — car's 3 hour-bucket profiles fit together
# under `--variant car` (they share the graph load; splitting into
# separate variants would fragment the shared coefs / properties files).
variants = Variants([('mode', str)])
for _cfg in _PROFILE_CONFIG:
    variants.add(name=_cfg.mode, mode=_cfg.mode)


_VALID_TRIP_COL = 'is_valid_domestic_trip'
_SPEED_ENVELOPE_COL = 'is_within_speed_envelope'
_DETOUR_ENVELOPE_COL = 'is_within_detour_envelope'
_LAND_BASED_COL = 'is_land_based'
_ELEVATION_BAND_COL = 'is_within_elevation_band'
# Survey `mode_simplified` labels fitted per graph mode (bike: regular bicycles only).
_SURVEY_MODES: dict[str, set[str]] = {'walk': {'walk'}, 'bike': {'rbike'}, 'car': {'car'}}


def _apply_survey_gates(legs: pd.DataFrame) -> pd.DataFrame:
    """Trip-level quality gates common to MTMC + MOBIS calibration
    training. Applied at load time because 04 reads both sources
    straight from the preparation surveys directory (not through
    `survey/02d_prepare_survey_legs.py`'s pre-filter), so every flag is
    enforced here, symmetrically. Gates:

      - is_valid_domestic_trip == 1 (dist/time OK, in Switzerland,
        MOBIS `implausible` False)
      - is_within_speed_envelope == 1 (per-mode speed envelope)
      - is_within_detour_envelope == 1 (dist_measured/dist_line ratio
        within per-mode cap — catches errand-stop legs)
      - is_land_based == 1 (drops plane / boat / aerialway)
      - is_within_elevation_band == 1 (DEM-valid endpoints + not
        alpine; composite gate from `standardize.attach_within_
        elevation_band_flag` — thresholds live there as single
        source of truth, shared with 07a + 02d)

    Mode filtering is applied by each caller (they know which mode
    set they're training on). Person-level diary integrity is NOT
    enforced here — it's a person-stats concern, not a training
    concern (see `standardize.person_has_any_bad_trip` for the
    downstream helper).

    Flags checked defensively — a missing column is treated as
    all-pass so this survives future schema changes.
    """
    if _VALID_TRIP_COL in legs.columns:
        legs = legs[legs[_VALID_TRIP_COL] == 1]
    if _SPEED_ENVELOPE_COL in legs.columns:
        legs = legs[legs[_SPEED_ENVELOPE_COL] == 1]
    if _DETOUR_ENVELOPE_COL in legs.columns:
        legs = legs[legs[_DETOUR_ENVELOPE_COL] == 1]
    if _LAND_BASED_COL in legs.columns:
        legs = legs[legs[_LAND_BASED_COL].fillna(1) == 1]
    if _ELEVATION_BAND_COL in legs.columns:
        legs = legs[legs[_ELEVATION_BAND_COL] == 1]
    return legs


def _load_mtmc_legs(context, mode: str) -> pd.DataFrame:
    """MZMV 2015 + 2021 legs, mode-filtered. Self-reported durations.
    MZMV uses `rbike` for regular bike in `mode_simplified`."""
    surveys_ctx = context.source(
        'preparation/switzerland/surveys', storage=Storage.PRIVATE)
    dfs = []
    for subpath in ('mzmv_2015/legs.csv', 'mzmv_2021/legs.csv'):
        path = surveys_ctx.path_for(Storage.PRIVATE, subpath)
        dfs.append(pd.read_csv(path, low_memory=False))
    legs = pd.concat(dfs, ignore_index=True, sort=False)
    legs = _apply_survey_gates(legs)
    legs = legs[legs['mode_simplified'].isin(_SURVEY_MODES[mode])]
    return legs


def _load_mobis_legs(context, mode: str, cohort: str) -> pd.DataFrame:
    """MOBIS legs from the specified `cohort` folder (`mobis_precovid`
    or `mobis_covid`), mode-filtered. GPS-tracked durations.

    Bike training is confirmed-mechanical only (`mode_simplified ==
    'rbike'`) — symmetric with MTMC. MOBIS introduced the Mode::Bicycle
    vs Mode::Ebicycle split only in 2020-07:
      - `mobis_precovid` (< 2020): NO post-split Mode::Bicycle rows;
        all Mode::Bicycle labelled `'anybike'` (mixed unknown sub-type),
        so the bike training set is effectively EMPTY for this cohort.
        That's intentional; use MTMC for bike here.
      - `mobis_covid` (2020-wk13 to 2022-03): has confirmed
        Mode::Bicycle → `'rbike'` rows. Note COVID-era traffic is
        anomalous — use with care for baseline calibration."""
    surveys_ctx = context.source(
        'preparation/switzerland/surveys', storage=Storage.PRIVATE)
    path = surveys_ctx.path_for(Storage.PRIVATE, f'{cohort}/legs.csv')
    legs = pd.read_csv(path, low_memory=False)
    legs = _apply_survey_gates(legs)
    legs = legs[legs['mode_simplified'].isin(_SURVEY_MODES[mode])]
    if mode == 'bike' and len(legs) == 0:
        logging.warning(
            f"  ⚠ MOBIS bike training set is EMPTY for cohort "
            f"{cohort!r} (no Mode::Bicycle → 'rbike' rows). Consider "
            f"using MTMC for bike (edge_weights_sources['bike']='mtmc') "
            f"or switching cohort.")
    return legs


def _load_survey_legs(
    context, mode: str, profile: str | None, source: str,
) -> pd.DataFrame:
    """Load ground-truth legs from the requested `source`, mode-filtered
    (and, for car, time-of-day-bucketed by `profile` via `peak_str`).
    Returns the schema `aperta.calibration.calibrate_edge_weights`
    expects: `orig_x, orig_y, dest_x, dest_y, time_measured,
    dist_measured, dist_line` (LV95 coords; times in s; distances in m).
    """
    if source == 'mtmc':
        legs = _load_mtmc_legs(context, mode)
    elif source in ('mobis_precovid', 'mobis_covid'):
        legs = _load_mobis_legs(context, mode, cohort=source)
    else:
        raise ValueError(
            f"Unknown edge_weights source {source!r}; expected one of "
            f"'mtmc' | 'mobis_precovid' | 'mobis_covid'.")

    # Car time-of-day bucket: the survey prep's `peak_str` (weekday-aware), the same label 05,
    # 07a, 08a and the validation scripts use to pick a car profile. Unknown-hour legs
    # (`hour_of_day == -1`, labelled 'base' by `peak_str`) are left out of training.
    if mode == 'car' and profile is not None:
        bucket = profile.removeprefix('car_')
        if bucket not in ('peak', 'base', 'night'):
            raise ValueError(
                f"Unknown car profile {profile!r}; expected 'car_night', "
                f"'car_base', or 'car_peak'.")
        legs = legs[(legs['peak_str'] == bucket) & (legs['hour_of_day'] >= 0)]

    legs = legs[(legs['time_measured'] > 0) & (legs['dist_measured'] > 0)]

    if 'dist_line' not in legs.columns:
        dx = legs['dest_x'] - legs['orig_x']
        dy = legs['dest_y'] - legs['orig_y']
        legs = legs.assign(dist_line=np.sqrt(dx * dx + dy * dy))

    logging.info(
        f"  → loaded {len(legs):,} legs (mode={mode!r}, profile={profile!r}, "
        f"source={source!r})")
    # Return the FULL frame — `calibrate_edge_weights` ignores extra
    # columns, and `_fit_coefs` needs `weight_person` for WLS.
    return legs.reset_index(drop=True)


def _prepare_edge_features(graph, baseline_speed_kph: float | None) -> None:
    """Write a uniform `speed_kph` onto every edge of `graph`, in place.
    No-op when `baseline_speed_kph` is None (car keeps the OSM speeds).
    """
    if baseline_speed_kph is not None:
        nx.set_edge_attributes(graph, baseline_speed_kph, 'speed_kph')


_KMH_TO_MS = 1.0 / 3.6


def _stash_osm_speed_kph(graph) -> None:
    """Copy each edge's current `speed_kph` to `osm_speed_kph` before
    `_prepare_edge_features` overwrites it with a uniform baseline.
    `_set_per_edge_derived_baseline` uses it to cap derived profiles'
    per-edge baselines at the original speed limit."""
    osm = nx.get_edge_attributes(graph, 'speed_kph')
    nx.set_edge_attributes(graph, osm, 'osm_speed_kph')


def _set_per_edge_derived_baseline(
    graph, derivation: DerivedFromCalibrated,
) -> None:
    """For each edge: `eff_speed = clip(min(target, osm_speed_kph),
    [min_s, max_s])` and `__derived_baseline_duration = length /
    (eff_speed / 3.6)`. Writes `__derived_effective_speed_kph` and
    `__derived_baseline_duration` per edge. Edges with missing /
    non-positive `osm_speed_kph` fall back to `target_speed_kph`."""
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = float(data['length'])
        osm = data.get('osm_speed_kph')
        target = derivation.target_speed_kph
        if osm is None or not pd.notna(osm) or float(osm) <= 0:
            eff = target
        else:
            eff = min(target, float(osm))
        eff = max(derivation.min_speed_kph,
                  min(derivation.max_speed_kph, eff))
        data['__derived_effective_speed_kph'] = eff
        data['__derived_baseline_duration'] = length / (eff * _KMH_TO_MS)


def _derive_ebike_profile(
    graph,
    rbike_cfg: ProfileCalibration,
    rbike_coefs: pd.DataFrame,
    derivation: DerivedFromCalibrated,
) -> tuple[dict, dict, pd.DataFrame]:
    """Apply derivation rules to the parent profile's fitted coefficients
    (rbike → ebike25 / ebike45, rwalk → walk_prm) and write
    `duration_calibrated_<derivation.profile>` on the graph. Returns
    `(durations_dict, effective_speeds_dict, derived_coefs_df)` suitable
    for stacking into 04's output frames.

    `rbike_coefs` is the parent's `CalibrationResult.coefficients`
    DataFrame: indexed by feature name (`baseline_time`, `is_4way`,
    `elevation_gain`, …) with columns `coef`, `p`.
    """
    alpha = float(rbike_coefs.loc['baseline_time', 'coef'])
    mult = {
        name: float(rbike_coefs.loc[name, 'coef'])
              * derivation.coef_multipliers.get(name, 1.0)
        for name in rbike_cfg.multiplier_features
        if name in rbike_coefs.index
    }
    add_route = {
        name: float(rbike_coefs.loc[name, 'coef'])
              * derivation.coef_multipliers.get(name, 1.0)
        for name in rbike_cfg.additive_route_features
        if name in rbike_coefs.index
    }
    missing = ((set(rbike_cfg.multiplier_features)
                | set(rbike_cfg.additive_route_features))
               - set(rbike_coefs.index))
    if missing:
        logging.warning(
            f"  ⚠ derived profile={derivation.profile}: features in "
            f"rbike cfg but not in rbike coefs (dropped): {sorted(missing)}")

    with step(
        f'derive {derivation.profile}: '
        f'target={derivation.target_speed_kph:.0f} km/h (capped by OSM), '
        f'multipliers={derivation.coef_multipliers}'
    ):
        _set_per_edge_derived_baseline(graph, derivation)
        edge_duration_attr = f'duration_calibrated_{derivation.profile}'
        calibration.apply_edge_durations(
            graph,
            multiplier_features=mult,
            additive_route_features=add_route,
            alpha=alpha,
            out_attr=edge_duration_attr,
            baseline_duration_attr='__derived_baseline_duration',
            min_speed_kph=derivation.min_speed_kph,
            max_speed_kph=derivation.max_speed_kph,
        )
        for other in derivation.not_slower_than:
            other_attr = f'duration_calibrated_{other}'
            n_capped = 0
            for _u, _v, _k, data in graph.edges(keys=True, data=True):
                cap = data.get(other_attr)
                if cap is not None and cap < data[edge_duration_attr]:
                    data[edge_duration_attr] = cap
                    n_capped += 1
            logging.info(f"  → {derivation.profile}: {n_capped:,} edges capped at "
                         f"{other}'s duration")
        # Apply hard exclusions: any predicate matching → duration = inf.
        if derivation.edge_exclusions:
            edges_df = ox.graph_to_gdfs(graph, nodes=False)
            excluded_mask = pd.Series(False, index=edges_df.index)
            for predicate in derivation.edge_exclusions:
                excluded_mask |= predicate(edges_df).fillna(False).astype(bool)
            n_excluded = int(excluded_mask.sum())
            if n_excluded:
                inf = float('inf')
                for (u, v, k) in edges_df.index[excluded_mask]:
                    graph[u][v][k][edge_duration_attr] = inf
            logging.info(
                f"  → {derivation.profile}: {n_excluded:,} of "
                f"{len(edges_df):,} edges hard-excluded (duration=inf)")

        durations = nx.get_edge_attributes(graph, edge_duration_attr)
        lengths = nx.get_edge_attributes(graph, 'length')
        eff_speed_kph = {k: lengths[k] / v * 3.6
                         for k, v in durations.items() if v != float('inf')}

    # Derived coefs: scale coef (and mean_effect, if present) by the same
    # factor; NaN out p-values since derived profiles aren't fitted.
    derived_coefs = rbike_coefs.copy()
    for name in (*mult.keys(), *add_route.keys()):
        factor = derivation.coef_multipliers.get(name, 1.0)
        derived_coefs.loc[name, 'coef'] = (rbike_coefs.loc[name, 'coef'] * factor)
        if 'mean_effect' in derived_coefs.columns:
            derived_coefs.loc[name, 'mean_effect'] = (rbike_coefs.loc[name, 'mean_effect'] * factor)
        if 'p' in derived_coefs.columns:
            derived_coefs.loc[name, 'p'] = float('nan')

    med_dur = float(np.median(list(durations.values())))
    med_speed = float(np.median(list(eff_speed_kph.values())))
    logging.info(f"  → {derivation.profile}: median duration {med_dur:.2f} s, "
                 f"median effective speed {med_speed:.1f} km/h")
    return durations, eff_speed_kph, derived_coefs


def _fit_coefs(context, cfg: ProfileCalibration) -> pd.DataFrame:
    """Fit OLS per profile in `cfg`; return the coef DataFrame with MultiIndex
    columns `(profile, coef|p)`. Graph reloaded fresh per profile — calibrate
    mutates per-edge attrs. `cfg.derived_profiles` (ebike25 / ebike45,
    walk_prm) are rule-derived from the fit. Does NOT write to disk;
    `coefs.resolve` handles that.
    """
    profiles = cfg.profile if isinstance(cfg.profile, list) else [cfg.profile]
    out_coef_dfs: dict[str, pd.DataFrame] = {}
    # Scope training legs to the scenario's AOI (both endpoints inside). For
    # switzerland-h10 this drops legs leaving Switzerland; for the CV scenarios
    # and other sub-country AOIs it also restricts the region.
    scenario = get_scenario(context.scenario)
    with step('load AOI polygon (area outline, as in 01)'):
        aoi_polygon = load_aoi_polygon(context)

    for profile in profiles:
        with step(f'load {cfg.mode} graph fresh for profile={profile}'):
            graph = _load_mode_graph(context, cfg, allow_cache=False)
            logging.info(f"  → {graph.number_of_nodes():,} nodes, "
                         f"{graph.number_of_edges():,} edges")
        if cfg.derived_profiles:
            _stash_osm_speed_kph(graph)
        _prepare_edge_features(graph, cfg.baseline_speed_kph)

        with step(f'profile={profile}: derive snap-eligible node set'):
            snap_eligible = routing_prep.compute_snap_eligible_nodes(
                graph,
                directedness='directed_scc',
                cost_excluded_flag=f'cost_excluded_{cfg.mode}',
            )
            logging.info(f"  → {len(snap_eligible):,} of "
                         f"{graph.number_of_nodes():,} nodes eligible")

        source = scenario.edge_weights_sources[cfg.mode]
        with step(f'profile={profile}: OLS fit (source={source})'):
            legs = _load_survey_legs(context, cfg.mode, profile, source)
            legs = filter_legs_by_aoi(
                legs, aoi_polygon, crs=scenario.crs_main,
                label=f'{source} {cfg.mode}')
            leg_filter = ((legs.dist_line > cfg.min_trip_distance) &
                          (legs.dist_line <= cfg.max_trip_distance))
            # WLS weight = `weight_person`: BFS's stratification correction
            # on MTMC (0.3-3.0 range), uniformly 1 on MOBIS (WLS = OLS).
            # Without the column, `weight_col=None` skips weighting.
            if 'weight_person' in legs.columns:
                weight_col = 'weight_person'
            else:
                weight_col = None
                weight_like_cols = sorted(
                    c for c in legs.columns if 'weight' in c.lower())
                logging.warning(
                    f"  ⚠ 'weight_person' NOT found on legs — WLS disabled. "
                    f"weight-like columns present: {weight_like_cols!r}")
            result = calibration.calibrate_edge_weights(
                graph, legs[leg_filter],
                baseline_speed_attr='speed_kph',
                multiplier_features=cfg.multiplier_features,
                additive_route_features=cfg.additive_route_features,
                additive_endpoint_features=cfg.additive_endpoint_features,
                min_speed_kph=cfg.min_speed_kph,
                max_speed_kph=cfg.max_speed_kph,
                constant=cfg.constant,
                n_iterations=_N_ITERATIONS,
                r2_tolerance=_R2_TOLERANCE,
                max_distance=cfg.max_snap_distance,
                edge_duration_attr=f"duration_calibrated_{profile}",
                eligible_node_ids=snap_eligible,
                weight_col=weight_col,
            )
            out_coef_dfs[profile] = result.coefficients

        # Only coefs are kept here; per-edge durations are recomputed on the
        # LOCAL graph in `_apply_coefs_and_save`.
        for derivation in cfg.derived_profiles:
            _, _, derived_coefs = _derive_ebike_profile(
                graph, cfg, result.coefficients, derivation)
            out_coef_dfs[derivation.profile] = derived_coefs

    return pd.concat(out_coef_dfs.values(), axis=1, keys=out_coef_dfs.keys())


def _apply_coefs_and_save(
    context, cfg: ProfileCalibration, coefs_df: pd.DataFrame,
) -> None:
    """Apply coefs to this scenario's local network, write
    `properties/edges_<mode>_calibrated.csv`. `coefs_df` is the
    `(profile, coef|p)` MultiIndex frame from `coefs.resolve`, uniform
    regardless of Calibrate / ImportFrom / HandWritten provenance.
    """
    profiles = cfg.profile if isinstance(cfg.profile, list) else [cfg.profile]
    out_edge_dfs: list[pd.DataFrame] = []

    for profile in profiles:
        with step(f'apply {cfg.mode}/{profile} coefs to {context.scenario}'):
            graph = _load_mode_graph(context, cfg, allow_cache=False)
            if cfg.derived_profiles:
                _stash_osm_speed_kph(graph)
            _prepare_edge_features(graph, cfg.baseline_speed_kph)

            # apply_edge_durations expects a duration attr, not speed.
            for *_, d in graph.edges(keys=True, data=True):
                d['__baseline_duration'] = (
                    float(d['length']) / (float(d['speed_kph']) * _KMH_TO_MS)
                )

            col = (profile, 'coef')
            if col not in coefs_df.columns:
                raise KeyError(
                    f"profile {profile!r} missing from coefs for mode "
                    f"{cfg.mode!r}. Available: "
                    f"{sorted(set(c[0] for c in coefs_df.columns))}")
            alpha = float(coefs_df.loc['baseline_time', col])
            mult: dict[str, float] = {
                feat: float(coefs_df.loc[feat, col])
                for feat in cfg.multiplier_features
                if feat in coefs_df.index
            }
            add_route: dict[str, float] = {
                feat: float(coefs_df.loc[feat, col])
                for feat in cfg.additive_route_features
                if feat in coefs_df.index
            }
            missing = (set(cfg.multiplier_features) | set(cfg.additive_route_features)) - set(coefs_df.index)
            if missing:
                logging.warning(f"  ⚠ profile={profile}: features in cfg but not in "
                                f"coefs (dropped): {sorted(missing)}")

            edge_duration_attr = f'duration_calibrated_{profile}'
            calibration.apply_edge_durations(
                graph,
                multiplier_features=mult,
                additive_route_features=add_route,
                alpha=alpha,
                out_attr=edge_duration_attr,
                baseline_duration_attr='__baseline_duration',
                min_speed_kph=cfg.min_speed_kph,
                max_speed_kph=cfg.max_speed_kph,
            )
            durations = nx.get_edge_attributes(graph, edge_duration_attr)
            baseline_durations = nx.get_edge_attributes(graph, '__baseline_duration')
            lengths = nx.get_edge_attributes(graph, 'length')
            eff_speed = {
                k: (lengths[k] / v * 3.6) if v > 0 else float('nan')
                for k, v in durations.items()
            }
            out_edge_dfs.append(pd.DataFrame.from_dict({
                edge_duration_attr: durations,
                f'effective_speed_kph_{profile}': eff_speed,
                # Per-edge naive baseline duration (length / baseline_speed).
                # For car: baseline = OSM speed_kph. For walk/bike: uniform
                # `cfg.baseline_speed_kph`. Used by 05 for the naive-routing
                # comparison in technical_validation/times_vs_* scripts.
                f'duration_baseline_{profile}': baseline_durations,
            }))

        for derivation in cfg.derived_profiles:
            rbike_coefs = coefs_df.xs(profile, axis=1, level=0).copy()
            derived_durations, derived_speeds, _ = _derive_ebike_profile(
                graph, cfg, rbike_coefs, derivation)
            # `_derive_ebike_profile` sets `__derived_baseline_duration`
            # via `_set_per_edge_derived_baseline` before writing the
            # calibrated attribute; capture it now (next derivation would
            # overwrite it).
            derived_baseline = nx.get_edge_attributes(
                graph, '__derived_baseline_duration')
            out_edge_dfs.append(pd.DataFrame.from_dict({
                f'duration_calibrated_{derivation.profile}': derived_durations,
                f'effective_speed_kph_{derivation.profile}': derived_speeds,
                f'duration_baseline_{derivation.profile}': derived_baseline,
            }))

    with step(f'save edges_{cfg.mode}_calibrated.csv'):
        out_edge_df = pd.concat(out_edge_dfs, axis=1)
        edge_id_index = pd.Index(
            [_edge_id(u, v, k) for u, v, k in out_edge_df.index],
            name='edge_id')
        out_edge_df.index = edge_id_index
        context.create_properties(
            out_edge_df, data_name=f'{cfg.mode}_calibrated',
            float_format='%.2f')


def _load_mode_graph(context, cfg: ProfileCalibration, allow_cache: bool):
    """Load `cfg.mode`'s graph. Car adds `flows` on top of the standard
    core/from_nodes/extended overlays."""
    edge_cols = ['core', 'from_nodes', 'extended']
    if cfg.mode == 'car':
        edge_cols.append('flows')
    return context.get_nw(
        data_name=cfg.mode,
        add_node_properties='core',
        add_edge_properties=edge_cols,
        allow_cache=allow_cache,
    )


def main(variant) -> None:
    """Resolve coefs for one mode (walk / bike / car) per the scenario's
    declaration in scenarios.py, then apply to the local graph."""
    context = init_context(variant)
    scenario = get_scenario(context.scenario)
    mode = variant.mode
    if mode not in scenario.mode_configs:
        logging.info(f"mode={mode!r} not in scenario mode_configs "
                     f"({list(scenario.mode_configs)}); skipping")
        context.close()
        return
    cfg = _profile_config(mode, scenario)
    coefs_df = coefs.resolve(
        context,
        name=f'edge_weights_{cfg.mode}',
        calibrate_fn=(lambda: _fit_coefs(context, cfg)),
    )
    _apply_coefs_and_save(context, cfg, coefs_df)
    context.close()


if __name__ == '__main__':
    variants.run(main)
