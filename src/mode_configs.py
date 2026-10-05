"""Per-mode pipeline configuration for the Accessibility Atlas.

A `ModeConfig` bundles one mode's network, per-profile cost variants,
OD-tier radii, cost cutoff, and min-route floors. `MODE_CONFIGS` provides
defaults; per-scenario overrides happen via `Scenario.mode_configs`.
Routing/accessibility scripts iterate `scenario.mode_configs.values()`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OdmRadii:
    """Per-mode tier radii (CRS units, meters). Must satisfy
    `r_cells ≤ r_medium ≤ r_zones`. See `aperta.od_pairs.get_pairs`."""
    r_cells: float
    r_medium: float
    r_zones: float


@dataclass(frozen=True)
class CalibratedSource:
    """Where this mode+profile's calibrated edges live, plus the label
    used for output filenames and overhead-coefs lookup.

    `name`: profile label — suffix in `cells_access_*_<name>.csv` AND
        column key in `coefs/overheads_road.csv` (must match 07's
        `_ROAD_CONFIGS` entry).
    `edges_data_name`: `data_name` for `context.get_properties('edges', …)`.
    `edge_column`: column inside that file (typically `duration_calibrated_<name>`).
    `cost_data_name`: `data_name` for the 05 cost ODM.
    """
    name: str
    edges_data_name: str
    edge_column: str
    cost_data_name: str


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for one mode's accessibility pipeline. One `ModeConfig`
    bundles all profiles that share the mode's network (e.g. all three
    car time-of-day profiles route on `car.graphml`).

    `mode`: graph + snap-column root. `profiles`: profile labels (`[None]`
    for single-profile modes). `radii`: OD-tier radii. `time_cutoff_s`:
    max routing horizon. `calibrated_sources`: per-profile source overrides
    (default = one file per mode, profile-suffixed column). `min_route_time_s`
    / `min_route_disutility`: physical floors on trip cost, applied via
    `routing.floor_intrazonal_costs` — prevents degenerate `exp(-β·0)`
    and log-blowups on self/very-close pairs.
    """
    mode: str
    profiles: list[str | None]
    radii: OdmRadii
    time_cutoff_s: float
    calibrated_sources: dict[str | None, CalibratedSource] | None = None
    min_route_time_s: float = 60.0
    min_route_disutility: float | None = None

    def source_for(self, profile: str | None) -> CalibratedSource:
        """Resolve the `CalibratedSource` for `profile`. Uses
        `calibrated_sources[profile]` if set, else 05's default convention
        (one file per mode, profile-suffixed column)."""
        if (self.calibrated_sources is not None
                and profile in self.calibrated_sources):
            return self.calibrated_sources[profile]
        if profile is None:
            return CalibratedSource(
                name=self.mode,
                edges_data_name=f'{self.mode}_calibrated',
                edge_column=f'duration_calibrated_{self.mode}',
                cost_data_name='time_net',
            )
        return CalibratedSource(
            name=f'{self.mode}_{profile}',
            edges_data_name=f'{self.mode}_calibrated',
            edge_column=f'duration_calibrated_{self.mode}_{profile}',
            cost_data_name=f'time_net_{profile}',
        )


# Bike: rbike fit against the scenario's bike ground truth (`edge_weights_sources['bike']`);
# ebike25 / ebike45 derived (05's
# `_EBIKE_DERIVATIONS`). All three columns share `edges_bike_calibrated.csv`;
# each profile gets its own cost ODM.
_BIKE_SOURCES: dict[str | None, CalibratedSource] = {
    'rbike': CalibratedSource(
        name='rbike',
        edges_data_name='bike_calibrated',
        edge_column='duration_calibrated_rbike',
        cost_data_name='time_net_rbike',
    ),
    'ebike25': CalibratedSource(
        name='ebike25',
        edges_data_name='bike_calibrated',
        edge_column='duration_calibrated_ebike25',
        cost_data_name='time_net_ebike25',
    ),
    'ebike45': CalibratedSource(
        name='ebike45',
        edges_data_name='bike_calibrated',
        edge_column='duration_calibrated_ebike45',
        cost_data_name='time_net_ebike45',
    ),
}


# Walk: rwalk (calibrated) + walk_prm (persons-with-reduced-mobility,
# derived from rwalk in 05). Both share `edges_walk_calibrated.csv`.
_WALK_SOURCES: dict[str | None, CalibratedSource] = {
    'rwalk': CalibratedSource(
        name='rwalk',
        edges_data_name='walk_calibrated',
        edge_column='duration_calibrated_rwalk',
        cost_data_name='time_net_rwalk',
    ),
    'walk_prm': CalibratedSource(
        name='walk_prm',
        edges_data_name='walk_calibrated',
        edge_column='duration_calibrated_walk_prm',
        cost_data_name='time_net_walk_prm',
    ),
}


# Unified radii → downstream cell-baked ODMs share a common structure so
# cross-modal aggregation (fastest-mode / min-cost) works trivially at
# the cell level. Sized for car; per-mode `time_cutoff_s` prunes further.
_UNIFIED_RADII = OdmRadii(r_cells=1_500.0, r_medium=7_500.0, r_zones=100_000.0)

MODE_CONFIGS: dict[str, ModeConfig] = {
    'walk': ModeConfig(
        mode='walk',
        profiles=['rwalk', 'walk_prm'],
        radii=_UNIFIED_RADII,
        time_cutoff_s=3_600.0,
        calibrated_sources=_WALK_SOURCES,
        min_route_disutility=3.0,
    ),
    'bike': ModeConfig(
        mode='bike',
        profiles=['rbike', 'ebike25', 'ebike45'],
        radii=_UNIFIED_RADII,
        time_cutoff_s=3_600.0,
        calibrated_sources=_BIKE_SOURCES,
        # 2 min floor: unlock + mount + park.
        min_route_time_s=120.0,
        min_route_disutility=3.0,
    ),
    'car': ModeConfig(
        mode='car',
        profiles=['night', 'base', 'peak'],
        radii=_UNIFIED_RADII,
        time_cutoff_s=3_600.0,
        # 3 min floor: walk to car + park at dest.
        min_route_time_s=180.0,
        min_route_disutility=3.0,
    ),
}


# Transit is kept out of `MODE_CONFIGS` — no local graph (NPVM provides
# z2z directly). Transit-aware scripts import `TRANSIT_MODE_CONFIG`.
TRANSIT_MODE_CONFIG: ModeConfig = ModeConfig(
    mode='transit',
    profiles=[None],
    radii=_UNIFIED_RADII,
    time_cutoff_s=5_400.0,
    # 5 min floor: walk to stop + wait + board + ride + walk. Also
    # protects against NPVM same-zone = 0 s + negative overhead.
    min_route_time_s=300.0,
    min_route_disutility=3.0,
)
