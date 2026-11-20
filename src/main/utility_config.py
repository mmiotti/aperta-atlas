"""
Single source of truth for the utility spec's feature-column mapping
+ pre-fit numeric rescaling.

Both 09a (fit-time producer) and 09b / 10 (runtime consumers) must
agree on which node file each feature is joined from
(`POINT_FEATURE_SOURCES`) and what divisor 09a applied before fitting β
(`FEATURE_SCALE`). Consumers apply the identical rescale.
"""


# Per-node point features used at origin + destination anchors, keyed
# by the un-oriented source column and mapped to
# (source_mode, node-props data_name). Each feature lives on its
# NATURAL graph's node file; the endpoint join uses
# `node_id_<source_mode>` (populated on cells / zones by the 'snap'
# properties load) as the lookup key.
POINT_FEATURE_SOURCES: dict[str, tuple[str, str]] = {
    'elevation':                  ('walk', 'walk_extended'),
    'density_r250_norm':          ('walk', 'walk_extended'),
    'density_r500_norm':          ('walk', 'walk_extended'),
    'mean_abs_slope_r100':        ('walk', 'walk_extended'),
    'mean_abs_slope_r250':        ('walk', 'walk_extended'),
    'bike_stress_share_01_r100':  ('bike', 'bike_extended'),
    'bike_stress_share_01_r250':  ('bike', 'bike_extended'),
    'bike_stress_share_45_r100':  ('bike', 'bike_extended'),
    'bike_stress_share_45_r250':  ('bike', 'bike_extended'),
    'speed_limit_avg_r100':       ('car',  'car_extended'),
    'speed_limit_avg_r250':       ('car',  'car_extended'),
    'traffic_flow_avg_r250':      ('car',  'car_flows_avg'),
    'traffic_flow_avg_r500':      ('car',  'car_flows_avg'),
    'vc_beta_2.0_avg_r250':       ('car',  'car_flows_avg'),
    'vc_beta_2.0_avg_r500':       ('car',  'car_flows_avg'),
}

# 09a divides raw feature values by these constants before fitting so
# β magnitudes land in a sane numeric range. Consumers MUST apply the
# same divisor when looking up feature values — otherwise `β × raw` is
# off by `scale`× and utilities blow up (e.g. car `traffic_flow_avg_r250`
# with raw ~10k veh/day + scale 10k = a 10000× error).
#
# The keys mix un-oriented source columns (looked up in
# `common.join_util_node_features` and 09a's endpoint join) and the route
# features' biogeme names (`*_route`, 09a's route join). A missing entry
# means `scale = 1.0` (no rescale).
FEATURE_SCALE: dict[str, float] = {
    'traffic_flow_avg_r250':        10_000.0,
    'traffic_flow_avg_r250_route':  10_000.0,
    'traffic_flow_avg_r500':        10_000.0,
    'speed_limit_avg_r250':         10.0,       # per 10 km/h
    'speed_limit_avg_r250_route':   10.0,
    'mean_abs_slope_r250':          0.01,       # m/m → percent
    'mean_abs_slope_r250_route':    0.01,
}
