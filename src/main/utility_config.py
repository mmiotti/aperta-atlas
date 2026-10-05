"""
Single source of truth for the utility spec's feature-column mapping
+ pre-fit numeric rescaling.

Both 09a (fit-time producer) and 09b / 10 (runtime consumers) must
agree on which node file each feature is joined from (`POINT_FEATURE_
SOURCES`) and what divisor 09a applied before fitting β (`FEATURE_
SCALE`). Consumers apply the identical rescale.
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
    'bike_infra_score_avg_r100':  ('bike', 'bike_extended'),
    'bike_infra_score_avg_r250':  ('bike', 'bike_extended'),
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
# The keys mix un-oriented source columns (looked up in `common.join_
# util_node_features`) and fully-oriented biogeme names (looked up at
# 09a's fit-time join). The producer / consumer sides each look up
# whichever key applies at their point in the pipeline; a missing entry
# means `scale = 1.0` (no rescale).
FEATURE_SCALE: dict[str, float] = {
    'elev_gain_net':                1_000.0,
    'elev_loss_net':                1_000.0,
    'elev_gain_route':              1_000.0,
    'elev_loss_route':              1_000.0,
    'traffic_flow_avg_r250':        10_000.0,
    'traffic_flow_avg_r250_route':  10_000.0,
    'traffic_flow_avg_r500':        10_000.0,
    'traffic_flow_avg_r500_route':  10_000.0,
}
