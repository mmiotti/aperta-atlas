"""Tests for `aperta_atlas.osm` — OSM tag rules and post-consolidation helpers.

Run with (from the aperta-atlas repo root):
    python -m unittest discover -s tests -t .
"""

import unittest

import networkx as nx
from shapely.geometry import LineString

from aperta_atlas.osm import clean_consolidated_edges, lanes_per_direction


class LanesPerDirectionTestCase(unittest.TestCase):
    """`lanes_per_direction` corrects OSM's bidirectional `lanes` tag for
    use in per-direction quantities (directional AADT, per-lane capacity).
    """

    def test_oneway_returns_lanes_unchanged(self):
        # Motorway: 3 lanes, oneway -> all 3 lanes in this direction.
        self.assertEqual(lanes_per_direction({"lanes": 3, "oneway": True}), 3.0)

    def test_twoway_halves_lanes(self):
        # Two-way primary: 4 total lanes -> 2 per direction.
        self.assertEqual(lanes_per_direction({"lanes": 4, "oneway": False}), 2.0)

    def test_twoway_with_one_lane_returns_one(self):
        # Narrow shared road: 1 lane both ways -> can't split.
        self.assertEqual(lanes_per_direction({"lanes": 1, "oneway": False}), 1.0)

    def test_missing_lanes_defaults_to_one(self):
        # No lanes tag -> OSM implicit default = 1 per direction.
        self.assertEqual(lanes_per_direction({"oneway": False}), 1.0)
        self.assertEqual(lanes_per_direction({"oneway": True}), 1.0)

    def test_string_lanes_parsed(self):
        # OSM often stores lanes as strings.
        self.assertEqual(lanes_per_direction({"lanes": "4", "oneway": False}), 2.0)

    def test_list_lanes_takes_first(self):
        # Post-OSMnx merges occasionally leave list-valued tags.
        self.assertEqual(lanes_per_direction({"lanes": ["4", "4"], "oneway": False}), 2.0)

    def test_unparseable_lanes_defaults_to_one(self):
        self.assertEqual(lanes_per_direction({"lanes": "unknown"}), 1.0)

    def test_nan_lanes_defaults_to_one(self):
        # NaN floats (pandas/numpy) reach lanes_per_direction via CSV-loaded
        # edge properties. `float(nan)` doesn't raise, so a naive
        # `_parse_lanes` would propagate the NaN through `lanes / 2.0`
        # and silently NaN out the lanes_per_direction column. Treat
        # NaN as missing → return the 1.0 default.
        nan = float('nan')
        self.assertEqual(lanes_per_direction({"lanes": nan, "oneway": False}), 1.0)
        self.assertEqual(lanes_per_direction({"lanes": nan, "oneway": True}), 1.0)


class CleanConsolidatedEdgesTestCase(unittest.TestCase):
    """`clean_consolidated_edges` is the standalone post-cleanup for
    OSMnx-consolidated graphs. Tests target it in isolation — feed a
    graph with known dirty attrs, verify they're cleaned according to
    spec."""

    def _make_edge(self, **attrs):
        g = nx.MultiDiGraph()
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=100.0, y=0.0)
        # Default geometry: straight 100m line in a metric CRS.
        attrs.setdefault("geometry", LineString([(0, 0), (100, 0)]))
        g.add_edge(0, 1, key=0, **attrs)
        return g

    def test_default_drop_attrs_removes_name(self):
        g = self._make_edge(name="Bahnhofstrasse", highway="primary")
        clean_consolidated_edges(g)
        self.assertNotIn("name", g[0][1][0])
        # Other attrs preserved.
        self.assertEqual(g[0][1][0]["highway"], "primary")

    def test_custom_drop_attrs(self):
        g = self._make_edge(name="X", ref="A1", highway="motorway")
        clean_consolidated_edges(g, drop_edge_attrs=["ref"])
        self.assertIn("name", g[0][1][0])  # kept (not in custom drop list)
        self.assertNotIn("ref", g[0][1][0])  # dropped

    def test_collapses_list_valued_lanes(self):
        g = self._make_edge(lanes=[2, 4], oneway=True)
        clean_consolidated_edges(g)
        # _mean_numeric averages the list to a scalar.
        self.assertEqual(g[0][1][0]["lanes"], 3.0)

    def test_collapses_list_valued_maxspeed(self):
        g = self._make_edge(maxspeed=["50", "70"])
        clean_consolidated_edges(g)
        self.assertEqual(g[0][1][0]["maxspeed"], 60.0)

    def test_recomputes_length_from_geometry(self):
        # Pre-set a wrong length to confirm it gets overwritten.
        g = self._make_edge(highway="primary", length=9999.0)
        clean_consolidated_edges(g)
        # geometry is 100m straight line -> length should be 100.
        self.assertAlmostEqual(g[0][1][0]["length"], 100.0, places=3)

    def test_no_geometry_skips_length_recompute(self):
        # An edge without geometry: existing length should be untouched.
        g = nx.MultiDiGraph()
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=100.0, y=0.0)
        g.add_edge(0, 1, key=0, highway="primary", length=42.0)
        # Remove geometry so the function falls through.
        if "geometry" in g[0][1][0]:
            del g[0][1][0]["geometry"]
        clean_consolidated_edges(g)
        self.assertEqual(g[0][1][0]["length"], 42.0)

    def test_does_not_write_lanes_per_direction(self):
        # `clean_consolidated_edges` is cleanup-only — derived per-edge
        # attributes like `lanes_per_direction` belong in
        # `networks_decorate.py`. Confirm this function doesn't add it.
        g = self._make_edge(lanes=4, oneway=False, highway="primary")
        clean_consolidated_edges(g)
        self.assertNotIn("lanes_per_direction", g[0][1][0])

    def test_no_edges_is_noop(self):
        g = nx.MultiDiGraph()
        g.add_node(0, x=0.0, y=0.0)
        clean_consolidated_edges(g)  # should not raise
        self.assertEqual(g.number_of_nodes(), 1)

    # ----- bridge / tunnel as numeric attrs from upstream -----
    # Real bridge/tunnel float conversion happens in `networks_from_pbf.py`
    # before `collapse_degree_2_chains`, and the chain collapser produces
    # the true length-weighted fraction. By the time the graph reaches
    # `clean_consolidated_edges`, `bridge` and `tunnel` are already
    # numeric. The only thing this function needs to handle is the rare
    # list case `osmnx.consolidate_intersections` can produce for
    # parallel-edge merges — handled by `_mean_numeric` registered in
    # `_DEFAULT_EDGE_ATTR_AGGS`.

    def test_bridge_float_singleton_preserved(self):
        """A float `bridge` (the typical case after chain collapse) passes
        through unchanged."""
        g = self._make_edge(highway="primary", bridge=0.5, tunnel=0.0)
        clean_consolidated_edges(g)
        self.assertEqual(g[0][1][0]["bridge"], 0.5)
        self.assertEqual(g[0][1][0]["tunnel"], 0.0)

    def test_bridge_list_of_floats_averaged(self):
        """Parallel-edge merge in `consolidate_intersections` may produce
        `bridge=[0.0, 1.0]` — `_mean_numeric` collapses to the mean."""
        g = self._make_edge(highway="primary", bridge=[0.0, 1.0])
        clean_consolidated_edges(g)
        self.assertEqual(g[0][1][0]["bridge"], 0.5)

    def test_tunnel_list_of_floats_averaged(self):
        g = self._make_edge(highway="primary", tunnel=[1.0, 0.5, 0.0])
        clean_consolidated_edges(g)
        self.assertAlmostEqual(g[0][1][0]["tunnel"], 0.5, places=6)


# =====================================================================
# `flag_node_osm_classification`
# =====================================================================

from aperta.network_processing import flag_node_intersection_topology
from aperta_atlas.osm import flag_node_osm_classification


def _flag_all(g):
    """Test helper: topology + OSM classification in one call."""
    flag_node_intersection_topology(g)
    flag_node_osm_classification(g)


class FlagNodeIntersectionsTestCase(unittest.TestCase):
    """`flag_node_intersection_topology` + `flag_node_osm_classification`
    write per-node attributes describing intersection type (`n_streets`,
    `is_t_junction`, `is_4way`), their rank-conditional variants (`_major`,
    `_anchor`), and per-node max / min OSM highway-rank.
    """

    def _graph(self) -> nx.MultiDiGraph:
        """Mixed-degree fixture:
            1: 4-way intersection (n_streets=4) — primary + residential
            2: passthrough (n_streets=2) — primary (1↔2) + residential (2↔6)
            3, 4, 5: leaves (n_streets=1) on residential
            6: leaf (n_streets=1) on residential
        Highway tags chosen so node 1 sees both primary (rank 5) and
        residential (rank 2) — tests max/min rank.
        """
        g = nx.MultiDiGraph()
        for n, x, y in [(1, 0, 0), (2, 1, 0), (3, -1, 0), (4, 0, 1), (5, 0, -1), (6, 2, 0)]:
            g.add_node(n, x=float(x), y=float(y))
        for u, v, hw in [
            (1, 2, "primary"),
            (1, 3, "residential"),
            (1, 4, "residential"),
            (1, 5, "residential"),
            (2, 6, "residential"),
        ]:
            g.add_edge(u, v, highway=hw)
            g.add_edge(v, u, highway=hw)
        return g

    def test_basic_intersection_flags_mutually_exclusive(self):
        """is_t_junction = exactly 3 distinct neighbours; is_4way = ≥ 4;
        never both set on the same node. Leaves and passthroughs get neither."""
        g = self._graph()
        _flag_all(g)
        # Node 1: 4 distinct neighbours → only is_4way.
        self.assertEqual(g.nodes[1]["n_streets"], 4.0)
        self.assertEqual(g.nodes[1]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[1]["is_4way"], 1.0)
        # Node 2: passthrough → neither.
        self.assertEqual(g.nodes[2]["n_streets"], 2.0)
        self.assertEqual(g.nodes[2]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[2]["is_4way"], 0.0)
        # Leaf node 6: 1 neighbour → neither.
        self.assertEqual(g.nodes[6]["n_streets"], 1.0)
        self.assertEqual(g.nodes[6]["is_t_junction"], 0.0)
        self.assertEqual(g.nodes[6]["is_4way"], 0.0)

    def test_t_junction_fires_at_exactly_three(self):
        """3-way intersection (T-junction) lights up is_t_junction."""
        g = nx.MultiDiGraph()
        for n, (x, y) in enumerate([(0, 0), (1, 0), (-1, 0), (0, 1)]):
            g.add_node(n, x=float(x), y=float(y))
        for u, v in [(0, 1), (0, 2), (0, 3)]:
            g.add_edge(u, v)
            g.add_edge(v, u)
        _flag_all(g)
        self.assertEqual(g.nodes[0]["n_streets"], 3.0)
        self.assertEqual(g.nodes[0]["is_t_junction"], 1.0)
        self.assertEqual(g.nodes[0]["is_4way"], 0.0)

    def test_max_min_highway_rank(self):
        """max/min from OSM_HIGHWAY_RANKS over incident edges."""
        g = self._graph()
        _flag_all(g)
        from aperta_atlas.osm import OSM_HIGHWAY_RANKS

        # Node 1: edges of types {primary, residential} → max=5, min=2.
        self.assertEqual(g.nodes[1]["max_highway_rank"], float(OSM_HIGHWAY_RANKS["primary"]))
        self.assertEqual(g.nodes[1]["min_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))
        # Node 6: only residential edges → max=min=2.
        self.assertEqual(g.nodes[6]["max_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))
        self.assertEqual(g.nodes[6]["min_highway_rank"], float(OSM_HIGHWAY_RANKS["residential"]))

    def test_undirected_graph_works(self):
        """Undirected graphs use `graph.neighbors`, not predecessors/successors."""
        g = nx.MultiGraph()
        g.add_node(0, x=0.0, y=0.0)
        for i, (x, y) in enumerate([(1, 0), (-1, 0), (0, 1)], start=1):
            g.add_node(i, x=float(x), y=float(y))
            g.add_edge(0, i)
        _flag_all(g)
        # 3 distinct neighbours of node 0 → is_t_junction set, is_4way clear.
        self.assertEqual(g.nodes[0]["n_streets"], 3.0)
        self.assertEqual(g.nodes[0]["is_t_junction"], 1.0)
        self.assertEqual(g.nodes[0]["is_4way"], 0.0)

    def test_major_requires_min_rank_ge_3(self):
        """`_major` variants need every incident edge to be tertiary or better."""
        # Node A: 3-way T-junction with three primary edges → major qualifies.
        g_pure_t = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_t.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_t.add_edge(u, v, highway="primary")
            g_pure_t.add_edge(v, u, highway="primary")
        _flag_all(g_pure_t)
        self.assertEqual(g_pure_t.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_t.nodes["A"]["is_t_junction_major"], 1.0)

        # Node from `_graph` fixture: 4-way with one primary + three
        # residential → min_rank = 2 (residential) → major fails.
        g_mixed = self._graph()
        _flag_all(g_mixed)
        self.assertEqual(g_mixed.nodes[1]["is_4way"], 1.0)
        self.assertEqual(g_mixed.nodes[1]["is_4way_major"], 0.0)  # has a residential branch

    def test_anchor_requires_max_rank_ge_3_and_min_rank_le_5(self):
        """`_anchor` variants need ≥1 tertiary+ edge AND not pure trunk/motorway."""
        # Mixed residential + primary 4-way: max=5 (primary), min=2 (residential).
        # 5 >= 3 ✓ and 2 <= 5 ✓ → anchor qualifies.
        g_mixed = self._graph()
        _flag_all(g_mixed)
        self.assertEqual(g_mixed.nodes[1]["is_4way"], 1.0)
        self.assertEqual(g_mixed.nodes[1]["is_4way_anchor"], 1.0)

        # Pure-residential T-junction: max=2 → fails `max >= 3` → anchor=0.
        g_pure_res = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_res.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_res.add_edge(u, v, highway="residential")
            g_pure_res.add_edge(v, u, highway="residential")
        _flag_all(g_pure_res)
        self.assertEqual(g_pure_res.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_res.nodes["A"]["is_t_junction_anchor"], 0.0)

        # Pure-motorway T-junction: max=min=7 → fails `min <= 5` → anchor=0.
        g_pure_mw = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D"):
            g_pure_mw.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D")]:
            g_pure_mw.add_edge(u, v, highway="motorway")
            g_pure_mw.add_edge(v, u, highway="motorway")
        _flag_all(g_pure_mw)
        self.assertEqual(g_pure_mw.nodes["A"]["is_t_junction"], 1.0)
        self.assertEqual(g_pure_mw.nodes["A"]["is_t_junction_anchor"], 0.0)

    def test_major_is_a_subset_of_anchor_when_max_rank_le_5(self):
        """If every edge is tertiary–primary (rank 3–5), the node is BOTH
        major (min >= 3) AND anchor (max >= 3, min <= 5)."""
        g = nx.MultiDiGraph()
        for n in ("A", "B", "C", "D", "E"):
            g.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]:
            g.add_edge(u, v, highway="tertiary")
            g.add_edge(v, u, highway="tertiary")
        _flag_all(g)
        self.assertEqual(g.nodes["A"]["is_4way"], 1.0)
        self.assertEqual(g.nodes["A"]["is_4way_major"], 1.0)
        self.assertEqual(g.nodes["A"]["is_4way_anchor"], 1.0)

    def test_passthrough_node_gets_no_intersection_flags(self):
        """Passthrough (n_streets=2) is neither T-junction nor 4-way, regardless of rank."""
        g = nx.MultiDiGraph()
        for n in ("A", "B", "C"):
            g.add_node(n, x=0.0, y=0.0)
        for u, v in [("A", "B"), ("B", "C")]:
            g.add_edge(u, v, highway="primary")
            g.add_edge(v, u, highway="primary")
        _flag_all(g)
        self.assertEqual(g.nodes["B"]["n_streets"], 2.0)
        # All intersection flags off:
        for flag in (
            "is_t_junction",
            "is_4way",
            "is_t_junction_major",
            "is_4way_major",
            "is_t_junction_anchor",
            "is_4way_anchor",
        ):
            self.assertEqual(g.nodes["B"][flag], 0.0, f"{flag} should be 0 for passthrough")


from aperta_atlas.osm import is_usable  # noqa: E402


class AccessOverrideTestCase(unittest.TestCase):
    """A mode-specific tag overrides `access=no|private` (OSM access hierarchy)."""

    def test_closed_to_cars_open_to_bikes_and_walkers(self):
        tags = {"highway": "residential", "access": "no", "bicycle": "yes", "foot": "yes"}
        self.assertTrue(is_usable(tags, "bike"))
        self.assertTrue(is_usable(tags, "walk"))
        self.assertFalse(is_usable(tags, "drive"))

    def test_plain_access_no_still_excludes(self):
        tags = {"highway": "service", "access": "private"}
        for mode in ("walk", "bike", "drive"):
            self.assertFalse(is_usable(tags, mode))

    def test_explicit_no_wins(self):
        self.assertFalse(is_usable({"highway": "track", "access": "no", "bicycle": "no"}, "bike"))

    def test_drive_needs_general_traffic(self):
        self.assertTrue(is_usable(
            {"highway": "residential", "access": "no", "motor_vehicle": "destination"}, "drive"))
        self.assertFalse(is_usable(
            {"highway": "unclassified", "access": "no", "motor_vehicle": "forestry"}, "drive"))


class MountainTrailsTestCase(unittest.TestCase):
    """Mountain-hiking trails are not in the bike network unless in a mountain-bike route."""

    def test_sac_scale(self):
        trail = {"highway": "path", "sac_scale": "mountain_hiking", "bicycle": "yes"}
        self.assertFalse(is_usable(trail, "bike"))
        self.assertTrue(is_usable(trail, "walk"))
        self.assertTrue(is_usable({**trail, "mtb_route": "yes"}, "bike"))
        self.assertTrue(is_usable({"highway": "path", "sac_scale": "hiking"}, "bike"))

    def test_via_ferrata(self):
        for mode in ("walk", "bike"):
            self.assertFalse(is_usable({"highway": "via_ferrata"}, mode))


from aperta_atlas.osm import (  # noqa: E402
    bike_push, bike_stress, bike_stress_beside, cycleway_in_direction, lanes_in_direction,
    parse_maxspeed_kmh, surface_type,
)


class SurfaceTypeTestCase(unittest.TestCase):

    def test_values(self):
        cases = [({"surface": "asphalt"}, "paved"), ({"surface": "sett"}, "cobble"),
                 ({"surface": "gravel"}, "unpaved_smooth"),
                 ({"surface": "dirt"}, "unpaved_rough"),
                 ({"tracktype": "grade1"}, "paved"), ({"tracktype": "grade2"}, "unpaved_smooth"),
                 ({"tracktype": "grade4"}, "unpaved_rough"),
                 ({"surface": "asphalt", "tracktype": "grade4"}, "paved"),    # surface first
                 ({"surface": "asphalt;gravel", "tracktype": "grade3"}, "unpaved_rough"),
                 # smoothness decides for unpaved ways only
                 ({"surface": "gravel", "smoothness": "bad"}, "unpaved_rough"),
                 ({"surface": "ground", "smoothness": "intermediate"}, "unpaved_smooth"),
                 ({"surface": "asphalt", "smoothness": "bad"}, "paved"),
                 # merged edges: the first value with a type
                 ({"surface": [float("nan"), "gravel"], "smoothness": ["bad"]}, "unpaved_rough"),
                 ({"surface": "asphalt;gravel"}, None), ({"surface": float("nan")}, None),
                 ({"highway": "track"}, None)]
        for tags, expected in cases:
            self.assertEqual(surface_type(tags), expected, msg=tags)


class CyclewayInDirectionTestCase(unittest.TestCase):
    """Which direction a side-specific bike lane serves."""

    def both(self, tags, **kw):
        return (cycleway_in_direction(tags, True, **kw), cycleway_in_direction(tags, False, **kw))

    def test_two_way(self):
        self.assertEqual(self.both({"cycleway:right": "lane"}), ("lane", "no"))
        self.assertEqual(self.both({"cycleway:right": "lane"}, left_hand_traffic=True),
                         ("no", "lane"))
        self.assertEqual(self.both({"cycleway:left": "track", "cycleway:right": "lane"}),
                         ("lane", "track"))
        self.assertEqual(self.both({"cycleway": "lane"}), ("lane", "lane"))
        self.assertEqual(self.both({"cycleway:both": "separate"}), ("no", "no"))

    def test_one_way(self):
        one_way = {"oneway": "yes", "oneway:bicycle": "no"}
        self.assertEqual(self.both({**one_way, "cycleway:right": "lane",
                                    "cycleway:left": "opposite_lane"}), ("lane", "opposite_lane"))
        self.assertEqual(self.both({**one_way, "cycleway:left": "lane"}), ("no", "lane"))
        self.assertEqual(self.both({"oneway": "yes", "cycleway:left": "lane"}), ("lane", "no"))
        self.assertEqual(self.both({**one_way, "cycleway": "opposite"}), ("no", "opposite"))
        self.assertEqual(self.both({"oneway": "-1", "cycleway:right": "lane"}), ("no", "lane"))
        # oneway=-1: traffic runs against the way, so its traffic side is the way's left
        self.assertEqual(self.both({"oneway": "-1", "cycleway:left": "lane"}), ("no", "lane"))
        self.assertEqual(self.both({"oneway": "-1", "oneway:bicycle": "no",
                                    "cycleway:right": "lane"}), ("lane", "no"))



class MaxspeedTestCase(unittest.TestCase):

    def test_values(self):
        cases = {"50": 50, "30 mph": 30 * 1.609344, "CH:urban": 50, "CH:rural": 80,
                 "DE:rural": 100, "CH:zone30": 30, "GB:nsl_single": 60 * 1.609344,
                 "50;80": 80, "80": 80}
        for raw, kmh in cases.items():
            self.assertAlmostEqual(parse_maxspeed_kmh(raw), kmh, places=3, msg=raw)
        for raw in (None, float("nan"), "", "signals", "0", 0.0, "0 mph"):
            self.assertIsNone(parse_maxspeed_kmh(raw))


class BikeStressTestCase(unittest.TestCase):
    """Bike traffic stress, 0 (none) to 5 (very high)."""

    def test_mixed_traffic(self):
        self.assertEqual(bike_stress({"highway": "residential"}), 2)          # untagged: quiet
        self.assertEqual(bike_stress({"highway": "residential", "maxspeed": "50"}), 3)
        self.assertEqual(bike_stress({"highway": "tertiary", "maxspeed": "30"}), 3)
        self.assertEqual(bike_stress({"highway": "secondary"}), 4)           # untagged main: 50
        self.assertEqual(bike_stress({"highway": "secondary", "lanes_dir": 2.0}), 5)
        self.assertEqual(bike_stress({"highway": "primary", "maxspeed": "80"}), 5)
        self.assertEqual(bike_stress({"highway": "unclassified", "maxspeed": "80"}), 3)
        self.assertEqual(bike_stress({"highway": "residential", "lanes_dir": 2.0}), 2)
        self.assertEqual(bike_stress({"highway": "living_street"}), 1)
        self.assertEqual(bike_stress({"highway": "residential", "cyclestreet": "yes"}), 1)

    def test_painted_lane(self):
        self.assertEqual(bike_stress({"highway": "secondary", "cycleway_dir": "lane"}), 3)
        self.assertEqual(bike_stress({"highway": "secondary", "cycleway_dir": "share_busway"}), 3)
        self.assertEqual(bike_stress({"highway": "secondary", "cycleway_dir": "no"}), 4)
        self.assertEqual(bike_stress({"highway": "residential", "cycleway_dir": "lane"}), 1)
        self.assertEqual(bike_stress(
            {"highway": "tertiary", "maxspeed": "30", "cycleway_dir": "lane"}), 2)
        self.assertEqual(bike_stress(
            {"highway": "primary", "maxspeed": "CH:rural", "cycleway_dir": "lane"}), 4)
        self.assertEqual(bike_stress(                       # sharrows are mixed traffic
            {"highway": "secondary", "cycleway_dir": "shared_lane"}), 4)

    def test_separated_by_road_beside(self):
        self.assertEqual(bike_stress({"highway": "cycleway"}), 0)
        self.assertEqual(bike_stress({"highway": "cycleway"}, beside=3), 3)
        self.assertEqual(bike_stress({"highway": "secondary", "cycleway_dir": "track"}), 2)
        self.assertEqual(bike_stress_beside({"highway": "residential"}), 0)
        self.assertEqual(bike_stress_beside({"highway": "primary"}), 2)
        self.assertEqual(bike_stress_beside({"highway": "primary", "maxspeed": "80"}), 3)

    def test_one_way_and_climb_move_one_row(self):
        street = {"highway": "residential"}                          # row (0, 1, 2)
        self.assertEqual(bike_stress({**street, "oneway_motor": 1.0}), 1)
        self.assertEqual(bike_stress(street, slope=0.06), 3)
        self.assertEqual(bike_stress(street, slope=0.04), 2)         # below the threshold
        self.assertEqual(bike_stress(street, slope=-0.08), 2)        # descents are neutral
        main = {"highway": "secondary", "cycleway_dir": "lane"}      # row (2, 3, 4)
        self.assertEqual(bike_stress({**main, "oneway_motor": 1.0}), 2)
        self.assertEqual(bike_stress(main, slope=0.06), 4)
        self.assertEqual(bike_stress({**main, "oneway_motor": 1.0}, slope=0.06), 3)   # cancel
        self.assertEqual(bike_stress({"highway": "primary", "maxspeed": "80"}, slope=0.06), 5)
        self.assertEqual(bike_stress({"highway": "living_street", "oneway_motor": 1.0}), 1)

    def test_separated_gets_no_modifiers(self):
        track = {"highway": "secondary", "cycleway_dir": "track", "oneway_motor": 1.0}
        self.assertEqual(bike_stress(track, slope=-0.05), 2)
        self.assertEqual(bike_stress({"highway": "cycleway"}, beside=2, slope=0.08), 2)

    def test_lanes_in_direction(self):
        self.assertEqual(lanes_in_direction({"lanes": "4"}, True), 2)
        self.assertEqual(lanes_in_direction(
            {"lanes": "5", "lanes:forward": "3", "lanes:backward": "2"}, True), 3)
        self.assertEqual(lanes_in_direction(                 # minus the bus lane
            {"lanes": "4", "lanes:bus:forward": "1"}, True), 1)
        self.assertEqual(lanes_in_direction({"lanes": "2", "oneway": "yes"}, True), 2)
        self.assertEqual(lanes_in_direction({"lanes": "2", "oneway": "yes"}, False), 0)
        self.assertIsNone(lanes_in_direction({}, True))

    def test_car_free_road_and_push(self):
        self.assertEqual(bike_stress(
            {"highway": "residential", "access": "no", "bicycle": "yes"}), 0)
        self.assertTrue(bike_push({"highway": "footway"}))
        self.assertFalse(bike_push({"highway": "footway", "bicycle": "yes"}))
        self.assertTrue(bike_push({"highway": "residential", "bicycle": "dismount"}))
        self.assertFalse(bike_push({"highway": ["footway", "residential"]}))
        self.assertFalse(bike_push({"highway": "path", "bicycle": ["dismount", "designated"]}))
        # Hard mountain-bike trails, whatever the bike access; a float when read back from CSV.
        self.assertTrue(bike_push({"highway": "path", "bicycle": "designated", "mtb:scale": "2+"}))
        self.assertTrue(bike_push({"highway": "track", "mtb:scale": 3.0}))
        self.assertFalse(bike_push({"highway": "path", "mtb:scale": "1"}))
        self.assertFalse(bike_push({"highway": "path", "mtb:scale": ["3", "0"]}))
        self.assertEqual(bike_stress(                       # destination traffic: not car-free
            {"highway": "residential", "access": "no", "motor_vehicle": "destination"}), 2)
        self.assertTrue(bike_stress({"highway": "footway"}) != bike_stress({"highway": "footway"}))


from aperta_atlas.osm import drop_sidepath_ways  # noqa: E402


class DropSidepathWaysTestCase(unittest.TestCase):
    """A 150 m street in WGS84 near Bern: road A–M–B, a sidewalk SA–SB 8 m north of it,
    linked to the road at both ends."""

    LAT, DLAT = 46.95, 0.00007                     # 0.00007° latitude ≈ 8 m

    def _graph(self, road_tags, sidewalk_tags, sidewalk_dlat=DLAT, extra=()):
        g = nx.MultiDiGraph(crs="EPSG:4326")
        pts = {"A": (7.4400, self.LAT), "M": (7.4410, self.LAT), "B": (7.4420, self.LAT),
               "SA": (7.4400, self.LAT + sidewalk_dlat), "SB": (7.4420, self.LAT + sidewalk_dlat),
               "X": (7.4410, self.LAT - 0.0005)}
        for n, (x, y) in pts.items():
            g.add_node(n, x=x, y=y)

        def way(osmid, nodes, tags):
            for a, b in zip(nodes, nodes[1:]):
                for u, v in ((a, b), (b, a)):
                    g.add_edge(u, v, osmid=osmid, geometry=LineString([pts[u], pts[v]]), **tags)

        way(1, ["A", "M", "B"], road_tags)
        way(2, ["SA", "SB"], sidewalk_tags)
        way(3, ["SA", "A"], {"highway": "footway"})
        way(4, ["SB", "B"], {"highway": "footway"})
        for osmid, nodes, tags in extra:
            way(osmid, nodes, tags)
        return g

    def _ways(self, g):
        return sorted({d["osmid"] for *_, d in g.edges(data=True)})

    def test_bike_drops_sidewalk_beside_road(self):
        g = self._graph({"highway": "residential"}, {"highway": "footway", "footway": "sidewalk"})
        stats = drop_sidepath_ways(g, "bike")
        self.assertEqual(self._ways(g), [1, 3, 4])
        self.assertEqual(stats["ways_dropped"], 1)

    def test_bike_keeps_sidewalk_with_any_bicycle_tag(self):
        g = self._graph({"highway": "residential"},
                        {"highway": "footway", "footway": "sidewalk", "bicycle": "dismount"})
        drop_sidepath_ways(g, "bike")
        self.assertEqual(self._ways(g), [1, 2, 3, 4])

    def test_bike_keeps_sidewalk_without_road_alongside(self):
        g = self._graph({"highway": "residential"}, {"highway": "footway", "footway": "sidewalk"},
                        sidewalk_dlat=0.0004)          # ≈ 45 m away
        stats = drop_sidepath_ways(g, "bike")
        self.assertEqual(self._ways(g), [1, 2, 3, 4])
        self.assertEqual(stats["ways_kept_no_alongside"], 1)

    def test_bike_use_sidepath_road_needs_track_alongside(self):
        road = {"highway": "secondary", "bicycle": "use_sidepath"}
        g = self._graph(road, {"highway": "cycleway"})
        drop_sidepath_ways(g, "bike")
        self.assertEqual(self._ways(g), [2, 3, 4])
        g = self._graph(road, {"highway": "footway"})   # no bike way beside it: keep the road
        drop_sidepath_ways(g, "bike")
        self.assertEqual(self._ways(g), [1, 2, 3, 4])

    def test_bike_keeps_sidewalk_beside_one_way_road(self):
        """Against a one-way road, pushing on the sidewalk is the only way: keep it."""
        g = self._graph({"highway": "residential"}, {"highway": "footway", "footway": "sidewalk"})
        g.remove_edges_from([(v, u) for u, v, d in list(g.edges(data=True)) if d["osmid"] == 1
                             and (u, v) in (("A", "M"), ("M", "B"))])   # road one-way A→B
        drop_sidepath_ways(g, "bike")
        self.assertIn(2, self._ways(g))

    def test_walk_drops_road_with_separate_sidewalks(self):
        g = self._graph({"highway": "residential", "sidewalk": "separate"},
                        {"highway": "footway", "footway": "sidewalk"})
        drop_sidepath_ways(g, "walk")
        self.assertEqual(self._ways(g), [2, 3, 4])

    def test_road_put_back_if_a_side_road_hangs_on_it(self):
        """The side road M–X only connects to the road at M, which the sidewalk does not touch."""
        g = self._graph({"highway": "residential", "sidewalk": "separate"},
                        {"highway": "footway", "footway": "sidewalk"},
                        extra=[(5, ["M", "X"], {"highway": "service"})])
        stats = drop_sidepath_ways(g, "walk")
        self.assertEqual(self._ways(g), [1, 2, 3, 4, 5])
        self.assertEqual((stats["ways_dropped"], stats["ways_put_back"]), (0, 1))

    def test_car_unchanged(self):
        g = self._graph({"highway": "residential", "sidewalk": "separate"},
                        {"highway": "footway", "footway": "sidewalk"})
        drop_sidepath_ways(g, "drive")
        self.assertEqual(len(self._ways(g)), 4)


if __name__ == "__main__":
    unittest.main()
