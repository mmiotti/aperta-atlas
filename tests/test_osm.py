"""Tests for `aperta_atlas.osm` — OSM-specific post-consolidation helpers.

Run with (from the aperta-atlas repo root):
    python -m unittest discover -s tests -t .

`clean_consolidated_edges` + `lanes_per_direction` were moved here from
`aperta.network_processing` on 2026-06-05: the `lanes` / `oneway`
semantics they rely on are OSM conventions, so the public helpers belong
in aperta-atlas's OSM-aware layer rather than in the algorithm library.
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
# Tests moved from aperta/tests/test_network_processing.py 2026-06-07
# as part of the sweeping cleanup that pushed all OSM-aware code to
# aperta-atlas. These exercise `flag_node_osm_classification`,
# `consolidate_intersections`, and `load_consolidated_graphml` —
# all now in `aperta_atlas.osm`.
# =====================================================================

import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString as _LineString, Point, box  # noqa: F401

from aperta.network_processing import flag_node_intersection_topology
from aperta_atlas.osm import (
    consolidate_intersections,
    flag_node_osm_classification,
    load_consolidated_graphml,
)


def _flag_all(g):
    """Test helper: topology + OSM classification in one call."""
    flag_node_intersection_topology(g)
    flag_node_osm_classification(g)


class FlagNodeIntersectionsTestCase(unittest.TestCase):
    """`flag_node_intersection_topology` + `flag_node_osm_classification`
    write per-node attributes describing intersection type (`n_streets`,
    `is_t_junction`, `is_4way`), their rank-conditional variants (`_major`,
    `_anchor`), and per-node max / min OSM highway-rank. Obstacle flags
    (traffic signals etc.) live in `consolidate_intersections`, not here.
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


class ConsolidateIntersectionsTestCase(unittest.TestCase):
    """`consolidate_intersections` wraps `osmnx.consolidate_intersections`,
    plus reattaches obstacle flags (traffic signals, stops, roundabouts)
    that OSMnx alone would drop when their host nodes are merged away.
    """

    def _graph_with_signal_and_roundabout(self) -> nx.MultiDiGraph:
        """4-arm intersection at (1000, 1000) with a traffic_signal node 5 m
        offset (typical OSM pattern — signals tagged on the approach, not
        the centre). Separately, a small roundabout (two nodes 11 m apart,
        connected by a `junction=roundabout` edge) at (2000, 2000).
        """
        g = nx.MultiDiGraph(crs="EPSG:2056")
        g.add_node(1, x=1000.0, y=1000.0)
        for n, (x, y) in zip([2, 3, 4, 5], [(1100, 1000), (900, 1000), (1000, 1100), (1000, 900)]):
            g.add_node(n, x=float(x), y=float(y))
        # Signal sits 5√2 ≈ 7 m east-northeast of the intersection centre.
        g.add_node(6, x=1005.0, y=1005.0, highway="traffic_signals")
        for u, v in [(1, 2), (1, 3), (1, 4), (1, 5)]:
            g.add_edge(u, v)
            g.add_edge(v, u)
        # East arm goes through the signal node.
        g.add_edge(2, 6)
        g.add_edge(6, 1)
        g.add_edge(1, 6)
        g.add_edge(6, 2)
        # Roundabout: two nodes ~11 m apart with junction=roundabout edges.
        g.add_node(10, x=2000.0, y=2000.0)
        g.add_node(11, x=2010.0, y=2005.0)
        g.add_edge(10, 11, junction="roundabout")
        g.add_edge(11, 10, junction="roundabout")
        return g

    def test_signal_reallocated_to_consolidated_node(self):
        """The off-centre traffic_signal node is dropped during consolidation
        but its flag re-attaches to the consolidated 4-way intersection."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        # Find the consolidated central intersection (degree ≥ 4 near 1000,1000).
        central = None
        for nid, d in consolidated.nodes(data=True):
            if abs(d["x"] - 1000) < 30 and abs(d["y"] - 1000) < 30 and d.get("is_4way") == 1.0:
                central = nid
                break
        self.assertIsNotNone(central, "no consolidated 4-way intersection found")
        self.assertEqual(consolidated.nodes[central]["is_traffic_signal"], 1.0)

    def test_roundabout_detected_from_edge_tag(self):
        """A node consolidated from a `junction=roundabout` edge is flagged."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        rb_nodes = [
            nid for nid, d in consolidated.nodes(data=True) if d.get("is_roundabout") == 1.0
        ]
        self.assertEqual(len(rb_nodes), 1)
        self.assertAlmostEqual(consolidated.nodes[rb_nodes[0]]["x"], 2005, delta=10)
        self.assertAlmostEqual(consolidated.nodes[rb_nodes[0]]["y"], 2002.5, delta=10)

    def test_non_intersection_nodes_have_zero_flags(self):
        """Arm-tip nodes (degree 1 in the original) carry no obstacle flags."""
        g = self._graph_with_signal_and_roundabout()
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=30.0)
        # Whichever nodes ended up near the arm tips (not within tolerance of
        # the centre) should have all flags 0.
        for nid, d in consolidated.nodes(data=True):
            if abs(d["x"] - 1000) > 50 and abs(d["x"] - 2005) > 30:
                self.assertEqual(d.get("is_traffic_signal", 0.0), 0.0)
                self.assertEqual(d.get("is_roundabout", 0.0), 0.0)

    def test_obstacle_buffer_excludes_far_signals(self):
        """A signal further than `obstacle_buffer` is NOT attached."""
        g = self._graph_with_signal_and_roundabout()
        # With buffer=2 m the signal at (1005,1005) is too far from the
        # consolidated central node at ~(1001,1001).
        consolidated = consolidate_intersections(g, tolerance=20.0, obstacle_buffer=2.0)
        any_signal = any(
            d.get("is_traffic_signal") == 1.0 for _, d in consolidated.nodes(data=True)
        )
        self.assertFalse(any_signal)


class GraphmlBoolRoundtripTestCase(unittest.TestCase):
    """Bool-typed per-mode flags (`is_snap_eligible_<mode>`,
    `cost_excluded_<mode>`, `is_virtual`) round-trip through .graphml as
    integers (0 / 1), not the literal strings 'True' / 'False' — thanks to
    the prefix-scan dtype helper in `load_consolidated_graphml`."""

    def _g(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph(crs="EPSG:4326")
        for n, (x, y) in {0: (0, 0), 1: (1, 0), 2: (2, 0)}.items():
            g.add_node(n, x=float(x), y=float(y), osmid=n)
        g.add_edge(0, 1, key=0, highway="residential", length=1.0)
        g.add_edge(1, 2, key=0, highway="motorway", length=1.0)
        return g

    def test_bool_flags_roundtrip_as_int(self):
        import tempfile

        import osmnx as ox

        from aperta.routing_prep import prepare_network
        from aperta_atlas.osm import load_consolidated_graphml

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepared = prepare_network(self._g(), "walk")
            # Tag one node as virtual to exercise the is_virtual prefix.
            prepared.graph.nodes[1]["is_virtual"] = 1
            ox.save_graphml(prepared.graph, tmp_path)

            loaded = load_consolidated_graphml(tmp_path)
            # Per-node is_snap_eligible_walk: every node should have a real int.
            for n in loaded.nodes:
                val = loaded.nodes[n].get("is_snap_eligible_walk")
                self.assertIsInstance(val, int, f"node {n} has wrong dtype")
            # Per-edge cost_excluded_walk: every edge.
            if loaded.is_multigraph():
                for _u, _v, _k, d in loaded.edges(keys=True, data=True):
                    val = d.get("cost_excluded_walk")
                    self.assertIsInstance(val, int)
            # is_virtual=1 round-trips as int 1 on node 1.
            self.assertEqual(loaded.nodes[1].get("is_virtual"), 1)
        finally:
            import os

            os.unlink(tmp_path)


# =====================================================================
# Tests moved from aperta-atlas/tests/test_osm_helpers.py 2026-06-07
# (file deleted) as part of the final osm_helpers.py cleanup.
# These exercise `osm_tag_query_for_categories` + `categorize_pois`,
# both now in `aperta_atlas.osm`.
# =====================================================================

from aperta_atlas.osm import categorize_pois, osm_tag_query_for_categories

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "groceries": [
        ("shop:supermarket", 1.0),
        ("shop:convenience", 0.5),
        ("shop:bakery", 0.5),
    ],
    "schools": [
        ("amenity:school", 1.0),
    ],
    # Test multi-tag-key category and integer weight.
    "transit_rail": [
        ("railway:station", 3),
        ("railway:halt", 1),
    ],
}


def _toy_pois() -> gpd.GeoDataFrame:
    """Six POIs covering: 1 supermarket, 1 convenience (also has a bus stop
    tag), 1 bakery, 1 school, 1 railway station, 1 unrelated (fuel)."""
    data = {
        "amenity": ["fuel", None, None, "school", None, "restaurant"],
        "shop": [None, "convenience", "bakery", None, None, None],
        "railway": [None, None, None, None, "station", None],
        "highway": [None, "bus_stop", None, None, None, None],
    }
    geom = [Point(i, 0) for i in range(len(data["amenity"]))]
    return gpd.GeoDataFrame(
        data,
        geometry=geom,
        index=pd.Index(["p0", "p1", "p2", "p3", "p4", "p5"], name="osm_id"),
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# osm_tag_query_for_categories
# ---------------------------------------------------------------------------


class OsmTagQueryTestCase(unittest.TestCase):
    def test_unions_keys_across_categories(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        self.assertSetEqual(set(q.keys()), {"shop", "amenity", "railway"})

    def test_unions_values_per_key(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        self.assertSetEqual(set(q["shop"]), {"supermarket", "convenience", "bakery"})
        self.assertSetEqual(set(q["amenity"]), {"school"})
        self.assertSetEqual(set(q["railway"]), {"station", "halt"})

    def test_values_are_sorted_for_determinism(self):
        q = osm_tag_query_for_categories(CATEGORY_MAP)
        for key, vals in q.items():
            self.assertEqual(vals, sorted(vals), f"values for {key!r} not sorted")

    def test_duplicate_tag_across_categories_deduped(self):
        """Same (tag:value) reused across categories collapses to one value."""
        cm = {
            "a": [("shop:supermarket", 1.0)],
            "b": [("shop:supermarket", 0.5)],  # same OSM tag, different weight
        }
        q = osm_tag_query_for_categories(cm)
        self.assertEqual(q["shop"], ["supermarket"])

    def test_missing_colon_raises(self):
        with self.assertRaisesRegex(ValueError, "'key:value'"):
            osm_tag_query_for_categories({"bad": [("no_colon_here", 1.0)]})


# ---------------------------------------------------------------------------
# categorize_pois
# ---------------------------------------------------------------------------


class CategorizePoisTestCase(unittest.TestCase):
    def test_count_and_weight_columns_added(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP)
        for cat in CATEGORY_MAP:
            self.assertIn(cat, out.columns)
            self.assertIn(f"{cat}_weight", out.columns)

    def test_count_values(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=False)
        # p0 (fuel): no match anywhere.
        self.assertEqual(out.loc["p0", "groceries"], 0)
        # p1 (shop:convenience): matches groceries (1 listed pair).
        self.assertEqual(out.loc["p1", "groceries"], 1)
        # p2 (shop:bakery): matches groceries (1 listed pair).
        self.assertEqual(out.loc["p2", "groceries"], 1)
        # p3 (amenity:school): matches schools, not groceries.
        self.assertEqual(out.loc["p3", "schools"], 1)
        self.assertEqual(out.loc["p3", "groceries"], 0)
        # p4 (railway:station): matches transit_rail.
        self.assertEqual(out.loc["p4", "transit_rail"], 1)

    def test_weight_values(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=False)
        # p1 (convenience): weight = 0.5.
        self.assertEqual(out.loc["p1", "groceries_weight"], 0.5)
        # p2 (bakery): weight = 0.5.
        self.assertEqual(out.loc["p2", "groceries_weight"], 0.5)
        # p4 (railway:station): weight = 3 (integer weight handled).
        self.assertEqual(out.loc["p4", "transit_rail_weight"], 3.0)

    def test_drop_unmatched(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, drop_unmatched=True)
        # p0 (fuel) and p5 (restaurant) match nothing → dropped.
        self.assertNotIn("p0", out.index)
        self.assertNotIn("p5", out.index)
        # The rest stay.
        self.assertSetEqual(set(out.index), {"p1", "p2", "p3", "p4"})

    def test_drop_unmatched_false_keeps_all(self):
        toy = _toy_pois()
        out = categorize_pois(toy, CATEGORY_MAP, drop_unmatched=False)
        self.assertEqual(len(out), len(toy))

    def test_missing_tag_column_silently_skipped(self):
        """If a category references a tag key the DataFrame doesn't have,
        that pair simply doesn't match — no error."""
        toy = _toy_pois().drop(columns=["railway"])
        out = categorize_pois(toy, CATEGORY_MAP, drop_unmatched=False)
        # transit_rail columns still added, but always zero.
        self.assertEqual(out["transit_rail"].sum(), 0)
        self.assertEqual(out["transit_rail_weight"].sum(), 0.0)

    def test_multi_match_within_category(self):
        """A row matching multiple (tag:value) pairs in one category gets
        count > 1 and summed weight."""
        # Make a single row that matches both shop:supermarket AND shop:bakery
        # (unusual but possible — large food hall etc.). Since we have one
        # `shop` column with a single value, simulate via a custom category.
        cm = {
            "groceries": [
                ("shop:supermarket", 1.0),
                ("amenity:supermarket", 0.5),  # second match condition
            ],
        }
        gdf = gpd.GeoDataFrame(
            {"shop": ["supermarket"], "amenity": ["supermarket"]},
            geometry=[Point(0, 0)],
            index=pd.Index(["p"], name="id"),
            crs="EPSG:4326",
        )
        out = categorize_pois(gdf, cm, drop_unmatched=False)
        self.assertEqual(out.loc["p", "groceries"], 2)
        self.assertEqual(out.loc["p", "groceries_weight"], 1.5)

    def test_custom_weight_suffix(self):
        out = categorize_pois(_toy_pois(), CATEGORY_MAP, weight_suffix="_w", drop_unmatched=False)
        self.assertIn("groceries_w", out.columns)
        self.assertNotIn("groceries_weight", out.columns)

    def test_column_name_collision_raises(self):
        """If a category name already exists as a column, raise instead of
        silently overwriting (would erase OSM tag data)."""
        toy = _toy_pois().assign(groceries="preexisting")
        with self.assertRaisesRegex(ValueError, "groceries"):
            categorize_pois(toy, CATEGORY_MAP)

    def test_input_not_mutated(self):
        toy = _toy_pois()
        cols_before = list(toy.columns)
        _ = categorize_pois(toy, CATEGORY_MAP)
        self.assertEqual(list(toy.columns), cols_before)

    def test_empty_category_map(self):
        """Empty category map = no new columns, no rows dropped."""
        toy = _toy_pois()
        out = categorize_pois(toy, {}, drop_unmatched=True)
        self.assertEqual(len(out), len(toy))
        for c in toy.columns:
            self.assertIn(c, out.columns)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
