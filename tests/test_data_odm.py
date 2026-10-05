"""Tests for `Context.create_tiered_odm` / `Context.get_tiered_odm` and
`aperta.network_processing.verify_odm_against_network`.

Run with:
    cd src && python -m unittest tests.test_data_odm

A tiered ODM is tied to a specific network (`network_name`) and to what its
values mean (`data_name`). Files land at `odm/<network_name>_<data_name>.npz`
— ONE file per (network, data_name), holding every populated tier under a
`<tier>__keys`/`<tier>__offsets`/`<tier>__values` prefix, plus a top-level
`__tier_names__` manifest. Variants ('default' / 'detoured' / etc.) are
encoded directly in `data_name` if needed (e.g. `'travel_time_detoured'`).
"""
import os
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aperta.errors import DataError
from aperta.network_processing import verify_odm_against_network
from aperta.od_pairs import TieredODNodePairs
from aperta_atlas.context import Context, Storage, cache


def _make_prep_context(working_dir: str, data_dir: str) -> Context:
    """Build a preparation-namespace Context for tests; doesn't touch the filesystem
    until create_/get_ are called.

    `default_storage_override=Storage.PRIVATE` mirrors what `init_context`
    would set at runtime by reading the `STORAGE` constant from
    `preparation/switzerland/private/historic/__init__.py`. Direct Context
    construction bypasses that, so the test wires it explicitly."""
    return Context(
        parent=None, variant=None, project=None, scenario=None,
        namespace='preparation/switzerland/historic',
        caller_root_path='', caller_file_path='x.py', caller_base_name='x.py',
        data_dir_public=data_dir, data_dir_private=data_dir + '_prot',
        working_dir=working_dir, read_only=False, track_dependencies=False,
        start_time=0.0, env={},
        created_data=set(), used_data=set(),
        default_storage_override=Storage.PRIVATE,
    )


def _network_nodes(node_ids):
    """Synthetic nodes GeoDataFrame indexed by `node_ids` (Point geometries at 0,0)."""
    return gpd.GeoDataFrame(
        {'geometry': [Point(0, 0) for _ in node_ids]},
        index=pd.Index(list(node_ids), name='node_id'),
        crs='EPSG:2056',
    )


class CreateGetTieredOdmTestCase(unittest.TestCase):
    """Round-trip + file-naming tests for the tiered ODM API."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.working_dir = self._tmp.name
        os.makedirs(f'{self.working_dir}/config', exist_ok=True)
        with open(f'{self.working_dir}/config/general.yml', 'w') as f:
            f.write("---\n")
        self.data_dir = f'{self.working_dir}/D'
        self.ctx = _make_prep_context(self.working_dir, self.data_dir)
        cache.clear()

    def tearDown(self):
        self._tmp.cleanup()
        cache.clear()

    # -------- file naming --------

    def test_filename_uses_network_data_naming(self):
        pairs = TieredODNodePairs(
            cells_to_cells={'N0': np.array([10, 20], dtype=np.int32)},
        )
        self.ctx.create_tiered_odm(pairs, network_name='nw_2020', data_name='dist_line')
        # Test ctx has namespace 'preparation/switzerland/historic', whose STORAGE
        # constant is PRIVATE → outputs land under data_dir_private (data_dir + '_prot').
        expected = (
            f'{self.data_dir}_prot/preparation/switzerland/historic/odm/'
            f'nw_2020_dist_line.npz'
        )
        self.assertTrue(os.path.exists(expected), f'missing: {expected}')

    # -------- round-trips, single tier --------

    def test_round_trip_int32_values_cells_only(self):
        pairs = TieredODNodePairs(
            cells_to_cells={'N0': np.array([10, 20, 30], dtype=np.int32),
                            'N1': np.array([40, 50], dtype=np.int32)},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'dist_line')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'dist_line')
        self.assertIsNone(loaded.cells_to_zones)
        self.assertIsNone(loaded.zones_to_zones)
        self.assertEqual(set(loaded.cells_to_cells), {'N0', 'N1'})
        np.testing.assert_array_equal(loaded.cells_to_cells['N0'], [10, 20, 30])
        self.assertEqual(loaded.cells_to_cells['N0'].dtype, np.int32)

    def test_round_trip_float32_values(self):
        pairs = TieredODNodePairs(
            cells_to_cells={'N0': np.array([1.5, 2.5], dtype=np.float32)},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'travel_time_car')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'travel_time_car')
        self.assertEqual(loaded.cells_to_cells['N0'].dtype, np.float32)
        np.testing.assert_array_almost_equal(loaded.cells_to_cells['N0'], [1.5, 2.5])

    def test_round_trip_int_origin_keys(self):
        """OSM-style int origin keys must round-trip as ints (not silently coerced
        to strings by the underlying np.savez_compressed string-keyword requirement).
        """
        pairs = TieredODNodePairs(
            cells_to_cells={12345: np.array([1.0, 2.0], dtype=np.float32),
                            67890: np.array([3.0], dtype=np.float32)},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'dist_line')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'dist_line')
        self.assertEqual(set(loaded.cells_to_cells), {12345, 67890})
        self.assertIsInstance(next(iter(loaded.cells_to_cells)), int)

    def test_round_trip_idx_list_str_destinations(self):
        """`data_name='idx'` round-trips list[str] (cast to U-dtype, restored as list)."""
        pairs = TieredODNodePairs(
            cells_to_cells={'N0': ['N1', 'N2', 'N3'], 'N1': ['N0', 'N2']},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'idx')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'idx')
        self.assertIsInstance(loaded.cells_to_cells['N0'], list)
        self.assertEqual(loaded.cells_to_cells['N0'], ['N1', 'N2', 'N3'])

    # -------- round-trips, multiple tiers --------

    def test_round_trip_all_three_tiers(self):
        pairs = TieredODNodePairs(
            cells_to_cells={'C0': np.array([1.0, 2.0], dtype=np.float32),
                            'C1': np.array([3.0], dtype=np.float32)},
            cells_to_zones={'C0': np.array([100.0], dtype=np.float32)},
            zones_to_zones={'Z0': np.array([1000.0, 2000.0], dtype=np.float32)},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'travel_time')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'travel_time')
        np.testing.assert_array_almost_equal(loaded.cells_to_cells['C0'], [1.0, 2.0])
        np.testing.assert_array_almost_equal(loaded.cells_to_zones['C0'], [100.0])
        np.testing.assert_array_almost_equal(loaded.zones_to_zones['Z0'], [1000.0, 2000.0])

    def test_round_trip_two_tiers_missing_middle(self):
        """cells_to_zones absent → comes back None; cells_to_cells + zones_to_zones populate."""
        pairs = TieredODNodePairs(
            cells_to_cells={'C0': np.array([1.0], dtype=np.float32)},
            zones_to_zones={'Z0': np.array([10.0], dtype=np.float32)},
        )
        self.ctx.create_tiered_odm(pairs, 'nw_2020', 'travel_time')
        cache.clear()
        loaded = self.ctx.get_tiered_odm('nw_2020', 'travel_time')
        self.assertIsNotNone(loaded.cells_to_cells)
        self.assertIsNone(loaded.cells_to_zones)
        self.assertIsNotNone(loaded.zones_to_zones)

    # -------- validation --------

    def test_write_with_no_populated_tiers_raises(self):
        pairs = TieredODNodePairs()  # all tiers None
        with self.assertRaises(DataError):
            self.ctx.create_tiered_odm(pairs, 'nw_2020', 'travel_time')


class VerifyOdmAgainstNetworkTestCase(unittest.TestCase):
    """Tests for the standalone `verify_odm_against_network` checker (no I/O)."""

    def test_passes_when_origins_in_network(self):
        nodes = _network_nodes(['N0', 'N1', 'N2'])
        odm = {'N0': np.array([1, 2], dtype=np.int32),
               'N1': np.array([3], dtype=np.int32)}
        verify_odm_against_network(odm, nodes)  # should not raise

    def test_raises_on_unknown_origin(self):
        nodes = _network_nodes(['N0', 'N1'])
        odm = {'N0': np.array([1], dtype=np.int32),
               'N99': np.array([2], dtype=np.int32)}
        with self.assertRaises(DataError) as ctx:
            verify_odm_against_network(odm, nodes)
        self.assertIn('N99', str(ctx.exception))

    def test_raises_on_unknown_destination_when_check_destinations(self):
        nodes = _network_nodes(['N0', 'N1'])
        odm = {'N0': ['N1', 'N99']}
        with self.assertRaises(DataError) as ctx:
            verify_odm_against_network(odm, nodes, check_destinations=True)
        self.assertIn('N99', str(ctx.exception))

    def test_destinations_not_validated_by_default(self):
        """`check_destinations=False` (default) skips destination validation — useful
        for value-style ODMs where the values aren't node IDs.
        """
        nodes = _network_nodes(['N0'])
        odm = {'N0': np.array([1.5, 2.5, 3.5], dtype=np.float32)}
        verify_odm_against_network(odm, nodes)  # should not raise


if __name__ == '__main__':
    unittest.main()
