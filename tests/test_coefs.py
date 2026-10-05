"""Tests for `aperta_atlas.coefs` — the three CoefSource kinds and the
`resolve` dispatcher.

Run with:
    cd src && python -m unittest tests.test_coefs

These tests stub a minimal `scenarios` module in a temp dir + put it on
sys.path so `coefs._get_source` resolves the way real callers do. Each
test cleans the stub up before exiting.
"""
import importlib
import os
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import pandas as pd

from aperta_atlas import coefs
from aperta_atlas.context import Context, Storage


# --- helpers -----------------------------------------------------------------


def _make_ctx(working_dir: str, data_dir: str, scenario: str) -> Context:
    """Project Context for tests; no filesystem touched until create_/get_."""
    return Context(
        parent=None, variant=None, project='atlas', scenario=scenario,
        namespace=f'atlas/main', caller_root_path='', caller_file_path='x.py',
        caller_base_name='x.py',
        data_dir_public=data_dir, data_dir_private=data_dir + '_priv',
        working_dir=working_dir, read_only=False, track_dependencies=False,
        start_time=0.0, env={},
        created_data=set(), used_data=set(),
        default_storage_override=Storage.PUBLIC,
    )


def _stub_scenarios_module(tmp_dir: str, body: str) -> None:
    """Write a stub `scenarios.py` into `tmp_dir` and put it first on sys.path
    so `importlib.import_module('scenarios')` picks it up."""
    with open(os.path.join(tmp_dir, 'scenarios.py'), 'w') as f:
        f.write(textwrap.dedent(body))
    sys.path.insert(0, tmp_dir)
    # Drop any prior `scenarios` from the import cache.
    sys.modules.pop('scenarios', None)


def _unstub_scenarios_module(tmp_dir: str) -> None:
    if tmp_dir in sys.path:
        sys.path.remove(tmp_dir)
    sys.modules.pop('scenarios', None)


_STUB_BASE = """
    from dataclasses import dataclass, field
    from aperta_atlas.coefs import Calibrate, CoefSource, HandWritten, ImportFrom

    @dataclass(frozen=True)
    class Scenario:
        name: str
        coefs: dict = field(default_factory=dict)

    SCENARIOS = {{
        'src-cal': Scenario(name='src-cal', coefs={{
            'my_coef': Calibrate(),
        }}),
        'target': Scenario(name='target', coefs={{
            'my_coef': {target_source},
        }}),
    }}
"""


# --- Calibrate path ---------------------------------------------------------


class CalibrateTestCase(unittest.TestCase):
    """`Calibrate()` source: runs `calibrate_fn`, writes to calibrated/."""

    def setUp(self):
        self._tmp_data = tempfile.TemporaryDirectory()
        self._tmp_src = tempfile.TemporaryDirectory()
        _stub_scenarios_module(
            self._tmp_src.name,
            _STUB_BASE.format(target_source='Calibrate()'),
        )
        self.ctx = _make_ctx(self._tmp_data.name, self._tmp_data.name, 'target')

    def tearDown(self):
        _unstub_scenarios_module(self._tmp_src.name)
        self._tmp_data.cleanup()
        self._tmp_src.cleanup()

    def test_calibrate_writes_to_calibrated_subfolder(self):
        df = pd.DataFrame({'profile_a': [1.0, 2.0]},
                          index=pd.Index(['baseline_time', 'feature_x'], name='param'))
        result = coefs.resolve(
            self.ctx, name='my_coef', calibrate_fn=lambda: df)
        pd.testing.assert_frame_equal(result, df)

        out_path = os.path.join(
            self._tmp_data.name, 'atlas', 'target', 'coefs', 'calibrated',
            'my_coef.csv')
        self.assertTrue(os.path.exists(out_path), f'missing: {out_path}')

    def test_get_coefs_reads_back_calibrated(self):
        df = pd.DataFrame({'profile_a': [1.5, 2.5]},
                          index=pd.Index(['baseline_time', 'feature_x'], name='param'))
        coefs.resolve(self.ctx, name='my_coef', calibrate_fn=lambda: df)
        loaded = self.ctx.get_coefs('my_coef')
        pd.testing.assert_frame_equal(loaded, df)


# --- HandWritten path ------------------------------------------------------


class HandWrittenTestCase(unittest.TestCase):
    """`HandWritten()` source: user places file at manual/, resolve verifies."""

    def setUp(self):
        self._tmp_data = tempfile.TemporaryDirectory()
        self._tmp_src = tempfile.TemporaryDirectory()
        _stub_scenarios_module(
            self._tmp_src.name,
            _STUB_BASE.format(target_source='HandWritten()'),
        )
        self.ctx = _make_ctx(self._tmp_data.name, self._tmp_data.name, 'target')

    def tearDown(self):
        _unstub_scenarios_module(self._tmp_src.name)
        self._tmp_data.cleanup()
        self._tmp_src.cleanup()

    def test_handwritten_reads_existing_file(self):
        manual_dir = os.path.join(
            self._tmp_data.name, 'atlas', 'target', 'coefs', 'manual')
        os.makedirs(manual_dir)
        df = pd.DataFrame({'transit': [0.5, 0.5, 300.0]},
                          index=pd.Index(['walk_coef', 'bike_coef', 'cap_s'],
                                         name='param'))
        df.to_csv(os.path.join(manual_dir, 'my_coef.csv'))

        # resolve doesn't call calibrate_fn for HandWritten; mark with a
        # callable that would fail if invoked.
        result = coefs.resolve(
            self.ctx, name='my_coef',
            calibrate_fn=lambda: (_ for _ in ()).throw(AssertionError('should not be called')))
        pd.testing.assert_frame_equal(result, df)

    def test_handwritten_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            coefs.resolve(self.ctx, name='my_coef',
                          calibrate_fn=lambda: pd.DataFrame())


# --- ImportFrom path -------------------------------------------------------


class ImportFromTestCase(unittest.TestCase):
    """`ImportFrom(other)` source: reads other scenario's coefs, writes to
    this scenario's transferred/ with provenance sidecar."""

    def setUp(self):
        self._tmp_data = tempfile.TemporaryDirectory()
        self._tmp_src = tempfile.TemporaryDirectory()
        _stub_scenarios_module(
            self._tmp_src.name,
            _STUB_BASE.format(target_source="ImportFrom('src-cal')"),
        )

    def tearDown(self):
        _unstub_scenarios_module(self._tmp_src.name)
        self._tmp_data.cleanup()
        self._tmp_src.cleanup()

    def test_importfrom_copies_from_source_and_writes_transferred(self):
        # First populate src-cal's calibrated coef.
        src_ctx = _make_ctx(self._tmp_data.name, self._tmp_data.name, 'src-cal')
        df = pd.DataFrame({'profile_a': [10.0, 20.0]},
                          index=pd.Index(['baseline_time', 'feature_x'], name='param'))
        coefs.resolve(src_ctx, name='my_coef', calibrate_fn=lambda: df)

        # Now resolve from target — should copy to transferred/.
        target_ctx = _make_ctx(self._tmp_data.name, self._tmp_data.name, 'target')
        result = coefs.resolve(
            target_ctx, name='my_coef',
            calibrate_fn=lambda: (_ for _ in ()).throw(AssertionError('should not be called')))
        pd.testing.assert_frame_equal(result, df)

        out_path = os.path.join(
            self._tmp_data.name, 'atlas', 'target', 'coefs', 'transferred',
            'my_coef.csv')
        self.assertTrue(os.path.exists(out_path), f'missing: {out_path}')
        # Provenance = the `transferred/` subfolder itself; no sidecar.


# --- CSV header sniffing ---------------------------------------------------


class MultiHeaderCsvTestCase(unittest.TestCase):
    """`_read_coefs_csv` auto-detects single- vs 2-row header layouts."""

    def setUp(self):
        self._tmp_data = tempfile.TemporaryDirectory()
        self._tmp_src = tempfile.TemporaryDirectory()
        _stub_scenarios_module(
            self._tmp_src.name,
            _STUB_BASE.format(target_source='Calibrate()'),
        )
        self.ctx = _make_ctx(self._tmp_data.name, self._tmp_data.name, 'target')

    def tearDown(self):
        _unstub_scenarios_module(self._tmp_src.name)
        self._tmp_data.cleanup()
        self._tmp_src.cleanup()

    def test_round_trip_multi_index_columns(self):
        # Bike's calibrated frame uses a (profile, coef|p) MultiIndex.
        cols = pd.MultiIndex.from_tuples(
            [('rbike', 'coef'), ('rbike', 'p'),
             ('ebike25', 'coef'), ('ebike25', 'p')])
        df = pd.DataFrame(
            [[0.5, 0.01, 0.33, 0.02], [1.5, 0.10, 0.50, 0.20]],
            index=pd.Index(['baseline_time', 'feature_x'], name='param'),
            columns=cols,
        )
        coefs.resolve(self.ctx, name='my_coef', calibrate_fn=lambda: df)
        loaded = self.ctx.get_coefs('my_coef')
        pd.testing.assert_frame_equal(loaded, df)


# --- Namespace-scoped coefs (preparation) ----------------------------------


def _make_namespace_ctx(working_dir: str, data_dir: str,
                        namespace: str = 'preparation/world/land_use') -> Context:
    """Preparation-namespace Context for tests; project + scenario both None
    (the precondition that gates the namespace-scoped coef path)."""
    return Context(
        parent=None, variant=None, project=None, scenario=None,
        namespace=namespace, caller_root_path='', caller_file_path='x.py',
        caller_base_name='x.py',
        data_dir_public=data_dir, data_dir_private=data_dir + '_priv',
        working_dir=working_dir, read_only=False, track_dependencies=False,
        start_time=0.0, env={},
        created_data=set(), used_data=set(),
        default_storage_override=Storage.PUBLIC,
    )


class NamespaceCoefsTestCase(unittest.TestCase):
    """Preparation-namespace coefs: flat layout, no dispatcher, sidecar still
    written. No `scenarios.py` stub needed — the namespace path skips dispatch."""

    def setUp(self):
        self._tmp_data = tempfile.TemporaryDirectory()
        self.ctx = _make_namespace_ctx(self._tmp_data.name, self._tmp_data.name)

    def tearDown(self):
        self._tmp_data.cleanup()

    def test_create_coefs_writes_to_flat_path(self):
        df = pd.DataFrame({'intensity': [0.5, 1.2, 0.0]},
                          index=pd.Index(['tag_a', 'tag_b', 'tag_c'], name='tag'))
        self.ctx.create_coefs(df, 'my_intensities')

        out_path = os.path.join(
            self._tmp_data.name, 'preparation', 'world', 'land_use',
            'coefs', 'my_intensities.csv')
        self.assertTrue(os.path.exists(out_path), f'missing: {out_path}')

        # No <kind>/ subfolder for namespace coefs.
        bad_path = os.path.join(
            self._tmp_data.name, 'preparation', 'world', 'land_use',
            'coefs', 'calibrated', 'my_intensities.csv')
        self.assertFalse(os.path.exists(bad_path),
                         f'unexpected <kind>/ subfolder: {bad_path}')

    def test_get_coefs_reads_back_from_flat_path(self):
        df = pd.DataFrame({'intensity': [0.5, 1.2]},
                          index=pd.Index(['tag_a', 'tag_b'], name='tag'))
        self.ctx.create_coefs(df, 'my_intensities')
        loaded = self.ctx.get_coefs('my_intensities')
        pd.testing.assert_frame_equal(loaded, df)

    def test_get_coefs_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.ctx.get_coefs('not_a_coef')

    def test_kind_argument_rejected_in_namespace(self):
        """`kind=` is meaningless for preparation — no dispatcher to dispatch."""
        df = pd.DataFrame({'x': [1.0]}, index=pd.Index(['a'], name='tag'))
        with self.assertRaisesRegex(Exception, 'project-only'):
            self.ctx.create_coefs(df, 'my_intensities', kind='transferred')

    def test_cross_namespace_read_via_source(self):
        """Common pattern: switzerland writer + world reader, both namespaces.
        `context.source('<other namespace>').get_coefs(name)` resolves.

        Mirrors the live `employment_per_building_from_coef` →
        `employment_statent` consumer pattern.
        """
        writer = _make_namespace_ctx(
            self._tmp_data.name, self._tmp_data.name,
            namespace='preparation/switzerland/land_use')
        df = pd.DataFrame({'intensity': [0.5, 1.2]},
                          index=pd.Index(['tag_a', 'tag_b'], name='tag'))
        writer.create_coefs(df, 'employment_statent_2022')

        reader = _make_namespace_ctx(
            self._tmp_data.name, self._tmp_data.name,
            namespace='preparation/world/land_use')
        # `source(storage=)` keeps both contexts on the same root so the
        # cross-namespace read finds the file regardless of which storage
        # tier the Swiss namespace defaults to. Real callers don't need
        # this — production storage tier is namespace-stable.
        src = reader.source('preparation/switzerland/land_use', storage=Storage.PUBLIC)
        loaded = src.get_coefs('employment_statent_2022')
        pd.testing.assert_frame_equal(loaded, df)


if __name__ == '__main__':
    unittest.main()
