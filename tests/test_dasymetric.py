"""Tests for the orchestrator-facing helpers in `aperta_atlas.dasymetric`.

Covered here:
  - `learn_coefficients`: single-column NNLS regression for per-tag
    per-m² intensities, with nearest-fallback rows for unmatched cells.
    Verifies basic fit, non-negativity, CI shape, contribution computation,
    fallback rescue, collinearity warning.

  - `combine_learned_per_category`: wide-form merge of per-category
    learned tables for the published CSV.

  - `merge_per_category_outputs`: merge per-category `per_building`
    outputs onto the original buildings index.

  - `per_building_from_coefficients`: apply learned per-tag intensities
    × area for the world-side path.

  - `apply_proportional_split`: per-cell ratio expansion of a coarse
    output into finer sub-categories.

  - `per_building`: in-coverage filter (overlap or fallback).
"""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from aperta_atlas.dasymetric import (
    apply_proportional_split,
    learn_coefficients,
    per_building,
    per_building_from_coefficients,
)
from preparation.switzerland.common import (
    combine_learned_per_category,
    merge_per_category_outputs,
)


def _square(x, y, side=10):
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def _make_buildings_in_grid(records, side=10, crs='EPSG:2056'):
    """`records` = list of dicts each with `building`, `x`, `y`, plus
    optional `area_m2` (defaults to side²) and value columns."""
    rows = []
    geoms = []
    for r in records:
        x, y = r.pop('x'), r.pop('y')
        if 'area_m2' not in r:
            r['area_m2'] = float(side * side)
        rows.append(r)
        geoms.append(_square(x, y, side))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def _make_cells(coords, side=100, crs='EPSG:2056'):
    polys = [_square(x, y, side) for x, y in coords]
    return gpd.GeoDataFrame(
        {'cell_id': np.arange(len(coords))},
        geometry=polys, crs=crs,
    )


class LearnCoefficientsTestCase(unittest.TestCase):
    """`learn_coefficients` runs single-column NNLS with overlay rows
    + nearest-fallback rows."""

    def test_recovers_known_intensities(self):
        # Ground truth: office 0.05, residential 0.01. NNLS should
        # recover these from cell-level totals.
        cells = _make_cells([(0, 0), (200, 0), (0, 200), (200, 200)])
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 10, 'y': 10},
            {'building': 'office', 'x': 210, 'y': 10},
            {'building': 'residential', 'x': 250, 'y': 10},
            {'building': 'residential', 'x': 10, 'y': 210},
            {'building': 'residential', 'x': 30, 'y': 210},
            {'building': 'residential', 'x': 50, 'y': 210},
            {'building': 'office', 'x': 210, 'y': 210},
            {'building': 'office', 'x': 230, 'y': 210},
            {'building': 'residential', 'x': 260, 'y': 210},
        ])
        cell_totals = pd.DataFrame(
            {'val': [5.0, 6.0, 3.0, 11.0]},
            index=pd.Index([0, 1, 2, 3], name='cell_id'),
        )
        learned = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office', 'residential'}),
            nearest_fallback_max_m=None,  # no fallback needed here
        )
        self.assertAlmostEqual(
            learned.loc['office', 'intensity_val'], 0.05, places=4)
        self.assertAlmostEqual(
            learned.loc['residential', 'intensity_val'], 0.01, places=4)
        # Contribution = β × Σ building area for tag.
        # office: 4 buildings × 100 m² = 400; 0.05 × 400 = 20.
        self.assertAlmostEqual(
            learned.loc['office', 'contribution_val'], 20.0, places=4)
        self.assertAlmostEqual(
            learned.loc['residential', 'contribution_val'], 5.0, places=4)

    def test_nonnegativity_constraint(self):
        # Set up data that would give a negative coef under OLS;
        # NNLS clips it to 0.
        cells = _make_cells([(0, 0), (200, 0)])
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 10, 'y': 10},
            {'building': 'shed', 'x': 40, 'y': 10},
            {'building': 'office', 'x': 210, 'y': 10},
        ])
        cell_totals = pd.DataFrame(
            {'val': [5.0, 6.0]}, index=pd.Index([0, 1], name='cell_id'),
        )
        learned = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office', 'shed'}),
            nearest_fallback_max_m=None,
        )
        self.assertGreaterEqual(learned.loc['shed', 'intensity_val'], 0.0)
        self.assertGreaterEqual(learned.loc['office', 'intensity_val'], 0.0)

    def test_p_value_in_unit_interval_and_small_for_strong_signal(self):
        # 5 cells with consistent β ≈ 0.05 (5 FTE on 100 m²) and very
        # little noise → strong signal → p-value near 0.
        cells = _make_cells([(i * 200, 0) for i in range(5)])
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': i * 200 + 10, 'y': 10}
            for i in range(5)
        ])
        cell_totals = pd.DataFrame(
            {'val': [5.0, 5.5, 4.8, 5.2, 4.9]},
            index=pd.Index(range(5), name='cell_id'),
        )
        learned = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office'}),
            nearest_fallback_max_m=None,
        )
        p = learned.loc['office', 'p_value_val']
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertLess(p, 0.01)   # strong signal → reject H0

    def test_collinearity_warning(self):
        # Two tags with proportional area patterns across cells →
        # collinear. Should log a WARNING.
        cells = _make_cells([(0, 0), (200, 0), (0, 200)])
        buildings = _make_buildings_in_grid([
            {'building': 'a', 'x': 10, 'y': 10},
            {'building': 'b', 'x': 40, 'y': 10},
            {'building': 'a', 'x': 210, 'y': 10},
            {'building': 'a', 'x': 230, 'y': 10},
            {'building': 'b', 'x': 260, 'y': 10},
            {'building': 'b', 'x': 280, 'y': 10},
            {'building': 'a', 'x': 10, 'y': 210},
            {'building': 'a', 'x': 30, 'y': 210},
            {'building': 'a', 'x': 50, 'y': 210},
            {'building': 'b', 'x': 70, 'y': 210},
            {'building': 'b', 'x': 90, 'y': 220},
            {'building': 'b', 'x': 10, 'y': 240},
        ])
        cell_totals = pd.DataFrame(
            {'val': [10.0, 22.0, 31.0]},
            index=pd.Index([0, 1, 2], name='cell_id'),
        )
        with self.assertLogs(level='WARNING') as cm:
            learn_coefficients(
                buildings, cells, cell_totals,
                cell_id_col='cell_id', column='val',
                relevant_tags=frozenset({'a', 'b'}),
                nearest_fallback_max_m=None,
            )
        self.assertIn('collinearity', '\n'.join(cm.output).lower())

    def test_output_schema(self):
        cells = _make_cells([(0, 0), (200, 0)])
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 10, 'y': 10},
            {'building': 'office', 'x': 210, 'y': 10},
        ])
        cell_totals = pd.DataFrame(
            {'val': [3.0, 3.0]}, index=pd.Index([0, 1], name='cell_id'),
        )
        learned = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office'}),
            nearest_fallback_max_m=None,
        )
        expected = [
            'intensity_val', 'p_value_val', 'contribution_val', 'n_observations',
        ]
        self.assertEqual(list(learned.columns), expected)
        self.assertEqual(learned.index.name, 'building')
        self.assertEqual(learned.loc['office', 'n_observations'], 2)

    def test_fallback_rescues_unmatched_cell(self):
        # Cell 0: contains an `office` building → overlay row.
        # Cell 1 (at (200,0)-(300,100)): NO `office` building inside, but
        # there's an `office` at (190, 50) — 10 m from cell 1's left edge,
        # < 100 m from centroid. With fallback enabled, NNLS should rescue
        # cell 1 via a synthetic row.
        cells = _make_cells([(0, 0), (200, 0)])
        # Office inside cell 0:
        b_in_cell_0 = Polygon([(20, 20), (40, 20), (40, 40), (20, 40)])
        # Office just outside cell 1 (to the left):
        b_outside_cell_1 = Polygon(
            [(185, 45), (195, 45), (195, 55), (185, 55)])
        buildings = gpd.GeoDataFrame(
            {'building': ['office', 'office'], 'area_m2': [400.0, 100.0]},
            geometry=[b_in_cell_0, b_outside_cell_1], crs='EPSG:2056',
        )
        cell_totals = pd.DataFrame(
            {'val': [20.0, 5.0]}, index=pd.Index([0, 1], name='cell_id'),
        )
        # WITHOUT fallback: only cell 0 has overlap; cell 1 is unmatched.
        # NNLS β (pre-rescale) = 20/400 = 0.05.
        # Bias rescale uses area_sum_per_tag (full area, both office
        # buildings) = 400 + 100 = 500. Target sum = 25 (cell 0 + cell 1).
        # Scale = 25 / (0.05 × 500) = 1.0 → β_eff = 0.05 unchanged.
        # Per the new contract, the input buildings ARE the calibration
        # area, so both offices count toward A_T regardless of overlap.
        learned_no_fb = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office'}),
            nearest_fallback_max_m=None,
        )
        beta_no_fb = learned_no_fb.loc['office', 'intensity_val']
        self.assertAlmostEqual(beta_no_fb, 0.05, places=4)

        # WITH fallback: cell 1 is rescued via the outside office.
        # NNLS β (pre-rescale): both rows agree on β = 0.05. Same
        # area_sum (500) and target sum (25) → scale 1.0 → β_eff = 0.05.
        learned_fb = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office'}),
            nearest_fallback_max_m=200.0,
        )
        beta_fb = learned_fb.loc['office', 'intensity_val']
        self.assertAlmostEqual(beta_fb, 0.05, places=4)

    def test_fallback_skipped_when_no_relevant_building_nearby(self):
        # Cell 0: contains an office. Cell 1 (at (10000, 10000)): no
        # office inside; nearest office is the one in cell 0, but it's
        # ~14000 m away — beyond max_m. Cell 1 is truly unreachable.
        cells = _make_cells([(0, 0), (10000, 10000)])
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 20, 'y': 20},
        ])
        cell_totals = pd.DataFrame(
            {'val': [10.0, 5.0]}, index=pd.Index([0, 1], name='cell_id'),
        )
        learned = learn_coefficients(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            relevant_tags=frozenset({'office'}),
            nearest_fallback_max_m=200.0,
        )
        # NNLS β (pre-rescale) = cell 0's 10/100 = 0.10.
        # Bias rescale: overlay_area=100, cal_target_sum=15 (incl. cell 1's
        # unreachable 5), scale = 15/10 = 1.5 → β_eff = 0.15. The unreachable
        # cell's target is implicitly absorbed into the β scaling — fine
        # in aggregate but a known limitation when only a single building
        # is in coverage.
        self.assertAlmostEqual(
            learned.loc['office', 'intensity_val'], 0.15, places=4)


class CombineLearnedPerCategoryTestCase(unittest.TestCase):
    def test_merges_per_column_tables(self):
        learned_a = pd.DataFrame({
            'intensity_a': [0.5, 0.3],
            'p_value_a': [0.001, 0.02],
            'contribution_a': [50.0, 30.0],
            'n_observations': [100, 50],
        }, index=pd.Index(['office', 'shop'], name='building'))
        learned_b = pd.DataFrame({
            'intensity_b': [0.7],
            'p_value_b': [0.005],
            'contribution_b': [70.0],
            'n_observations': [80],
        }, index=pd.Index(['farm'], name='building'))
        merged = combine_learned_per_category({'a': learned_a, 'b': learned_b})
        # All three tags present.
        self.assertEqual(sorted(merged.index), ['farm', 'office', 'shop'])
        # 'farm' missing in 'a' columns → NaN.
        self.assertTrue(np.isnan(merged.loc['farm', 'intensity_a']))
        # 'office' missing in 'b' columns → NaN.
        self.assertTrue(np.isnan(merged.loc['office', 'intensity_b']))
        # 'office' present in 'a': intensity preserved.
        self.assertAlmostEqual(merged.loc['office', 'intensity_a'], 0.5)
        # n_observations preserved.
        self.assertEqual(merged.loc['office', 'n_observations'], 100)
        self.assertEqual(merged.loc['farm', 'n_observations'], 80)


class MergePerCategoryOutputsTestCase(unittest.TestCase):
    def test_merges_per_category_per_building_outputs(self):
        buildings = gpd.GeoDataFrame(
            {'building': ['a', 'b', 'c']},
            geometry=[Point(0, 0).buffer(1), Point(10, 0).buffer(1),
                      Point(20, 0).buffer(1)],
            crs='EPSG:2056',
        )
        # Sector P: only buildings 'a' and 'b' in coverage.
        out_p = buildings.iloc[[0, 1]].copy()
        out_p['primary'] = [10.0, 20.0]
        # Sector Q: only buildings 'b' and 'c' in coverage.
        out_q = buildings.iloc[[1, 2]].copy()
        out_q['secondary'] = [5.0, 15.0]
        merged = merge_per_category_outputs(
            buildings, {'primary': out_p, 'secondary': out_q})
        # Union of in-coverage buildings = all three.
        self.assertEqual(len(merged), 3)
        # 'a' has primary but not secondary → secondary should be 0.
        self.assertAlmostEqual(merged.iloc[0]['primary'], 10.0)
        self.assertAlmostEqual(merged.iloc[0]['secondary'], 0.0)
        # 'c' has secondary but not primary → primary should be 0.
        self.assertAlmostEqual(merged.iloc[2]['primary'], 0.0)
        self.assertAlmostEqual(merged.iloc[2]['secondary'], 15.0)


class PerBuildingFromCoefficientsTestCase(unittest.TestCase):
    def _calibrated(self, intensity_per_tag, col='val'):
        return pd.DataFrame(
            {f'intensity_{col}': list(intensity_per_tag.values())},
            index=pd.Index(list(intensity_per_tag.keys()), name='building'),
        )

    def test_basic_application(self):
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 0, 'y': 0, 'area_m2': 100.0},
            {'building': 'shop', 'x': 100, 'y': 0, 'area_m2': 200.0},
        ])
        cal = self._calibrated({'office': 0.5, 'shop': 0.2})
        out = per_building_from_coefficients(
            buildings, columns=('val',), calibrated_coeffs=cal,
        )
        self.assertAlmostEqual(out['val'].iloc[0], 50.0)
        self.assertAlmostEqual(out['val'].iloc[1], 40.0)

    def test_unknown_tag_filtered_out(self):
        buildings = _make_buildings_in_grid([
            {'building': 'mystery', 'x': 0, 'y': 0},
        ])
        cal = self._calibrated({'office': 0.5})
        out = per_building_from_coefficients(
            buildings, columns=('val',), calibrated_coeffs=cal,
        )
        self.assertEqual(len(out), 0)

    def test_nan_intensity_yields_zero(self):
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 0, 'y': 0, 'area_m2': 100.0},
        ])
        cal = pd.DataFrame({
            'intensity_a': [0.5],
            'intensity_b': [np.nan],
        }, index=pd.Index(['office'], name='building'))
        out = per_building_from_coefficients(
            buildings, columns=('a', 'b'), calibrated_coeffs=cal,
        )
        self.assertAlmostEqual(out['a'].iloc[0], 50.0)
        self.assertAlmostEqual(out['b'].iloc[0], 0.0)

    def test_missing_calibrated_raises(self):
        buildings = _make_buildings_in_grid([
            {'building': 'office', 'x': 0, 'y': 0},
        ])
        with self.assertRaises(ValueError):
            per_building_from_coefficients(
                buildings, columns=('val',), calibrated_coeffs=None,  # type: ignore
            )

    def test_precision_rounds_default_to_two_decimals(self):
        # Small β × area should round to 0 with default precision=2.
        # Larger should keep 2-decimal precision.
        buildings = _make_buildings_in_grid([
            {'building': 'a', 'x': 0,   'y': 0, 'area_m2': 100.0},   # 100 m²
            {'building': 'b', 'x': 100, 'y': 0, 'area_m2': 100.0},
            {'building': 'c', 'x': 200, 'y': 0, 'area_m2': 100.0},
        ])
        cal = self._calibrated({
            'a': 0.00004,   # × 100 = 0.004 → rounds to 0.00
            'b': 0.00007,   # × 100 = 0.007 → rounds to 0.01
            'c': 0.12349,   # × 100 = 12.349 → rounds to 12.35
        })
        out = per_building_from_coefficients(
            buildings, columns=('val',), calibrated_coeffs=cal,
        )
        self.assertAlmostEqual(out['val'].iloc[0], 0.0, places=4)
        self.assertAlmostEqual(out['val'].iloc[1], 0.01, places=4)
        self.assertAlmostEqual(out['val'].iloc[2], 12.35, places=4)

    def test_precision_none_returns_raw_floats(self):
        buildings = _make_buildings_in_grid([
            {'building': 'a', 'x': 0, 'y': 0, 'area_m2': 100.0},
        ])
        cal = self._calibrated({'a': 0.00007})
        out = per_building_from_coefficients(
            buildings, columns=('val',), calibrated_coeffs=cal,
            precision=None,
        )
        self.assertAlmostEqual(out['val'].iloc[0], 0.007, places=6)


class ApplyProportionalSplitTestCase(unittest.TestCase):
    def _two_cells_with_buildings(self):
        cell0 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        cell1 = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
        cells = gpd.GeoDataFrame(
            {'cell_id': [0, 1]},
            geometry=[cell0, cell1], crs='EPSG:2056',
        )
        buildings = gpd.GeoDataFrame(
            {'building': ['office', 'house'], 'parent': [100.0, 50.0]},
            geometry=[Point(50, 50).buffer(5), Point(150, 50).buffer(5)],
            crs='EPSG:2056',
        )
        return buildings, cells

    def test_single_parent_split(self):
        buildings, cells = self._two_cells_with_buildings()
        cell_totals = pd.DataFrame({
            'total': [100.0, 100.0],
            'band_a': [70.0, 20.0],
            'band_b': [30.0, 80.0],
        }, index=pd.Index([0, 1], name='cell_id'))
        buildings = buildings.rename(columns={'parent': 'total'})
        out = apply_proportional_split(
            buildings, cells, cell_totals,
            cell_id_col='cell_id',
            splits={'band_a': 'total', 'band_b': 'total'},
        )
        self.assertAlmostEqual(out['band_a'].iloc[0], 70.0)
        self.assertAlmostEqual(out['band_b'].iloc[0], 30.0)
        self.assertAlmostEqual(out['band_a'].iloc[1], 10.0)
        self.assertAlmostEqual(out['band_b'].iloc[1], 40.0)

    def test_non_sequential_building_index(self):
        buildings, cells = self._two_cells_with_buildings()
        buildings = buildings.rename(columns={'parent': 'total'})
        buildings.index = pd.Index([100_001, 200_002], name='building_id')
        cell_totals = pd.DataFrame({
            'total': [50.0, 50.0],
            'band_a': [40.0, 10.0],
        }, index=pd.Index([0, 1], name='cell_id'))
        out = apply_proportional_split(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', splits={'band_a': 'total'},
        )
        self.assertAlmostEqual(out['band_a'].iloc[0], 80.0)
        self.assertAlmostEqual(out['band_a'].iloc[1], 10.0)
        self.assertEqual(out.index.tolist(), [100_001, 200_002])


class PerBuildingCoverageFilterTestCase(unittest.TestCase):
    def test_out_of_coverage_buildings_dropped(self):
        cell = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        cells = gpd.GeoDataFrame(
            {'cell_id': [0]}, geometry=[cell], crs='EPSG:2056',
        )
        cell_totals = pd.DataFrame(
            {'val': [50.0]}, index=pd.Index([0], name='cell_id'),
        )
        b_a = Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
        b_b = Polygon([(1000, 1000), (1010, 1000), (1010, 1010), (1000, 1010)])
        b_c = Polygon([(2000, 2000), (2010, 2000), (2010, 2010), (2000, 2010)])
        buildings = gpd.GeoDataFrame(
            {'building': ['office', 'office', 'office']},
            geometry=[b_a, b_b, b_c], crs='EPSG:2056',
        )
        out = per_building(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            coeffs={'office': 1.0},
            nearest_fallback_max_m=10,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out['val'].iloc[0], 50.0)

    def test_nearest_fallback_brings_into_coverage(self):
        cell = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        cells = gpd.GeoDataFrame(
            {'cell_id': [0]}, geometry=[cell], crs='EPSG:2056',
        )
        cell_totals = pd.DataFrame(
            {'val': [30.0]}, index=pd.Index([0], name='cell_id'),
        )
        b_far = Polygon([(150, 40), (160, 40), (160, 60), (150, 60)])
        buildings = gpd.GeoDataFrame(
            {'building': ['office']},
            geometry=[b_far], crs='EPSG:2056',
        )
        out = per_building(
            buildings, cells, cell_totals,
            cell_id_col='cell_id', column='val',
            coeffs={'office': 1.0},
            nearest_fallback_max_m=200,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out['val'].iloc[0], 30.0)


if __name__ == '__main__':
    unittest.main()
