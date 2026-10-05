"""Tests for `aperta_atlas.pipeline` — focused on the ${area} substitution
the preparation/world pipeline relies on. Existing scenario / --from / --only
plumbing isn't covered here (older, exercised by the live switzerland prep
pipelines).

Run with:
    cd src && python -m unittest tests.test_pipeline
"""
import unittest

from aperta.errors import ProcessingError

from aperta_atlas.pipeline import (
    Pipeline,
    Stage,
    _resolve_variants,
    _stage_invocations,
    run_pipeline,
)


def _stage(name: str, variants=()) -> Stage:
    return Stage(name=name, module=f'pkg.{name}', variants=list(variants))


class ResolveVariantsTestCase(unittest.TestCase):
    """`_resolve_variants` is the single substitution point."""

    def test_no_placeholder_no_area_passthrough(self):
        s = _stage('a', variants=['bern_walk', 'bern_bike'])
        self.assertEqual(_resolve_variants(s, area=None),
                         ['bern_walk', 'bern_bike'])

    def test_no_placeholder_with_area_passthrough(self):
        """`--area` passed but stage has no placeholder → variants unchanged.
        The pipeline-level guard rejects the run; this helper only resolves."""
        s = _stage('a', variants=['bern_walk'])
        self.assertEqual(_resolve_variants(s, area='zurich'), ['bern_walk'])

    def test_placeholder_substituted_when_area_given(self):
        s = _stage('a', variants=['${area}_walk', '${area}_car'])
        self.assertEqual(_resolve_variants(s, area='bern'),
                         ['bern_walk', 'bern_car'])

    def test_placeholder_without_area_raises(self):
        s = _stage('a', variants=['${area}_walk'])
        with self.assertRaisesRegex(ProcessingError, r'\$\{area\}'):
            _resolve_variants(s, area=None)

    def test_empty_variants_no_substitution(self):
        s = _stage('a', variants=[])
        self.assertEqual(_resolve_variants(s, area='bern'), [])


class StageInvocationsTestCase(unittest.TestCase):
    """`_stage_invocations` builds the argv list per stage; verify --area flows
    through into the resolved `--variant` arguments."""

    def test_area_substitutes_into_variant_args(self):
        s = _stage('a', variants=['${area}_walk', '${area}_car'])
        argvs = _stage_invocations(s, python='/py', scenario=None, area='bern')
        self.assertEqual(argvs, [
            ['/py', '-m', 'pkg.a', '--variant', 'bern_walk'],
            ['/py', '-m', 'pkg.a', '--variant', 'bern_car'],
        ])

    def test_scenario_and_area_compose(self):
        s = _stage('a', variants=['${area}_walk'])
        argvs = _stage_invocations(s, python='/py', scenario='target', area='bern')
        self.assertEqual(argvs, [
            ['/py', '-m', 'pkg.a', '--variant', 'bern_walk',
             '--scenario', 'target'],
        ])


class PipelineRunGuardsTestCase(unittest.TestCase):
    """`run_pipeline` rejects mismatched --area usage before any subprocess fires."""

    def _make(self, stages: list[Stage]) -> Pipeline:
        return Pipeline(name='t', stages=stages,
                        working_dir='.', python='/py', yaml_path=None)

    def test_area_passed_but_no_stage_uses_it(self):
        p = self._make([_stage('a', variants=['bern_walk'])])
        with self.assertRaisesRegex(ProcessingError, 'no selected stage uses'):
            run_pipeline(p, area='zurich')

    def test_area_required_but_missing(self):
        """A stage with ${area} but no --area raises at first invocation."""
        p = self._make([_stage('a', variants=['${area}_walk'])])
        with self.assertRaisesRegex(ProcessingError, r'\$\{area\}'):
            run_pipeline(p)


if __name__ == '__main__':
    unittest.main()
