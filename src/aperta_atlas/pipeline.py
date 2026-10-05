"""
Lightweight YAML-driven pipeline runner.

Reads a pipeline.yml that lists stages (scripts to run, in order, optionally
parameterized by `Variants` names) and invokes each as a subprocess. Sequential
and intentionally dumb: no caching, no parallelism inference, no stale-detection.
Partial runs via --from / --only / --skip. For parallelism, run multiple
invocations from bash with `&`.

YAML schema:

    name: <pipeline name>                      # optional; defaults to filename stem
    description: <text>                        # optional
    working_dir: <path>                        # optional; relative to pipeline.yml's
                                               # parent directory; default '.'.
                                               # Used as cwd for stage subprocesses.
    python: <executable>                       # optional; default = sys.executable
    stages:
      - name: <stage name>                     # optional; auto-derived if absent
        script: <path>                         # path relative to working_dir, OR
        module: <a.b.c>                        # python module (runs via `python -m`)
        variants: [<variant name>, ...]        # optional; one subprocess per variant,
                                               # passed as `--variant <name>`
        cwd: <path>                            # optional per-stage cwd override
                                               # (relative to pipeline.yml's parent)

`--scenario <name>` is forwarded to every stage subprocess. Preparation-namespace
scripts are scenario-free and will reject it via `init_context`, so a pipeline
that mixes preparation and project stages should omit `--scenario` and let each
project stage pick up its `DEFAULT_SCENARIO` from `src/scenarios.py`.

`--area <name>` substitutes `${area}` placeholders in every stage's `variants`
list before invocation. Lets a single preparation pipeline run for any area
defined in `areas.py` — the scripts already follow a `<area_name>_<suffix>`
variant convention, so `variants: ['${area}_walk']` resolves to the actual
variant name at runtime. The runner errors out if any stage uses `${area}` but
`--area` wasn't passed (or vice versa).

CLI:
    python -m aperta_atlas.pipeline run <pipeline.yml>
    python -m aperta_atlas.pipeline run <pipeline.yml> --scenario <name>
    python -m aperta_atlas.pipeline run <pipeline.yml> --area <name>
    python -m aperta_atlas.pipeline run <pipeline.yml> --from <stage>
    python -m aperta_atlas.pipeline run <pipeline.yml> --only <stage>
    python -m aperta_atlas.pipeline run <pipeline.yml> --skip <stage> [--skip <stage> ...]
    python -m aperta_atlas.pipeline list <pipeline.yml>
"""

import sys
import yaml
import subprocess
import argparse
import logging

from pathlib import Path
from dataclasses import dataclass, field

from aperta.errors import ProcessingError


@dataclass
class Stage:
    name: str
    script: str | None = None
    module: str | None = None
    variants: list[str] = field(default_factory=list)
    cwd: str | None = None

    def __post_init__(self):
        if self.script and self.module:
            raise ProcessingError(f"Stage '{self.name}': specify either `script` or `module`, not both.")
        if not self.script and not self.module:
            raise ProcessingError(f"Stage '{self.name}': must specify `script` or `module`.")


@dataclass
class Pipeline:
    name: str
    stages: list[Stage]
    working_dir: Path
    python: str
    yaml_path: Path
    description: str = ''

    @classmethod
    def load(cls, path: str | Path) -> 'Pipeline':
        path = Path(path).resolve()
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        working_dir = (path.parent / doc.get('working_dir', '.')).resolve()
        stages = []
        seen_names = set()
        for s in doc.get('stages') or []:
            name = s.get('name')
            if name is None:
                key = s.get('script') or s.get('module', '')
                name = Path(key).stem if s.get('script') else key.split('.')[-1]
            if name in seen_names:
                raise ProcessingError(f"Duplicate stage name: '{name}'.")
            seen_names.add(name)
            stages.append(Stage(name=name,
                                script=s.get('script'),
                                module=s.get('module'),
                                variants=s.get('variants') or [],
                                cwd=s.get('cwd')))
        return cls(name=doc.get('name', path.stem),
                   stages=stages,
                   working_dir=working_dir,
                   python=doc.get('python', sys.executable),
                   yaml_path=path,
                   description=doc.get('description', ''))


_AREA_PLACEHOLDER = '${area}'


def _resolve_variants(stage: Stage, area: str | None) -> list[str]:
    """Substitute `${area}` placeholders in this stage's `variants` list.

    Errors out on a mismatch — `${area}` used without `--area`, or `--area` passed
    when no stage uses it. The runtime check beats silent no-op behavior.
    """
    has_placeholder = any(_AREA_PLACEHOLDER in v for v in stage.variants)
    if has_placeholder and area is None:
        raise ProcessingError(
            f"Stage '{stage.name}' uses {_AREA_PLACEHOLDER} in `variants` but no "
            f"--area was passed. Run with `--area <name>`.")
    if area is None:
        return list(stage.variants)
    return [v.replace(_AREA_PLACEHOLDER, area) for v in stage.variants]


def _stage_invocations(stage: Stage, python: str, scenario: str | None,
                       area: str | None) -> list[list[str]]:
    """Build the subprocess argv list for a stage (one per variant, or one if none).

    `scenario`, if provided, is appended as `--scenario <scenario>` to every
    invocation. Preparation-namespace scripts will reject it at `init_context`;
    the runner does not detect the namespace itself.

    `area`, if provided, substitutes `${area}` placeholders in this stage's
    `variants` list — see `_resolve_variants`.
    """
    base = [python]
    if stage.script:
        base.append(stage.script)
    else:
        base.extend(['-m', stage.module])
    scenario_args = ['--scenario', scenario] if scenario else []
    variants = _resolve_variants(stage, area)
    if variants:
        return [base + ['--variant', v] + scenario_args for v in variants]
    return [base + scenario_args]


def run_stage(stage: Stage, pipeline: Pipeline,
              scenario: str | None = None, area: str | None = None) -> None:
    if stage.cwd:
        cwd = (pipeline.yaml_path.parent / stage.cwd).resolve()
    else:
        cwd = pipeline.working_dir
    invocations = _stage_invocations(stage, pipeline.python, scenario, area)
    n = len(invocations)
    for i, argv in enumerate(invocations, start=1):
        suffix = f' [{i}/{n}]' if n > 1 else ''
        logging.info(f"--- stage '{stage.name}'{suffix}: {' '.join(argv)}  (cwd: {cwd}) ---")
        result = subprocess.run(argv, cwd=cwd)
        if result.returncode != 0:
            raise ProcessingError(f"Stage '{stage.name}'{suffix} failed (exit {result.returncode}).")


def select_stages(
    pipeline: Pipeline,
    start_from: str | None = None,
    only: str | None = None,
    skip: list[str] | None = None,
) -> list[Stage]:
    skip = skip or []
    if only and start_from:
        raise ProcessingError("Use --only OR --from, not both.")
    all_names = [s.name for s in pipeline.stages]
    for name in [n for n in (only, start_from, *skip) if n is not None]:
        if name not in all_names:
            raise ProcessingError(f"Unknown stage '{name}'. Known: {all_names}.")
    if only:
        return [s for s in pipeline.stages if s.name == only]
    selected = pipeline.stages
    if start_from:
        i = all_names.index(start_from)
        selected = pipeline.stages[i:]
    return [s for s in selected if s.name not in skip]


def run_pipeline(
    pipeline: Pipeline,
    start_from: str | None = None,
    only: str | None = None,
    skip: list[str] | None = None,
    scenario: str | None = None,
    area: str | None = None,
) -> None:
    stages = select_stages(pipeline, start_from=start_from, only=only, skip=skip)
    if area is not None:
        unused = not any(_AREA_PLACEHOLDER in v for s in stages for v in s.variants)
        if unused:
            raise ProcessingError(
                f"--area {area!r} passed but no selected stage uses "
                f"{_AREA_PLACEHOLDER} in `variants`. Drop --area or pick a "
                f"pipeline that needs it.")
    parts = []
    if scenario:
        parts.append(f"scenario={scenario}")
    if area:
        parts.append(f"area={area}")
    suffix = f" ({', '.join(parts)})" if parts else ""
    logging.info(f"=== Pipeline '{pipeline.name}': {len(stages)} stage(s) selected{suffix} ===")
    for stage in stages:
        run_stage(stage, pipeline, scenario=scenario, area=area)
    logging.info(f"=== Pipeline '{pipeline.name}': done ===")


def list_stages(pipeline: Pipeline) -> None:
    print(f"Pipeline: {pipeline.name}")
    if pipeline.description:
        print(f"  {pipeline.description}")
    print(f"Working dir: {pipeline.working_dir}")
    print(f"Stages ({len(pipeline.stages)}):")
    for s in pipeline.stages:
        target = s.script if s.script else f"-m {s.module}"
        var_str = f"  variants: {s.variants}" if s.variants else ''
        cwd_str = f"  cwd: {s.cwd}" if s.cwd else ''
        print(f"  - {s.name}: {target}{var_str}{cwd_str}")


def _setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-9s %(message)s',
                        stream=sys.stdout)


def main():
    _setup_logging()
    p = argparse.ArgumentParser(prog='python -m aperta.pipeline')
    sub = p.add_subparsers(dest='command', required=True)

    p_run = sub.add_parser('run', help='Execute pipeline stages')
    p_run.add_argument('pipeline', help='Path to pipeline.yml')
    p_run.add_argument('--from', dest='start_from', help='Start from this stage name')
    p_run.add_argument('--only', help='Run only this stage')
    p_run.add_argument('--skip', action='append', default=[],
                       help='Skip this stage (repeat for multiple)')
    p_run.add_argument('--scenario',
                       help='Scenario forwarded to each project stage as --scenario <name>. '
                            'Preparation stages reject it; pipelines that mix prep + project '
                            'stages should omit this and let project stages use their '
                            '`DEFAULT_SCENARIO`.')
    p_run.add_argument('--area',
                       help='Substitutes ${area} placeholders in each stage\'s variants list. '
                            'Use with pipelines like preparation/world/pipeline.yml that '
                            'parameterise across the areas defined in areas.py.')

    p_list = sub.add_parser('list', help='List pipeline stages')
    p_list.add_argument('pipeline', help='Path to pipeline.yml')

    args = p.parse_args()
    pipeline = Pipeline.load(args.pipeline)
    if args.command == 'run':
        run_pipeline(pipeline, start_from=args.start_from, only=args.only,
                     skip=args.skip, scenario=args.scenario, area=args.area)
    elif args.command == 'list':
        list_stages(pipeline)


if __name__ == '__main__':
    main()
