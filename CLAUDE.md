# CLAUDE.md

Current-state guidance for Claude Code working in this repo.

## Project nature

This repo (formerly `aperta-lab`) holds (a) `aperta_atlas`, opinionated
project scaffolding on `aperta` (context / typed I/O / coefs / pipeline /
variants), and (b) the project built on it, the Swiss Aperta Accessibility
Atlas pipeline, flat in `src/`. The `aperta` library is the sibling repo
[`../aperta/`](../aperta/) (pinned in [pyproject.toml](pyproject.toml)). The
workspace guide `../home/CLAUDE.md` covers the other repos and the shared
working norms.

Research codebase. Combines transport networks + land-use data to compute
accessibility metrics at high spatial resolution. Designed Swiss-first, the
library is region-agnostic. Data inputs/outputs live outside the repo
(`DATA_DIR_PUBLIC` / `DATA_DIR_PRIVATE` in `.env`) and are typically GB-scale.

## The aperta workflow

aperta 0.4 is organized around a six-phase workflow. Every library module
slots into one of these phases.

1. **Load and prepare data** — networks (per mode); land use; topography.
2. **Map data to units** — `cells → zones` aggregation hierarchy
   via `geo_mapping` + `network_snap.snap_to_network_nodes`.
3. **Build sparse OD pairs** — `od_pairs.get_pairs` returns a
   `TieredODNodePairs` (node-keyed). Lift to `TieredODGeoPairs`
   (cell/zone-keyed) via `od_pairs.reindex_by_geo_unit` for cross-modal
   alignment. Three distance tiers, each a dict-of-arrays at its own
   resolution: `cells_to_cells` (close), `cells_to_zones` (medium —
   cell-resolution origin, zone-resolution dest), `zones_to_zones` (far).
   Consumed by `routing.tiered_path_costs`, `accessibility.*`,
   `traffic_flows.nested_node_sample`.
4. **Estimate traffic flows** — `traffic_flows.nested_node_sample` +
   betweenness via `network_processing.get_nested_edge_betweenness`.
5. **Estimate travel costs** — `routing.tiered_path_costs` /
   `routing.tiered_path_aggregate` (Dijkstra via `scipy.sparse.csgraph`) +
   `overhead.add_geo_overheads` (per-geo-unit origin / destination
   first-mile and last-mile costs added onto a geo-keyed cost ODM).
   Plus `utility.route_utility` / `add_endpoint_utility` for
   utility-based costs.
6. **Calculate accessibilities** — `accessibility.cumulative_opportunities`,
   `accessibility.gravity`, `accessibility.nearest_k`. Cross-modal:
   `od_pairs.aggregate_across_modes` on per-mode `TieredODGeoPairs`,
   then any accessibility primitive on the combined ODM.

`../aperta/tests/test_workflow.py` is the runnable toy-world example (and
integration test); real-world examples:
[`../aperta/examples/README.md`](../aperta/examples/README.md). The 0.5
refactor (`../home/planning/refactor-v0.5/aperta_refactor_plan.md`) will
replace much of this API; it is planned, not started, so the 0.4 names above
are current.

## Source layout

```
src/
    aperta_atlas/                  # scaffolding package; module map in its __init__.py
        context.py tracking.py coefs.py pipeline.py variant.py utils.py   # framework
        osm.py graph_simplification.py dasymetric.py stats_helpers.py routing_mp.py  # helpers
    preparation/                   # scenario-free prep (raw → reusable prepared assets)
        switzerland/
            common.py              # Swiss-wide constants (CRS_CH, CRS_LATLON) + helpers
            public/                # STORAGE = Storage.PUBLIC (declared per subfolder)
                general/           # political boundaries, traffic zones, TLM, income, vehicles
                land_use/          # STATPOP / STATENT dasymetric mapping + intensity coefs
                npvm/              # NPVM zones + transit ODM
                tlmregio/          # swissTLMRegio car network (network-tlmregio scenario)
            private/               # STORAGE = Storage.PRIVATE
                surveys/           # MZMV + MOBIS preprocessing
                traffic_counters/  # pooled Swiss traffic counts (NPVM zaehldaten)
        world/
            areas.py               # `Area` dataclass + `AREAS` dict
            pipeline.yml           # public-data prep chain, `--area <name>`
            osm/, elevation/, land_use/           — STORAGE = Storage.PUBLIC

    # The atlas project — flat at src/ root (one repo = one project):
    scenarios.py                   # PROJECT_NAME + `Scenario` + SCENARIOS + DEFAULT_SCENARIO + per-scenario `coefs` declarations
    scenarios_technical_validation.py  # res-* / radii-* / cv-* / data-* / network-* scenarios (merged into SCENARIOS)
    mode_configs.py                # `ModeConfig` per road mode + `TRANSIT_MODE_CONFIG`
    aoi_filter.py                  # restrict calibration / validation data to the scenario AOI
    main/                          # accessibility pipeline (build outputs); main/pipeline.yml runs all stages
    survey/                        # legs → routed times → overheads → stats (02d, 05, 08a-c;
                                   #   stages of main/pipeline.yml) + 08d (R5 timetable reference
                                   #   for transit trips; standalone; needs r5py + Java, not in
                                   #   pyproject); variants per leg set: mtmc (default),
                                   #   mobis_precovid, mobis_covid
    technical_validation/          # checks of a scenario: vs surveys, transit trips, counters; exact vs tiered OD;
                                   #   scenario vs reference (plan: scenarios_technical_validation.py)
    data_paper/                    # figures + LaTeX table bodies of the Scientific Data descriptor
                                   #   (papers/atlas-pipeline)
    research/                      # gitignored: private experimental scripts (`python -m research.<script>`)
    wctr_paper/                    # gitignored: figure scripts of papers/bike-times-wctr
    _archive/, misc/               # gitignored local-only: legacy code, diagnostics; never import
tests/                             # scaffolding unit tests
status/status.json                 # dependency tracker output (when tracking enabled)
results/<project>/<scenario>/      # figure outputs
pyproject.toml                     # declares the `aperta_atlas` package + extras
```

All project / scenario / area / CRS / storage knobs live as **typed
Python** in `src/scenarios.py` and `src/preparation/world/areas.py`. No
project-config YAML — `init_context` dynamic-imports the top-level
`scenarios` module to read `PROJECT_NAME` (first segment of on-disk
output paths), `DEFAULT_SCENARIO` (fallback when `--scenario` isn't
passed), and `SCENARIOS[<active>].storage` (for the project context's
`default_storage`). Preparation namespaces each declare a `STORAGE`
constant in their `__init__.py` (e.g. `preparation/switzerland/private/surveys`
sets `STORAGE = Storage.PRIVATE` because the survey data is proprietary).

The only YAML files are the `pipeline.yml` files (per-pipeline
definitions; consumed by the optional pipeline runner).

## Conventions

**Library / preparation boundary.** If the code could run 1:1 on a different
country's data, it goes in the `aperta` library (sibling repo) or in
`aperta_atlas` (scaffolding). If it knows the name of a specific input file
or schema, it goes in `preparation/<country>/`. Test: would this work for
a US user with their road network?

**Preparation / project boundary.** Preparation is *scenario-free* — outputs
land under `<DATA_DIR_*>/preparation/<region>/<sub>/<rel>` and are
project-agnostic (any project repo can read them). Project code is
*scenario-bound* — typed I/O writes under
`<DATA_DIR_*>/<PROJECT_NAME>/<scenario>/<rel>`. The `public/` / `private/`
folders only set the storage class and are dropped from the namespace: a
script in `preparation/switzerland/private/surveys/` is read via
`context.source('preparation/switzerland/surveys', storage=Storage.PRIVATE)`.

**Project subfolders are organisational.** `main/`, `technical_validation/`,
`data_paper/` etc. don't become scenarios and don't
appear in the output path. They DO appear in `status.json` keys (e.g.
`atlas/main/04_edge_weights.py` vs `atlas/technical_validation/flows_vs_counters.py`)
so dependency tracking distinguishes scripts by role.

**Scenario vs variant.** *Scenario* = project-wide runtime axis with a
typed `Scenario` entry in `src/scenarios.py`, selected via `--scenario`.
*Variant* = script-local toggle, no project-level config, selected via
`--variant`. Most scripts are *not* scenario-specific. If behavior
differs per scenario, branch on `context.scenario` (or read fields off
`get_scenario(context.scenario)`) inside the script — don't duplicate
files.

**Canonical geo_units.** `cells`, `zones`, `nodes`, `edges` for the
**aperta library** (routing primitives — anything else maps to one of
these during preparation). Plus `buildings` as a **scaffolding-only
addition** in `aperta_atlas.context.CANONICAL_UNITS` — recognised
by the typed-I/O methods (`create_shapes` /
`create_properties` / `get_shapes` / `get_properties`) so multiple
per-building property files can layer onto one geometry file without
duplication. Buildings are NOT routed on by aperta; the algorithm
library's contract remains 4 units. Municipalities / cantons /
locations etc. still map down to a canonical unit in preparation.
Keep the contract tight — no further additions without explicit
discussion.

**ODM convention.** An ODM is a `dict` keyed by origin id, with values that
are either `list[str]` (destination ids; `data_name='idx'`) or `np.ndarray`
of per-OD-pair values aligned to that origin's dest list. Dict-of-arrays is
deliberate — ~100× faster row access than scipy sparse, string keys are
readable. Do not silently convert to sparse / 2D arrays.

**Coordinate systems (Switzerland).** `EPSG:2056` (LV95, meters) is the
canonical CRS for internal processing. `EPSG:4326` only at lat/lon
boundaries (external APIs).

**Custom errors.** `ContextError`, `DataError`, `ProcessingError` (all in
`aperta.errors`) for context-shaped / data-shaped / aperta-logic-shaped
issues respectively. Use these, not bare `ValueError`.

**Calibration coefficients.** Produced/consumed per scenario via the
`aperta_atlas.coefs` system. Each `Scenario.coefs: dict[str, CoefSource]`
declares one of `Calibrate()` / `ImportFrom('<other>')` / `HandWritten()`
per coef name. Calibration scripts call `coefs.resolve(context, name,
calibrate_fn)`; consumers call `context.get_coefs(name)`. On disk:
`<DATA_DIR_*>/<project>/<scenario>/coefs/{calibrated,transferred,manual}/<name>.csv`
— the `<kind>/` subfolder itself is the provenance signal. See the
[README "Coefficients" section](README.md#coefficients) for the
conceptual inventory + new-scenario recipe.

## Context layer (opt-in opinionated)

aperta's algorithm modules need no Context; the `aperta_atlas` scaffolding
layer adds:

- **Resolved filesystem paths** derived from `.env` + project/scenario.
- **Typed I/O**: `context.create_*` / `get_*` for shapes, properties, ODMs,
  networks, generic files, and project results.
- **`Storage` classes**: `PUBLIC`, `PRIVATE`, `SCRATCH`, `RESULTS` — distinct
  on-disk roots. Each project `Scenario` declares its `storage` field; each
  preparation namespace declares a `STORAGE` constant in its `__init__.py`.
  Both are read by `Context.default_storage` so scripts don't need to pass
  `storage=Storage.X` on every call.
- **Opt-in dependency tracking** (`APERTA_TRACK_DEPENDENCIES=1` in `.env`):
  `status/status.json` records script hashes and data I/O and warns about
  stale upstream data.

Script template:

```python
from aperta_atlas.context import init_context

def main():
    context = init_context()
    # ... do work via context.* I/O calls or pure-algorithm calls ...
    context.close()

if __name__ == '__main__':
    main()
```

Cross-namespace reads use `context.source('<path>', storage=...)` where
`<path>` mirrors the on-disk layout:

    context.source('preparation/switzerland/general')             # preparation
    context.source(f'{context.project}/switzerland-h10')          # same project,
                                                                  # other scenario
    context.source('lumos/2020')                                  # cross-project (rare)

The optional `storage=` kwarg pins the source ctx's `default_storage` —
useful when a namespace's `STORAGE` constant is wrong for the read at
hand (mixed-storage preparation namespace) or unreachable cross-repo:

    context.source('preparation/switzerland/surveys',
                   storage=Storage.PRIVATE)

## Setup

- Python **3.12**, the tested version (`pyproject.toml`: `>=3.12`,
  `aperta>=0.4.4a0,<0.5`). Stage 09a needs the `[estimation]` extra
  (biogeme 3.2, which only installs on 3.12). aperta's CI also covers 3.13.
- **Conda-forge** for `osmium-tool` (the `osmium` CLI used by
  `preparation/world/osm/clip_pbf.py`; not on PyPI):
  `conda install -c conda-forge osmium-tool`. `pyosmium` (imported as `osmium`
  by the `*_from_pbf.py` scripts) comes from PyPI via the `[projects]` extra.
- Install: `../home/CLAUDE.md` (all repos) or [README "Setup"](README.md#setup)
  (this repo alone).
- Mark `src/` as the IDE's Sources Root.
- **Run from terminal, not VSCode Cmd+R**, for any script that calls
  `osmium` or other conda-only binaries. VSCode launches don't activate
  the conda env consistently across machines, so `osmium` ends up
  missing from PATH (see `feedback-shell-over-helpers` memory). Use
  `cd src && python -m preparation.world.osm.<script>` from a shell
  with the env active.
- `.env` (gitignored) at the repo root: `WORKING_DIR`, `DATA_DIR_PUBLIC`,
  `DATA_DIR_PRIVATE`, `LOGGING_LEVEL`, `APERTA_TRACK_DEPENDENCIES`.
  Optional per-machine overrides via `*_MACHINE2` + `OS_NAME_MACHINE2`.

## Running scripts

```bash
cd src
python -m main.03b_traffic_flows --scenario switzerland-h10          # one script
python -m aperta_atlas.pipeline run main/pipeline.yml --scenario switzerland-h10
python -m aperta_atlas.pipeline run preparation/world/pipeline.yml --area <name>
```

The pipeline runner takes `--from` / `--only` / `--skip <stage>` for partial
runs. Scenarios: pass `--scenario <name>` or set `DEFAULT_SCENARIO = '<name>'`
in `src/scenarios.py`. Variants: pass `--variant <name>`
for scripts that declare a `Variants(...)` registry. Preparation scripts
reject `--scenario` (they're scenario-free).

## Tests

`python -m unittest discover -s tests -t .` from the repo root (~100 tests):
tiered-ODM round-trips, coefs resolution, pipeline runner, dasymetric mapping,
OSM helpers. Other `Context` methods and `variant` are exercised only by
project scripts.

## Docstring convention for scripts

Every script under `src/` opens with a docstring: one-line present-tense
summary; 1–2 paragraphs on what and why; `Inputs (<STORAGE>, under <path>/):`
and `Outputs (...)` blocks of filename patterns, each with a one-line comment
(outputs name their `Storage` class); then the `python -m` run lines (all
variants, `--variant <name>`, `--scenario <name>` for project scripts).
Canonical example: [clip_pbf.py](src/preparation/world/osm/clip_pbf.py).
