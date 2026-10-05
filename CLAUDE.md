# CLAUDE.md

Current-state guidance for Claude Code working in this repo.

## Project nature

Three interlinked but distinct names:

- **`aperta`** — the reusable Python library, in the sibling
  [`../aperta/`](../aperta/) directory (also at `git@github.com:mmiotti/aperta.git`).
  Pure algorithms — routing, accessibility, OD pairs, calibration. On PyPI
  alongside the toolkit paper. Brought into this repo as a pinned
  dependency declared in [pyproject.toml](pyproject.toml).
- **`aperta-atlas`** — this repository (formerly `aperta-lab`). Holds
  (a) the `aperta_atlas` Python package — opinionated project
  scaffolding on top of `aperta`: context / typed I/O / coefs /
  pipeline / variants — and (b) the canonical project built on it,
  the Swiss "Urban Mobility Atlas" pipeline under `src/`.
- **"Urban Mobility Atlas"** — the human-facing publication name for
  the deliverable produced from this repo's `atlas/`.
- **`aperta-lumos`** (sibling repo, **private**) — historic per-year
  Swiss accessibility, uses proprietary BFS data. Imports the
  `aperta_atlas` scaffolding from this repo; not reproducible by
  third parties. See `../aperta-lumos/`.

Research codebase. Combines transport networks + land-use data to compute
accessibility metrics at high spatial resolution. Designed Swiss-first, the
library is region-agnostic. Data inputs/outputs live outside the repo
(`DATA_DIR_PUBLIC` / `DATA_DIR_PRIVATE` in `.env`) and are typically GB-scale.

## The aperta workflow

aperta is organized around a six-phase workflow. Every library module slots
into one of these phases.

1. **Load and prepare data** — networks (per mode); land use; topography.
2. **Map data to units** — `cells → zones` aggregation hierarchy
   via `geo_mapping` + `network_processing.snap_to_network_nodes` /
   `assign_to_eligible_centroid`.
3. **Build sparse OD pairs** — `od_pairs.get_pairs` returns a
   `TieredODNodePairs` (three distance tiers: cells_to_cells,
   cells_to_zones, zones_to_zones — node-keyed). Lift to
   `TieredODGeoPairs` (cell/zone-keyed) via
   `od_pairs.reindex_by_geo_unit` for cross-modal alignment.
4. **Estimate traffic flows** — `traffic_flows.nested_node_sample` +
   betweenness via `network_processing.get_*_betweenness*`.
5. **Estimate travel costs** — `routing.tiered_path_costs` /
   `routing.tiered_path_aggregate` (Dijkstra on any networkx graph) +
   `overhead.add_geo_overheads` (per-geo-unit origin / destination
   first-mile and last-mile costs added onto a geo-keyed cost ODM).
   Plus `utility.route_utility` / `add_endpoint_utility` for
   utility-based costs.
6. **Calculate accessibilities** — `accessibility.count_in_bins`,
   `accessibility.gravity`, `accessibility.nearest_k`. Cross-modal:
   `od_pairs.aggregate_across_modes` on per-mode `TieredODGeoPairs`,
   then any accessibility primitive on the combined ODM.

For the `aperta` library itself see [`../aperta/README.md`](../aperta/README.md).
The library's `tests/test_workflow.py` is the runnable ~150-line minimal
example showing the full workflow on a toy world; it doubles as the
integration test. Real-world examples live in [`../aperta/examples/`](../aperta/examples/):
`minimal/` is the simplest walking-only example (Cambridge MA, ~10 s),
`walkthrough/` covers every primitive including multi-modal walk + bike
and cross-modal logsum (Central Paris, ~40 s), `calibration/` is the
traffic-flow + edge-weight calibration showcase (Canton of Zurich), and
`benchmarks/` compares routing performance.

## Reference structures

- **Scaffolding package** (in this repo): [src/aperta_atlas/](src/aperta_atlas/).
- **Reference project**: [src/](src/) (the multi-modal accessibility atlas).
- **Reference preparation**: [src/preparation/world/](src/preparation/world/)
  (public-data chain, driven by `preparation/world/pipeline.yml --area <name>`).
- **Example notebooks**: live in the sibling
  [`../aperta/examples/`](../aperta/examples/) — lightweight notebooks using
  `aperta` directly (no scaffolding).

`src/_archive/` and `src/misc/` are gitignored, local-only folders
(pre-refactor legacy code, one-off diagnostics). Do not import from them
in new code.

## Source layout

```
src/
    aperta_atlas/                  # the scaffolding package (installed as `aperta_atlas`)
        context.py                 # Context + init_context (filesystem + paths + typed I/O methods)
        tracking.py                # opt-in dependency tracking (status.json); Context delegates here
        coefs.py                   # CoefSource (Calibrate/ImportFrom/HandWritten) + resolve dispatcher
        pipeline.py                # YAML-driven pipeline runner (optional)
        variant.py                 # `Variants` class for per-script run-variants (optional)
        utils.py                   # generic helpers (step ctx manager, named-tuple tracking)
    preparation/                   # scenario-free prep (raw → reusable prepared assets)
        switzerland/
            common.py              # Swiss-wide constants (CRS_CH, CRS_LATLON) + helpers
            public/                # STORAGE = Storage.PUBLIC (declared per subfolder)
                general/           # political boundaries, traffic zones, TLM, income, vehicles
                land_use/          # STATPOP / STATENT dasymetric mapping + intensity coefs
                npvm/              # NPVM zones + transit ODM
            private/               # STORAGE = Storage.PRIVATE
                surveys/           # MZMV + MOBIS preprocessing
                traffic_counters/  # pooled Swiss traffic counts (NPVM zaehldaten)
        world/
            areas.py               # `Area` dataclass + `AREAS` dict
            pipeline.yml           # public-data prep chain, `--area <name>`
            osm/, elevation/, land_use/           — STORAGE = Storage.PUBLIC

    # The atlas project — flat at src/ root (one repo = one project):
    scenarios.py                   # PROJECT_NAME + `Scenario` + SCENARIOS + DEFAULT_SCENARIO + per-scenario `coefs` declarations
    scenarios_technical_validation.py  # res-* / radii-* / cv-* / data-* scenarios (merged into SCENARIOS)
    mode_configs.py                # `ModeConfig` per road mode + `TRANSIT_MODE_CONFIG`
    aoi_filter.py                  # restrict calibration / validation data to the scenario AOI
    main/                          # accessibility pipeline (build outputs); main/pipeline.yml runs all stages
    survey/                        # survey legs → routed times → overheads → stats (02d, 05, 08a, 08b, 08c);
                                   #   variants per leg set: mtmc (default), mobis_precovid, mobis_covid
    validation/                    # ground-truth comparison (times vs survey, transit trips, flows vs counters, …)
    visualization/                 # figures (story.py, story_condensed.py, fastest_mode.py)
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

The lone remaining YAML is `pipeline.yml` files (per-pipeline
definitions; consumed by the optional pipeline runner).

The `aperta` library itself is **not** in this repo — it lives in the
sibling `../aperta/` directory (or installed from PyPI). See [pyproject.toml](pyproject.toml)
for the version pin.

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
`<DATA_DIR_*>/<PROJECT_NAME>/<scenario>/<rel>`.

**Project subfolders are organisational.** `main/`, `validation/`,
`visualization/`, `analysis/` etc. don't become scenarios and don't
appear in the output path. They DO appear in `status.json` keys (e.g.
`atlas/main/04_edge_weights.py` vs `atlas/validation/flows_vs_counters.py`)
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
addition** in `aperta_atlas.context.CANONICAL_UNITS` (added 2026-06-07)
— recognised by the typed-I/O methods (`create_shapes` /
`create_properties` / `get_shapes` / `get_properties`) so multiple
per-building property files can layer onto one geometry file without
duplication. Buildings are NOT routed on by aperta; the algorithm
library's contract remains 4 units. Municipalities / cantons /
locations etc. still map down to a canonical unit in preparation.
Keep the contract tight — no further additions without explicit
discussion. (The historical `regions` tier was removed when aperta
moved to its two-geo-layer / three-distance-tier OD structure; some
aperta-atlas code may still carry `region` naming for backward
compatibility.)

**ODM convention.** An ODM is a `dict` keyed by origin id, with values that
are either `list[str]` (destination ids; `data_name='idx'`) or `np.ndarray`
of per-OD-pair values aligned to that origin's dest list. Dict-of-arrays is
deliberate — ~100× faster row access than scipy sparse, string keys are
readable. Do not silently convert to sparse / 2D arrays.

**TieredODPairs.** Three distance tiers: `cells_to_cells` (close),
`cells_to_zones` (medium — cell-resolution origin, zone-resolution dest),
`zones_to_zones` (far). Each tier is a dict-of-arrays at its own resolution.
Built by `od_pairs.get_pairs`; consumed by `routing.tiered_path_costs`,
`accessibility.count_in_bins`, `traffic_flows.nested_node_sample`.

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

The aperta algorithm modules (`routing`, `accessibility`, `od_pairs`,
`traffic_flows`, the `geo_*` and `network_processing` modules) work on plain
numpy / pandas / networkx — **no Context required**. Use them directly when
you just want algorithms.

The `aperta_atlas` scaffolding layer adds:

- **Resolved filesystem paths** derived from `.env` + project/scenario.
- **Typed I/O**: `context.create_*` / `get_*` for shapes, properties, ODMs,
  networks, generic files, and project results.
- **`Storage` classes**: `PUBLIC`, `PRIVATE`, `SCRATCH`, `RESULTS` — distinct
  on-disk roots. Each project `Scenario` declares its `storage` field; each
  preparation namespace declares a `STORAGE` constant in its `__init__.py`.
  Both are read by `Context.default_storage` so scripts don't need to pass
  `storage=Storage.X` on every call.
- **Optional dependency tracking**: opt in via `APERTA_TRACK_DEPENDENCIES=1`
  in `.env`. When on, `status/status.json` records script hashes and
  created/consumed data per run; subsequent runs warn about stale upstream
  data. Off by default — newbie scripts run as one-shot with no
  `status.json` clutter.

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

- Python **3.12** (3.13 not yet supported — `pyosmium` and a few other
  geo / scientific deps don't have 3.13 wheels on conda-forge yet).
  `pyproject.toml` declares ≥ 3.12 and `aperta>=0.4.0a0`.
- Dependencies declared in [pyproject.toml](pyproject.toml) (this repo, for
  `aperta_atlas` + project deps) and in the sibling [../aperta/pyproject.toml](../aperta/pyproject.toml)
  (for the `aperta` library).
- **Conda-forge** for the PBF tooling — `osmium-tool` (CLI used by
  `preparation/world/osm/clip_pbf.py`) and `pyosmium` (Python bindings
  used by the `*_from_pbf.py` scripts) are not on PyPI. Install both via
  `conda install -c conda-forge osmium-tool pyosmium`.
- Development install (both packages from local checkouts):
  ```bash
  conda create -n aperta python=3.12
  conda activate aperta
  conda install -c conda-forge osmium-tool pyosmium    # PBF tooling, not on PyPI
  pip install -e ../aperta              # the algorithm library
  pip install -e '.[projects]'          # this repo: scaffolding + heavy project deps
  ```
- Mark `src/` as the IDE's Sources Root.
- **Run from terminal, not VSCode Cmd+R**, for any script that calls
  `osmium` or other conda-only binaries. VSCode launches don't activate
  the conda env consistently across machines, so `osmium` ends up
  missing from PATH (see `feedback-shell-over-helpers` memory). Use
  `cd src && python -m preparation.world.osm.<script>` from a shell
  with the env active.
- `.env` (gitignored) at the repo root: `WORKING_DIR`, `DATA_DIR_PUBLIC`,
  `DATA_DIR_PRIVATE`, `LOGGING_LEVEL`, `APERTA_TRACK_DEPENDENCIES`, and
  optionally `CARTO_API_KEY` (basemap tiles for the story figures — never in
  source code). Optional per-machine overrides via `*_MACHINE2` +
  `OS_NAME_MACHINE2`.

## Running scripts

```bash
cd src && python -m main.03b_traffic_flows --scenario switzerland-h10
```

Scenarios: pass `--scenario <name>` or set `DEFAULT_SCENARIO = '<name>'`
in `src/scenarios.py`. Variants: pass `--variant <name>`
for scripts that declare a `Variants(...)` registry. Preparation scripts
reject `--scenario` (they're scenario-free).

## Tests

Tests live with each package.

- **aperta** (in sibling repo): ~455 tests under `aperta/tests/`. Run from
  the aperta clone: `python -m unittest discover -s tests -t .`. Also runs
  in GitHub Actions on every push to `mmiotti/aperta`, alongside Ruff
  lint/format and mypy type-checking jobs.
- **aperta_atlas** (this repo): 99 tests under [tests/](tests/). Run from
  the repo root:
  ```bash
  python -m unittest discover -s tests -t .
  ```
  Covers `Context.create_/get_odm` round-trips, `coefs` resolution, the
  pipeline runner, dasymetric mapping, and OSM helpers. Other `Context`
  methods and `variant` remain untested (exercised indirectly by project
  scripts).

## Docstring convention for `preparation/` and `projects/` scripts

Every refactored script in `src/preparation/` and `src/` has a
top-of-file docstring with this shape:

```python
"""
<One-line summary in present tense.>

<1-2 paragraphs explaining what the script does and why.>

Inputs[ (under <DATA_DIR_*>/<storage_root>/<path>/)]:
    [<typed-subfolder/>]<filename pattern>      # optional comment
    ...

Outputs:
    <filename pattern>      # <Storage class> — one-line description
    ...

Run all variants sequentially (default):
    python -m <namespace>.<filename>
Single variant:
    python -m <namespace>.<filename> --variant <name>
[Project scripts only — pick a scenario:]
    python -m <namespace>.<filename> --scenario <name>
```

Canonical example: [preparation/world/osm/clip_pbf.py](src/preparation/world/osm/clip_pbf.py).

## Working norms

- **Step-by-step refactor.** Don't propose sweeping rewrites in a single change.
- **Suggestions are wanted.** Surface trim/consolidate opportunities you
  notice along the way; the user wants suggestions in addition to execution.
- **Ask when unclear** — especially about destination layout and naming for
  ongoing reorganization. A quick clarifying question beats guessing wrong.
- **Docs may lie.** When code disagrees with a docstring or this CLAUDE.md,
  trust the code and offer to fix the doc.
- **KISS / DRY / YAGNI.** Library complexity is at the upper limit the
  maintainer wants to carry; new abstractions need a clear, concrete payoff.
