# aperta-atlas

> **Published Swiss outputs** — the pipeline's key deliverables for Switzerland are available on Zenodo:
> - **[Accessibility metrics for Switzerland](https://doi.org/10.5281/zenodo.21410511)** — per-cell multi-modal accessibility metrics (time-, utility-, and distance-based) at Uber H3 resolution 10, CC BY 4.0.
> - **[Calibrated transport networks for Switzerland](https://doi.org/10.5281/zenodo.21410967)** — walk / bike / car networks with per-edge calibrated travel-time weights per mode profile, ODbL.
>
> Both links resolve to the latest version of each dataset.

**aperta-atlas** is a pipeline implementation of the
[`aperta`](https://github.com/mmiotti/aperta) accessibility-analysis library, designed to
produce multi-modal (walk / bike / car / transit) urban accessibility
metrics for any region in the world.

The current best-calibrated case is **Switzerland**, including a
public-transit accessibility component built on NPVM zone-to-zone
travel times plus per-cell endpoint adjustments calibrated against
survey trips. The Swiss calibration relies on several types of
proprietary input:

- **Travel survey legs** — MZMV / MTMC trip records (origin, destination,
  mode, reported time), used as edge-level ground truth for walk and
  bike, for the road and transit overheads, the traffic-flow trip
  distribution, and the utility model
- **GPS-tracked trips** — MOBIS legs, used as edge-level ground truth
  for car (pre-COVID-19 cohort) and as out-of-sample validation sets
- **Traffic counter readings** — Swiss ASTRA counters, used for
  traffic-flow validation

Scenarios are declared in [`src/scenarios.py`](src/scenarios.py):

- **`switzerland-h10`** (default): all of Switzerland on H3 resolution-10
  cells, STATPOP + STATENT dasymetric mapping for population and
  employment per building. The calibration scenario — it fits every
  coefficient set the other scenarios reuse.
- **`cambridgeuk-public`**: Cambridge (UK) on public data only (OSM,
  GHS-POP), with all coefficients transferred from `switzerland-h10`.
  Demonstrates running a new region without local calibration data.
- **Technical-validation scenarios**
  ([`src/scenarios_technical_validation.py`](src/scenarios_technical_validation.py)):
  `res-*` (cell resolution: H3 9 / 10 / 11 and buildings), `radii-*`
  (OD-tier radii), `cv-{de,fr}-{train,test}` (spatial cross-validation
  between language regions), and `data-{coef,ghs}` (public-data
  population / employment sources).

**Anyone can recreate the pipeline for any region** with OSM data and
a chosen calibration strategy per coefficient set: fit from local data,
transfer from an existing scenario (e.g. Swiss-calibrated weights as a
starting point), or hand-write. See [Coefficients](#coefficients) for
the transferability mechanism.

## What's in this repo

1. The **Accessibility Atlas pipeline** itself ([`src/`](src/)) — the
   numbered stages 01 → 10 that produce accessibility metrics per cell
   × destination type × mode × scenario. Split across
   [`src/main/`](src/main/) (the always-run stack) and
   [`src/survey/`](src/survey/) (survey preprocessing needed only for
   fresh calibration and for validation). The Swiss instantiation is
   published as the *Urban Mobility Atlas*.
2. **Region-specific data preparation** under
   [`src/preparation/`](src/preparation/) — Switzerland-specific
   (`switzerland/`) and region-agnostic OSM / elevation / land-use
   (`world/`) prep that produces the inputs the atlas consumes. The
   `preparation/world/` chain runs on fully public data; some
   `preparation/switzerland/` subfolders consume restricted-access
   inputs (MZMV / MOBIS surveys, ASTRA traffic counters, BFS
   hectare-level STATPOP/STATENT). Those scripts ship
   as documentation of the method.
3. The **`aperta_atlas` Python package**
   ([`src/aperta_atlas/`](src/aperta_atlas/)) — opinionated scaffolding
   (context / typed I/O / variants / coefs / pipeline runner) that the
   atlas pipeline builds on.

## Status

**Active work in progress.** The atlas pipeline runs end-to-end for
multiple scenarios and produces published outputs. APIs and on-disk
paths still evolve. Releases will be tagged as scenarios reach stable
form.

This code is **documentation for the atlas datasets it produces**, not
a third-party API. Third parties may read the pipeline to understand
how an atlas was made, fork it to reproduce a public scenario, or
adapt it to new regions — but should not depend on the scaffolding
APIs as if it were a library. The reusable algorithmic primitives
live in [`aperta`](https://github.com/mmiotti/aperta), which IS treated as a library
(rigorous tests, semver, careful API).

## Relationship to `aperta`

`aperta-atlas` builds on [`aperta`](https://github.com/mmiotti/aperta) — the reusable algorithm library that provides the actual routing, OD pairs, accessibility metrics, calibration, and traffic-flow primitives. `aperta` is region-agnostic pure Python (numpy / pandas / networkx); `aperta-atlas` adds the opinionated project scaffolding, the Swiss data-preparation chain, and the calibration workflow that ties everything into a runnable end-to-end pipeline.

## Layout

```
aperta-atlas/
    src/
        aperta_atlas/                # the scaffolding package (importable as `aperta_atlas`)
            context.py               # Context + init_context (filesystem + paths + typed I/O)
            tracking.py              # opt-in dependency tracking (status.json)
            coefs.py                 # CoefSource (Calibrate/ImportFrom/HandWritten) + resolve
            pipeline.py              # YAML-driven pipeline runner (optional)
            variant.py               # `Variants` class for per-script run-variants
            ...
        preparation/                 # scenario-free prep (raw → reusable prepared assets)
            world/
                areas.py             # `Area` dataclass + `AREAS` dict
                pipeline.yml         # public-data prep chain, parameterised by --area
                osm/, elevation/, land_use/
            switzerland/
                common.py            # Swiss-wide constants (CRS_CH, CRS_LATLON) + helpers
                public/              # STORAGE = PUBLIC (declared in each subfolder's __init__.py)
                    general/         # political boundaries, traffic zones, TLM, income, vehicles
                    land_use/        # STATPOP / STATENT dasymetric mapping + per-OSM-tag intensities
                    npvm/            # NPVM travel-demand model (zones + transit ODM)
                private/             # STORAGE = PRIVATE
                    surveys/         # MZMV + MOBIS preprocessing
                    traffic_counters/

        # The atlas project — flat at src/ root (one repo = one project):
        scenarios.py                 # PROJECT_NAME + Scenario dataclass + SCENARIOS + per-scenario coefs
        scenarios_technical_validation.py  # res-* / radii-* / cv-* / data-* scenarios
        mode_configs.py              # per-mode ModeConfig (walk/bike/car) + TRANSIT_MODE_CONFIG
        aoi_filter.py                # restrict calibration / validation data to the scenario's AOI
        main/                        # the accessibility pipeline: 01_cells_zones → … → 10_accessibilities
            pipeline.yml             # all main + survey stages in order
        survey/                      # survey legs → routed times → overheads → stats (02d, 05, 08a, 08b, 08c)
        validation/                  # predicted vs observed: times (MTMC / MOBIS), transit trips,
                                     #   flows vs counters, accessibility vs reference
        visualization/               # figures (story.py, story_condensed.py, fastest_mode.py)
    tests/                           # unit tests for the scaffolding
    pyproject.toml                   # package = `aperta_atlas`, distribution = `aperta-atlas`
    .env                             # gitignored; per-machine paths (DATA_DIR_PUBLIC, DATA_DIR_PRIVATE, …)
```

The `aperta` library is **not** in this repo — install it from PyPI or
from a local clone of [mmiotti/aperta](https://github.com/mmiotti/aperta).
See [pyproject.toml](pyproject.toml) for the minimum version.

## Setup

Requires **Python 3.12** (3.13 not yet supported — `pyosmium` and a few
other geo/scientific deps lack 3.13 wheels on conda-forge).

```bash
conda create -n aperta python=3.12
conda activate aperta

# PBF tooling (not on PyPI):
conda install -c conda-forge osmium-tool pyosmium

pip install -e '.[projects]'          # this repo: scaffolding + project deps (pulls aperta from PyPI)
# For aperta development, install a local clone instead: pip install -e ../aperta
```

Mark `src/` as the IDE's Sources Root.

`.env` at the repository root (gitignored) — minimal example:

```
WORKING_DIR=/absolute/path/to/aperta-atlas
DATA_DIR_PUBLIC=/absolute/path/to/your/data/root
DATA_DIR_PRIVATE=/absolute/path/to/your/protected/data/root
LOGGING_LEVEL=INFO
APERTA_TRACK_DEPENDENCIES=1            # optional: opt into status.json tracking
CARTO_API_KEY=...                      # optional: CARTO basemap tiles in visualization/story*.py
```

Optional per-machine overrides via `*_MACHINE2` + `OS_NAME_MACHINE2`.

## Running scripts

Project-namespace scripts are scenario-bound; preparation-namespace
scripts are scenario-free.

**Atlas pipeline (single script, single scenario):**

```bash
cd src
python -m main.01_cells_zones --scenario cambridgeuk-public
python -m main.02a_networks_snap --scenario cambridgeuk-public
# ... etc.
```

`--scenario` is optional when running the project's `DEFAULT_SCENARIO`
(see [`src/scenarios.py`](src/scenarios.py)). Scripts with run-variants
take `--variant <name>` (default variant if omitted, `--variant all`
for every variant), e.g. the MOBIS validation leg sets:

```bash
python -m survey.05_leg_times --scenario switzerland-h10 --variant mobis_precovid
```

**Via the pipeline runner** (optional — declarative stage list in a
`pipeline.yml`):

```bash
cd src
python -m aperta_atlas.pipeline run main/pipeline.yml --scenario switzerland-h10
python -m aperta_atlas.pipeline run main/pipeline.yml --scenario switzerland-h10 --from edge_weights
python -m aperta_atlas.pipeline run preparation/world/pipeline.yml --area cambridgeuk
```

Run from a terminal with the conda env active, NOT VSCode Cmd+R — the
PBF-handling scripts (`osmium`, `pyosmium`) need conda's PATH which
VSCode launches don't always pick up.

## Configuration model

All project / scenario / area / CRS / storage knobs are **typed Python**
— there is no project-config YAML.

- `src/preparation/world/areas.py` declares the `Area` dataclass and
  `AREAS` dict — geographic targets with OSM place names, per-mode
  network buffers, AOI buffer, GHSL tiles.
- `src/scenarios.py` declares the project's `Scenario` dataclass, the
  `SCENARIOS` dict (one entry per scenario), the `DEFAULT_SCENARIO`
  constant (fallback when `--scenario` is omitted), and each scenario's
  `coefs` dict (see [Coefficients](#coefficients) below).
- `src/mode_configs.py` declares per-mode routing config (`ModeConfig`
  per mode + `OdmRadii` + `CalibratedSource`, plus
  `TRANSIT_MODE_CONFIG`).

## Coefficients

The pipeline relies on a small number of **coefficient sets** that
parametrise the route-cost, overhead, traffic-flow and utility models.
Each set is produced by one calibration stage and consumed by one or
more downstream stages. To run the full pipeline for a scenario, each
of these sets must be available.

**Per-edge route-cost weights** (`edge_weights_<mode>`) — one weight vector per profile (walk, bike + ebike variants, car peak / base / night)
- *Produced by:* edge-weight calibration (stage **04**)
- *Consumed by:* road OD times (**05**), survey leg routing (**survey/05**), visualization
- *Calibration data:* per-mode travel-time observations — MZMV legs for walk and bike, MOBIS GPS legs for car

**Traffic-flow coefficients** (`node_trip_weights_car`, `flow_cost_bins_car`)
- *Produced by:* flow-coefficient calibration (stage **03a**): a Poisson GLM for per-cell trip generation, and percentile bins of the observed car trip-time distribution
- *Consumed by:* traffic-flow estimation (**03b**)
- *Calibration data:* MZMV car trips + per-cell population / employment (counters are used for validation only, in `validation/flows_vs_counters.py`)

**Per-cell road-trip overheads** (`overheads_road`, per profile)
- *Produced by:* road-overhead calibration (stage **07a**)
- *Consumed by:* gross OD times (**08a**), survey-leg gross-time reconstruction (**survey/08a**)
- *Calibration data:* travel-survey legs (observed times vs. modelled routed times)

**Per-cell transit overheads** (`overheads_transit`)
- *Produced by:* transit-overhead calibration (stage **07b**): trip-level fit of (observed − NPVM zone-to-zone time) on per-cell access features
- *Consumed by:* gross OD times (**08a**, transit branch), survey-trip transit times (**survey/08b**)
- *Calibration data:* MZMV trips with at least one transit leg (door-to-door totals)

**Utility-based travel-cost coefficients** (`utility_<variant>`)
- *Produced by:* biogeme mode-choice fit (stage **09a**)
- *Consumed by:* utility ODMs (**09b**), accessibility (**10**)
- *Calibration data:* MZMV survey legs with per-mode net + gross times, endpoint + route features, sociodemographics

**Per-(OSM tag, category) population and employment intensities**
- *Produced by:* Swiss STATPOP / STATENT dasymetric mapping (`preparation/switzerland/public/land_use/population_statpop.py`, `employment_statent.py`)
- *Consumed by:* public-data per-building estimates (`preparation/world/land_use/{population,employment}_per_building_from_coef.py`), e.g. the `data-coef` scenario
- *Calibration data:* BFS hectare-level STATPOP / STATENT

**Per-area GHS-POP population intensities**
- *Produced by:* GHS-POP dasymetric mapping (`preparation/world/land_use/population_per_building_from_ghs.py`)
- *Consumed by:* — saved for diagnostics; the per-building population from the same script feeds `population_source='ghs'` scenarios (e.g. `data-ghs`, `cambridgeuk-public`)
- *Calibration data:* GHS-POP 100 m raster cell totals

**Transferability.** Each coefficient set in each scenario is sourced
in one of three ways (declared in `src/scenarios.py` per scenario):

- **`Calibrate()`** — fit from the scenario's own calibration data.
  Requires the data listed above to be available for the scenario's
  region (e.g. a local travel survey or GPS-tracking study).
- **`ImportFrom('<other_scenario>')`** — reuse another scenario's
  coefficients verbatim. The copy lands under `coefs/transferred/`,
  so the on-disk subfolder itself records provenance. Lets a new
  region run end-to-end with a sensible starting point (e.g.
  Swiss-calibrated edge weights for a small European city) without
  needing its own calibration data.
- **`HandWritten()`** — place the coefficient file by hand. Useful for
  initial exploration, or for sets without a calibration pipeline in
  the target region.

The three sources mean a new analysis area can be added with **no
calibration data** at all — transfer every coefficient set from
Switzerland, as `cambridgeuk-public` does. Conversely, anyone with
local ground-truth data can re-calibrate any subset.

### On-disk layout

**Project-scoped coefs** live under each scenario's data folder, split
by source kind (the dispatcher choosing between `Calibrate` /
`ImportFrom` / `HandWritten` writes into the matching subfolder):

```
<DATA_DIR_*>/<project>/<scenario>/coefs/
    calibrated/   # fitted by this scenario's own calibration run
    transferred/  # copied verbatim from another scenario
    manual/       # placed by hand
```

**Namespace-scoped coefs** (preparation-side, e.g. STATENT or GHS-POP
calibrations) live under the preparation namespace's folder with a
**flat** layout — preparation has a single source ("computed during
this run"), no dispatcher choosing between options:

```
<DATA_DIR_*>/preparation/<namespace>/coefs/
    <name>.csv
```

Both shapes use the same `param × profile` CSV format. The `<kind>/`
subfolder itself IS the provenance signal — no per-file sidecar is
written. Read either via `context.get_coefs(name)`; the path-shape
branches automatically on whether the context is project-scoped or
namespace-scoped.

### Per-scenario declaration

```python
from aperta_atlas.coefs import Calibrate, ImportFrom, HandWritten

Scenario(
    name='my-region',
    ...,
    coefs={
        'edge_weights_walk':     ImportFrom('switzerland-h10'),
        'edge_weights_bike':     ImportFrom('switzerland-h10'),
        'edge_weights_car':      ImportFrom('switzerland-h10'),
        'overheads_road':        ImportFrom('switzerland-h10'),
        'overheads_transit':     ImportFrom('switzerland-h10'),
        'node_trip_weights_car': ImportFrom('switzerland-h10'),
        'flow_cost_bins_car':    ImportFrom('switzerland-h10'),
        'utility_default':       ImportFrom('switzerland-h10'),
        'utility_default_stats': ImportFrom('switzerland-h10'),
    },
)
```

Scripts consuming coefs (`03b`, `05`, `08a`, `09b`, `10`, the figures,
…) call `context.get_coefs(name)` — they don't need to know whether the
coef was calibrated, transferred, or hand-written.

### Adding a new scenario

1. Add a `Scenario(...)` entry to `SCENARIOS` in `src/scenarios.py`,
   including the `coefs={...}` dict.
2. For each coefficient set, pick one of the three sources above.
3. Run the pipeline (`main/pipeline.yml`, stages `01` → `10`). The
   calibration stages (`03a`, `04`, `07a`, `07b`, `09a`) resolve each
   declared source: fitting where `Calibrate()`, copying-and-tagging
   where `ImportFrom`, or verifying-the-file-exists where
   `HandWritten`. Fresh calibration also needs the `survey/`
   stages (`02d`, `05`, `08a`, `08b`, `08c`), which the pipeline file
   runs in order; for transferred coefficients they short-circuit.

### `overheads_transit` — format

Stage `07b` writes `overheads_transit.csv` with one column per fitted
specification plus `transit`, the column the pipeline applies. For a
`HandWritten()` scenario, a minimal file at
`<DATA_DIR_*>/<project>/<scenario>/coefs/manual/overheads_transit.csv`:

```csv
param,transit
cap_s,900.0
const_s,0.0
t_walk_to_transit_nearest_zone_dev,0.25
t_bike_to_train_nearest_zone_dev,0.35
```

Reserved rows: `cap_s` (required, symmetric clipping bound in seconds),
`const_s` (optional trip-level intercept) and `t_routed` (optional scale
on the routed zone-to-zone time). Every other row names a column in the
cells frame (produced by stage 06 — the `*_zone_dev` columns). The
per-cell transit overhead is `const_s / 2 + Σ coef × cells[column]`,
clipped to ±`cap_s` seconds and added at origin and destination to
NPVM zone-to-zone times.

## Tests

```bash
python -m unittest discover -s tests -t .
```

99 tests covering `Context` round-trips, coefficient resolution, the
pipeline runner, dasymetric mapping, and OSM helpers.

## Acknowledgments

Aperta-atlas was developed at the [Chair of Ecological Systems Design](https://esd.ifu.ethz.ch/) at [ETH Zurich](https://ethz.ch) in the context of the [BlueCity](https://www.epfl.ch/schools/enac/blue-city-project/) project and [LUMOS](https://csfm.ethz.ch/en/research/projects/lumos.html).

## License

MIT. See [LICENSE](LICENSE).
