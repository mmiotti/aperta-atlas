# aperta-atlas

> **Published Swiss outputs** — the pipeline's key deliverables for Switzerland are available on Zenodo:
> - **[Accessibility metrics for Switzerland](https://doi.org/10.5281/zenodo.21410511)** — per-cell multi-modal accessibility metrics (time-, utility-, and distance-based) at Uber H3 resolution 10, CC BY 4.0.
> - **[Calibrated transport networks for Switzerland](https://doi.org/10.5281/zenodo.21410967)** — walk / bike / car networks with per-edge calibrated travel-time weights per mode profile, ODbL.
>
> Both links resolve to the latest version of each dataset.

**aperta-atlas** is the pipeline that produces the *Urban Mobility Atlas*:
multi-modal accessibility metrics at high spatial resolution, built on
the [`aperta`](https://github.com/mmiotti/aperta) accessibility-analysis
library. `aperta` provides the algorithms (routing, OD pairs,
accessibility metrics, calibration, traffic flows); this repository adds
the data preparation, the calibration workflow, and the project
scaffolding that tie them into an end-to-end pipeline.

Walk, bike and car run on OpenStreetMap data for any region. The
best-calibrated case is **Switzerland**, which also includes public
transit (NPVM zone-to-zone travel times plus per-cell access overheads
calibrated against survey trips). The Swiss calibration uses
restricted-access data:

- **Travel survey** — the Swiss Mobility and Transport Microcensus
  (MTMC / MZMV): ground truth for walk and bike travel times, the road
  and transit overheads, the traffic-flow trip distribution, and the
  utility model
- **GPS-tracked trips** — MOBIS legs: ground truth for car travel times
  (pre-COVID-19 cohort) and out-of-sample validation
- **Traffic counters** — Swiss ASTRA counters, for traffic-flow validation

This code is **documentation for the datasets it produces**: read it to
understand how an atlas was made, fork it to reproduce a public
scenario, or adapt it to a new region. Its APIs and on-disk paths still
evolve; the reusable, stable primitives live in `aperta`.

## Scenarios

Declared in [`src/scenarios.py`](src/scenarios.py):

- **`switzerland-h10`** (default) — all of Switzerland on H3
  resolution-10 cells, with STATPOP / STATENT population and employment
  per building. The calibration scenario: it fits every coefficient set
  the other scenarios reuse.
- **`cambridgeuk-public`** — Cambridge (UK) on public data only (OSM,
  GHS-POP), walk / bike / car, with all coefficients transferred from
  `switzerland-h10`. Shows how to run a new region without local
  calibration data.
- **Technical-validation scenarios**
  ([`src/scenarios_technical_validation.py`](src/scenarios_technical_validation.py)):
  `res-*` (cell resolution), `radii-*` (OD-tier radii),
  `cv-{de,fr}-{train,test}` (spatial cross-validation between language
  regions), and `data-{coef,ghs}` (public-data population / employment
  sources).

## Repository layout

```
src/
    main/            # the pipeline: 01_cells_zones → … → 10_accessibilities (+ pipeline.yml)
    survey/          # survey legs → routed times → overheads (needed for calibration + validation)
    validation/      # predicted vs observed: travel times, transit trips, flows vs counters
    visualization/   # figures
    preparation/     # scenario-free input preparation: world/ (public data, any region)
                     #   and switzerland/ (Swiss sources, partly restricted-access)
    aperta_atlas/    # project scaffolding: context + typed I/O, coefficients, variants, pipeline runner
    scenarios.py, scenarios_technical_validation.py, mode_configs.py
tests/               # scaffolding unit tests
```

Project settings (areas, scenarios, per-mode routing) are typed Python
in `preparation/world/areas.py`, `scenarios.py` and `mode_configs.py`;
there is no configuration YAML.

## Setup

Tested on **Python 3.12**.

```bash
conda create -n aperta python=3.12
conda activate aperta
conda install -c conda-forge osmium-tool pyosmium   # PBF tooling, not on PyPI
pip install -e '.[projects]'                        # this repo + project deps (pulls aperta from PyPI)
```

For `aperta` development, install a local clone instead
(`pip install -e ../aperta`). Mark `src/` as the IDE's Sources Root.

Create a `.env` at the repository root (gitignored):

```
WORKING_DIR=/absolute/path/to/aperta-atlas
DATA_DIR_PUBLIC=/absolute/path/to/your/data/root
DATA_DIR_PRIVATE=/absolute/path/to/your/protected/data/root
LOGGING_LEVEL=INFO
APERTA_TRACK_DEPENDENCIES=1      # optional: dependency tracking in status/status.json
CARTO_API_KEY=...                # optional: basemap tiles for visualization/story*.py
```

## Running

Run from `src/` in a terminal with the conda env active (VS Code launches
don't always pick up conda's `osmium`).

**Whole pipeline** for a scenario, or the public-data preparation for an area:

```bash
cd src
python -m aperta_atlas.pipeline run main/pipeline.yml --scenario switzerland-h10
python -m aperta_atlas.pipeline run main/pipeline.yml --scenario switzerland-h10 --from edge_weights
python -m aperta_atlas.pipeline run preparation/world/pipeline.yml --area cambridgeuk
```

**Single scripts:**

```bash
python -m main.01_cells_zones --scenario cambridgeuk-public
python -m survey.05_leg_times --scenario switzerland-h10 --variant mobis_precovid
```

`--scenario` defaults to `DEFAULT_SCENARIO`; scripts with run-variants
take `--variant <name>` (or `--variant all`).

## Coefficients

Each scenario declares, per coefficient set, where it comes from:
**`Calibrate()`** (fit from the scenario's own data), **`ImportFrom('<scenario>')`**
(reuse another scenario's coefficients), or **`HandWritten()`** (a file
placed by hand). A new region can therefore run with no calibration data
at all, by transferring everything from Switzerland — see
`cambridgeuk-public` in `scenarios.py`.

| Coefficient set | Calibrated in | Calibration data |
|---|---|---|
| `edge_weights_{walk,bike,car}` — per-edge travel times per profile | `main/04` | MTMC legs (walk, bike), MOBIS GPS legs (car) |
| `node_trip_weights_car`, `flow_cost_bins_car` — trip generation + trip-time distribution | `main/03a` | MTMC car trips, cell population / employment |
| `overheads_road` — per-cell origin / destination overheads | `main/07a` | MTMC legs |
| `overheads_transit` — per-cell transit access overheads | `main/07b` | MTMC trips with a transit leg |
| `utility_*` — mode-choice utility model | `main/09a` | MTMC legs + sociodemographics |
| Population / employment intensities per OSM building tag | `preparation/switzerland/public/land_use/` | BFS STATPOP / STATENT |

Coefficients are stored per scenario under `coefs/calibrated/`,
`coefs/transferred/` or `coefs/manual/`, so the folder records their
provenance.

**Adding a scenario:**

1. Add a `Scenario(...)` entry to `SCENARIOS` in `src/scenarios.py`,
   with a `coefs={...}` source per coefficient set.
2. Prepare its area (`preparation/world/pipeline.yml --area <name>`).
3. Run `main/pipeline.yml --scenario <name>`. Calibration stages fit,
   copy, or check each coefficient set according to its declared source.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Acknowledgments

Aperta-atlas was developed by Marco Miotti at the [Chair of Ecological Systems Design](https://esd.ifu.ethz.ch/), [ETH Zurich](https://ethz.ch), in the context of the [BlueCity](https://www.epfl.ch/schools/enac/blue-city-project/) project and [LUMOS](https://csfm.ethz.ch/en/research/projects/lumos.html).

## License

MIT. See [LICENSE](LICENSE).
