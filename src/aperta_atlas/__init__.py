"""aperta_atlas — opinionated project scaffolding on top of `aperta`.

The framework half (use any subset; nothing requires all of them):

- `aperta_atlas.context`  — `Context` dataclass + `init_context()` entry
                            point. Resolves paths from `.env`, reads
                            `PROJECT_NAME` / `DEFAULT_SCENARIO` /
                            `SCENARIOS` from the top-level `scenarios`
                            module, and owns the typed I/O surface:
                            `create_/get_shapes` (`.gpkg`),
                            `create_/get_properties` (`.csv`),
                            `create_/get_tiered_odm` (TieredODPairs as
                            one compressed `.npz`), `create_/get_nw`
                            (`.graphml` skeleton + companion property
                            CSVs), `create_/get_generic`,
                            `create_/get_results`. Plus the `Storage`
                            enum (`PUBLIC` / `PRIVATE` / `SCRATCH` /
                            `RESULTS`).
- `aperta_atlas.tracking` — Opt-in dependency tracking via
                            `status/status.json`. Activated by
                            `APERTA_TRACK_DEPENDENCIES=1` in `.env`;
                            `Context` delegates here.
- `aperta_atlas.coefs`    — Coefficient sources + dispatcher. Each
                            `Scenario.coefs` declares per-coef source
                            (`Calibrate()` / `ImportFrom('<other>')` /
                            `HandWritten()`); calibration scripts call
                            `coefs.resolve(...)` to route to the right
                            path. On-disk shape:
                            `coefs/{calibrated,transferred,manual}/<name>.csv`.
- `aperta_atlas.pipeline` — Stage-list runner driven by `pipeline.yml`
                            (`python -m aperta_atlas.pipeline`).
                            Sequential subprocess execution; partial
                            runs via `--from` / `--only` / `--skip`.
- `aperta_atlas.variant`  — `Variants` registry for declaring per-script
                            run-variants, with CLI dispatch via
                            `Variants.run(main)` (`--variant <name>`).
- `aperta_atlas.utils`    — `step` context manager + a few generic
                            helpers; mostly used by project scripts.

The OSM-aware / domain-helper half (used by the preparation scripts in
this repo; not strictly part of the framework):

- `aperta_atlas.osm`               — OSM tag conventions, highway-rank
                                     dict, consolidation cleanup,
                                     graphml dtype handling.
- `aperta_atlas.graph_simplification` — PBF-pipeline graph cleanup
                                     (collapse degree-2 chains, prune
                                     short dead ends).
- `aperta_atlas.dasymetric`        — Building-level allocation of
                                     spatial-unit data (e.g. STATPOP).
- `aperta_atlas.stats_helpers`     — Bin-edge helpers for the
                                     traffic-flow calibration.
- `aperta_atlas.routing_mp`        — Multi-process Dijkstra wrapper
                                     using fork-based COW graph sharing.

The algorithm modules under `aperta.*` are the substrate `aperta_atlas`
sits on top of. They take plain numpy / pandas / networkx inputs and
don't know anything about Context / filesystem / scenarios.
"""
