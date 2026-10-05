"""Coefficient sources + cross-scenario dispatch for the atlas pipeline.

Each `Scenario` declares where every coef comes from via a
`Scenario.coefs: dict[str, CoefSource]` mapping. The three source kinds are:

- **`Calibrate()`** — this scenario calibrates the coef from its own data
  via the appropriate calibration script (04 for `edge_weights_<mode>`,
  08a for `overheads_road`, etc.). Output lands at
  `<scenario>/coefs/calibrated/<name>.csv`.

- **`ImportFrom('<other_scenario>')`** — this scenario uses the coef
  fitted by another scenario in the same project. The calibration script
  reads the foreign coef via that scenario's own source declaration (so
  chains are handled), then writes the copy to
  `<scenario>/coefs/transferred/<name>.csv`.

- **`HandWritten()`** — the user places the coef file at
  `<scenario>/coefs/manual/<name>.csv` directly. The calibration script
  verifies the file exists and doesn't touch it.

The on-disk `<kind>/` subfolder (`calibrated` / `transferred` /
`manual`) IS the provenance signal — no per-file sidecar is written.

`Context.get_coefs(name)` is the read API: it consults the scenario's
declaration to figure out which subfolder to read from. Consumers
(survey/08a / 09a / story.py / …) don't need to know which kind a coef is.

`coefs.resolve(context, name, calibrate_fn)` is the write/dispatch API:
calibration scripts call it once per coef, passing a `calibrate_fn`
callable that runs the OLS (used only when the source is `Calibrate`).
For `ImportFrom` it performs the cross-scenario copy; for `HandWritten`
it verifies the file exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd

from aperta.errors import ContextError

if TYPE_CHECKING:
    from aperta_atlas.context import Context


# ---------------------------------------------------------------------------
# Source declarations (used in scenarios.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefSource:
    """Marker base for coef source declarations. Don't instantiate directly —
    use `Calibrate`, `ImportFrom`, or `HandWritten`."""


@dataclass(frozen=True)
class Calibrate(CoefSource):
    """Source: this scenario fits the coef from its own data via the
    appropriate calibration script. Output → `coefs/calibrated/<name>.csv`."""


@dataclass(frozen=True)
class ImportFrom(CoefSource):
    """Source: copy the coef from another scenario in the same project.
    Output → `coefs/transferred/<name>.csv` (verbatim copy of the source
    scenario's effective coef; the `transferred/` subfolder itself
    records the provenance)."""
    scenario: str


@dataclass(frozen=True)
class HandWritten(CoefSource):
    """Source: the user places the coef file at `coefs/manual/<name>.csv`
    by hand. The calibration script doesn't touch the file; it only
    verifies it exists before downstream stages try to read it."""


# Subfolder name per source kind. Kept here so callers don't hand-roll the
# convention.
_KIND_FOR_SOURCE: dict[type[CoefSource], str] = {
    Calibrate: 'calibrated',
    ImportFrom: 'transferred',
    HandWritten: 'manual',
}


def _kind_for(source: CoefSource) -> str:
    """Map a `CoefSource` instance to its on-disk subfolder name."""
    try:
        return _KIND_FOR_SOURCE[type(source)]
    except KeyError as exc:
        raise ContextError(
            f"Unknown CoefSource subtype {type(source).__name__!r}. Use one of "
            f"{sorted(c.__name__ for c in _KIND_FOR_SOURCE)}."
        ) from exc


# ---------------------------------------------------------------------------
# Scenario lookup
# ---------------------------------------------------------------------------


def _get_source(context: 'Context', name: str) -> CoefSource:
    """Look up `scenarios.SCENARIOS[context.scenario].coefs[name]`."""
    if context.project is None or context.scenario is None:
        raise ContextError(
            f"Coefs are project-bound (current namespace: {context.namespace!r}). "
            f"Call `get_coefs` / `coefs.resolve` from a project context only.")
    from aperta_atlas.context import _read_scenarios_module
    scenarios_mod = _read_scenarios_module()
    if scenarios_mod is None:
        raise ContextError(
            "Cannot import top-level `scenarios` module to look up coefs.")
    scenarios_dict = getattr(scenarios_mod, 'SCENARIOS', None)
    if scenarios_dict is None:
        raise ContextError("scenarios.SCENARIOS not defined.")
    scenario = scenarios_dict.get(context.scenario)
    if scenario is None:
        raise ContextError(
            f"scenario {context.scenario!r} not found in SCENARIOS.")
    coefs_map = getattr(scenario, 'coefs', None)
    if coefs_map is None or name not in coefs_map:
        raise ContextError(
            f"scenario {context.scenario!r} doesn't declare a source for "
            f"coef {name!r}. Add it to the scenario's `coefs={{...}}` dict "
            f"in src/scenarios.py: one of `Calibrate()`, "
            f"`ImportFrom('<other>')`, `HandWritten()`.")
    return coefs_map[name]


# ---------------------------------------------------------------------------
# resolve() — main dispatcher for calibration scripts
# ---------------------------------------------------------------------------


def resolve(
    context: 'Context',
    name: str,
    calibrate_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Resolve and persist this scenario's coef `name`.

    Calibration scripts (04 / 08a / 08b / utility / …) call this once per
    coef. Looks up `scenario.coefs[name]` and:

      - `Calibrate()`         → runs `calibrate_fn()`, writes the result to
                                 `coefs/calibrated/<name>.csv`, returns it.
      - `ImportFrom(other)`   → reads the foreign coef via the source
                                 scenario's own declaration, writes a copy
                                 to `coefs/transferred/<name>.csv`, returns
                                 the copy. `calibrate_fn` is NOT called.
      - `HandWritten()`       → reads `coefs/manual/<name>.csv` (raises if
                                 missing), returns it. `calibrate_fn` is
                                 NOT called.

    `calibrate_fn` is a zero-arg callable returning a `param × profile`
    DataFrame (param index, one column per profile).
    """
    source = _get_source(context, name)
    if isinstance(source, Calibrate):
        df = calibrate_fn()
        context.create_coefs(df, name, kind='calibrated')
        return df
    if isinstance(source, ImportFrom):
        src_ctx = context.source(f'{context.project}/{source.scenario}')
        try:
            src_df = src_ctx.get_coefs(name)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import coef {name!r} from scenario "
                f"{source.scenario!r}: {exc}. Run the relevant calibration "
                f"step for {source.scenario!r} first."
            ) from exc
        context.create_coefs(src_df, name, kind='transferred')
        return src_df
    if isinstance(source, HandWritten):
        # Verify file exists; load and return.
        try:
            return context.get_coefs(name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"HandWritten coef {name!r} expected at "
                f"coefs/manual/{name}.csv but not found. Create the file "
                f"(param index, one column per profile) before running "
                f"downstream stages."
            ) from exc
    raise ContextError(f"Unhandled CoefSource: {source!r}")
