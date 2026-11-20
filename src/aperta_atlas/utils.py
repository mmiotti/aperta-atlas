"""Small generic utilities for aperta-atlas pipelines.

Grouped:

- **Pipeline progress logging**: `step` context manager.
- **Numeric / pandas helpers**: `round_to_significant_figures`,
  `most_common`.
- **Dependency-tracking NamedTuple factory**: `tracked_namedtuple` —
  produces a NamedTuple subclass that records which fields were accessed;
  used by the reproducibility scaffold (Context layer).
"""

import contextlib
import logging
import math
import time
from typing import Any, NamedTuple

import pandas as pd


# Steps faster than this are not annotated with a duration on completion.
# Slow steps get a `… (Xs)` line at the end so bottlenecks stay visible.
_STEP_DURATION_THRESHOLD_S = 60.0


@contextlib.contextmanager
def step(label: str, duration_threshold: int | None = None):
    """Context manager that logs a single header line on ENTRY —
    `▶ <label>` — so substantive sub-logs read naturally underneath it.
    On exit, prints `… <label> (Xs)` ONLY if the block took longer than
    `_STEP_DURATION_THRESHOLD_S` seconds — keeps the routine fast-step
    output uncluttered while still surfacing slow bottlenecks.

    Example:
        with step('osmium extract'):
            subprocess.run(['osmium', 'extract', ...], check=True)

    Logs at INFO level via the standard `logging` module — assumes
    `init_context(...)` (or equivalent) has set up the logging config.
    """
    logging.info(f"▶ {label}")
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        if dt >= (duration_threshold or _STEP_DURATION_THRESHOLD_S):
            logging.info(f"… {label} ({dt:.1f}s)")


def tracked_namedtuple(labels_and_types: list[tuple[str, type]]) -> type:
    """Factory for creating named tuples that track access to its fields through self._accessed_fields.

    There may be some unexpected behavior when using a debugger/IDE, which may access object attributes outside the
    actual code. This implementation using named tuples works better in PyCharm than an implementation using
    (frozen) data classes, as starting the debug console will access all attributes of any instances of the latter.

    This feature is used to keep track of dependencies (more specifically, flagging outdated output data if input
    parameters that were used to create that data have changed). False positives are therefore not a big deal; they
    will mostly lead to false positive dependency flags.
    """

    # Dynamic NamedTuple construction — mypy expects a literal arg list here,
    # but the whole point of this factory is the runtime field list.
    nt = NamedTuple("nt", labels_and_types)  # type: ignore[misc]

    class TrackedNamedTuple(nt):
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            cls._accessed_fields = set()
            return super().__new__(cls, *args, **kwargs)

        def __getattribute__(self, key):
            if key != "_fields" and key in self._fields:
                self._accessed_fields.add(key)
            return super().__getattribute__(key)

        def used_fields_as_dict(self) -> dict:
            return {k: v for k, v in self._asdict().items() if k in self._accessed_fields}

    return TrackedNamedTuple


def round_to_significant_figures(x, n):
    r"""Round `x` to `n` significant figures (e.g.\ `round_to_significant_figures(1234, 2) == 1200`).

    Zero is returned as-is. For non-zero `x`, the number of decimal places to
    round to is derived from `log10(abs(x))`.
    """
    return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def most_common(a: pd.Series) -> Any:
    """Return the most-frequent value in `a` (the modal value).

    Ties are broken by `value_counts`'s default ordering (first occurrence wins).
    Use in `groupby().agg()` patterns where you want the modal value per group.
    """
    return a.value_counts().index[0]


