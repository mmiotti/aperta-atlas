"""
Generic statistical helpers shared across aperta-atlas pipelines.

Lightweight, dependency-light utilities that don't fit any single
algorithm module but are useful to multiple. Each function is pure
(except for `logging.warning` side effects where flagged).

Currently:
  - `warn_high_collinearity`: flag pairs of regressor columns whose
    pairwise correlation exceeds a threshold (indicates the regression
    can't cleanly separate their coefficients).
"""

import logging

import numpy as np


def warn_high_collinearity(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.99,
    context_label: str = '',
) -> None:
    """Log a WARNING for each pair of feature columns in `X` whose
    pairwise Pearson correlation exceeds `threshold` in absolute value.

    High collinearity → the regression can fit the data well as a
    joint system but can't reliably attribute the signal to either
    feature individually. Standard errors blow up; coefficient signs
    may flip with small data changes. The warning surfaces this so
    downstream consumers know not to over-interpret individual β's.

    Args:
        X: 2D array, columns = features (regressors).
        feature_names: column labels, len(...) == X.shape[1].
        threshold: |corr| above which to warn. Default 0.99 (very
            high — only flags near-duplicate features). Lower (e.g.
            0.95) for stricter checks.
        context_label: optional prefix for the log message (e.g. the
            calling regression's name). Empty string → bare warning.
    """
    if X.shape[1] < 2:
        return
    with np.errstate(invalid='ignore'):
        corr = np.corrcoef(X.T)
    prefix = f"{context_label}: " if context_label else ""
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) > threshold:
                logging.warning(
                    f"  → {prefix}high collinearity ({c:+.3f}) "
                    f"between features {feature_names[i]!r} and "
                    f"{feature_names[j]!r}. Their individual "
                    f"coefficients may not be reliably separable.")
