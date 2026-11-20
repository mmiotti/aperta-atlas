"""Shared helpers for the validation scripts.

- `agreement_stats`: the bias / MAE / RMSE / slope / R² / MAE% set
  reported by every ground-truth comparison (`times_vs_survey*.py`,
  `flows_vs_counters.py`) and by the accessibility comparisons
  (`access_vs_reference.column_stats`).
- `DIST_BANDS`, `STATS_RENAME`, `STATS_FORMATTERS`: distance bands and the
  table columns of `times_vs_survey*.py`.
"""

import numpy as np

# Straight-line distance bands (metres). Coarse on purpose: MTMC self-report is
# 5-min-rounded, so finer bands mostly add rounding noise; shared across leg sets so
# MTMC and MOBIS rows compare. <2 km and 2-5 km split the short legs, where per-trip
# overheads weigh most.
DIST_BANDS: list[tuple[str, float, float]] = [
    ('   <2km',       0.0,    2_000.0),
    ('   2-5km',  2_000.0,    5_000.0),
    ('  5-25km',  5_000.0,   25_000.0),
    ('   >25km', 25_000.0,  np.inf),
]

# `agreement_stats` keys → table headers (plus the script's own name for `n`).
STATS_RENAME: dict[str, str] = {
    'bias_pct': 'bias%', 'med_resid_pct': 'med_resid%',
    'r2': 'R²_1to1', 'r2_fit': 'R²_fit', 'mae_pct': 'MAE%',
}

# Console formatters for the table headers (plus the script's own for `n`).
STATS_FORMATTERS = {
    'bias':       '{:+8.1f}'.format,
    'bias%':      '{:+.1f}%'.format,
    'med_resid':  '{:+8.1f}'.format,
    'med_resid%': '{:+.1f}%'.format,
    'R²_1to1':    '{:.4f}'.format,
    'R²_fit':     '{:.4f}'.format,
    'MAE%':       '{:.1f}%'.format,
}


def agreement_stats(measured: np.ndarray, predicted: np.ndarray) -> dict:
    """Agreement of `predicted` with `measured` on the finite-in-both subset.

    `bias` = mean(predicted − measured) (positive → model over-predicts);
    `slope` = OLS slope of predicted on measured; `r2` = 1:1 R²
    (`1 − SS_res / SS_tot`, as in `aperta.calibration`; penalises bias and
    scale, negative when worse than the measured mean); `r2_fit` = squared
    Pearson; `*_pct` relative to the mean measured value. `med_resid` =
    median(predicted − measured): where the typical case lies relative to the
    prediction. The fits target the mean, so in sample `bias` ≈ 0 while a
    right-skewed tail of long measured values puts `med_resid` above 0;
    `med_resid_pct` is relative to the median measured value.
    """
    mask = np.isfinite(measured) & np.isfinite(predicted)
    n = int(mask.sum())
    if n < 2:
        return {'n': n, 'bias': np.nan, 'bias_pct': np.nan,
                'med_resid': np.nan, 'med_resid_pct': np.nan,
                'mae': np.nan, 'rmse': np.nan,
                'slope': np.nan, 'r2': np.nan, 'r2_fit': np.nan, 'mae_pct': np.nan}
    m, p = measured[mask], predicted[mask]
    resid = p - m
    m_mean = float(m.mean())
    bias = float(resid.mean())
    m_c = m - m_mean
    p_c = p - float(p.mean())
    ss_m = float((m_c ** 2).sum())
    slope = float((m_c * p_c).sum() / ss_m) if ss_m > 0 else np.nan
    r2 = 1.0 - float((resid ** 2).sum()) / ss_m if ss_m > 0 else np.nan
    ss_p = float((p_c ** 2).sum())
    r2_fit = float((m_c * p_c).sum()) ** 2 / (ss_m * ss_p) if ss_m > 0 and ss_p > 0 else np.nan
    mae = float(np.abs(resid).mean())
    m_abs_mean = float(np.abs(m).mean())
    mae_pct = 100.0 * mae / m_abs_mean if m_abs_mean > 0 else np.nan
    bias_pct = 100.0 * bias / m_mean if m_mean != 0 else np.nan
    med_resid = float(np.median(resid))
    m_median = float(np.median(m))
    med_resid_pct = 100.0 * med_resid / m_median if m_median != 0 else np.nan
    return {
        'n': n, 'bias': bias, 'bias_pct': bias_pct,
        'med_resid': med_resid, 'med_resid_pct': med_resid_pct,
        'mae': mae, 'rmse': float(np.sqrt((resid ** 2).mean())),
        'slope': slope, 'r2': r2, 'r2_fit': r2_fit, 'mae_pct': mae_pct,
    }
