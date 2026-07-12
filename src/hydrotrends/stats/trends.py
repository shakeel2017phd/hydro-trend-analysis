"""Trend detection for :mod:`hydrotrends`.

Faithful, typed ports of the trend tests from the original script — each
returning a small dataclass instead of a loose dict:

* :func:`mann_kendall` — the (rank-based) Mann-Kendall test with tie-corrected
  variance and the continuity-corrected Z statistic.
* :func:`mann_kendall_modified` — the Hamed-Rao autocorrelation-corrected
  variant; for series shorter than 10 it transparently falls back to the
  original test (there isn't enough data to estimate the correction).
* :func:`sens_slope` — the Theil-Sen median slope and intercept.
* :func:`linear_regression` / :func:`log_linear_regression` — ordinary least
  squares on the values, and on the logs of the positive values.
* :func:`percent_slope` — a slope expressed as % of the mean (per step).

All tests follow the "soft guard" convention: on a series too short to be
meaningful they return ``NaN`` / ``NO_TREND`` rather than raising, because they
run inside loops over many short per-period sub-series where skipping is the
right behaviour. Time is the observation index (0, 1, 2, ...); when the input is
one value per year (as in the across-years-per-period analysis), the slope is
therefore per year.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..core.constants import DEFAULT_ALPHA, MOVING_AVERAGE_WINDOW, TrendDirection
from ..core.utils import to_float_array as _to_array
from ..core.validation import ArrayLike

__all__ = [
    "MannKendallResult",
    "SenSlope",
    "LinearFit",
    "mann_kendall",
    "mann_kendall_modified",
    "sens_slope",
    "linear_regression",
    "log_linear_regression",
    "percent_slope",
    "moving_average_trend",
]


@dataclass(frozen=True)
class MannKendallResult:
    trend: TrendDirection
    p_value: float
    z_score: float
    s: float


@dataclass(frozen=True)
class SenSlope:
    slope: float
    intercept: float


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r2: float
    p_value: float


# ─────────────────────────────────────────────────────────────────────────────
# Mann-Kendall internals
# ─────────────────────────────────────────────────────────────────────────────
def _mk_s(x: np.ndarray) -> float:
    """Mann-Kendall S: signed count of concordant/discordant pairs."""
    n = len(x)
    s = 0.0
    for k in range(n - 1):
        s += float(np.sign(x[k + 1:] - x[k]).sum())
    return s


def _mk_var(x: np.ndarray) -> float:
    """Variance of S with the standard tie correction."""
    n = len(x)
    tie_counts = Counter(x.tolist())
    tie_term = sum(v * (v - 1) * (2 * v + 5) for v in tie_counts.values())
    return (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0


def _z_and_p(s: float, var_s: float) -> tuple[float, float]:
    """Continuity-corrected Z and its two-sided normal p-value."""
    if var_s <= 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / np.sqrt(var_s)
    p = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _trend_label(p_value: float, s: float, alpha: float) -> TrendDirection:
    if p_value < alpha and s > 0:
        return TrendDirection.INCREASING
    if p_value < alpha and s < 0:
        return TrendDirection.DECREASING
    return TrendDirection.NO_TREND


# ─────────────────────────────────────────────────────────────────────────────
# Public tests
# ─────────────────────────────────────────────────────────────────────────────
def mann_kendall(data: ArrayLike, *, alpha: float = DEFAULT_ALPHA) -> MannKendallResult:
    """Original Mann-Kendall trend test."""
    x = _to_array(data)
    s = _mk_s(x)
    var_s = _mk_var(x)
    z, p = _z_and_p(s, var_s)
    return MannKendallResult(_trend_label(p, s, alpha), p, z, s)


def sens_slope(data: ArrayLike) -> SenSlope:
    """Theil-Sen median slope and intercept (NaN if fewer than 2 points)."""
    x = _to_array(data)
    n = len(x)
    if n < 2:
        return SenSlope(float("nan"), float("nan"))
    t = np.arange(n, dtype="float64")
    slopes = [
        (x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)
    ]
    if not slopes:
        return SenSlope(float("nan"), float("nan"))
    slope = float(np.median(slopes))
    intercept = float(np.median(x - slope * t))
    return SenSlope(slope, intercept)


def mann_kendall_modified(
    data: ArrayLike, *, alpha: float = DEFAULT_ALPHA
) -> MannKendallResult:
    """Hamed-Rao variance-corrected Mann-Kendall.

    Corrects the variance of S for lag autocorrelation in the detrended,
    rank-transformed series. For n < 10 it falls back to :func:`mann_kendall`.
    """
    x = _to_array(data)
    n = len(x)
    if n < 10:
        return mann_kendall(x, alpha=alpha)

    slope = sens_slope(x).slope
    detrended = x - np.arange(n) * slope
    ranks = scipy_stats.rankdata(detrended)
    s = _mk_s(x)
    var_s = _mk_var(x)

    n_ns = 1.0
    threshold = scipy_stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    mean_rank = ranks.mean()
    den = float(np.sum((ranks - mean_rank) ** 2))
    for lag in range(1, n - 3):
        num = float(np.sum((ranks[: n - lag] - mean_rank) * (ranks[lag:] - mean_rank)))
        rho = num / den if den != 0 else 0.0
        if abs(rho) >= threshold:
            n_ns += (
                2 * (n - lag) * (n - lag - 1) * (n - lag - 2) * rho
                / (n * (n - 1) * (n - 2))
            )
    z, p = _z_and_p(s, var_s * n_ns)
    return MannKendallResult(_trend_label(p, s, alpha), p, z, s)


def linear_regression(data: ArrayLike) -> LinearFit:
    """OLS of the series against time index (slope, intercept, R^2, p)."""
    y = _to_array(data)
    n = len(y)
    if n < 2:
        return LinearFit(float("nan"), float("nan"), float("nan"), float("nan"))
    t = np.arange(n, dtype="float64")
    t_bar, y_bar = t.mean(), y.mean()
    sxy = float(((t - t_bar) * (y - y_bar)).sum())
    sxx = float(((t - t_bar) ** 2).sum())
    sst = float(((y - y_bar) ** 2).sum())
    if sxx == 0:
        return LinearFit(float("nan"), float("nan"), float("nan"), float("nan"))
    slope = sxy / sxx
    intercept = y_bar - slope * t_bar
    sse = float(((y - (slope * t + intercept)) ** 2).sum())
    r2 = max(0.0, 1 - sse / sst) if sst > 0 else 0.0
    se = np.sqrt(sse / max(n - 2, 1) / sxx)
    t_stat = slope / se if se > 0 else 0.0
    p = 2.0 * float(scipy_stats.t.sf(abs(t_stat), df=n - 2))
    return LinearFit(float(slope), float(intercept), float(r2), p)


def log_linear_regression(data: ArrayLike) -> LinearFit:
    """OLS on the natural log of the positive values (needs >= 4 positives)."""
    x = _to_array(data)
    positive = x[x > 0]
    if positive.size < 4:
        return LinearFit(float("nan"), float("nan"), float("nan"), float("nan"))
    return linear_regression(np.log(positive))


def percent_slope(slope: float, mean: float) -> float:
    """Express a per-step slope as a percentage of the mean."""
    if np.isfinite(slope) and np.isfinite(mean) and mean != 0:
        return slope / mean * 100.0
    return float("nan")


def moving_average_trend(
    data: ArrayLike,
    *,
    window: int = MOVING_AVERAGE_WINDOW,
    alpha: float = DEFAULT_ALPHA,
) -> MannKendallResult | None:
    """Mann-Kendall trend of the backward moving average.

    Returns ``None`` when there are fewer than 3 smoothed points to test.
    """
    ma = (
        pd.to_numeric(pd.Series(data), errors="coerce")
        .rolling(window=window, min_periods=window)
        .mean()
        .dropna()
    )
    if len(ma) < 3:
        return None
    return mann_kendall(ma, alpha=alpha)