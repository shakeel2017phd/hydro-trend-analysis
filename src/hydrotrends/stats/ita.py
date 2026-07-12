"""Innovative Trend Analysis (ITA) for :mod:`hydrotrends`.

Şen's method: split the series in time into two equal halves, and compare them.
The aggregate slope indicator reproduces the original script exactly::

    slope = 2 * (mean(second_half) - mean(first_half)) / n

(an odd-length series drops its first point; a slope within ``ITA_SLOPE_EPS`` of
zero is reported as no trend).

On top of the aggregate, this port adds the standard ITA refinement that the
script omitted: sorting each half and comparing by magnitude rank, then
reporting sub-trends for the **low / medium / high** thirds. This matters in
hydrology because low flows often trend opposite to high flows — a distinction
the single aggregate slope hides. The sorted halves are also returned so
``viz`` can draw the classic ITA scatter against the 1:1 line. Because the
aggregate uses means (which are sort-invariant), adding the sorted analysis does
not change it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import ITA_SLOPE_EPS, TrendDirection
from ..core.utils import to_float_array
from ..core.validation import ArrayLike

__all__ = ["ITAResult", "innovative_trend_analysis"]


@dataclass(frozen=True)
class ITAResult:
    """Result of an Innovative Trend Analysis.

    ``first_half``/``second_half`` are the two time-halves **sorted ascending**
    (paired by rank for the scatter and the sub-trends). The sub-trend slopes
    are ``NaN`` when there are too few points to form three groups (n < 6).
    """

    slope: float
    trend: TrendDirection
    n_used: int
    first_half: np.ndarray
    second_half: np.ndarray
    low_slope: float
    medium_slope: float
    high_slope: float


def _label(slope: float) -> TrendDirection:
    if slope > ITA_SLOPE_EPS:
        return TrendDirection.INCREASING
    if slope < -ITA_SLOPE_EPS:
        return TrendDirection.DECREASING
    return TrendDirection.NO_TREND


def _empty() -> ITAResult:
    empty = np.empty(0, dtype="float64")
    nan = float("nan")
    return ITAResult(
        slope=nan, trend=TrendDirection.NO_TREND, n_used=0,
        first_half=empty, second_half=empty,
        low_slope=nan, medium_slope=nan, high_slope=nan,
    )


def innovative_trend_analysis(data: ArrayLike) -> ITAResult:
    """Compute the ITA aggregate slope, trend, and low/medium/high sub-trends."""
    x = to_float_array(data)
    n = len(x)
    if n < 4:
        return _empty()
    if n % 2 != 0:  # drop the first point so the halves are equal (v26 rule)
        x = x[1:]
        n = len(x)

    half = n // 2
    first, second = x[:half], x[half:]
    slope = 2.0 * (float(second.mean()) - float(first.mean())) / n

    xs, ys = np.sort(first), np.sort(second)
    if half >= 3:
        groups = np.array_split(np.arange(half), 3)
        low, medium, high = (
            float(2.0 * (ys[g].mean() - xs[g].mean()) / n) for g in groups
        )
    else:
        low = medium = high = float("nan")

    return ITAResult(
        slope=slope,
        trend=_label(slope),
        n_used=n,
        first_half=xs,
        second_half=ys,
        low_slope=low,
        medium_slope=medium,
        high_slope=high,
    )