"""Change-point detection for :mod:`hydrotrends`.

Faithful, typed ports of the three change-point methods from the original
script:

* :func:`pettitt_test` — the rank-based Pettitt homogeneity test, returning the
  change-point position, the ``K`` statistic, and its (approximate) p-value.
* :func:`cusum_change_point` — the location of the largest cumulative deviation
  from the mean.
* :func:`bai_perron_change_point` — a single least-squares break: the split that
  minimises the within-segment sum of squares.

Each returns the change-point as a **position** in the (NaN-dropped) series;
mapping it to a calendar/hydrological year is the caller's job (it holds the
year axis). Series shorter than the method's minimum return ``None`` / ``NaN``
rather than raising, matching the soft-guard convention of :mod:`hydrotrends.stats`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from ..core.constants import BAI_PERRON_MIN_SIZE, BAI_PERRON_N_BREAKS
from ..core.utils import to_float_array
from ..core.validation import ArrayLike

__all__ = [
    "PettittResult",
    "pettitt_test",
    "cusum_change_point",
    "bai_perron_change_point",
]

_MIN_N = 5  # Pettitt / CUSUM need at least this many points


@dataclass(frozen=True)
class PettittResult:
    """Pettitt test outcome. ``index`` is ``None`` when the series is too short."""

    index: int | None
    k: float
    p_value: float


def pettitt_test(data: ArrayLike) -> PettittResult:
    """Rank-based Pettitt test for a single change point."""
    x = to_float_array(data)
    n = len(x)
    if n < _MIN_N:
        return PettittResult(None, float("nan"), float("nan"))
    ranks = scipy_stats.rankdata(x)
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    abs_u = np.abs(u)
    k = float(abs_u.max())
    cp = int(abs_u.argmax()) + 1
    if cp >= n:
        cp = n - 1
    p = min(2.0 * float(np.exp(-6 * k**2 / (n**3 + n**2))), 1.0)
    return PettittResult(cp, k, p)


def cusum_change_point(data: ArrayLike) -> int | None:
    """Position of the largest cumulative deviation from the mean (CUSUM)."""
    x = to_float_array(data)
    n = len(x)
    if n < _MIN_N:
        return None
    cusum = np.cumsum(x - x.mean())
    cp = int(np.argmax(np.abs(cusum))) + 1
    return n - 1 if cp >= n else cp


def bai_perron_change_point(
    data: ArrayLike,
    *,
    n_breaks: int = BAI_PERRON_N_BREAKS,
    min_size: int = BAI_PERRON_MIN_SIZE,
) -> int | None:
    """Single least-squares break: the split minimising within-segment SSE.

    Only a single break is supported (as in the source); ``n_breaks`` other than
    1 raises. Returns ``None`` when the series is shorter than ``2 * min_size``.
    """
    if n_breaks != 1:
        raise NotImplementedError("only a single break (n_breaks=1) is supported")
    x = to_float_array(data)
    n = len(x)
    if n < 2 * min_size:
        return None
    best_cost = np.inf
    best_t: int | None = None
    for t in range(min_size, n - min_size + 1):
        seg1, seg2 = x[:t], x[t:]
        cost = float(
            np.sum((seg1 - seg1.mean()) ** 2) + np.sum((seg2 - seg2.mean()) ** 2)
        )
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t