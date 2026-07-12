"""Flow/volume duration and exceedance analysis for :mod:`hydrotrends`.

Reproduces the original script's frequency tools exactly:

* :func:`flow_duration_curve` — values sorted descending with the **Weibull**
  plotting-position exceedance probability ``rank / (n + 1) * 100``.
* :func:`exceedance_counts` — how many observations meet or exceed each flood
  threshold (LF/MF/HF/VHF/EHF).
* :func:`exceedance_probability` — the % of time a single threshold is met or
  exceeded (used to annotate flood lines on the duration curve).

Added on top (standard hydrological indices the script didn't expose):

* :func:`flow_percentiles` — the ``Q_p`` flows, i.e. the value exceeded ``p`` %
  of the time. By the exceedance convention ``Q_p = percentile(flows, 100 - p)``
  (same percentile method as :mod:`hydrotrends.stats.descriptive`), so ``Q50`` is
  the median, ``Q90`` a low-flow index, and ``Q10`` a high-flow index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.utils import to_float_array
from ..core.validation import ArrayLike

__all__ = [
    "COL_FDC_VALUE",
    "COL_FDC_RANK",
    "COL_FDC_EXCEEDANCE",
    "flow_duration_curve",
    "exceedance_counts",
    "exceedance_probability",
    "flow_percentiles",
]

# Column names for the flow-duration-curve frame (FDC-local, not the main schema).
COL_FDC_VALUE = "Value"
COL_FDC_RANK = "Rank"
COL_FDC_EXCEEDANCE = "Exceedance_Probability"

_DEFAULT_EXCEEDANCES: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)


def flow_duration_curve(data: ArrayLike) -> pd.DataFrame:
    """Flow/volume duration curve: values descending with Weibull exceedance %.

    Returns a frame with columns ``Value`` (descending), ``Rank`` (1..n), and
    ``Exceedance_Probability`` (``rank / (n + 1) * 100``).
    """
    s = (
        pd.to_numeric(pd.Series(data), errors="coerce")
        .dropna()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    n = int(s.size)
    ranks = np.arange(1, n + 1)
    exceedance = ranks / (n + 1) * 100.0 if n else np.empty(0, dtype="float64")
    return pd.DataFrame(
        {
            COL_FDC_VALUE: s.to_numpy(dtype="float64"),
            COL_FDC_RANK: ranks,
            COL_FDC_EXCEEDANCE: exceedance,
        }
    )


def exceedance_counts(
    data: ArrayLike,
    thresholds: Mapping[str, float],
) -> dict[str, int]:
    """Count observations meeting or exceeding each named threshold.

    ``thresholds`` maps a label (e.g. flood class) to a value *in the same unit
    as ``data``* — convert flood limits to the active flow unit before calling.
    """
    values = to_float_array(data)
    return {
        label: int((values >= threshold).sum())
        for label, threshold in thresholds.items()
    }


def exceedance_probability(data: ArrayLike, threshold: float) -> float:
    """Percentage of observations meeting or exceeding ``threshold``."""
    values = to_float_array(data)
    if values.size == 0:
        return float("nan")
    return float((values >= threshold).mean() * 100.0)


def flow_percentiles(
    data: ArrayLike,
    exceedances: Sequence[int] = _DEFAULT_EXCEEDANCES,
) -> dict[str, float]:
    """``Q_p`` flow indices: the value exceeded ``p`` % of the time.

    Keyed ``"Q{p}"``; ``Q_p = percentile(values, 100 - p)``.
    """
    values = to_float_array(data)
    if values.size == 0:
        return {f"Q{p}": float("nan") for p in exceedances}
    return {f"Q{p}": float(np.percentile(values, 100 - p)) for p in exceedances}
