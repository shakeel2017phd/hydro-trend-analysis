"""Descriptive statistics for :mod:`hydrotrends`.

Replaces the original script's ``HSTAT_LABELS`` list of bare lambdas with a
single, named, typed :func:`describe` returning a :class:`DescriptiveStats`
record. Formulas match the source exactly so numbers are reproducible:

* ``std`` and ``cv`` use the **sample** standard deviation (``ddof=1``);
* ``cv`` (%) is ``std / mean * 100`` and is ``NaN`` when the mean is 0;
* percentiles use :func:`numpy.percentile` (linear interpolation);
* ``skewness``/``kurtosis`` use SciPy with ``bias=False`` (bias-corrected).

Degenerate inputs degrade gracefully rather than raising: statistics that need
a minimum sample size return ``NaN`` (std needs >= 2 points, skewness >= 3,
kurtosis >= 4), and an all-empty input yields ``n = 0`` with ``NaN`` fields.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..core.constants import REPORTED_PERCENTILES
from ..core.validation import ArrayLike

__all__ = ["DescriptiveStats", "describe", "describe_by"]

# Pretty labels (source order) for tabular output.
_LABELS: dict[str, str] = {
    "n": "N",
    "mean": "Mean",
    "median": "Median",
    "std": "Std Dev",
    "cv_pct": "CV (%)",
    "minimum": "Min",
    "p10": "P10",
    "p25": "P25",
    "p75": "P75",
    "p90": "P90",
    "maximum": "Max",
    "iqr": "IQR",
    "value_range": "Range",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "total": "Sum",
}


@dataclass(frozen=True)
class DescriptiveStats:
    """A full descriptive summary of one numeric series."""

    n: int
    mean: float
    median: float
    std: float
    cv_pct: float
    minimum: float
    p10: float
    p25: float
    p75: float
    p90: float
    maximum: float
    iqr: float
    value_range: float
    skewness: float
    kurtosis: float
    total: float

    def to_dict(self) -> dict[str, float]:
        """Field-name -> value mapping."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Pretty-labelled Series (source column order), handy for tables."""
        data = asdict(self)
        return pd.Series({label: data[field] for field, label in _LABELS.items()})


def _empty() -> DescriptiveStats:
    nan = float("nan")
    return DescriptiveStats(
        n=0,
        mean=nan,
        median=nan,
        std=nan,
        cv_pct=nan,
        minimum=nan,
        p10=nan,
        p25=nan,
        p75=nan,
        p90=nan,
        maximum=nan,
        iqr=nan,
        value_range=nan,
        skewness=nan,
        kurtosis=nan,
        total=nan,
    )


def describe(data: ArrayLike) -> DescriptiveStats:
    """Compute descriptive statistics for ``data`` (NaNs are dropped)."""
    s = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    n = int(s.size)
    if n == 0:
        return _empty()

    arr = s.to_numpy(dtype="float64")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    cv_pct = (std / mean * 100.0) if mean != 0 and np.isfinite(mean) else float("nan")

    p10, p25, p75, p90 = (
        float(x) for x in np.percentile(arr, list(REPORTED_PERCENTILES))
    )
    minimum = float(arr.min())
    maximum = float(arr.max())

    # skew/kurtosis on tiny, near-constant samples can trigger a numerical
    # "catastrophic cancellation" RuntimeWarning from SciPy. The n>=3 / n>=4
    # guards already flag the small-sample limitation, so suppress just that
    # narrow warning here rather than letting it flood the logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skewness = float(scipy_stats.skew(arr, bias=False)) if n >= 3 else float("nan")
        kurtosis = (
            float(scipy_stats.kurtosis(arr, bias=False)) if n >= 4 else float("nan")
        )

    return DescriptiveStats(
        n=n,
        mean=mean,
        median=float(np.median(arr)),
        std=std,
        cv_pct=cv_pct,
        minimum=minimum,
        p10=p10,
        p25=p25,
        p75=p75,
        p90=p90,
        maximum=maximum,
        iqr=p75 - p25,
        value_range=maximum - minimum,
        skewness=skewness,
        kurtosis=kurtosis,
        total=float(arr.sum()),
    )


def describe_by(
    df: pd.DataFrame,
    *,
    value_col: str,
    by: str,
) -> pd.DataFrame:
    """Descriptive statistics of ``value_col`` for each group in ``by``.

    Returns a DataFrame indexed by group value, one column per statistic — the
    building block for per-year / per-month / per-season summary tables.
    """
    summaries = {
        key: describe(group[value_col]).to_dict()
        for key, group in df.groupby(by, sort=True)
    }
    return pd.DataFrame.from_dict(summaries, orient="index")
