"""Extended descriptive statistics for the Phase-3 Descriptive Statistics
Summary (:mod:`hydrotrends`).

A superset of :func:`~hydrotrends.stats.descriptive.describe`, covering Data
Quality, Central Tendency, Dispersion, Estimation Uncertainty, Extremes,
Quantiles, Distribution Shape, Flow Duration, and Totals. Kept in its own
module (rather than extending ``describe``) because it is a distinct, newer
feature with its own formula choices, and ``describe`` backs the
v26-parity-tested Trends/Data sheets that must stay bit-for-bit stable.

Formula notes (confirmed choices, not arbitrary defaults):

* Weighted Mean is omitted: for equally-spaced observations it equals the
  arithmetic mean, so it would be a redundant column.
* ``std``/``variance``/``cv_pct`` use the **sample** (``ddof=1``) estimator,
  matching :func:`~hydrotrends.stats.descriptive.describe`.
* Geometric Mean / Harmonic Mean are computed over **strictly positive**
  values only (undefined for zero/negative flow); ``NaN`` if none are positive.
* No-Flow Count (``NF``) and Negative Count (``NNeg``) are tracked
  separately -- a zero-flow day is not a data error, a negative one usually is.
* Record Completeness is ``NaN`` unless the caller supplies ``expected_n``
  (the year-type/resolution-specific expected span), since that denominator
  isn't derivable from the values alone.
* Last-N-Years Mean is the mean of the last ``last_n_window`` *valid*
  observations of ``data``, in the order given -- meaningful only when the
  caller passes a chronologically-ordered series (e.g. one period's value
  across years); ``NaN`` if ``last_n_window`` is ``None``.
* Confidence interval is the normal-theory 95% interval
  (``mean +/- 1.96 * standard error``).
* Pearson's 2nd skewness coefficient is ``3 * (mean - median) / std``;
  Bowley's skewness (Yule-Kendall index) is ``(P75 + P25 - 2*median) / IQR``;
  L-skewness is the L-moment ratio ``l3 / l2`` (Hosking's unbiased sample
  L-moments via probability-weighted moments).
* Shapiro-Wilk and Anderson-Darling are normality-diagnostic fields, not
  formal hypothesis-test gates -- report the raw statistic(s) for the reader
  to interpret, since "is flow normally distributed" is rarely a yes/no
  question worth hard-coding a significance threshold for.
* Flow Duration ``Q90``/``Q95``/``Q99`` are exceedance flows (the flow equalled
  or exceeded 90%/95%/99% of the time), i.e. the 10th/5th/1st percentile of
  the value distribution -- not a typo of P90 (data quantile).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..core.constants import REPORTED_PERCENTILES
from ..core.validation import ArrayLike

__all__ = [
    "ExtendedDescriptiveStats",
    "describe_extended",
    "describe_extended_by",
    "EXTENDED_STATS_COLUMNS",
]

_CI_Z = 1.96  # normal-theory 95% CI multiplier
_FLOW_DURATION_PERCENTILES = (10, 5, 1)  # -> Q90, Q95, Q99

# Field name -> pretty label, in report column order. Public so the report
# layer (viz.reports) can build a matching table without duplicating the
# field list -- mirrors reports._DESCRIPTIVE_COLUMNS's role for `describe`.
EXTENDED_STATS_COLUMNS: tuple[tuple[str, str], ...] = (
    # Data Quality
    ("n", "N"),
    ("missing", "Missing Count"),
    ("no_flow_count", "NF (No-Flow Count)"),
    ("negative_count", "NNeg (Negative Count)"),
    ("completeness_pct", "Record Completeness (%)"),
    # Central Tendency
    ("mean", "Mean"),
    ("median", "Median"),
    ("mode", "Mode"),
    ("geometric_mean", "Geometric Mean (GM)"),
    ("harmonic_mean", "Harmonic Mean (HM)"),
    # Dispersion
    ("std", "Std Dev"),
    ("variance", "Variance"),
    ("cv_pct", "CV (%)"),
    ("iqr", "IQR"),
    ("value_range", "Range"),
    ("mad", "MAD"),
    # Estimation Uncertainty
    ("se", "Standard Error"),
    ("ci_lower", "95% CI Lower"),
    ("ci_upper", "95% CI Upper"),
    # Extremes
    ("minimum", "Min"),
    ("maximum", "Max"),
    ("last_n_mean", "Last N-Years Mean"),
    # Quantiles
    ("p10", "P10"),
    ("p25", "P25"),
    ("p75", "P75"),
    ("p90", "P90"),
    # Distribution Shape
    ("skewness", "Skewness"),
    ("kurtosis", "Kurtosis"),
    ("pearson_skew", "Pearson's 2nd Skewness"),
    ("bowley_skew", "Bowley's Skewness"),
    ("l_skewness", "L-Skewness"),
    ("shapiro_stat", "Shapiro-Wilk Statistic"),
    ("shapiro_p", "Shapiro-Wilk p-value"),
    ("anderson_stat", "Anderson-Darling Statistic"),
    # Flow Duration
    ("q90", "Q90"),
    ("q95", "Q95"),
    ("q99", "Q99"),
    # Totals
    ("total", "Sum (Volume)"),
)
_LABELS: dict[str, str] = dict(EXTENDED_STATS_COLUMNS)


@dataclass(frozen=True)
class ExtendedDescriptiveStats:
    """The Phase-3 Descriptive Statistics Summary fields for one series."""

    # Data Quality
    n: int
    missing: int
    no_flow_count: int
    negative_count: int
    completeness_pct: float
    # Central Tendency
    mean: float
    median: float
    mode: float
    geometric_mean: float
    harmonic_mean: float
    # Dispersion
    std: float
    variance: float
    cv_pct: float
    iqr: float
    value_range: float
    mad: float
    # Estimation Uncertainty
    se: float
    ci_lower: float
    ci_upper: float
    # Extremes
    minimum: float
    maximum: float
    last_n_mean: float
    # Quantiles
    p10: float
    p25: float
    p75: float
    p90: float
    # Distribution Shape
    skewness: float
    kurtosis: float
    pearson_skew: float
    bowley_skew: float
    l_skewness: float
    shapiro_stat: float
    shapiro_p: float
    anderson_stat: float
    # Flow Duration
    q90: float
    q95: float
    q99: float
    # Totals
    total: float

    def to_dict(self) -> dict[str, float]:
        """Field-name -> value mapping."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Pretty-labelled Series (report column order), handy for tables."""
        data = asdict(self)
        return pd.Series({label: data[field] for field, label in _LABELS.items()})


def _mode(arr: np.ndarray, ndigits: int) -> float:
    """Most frequent rounded value; falls back to a KDE peak when every
    value is distinct (rounding produced no repeats to count)."""
    rounded = np.round(arr, ndigits)
    values, counts = np.unique(rounded, return_counts=True)
    if counts.max() > 1:
        return float(values[np.argmax(counts)])
    if len(arr) < 2 or np.ptp(arr) == 0:
        return float(arr[0])
    try:
        kde = scipy_stats.gaussian_kde(arr)
    except np.linalg.LinAlgError:
        return float(values[np.argmax(counts)])
    grid = np.linspace(arr.min(), arr.max(), 512)
    return float(grid[np.argmax(kde(grid))])


def _geometric_mean(positive: np.ndarray) -> float:
    if positive.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(positive))))


def _harmonic_mean(positive: np.ndarray) -> float:
    if positive.size == 0:
        return float("nan")
    return float(positive.size / np.sum(1.0 / positive))


def _l_skewness(sorted_arr: np.ndarray) -> float:
    """L-skewness (tau3 = l3/l2) via Hosking's sample L-moments (unbiased PWMs)."""
    n = len(sorted_arr)
    if n < 3:
        return float("nan")
    i = np.arange(1, n + 1, dtype="float64")
    b0 = float(np.mean(sorted_arr))
    b1 = float(np.sum((i - 1) * sorted_arr) / (n * (n - 1)))
    b2 = float(np.sum((i - 1) * (i - 2) * sorted_arr) / (n * (n - 1) * (n - 2)))
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return l3 / l2 if l2 != 0 else float("nan")


def _empty(missing: int) -> ExtendedDescriptiveStats:
    nan = float("nan")
    return ExtendedDescriptiveStats(
        n=0,
        missing=missing,
        no_flow_count=0,
        negative_count=0,
        completeness_pct=nan,
        mean=nan,
        median=nan,
        mode=nan,
        geometric_mean=nan,
        harmonic_mean=nan,
        std=nan,
        variance=nan,
        cv_pct=nan,
        iqr=nan,
        value_range=nan,
        mad=nan,
        se=nan,
        ci_lower=nan,
        ci_upper=nan,
        minimum=nan,
        maximum=nan,
        last_n_mean=nan,
        p10=nan,
        p25=nan,
        p75=nan,
        p90=nan,
        skewness=nan,
        kurtosis=nan,
        pearson_skew=nan,
        bowley_skew=nan,
        l_skewness=nan,
        shapiro_stat=nan,
        shapiro_p=nan,
        anderson_stat=nan,
        q90=nan,
        q95=nan,
        q99=nan,
        total=nan,
    )


def describe_extended(
    data: ArrayLike,
    *,
    expected_n: int | None = None,
    last_n_window: int | None = None,
    mode_ndigits: int = 2,
) -> ExtendedDescriptiveStats:
    """Compute the Phase-3 Descriptive Statistics Summary for ``data``.

    ``data`` may contain NaNs (dropped for every statistic except ``missing``,
    which counts them). ``expected_n`` is the year-type/resolution-specific
    expected observation count (e.g. 366 for a hydro year of daily data);
    omit it to leave Record Completeness as ``NaN``. ``last_n_window`` (e.g.
    5) computes the mean of the last N valid observations *in the order
    ``data`` is given* -- pass a chronologically-ordered series (one period's
    values across years) for this to be meaningful; omit it for a
    within-year (across-periods) series, where "last N years" doesn't apply.
    """
    raw = pd.Series(data)
    numeric = pd.to_numeric(raw, errors="coerce")
    valid = numeric.dropna()
    missing = int(numeric.isna().sum())
    n = int(valid.size)
    if n == 0:
        return _empty(missing)

    arr = valid.to_numpy(dtype="float64")
    sorted_arr = np.sort(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    variance = float(arr.var(ddof=1)) if n >= 2 else float("nan")
    cv_pct = (std / mean * 100.0) if mean != 0 and np.isfinite(mean) else float("nan")

    p10, p25, p75, p90 = (
        float(x) for x in np.percentile(arr, list(REPORTED_PERCENTILES))
    )
    q90, q95, q99 = (
        float(x) for x in np.percentile(arr, list(_FLOW_DURATION_PERCENTILES))
    )
    minimum = float(arr.min())
    maximum = float(arr.max())
    iqr = p75 - p25

    positive = arr[arr > 0]
    se = float(std / np.sqrt(n)) if n >= 2 and np.isfinite(std) else float("nan")
    ci_lower = mean - _CI_Z * se if np.isfinite(se) else float("nan")
    ci_upper = mean + _CI_Z * se if np.isfinite(se) else float("nan")

    if last_n_window is not None and last_n_window > 0:
        tail = arr[-last_n_window:]
        last_n_mean = float(tail.mean()) if tail.size else float("nan")
    else:
        last_n_mean = float("nan")

    completeness_pct = (
        n / expected_n * 100.0 if expected_n and expected_n > 0 else float("nan")
    )

    # skew/kurtosis/normality diagnostics on tiny, near-constant samples can
    # trigger numerical RuntimeWarnings (and SciPy warns on zero-range input
    # to shapiro, and on its own anderson() p-value-method deprecation) --
    # the size/range guards already flag those limitations, so suppress the
    # noise here rather than letting it flood the logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        skewness = float(scipy_stats.skew(arr, bias=False)) if n >= 3 else float("nan")
        kurtosis = (
            float(scipy_stats.kurtosis(arr, bias=False)) if n >= 4 else float("nan")
        )
        pearson_skew = (
            3 * (mean - median) / std if n >= 2 and std != 0 else float("nan")
        )
        bowley_skew = (p75 + p25 - 2 * median) / iqr if iqr != 0 else float("nan")
        l_skewness = _l_skewness(sorted_arr)
        if n >= 3:
            shapiro_result = scipy_stats.shapiro(arr)
            shapiro_stat = float(shapiro_result.statistic)
            shapiro_p = float(shapiro_result.pvalue)
        else:
            shapiro_stat = float("nan")
            shapiro_p = float("nan")
        try:
            anderson_stat = float(scipy_stats.anderson(arr, dist="norm").statistic)
        except (ValueError, np.linalg.LinAlgError):
            anderson_stat = float("nan")

    return ExtendedDescriptiveStats(
        n=n,
        missing=missing,
        no_flow_count=int(np.sum(arr == 0)),
        negative_count=int(np.sum(arr < 0)),
        completeness_pct=completeness_pct,
        mean=mean,
        median=median,
        mode=_mode(arr, mode_ndigits),
        geometric_mean=_geometric_mean(positive),
        harmonic_mean=_harmonic_mean(positive),
        std=std,
        variance=variance,
        cv_pct=cv_pct,
        iqr=iqr,
        value_range=maximum - minimum,
        mad=float(np.mean(np.abs(arr - mean))),
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        minimum=minimum,
        maximum=maximum,
        last_n_mean=last_n_mean,
        p10=p10,
        p25=p25,
        p75=p75,
        p90=p90,
        skewness=skewness,
        kurtosis=kurtosis,
        pearson_skew=pearson_skew,
        bowley_skew=bowley_skew,
        l_skewness=l_skewness,
        shapiro_stat=shapiro_stat,
        shapiro_p=shapiro_p,
        anderson_stat=anderson_stat,
        q90=q90,
        q95=q95,
        q99=q99,
        total=float(arr.sum()),
    )


def describe_extended_by(
    df: pd.DataFrame,
    *,
    value_col: str,
    by: str,
    sort_col: str | None = None,
    expected_n: int | None = None,
    last_n_window: int | None = None,
) -> pd.DataFrame:
    """:func:`describe_extended` of ``value_col`` for each group in ``by``.

    Mirrors :func:`~hydrotrends.stats.descriptive.describe_by`, with the full
    Phase-3 field set. ``expected_n``/``last_n_window`` are passed through to
    every group unchanged (see :func:`describe_extended` for what each means)
    -- a Horizontal summary (one row per year, across that year's periods)
    typically sets ``expected_n`` to the period count and leaves
    ``last_n_window`` unset; a Vertical summary (one row per period, across
    years) typically sets ``last_n_window`` and passes ``sort_col`` (a year
    column) so "last N" means the most recent years, not row order.
    """
    summaries: dict[object, dict[str, float]] = {}
    for key, group in df.groupby(by, sort=True):
        ordered = group.sort_values(sort_col) if sort_col else group
        summaries[key] = describe_extended(
            ordered[value_col], expected_n=expected_n, last_n_window=last_n_window
        ).to_dict()
    return pd.DataFrame.from_dict(summaries, orient="index")
