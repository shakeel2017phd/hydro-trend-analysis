"""Analysis orchestration for :mod:`hydrotrends`.

The compute layer that turns preprocessed frames into the results tables the
report and plotting layers consume. :func:`analyze_series` is the typed,
de-duplicated successor to the original ``full_analysis``: it runs the
descriptive statistics and every trend test on one series and returns a row
keyed by the report schema (:data:`hydrotrends.viz.reports.STAT_GROUPS`).
:func:`analyze_by_period` applies it across periods (the across-years series for
each fixed day/dekad), and :func:`analyze_preprocessed` wires it to a
:class:`~hydrotrends.data.preprocessing.PreprocessedData`, choosing the correct
period-label order for the resolution.

Design notes
------------
* Values are stored at **full precision**; rounding is a display concern handled
  by the report writer, so an exported results frame keeps its precision.
* Trend directions are stored as plain strings (``"increasing"`` ...), matching
  the report's colour-lookup keys.
* Change-point columns (Pettitt/CUSUM/Bai-Perron) are intentionally absent until
  a change-point module exists; they render blank in the report.
* The ``n < 4`` guard mirrors the source: too short for any trend test, so only
  the descriptive block is produced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

from .core.constants import (
    COL_HYDRO_YEAR,
    COL_PERIOD,
    COL_YEAR,
    DEFAULT_ALPHA,
    MOVING_AVERAGE_WINDOW,
    RESOLUTION_INFO,
)
from .core.utils import significance_stars, to_float_array
from .data.preprocessing import PreprocessedData
from .stats.changepoint import (
    bai_perron_change_point,
    cusum_change_point,
    pettitt_test,
)
from .stats.descriptive import describe, describe_by
from .stats.ita import innovative_trend_analysis
from .stats.trends import (
    linear_regression,
    log_linear_regression,
    mann_kendall,
    mann_kendall_modified,
    percent_slope,
    sens_slope,
)
from .viz.reports import write_cover_sheet, write_descriptive_sheet, write_results_sheet

__all__ = [
    "analyze_series",
    "analyze_by_period",
    "analyze_preprocessed",
    "ReportColumn",
    "generate_report",
]

_MIN_FOR_TRENDS = 4  # matches the source's n < 4 guard


def _reindex_to_order(df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Reindex ``df`` to ``order``, appending any periods not listed in it."""
    order_set = set(order)
    ordered = [p for p in order if p in df.index]
    extras = [p for p in df.index if p not in order_set]
    return df.reindex(ordered + extras)


def _change_point_keys(arr: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    """Change-point positions mapped to years, keyed for the report schema."""

    def to_year(index: int | None) -> Any:
        if index is None or index >= len(years):
            return math.nan
        return int(years[index])

    pettitt = pettitt_test(arr)
    return {
        "pettitt_cp_year": to_year(pettitt.index),
        "pettitt_K": pettitt.k,
        "pettitt_p": pettitt.p_value,
        "pettitt_sig": significance_stars(pettitt.p_value),
        "cusum_cp_year": to_year(cusum_change_point(arr)),
        "bpcp_year": to_year(bai_perron_change_point(arr)),
    }


def analyze_series(
    values: Any,
    *,
    years: Any = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Descriptive stats + every trend test for one series, as a report-keyed row.

    Below :data:`_MIN_FOR_TRENDS` valid points only the descriptive block is
    returned (the trend keys are absent and render blank downstream). When
    ``years`` is given (aligned to ``values``), change-point results are mapped
    to years; otherwise the positional index is used as a fallback.
    """
    arr = to_float_array(values)
    n = len(arr)
    d = describe(arr)
    row: dict[str, Any] = {
        "n": d.n,
        "mean": d.mean,
        "median": d.median,
        "std": d.std,
        "cv_pct": d.cv_pct,
        "min": d.minimum,
        "max": d.maximum,
        "p10": d.p10,
        "p25": d.p25,
        "p75": d.p75,
        "p90": d.p90,
        "skew": d.skewness,
        "kurt": d.kurtosis,
    }
    if n < _MIN_FOR_TRENDS:
        return row

    mk = mann_kendall(arr, alpha=alpha)
    sen = sens_slope(arr)
    row.update(
        {
            "trend": str(mk.trend),
            "p_value": mk.p_value,
            "z_score": mk.z_score,
            "sens_slope": sen.slope,
            "sens_slope_pct": percent_slope(sen.slope, d.mean),
            "significance": significance_stars(mk.p_value),
        }
    )

    mmk = mann_kendall_modified(arr, alpha=alpha)
    row.update(
        {
            "mmk_trend": str(mmk.trend),
            "mmk_p": mmk.p_value,
            "mmk_z": mmk.z_score,
            "mmk_sig": significance_stars(mmk.p_value),
        }
    )

    lin = linear_regression(arr)
    row.update(
        {
            "lin_slope": lin.slope,
            "lin_intercept": lin.intercept,
            "lin_r2": lin.r2,
            "lin_p": lin.p_value,
        }
    )

    log = log_linear_regression(arr)
    row.update({"log_slope": log.slope, "log_r2": log.r2, "log_p": log.p_value})

    ma = (
        pd.Series(arr)
        .rolling(window=MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_WINDOW)
        .mean()
        .dropna()
    )
    if len(ma) >= 3:
        row.update(
            {
                "ma5_mean": float(ma.mean()),
                "ma5_std": float(ma.std(ddof=1)),
                "ma5_trend": str(mann_kendall(ma.to_numpy(), alpha=alpha).trend),
            }
        )

    ita = innovative_trend_analysis(arr)
    row.update({"ita_slope": ita.slope, "ita_trend": str(ita.trend)})

    year_axis = (
        np.asarray(years)[:n]
        if (years is not None and len(np.asarray(years)) >= n)
        else np.arange(n)
    )
    row.update(_change_point_keys(arr, year_axis))
    return row


def analyze_by_period(
    df: pd.DataFrame,
    *,
    value_col: str,
    period_col: str = COL_PERIOD,
    year_col: str = COL_HYDRO_YEAR,
    order: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Run :func:`analyze_series` on each period's across-years series.

    For each ``period_col`` value, the series is that period's observations over
    all years, sorted by ``year_col`` (chronological order matters for the trend
    tests). ``order`` reindexes the result (e.g. to hydrological-month order);
    any periods present in the data but absent from ``order`` are appended.
    """
    rows: dict[str, dict[str, Any]] = {}
    for period, group in df.groupby(period_col, sort=False):
        ordered_group = group.sort_values(year_col)
        rows[str(period)] = analyze_series(
            ordered_group[value_col],
            years=ordered_group[year_col].to_numpy(),
            alpha=alpha,
        )

    result = pd.DataFrame.from_dict(rows, orient="index")
    if order is not None:
        result = _reindex_to_order(result, order)
    result.index.name = "Period"
    return result


def analyze_preprocessed(
    pre: PreprocessedData,
    *,
    value_col: str,
    calendar: bool = False,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Per-period analysis of a preprocessed input, in the right period order.

    Uses the hydrological-year frame by default (``calendar=True`` for the
    calendar-year frame), and picks the matching period-label order and year
    column for the input's resolution.
    """
    frame = pre.calendar if calendar else pre.hydro
    info = RESOLUTION_INFO[pre.resolution]
    order = list(info.cal_periods if calendar else info.hydro_periods)
    year_col = COL_YEAR if calendar else COL_HYDRO_YEAR
    return analyze_by_period(
        frame,
        value_col=value_col,
        period_col=COL_PERIOD,
        year_col=year_col,
        order=order,
        alpha=alpha,
    )


@dataclass(frozen=True)
class ReportColumn:
    """One column to analyse in a report, with its display unit label."""

    column: str  # DataFrame column, e.g. COL_FLOW_CUSECS
    unit_label: str  # e.g. "Cusecs", "Cumecs", "MAF"


def generate_report(
    pre: PreprocessedData,
    output_path: str | Path,
    *,
    columns: Sequence[ReportColumn],
    title: str = "Hydrological Trend Analysis",
    subtitle: str = "",
    calendar: bool = False,
    include_descriptive: bool = True,
    alpha: float = DEFAULT_ALPHA,
) -> Path:
    """Build a complete .xlsx report from a preprocessed input.

    Writes a cover sheet, then for each column a trend-analysis sheet (and, when
    ``include_descriptive`` is set, a descriptive-statistics sheet). Periods are
    ordered by the resolution's hydrological (or calendar) sequence. Returns the
    output path.
    """
    if not columns:
        raise ValueError("generate_report needs at least one ReportColumn")

    frame = pre.calendar if calendar else pre.hydro
    info = RESOLUTION_INFO[pre.resolution]
    order = list(info.cal_periods if calendar else info.hydro_periods)

    wb = Workbook()
    default = wb.active
    write_cover_sheet(wb.create_sheet("Cover"), title=title, subtitle=subtitle)

    for rc in columns:
        trends = analyze_preprocessed(
            pre, value_col=rc.column, calendar=calendar, alpha=alpha
        )
        write_results_sheet(
            wb.create_sheet(f"Trends ({rc.unit_label})"),
            trends,
            title=f"{title} — {rc.unit_label}",
            index_label="Period",
            unit=rc.unit_label,
        )
        if include_descriptive:
            desc = _reindex_to_order(
                describe_by(frame, value_col=rc.column, by=COL_PERIOD), order
            )
            write_descriptive_sheet(
                wb.create_sheet(f"Descriptive ({rc.unit_label})"),
                desc,
                title=f"Descriptive Statistics — {rc.unit_label}",
                index_label="Period",
                unit=rc.unit_label,
            )

    wb.remove(default)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
