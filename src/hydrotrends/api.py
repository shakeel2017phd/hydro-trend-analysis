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

from .core.config import SeasonScheme
from .core.constants import (
    COL_HYDRO_YEAR,
    COL_PERIOD,
    COL_YEAR,
    DEFAULT_ALPHA,
    HYDRO_MONTHS,
    MOVING_AVERAGE_WINDOW,
    RESOLUTION_INFO,
)
from .core.utils import significance_stars, to_float_array
from .data.preprocessing import (
    PreprocessedData,
    hydro_seasonal_volumes,
    met_seasonal_volumes,
    monthly_volumes,
)
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
    "analyze_monthly_volumes",
    "analyze_hydro_seasonal_volumes",
    "analyze_met_seasonal_volumes",
    "describe_monthly_volumes",
    "describe_hydro_seasonal_volumes",
    "describe_met_seasonal_volumes",
    "ReportColumn",
    "generate_report",
]

# Hydrological (cropping) season/annual rollups in report order, with the
# source's display labels (``build_workbook``'s ``slabels`` / ``season_order``).
# Named ``_HYDRO_SEASON_*`` to keep them unambiguous next to the
# ``_MET_SEASON_*`` (meteorological) rollups below — the source script only
# ever means Kharif/Rabi by "season".
_HYDRO_SEASON_ORDER: tuple[str, ...] = (
    "Early_Kharif",
    "Late_Kharif",
    "Kharif",
    "Rabi",
    "Annual",
)
_HYDRO_SEASON_LABELS: dict[str, str] = {
    "Early_Kharif": "Early Kharif  (Apr1–Jun10)",
    "Late_Kharif": "Late Kharif   (Jun11–Sep30)",
    "Kharif": "Kharif        (Apr1–Sep30)",
    "Rabi": "Rabi          (Oct1–Mar31)",
    "Annual": "Annual        (Apr1–Mar31)",
}

# Meteorological seasons (calendar-based; see met_seasonal_volumes for the
# Winter year-boundary convention), in calendar order.
_MET_SEASON_ORDER: tuple[str, ...] = ("Winter", "Spring", "Summer", "Monsoon", "Autumn")
_MET_SEASON_LABELS: dict[str, str] = {
    "Winter": "Winter   (Dec–Feb)",
    "Spring": "Spring   (Mar–Apr)",
    "Summer": "Summer   (May–Jun)",
    "Monsoon": "Monsoon  (Jul–Sep)",
    "Autumn": "Autumn   (Oct–Nov)",
}

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
    calendar-year frame), and picks the matching period-label order for the
    input's resolution.

    The year axis (used for sorting each period's across-years series and for
    change-point-year mapping) is always the plain calendar ``Year`` column,
    matching the source script exactly: its per-period daily/10-daily tables
    group the hydro-year-filtered frame by Period but always sort by and
    report calendar ``Year`` (``sub["Year"].values``), never ``HydroYear`` --
    even though the frame itself is hydro-year-filtered. This only affects the
    reported change-point *year label* for periods that fall in Jan/Feb/Mar
    (where ``Year == HydroYear + 1``); trend-test results are identical either
    way, since ``Year`` and ``HydroYear`` differ by the same constant offset
    for every row of a given Period, so sorting by one or the other yields the
    same row order.
    """
    frame = pre.calendar if calendar else pre.hydro
    info = RESOLUTION_INFO[pre.resolution]
    order = list(info.cal_periods if calendar else info.hydro_periods)
    return analyze_by_period(
        frame,
        value_col=value_col,
        period_col=COL_PERIOD,
        year_col=COL_YEAR,
        order=order,
        alpha=alpha,
    )


def analyze_monthly_volumes(
    hydro: pd.DataFrame,
    *,
    value_col: str,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Trend analysis of total monthly volume, across hydrological years.

    One row per hydrological month (``HYDRO_MONTHS`` order: Apr...Mar). Each
    row's series is that calendar month's total volume (``value_col``, e.g.
    ``Vol_MAF``) for every hydrological year on record — the across-years
    series :func:`analyze_series` runs its trend tests on. Mirrors the
    source's ``RESULTS["monthly_<col>"]``.
    """
    monthly = monthly_volumes(hydro)
    rows: dict[str, dict[str, Any]] = {}
    for month, group in monthly.groupby("Month", sort=False):
        ordered = group.sort_values(COL_HYDRO_YEAR)
        rows[str(month)] = analyze_series(
            ordered[value_col], years=ordered[COL_HYDRO_YEAR].to_numpy(), alpha=alpha
        )
    result = pd.DataFrame.from_dict(rows, orient="index")
    result = _reindex_to_order(result, HYDRO_MONTHS)
    result.index.name = "Month"
    return result


def analyze_hydro_seasonal_volumes(
    hydro: pd.DataFrame,
    *,
    value_col: str,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Trend analysis of total cropping-seasonal/annual volume, across hydro years.

    One row per season, in ``Early_Kharif``/``Late_Kharif``/``Kharif``/``Rabi``/
    ``Annual`` order (the source's ``season_order``); each row's series is that
    season's yearly total volume (``value_col``). Mirrors the source's
    ``RESULTS["seasonal_<name>_<col>"]``. For the calendar-based meteorological
    seasons instead, see :func:`analyze_met_seasonal_volumes`.
    """
    seasonal = hydro_seasonal_volumes(hydro)
    rows = {
        _HYDRO_SEASON_LABELS[name]: analyze_series(
            seasonal[name][value_col].sort_index(),
            years=seasonal[name].sort_index().index.to_numpy(),
            alpha=alpha,
        )
        for name in _HYDRO_SEASON_ORDER
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Season"
    return result


def analyze_met_seasonal_volumes(
    hydro: pd.DataFrame,
    *,
    value_col: str,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Trend analysis of total meteorological-season volume, across met years.

    One row per season, in ``Winter``/``Spring``/``Summer``/``Monsoon``/
    ``Autumn`` order; each row's series is that season's per-met-year total
    volume (``value_col``, see
    :func:`~hydrotrends.data.preprocessing.met_seasonal_volumes` for how a
    "met year" is defined). Not present in the source script — an addition
    alongside the cropping-season analysis it does have.
    """
    met = met_seasonal_volumes(hydro)
    rows = {
        _MET_SEASON_LABELS[name]: analyze_series(
            met[name][value_col].sort_index(),
            years=met[name].sort_index().index.to_numpy(),
            alpha=alpha,
        )
        for name in _MET_SEASON_ORDER
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Met Season"
    return result


def describe_monthly_volumes(hydro: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    """Descriptive statistics of total monthly volume, across hydrological years.

    Companion to :func:`analyze_monthly_volumes`: descriptive-only (no trend
    tests), one row per hydrological month.
    """
    monthly = monthly_volumes(hydro)
    result = _reindex_to_order(
        describe_by(monthly, value_col=value_col, by="Month"), HYDRO_MONTHS
    )
    result.index.name = "Month"
    return result


def describe_hydro_seasonal_volumes(
    hydro: pd.DataFrame, *, value_col: str
) -> pd.DataFrame:
    """Descriptive statistics of total cropping-seasonal/annual volume.

    Companion to :func:`analyze_hydro_seasonal_volumes`: descriptive-only (no
    trend tests), one row per season in ``_HYDRO_SEASON_ORDER``.
    """
    seasonal = hydro_seasonal_volumes(hydro)
    rows = {
        _HYDRO_SEASON_LABELS[name]: describe(seasonal[name][value_col]).to_dict()
        for name in _HYDRO_SEASON_ORDER
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Season"
    return result


def describe_met_seasonal_volumes(
    hydro: pd.DataFrame, *, value_col: str
) -> pd.DataFrame:
    """Descriptive statistics of total meteorological-season volume.

    Companion to :func:`analyze_met_seasonal_volumes`: descriptive-only (no
    trend tests), one row per season in ``_MET_SEASON_ORDER``.
    """
    met = met_seasonal_volumes(hydro)
    rows = {
        _MET_SEASON_LABELS[name]: describe(met[name][value_col]).to_dict()
        for name in _MET_SEASON_ORDER
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Met Season"
    return result


@dataclass(frozen=True)
class ReportColumn:
    """One column to analyse in a report, with its display unit label.

    ``volume_column``/``volume_unit_label`` are optional: when set, the report
    also gets monthly and seasonal/annual *volume* trend (and, if
    ``include_descriptive``, descriptive) sheets for this flow/volume pairing —
    e.g. ``COL_FLOW_CUSECS`` paired with ``COL_VOL_MAF``, matching the source's
    per-workbook flow/volume unit pairing (see
    :data:`~hydrotrends.core.constants.UNIT_PAIRS`).
    """

    column: str  # DataFrame column, e.g. COL_FLOW_CUSECS
    unit_label: str  # e.g. "Cusecs", "Cumecs", "MAF"
    volume_column: str | None = None  # e.g. COL_VOL_MAF; None skips volume sheets
    volume_unit_label: str | None = None  # e.g. "MAF"


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
    season_schemes: Sequence[SeasonScheme] = (
        SeasonScheme.CROPPING,
        SeasonScheme.METEOROLOGICAL,
    ),
) -> Path:
    """Build a complete .xlsx report from a preprocessed input.

    Writes a cover sheet, then for each column a trend-analysis sheet (and, when
    ``include_descriptive`` is set, a descriptive-statistics sheet). Periods are
    ordered by the resolution's hydrological (or calendar) sequence. When a
    column is volume-paired (see :class:`ReportColumn`), monthly volume trend
    sheets are always added, plus cropping-season (``Early_Kharif``/
    ``Late_Kharif``/``Kharif``/``Rabi``/``Annual``) and/or meteorological-season
    (``Winter``/``Spring``/``Summer``/``Monsoon``/``Autumn``) trend sheets
    depending on which schemes ``season_schemes`` selects — both by default,
    matching :attr:`~hydrotrends.core.config.Config.season_schemes`. Returns
    the output path.
    """
    if not columns:
        raise ValueError("generate_report needs at least one ReportColumn")
    include_cropping = SeasonScheme.CROPPING in season_schemes
    include_meteorological = SeasonScheme.METEOROLOGICAL in season_schemes

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

        if rc.volume_column is not None:
            vol_unit = rc.volume_unit_label or ""
            monthly_trend = analyze_monthly_volumes(
                pre.hydro, value_col=rc.volume_column, alpha=alpha
            )
            write_results_sheet(
                wb.create_sheet(f"Monthly Trends ({vol_unit})"),
                monthly_trend,
                title=f"{title} — Monthly Volume Trend ({vol_unit})",
                index_label="Month",
                unit=vol_unit,
            )
            if include_descriptive:
                write_descriptive_sheet(
                    wb.create_sheet(f"Monthly Descriptive ({vol_unit})"),
                    describe_monthly_volumes(pre.hydro, value_col=rc.volume_column),
                    title=f"Descriptive Statistics — Monthly Volume ({vol_unit})",
                    index_label="Month",
                    unit=vol_unit,
                )

            if include_cropping:
                hydro_trend = analyze_hydro_seasonal_volumes(
                    pre.hydro, value_col=rc.volume_column, alpha=alpha
                )
                write_results_sheet(
                    wb.create_sheet(f"Hydro Season Trends ({vol_unit})"),
                    hydro_trend,
                    title=(
                        f"{title} — Hydro Seasonal & Annual Volume Trend ({vol_unit})"
                    ),
                    index_label="Season",
                    unit=vol_unit,
                )
                if include_descriptive:
                    write_descriptive_sheet(
                        wb.create_sheet(f"Hydro Season Descriptive ({vol_unit})"),
                        describe_hydro_seasonal_volumes(
                            pre.hydro, value_col=rc.volume_column
                        ),
                        title=(
                            f"Descriptive Statistics — Hydro Seasonal & Annual "
                            f"Volume ({vol_unit})"
                        ),
                        index_label="Season",
                        unit=vol_unit,
                    )

            if include_meteorological:
                met_trend = analyze_met_seasonal_volumes(
                    pre.hydro, value_col=rc.volume_column, alpha=alpha
                )
                write_results_sheet(
                    wb.create_sheet(f"Met Season Trends ({vol_unit})"),
                    met_trend,
                    title=f"{title} — Meteorological Season Volume Trend ({vol_unit})",
                    index_label="Met Season",
                    unit=vol_unit,
                )
                if include_descriptive:
                    write_descriptive_sheet(
                        wb.create_sheet(f"Met Season Descriptive ({vol_unit})"),
                        describe_met_seasonal_volumes(
                            pre.hydro, value_col=rc.volume_column
                        ),
                        title=(
                            f"Descriptive Statistics — Meteorological Season "
                            f"Volume ({vol_unit})"
                        ),
                        index_label="Met Season",
                        unit=vol_unit,
                    )

    wb.remove(default)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
