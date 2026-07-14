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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

from .core.config import SeasonScheme
from .core.constants import (
    CAL_DEKADS,
    CAL_MONTHS,
    COL_HYDRO_YEAR,
    COL_MET_YEAR,
    COL_MONTH,
    COL_PERIOD,
    COL_YEAR,
    DEFAULT_ALPHA,
    HYDRO_DEKADS,
    HYDRO_MONTHS,
    MET_DEKADS,
    MET_MONTHS,
    MOVING_AVERAGE_WINDOW,
    RESOLUTION_INFO,
    TimeResolution,
)
from .core.utils import (
    hydro_year_label,
    met_year_label,
    significance_stars,
    to_float_array,
)
from .data.preprocessing import (
    PreprocessedData,
    dekads_from_daily,
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
from .stats.extended_descriptive import describe_extended, describe_extended_by
from .stats.ita import innovative_trend_analysis
from .stats.trends import (
    linear_regression,
    log_linear_regression,
    mann_kendall,
    mann_kendall_modified,
    percent_slope,
    sens_slope,
)
from .viz.reports import (
    write_annual_data_sheet,
    write_cover_sheet,
    write_extended_stats_sheet,
    write_monthly_data_sheet,
    write_period_data_sheet,
    write_results_sheet,
    write_seasonal_data_sheet,
)

__all__ = [
    "analyze_series",
    "analyze_by_period",
    "analyze_preprocessed",
    "analyze_10daily_from_daily",
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
# Winter year-boundary convention), in calendar order, plus the met-year
# annual total as the final row (mirrors _HYDRO_SEASON_ORDER's "Annual").
_MET_SEASON_ORDER: tuple[str, ...] = (
    "Winter",
    "Spring",
    "Summer",
    "Monsoon",
    "Autumn",
    "Annual",
)
_MET_SEASON_LABELS: dict[str, str] = {
    "Winter": "Winter   (Dec–Feb)",
    "Spring": "Spring   (Mar–Apr)",
    "Summer": "Summer   (May–Jun)",
    "Monsoon": "Monsoon  (Jul–Sep)",
    "Autumn": "Autumn   (Oct–Nov)",
    "Annual": "Annual   (Dec–Nov)",
}

_MIN_FOR_TRENDS = 4  # matches the source's n < 4 guard
_LAST_N_YEARS = 5  # "Last 5-Years Mean" window in the Descriptive Statistics Summary


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


def analyze_10daily_from_daily(
    daily_hydro: pd.DataFrame,
    *,
    value_col: str,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Trend analysis of dekad-averaged flow, derived from a daily record.

    One row per dekad (``HYDRO_DEKADS`` order: Apr1...Mar3). Aggregates the
    daily hydro frame into dekads (see
    :func:`~hydrotrends.data.preprocessing.dekads_from_daily`), then runs the
    same per-period analysis the native 10-daily pathway does — sorted by,
    and reporting, calendar ``Year`` (matching :func:`analyze_preprocessed`'s
    year-axis convention). Mirrors the source's ``RESULTS["10daily_<col>"]``.
    """
    dekadal = dekads_from_daily(daily_hydro)
    return analyze_by_period(
        dekadal,
        value_col=value_col,
        period_col=COL_PERIOD,
        year_col=COL_YEAR,
        order=list(HYDRO_DEKADS),
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
class _YearFraming:
    """One Cal/Hydro/Met Year framing of a Daily or 10-Daily raw-value pivot."""

    key: str  # "Cal_Year" / "Hydro_Year" / "Met_Year" -- goes in the sheet name
    row_label: str  # "Year" / "Hydro Year" / "Met Year"
    year_col: str
    frame: pd.DataFrame
    period_order: tuple[str, ...]
    month_order: tuple[str, ...]
    label_fn: Callable[[Any], str] | None


def _year_framings(pre: PreprocessedData) -> tuple[_YearFraming, ...]:
    """The 3 Cal/Hydro/Met framings of ``pre`` at its own resolution."""
    info = RESOLUTION_INFO[pre.resolution]
    return (
        _YearFraming(
            "Cal_Year",
            "Year",
            COL_YEAR,
            pre.calendar,
            info.cal_periods,
            CAL_MONTHS,
            None,
        ),
        _YearFraming(
            "Hydro_Year",
            "Hydro Year",
            COL_HYDRO_YEAR,
            pre.hydro,
            info.hydro_periods,
            HYDRO_MONTHS,
            hydro_year_label,
        ),
        _YearFraming(
            "Met_Year",
            "Met Year",
            COL_MET_YEAR,
            pre.met,
            info.met_periods,
            MET_MONTHS,
            met_year_label,
        ),
    )


def _dekad_year_framings(pre: PreprocessedData) -> tuple[_YearFraming, ...]:
    """The 3 Cal/Hydro/Met framings of dekads derived from a *daily* ``pre``.

    Lets a daily-only record also produce 10-Daily raw-value sheets, the same
    way :func:`analyze_10daily_from_daily` derives 10-Daily trend sheets.
    """
    return (
        _YearFraming(
            "Cal_Year",
            "Year",
            COL_YEAR,
            dekads_from_daily(pre.calendar),
            CAL_DEKADS,
            CAL_MONTHS,
            None,
        ),
        _YearFraming(
            "Hydro_Year",
            "Hydro Year",
            COL_HYDRO_YEAR,
            dekads_from_daily(pre.hydro),
            HYDRO_DEKADS,
            HYDRO_MONTHS,
            hydro_year_label,
        ),
        _YearFraming(
            "Met_Year",
            "Met Year",
            COL_MET_YEAR,
            dekads_from_daily(pre.met),
            MET_DEKADS,
            MET_MONTHS,
            met_year_label,
        ),
    )


# Excel's sheet-tab-name limit is 31 characters. "10Daily_Mean_Data_..._Cusecs"
# is the one sheet-tag/unit combination long enough to blow past it (up to 35
# chars for the Hydro_Year case), so only those sheet *names* (not their
# titles or row labels) abbreviate the year-type -- CY/HY/MY instead of
# Cal_Year/Hydro_Year/Met_Year.
_ABBREVIATED_FRAMING_KEY = {"Cal_Year": "CY", "Hydro_Year": "HY", "Met_Year": "MY"}


def _write_period_data_sheets(
    wb: Workbook,
    framings: Sequence[_YearFraming],
    *,
    sheet_tag: str,
    quantity: str,
    value_col: str,
    unit_label: str,
    title: str,
    abbreviate_sheet_key: bool = False,
) -> None:
    """Write one Data pivot sheet per Cal/Hydro/Met Year framing.

    ``sheet_tag`` is the sheet-name prefix (e.g. ``"Daily"``/``"10Daily_Mean"``);
    ``quantity`` is the human title fragment (e.g. ``"Daily Inflow"``).
    """
    for framing in framings:
        pivot = framing.frame.pivot_table(
            index=framing.year_col,
            columns=COL_PERIOD,
            values=value_col,
            aggfunc="first",
        )
        sheet_key = (
            _ABBREVIATED_FRAMING_KEY[framing.key]
            if abbreviate_sheet_key
            else framing.key
        )
        write_period_data_sheet(
            wb.create_sheet(f"{sheet_tag}_Data_{sheet_key}_{unit_label}"),
            pivot,
            title=f"{quantity} ({unit_label}) - {title} [{framing.row_label}]",
            period_order=list(framing.period_order),
            month_order=list(framing.month_order),
            row_label=framing.row_label,
            unit=unit_label,
            label_fn=framing.label_fn,
        )


_HYDRO_SEASON_COL_HEADERS: tuple[str, ...] = (
    "Early Kharif",
    "Late Kharif",
    "Kharif",
    "Rabi",
    "Annual",
)
_HYDRO_SEASON_SUB_LABELS: tuple[str, ...] = (
    "(Apr1–Jun10)",
    "(Jun11–Sep30)",
    "(Apr1–Sep30)",
    "(Oct1–Mar31)",
    "(Apr1–Mar31)",
)
_MET_SEASON_COL_HEADERS: tuple[str, ...] = (
    "Winter",
    "Spring",
    "Summer",
    "Monsoon",
    "Autumn",
    "Annual",
)
_MET_SEASON_SUB_LABELS: tuple[str, ...] = (
    "(Dec–Feb)",
    "(Mar–Apr)",
    "(May–Jun)",
    "(Jul–Sep)",
    "(Oct–Nov)",
    "(Dec–Nov)",
)


def _write_monthly_data_sheet(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Monthly_Data_{unit}``: monthly volume by hydrological year."""
    pivot = (
        monthly_volumes(pre.hydro)
        .pivot_table(
            index=COL_HYDRO_YEAR, columns=COL_MONTH, values=value_col, aggfunc="first"
        )
        .reindex(columns=HYDRO_MONTHS)
    )
    write_monthly_data_sheet(
        wb.create_sheet(f"Monthly_Data_{unit_label}"),
        pivot,
        title=f"Monthly Inflow Volume ({unit_label}) - {title}",
        month_order=list(HYDRO_MONTHS),
        unit=unit_label,
        label_fn=hydro_year_label,
    )


def _write_hydro_season_data_sheet(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Hydro_Season_Data_{unit}``: cropping-seasonal & annual volume."""
    seasonal = hydro_seasonal_volumes(pre.hydro)
    all_years = sorted(seasonal["Annual"].index)
    data = {name: seasonal[name][value_col] for name in _HYDRO_SEASON_ORDER}
    write_seasonal_data_sheet(
        wb.create_sheet(f"Hydro_Season_Data_{unit_label}"),
        data,
        title=f"Hydro-Seasonal & Annual Inflow Volume ({unit_label}) - {title}",
        col_headers=list(_HYDRO_SEASON_COL_HEADERS),
        sub_labels=list(_HYDRO_SEASON_SUB_LABELS),
        all_years=all_years,
        col_keys=list(_HYDRO_SEASON_ORDER),
        unit=unit_label,
        label_fn=hydro_year_label,
        row_label="Hydro Year",
    )


def _write_met_season_data_sheet(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Met_Season_Data_{unit}``: meteorological-seasonal & annual volume."""
    met = met_seasonal_volumes(pre.hydro)
    all_years = sorted(met["Annual"].index)
    data = {name: met[name][value_col] for name in _MET_SEASON_ORDER}
    write_seasonal_data_sheet(
        wb.create_sheet(f"Met_Season_Data_{unit_label}"),
        data,
        title=f"Meteorological-Seasonal Inflow Volume ({unit_label}) - {title}",
        col_headers=list(_MET_SEASON_COL_HEADERS),
        sub_labels=list(_MET_SEASON_SUB_LABELS),
        all_years=all_years,
        col_keys=list(_MET_SEASON_ORDER),
        unit=unit_label,
        label_fn=met_year_label,
        row_label="Met Year",
    )


def _write_annual_data_sheet(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Annual_Data_{unit}``: annual volume, anomaly, and 5-yr backward MA."""
    ann = hydro_seasonal_volumes(pre.hydro)["Annual"][value_col].sort_index()
    mean_vol = ann.mean()
    ma5 = ann.rolling(MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_WINDOW).mean()
    pct_col = "Anomaly\n(%)"
    ann_df = pd.DataFrame(
        {
            f"Annual Vol\n({unit_label})": ann.round(2),
            f"Anomaly\n({unit_label})": (ann - mean_vol).round(2),
            pct_col: ((ann - mean_vol) / mean_vol * 100).round(2),
            f"5-yr Bwd MA\n({unit_label})": ma5.round(2),
        }
    )
    write_annual_data_sheet(
        wb.create_sheet(f"Annual_Data_{unit_label}"),
        ann_df,
        title=f"Annual Inflow Volume by Hydrological Year - {title} [{unit_label}]",
        data_cols=list(ann_df.columns),
        label_fn=hydro_year_label,
        highlight_cols=frozenset({pct_col}),
        unit=unit_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive Statistics Summary sheets (Phase 3): Horizontal (one row per
# year, across that year's periods) + Vertical (one row per period, across
# years) full-field summaries. Always Hydro-Year framed -- unlike the raw Data
# sheets, these aren't tripled across Cal/Hydro/Met Year, to keep the sheet
# count bounded (see the Phase 3 commit message for the reasoning).
# ─────────────────────────────────────────────────────────────────────────────
def _write_period_stats_summary(
    wb: Workbook,
    frame: pd.DataFrame,
    *,
    sheet_tag: str,
    period_order: Sequence[str],
    value_col: str,
    unit_label: str,
    title: str,
) -> None:
    """Write ``{tag}_Summary_H_{unit}`` / ``{tag}_Summary_V_{unit}`` for a
    Daily/10-Daily period frame (one row per Hydro Year x one column per
    period, or vice versa).
    """
    horiz = describe_extended_by(
        frame, value_col=value_col, by=COL_HYDRO_YEAR, expected_n=len(period_order)
    )
    write_extended_stats_sheet(
        wb.create_sheet(f"{sheet_tag}_Summary_H_{unit_label}"),
        horiz,
        title=(
            f"{sheet_tag} Descriptive Statistics Summary (Horizontal) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Hydro Year",
        unit=unit_label,
        label_fn=hydro_year_label,
    )

    n_years = int(frame[COL_HYDRO_YEAR].nunique())
    vert = describe_extended_by(
        frame,
        value_col=value_col,
        by=COL_PERIOD,
        sort_col=COL_HYDRO_YEAR,
        expected_n=n_years,
        last_n_window=_LAST_N_YEARS,
    )
    vert = _reindex_to_order(vert, period_order)
    write_extended_stats_sheet(
        wb.create_sheet(f"{sheet_tag}_Summary_V_{unit_label}"),
        vert,
        title=(
            f"{sheet_tag} Descriptive Statistics Summary (Vertical) - "
            f"{title} [{unit_label}]"
        ),
        index_label=sheet_tag,
        unit=unit_label,
    )


def _write_monthly_stats_summary(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Monthly_Summary_H_{unit}`` / ``Monthly_Summary_V_{unit}``."""
    monthly = monthly_volumes(pre.hydro)
    horiz = describe_extended_by(
        monthly, value_col=value_col, by=COL_HYDRO_YEAR, expected_n=len(HYDRO_MONTHS)
    )
    write_extended_stats_sheet(
        wb.create_sheet(f"Monthly_Summary_H_{unit_label}"),
        horiz,
        title=(
            f"Monthly Descriptive Statistics Summary (Horizontal) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Hydro Year",
        unit=unit_label,
        label_fn=hydro_year_label,
    )

    n_years = int(monthly[COL_HYDRO_YEAR].nunique())
    vert = describe_extended_by(
        monthly,
        value_col=value_col,
        by=COL_MONTH,
        sort_col=COL_HYDRO_YEAR,
        expected_n=n_years,
        last_n_window=_LAST_N_YEARS,
    )
    vert = _reindex_to_order(vert, HYDRO_MONTHS)
    write_extended_stats_sheet(
        wb.create_sheet(f"Monthly_Summary_V_{unit_label}"),
        vert,
        title=(
            f"Monthly Descriptive Statistics Summary (Vertical) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Month",
        unit=unit_label,
    )


def _seasonal_long_frame(
    seasonal: dict[str, pd.DataFrame],
    order: Sequence[str],
    *,
    year_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Tidy ``(year_col, "Season", value_col)`` frame from a per-season dict
    of year-indexed frames (as returned by ``hydro_seasonal_volumes``/
    ``met_seasonal_volumes``) -- lets the season scales reuse
    :func:`~hydrotrends.stats.extended_descriptive.describe_extended_by` the
    same way the Monthly scale does.
    """
    parts = [
        pd.DataFrame(
            {
                year_col: seasonal[name].index,
                "Season": name,
                value_col: seasonal[name][value_col].to_numpy(),
            }
        )
        for name in order
    ]
    return pd.concat(parts, ignore_index=True)


def _write_hydro_season_stats_summary(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Hydro_Season_Summary_H_{unit}`` / ``Hydro_Season_Summary_V_{unit}``."""
    long = _seasonal_long_frame(
        hydro_seasonal_volumes(pre.hydro),
        _HYDRO_SEASON_ORDER,
        year_col=COL_HYDRO_YEAR,
        value_col=value_col,
    )
    horiz = describe_extended_by(
        long,
        value_col=value_col,
        by=COL_HYDRO_YEAR,
        expected_n=len(_HYDRO_SEASON_ORDER),
    )
    write_extended_stats_sheet(
        wb.create_sheet(f"Hydro_Season_Summary_H_{unit_label}"),
        horiz,
        title=(
            f"Hydro-Season Descriptive Statistics Summary (Horizontal) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Hydro Year",
        unit=unit_label,
        label_fn=hydro_year_label,
    )

    n_years = int(long[COL_HYDRO_YEAR].nunique())
    vert = describe_extended_by(
        long,
        value_col=value_col,
        by="Season",
        sort_col=COL_HYDRO_YEAR,
        expected_n=n_years,
        last_n_window=_LAST_N_YEARS,
    )
    vert = _reindex_to_order(vert, _HYDRO_SEASON_ORDER)
    write_extended_stats_sheet(
        wb.create_sheet(f"Hydro_Season_Summary_V_{unit_label}"),
        vert,
        title=(
            f"Hydro-Season Descriptive Statistics Summary (Vertical) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Season",
        unit=unit_label,
    )


def _write_met_season_stats_summary(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Met_Season_Summary_H_{unit}`` / ``Met_Season_Summary_V_{unit}``."""
    long = _seasonal_long_frame(
        met_seasonal_volumes(pre.hydro),
        _MET_SEASON_ORDER,
        year_col=COL_MET_YEAR,
        value_col=value_col,
    )
    horiz = describe_extended_by(
        long, value_col=value_col, by=COL_MET_YEAR, expected_n=len(_MET_SEASON_ORDER)
    )
    write_extended_stats_sheet(
        wb.create_sheet(f"Met_Season_Summary_H_{unit_label}"),
        horiz,
        title=(
            f"Met-Season Descriptive Statistics Summary (Horizontal) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Met Year",
        unit=unit_label,
        label_fn=met_year_label,
    )

    n_years = int(long[COL_MET_YEAR].nunique())
    vert = describe_extended_by(
        long,
        value_col=value_col,
        by="Season",
        sort_col=COL_MET_YEAR,
        expected_n=n_years,
        last_n_window=_LAST_N_YEARS,
    )
    vert = _reindex_to_order(vert, _MET_SEASON_ORDER)
    write_extended_stats_sheet(
        wb.create_sheet(f"Met_Season_Summary_V_{unit_label}"),
        vert,
        title=(
            f"Met-Season Descriptive Statistics Summary (Vertical) - "
            f"{title} [{unit_label}]"
        ),
        index_label="Season",
        unit=unit_label,
    )


def _write_annual_stats_summary(
    wb: Workbook, pre: PreprocessedData, *, value_col: str, unit_label: str, title: str
) -> None:
    """Write ``Annual_Summary_{unit}``.

    Annual volume has no within-year sub-periods to summarise, so there's no
    Horizontal counterpart here -- just the across-years (Vertical) summary,
    as a single-row table.
    """
    ann = hydro_seasonal_volumes(pre.hydro)["Annual"][value_col].sort_index()
    stats = describe_extended(
        ann.to_numpy(), expected_n=int(ann.size), last_n_window=_LAST_N_YEARS
    )
    table = pd.DataFrame([stats.to_dict()], index=["All Years"])
    write_extended_stats_sheet(
        wb.create_sheet(f"Annual_Summary_{unit_label}"),
        table,
        title=f"Annual Descriptive Statistics Summary - {title} [{unit_label}]",
        index_label="Hydro Year",
        unit=unit_label,
    )


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
    alpha: float = DEFAULT_ALPHA,
    season_schemes: Sequence[SeasonScheme] = (
        SeasonScheme.CROPPING,
        SeasonScheme.METEOROLOGICAL,
    ),
) -> Path:
    """Build a complete .xlsx report from a preprocessed input.

    Writes a cover sheet, then for each column a trend-analysis sheet plus its
    raw-value Data sheets (Cal/Hydro/Met Year framings, with full row/column
    descriptive statistics -- see
    :func:`~hydrotrends.viz.reports.write_period_data_sheet`). Periods are
    ordered by the resolution's hydrological (or calendar) sequence.
    When a column is volume-paired (see :class:`ReportColumn`), monthly volume
    trend and Data sheets are always added, plus cropping-season
    (``Early_Kharif``/``Late_Kharif``/``Kharif``/``Rabi``/``Annual``) and/or
    meteorological-season (``Winter``/``Spring``/``Summer``/``Monsoon``/
    ``Autumn``) trend and Data sheets depending on which schemes
    ``season_schemes`` selects — both by default, matching
    :attr:`~hydrotrends.core.config.Config.season_schemes`. Returns the output
    path.
    """
    if not columns:
        raise ValueError("generate_report needs at least one ReportColumn")
    include_cropping = SeasonScheme.CROPPING in season_schemes
    include_meteorological = SeasonScheme.METEOROLOGICAL in season_schemes

    is_daily = pre.resolution is TimeResolution.DAILY
    primary_label = "Daily" if is_daily else "10Daily"
    primary_desc = "Daily Inflow" if is_daily else "10-Daily Mean Inflow"
    primary_vol_desc = "Daily Inflow Volume" if is_daily else "10-Daily Inflow Volume"

    wb = Workbook()
    default = wb.active
    write_cover_sheet(wb.create_sheet("Cover"), title=title, subtitle=subtitle)

    for rc in columns:
        trends = analyze_preprocessed(
            pre, value_col=rc.column, calendar=calendar, alpha=alpha
        )
        write_results_sheet(
            wb.create_sheet(f"{primary_label}_Trends_{rc.unit_label}"),
            trends,
            title=f"{primary_desc} Trend Analysis - {title} [{rc.unit_label}]",
            index_label=primary_label,
            unit=rc.unit_label,
        )
        if is_daily:
            # A daily record can also produce the 10-daily (dekad-averaged)
            # trend table, derived on the fly -- no separate 10-daily input
            # needed (see dekads_from_daily).
            dekadal_trends = analyze_10daily_from_daily(
                pre.hydro, value_col=rc.column, alpha=alpha
            )
            write_results_sheet(
                wb.create_sheet(f"10Daily_Trends_{rc.unit_label}"),
                dekadal_trends,
                title=(
                    f"10-Daily Mean Inflow Trend Analysis - {title} [{rc.unit_label}]"
                ),
                index_label="10Daily",
                unit=rc.unit_label,
            )

        # Raw-value Data sheets: one per Cal/Hydro/Met Year framing, always
        # (independent of the `calendar` trend-framing choice above).
        _write_period_data_sheets(
            wb,
            _year_framings(pre),
            sheet_tag="Daily" if is_daily else "10Daily_Mean",
            quantity=primary_desc,
            value_col=rc.column,
            unit_label=rc.unit_label,
            title=title,
            abbreviate_sheet_key=not is_daily,
        )
        if is_daily:
            _write_period_data_sheets(
                wb,
                _dekad_year_framings(pre),
                sheet_tag="10Daily_Mean",
                quantity="10-Daily Mean Inflow",
                value_col=rc.column,
                unit_label=rc.unit_label,
                title=title,
                abbreviate_sheet_key=True,
            )

        # Descriptive Statistics Summary (Horizontal + Vertical), Hydro-Year
        # framed -- see _write_period_stats_summary's docstring for why this
        # isn't tripled across Cal/Hydro/Met Year the way Data sheets are.
        _write_period_stats_summary(
            wb,
            pre.hydro,
            sheet_tag="Daily" if is_daily else "10Daily_Mean",
            period_order=list(RESOLUTION_INFO[pre.resolution].hydro_periods),
            value_col=rc.column,
            unit_label=rc.unit_label,
            title=title,
        )
        if is_daily:
            _write_period_stats_summary(
                wb,
                dekads_from_daily(pre.hydro),
                sheet_tag="10Daily_Mean",
                period_order=list(HYDRO_DEKADS),
                value_col=rc.column,
                unit_label=rc.unit_label,
                title=title,
            )

        if rc.volume_column is not None:
            vol_unit = rc.volume_unit_label or ""
            _write_period_data_sheets(
                wb,
                _year_framings(pre),
                sheet_tag=primary_label,
                quantity=primary_vol_desc,
                value_col=rc.volume_column,
                unit_label=vol_unit,
                title=title,
            )
            if is_daily:
                _write_period_data_sheets(
                    wb,
                    _dekad_year_framings(pre),
                    sheet_tag="10Daily",
                    quantity="10-Daily Inflow Volume",
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
            _write_period_stats_summary(
                wb,
                pre.hydro,
                sheet_tag=primary_label,
                period_order=list(RESOLUTION_INFO[pre.resolution].hydro_periods),
                value_col=rc.volume_column,
                unit_label=vol_unit,
                title=title,
            )
            if is_daily:
                _write_period_stats_summary(
                    wb,
                    dekads_from_daily(pre.hydro),
                    sheet_tag="10Daily",
                    period_order=list(HYDRO_DEKADS),
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
            monthly_trend = analyze_monthly_volumes(
                pre.hydro, value_col=rc.volume_column, alpha=alpha
            )
            write_results_sheet(
                wb.create_sheet(f"Monthly_Trends_{vol_unit}"),
                monthly_trend,
                title=f"Monthly Inflow Trend Analysis - {title} [{vol_unit}]",
                index_label="Monthly",
                unit=vol_unit,
            )
            _write_monthly_data_sheet(
                wb, pre, value_col=rc.volume_column, unit_label=vol_unit, title=title
            )
            _write_monthly_stats_summary(
                wb, pre, value_col=rc.volume_column, unit_label=vol_unit, title=title
            )

            if include_cropping:
                hydro_trend = analyze_hydro_seasonal_volumes(
                    pre.hydro, value_col=rc.volume_column, alpha=alpha
                )
                write_results_sheet(
                    wb.create_sheet(f"Hydro_Season_Trends_{vol_unit}"),
                    hydro_trend,
                    title=(
                        f"Hydro-Seasonal & Annual Inflow Volume Trend Analysis - "
                        f"{title} [{vol_unit}]"
                    ),
                    index_label="Season",
                    unit=vol_unit,
                )
                _write_hydro_season_data_sheet(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
                _write_annual_data_sheet(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
                _write_hydro_season_stats_summary(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
                _write_annual_stats_summary(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )

            if include_meteorological:
                met_trend = analyze_met_seasonal_volumes(
                    pre.hydro, value_col=rc.volume_column, alpha=alpha
                )
                write_results_sheet(
                    wb.create_sheet(f"Met_Season_Trends_{vol_unit}"),
                    met_trend,
                    title=(
                        f"Meteorological-Seasonal Inflow Volume Trend Analysis - "
                        f"{title} [{vol_unit}]"
                    ),
                    index_label="Met Season",
                    unit=vol_unit,
                )
                _write_met_season_data_sheet(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )
                _write_met_season_stats_summary(
                    wb,
                    pre,
                    value_col=rc.volume_column,
                    unit_label=vol_unit,
                    title=title,
                )

    wb.remove(default)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
