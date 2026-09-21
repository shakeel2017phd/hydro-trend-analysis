"""Plot-suite generation for :mod:`hydrotrends` (PNG/HTML, not Excel).

:func:`generate_plots` is the plotting counterpart to
:func:`~hydrotrends.api.generate_report`: instead of a workbook, it writes a
tree of PNG/interactive-HTML files, mirroring the source script's Section
11/12/13 plot suites (see the docstrings below for which section each part
covers). Not imported by ``hydrotrends/__init__.py`` -- like
:mod:`hydrotrends.viz.plotting`, which it depends on, this module pulls in
Matplotlib/Plotly/Seaborn, so importing the base :mod:`hydrotrends` package
stays lightweight. Import it explicitly::

    from hydrotrends.plots import generate_plots
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .api import (
    ReportColumn,
    analyze_10daily_from_daily,
    analyze_preprocessed,
    analyze_series,
)
from .core.constants import (
    COL_HYDRO_YEAR,
    COL_MONTH,
    COL_PERIOD,
    FULL_MONTHS,
    HYDRO_DEKADS,
    HYDRO_MONTHS,
    HYDRO_PERIODS,
    TimeResolution,
)
from .core.utils import flood_limits_for_unit, hydro_year_label
from .data.preprocessing import (
    PreprocessedData,
    dekads_from_daily,
    hydro_seasonal_volumes,
    monthly_volumes,
)
from .stats.changepoint import bai_perron_change_point, pettitt_test
from .stats.frequency import exceedance_counts
from .stats.ita import innovative_trend_analysis
from .stats.trends import sens_slope
from .viz.plotting import (
    anomaly_bar_chart,
    boxwhisker_grid,
    build_day_grid,
    compute_decadal_summary,
    compute_sliding_window_summary,
    daily_duration_curve_interactive,
    daily_duration_curve_static,
    daily_flood_heatmap,
    daily_flood_heatmap_interactive,
    daily_flood_overlay,
    daily_trend_heatmap,
    decadal_blocks_bar_chart,
    dekadal_trend_heatmap,
    distribution_grid,
    draw_anomaly_cell,
    draw_boxwhisker_cell,
    draw_decadal_blocks_cell,
    draw_histogram_kde_cell,
    draw_ita_scatter_cell,
    draw_parametric_bounds_cell,
    draw_recent_vs_longterm_cell,
    draw_robust_bounds_cell,
    draw_violin_cell,
    flood_heatmap,
    flood_heatmap_interactive,
    flood_overlay_interactive,
    flood_overlay_static,
    flow_duration_curve_interactive,
    flow_duration_curve_static,
    ita_scatter,
    monthly_trend_heatmap,
    parametric_bounds_plot,
    robust_bounds_plot,
    seasonal_divergence_lowess_chart,
    seasonal_trend_heatmap,
    single_histogram,
    sliding_windows_bar_chart,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["generate_plots"]

# Section 12's Seasonal scale grid excludes Annual (kept in Hydro_Season_*
# sheets/summaries instead) -- a 2x2 grid of the 4 sub-annual seasons only.
_SEASON_ORDER: tuple[str, ...] = ("Early_Kharif", "Late_Kharif", "Kharif", "Rabi")
_SEASON_LABELS: dict[str, str] = {
    "Early_Kharif": "Early Kharif",
    "Late_Kharif": "Late Kharif",
    "Kharif": "Kharif",
    "Rabi": "Rabi",
}


def _series_by_period(
    frame: pd.DataFrame, *, period_col: str, value_col: str
) -> dict[str, pd.Series]:
    """``{period label: values}`` from a long frame, dropping NaNs."""
    return {str(p): g[value_col].dropna() for p, g in frame.groupby(period_col)}


def _save_static(fig: Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_interactive(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def _min_max(series_by_key: dict[str, pd.Series]) -> tuple[float, float]:
    non_empty = [s for s in series_by_key.values() if len(s)]
    if not non_empty:
        return 0.0, 1.0
    return (
        float(min(s.min() for s in non_empty)),
        float(max(s.max() for s in non_empty)),
    )


def _write_flood_plots(
    out_dir: Path,
    file_prefix: str,
    scale_name: str,
    period_order: list[str],
    by_period: dict[str, pd.Series],
    recent_by_period: pd.Series,
    recent_label: str,
    unit_label: str,
    flood_limits: dict[str, float],
) -> None:
    """Flood-exceedance overlay + heatmap (Daily/10-Daily only)."""
    mean_s = pd.Series({p: s.mean() for p, s in by_period.items()})
    q1_s = pd.Series({p: s.quantile(0.25) for p, s in by_period.items()})
    q3_s = pd.Series({p: s.quantile(0.75) for p, s in by_period.items()})

    fig = flood_overlay_static(
        period_order,
        mean_s,
        recent_by_period,
        recent_label,
        q1_s,
        q3_s,
        title=f"{scale_name} Mean Inflow with Flood Limits ({unit_label})",
        unit_label=unit_label,
        flood_limits=flood_limits,
    )
    _save_static(fig, out_dir / f"{file_prefix}_Flood_Overlay.png", dpi=200)
    fig_i = flood_overlay_interactive(
        period_order,
        mean_s,
        recent_by_period,
        recent_label,
        q1_s,
        q3_s,
        title=f"{scale_name} Mean Inflow with Flood Limits ({unit_label})",
        unit_label=unit_label,
        period_label=f"{scale_name} Period",
        flood_limits=flood_limits,
    )
    _save_interactive(fig_i, out_dir / f"{file_prefix}_Flood_Overlay.html")

    counts = pd.DataFrame(
        {p: exceedance_counts(s, flood_limits) for p, s in by_period.items()}
    ).T.reindex(period_order)
    if not counts.empty:
        fig = flood_heatmap(
            counts,
            title=f"Flood Exceedance Counts — {scale_name}",
            period_label=f"{scale_name} Period",
        )
        _save_static(fig, out_dir / f"{file_prefix}_Flood_Heatmap.png", dpi=200)
        fig_i = flood_heatmap_interactive(
            counts,
            title=f"Flood Exceedance Counts — {scale_name}",
            period_label=f"{scale_name} Period",
        )
        _save_interactive(fig_i, out_dir / f"{file_prefix}_Flood_Heatmap.html")


def _write_duration_curve(
    out_dir: Path,
    file_prefix: str,
    scale_name: str,
    values: pd.Series,
    unit_label: str,
    flood_limits: dict[str, float] | None,
) -> None:
    fig = flow_duration_curve_static(
        values,
        title=f"{scale_name} Duration Curve ({unit_label})",
        unit_label=unit_label,
        flood_limits=flood_limits,
    )
    _save_static(fig, out_dir / f"{file_prefix}_Duration_Curve.png", dpi=300)
    fig_i = flow_duration_curve_interactive(
        values,
        title=f"{scale_name} Duration Curve ({unit_label})",
        unit_label=unit_label,
        flood_limits=flood_limits,
    )
    _save_interactive(fig_i, out_dir / f"{file_prefix}_Duration_Curve.html")


def _write_daily_scale_plots(
    dirs: dict[str, Path],
    pre: PreprocessedData,
    value_col: str,
    unit_label: str,
    max_hydro_year: int,
) -> None:
    """Section 12, Daily scale: grids split by month (366 periods total)."""
    by_period = _series_by_period(pre.hydro, period_col=COL_PERIOD, value_col=value_col)
    ymin, ymax = _min_max(by_period)
    ylim = (ymin * 0.95, ymax * 1.05)

    hydro_periods = [p for p in by_period if any(p.startswith(m) for m in HYDRO_MONTHS)]
    hydro_periods.sort(key=lambda p: (HYDRO_MONTHS.index(p.split("-")[0]), p))

    for month in HYDRO_MONTHS:
        month_periods = [p for p in hydro_periods if p.startswith(month)]
        if not month_periods:
            continue
        fig = distribution_grid(
            by_period,
            order=month_periods,
            title=f"Daily ({month}) Distribution ({unit_label})",
            unit_label=unit_label,
            ncols=6,
        )
        _save_static(fig, dirs["daily"] / f"Daily_Distribution_{month}.png", dpi=250)
        fig = boxwhisker_grid(
            by_period,
            order=month_periods,
            title=f"Daily ({month}) Box-Whisker ({unit_label})",
            unit_label=unit_label,
            ncols=6,
            ylim=ylim,
        )
        _save_static(fig, dirs["daily"] / f"Daily_BoxWhisker_{month}.png", dpi=250)

    all_values = (
        pd.concat(list(by_period.values())) if by_period else pd.Series(dtype="float64")
    )
    flood_limits = flood_limits_for_unit(unit_label)
    _write_duration_curve(
        dirs["daily"], "Daily", "Daily", all_values, unit_label, flood_limits
    )

    hydro_pivot = pre.hydro.pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_PERIOD, values=value_col, aggfunc="first"
    )
    recent = (
        hydro_pivot.loc[max_hydro_year]
        if max_hydro_year in hydro_pivot.index
        else pd.Series(dtype="float64")
    )
    _write_flood_plots(
        dirs["daily"],
        "Daily",
        "Daily",
        hydro_periods,
        by_period,
        recent,
        hydro_year_label(max_hydro_year),
        unit_label,
        flood_limits,
    )


def _write_dekadal_scale_plots(
    dirs: dict[str, Path],
    dekad_frame: pd.DataFrame,
    hydro_dekads: list[str],
    value_col: str,
    unit_label: str,
    max_hydro_year: int,
) -> None:
    """Section 12, 10-Daily scale: a single 6x6 grid (36 dekads)."""
    by_period = _series_by_period(
        dekad_frame, period_col=COL_PERIOD, value_col=value_col
    )
    ymin, ymax = _min_max(by_period)
    ylim = (ymin * 0.95, ymax * 1.05)

    fig = distribution_grid(
        by_period,
        order=hydro_dekads,
        title=f"10-Daily Distribution ({unit_label})",
        unit_label=unit_label,
        ncols=6,
    )
    _save_static(fig, dirs["dekadal"] / "10Daily_Distribution_Grid.png", dpi=250)
    fig = boxwhisker_grid(
        by_period,
        order=hydro_dekads,
        title=f"10-Daily Box-Whisker ({unit_label})",
        unit_label=unit_label,
        ncols=6,
        ylim=ylim,
    )
    _save_static(fig, dirs["dekadal"] / "10Daily_BoxWhisker_Grid.png", dpi=250)

    all_values = (
        pd.concat(list(by_period.values())) if by_period else pd.Series(dtype="float64")
    )
    flood_limits = flood_limits_for_unit(unit_label)
    _write_duration_curve(
        dirs["dekadal"], "10Daily", "10-Daily", all_values, unit_label, flood_limits
    )

    dek_pivot = dekad_frame.pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_PERIOD, values=value_col, aggfunc="first"
    )
    recent = (
        dek_pivot.loc[max_hydro_year]
        if max_hydro_year in dek_pivot.index
        else pd.Series(dtype="float64")
    )
    _write_flood_plots(
        dirs["dekadal"],
        "10Daily",
        "10-Daily",
        hydro_dekads,
        by_period,
        recent,
        hydro_year_label(max_hydro_year),
        unit_label,
        flood_limits,
    )


def _write_monthly_scale_plots(
    dirs: dict[str, Path], pre: PreprocessedData, value_col: str, unit_label: str
) -> None:
    """Section 12, Monthly scale: a single 3x4 grid, no flood exceedance."""
    monthly = monthly_volumes(pre.hydro)
    by_period = _series_by_period(monthly, period_col="Month", value_col=value_col)

    fig = distribution_grid(
        by_period,
        order=list(HYDRO_MONTHS),
        title=f"Monthly Distribution ({unit_label})",
        unit_label=unit_label,
        ncols=4,
    )
    _save_static(fig, dirs["monthly"] / "Monthly_Distribution_Grid.png", dpi=250)
    ymin, ymax = _min_max(by_period)
    fig = boxwhisker_grid(
        by_period,
        order=list(HYDRO_MONTHS),
        title=f"Monthly Box-Whisker ({unit_label})",
        unit_label=unit_label,
        ncols=4,
        ylim=(ymin * 0.95, ymax * 1.05),
    )
    _save_static(fig, dirs["monthly"] / "Monthly_BoxWhisker_Grid.png", dpi=250)

    all_values = (
        pd.concat(list(by_period.values())) if by_period else pd.Series(dtype="float64")
    )
    _write_duration_curve(
        dirs["monthly"], "Monthly", "Monthly", all_values, unit_label, None
    )


def _write_seasonal_scale_plots(
    dirs: dict[str, Path], pre: PreprocessedData, value_col: str, unit_label: str
) -> None:
    """Section 12, Seasonal scale: a single 2x2 grid (excludes Annual), no
    flood exceedance."""
    seasonal = hydro_seasonal_volumes(pre.hydro)
    by_period = {
        _SEASON_LABELS[key]: seasonal[key][value_col].dropna() for key in _SEASON_ORDER
    }
    labels = list(by_period)

    fig = distribution_grid(
        by_period,
        order=labels,
        title=f"Seasonal Distribution ({unit_label})",
        unit_label=unit_label,
        ncols=2,
    )
    _save_static(fig, dirs["seasonal"] / "Seasonal_Distribution_Grid.png", dpi=250)
    ymin, ymax = _min_max(by_period)
    fig = boxwhisker_grid(
        by_period,
        order=labels,
        title=f"Seasonal Box-Whisker ({unit_label})",
        unit_label=unit_label,
        ncols=2,
        ylim=(ymin * 0.95, ymax * 1.05),
    )
    _save_static(fig, dirs["seasonal"] / "Seasonal_BoxWhisker_Grid.png", dpi=250)

    all_values = (
        pd.concat(list(by_period.values())) if by_period else pd.Series(dtype="float64")
    )
    _write_duration_curve(
        dirs["seasonal"], "Seasonal", "Seasonal", all_values, unit_label, None
    )


def _write_annual_scale_plots(
    dirs: dict[str, Path], pre: PreprocessedData, value_col: str, unit_label: str
) -> None:
    """Section 12, Annual scale: a single histogram (no grid), no flood exceedance."""
    annual = hydro_seasonal_volumes(pre.hydro)["Annual"][value_col].dropna()
    fig = single_histogram(
        annual, title=f"Annual Distribution ({unit_label})", unit_label=unit_label
    )
    _save_static(fig, dirs["annual"] / "Annual_Distribution.png", dpi=300)
    _write_duration_curve(dirs["annual"], "Annual", "Annual", annual, unit_label, None)


def _write_distribution_and_flood_plots(
    pre: PreprocessedData, rc: ReportColumn, output_dir: Path, *, is_daily: bool
) -> None:
    """Section 12 (v23): distribution/box-whisker grids + duration curves for
    all 5 scales, plus Daily/10-Daily flood-exceedance overlay + heatmap.

    Always Hydro-Year framed, matching the source (Section 12 has no Cal/Met
    Year variant).
    """
    vol_unit = rc.volume_unit_label or ""
    unit_tag = f"{rc.unit_label}_{vol_unit}" if rc.volume_column else rc.unit_label
    root = output_dir / "Distribution_and_Flood_Plots" / unit_tag
    dirs = {
        name: root / label
        for name, label in {
            "daily": "Daily",
            "dekadal": "10Daily",
            "monthly": "Monthly",
            "seasonal": "Seasonal",
            "annual": "Annual",
        }.items()
    }

    max_hydro_year = int(pre.hydro[COL_HYDRO_YEAR].max())

    if is_daily:
        _write_daily_scale_plots(dirs, pre, rc.column, rc.unit_label, max_hydro_year)
        dekad_frame = dekads_from_daily(pre.hydro)
    else:
        dekad_frame = pre.hydro

    _write_dekadal_scale_plots(
        dirs, dekad_frame, list(HYDRO_DEKADS), rc.column, rc.unit_label, max_hydro_year
    )

    if rc.volume_column is not None:
        _write_monthly_scale_plots(dirs, pre, rc.volume_column, vol_unit)
        _write_seasonal_scale_plots(dirs, pre, rc.volume_column, vol_unit)
        _write_annual_scale_plots(dirs, pre, rc.volume_column, vol_unit)


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 (v24/v25): Daily-only per-calendar-month day-grids
# ─────────────────────────────────────────────────────────────────────────────
_DAILY_DETAILED_DIR_NAMES: tuple[str, ...] = (
    "BoxWhisker",
    "HistogramKDE",
    "Violin",
    "ParametricBounds",
    "RobustBounds",
    "Anomalies",
    "ITAScatter",
    "DecadalBlocks",
    "RecentVsLongTerm",
    "DurationCurve",
    "FloodHeatmap",
    "FloodOverlay",
)


def _months_periods(by_period: dict[str, pd.Series]) -> dict[str, list[str]]:
    """``{month: [period, ...]}``, each month's periods day-ordered."""
    grouped: dict[str, list[str]] = {}
    for period in by_period:
        grouped.setdefault(period.split("-")[0], []).append(period)
    for periods in grouped.values():
        periods.sort(key=lambda p: int(p.split("-")[1]))
    return grouped


def _write_distribution_style_month_grids(
    out_dir: Path,
    dir_name: str,
    title_template: str,
    draw_fn: Callable[..., None],
    by_period: dict[str, pd.Series],
    months_periods: dict[str, list[str]],
    unit_label: str,
) -> None:
    for month in HYDRO_MONTHS:
        periods = months_periods.get(month, [])
        if not periods:
            continue
        day_data = {i + 1: by_period[p] for i, p in enumerate(periods)}
        title = title_template.format(month=FULL_MONTHS[month], unit=unit_label)

        def cell(
            ax: Axes, day_num: int, _data: dict[int, pd.Series] = day_data
        ) -> None:
            draw_fn(ax, day_num, _data[day_num], unit_label=unit_label)

        fig = build_day_grid(len(periods), title=title, cell_draw_fn=cell)
        _save_static(fig, out_dir / dir_name / f"Daily_{dir_name}_{month}.png", dpi=180)


def _day_years_values(
    periods: list[str], by_period: dict[str, pd.Series]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    day_years = {i + 1: by_period[p].index.to_numpy() for i, p in enumerate(periods)}
    day_values = {i + 1: by_period[p].to_numpy() for i, p in enumerate(periods)}
    return day_years, day_values


def _write_parametric_bounds_month_grids(
    out_dir: Path,
    by_period: dict[str, pd.Series],
    months_periods: dict[str, list[str]],
    unit_label: str,
) -> None:
    for month in HYDRO_MONTHS:
        periods = months_periods.get(month, [])
        if not periods:
            continue
        day_years, day_values = _day_years_values(periods, by_period)
        title = (
            f"Daily Inflow ({unit_label}) — Parametric Bounds — {FULL_MONTHS[month]}"
        )

        def cell(
            ax: Axes,
            day_num: int,
            _dy: dict[int, np.ndarray] = day_years,
            _dv: dict[int, np.ndarray] = day_values,
        ) -> None:
            years, values = _dy[day_num], _dv[day_num]
            sen = sens_slope(values) if len(values) >= 2 else None
            bp_idx = bai_perron_change_point(values)
            cp_year = (
                float(years[bp_idx])
                if bp_idx is not None and bp_idx < len(years)
                else None
            )
            draw_parametric_bounds_cell(
                ax,
                day_num,
                years,
                values,
                unit_label=unit_label,
                sen_slope=sen,
                change_point_year=cp_year,
            )

        fig = build_day_grid(len(periods), title=title, cell_draw_fn=cell)
        _save_static(
            fig,
            out_dir / "ParametricBounds" / f"Daily_ParametricBounds_{month}.png",
            dpi=180,
        )


def _write_robust_bounds_month_grids(
    out_dir: Path,
    by_period: dict[str, pd.Series],
    months_periods: dict[str, list[str]],
    unit_label: str,
) -> None:
    for month in HYDRO_MONTHS:
        periods = months_periods.get(month, [])
        if not periods:
            continue
        day_years, day_values = _day_years_values(periods, by_period)
        title = f"Daily Inflow ({unit_label}) — Robust Bounds — {FULL_MONTHS[month]}"

        def cell(
            ax: Axes,
            day_num: int,
            _dy: dict[int, np.ndarray] = day_years,
            _dv: dict[int, np.ndarray] = day_values,
        ) -> None:
            years, values = _dy[day_num], _dv[day_num]
            sen = sens_slope(values) if len(values) >= 2 else None
            pettitt = pettitt_test(values)
            cp_idx = pettitt.index
            cp_year = (
                float(years[cp_idx])
                if cp_idx is not None and cp_idx < len(years)
                else None
            )
            draw_robust_bounds_cell(
                ax,
                day_num,
                years,
                values,
                unit_label=unit_label,
                sen_slope=sen,
                change_point_year=cp_year,
            )

        fig = build_day_grid(len(periods), title=title, cell_draw_fn=cell)
        _save_static(
            fig, out_dir / "RobustBounds" / f"Daily_RobustBounds_{month}.png", dpi=180
        )


def _write_trend_style_month_grids(
    out_dir: Path,
    dir_name: str,
    title_template: str,
    draw_fn: Callable[..., None],
    by_period: dict[str, pd.Series],
    months_periods: dict[str, list[str]],
    unit_label: str,
    *,
    needs_unit: bool,
) -> None:
    for month in HYDRO_MONTHS:
        periods = months_periods.get(month, [])
        if not periods:
            continue
        day_years, day_values = _day_years_values(periods, by_period)
        title = title_template.format(month=FULL_MONTHS[month], unit=unit_label)

        def cell(
            ax: Axes,
            day_num: int,
            _dy: dict[int, np.ndarray] = day_years,
            _dv: dict[int, np.ndarray] = day_values,
        ) -> None:
            years, values = _dy[day_num], _dv[day_num]
            if needs_unit:
                draw_fn(ax, day_num, years, values, unit_label=unit_label)
            else:
                draw_fn(ax, day_num, values)

        fig = build_day_grid(len(periods), title=title, cell_draw_fn=cell)
        _save_static(fig, out_dir / dir_name / f"Daily_{dir_name}_{month}.png", dpi=180)


def _write_daily_detailed_plots(
    output_dir: Path,
    pre: PreprocessedData,
    value_col: str,
    unit_label: str,
    max_hydro_year: int,
) -> None:
    """Section 13 (v24/v25): the only Daily-scale plots the source's v24+
    "reactivated everything" comment doesn't undo -- the richest, most novel
    piece: a grid per calendar month, one cell per calendar day, across 9
    cell types, plus 3 Daily-specific flood/duration plots (see
    :mod:`hydrotrends.viz.plotting`'s ``daily_*`` functions for how these
    differ from Section 12's generic 5-scale versions).
    """
    root = output_dir / "Daily_Detailed_Plots_v24" / unit_label.replace(" ", "")
    for name in _DAILY_DETAILED_DIR_NAMES:
        (root / name).mkdir(parents=True, exist_ok=True)

    hydro_pivot = pre.hydro.pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_PERIOD, values=value_col, aggfunc="first"
    )
    by_period = {str(p): hydro_pivot[p].dropna() for p in hydro_pivot.columns}
    months_periods = _months_periods(by_period)
    hydro_periods = [p for month in HYDRO_MONTHS for p in months_periods.get(month, [])]

    _write_distribution_style_month_grids(
        root,
        "BoxWhisker",
        "Daily Inflow Box-and-Whisker Plots for {month} ({unit})",
        draw_boxwhisker_cell,
        by_period,
        months_periods,
        unit_label,
    )
    _write_distribution_style_month_grids(
        root,
        "HistogramKDE",
        "Daily Inflow Histogram and KDE Plots for {month} ({unit})",
        draw_histogram_kde_cell,
        by_period,
        months_periods,
        unit_label,
    )
    _write_distribution_style_month_grids(
        root,
        "Violin",
        "Daily Violin Plots for {month} Inflow Data ({unit})",
        draw_violin_cell,
        by_period,
        months_periods,
        unit_label,
    )

    _write_parametric_bounds_month_grids(root, by_period, months_periods, unit_label)
    _write_robust_bounds_month_grids(root, by_period, months_periods, unit_label)
    _write_trend_style_month_grids(
        root,
        "Anomalies",
        "Daily Inflow Anomalies ({unit}) — {month}",
        draw_anomaly_cell,
        by_period,
        months_periods,
        unit_label,
        needs_unit=True,
    )
    _write_trend_style_month_grids(
        root,
        "ITAScatter",
        "Daily Inflow ITA Scatter Plots for {month} ({unit})",
        draw_ita_scatter_cell,
        by_period,
        months_periods,
        unit_label,
        needs_unit=False,
    )
    _write_trend_style_month_grids(
        root,
        "DecadalBlocks",
        "Daily Mean Inflow Plots by Decade for {month} ({unit})",
        draw_decadal_blocks_cell,
        by_period,
        months_periods,
        unit_label,
        needs_unit=True,
    )
    _write_trend_style_month_grids(
        root,
        "RecentVsLongTerm",
        "Daily Average Mean Inflow by Recent and Long-term ({unit}) — {month}",
        draw_recent_vs_longterm_cell,
        by_period,
        months_periods,
        unit_label,
        needs_unit=True,
    )

    flood_limits = flood_limits_for_unit(unit_label)
    all_values = (
        pd.concat(list(by_period.values())) if by_period else pd.Series(dtype="float64")
    )
    fig = daily_duration_curve_static(
        all_values, unit_label=unit_label, flood_limits=flood_limits
    )
    _save_static(fig, root / "DurationCurve" / "Daily_Duration_Curve.png", dpi=300)
    fig_i = daily_duration_curve_interactive(
        all_values, unit_label=unit_label, flood_limits=flood_limits
    )
    _save_interactive(fig_i, root / "DurationCurve" / "Daily_Duration_Curve.html")

    exceed_rows = {
        p: exceedance_counts(by_period.get(p, pd.Series(dtype="float64")), flood_limits)
        for p in hydro_periods
    }
    exceed_counts = pd.DataFrame.from_dict(exceed_rows, orient="index")
    fig = daily_flood_heatmap(exceed_counts, hydro_periods)
    _save_static(fig, root / "FloodHeatmap" / "Daily_Flood_Heatmap.png", dpi=200)
    fig_i = daily_flood_heatmap_interactive(exceed_counts, hydro_periods)
    _save_interactive(fig_i, root / "FloodHeatmap" / "Daily_Flood_Heatmap.html")

    mean_s = pd.Series({p: s.mean() for p, s in by_period.items()})
    q1_s = pd.Series({p: s.quantile(0.25) for p, s in by_period.items()})
    q3_s = pd.Series({p: s.quantile(0.75) for p, s in by_period.items()})
    min_s = pd.Series({p: s.min() for p, s in by_period.items()})
    max_s = pd.Series({p: s.max() for p, s in by_period.items()})
    std_s = pd.Series({p: s.std(ddof=1) for p, s in by_period.items()})
    recent = (
        hydro_pivot.loc[max_hydro_year]
        if max_hydro_year in hydro_pivot.index
        else pd.Series(dtype="float64")
    )
    fig = daily_flood_overlay(
        hydro_periods,
        mean_s,
        recent,
        hydro_year_label(max_hydro_year),
        q1_s,
        q3_s,
        min_s,
        max_s,
        std_s,
        unit_label=unit_label,
        flood_limits=flood_limits,
    )
    _save_static(fig, root / "FloodOverlay" / "Daily_Flood_Overlay.png", dpi=200)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 (v22): whole-series 6-plot trend suite for every individual
# series across all 5 scales, plus the 5 summary overview plots and the
# aggregated Mean_Shifts_Summary workbook.
# ─────────────────────────────────────────────────────────────────────────────
_TREND_SUITE_MIN_YEARS = 10  # matches the source's `_plot_timeseries_suite` guard


def _write_trend_suite(
    out_dir: Path,
    file_prefix: str,
    scale_name: str,
    years: np.ndarray,
    values: np.ndarray,
    unit_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Write one series' whole-series 6-plot trend suite (Parametric/Robust
    Bounds, Anomalies, ITA Scatter, Decadal Blocks, Sliding Windows) -- the
    source's ``_plot_timeseries_suite``. Returns this series' decadal-block
    and sliding-window summary rows for the aggregated Mean_Shifts_Summary
    export (both empty if there's under 10 years of data to plot at all).
    """
    if len(years) < _TREND_SUITE_MIN_YEARS:
        return [], []

    sen = sens_slope(values)
    bp_idx = bai_perron_change_point(values)
    bp_year = (
        float(years[bp_idx]) if bp_idx is not None and bp_idx < len(years) else None
    )
    pt_idx = pettitt_test(values).index
    pt_year = (
        float(years[pt_idx]) if pt_idx is not None and pt_idx < len(years) else None
    )
    unit = f" ({unit_label})" if unit_label else ""

    fig = parametric_bounds_plot(
        years,
        values,
        title=f"{scale_name} Inflow{unit} — Parametric Bounds",
        unit_label=unit_label,
        sen_slope=sen,
        change_point_year=bp_year,
    )
    _save_static(fig, out_dir / f"{file_prefix}_1_Parametric.png", dpi=300)

    fig = robust_bounds_plot(
        years,
        values,
        title=f"{scale_name} Inflow{unit} — Robust Bounds",
        unit_label=unit_label,
        sen_slope=sen,
        change_point_year=pt_year,
    )
    _save_static(fig, out_dir / f"{file_prefix}_2_Robust.png", dpi=300)

    fig = anomaly_bar_chart(
        years, values, title=f"{scale_name} Anomalies{unit}", unit_label=unit_label
    )
    _save_static(fig, out_dir / f"{file_prefix}_3_Anomalies.png", dpi=300)

    ita = innovative_trend_analysis(values)
    if len(ita.first_half) > 2:  # matches the source's `half_plot > 2` guard
        fig = ita_scatter(
            ita, title=f"ITA Scatter — {scale_name}", unit_label=unit_label
        )
        _save_static(fig, out_dir / f"{file_prefix}_4_ITA_Scatter.png", dpi=300)

    decadal_summary = compute_decadal_summary(years, values)
    overall_mean = float(np.asarray(values, dtype="float64").mean())
    fig = decadal_blocks_bar_chart(
        decadal_summary,
        overall_mean=overall_mean,
        title=f"Average Mean Value by Decade — {scale_name}{unit}",
        unit_label=unit_label,
    )
    _save_static(fig, out_dir / f"{file_prefix}_5_Decadal_Blocks.png", dpi=300)

    window_summary = compute_sliding_window_summary(years, values)
    fig = sliding_windows_bar_chart(
        window_summary,
        title=f"Average Mean Value by Recent Mean and Long-term Mean — {scale_name}",
        unit_label=unit_label,
    )
    _save_static(fig, out_dir / f"{file_prefix}_6_Sliding_Windows.png", dpi=300)

    dec_rows = [
        {
            "Scale": scale_name,
            "Decade_Block": str(row["Decade"]),
            "Years_Count": int(row["Years"]),
            f"Mean_Value_{unit_label}": float(row["Mean"]),
        }
        for _, row in decadal_summary.iterrows()
    ]
    win_rows = [
        {
            "Scale": scale_name,
            "Period": row["Period"],
            "Span": row["Span"],
            "Years_Included": int(row["Years_Included"]),
            f"Mean_Value_{unit_label}": float(row["Mean"]),
        }
        for _, row in window_summary.iterrows()
    ]
    return dec_rows, win_rows


_TrendSuiteRows = tuple[list[dict[str, object]], list[dict[str, object]]]


def _write_annual_trend_suite(
    pre: PreprocessedData, value_col: str, unit_label: str, out_dir: Path
) -> _TrendSuiteRows:
    annual = hydro_seasonal_volumes(pre.hydro)["Annual"][value_col].dropna()
    annual = annual.sort_index()
    return _write_trend_suite(
        out_dir,
        "Ann",
        "Annual",
        annual.index.to_numpy(dtype="float64"),
        annual.to_numpy(dtype="float64"),
        unit_label,
    )


def _write_seasonal_trend_suites(
    pre: PreprocessedData, value_col: str, unit_label: str, out_dir: Path
) -> _TrendSuiteRows:
    """One suite per cropping season, excluding Annual (kept in the Annual
    suite instead) -- matches the source's ``if s_name == "Annual": continue``.
    """
    seasonal = hydro_seasonal_volumes(pre.hydro)
    dec_rows: list[dict[str, object]] = []
    win_rows: list[dict[str, object]] = []
    for key in _SEASON_ORDER:
        s = seasonal[key][value_col].dropna().sort_index()
        d, w = _write_trend_suite(
            out_dir,
            f"Seas_{key}",
            f"Seasonal ({key.replace('_', ' ')})",
            s.index.to_numpy(dtype="float64"),
            s.to_numpy(dtype="float64"),
            unit_label,
        )
        dec_rows += d
        win_rows += w
    return dec_rows, win_rows


def _write_monthly_trend_suites(
    pre: PreprocessedData, value_col: str, unit_label: str, out_dir: Path
) -> _TrendSuiteRows:
    monthly_pivot = monthly_volumes(pre.hydro).pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_MONTH, values=value_col, aggfunc="first"
    )
    dec_rows: list[dict[str, object]] = []
    win_rows: list[dict[str, object]] = []
    for month in HYDRO_MONTHS:
        if month not in monthly_pivot.columns:
            continue
        s = monthly_pivot[month].dropna().sort_index()
        d, w = _write_trend_suite(
            out_dir,
            f"Mon_{month}",
            f"Monthly ({month})",
            s.index.to_numpy(dtype="float64"),
            s.to_numpy(dtype="float64"),
            unit_label,
        )
        dec_rows += d
        win_rows += w
    return dec_rows, win_rows


def _write_dekadal_trend_suites(
    dekad_frame: pd.DataFrame, value_col: str, unit_label: str, out_dir: Path
) -> _TrendSuiteRows:
    dek_pivot = dekad_frame.pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_PERIOD, values=value_col, aggfunc="first"
    )
    dec_rows: list[dict[str, object]] = []
    win_rows: list[dict[str, object]] = []
    for period in HYDRO_DEKADS:
        if period not in dek_pivot.columns:
            continue
        s = dek_pivot[period].dropna().sort_index()
        d, w = _write_trend_suite(
            out_dir,
            f"Dek_{period}",
            f"10-Daily ({period})",
            s.index.to_numpy(dtype="float64"),
            s.to_numpy(dtype="float64"),
            unit_label,
        )
        dec_rows += d
        win_rows += w
    return dec_rows, win_rows


def _write_daily_trend_suites(
    pre: PreprocessedData, value_col: str, unit_label: str, out_dir: Path
) -> _TrendSuiteRows:
    """366 whole-series suites, one per calendar day -- the very large,
    opt-out-able piece gated by ``generate_plots``'s
    ``include_daily_period_suites``.
    """
    hydro_pivot = pre.hydro.pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_PERIOD, values=value_col, aggfunc="first"
    )
    dec_rows: list[dict[str, object]] = []
    win_rows: list[dict[str, object]] = []
    for period in HYDRO_PERIODS:
        if period not in hydro_pivot.columns:
            continue
        s = hydro_pivot[period].dropna().sort_index()
        d, w = _write_trend_suite(
            out_dir,
            f"Daily_{period}",
            f"Daily ({period})",
            s.index.to_numpy(dtype="float64"),
            s.to_numpy(dtype="float64"),
            unit_label,
        )
        dec_rows += d
        win_rows += w
    return dec_rows, win_rows


def _write_mean_shifts_summary(
    out_dir: Path,
    unit_label: str,
    dec_rows: list[dict[str, object]],
    win_rows: list[dict[str, object]],
) -> None:
    """The aggregated ``Mean_Shifts_Summary_{unit}.xlsx`` -- every trend
    suite's Decadal-Blocks/Sliding-Windows rows, one workbook per volume unit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"Mean_Shifts_Summary_{unit_label}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(dec_rows).to_excel(
            writer, sheet_name="Decadal_Blocks", index=False
        )
        pd.DataFrame(win_rows).to_excel(
            writer, sheet_name="Sliding_Windows", index=False
        )


def _write_summary_heatmaps(
    pre: PreprocessedData, rc: ReportColumn, out_dir: Path, *, is_daily: bool
) -> None:
    """The source's 7A-7E summary overview plots: one multi-decadal LOWESS
    chart (Volume) plus 4 "Monotonic Trends" heatmaps -- Seasonal (Volume),
    Monthly (Volume), Dekadal (Flow), Daily (Flow, daily input only).

    The Dekadal/Daily heatmaps use the *flow* column/unit even though the
    Dekadal_Plots whole-series suite above is Volume-based -- a genuine
    asymmetry in the source script (its "Dekadal_Plots" suite plots
    ``vol_dek_hydro_pivot`` but its Dekadal summary heatmap uses the
    flow-based 10-daily trend results), reproduced here rather than
    "corrected" for consistency.
    """
    vol_col = rc.volume_column
    if vol_col is None:
        return
    vol_unit = rc.volume_unit_label or ""

    seasonal = hydro_seasonal_volumes(pre.hydro)
    annual = seasonal["Annual"][vol_col].dropna().sort_index()
    series_by_season = {key: seasonal[key][vol_col].dropna() for key in _SEASON_ORDER}
    fig = seasonal_divergence_lowess_chart(
        annual.index.to_numpy(), series_by_season, unit_label=vol_unit
    )
    _save_static(fig, out_dir / f"Summary_Seasonal_Shifts_{vol_unit}.png", dpi=300)

    trend_by_season: dict[str, tuple[str, str]] = {}
    for key, frames in seasonal.items():
        s = frames[vol_col].dropna().sort_index()
        row = analyze_series(s.to_numpy(), years=s.index.to_numpy())
        if "trend" in row:
            trend_by_season[key] = (row["trend"], row["significance"])
    fig = seasonal_trend_heatmap(trend_by_season, unit_label=vol_unit)
    _save_static(fig, out_dir / f"Summary_Seasonal_Heatmap_{vol_unit}.png", dpi=300)

    monthly_pivot = monthly_volumes(pre.hydro).pivot_table(
        index=COL_HYDRO_YEAR, columns=COL_MONTH, values=vol_col, aggfunc="first"
    )
    monthly_rows: dict[str, dict[str, object]] = {}
    for month in HYDRO_MONTHS:
        if month not in monthly_pivot.columns:
            continue
        s = monthly_pivot[month].dropna().sort_index()
        row = analyze_series(s.to_numpy(), years=s.index.to_numpy())
        if "trend" in row:
            monthly_rows[month] = row
    if monthly_rows:
        monthly_results = pd.DataFrame.from_dict(monthly_rows, orient="index")
        fig = monthly_trend_heatmap(monthly_results, unit_label=vol_unit)
        _save_static(fig, out_dir / f"Summary_Monthly_Heatmap_{vol_unit}.png", dpi=300)

    dekadal_results = (
        analyze_10daily_from_daily(pre.hydro, value_col=rc.column)
        if is_daily
        else analyze_preprocessed(pre, value_col=rc.column)
    )
    fig = dekadal_trend_heatmap(dekadal_results, unit_label=rc.unit_label)
    dekadal_unit_tag = rc.unit_label.replace(" ", "")
    _save_static(
        fig, out_dir / f"Summary_Dekadal_Heatmap_{dekadal_unit_tag}.png", dpi=300
    )

    if is_daily:
        daily_results = analyze_preprocessed(pre, value_col=rc.column)
        fig = daily_trend_heatmap(daily_results, unit_label=rc.unit_label)
        daily_unit_tag = rc.unit_label.replace(" ", "")
        _save_static(
            fig, out_dir / f"Summary_Daily_Heatmap_{daily_unit_tag}.png", dpi=300
        )


def _write_all_trend_plots(
    pre: PreprocessedData,
    rc: ReportColumn,
    output_dir: Path,
    *,
    is_daily: bool,
    include_daily_period_suites: bool,
) -> None:
    """Section 11 (v22): the whole-series 6-plot trend suite for every
    Annual/Seasonal/Monthly/10-Daily series (always) and every Daily series
    (only when ``include_daily_period_suites`` -- 366 extra series, matching
    the source's full parity but gated behind an opt-out for the very large
    extra cost that entails), plus the 5 summary overview plots and the
    aggregated Mean_Shifts_Summary workbook. Needs a volume-paired column
    (see :class:`~hydrotrends.api.ReportColumn`); a no-op without one, since
    every whole-series suite here is Volume-based except the Daily one.
    """
    if rc.volume_column is None:
        return
    vol_unit = rc.volume_unit_label or ""
    root = output_dir / f"Plots_Volume_{vol_unit}"
    scale_dirs = {
        "annual": root / "Annual_Plots",
        "seasonal": root / "Seasonal_Plots",
        "monthly": root / "Monthly_Plots",
        "dekadal": root / "Dekadal_Plots",
        "daily": root / "Daily_Plots",
        "summary": root / "Summary_Heatmaps",
    }
    for path in scale_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    dec_rows: list[dict[str, object]] = []
    win_rows: list[dict[str, object]] = []

    d, w = _write_annual_trend_suite(
        pre, rc.volume_column, vol_unit, scale_dirs["annual"]
    )
    dec_rows += d
    win_rows += w
    d, w = _write_seasonal_trend_suites(
        pre, rc.volume_column, vol_unit, scale_dirs["seasonal"]
    )
    dec_rows += d
    win_rows += w
    d, w = _write_monthly_trend_suites(
        pre, rc.volume_column, vol_unit, scale_dirs["monthly"]
    )
    dec_rows += d
    win_rows += w

    dekad_frame = dekads_from_daily(pre.hydro) if is_daily else pre.hydro
    d, w = _write_dekadal_trend_suites(
        dekad_frame, rc.volume_column, vol_unit, scale_dirs["dekadal"]
    )
    dec_rows += d
    win_rows += w

    if is_daily and include_daily_period_suites:
        d, w = _write_daily_trend_suites(
            pre, rc.column, rc.unit_label, scale_dirs["daily"]
        )
        dec_rows += d
        win_rows += w

    _write_mean_shifts_summary(root, vol_unit, dec_rows, win_rows)
    _write_summary_heatmaps(pre, rc, scale_dirs["summary"], is_daily=is_daily)


def generate_plots(
    pre: PreprocessedData,
    output_dir: str | Path,
    *,
    columns: list[ReportColumn],
    include_daily_period_suites: bool = True,
) -> Path:
    """Write the Section-12 distribution/duration/flood-exceedance plot suite,
    Section 13's Daily-only per-calendar-month day-grids, and Section 11's
    whole-series trend suites + summary heatmaps.

    For each :class:`~hydrotrends.api.ReportColumn`: a distribution histogram
    grid + box-whisker grid per scale (Daily split into one grid per
    calendar month; 10-Daily/Monthly/Seasonal each a single grid; Annual a
    single histogram), a duration curve (static + interactive HTML) per
    scale, and -- Daily/10-Daily only, where flood limits are physically
    meaningful for a flow unit -- a flood-exceedance overlay and heatmap
    (static + interactive). Monthly/Seasonal/Annual plots are only produced
    when the column is volume-paired (see :class:`~hydrotrends.api.ReportColumn`).

    When ``pre`` is a daily record, also writes Section 13's Daily-only
    per-calendar-month day-grids (flow-unit column only, one grid per month
    per cell type, plus the 3 Daily-specific flood/duration plots) --
    skipped for a native-10-daily input, same as Section 12's Daily scale,
    since there's no day-level data to derive it from.

    When the column is volume-paired, also writes Section 11's whole-series
    6-plot trend suite (Parametric/Robust Bounds, Anomalies, ITA Scatter,
    Decadal Blocks, Sliding Windows) for every Annual/Seasonal/Monthly/
    10-Daily series, the aggregated ``Mean_Shifts_Summary_{unit}.xlsx``, and
    the 5 summary overview plots. A daily input also gets one trend suite per
    calendar day (366 series) unless ``include_daily_period_suites=False`` --
    full source parity by default, with an opt-out for the very large extra
    cost (up to ~2,200 additional PNGs) that full parity entails.

    Files are written under
    ``<output_dir>/Distribution_and_Flood_Plots/<flow_unit>[_<vol_unit>]/{Daily,10Daily,Monthly,Seasonal,Annual}/``,
    ``<output_dir>/Daily_Detailed_Plots_v24/<flow_unit>/<PlotType>/`` (daily
    input only), and
    ``<output_dir>/Plots_Volume_<vol_unit>/{Annual,Seasonal,Monthly,Dekadal,Daily}_Plots/``
    plus ``.../Summary_Heatmaps/`` and ``.../Mean_Shifts_Summary_<vol_unit>.xlsx``
    (volume-paired columns only). Returns ``output_dir``.
    """
    if not columns:
        raise ValueError("generate_plots needs at least one ReportColumn")
    out = Path(output_dir)
    is_daily = pre.resolution is TimeResolution.DAILY

    for rc in columns:
        _write_distribution_and_flood_plots(pre, rc, out, is_daily=is_daily)
        if is_daily:
            max_hydro_year = int(pre.hydro[COL_HYDRO_YEAR].max())
            _write_daily_detailed_plots(
                out, pre, rc.column, rc.unit_label, max_hydro_year
            )
        _write_all_trend_plots(
            pre,
            rc,
            out,
            is_daily=is_daily,
            include_daily_period_suites=include_daily_period_suites,
        )

    return out
