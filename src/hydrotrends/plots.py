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

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .api import ReportColumn
from .core.constants import (
    COL_HYDRO_YEAR,
    COL_PERIOD,
    HYDRO_DEKADS,
    HYDRO_MONTHS,
    TimeResolution,
)
from .core.utils import flood_limits_for_unit, hydro_year_label
from .data.preprocessing import (
    PreprocessedData,
    dekads_from_daily,
    hydro_seasonal_volumes,
    monthly_volumes,
)
from .stats.frequency import exceedance_counts
from .viz.plotting import (
    boxwhisker_grid,
    distribution_grid,
    flood_heatmap,
    flood_heatmap_interactive,
    flood_overlay_interactive,
    flood_overlay_static,
    flow_duration_curve_interactive,
    flow_duration_curve_static,
    single_histogram,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go
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


def generate_plots(
    pre: PreprocessedData,
    output_dir: str | Path,
    *,
    columns: list[ReportColumn],
) -> Path:
    """Write the Section-12 distribution/duration/flood-exceedance plot suite.

    For each :class:`~hydrotrends.api.ReportColumn`: a distribution histogram
    grid + box-whisker grid per scale (Daily split into one grid per
    calendar month; 10-Daily/Monthly/Seasonal each a single grid; Annual a
    single histogram), a duration curve (static + interactive HTML) per
    scale, and -- Daily/10-Daily only, where flood limits are physically
    meaningful for a flow unit -- a flood-exceedance overlay and heatmap
    (static + interactive). Monthly/Seasonal/Annual plots are only produced
    when the column is volume-paired (see :class:`~hydrotrends.api.ReportColumn`).

    Files are written under
    ``<output_dir>/Distribution_and_Flood_Plots/<flow_unit>[_<vol_unit>]/{Daily,10Daily,Monthly,Seasonal,Annual}/``.
    Returns ``output_dir``.
    """
    if not columns:
        raise ValueError("generate_plots needs at least one ReportColumn")
    out = Path(output_dir)
    is_daily = pre.resolution is TimeResolution.DAILY

    for rc in columns:
        _write_distribution_and_flood_plots(pre, rc, out, is_daily=is_daily)

    return out
