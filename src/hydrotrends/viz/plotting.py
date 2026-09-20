"""Plot builders for :mod:`hydrotrends`.

Each function *returns* a figure (a Matplotlib ``Figure`` or a Plotly
``go.Figure``) — it never calls ``.show()``, writes a file, or touches global
state. Displaying, saving, and embedding are the caller's job, which keeps these
usable from scripts, notebooks, tests, and the reporting layer alike.

Matplotlib figures are built with :class:`matplotlib.figure.Figure` directly
(not via ``pyplot``) so that generating many figures in one run doesn't leak
them into pyplot's global registry.

This module is built up in pieces; this part covers the time-series views.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as scipy_stats

from ..core.constants import COL_DATE, FLOOD_CLASSES, FLOOD_COLORS, LOWESS_FRAC
from ..core.utils import significance_stars
from ..core.validation import ArrayLike
from ..stats.frequency import (
    COL_FDC_EXCEEDANCE,
    COL_FDC_VALUE,
    flow_duration_curve,
)
from ..stats.ita import ITAResult
from ..stats.trends import LinearFit, MannKendallResult, SenSlope

__all__ = [
    "timeseries_interactive",
    "timeseries_static",
    "ita_scatter",
    "flow_duration_curve_static",
    "flow_duration_curve_interactive",
    "trend_scatter",
    "compute_lowess",
    "parametric_bounds_plot",
    "robust_bounds_plot",
    "distribution_grid",
    "boxwhisker_grid",
    "single_histogram",
    "flood_heatmap",
    "flood_heatmap_interactive",
    "flood_overlay_static",
    "flood_overlay_interactive",
    "day_grid_layout",
    "build_day_grid",
    "hist_mode",
    "draw_boxwhisker_cell",
    "draw_histogram_kde_cell",
    "draw_violin_cell",
    "draw_parametric_bounds_cell",
    "draw_robust_bounds_cell",
    "draw_anomaly_cell",
    "draw_ita_scatter_cell",
    "draw_decadal_blocks_cell",
    "draw_recent_vs_longterm_cell",
    "daily_duration_curve_static",
    "daily_duration_curve_interactive",
    "daily_flood_heatmap",
    "daily_flood_heatmap_interactive",
    "daily_flood_overlay",
]

_LINE_COLOR = "#1f4e79"
_TEMPLATE = "plotly_white"

# Plotly range-selector buttons (from the reference notebook).
_RANGE_BUTTONS = [
    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
    {"count": 5, "label": "5y", "step": "year", "stepmode": "backward"},
    {"count": 10, "label": "10y", "step": "year", "stepmode": "backward"},
    {"step": "all"},
]


def timeseries_interactive(
    df: pd.DataFrame,
    *,
    value_col: str,
    date_col: str = COL_DATE,
    title: str | None = None,
    y_label: str | None = None,
    line_color: str = _LINE_COLOR,
) -> go.Figure:
    """Interactive Plotly time series with range slider + selector and unified hover.

    Parameters
    ----------
    df:
        Frame containing ``date_col`` (datetime) and ``value_col``.
    value_col, date_col:
        Column names to plot on the y and x axes.
    title, y_label:
        Optional labels; ``y_label`` defaults to ``value_col``.
    line_color:
        Trace colour.
    """
    fig = px.line(df, x=date_col, y=value_col, template=_TEMPLATE)
    fig.update_traces(line={"color": line_color, "width": 2})
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={"buttons": _RANGE_BUTTONS},
        tickformat="%d %b\n%Y",
    )
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title=y_label or value_col,
        hovermode="x unified",
    )
    return fig


def timeseries_static(
    df: pd.DataFrame,
    *,
    value_col: str,
    date_col: str = COL_DATE,
    title: str | None = None,
    y_label: str | None = None,
    line_color: str = _LINE_COLOR,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> Figure:
    """Static Matplotlib time series (returns a standalone ``Figure``)."""
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(df[date_col], df[value_col], color=line_color, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label or value_col)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig


_TERCILE_COLORS = ("#2ca02c", "#ff7f0e", "#d62728")  # low / medium / high
_TERCILE_LABELS = ("Low third", "Medium third", "High third")


def ita_scatter(
    result: ITAResult,
    *,
    title: str | None = None,
    unit_label: str = "",
    figsize: tuple[float, float] = (6.0, 6.0),
    point_color: str = "#1f77b4",
    highlight_terciles: bool = False,
) -> Figure:
    """Innovative Trend Analysis scatter of sorted first vs second half.

    Points on the 1:1 line indicate no trend; above it, increasing. The
    dashed "ITA trend shift" line is the 1:1 line offset by the mean difference
    between halves. Consumes an :class:`~hydrotrends.stats.ita.ITAResult`
    (its ``first_half``/``second_half`` are already sorted).

    Set ``highlight_terciles`` to colour the low/medium/high thirds — the
    regimes whose sub-trends :func:`~hydrotrends.stats.ita.innovative_trend_analysis`
    reports — so opposing low- vs high-flow shifts are visible.
    """
    fig = Figure(figsize=figsize)
    ax = fig.subplots()

    xs, ys = result.first_half, result.second_half
    if xs.size == 0:
        ax.text(0.5, 0.5, "insufficient data for ITA", ha="center", va="center")
        ax.set_axis_off()
        return fig

    if highlight_terciles and xs.size >= 3:
        for group, color, label in zip(
            np.array_split(np.arange(xs.size), 3),
            _TERCILE_COLORS,
            _TERCILE_LABELS,
            strict=False,
        ):
            ax.scatter(
                xs[group],
                ys[group],
                color=color,
                edgecolor="k",
                alpha=0.75,
                zorder=3,
                label=label,
            )
    else:
        ax.scatter(
            xs,
            ys,
            color=point_color,
            edgecolor="k",
            alpha=0.7,
            zorder=3,
            label="Sorted pairs",
        )

    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    margin = (hi - lo) * 0.05
    lims = (max(0.0, lo - margin), hi + margin)

    ax.plot(lims, lims, "k--", linewidth=2, label="1:1 line (no trend)")
    ax.plot(lims, [x * 1.10 for x in lims], "g:", linewidth=1.5, label="+10%")
    ax.plot(lims, [x * 0.90 for x in lims], "r:", linewidth=1.5, label="-10%")
    mean_diff = float(ys.mean() - xs.mean())
    ax.plot(
        lims,
        [x + mean_diff for x in lims],
        color="dodgerblue",
        linestyle="--",
        linewidth=2,
        label="ITA trend shift",
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(
        f"{title or 'ITA Scatter'}\nITA Slope: {result.slope:.3f}",
        fontweight="bold",
    )
    ax.set_xlabel(f"First Half Sorted{unit}")
    ax.set_ylabel(f"Second Half Sorted{unit}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Flow / volume duration curves
# ─────────────────────────────────────────────────────────────────────────────
_FDC_COLOR = "#1f4e79"


def _flood_annotation(label: str, threshold: float) -> str:
    """Flood-line label with just the threshold -- the plain (Section-12,
    all-5-scales) style. The Daily-only duration curve (Section 13) adds an
    exceedance-probability readout instead; see :func:`daily_duration_curve_static`.
    """
    return f"{label} ({threshold:,.0f})"


def flow_duration_curve_static(
    data: pd.Series | np.ndarray,
    *,
    title: str | None = None,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
    figsize: tuple[float, float] = (9.0, 5.5),
    line_color: str = _FDC_COLOR,
) -> Figure:
    """Static flow/volume duration curve (exceedance % on x, value on y).

    ``flood_limits`` maps a flood class to a threshold *in the data's unit*; each
    is drawn as a horizontal line annotated with its threshold value.
    """
    curve = flow_duration_curve(data)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(
        curve[COL_FDC_EXCEEDANCE],
        curve[COL_FDC_VALUE],
        color=line_color,
        linewidth=2,
    )

    if flood_limits:
        for label, threshold in flood_limits.items():
            color = FLOOD_COLORS.get(label, "gray")
            ax.axhline(threshold, linestyle=":", linewidth=1.3, color=color)
            ax.annotate(
                _flood_annotation(label, threshold),
                xy=(98, threshold),
                fontsize=8,
                color=color,
                ha="right",
                va="bottom",
            )

    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(title or f"Duration Curve{unit}", fontweight="bold")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel(f"Value{unit}")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def flow_duration_curve_interactive(
    data: pd.Series | np.ndarray,
    *,
    title: str | None = None,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
    line_color: str = _FDC_COLOR,
) -> go.Figure:
    """Interactive Plotly flow/volume duration curve."""
    curve = flow_duration_curve(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve[COL_FDC_EXCEEDANCE],
            y=curve[COL_FDC_VALUE],
            mode="lines",
            name="Duration curve",
            line={"color": line_color, "width": 2},
        )
    )
    if flood_limits:
        for label, threshold in flood_limits.items():
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color=FLOOD_COLORS.get(label, "gray"),
                annotation_text=_flood_annotation(label, threshold),
                annotation_position="right",
            )
    unit = f" ({unit_label})" if unit_label else ""
    fig.update_layout(
        title=title or f"Duration Curve{unit}",
        xaxis_title="Exceedance Probability (%)",
        yaxis_title=f"Value{unit}",
        template=_TEMPLATE,
        height=500,
        xaxis={"range": [0, 100]},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Trend scatter with Sen's-slope overlay
# ─────────────────────────────────────────────────────────────────────────────
def trend_scatter(
    x: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
    *,
    mk_result: MannKendallResult | None = None,
    sen_slope: SenSlope | None = None,
    linear_fit: LinearFit | None = None,
    title: str | None = None,
    unit_label: str = "",
    x_label: str = "Year",
    slope_unit: str = "yr",
    figsize: tuple[float, float] = (10.0, 5.0),
    point_color: str = "#1f4e79",
) -> Figure:
    """Series over time with trend overlays and significance in the title.

    The Sen's-slope line is anchored at the median point (``v26`` convention);
    the optional OLS line is anchored at the centroid (which OLS passes through).
    Pass ``mk_result`` to annotate the title with the trend direction, p-value,
    and significance stars. ``x`` should be evenly spaced (e.g. years) so the
    per-step slope maps onto the axis.
    """
    xa = np.asarray(x, dtype="float64")
    ya = np.asarray(values, dtype="float64")

    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(
        xa,
        ya,
        marker="o",
        markersize=4,
        color=point_color,
        linewidth=1.2,
        label="Observations",
    )

    if sen_slope is not None and np.isfinite(sen_slope.slope):
        intercept = float(np.median(ya) - sen_slope.slope * np.median(xa))
        ax.plot(
            xa,
            sen_slope.slope * xa + intercept,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"Sen's slope ({sen_slope.slope:.3f}/{slope_unit})",
        )

    if linear_fit is not None and np.isfinite(linear_fit.slope):
        intercept = float(ya.mean() - linear_fit.slope * xa.mean())
        ax.plot(
            xa,
            linear_fit.slope * xa + intercept,
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.8,
            label=f"OLS ({linear_fit.slope:.3f}/{slope_unit})",
        )

    unit = f" ({unit_label})" if unit_label else ""
    base = title or "Trend"
    if mk_result is not None:
        stars = significance_stars(mk_result.p_value)
        ax.set_title(
            f"{base}\n{mk_result.trend} — p={mk_result.p_value:.3g} {stars}",
            fontweight="bold",
        )
    else:
        ax.set_title(base, fontweight="bold")

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Value{unit}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Parametric / robust bounds plots (LOWESS + central-tendency band + Sen's
# slope + change-point marker) -- the source script's "Parametric Bounds" /
# "Robust Bounds" plot types.
# ─────────────────────────────────────────────────────────────────────────────
def compute_lowess(
    x: np.ndarray, y: np.ndarray, *, frac: float = LOWESS_FRAC
) -> np.ndarray:
    """Locally-weighted (tricube-kernel) linear smooth, one fit per point.

    A minimal, dependency-free LOWESS: for each ``x[i]``, fits a weighted
    linear regression to all points using a tricube weight on distance from
    ``x[i]`` (weight 0 beyond the ``frac``-fraction-nearest neighbour), and
    takes the fitted value at ``x[i]``. Ported verbatim (algorithmically) from
    the source script's own ``compute_lowess``, which this reproduces exactly
    rather than depending on ``statsmodels`` -- same reasoning as this
    package's other "self-contained" statistical methods.
    """
    n = len(x)
    y_smooth = np.zeros(n)
    reach = int(np.ceil(frac * n))
    for i in range(n):
        dist = np.abs(x - x[i])
        d_max = np.sort(dist)[reach] if reach < n else np.sort(dist)[-1]
        if d_max == 0:
            d_max = 1.0
        weights = np.clip(1 - (dist / d_max) ** 3, 0, 1) ** 3
        design = np.vstack((np.ones(n), x)).T
        weighted_design = weights[:, None] * design
        normal_matrix = design.T @ weighted_design
        try:
            beta = np.linalg.solve(normal_matrix, weighted_design.T @ y)
            y_smooth[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            y_smooth[i] = np.sum(weights * y) / np.sum(weights)
    return y_smooth


def _value_format(unit_label: str) -> str:
    return ".0f" if "Cusec" in unit_label else ".2f"


def _bounds_plot(
    years: ArrayLike,
    values: ArrayLike,
    *,
    center: str,
    title: str | None,
    default_title: str,
    unit_label: str,
    sen_slope: SenSlope | None,
    change_point_year: float | None,
    change_point_label: str,
    lowess_frac: float,
    figsize: tuple[float, float],
) -> Figure:
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")
    fmt = _value_format(unit_label)
    unit = f" ({unit_label})" if unit_label else ""

    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(
        years_arr,
        values_arr,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        alpha=0.3,
        label="Observed",
    )
    ax.plot(
        years_arr,
        compute_lowess(years_arr, values_arr, frac=lowess_frac),
        color="#9467bd",
        linewidth=3,
        label="LOWESS",
    )

    median_val = float(np.median(values_arr))
    if center == "mean":
        mean_val = float(values_arr.mean())
        std_val = float(values_arr.std(ddof=1))
        ax.axhline(
            mean_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean ({mean_val:{fmt}})",
        )
        ax.axhline(
            mean_val + 3 * std_val,
            color="lightgreen",
            linestyle="--",
            linewidth=1.5,
            label="± 3 Std Dev",
        )
        ax.axhline(
            mean_val - 3 * std_val, color="lightgreen", linestyle="--", linewidth=1.5
        )
    else:  # "median"
        q25, q75 = np.percentile(values_arr, 25), np.percentile(values_arr, 75)
        iqr = q75 - q25
        ax.axhline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median ({median_val:{fmt}})",
        )
        ax.axhline(
            median_val + 1.5 * iqr,
            color="paleturquoise",
            linestyle="--",
            linewidth=1.5,
            label="± 1.5 × IQR",
        )
        ax.axhline(
            median_val - 1.5 * iqr, color="paleturquoise", linestyle="--", linewidth=1.5
        )

    # Sen's-slope line is always anchored at the median point (the source's
    # own convention, in both the parametric and robust variants).
    if sen_slope is not None and np.isfinite(sen_slope.slope):
        intercept = median_val - sen_slope.slope * float(np.median(years_arr))
        ax.plot(
            years_arr,
            sen_slope.slope * years_arr + intercept,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"Sen's Slope ({sen_slope.slope:.3f}/yr)",
        )
    if change_point_year is not None and not math.isnan(change_point_year):
        cp = int(change_point_year)
        ax.axvline(
            x=cp,
            color="black",
            linestyle=":",
            linewidth=2,
            label=f"{change_point_label} ({cp})",
        )

    ax.set_title(title or default_title, fontweight="bold")
    ax.set_ylabel(f"Value{unit}")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def parametric_bounds_plot(
    years: ArrayLike,
    values: ArrayLike,
    *,
    title: str | None = None,
    unit_label: str = "",
    sen_slope: SenSlope | None = None,
    change_point_year: float | None = None,
    lowess_frac: float = LOWESS_FRAC,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> Figure:
    """LOWESS + mean ± 3 Std Dev band + Sen's slope + Bai-Perron break.

    The source script's "Parametric Bounds" plot. Pass a
    :class:`~hydrotrends.stats.trends.SenSlope` to draw the Sen's-slope
    overlay (anchored at the median point) and a ``change_point_year`` (e.g.
    from :func:`~hydrotrends.stats.changepoint.bai_perron_change_point`,
    mapped to a year by the caller) to mark the structural break.
    """
    unit = f" ({unit_label})" if unit_label else ""
    return _bounds_plot(
        years,
        values,
        center="mean",
        title=title,
        default_title=f"Inflow{unit} — Parametric Bounds",
        unit_label=unit_label,
        sen_slope=sen_slope,
        change_point_year=change_point_year,
        change_point_label="Bai-Perron Break",
        lowess_frac=lowess_frac,
        figsize=figsize,
    )


def robust_bounds_plot(
    years: ArrayLike,
    values: ArrayLike,
    *,
    title: str | None = None,
    unit_label: str = "",
    sen_slope: SenSlope | None = None,
    change_point_year: float | None = None,
    lowess_frac: float = LOWESS_FRAC,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> Figure:
    """LOWESS + median ± 1.5 × IQR band + Sen's slope + Pettitt break.

    The source script's "Robust Bounds" plot -- the median/IQR-based
    counterpart to :func:`parametric_bounds_plot`, less sensitive to
    outliers. Pass a ``change_point_year`` from
    :func:`~hydrotrends.stats.changepoint.pettitt_test` (mapped to a year by
    the caller) to mark the break.
    """
    unit = f" ({unit_label})" if unit_label else ""
    return _bounds_plot(
        years,
        values,
        center="median",
        title=title,
        default_title=f"Inflow{unit} — Robust Bounds",
        unit_label=unit_label,
        sen_slope=sen_slope,
        change_point_year=change_point_year,
        change_point_label="Pettitt Break",
        lowess_frac=lowess_frac,
        figsize=figsize,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Distribution grid and flood-exceedance heatmap
# ─────────────────────────────────────────────────────────────────────────────
def distribution_grid(
    data_by_period: Mapping[str, ArrayLike],
    *,
    order: list[str] | None = None,
    title: str | None = None,
    unit_label: str = "",
    ncols: int = 4,
    min_points: int = 3,
    panel_size: tuple[float, float] = (5.0, 4.2),
) -> Figure:
    """Histogram + KDE per period, arranged in a grid.

    ``data_by_period`` maps a period label to that period's values. Each panel
    shows the histogram with a KDE overlay and mean / median / +/-std lines;
    periods with fewer than ``min_points`` values are marked insufficient.
    """
    labels = order if order is not None else list(data_by_period)
    n = len(labels)
    ncols = max(1, ncols)
    nrows = max(1, math.ceil(n / ncols))

    fig = Figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    axes = np.asarray(fig.subplots(nrows, ncols, squeeze=False)).flatten()

    unit = f" ({unit_label})" if unit_label else ""
    for i, period in enumerate(labels):
        ax = axes[i]
        series = pd.Series(data_by_period.get(period, []), dtype="float64").dropna()
        if len(series) >= min_points:
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std(ddof=1))
            sns.histplot(
                series, kde=True, ax=ax, color="skyblue", stat="count", alpha=0.6
            )
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=1.3,
                label=f"Mean: {mean_val:.1f}",
            )
            ax.axvline(
                median_val,
                color="purple",
                linestyle="-",
                linewidth=1.3,
                label=f"Median: {median_val:.1f}",
            )
            ax.axvline(mean_val - std_val, color="green", linestyle=":", linewidth=1)
            ax.axvline(
                mean_val + std_val,
                color="green",
                linestyle=":",
                linewidth=1,
                label="+/-Std Dev",
            )
            ax.set_title(str(period), fontsize=11, fontweight="bold")
            ax.set_xlabel(f"Value{unit}", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        else:
            ax.set_title(f"{period} (insufficient data)", fontsize=10)
            ax.axis("off")

    for j in range(n, len(axes)):  # blank any unused panels
        axes[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def boxwhisker_grid(
    data_by_period: Mapping[str, ArrayLike],
    *,
    order: list[str] | None = None,
    title: str | None = None,
    unit_label: str = "",
    ncols: int = 4,
    min_points: int = 3,
    panel_size: tuple[float, float] = (3.2, 4.0),
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Box-and-whisker plot per period, arranged in a grid.

    ``data_by_period``/``order``/``min_points`` mirror :func:`distribution_grid`.
    ``ylim`` shares one y-axis range across every panel (the source's
    ``global_ylim``, e.g. the whole scale's min/max +/-5%) so panels are
    visually comparable; omit it to let each panel autoscale.
    """
    labels = order if order is not None else list(data_by_period)
    n = len(labels)
    ncols = max(1, ncols)
    nrows = max(1, math.ceil(n / ncols))

    fig = Figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    axes = np.asarray(fig.subplots(nrows, ncols, squeeze=False)).flatten()

    unit = f" ({unit_label})" if unit_label else ""
    for i, period in enumerate(labels):
        ax = axes[i]
        series = pd.Series(data_by_period.get(period, []), dtype="float64").dropna()
        if len(series) >= min_points:
            sns.boxplot(y=series, ax=ax, color="skyblue", width=0.5)
            ax.set_title(str(period), fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Value{unit}", fontsize=8)
            ax.set_xlabel("")
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.tick_params(labelbottom=False, labelsize=7)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.set_title(f"{period} (insufficient data)", fontsize=10)
            ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def single_histogram(
    series: ArrayLike,
    *,
    title: str | None = None,
    unit_label: str = "",
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """Histogram + KDE for a scale with only one period per year (Annual).

    Draws mean/median/+-std lines the same way :func:`distribution_grid`'s
    per-panel does, at single-plot scale.
    """
    s = pd.Series(series, dtype="float64").dropna()
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std(ddof=1))

    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    sns.histplot(s, kde=True, ax=ax, color="skyblue", stat="count", alpha=0.6)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="purple",
        linestyle="-",
        linewidth=1.5,
        label=f"Median: {median_val:.2f}",
    )
    ax.axvline(
        mean_val - std_val,
        color="green",
        linestyle=":",
        linewidth=1,
        label="+/-Std Dev",
    )
    ax.axvline(mean_val + std_val, color="green", linestyle=":", linewidth=1)

    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(title or f"Distribution{unit}", fontweight="bold")
    ax.set_xlabel(f"Value{unit}")
    ax.set_ylabel("Count (Years)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def flood_heatmap(
    counts: pd.DataFrame,
    *,
    flood_order: list[str] | None = None,
    title: str = "Flood Exceedance Counts",
    period_label: str = "Period",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Heatmap of flood-exceedance counts (flood classes as rows, periods as cols).

    ``counts`` is a frame indexed by period with one column per flood class
    (build it with :func:`hydrotrends.stats.frequency.exceedance_counts` per
    period). Rows are ordered by severity (``FLOOD_CLASSES``) unless overridden.
    """
    if flood_order is None:
        flood_order = [c for c in FLOOD_CLASSES if c in counts.columns]
    pivot = counts[flood_order].T  # rows = flood level, cols = period
    n = pivot.shape[1]

    if figsize is None:
        figsize = (min(22.0, max(12.0, n * 0.12)), 3.5)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Exceedance Count"},
        ax=ax,
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(period_label)
    ax.set_ylabel("Flood Level")

    tick_step = max(1, n // 60)
    positions = np.arange(n)[::tick_step]
    ax.set_xticks(positions + 0.5)
    ax.set_xticklabels(
        [str(pivot.columns[i]) for i in positions], rotation=90, fontsize=7
    )
    fig.tight_layout()
    return fig


def flood_heatmap_interactive(
    counts: pd.DataFrame,
    *,
    flood_order: list[str] | None = None,
    title: str = "Flood Exceedance Counts",
    period_label: str = "Period",
    height: int = 400,
) -> go.Figure:
    """Interactive Plotly counterpart to :func:`flood_heatmap`."""
    if flood_order is None:
        flood_order = [c for c in FLOOD_CLASSES if c in counts.columns]
    melted = counts.reset_index(names=period_label).melt(
        id_vars=period_label,
        value_vars=flood_order,
        var_name="Flood_Level",
        value_name="Exceedances",
    )
    fig = px.density_heatmap(
        melted,
        x=period_label,
        y="Flood_Level",
        z="Exceedances",
        title=title,
        labels={"Exceedances": "Count"},
        category_orders={"Flood_Level": flood_order},
        color_continuous_scale=[
            "white",
            "lightblue",
            "blue",
            "orange",
            "red",
            "darkred",
        ],
        height=height,
    )
    fig.update_layout(xaxis={"tickangle": -90}, template=_TEMPLATE)
    return fig


def flood_overlay_static(
    period_order: list[str],
    mean_by_period: pd.Series,
    recent_by_period: pd.Series,
    recent_label: str,
    q1_by_period: pd.Series,
    q3_by_period: pd.Series,
    *,
    title: str | None = None,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
) -> Figure:
    """Historical mean/IQR band + a recent year's series, with flood-limit lines.

    ``mean_by_period``/``q1_by_period``/``q3_by_period``/``recent_by_period``
    are all indexed by period label (reindexed to ``period_order`` here);
    build them from the raw per-period values before calling.
    """
    n = len(period_order)
    fig_width = min(22.0, max(12.0, n * 0.12))
    fig = Figure(figsize=(fig_width, 6.0))
    ax = fig.subplots()
    x = np.arange(n)

    ax.fill_between(
        x,
        q1_by_period.reindex(period_order),
        q3_by_period.reindex(period_order),
        color="steelblue",
        alpha=0.15,
        label="Q1-Q3 Range",
    )
    ax.plot(
        x,
        mean_by_period.reindex(period_order),
        color="navy",
        linewidth=2.5,
        label="Historical Mean",
    )
    ax.plot(
        x,
        recent_by_period.reindex(period_order),
        color="red",
        linewidth=2,
        label=f"Recent Year ({recent_label})",
    )
    if flood_limits:
        for label, threshold in flood_limits.items():
            color = FLOOD_COLORS.get(label, "gray")
            ax.axhline(threshold, linestyle=":", linewidth=1.2, color=color)
            ax.annotate(
                _flood_annotation(label, threshold),
                xy=(n - 1, threshold),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="left",
                va="center",
            )

    tick_step = max(1, n // 60)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(
        [str(period_order[i]) for i in x[::tick_step]], rotation=90, fontsize=7
    )
    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(title or f"Mean Inflow with Flood Limits{unit}", fontweight="bold")
    ax.set_ylabel(f"Inflow{unit}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def flood_overlay_interactive(
    period_order: list[str],
    mean_by_period: pd.Series,
    recent_by_period: pd.Series,
    recent_label: str,
    q1_by_period: pd.Series,
    q3_by_period: pd.Series,
    *,
    title: str | None = None,
    unit_label: str = "",
    period_label: str = "Period",
    flood_limits: Mapping[str, float] | None = None,
) -> go.Figure:
    """Interactive Plotly counterpart to :func:`flood_overlay_static`."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=period_order,
            y=q3_by_period.reindex(period_order),
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=period_order,
            y=q1_by_period.reindex(period_order),
            mode="lines",
            name="Q1-Q3 Range",
            fill="tonexty",
            fillcolor="rgba(0,100,255,0.15)",
            line={"width": 0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=period_order,
            y=mean_by_period.reindex(period_order),
            mode="lines",
            name="Historical Mean",
            line={"color": "navy", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=period_order,
            y=recent_by_period.reindex(period_order),
            mode="lines",
            name=f"Recent Year ({recent_label})",
            line={"color": "red", "width": 2},
        )
    )
    if flood_limits:
        for label, threshold in flood_limits.items():
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color=FLOOD_COLORS.get(label, "gray"),
                annotation_text=_flood_annotation(label, threshold),
                annotation_position="left",
                layer="below",
            )
    unit = f" ({unit_label})" if unit_label else ""
    fig.update_layout(
        title=title or f"Mean Inflow with Flood Limits{unit}",
        xaxis_title=period_label,
        yaxis_title=f"Inflow{unit}",
        hovermode="x unified",
        template=_TEMPLATE,
        height=550,
        xaxis={"tickangle": -90},
        margin={"l": 100, "r": 100},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Daily per-calendar-month day-grids (Section 13 / v24-v25)
# ─────────────────────────────────────────────────────────────────────────────
# One grid per calendar month, one cell per day-of-month (28-31 cells), across
# 9 cell "types" -- 3 distribution-style (Box-Whisker/Histogram+KDE/Violin,
# each cell summarising that day's values across every year on record) and 6
# trend-style (Parametric/Robust Bounds, Anomalies, ITA Scatter, Decadal
# Blocks, Recent-vs-Long-term, each cell showing that day's across-years
# series). The trend-style cells take precomputed Sen's-slope/change-point
# values rather than stats functions to call, matching
# parametric_bounds_plot/robust_bounds_plot's existing convention -- the
# caller computes them once per day with hydrotrends.stats and passes them in.
def day_grid_layout(n_days: int) -> tuple[list[tuple[int, int]], int, int]:
    """Cell (row, col) positions for a calendar month's day-grid.

    Returns ``(positions, nrows, ncols)``: ``positions[i]`` is the (row, col)
    for day ``i + 1``.

    * <=30 days: 3 rows x 10 cols, filled row-major (a short month's last row
      is partially empty, e.g. Feb).
    * 31 days: rows 0-1 use 10 cols each (days 1-20); row 2 uses 11 cols
      (days 21-31) -- an 11th column just for the last row, not a full 11-wide
      grid, so the figure isn't mostly-empty for every other month.
    """
    positions: list[tuple[int, int]] = []
    if n_days <= 30:
        for idx in range(n_days):
            positions.append((idx // 10, idx % 10))
        nrows, ncols = 3, 10
    else:  # 31
        for idx in range(20):
            positions.append((idx // 10, idx % 10))
        for idx in range(20, n_days):
            positions.append((2, idx - 20))
        nrows, ncols = 3, 11
    return positions, nrows, ncols


def build_day_grid(
    n_days: int,
    *,
    title: str,
    cell_draw_fn: Callable[[Axes, int], None],
    cell_size: tuple[float, float] = (4.2, 3.4),
) -> Figure:
    """Generic day-grid builder shared by every Section-13 plot type.

    ``cell_draw_fn(ax, day_num)`` (``day_num`` 1-based) draws one day's cell;
    call it with a closure over that day's data (see the ``draw_*_cell``
    functions below for what each cell type needs).
    """
    positions, nrows, ncols = day_grid_layout(n_days)
    fig = Figure(figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    for row in axes:
        for ax in row:
            ax.axis("off")
    for day_idx in range(n_days):
        row, col = positions[day_idx]
        ax = axes[row][col]
        ax.axis("on")
        cell_draw_fn(ax, day_idx + 1)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def hist_mode(values: np.ndarray, *, bins: int = 10) -> float:
    """Modal-bin-center estimate: the center of the histogram's tallest bin.

    A simple, robust-enough mode for continuous hydrological data in a small
    per-day sample -- distinct from
    :func:`~hydrotrends.stats.extended_descriptive.describe_extended`'s
    rounded-frequency-with-KDE-fallback mode, which serves a different
    (larger-sample, tabular-summary) purpose.
    """
    counts, edges = np.histogram(values, bins=bins)
    idx = int(np.argmax(counts))
    return float((edges[idx] + edges[idx + 1]) / 2)


def draw_boxwhisker_cell(
    ax: Axes, day_num: int, series: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's box-and-whisker cell: 5-number-summary legend, outliers flagged."""
    s = pd.Series(series, dtype="float64").dropna().to_numpy()
    if len(s) < 3:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    vmin, vmax = float(s.min()), float(s.max())
    outliers = s[(s < lo_fence) | (s > hi_fence)]

    sns.boxplot(y=s, ax=ax, color="skyblue", width=0.5, fliersize=3)
    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(f"({unit_label})" if unit_label else "", fontsize=7)
    ax.tick_params(labelbottom=False, labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    legend_handles = [
        Line2D([], [], color="none", label=f"Max: {vmax:{fmt}}"),
        Line2D([], [], color="none", label=f"Q3: {q3:{fmt}}"),
        Line2D([], [], color="none", label=f"Median: {med:{fmt}}"),
        Line2D([], [], color="none", label=f"Q1: {q1:{fmt}}"),
        Line2D([], [], color="none", label=f"Min: {vmin:{fmt}}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=5.5,
        frameon=True,
        framealpha=0.85,
        handlelength=0,
        handletextpad=0,
        borderpad=0.4,
        labelspacing=0.25,
    )

    n_out = len(outliers)
    out_txt = f"Outliers: {n_out}"
    if 0 < n_out <= 3:
        out_txt += "\n" + "\n".join(f"{v:{fmt}}" for v in sorted(outliers))
    ax.text(
        0.03,
        0.97,
        out_txt,
        transform=ax.transAxes,
        fontsize=5.5,
        ha="left",
        va="top",
        color="#8B0000",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
        },
    )


def draw_histogram_kde_cell(
    ax: Axes, day_num: int, series: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's histogram+KDE cell: Mean/Median/Mode lines, KDE overlay."""
    s = pd.Series(series, dtype="float64").dropna().to_numpy()
    if len(s) < 3:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    mean_val = float(s.mean())
    median_val = float(np.median(s))
    mode_val = hist_mode(s, bins=min(10, max(3, len(s) // 2)))

    sns.histplot(s, kde=False, ax=ax, color="skyblue", stat="count", alpha=0.6)
    if len(s) >= 5 and s.std() > 0:
        kde = scipy_stats.gaussian_kde(s)
        xs = np.linspace(s.min(), s.max(), 200)
        ax2 = ax.twinx()
        ax2.plot(xs, kde(xs), color="#2166AC", linewidth=1.8)
        ax2.set_yticks([])
        ax2.set_ylabel("")

    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {mean_val:.0f}",
    )
    ax.axvline(
        median_val,
        color="purple",
        linestyle="-",
        linewidth=1.2,
        label=f"Median: {median_val:.0f}",
    )
    ax.axvline(
        mode_val,
        color="darkorange",
        linestyle="-.",
        linewidth=1.2,
        label=f"Mode: {mode_val:.0f}",
    )

    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.set_xlabel(f"({unit_label})" if unit_label else "", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6.5)
    handles, _ = ax.get_legend_handles_labels()
    kde_handle = [Line2D([0], [0], color="#2166AC", linewidth=1.8, label="KDE Plot")]
    ax.legend(handles=handles + kde_handle, fontsize=5.5, loc="upper right")
    ax.grid(True, alpha=0.3)


def draw_violin_cell(
    ax: Axes, day_num: int, series: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's violin cell: same 5-number-summary legend as the box-whisker
    cell, plus the mean marked as a red diamond."""
    s = pd.Series(series, dtype="float64").dropna().to_numpy()
    if len(s) < 3:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    vmin, vmax = float(s.min()), float(s.max())
    outliers = s[(s < lo_fence) | (s > hi_fence)]
    mean_val = float(s.mean())

    sns.violinplot(y=s, ax=ax, color="skyblue", inner="box", cut=0)
    ax.scatter([0], [mean_val], color="red", marker="D", s=18, zorder=5)
    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(f"({unit_label})" if unit_label else "", fontsize=7)
    ax.tick_params(labelbottom=False, labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    legend_handles = [
        Line2D([], [], color="none", label=f"Max: {vmax:{fmt}}"),
        Line2D([], [], color="none", label=f"Q3: {q3:{fmt}}"),
        Line2D([], [], color="none", label=f"Median: {med:{fmt}}"),
        Line2D([], [], color="none", label=f"Q1: {q1:{fmt}}"),
        Line2D([], [], color="none", label=f"Min: {vmin:{fmt}}"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="red",
            linestyle="none",
            markersize=5,
            label=f"Mean: {mean_val:{fmt}}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=5.5,
        frameon=True,
        framealpha=0.85,
        handlelength=0.8,
        handletextpad=0.4,
        borderpad=0.4,
        labelspacing=0.25,
    )

    n_out = len(outliers)
    out_txt = f"Outliers: {n_out}"
    if 0 < n_out <= 3:
        out_txt += "\n" + "\n".join(f"{v:{fmt}}" for v in sorted(outliers))
    ax.text(
        0.03,
        0.97,
        out_txt,
        transform=ax.transAxes,
        fontsize=5.5,
        ha="left",
        va="top",
        color="#8B0000",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
        },
    )


def draw_parametric_bounds_cell(
    ax: Axes,
    day_num: int,
    years: ArrayLike,
    values: ArrayLike,
    *,
    unit_label: str = "",
    sen_slope: SenSlope | None = None,
    change_point_year: float | None = None,
    lowess_frac: float = 0.4,
) -> None:
    """One day's Parametric-Bounds cell -- :func:`parametric_bounds_plot` at
    cell scale (smaller fonts/markers, day number as the only title)."""
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")
    if len(years_arr) < 5:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    mean_val = float(values_arr.mean())
    std_val = float(values_arr.std(ddof=1))

    ax.plot(
        years_arr,
        values_arr,
        marker="o",
        markersize=2.5,
        linestyle="-",
        color="#1f77b4",
        alpha=0.35,
        linewidth=0.8,
    )
    ax.plot(
        years_arr,
        compute_lowess(years_arr, values_arr, frac=lowess_frac),
        color="#9467bd",
        linewidth=1.6,
    )
    ax.axhline(mean_val, color="green", linestyle="--", linewidth=1.2)
    sd3 = 3 * std_val
    ax.axhline(mean_val + sd3, color="lightgreen", linestyle="--", linewidth=1)
    ax.axhline(mean_val - sd3, color="lightgreen", linestyle="--", linewidth=1)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", alpha=0.5, label="Observed"),
        Line2D([0], [0], color="#9467bd", linewidth=1.6, label="LOWESS"),
        Line2D(
            [0], [0], color="green", linestyle="--", label=f"Mean ({mean_val:{fmt}})"
        ),
        Line2D(
            [0],
            [0],
            color="lightgreen",
            linestyle="--",
            label=f"3 Std Dev ({sd3:{fmt}})",
        ),
    ]
    if sen_slope is not None:
        intercept = float(np.median(values_arr)) - sen_slope.slope * float(
            np.median(years_arr)
        )
        ax.plot(
            years_arr,
            sen_slope.slope * years_arr + intercept,
            color="#d62728",
            linestyle="--",
            linewidth=1.3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle="--",
                label=f"Sen's Slope ({sen_slope.slope:.3f}/yr)",
            )
        )
    if change_point_year is not None and not math.isnan(change_point_year):
        ax.axvline(x=change_point_year, color="black", linestyle=":", linewidth=1.2)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                label=f"Bai-Perron ({int(change_point_year)})",
            )
        )

    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(handles=legend_handles, fontsize=5, loc="best")
    ax.grid(True, alpha=0.3)


def draw_robust_bounds_cell(
    ax: Axes,
    day_num: int,
    years: ArrayLike,
    values: ArrayLike,
    *,
    unit_label: str = "",
    sen_slope: SenSlope | None = None,
    change_point_year: float | None = None,
    lowess_frac: float = 0.4,
) -> None:
    """One day's Robust-Bounds cell -- :func:`robust_bounds_plot` at cell scale."""
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")
    if len(years_arr) < 5:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    median_val = float(np.median(values_arr))
    q1, q3 = np.percentile(values_arr, [25, 75])
    iqr = q3 - q1

    ax.plot(
        years_arr,
        values_arr,
        marker="o",
        markersize=2.5,
        linestyle="-",
        color="#1f77b4",
        alpha=0.35,
        linewidth=0.8,
    )
    ax.plot(
        years_arr,
        compute_lowess(years_arr, values_arr, frac=lowess_frac),
        color="#9467bd",
        linewidth=1.6,
    )
    ax.axhline(median_val, color="green", linestyle="--", linewidth=1.2)
    iqr15 = 1.5 * iqr
    ax.axhline(median_val + iqr15, color="paleturquoise", linestyle="--", linewidth=1)
    ax.axhline(median_val - iqr15, color="paleturquoise", linestyle="--", linewidth=1)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", alpha=0.5, label="Observed"),
        Line2D([0], [0], color="#9467bd", linewidth=1.6, label="LOWESS"),
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="--",
            label=f"Median ({median_val:{fmt}})",
        ),
        Line2D(
            [0],
            [0],
            color="paleturquoise",
            linestyle="--",
            label=f"1.5 IQR ({iqr15:{fmt}})",
        ),
    ]
    if sen_slope is not None:
        intercept = median_val - sen_slope.slope * float(np.median(years_arr))
        ax.plot(
            years_arr,
            sen_slope.slope * years_arr + intercept,
            color="#d62728",
            linestyle="--",
            linewidth=1.3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle="--",
                label=f"Sen's Slope ({sen_slope.slope:.3f}/yr)",
            )
        )
    if change_point_year is not None and not math.isnan(change_point_year):
        ax.axvline(x=change_point_year, color="black", linestyle=":", linewidth=1.2)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                label=f"Pettitt ({int(change_point_year)})",
            )
        )

    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(handles=legend_handles, fontsize=5, loc="best")
    ax.grid(True, alpha=0.3)


def draw_anomaly_cell(
    ax: Axes, day_num: int, years: ArrayLike, values: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's Anomalies cell: bars vs mean, +/-1 Std Dev shading."""
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")
    if len(years_arr) < 5:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    mean_val = float(values_arr.mean())
    std_val = float(values_arr.std(ddof=1))
    anomalies = values_arr - mean_val
    inside = np.abs(anomalies) <= std_val
    colors = [
        ("#95d095" if ins else "#2ca02c")
        if a >= 0
        else ("#f0a8a8" if ins else "#d62728")
        for a, ins in zip(anomalies, inside, strict=True)
    ]
    ax.bar(years_arr, anomalies, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(std_val, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-std_val, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    legend_handles = [
        Patch(facecolor="#2ca02c", label="Above Mean"),
        Patch(facecolor="#d62728", label="Below Mean"),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            label=f"+/- Std Dev ({std_val:{fmt}})",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=5, loc="best")
    ax.grid(True, alpha=0.3)


def draw_ita_scatter_cell(ax: Axes, day_num: int, values: ArrayLike) -> None:
    """One day's ITA Scatter cell: sorted first-half vs second-half."""
    values_arr = np.asarray(values, dtype="float64")
    n_plot = len(values_arr)
    if n_plot < 6:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    v_plot = values_arr[1:] if n_plot % 2 != 0 else values_arr
    half = len(v_plot) // 2
    x1_sort, x2_sort = np.sort(v_plot[:half]), np.sort(v_plot[half:])
    ax.scatter(
        x1_sort, x2_sort, color="#1f77b4", edgecolor="k", s=10, alpha=0.7, zorder=3
    )
    min_val, max_val = (
        min(x1_sort.min(), x2_sort.min()),
        max(x1_sort.max(), x2_sort.max()),
    )
    margin = (max_val - min_val) * 0.05 or 1.0
    lims = (max(0.0, min_val - margin), max_val + margin)
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.plot(lims, [x * 1.10 for x in lims], "g:", linewidth=1)
    ax.plot(lims, [x * 0.90 for x in lims], "r:", linewidth=1)
    mean_diff = float(x2_sort.mean() - x1_sort.mean())
    ax.plot(
        lims,
        [x + mean_diff for x in lims],
        color="dodgerblue",
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.01)
    ita_slope_val = 2 * mean_diff / len(v_plot)
    ax.set_title(f"{day_num:02d}", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=6, pad=1)
    legend_handles = [
        Line2D([0], [0], color="k", linestyle="--", label="1:1 Line"),
        Line2D([0], [0], color="g", linestyle=":", label="+10%"),
        Line2D([0], [0], color="r", linestyle=":", label="-10%"),
        Line2D(
            [0],
            [0],
            color="dodgerblue",
            linestyle="--",
            label=f"ITA Slope: {ita_slope_val:.3f}/year",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=5,
        loc="lower right",
        frameon=False,
        borderpad=0.15,
        handlelength=1.2,
        handletextpad=0.3,
        labelspacing=0.15,
    )


_DECADE_BOUNDARIES: tuple[int, ...] = (1980, 1990, 2000, 2010, 2020)


def _decade_label(year: int, min_year: int, max_year: int) -> str:
    for boundary in _DECADE_BOUNDARIES:
        if year <= boundary:
            lo = min_year if boundary == _DECADE_BOUNDARIES[0] else boundary - 9
            return f"{lo}–{boundary}"
    return f"{_DECADE_BOUNDARIES[-1] + 1}–{max_year}"


def draw_decadal_blocks_cell(
    ax: Axes, day_num: int, years: ArrayLike, values: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's Decadal-Blocks cell: mean by decade + overall-mean reference line."""
    years_arr = np.asarray(years, dtype="int64")
    values_arr = np.asarray(values, dtype="float64")
    if len(years_arr) < 5:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    min_year, max_year = int(years_arr.min()), int(years_arr.max())
    df = pd.DataFrame({"Year": years_arr, "Val": values_arr})
    df["Decade"] = df["Year"].apply(lambda y: _decade_label(y, min_year, max_year))
    decade_order = []
    for probe in (1975, 1985, 1995, 2005, 2015, 2025):
        label = _decade_label(probe, min_year, max_year)
        if label not in decade_order and (df["Decade"] == label).any():
            decade_order.append(label)
    df["Decade"] = pd.Categorical(df["Decade"], categories=decade_order, ordered=True)
    summary = df.groupby("Decade", observed=True)["Val"].mean().dropna()
    overall_mean = float(df["Val"].mean())

    bars = ax.bar(
        summary.index.astype(str),
        summary.to_numpy(),
        color=sns.color_palette("viridis", len(summary)),
    )
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.3)
    for rect in bars:
        height = rect.get_height()
        ax.annotate(
            format(height, fmt),
            (rect.get_x() + rect.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, summary.max() * 1.25 if len(summary) else 1)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                label=f"Overall Mean ({overall_mean:{fmt}})",
            )
        ],
        fontsize=5,
        loc="upper right",
    )
    ax.grid(True, alpha=0.3, axis="y")


def draw_recent_vs_longterm_cell(
    ax: Axes, day_num: int, years: ArrayLike, values: ArrayLike, *, unit_label: str = ""
) -> None:
    """One day's Recent-vs-Long-term cell: mean over sliding recent windows
    (last 5/10/20/40 years) vs the full record's long-term mean."""
    years_arr = np.asarray(years, dtype="int64")
    values_arr = np.asarray(values, dtype="float64")
    if len(years_arr) < 5:
        ax.set_title(f"{day_num:02d} (insufficient data)", fontsize=9)
        ax.axis("off")
        return
    fmt = _value_format(unit_label)
    order = np.argsort(years_arr)
    sorted_values = values_arr[order]
    n = len(sorted_values)
    windows = {"Long-term": n, "Last 40": 40, "Last 20": 20, "Last 10": 10, "Last 5": 5}
    labels, means = [], []
    for label, window in windows.items():
        window = min(window, n)
        labels.append(label)
        means.append(float(sorted_values[-window:].mean()))
    overall_mean = means[0]  # "Long-term"

    palette = ["#e74c3c" if lbl == "Long-term" else "#3498db" for lbl in labels]
    bars = ax.bar(labels, means, color=palette, alpha=0.9)
    for rect, mean_val in zip(bars, means, strict=True):
        height = rect.get_height()
        ax.annotate(
            format(mean_val, fmt),
            (rect.get_x() + rect.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.3)
    ax.set_title(f"{day_num:02d}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, max(means) * 1.3)
    ax.legend(
        handles=[
            Patch(facecolor="#e74c3c", label=f"Long-term Mean ({overall_mean:{fmt}})"),
            Patch(facecolor="#3498db", label="Recent Sliding Means"),
        ],
        fontsize=5,
        loc="upper right",
    )
    ax.grid(True, alpha=0.3, axis="y")


# ─────────────────────────────────────────────────────────────────────────────
# Daily-specific duration curve / flood heatmap / flood overlay (Section 13)
# ─────────────────────────────────────────────────────────────────────────────
# Richer than Section 12's generic 5-scale versions (flow_duration_curve_*/
# flood_heatmap*/flood_overlay_*): an exceedance-probability readout on the
# duration curve, a white-to-red (0 = white, not yellow) reversed-row
# colormap on the heatmap, and Range+IQR shaded bands plus 3-Std-Dev dashed
# lines (not a band) on the overlay. Daily scale only -- these are the exact
# v24 variants, distinct functions rather than a shared one with a flag,
# since the formulas/styling genuinely differ (see each docstring).
_WHITE_TO_RED = mcolors.LinearSegmentedColormap.from_list(
    "white_to_red", ["#FFFFFF", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026"]
)


def _probability_at_threshold(curve: pd.DataFrame, threshold: float) -> float:
    """Exceedance probability (%) at ``threshold``, interpolated from the
    duration curve's own plotted points -- distinct from
    :func:`~hydrotrends.stats.frequency.exceedance_probability`'s empirical
    ``mean(x >= threshold)``, which doesn't agree with the curve's Weibull
    plotting positions off the exact data points. NaN outside the data range.
    """
    values_asc = curve[COL_FDC_VALUE].to_numpy()[::-1]
    probs_asc = curve[COL_FDC_EXCEEDANCE].to_numpy()[::-1]
    if threshold < values_asc.min() or threshold > values_asc.max():
        return float("nan")
    return float(np.interp(threshold, values_asc, probs_asc))


def daily_duration_curve_static(
    data: pd.Series | np.ndarray,
    *,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
    figsize: tuple[float, float] = (9.0, 5.5),
    line_color: str = _FDC_COLOR,
) -> Figure:
    """Daily flow duration curve, each flood-limit line annotated with its
    exceedance probability (%) -- the Section-13 enhancement over
    :func:`flow_duration_curve_static`'s plain threshold label.
    """
    curve = flow_duration_curve(data)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(
        curve[COL_FDC_EXCEEDANCE], curve[COL_FDC_VALUE], color=line_color, linewidth=2
    )

    if flood_limits:
        for label, threshold in flood_limits.items():
            color = FLOOD_COLORS.get(label, "gray")
            prob = _probability_at_threshold(curve, threshold)
            prob_txt = f"{prob:.3f}%" if not math.isnan(prob) else "N/A"
            ax.axhline(threshold, linestyle=":", linewidth=1.3, color=color)
            ax.annotate(
                f"{label} ({threshold:,.0f})  |  "
                f"Exceedance Probability (%): {prob_txt}",
                xy=(98, threshold),
                fontsize=7.5,
                color=color,
                ha="right",
                va="bottom",
            )

    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(f"Daily Flow Duration Curve{unit}", fontweight="bold")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel(f"Value{unit}")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def daily_duration_curve_interactive(
    data: pd.Series | np.ndarray,
    *,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
    line_color: str = _FDC_COLOR,
) -> go.Figure:
    """Interactive Plotly counterpart to :func:`daily_duration_curve_static`."""
    curve = flow_duration_curve(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve[COL_FDC_EXCEEDANCE],
            y=curve[COL_FDC_VALUE],
            mode="lines",
            name="Duration curve",
            line={"color": line_color, "width": 2},
        )
    )
    if flood_limits:
        for label, threshold in flood_limits.items():
            prob = _probability_at_threshold(curve, threshold)
            prob_txt = f"{prob:.1f}%" if not math.isnan(prob) else "N/A"
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color=FLOOD_COLORS.get(label, "gray"),
                annotation_text=(
                    f"{label} ({threshold:,.0f}) | "
                    f"Exceedance Probability (%): {prob_txt}"
                ),
                annotation_position="right",
            )
    unit = f" ({unit_label})" if unit_label else ""
    fig.update_layout(
        title=f"Daily Flow Duration Curve{unit}",
        xaxis_title="Exceedance Probability (%)",
        yaxis_title=f"Value{unit}",
        template=_TEMPLATE,
        height=500,
        xaxis={"range": [0, 100]},
    )
    return fig


def _thin_ticks_10day(period_order: list[str]) -> tuple[list[int], list[str]]:
    """Tick positions/labels for every 10th day-of-month (day 01, 11, 21) --
    the Daily scale's 366 periods are too dense to label every one."""
    indices, labels = [], []
    for i, period in enumerate(period_order):
        if period[-2:] in ("01", "11", "21"):
            indices.append(i)
            labels.append(period)
    return indices, labels


def daily_flood_heatmap(
    counts: pd.DataFrame,
    period_order: list[str],
    *,
    title: str = "Daily Flood Exceedance Count",
) -> Figure:
    """Daily-specific flood-exceedance heatmap: white-to-red colormap (0 =
    white, not yellow -- a quiet color for "no exceedance" matters when most
    of 366 daily cells are 0), rows reversed (LF at bottom, EHF at top,
    matching flood severity reading top-down), colorbar on the left, and
    every-10th-day tick thinning. See :func:`flood_heatmap` for the Section-12
    (all-5-scales) plain-YlOrRd counterpart.
    """
    flood_cols_top_to_bottom = [
        c for c in reversed(FLOOD_CLASSES) if c in counts.columns
    ]
    pivot = counts[flood_cols_top_to_bottom].T.reindex(columns=period_order)
    n = pivot.shape[1]

    fig_width = min(22.0, max(12.0, n * 0.12))
    fig = Figure(figsize=(fig_width, 3.5))
    ax = fig.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="2%", pad=0.55)
    sns.heatmap(
        pivot,
        cmap=_WHITE_TO_RED,
        vmin=0,
        cbar_kws={"label": "Exceedance Count"},
        ax=ax,
        cbar_ax=cax,
        linewidths=0.3,
        linecolor="lightgray",
    )
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Daily Period")
    ax.set_ylabel("Flood Level")
    tick_idx, tick_labels = _thin_ticks_10day(period_order)
    ax.set_xticks([i + 0.5 for i in tick_idx])
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    fig.tight_layout()
    return fig


def daily_flood_heatmap_interactive(
    counts: pd.DataFrame,
    period_order: list[str],
    *,
    title: str = "Daily Flood Exceedance Count",
) -> go.Figure:
    """Interactive Plotly counterpart to :func:`daily_flood_heatmap`."""
    flood_cols = [c for c in FLOOD_CLASSES if c in counts.columns]
    melted = counts.reset_index(names="Period").melt(
        id_vars="Period",
        value_vars=flood_cols,
        var_name="Flood_Level",
        value_name="Exceedances",
    )
    fig = px.density_heatmap(
        melted,
        x="Period",
        y="Flood_Level",
        z="Exceedances",
        title=title,
        labels={"Exceedances": "Count"},
        category_orders={"Period": period_order, "Flood_Level": flood_cols},
        color_continuous_scale=["white", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026"],
        height=400,
    )
    _tick_idx, tick_labels = _thin_ticks_10day(period_order)
    fig.update_layout(
        xaxis={
            "tickangle": -90,
            "tickmode": "array",
            "tickvals": tick_labels,
            "ticktext": tick_labels,
        },
        template=_TEMPLATE,
    )
    return fig


def daily_flood_overlay(
    period_order: list[str],
    mean_by_period: pd.Series,
    recent_by_period: pd.Series,
    recent_label: str,
    q1_by_period: pd.Series,
    q3_by_period: pd.Series,
    min_by_period: pd.Series,
    max_by_period: pd.Series,
    std_by_period: pd.Series,
    *,
    unit_label: str = "",
    flood_limits: Mapping[str, float] | None = None,
) -> Figure:
    """Daily-specific flood overlay: Range (min-max) and IQR (Q1-Q3) as
    layered shaded bands, 3 Std Dev as two dashed blue lines (not a band,
    unlike :func:`flood_overlay_static`), the most recent hydrological
    year's series as a dashed red line, and every-10th-day tick thinning.
    No interactive counterpart -- the source only produces a static PNG here.
    """
    n = len(period_order)
    fig_width = min(16.0, max(10.0, n * 0.05))
    fig = Figure(figsize=(fig_width, 6.0))
    ax = fig.subplots()
    x = np.arange(n)

    range_lo = min_by_period.reindex(period_order)
    range_hi = max_by_period.reindex(period_order)
    mean_r = mean_by_period.reindex(period_order)
    std_r = std_by_period.reindex(period_order)
    sd_lo, sd_hi = mean_r - 3 * std_r, mean_r + 3 * std_r
    q1_r, q3_r = q1_by_period.reindex(period_order), q3_by_period.reindex(period_order)

    ax.fill_between(
        x,
        range_lo,
        range_hi,
        color="#DCEBFA",
        alpha=0.9,
        label="Range (Max-Min)",
        zorder=1,
    )
    ax.fill_between(
        x, q1_r, q3_r, color="#4E9BD8", alpha=0.85, label="IQR (Q3-Q1)", zorder=3
    )
    ax.plot(
        x,
        sd_hi,
        color="blue",
        linestyle="--",
        linewidth=1.2,
        label="3 Std Dev",
        zorder=2,
    )
    ax.plot(x, sd_lo, color="blue", linestyle="--", linewidth=1.2, zorder=2)

    ax.plot(x, mean_r, color="navy", linewidth=2.5, label="Historical Mean", zorder=4)
    ax.plot(
        x,
        recent_by_period.reindex(period_order),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"HydroYear {recent_label}",
        zorder=5,
    )

    if flood_limits:
        for label, threshold in flood_limits.items():
            color = FLOOD_COLORS.get(label, "gray")
            ax.axhline(threshold, linestyle=":", linewidth=1.2, color=color, zorder=6)
            ax.annotate(
                label, xy=(n - 1, threshold), fontsize=8, color=color, ha="left"
            )

    tick_idx, tick_labels = _thin_ticks_10day(period_order)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_xlim(0, n - 1)
    unit = f" ({unit_label})" if unit_label else ""
    ax.set_title(f"Daily Mean Flow with Flood Limits{unit}", fontweight="bold")
    ax.set_ylabel(f"Inflow{unit}")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
