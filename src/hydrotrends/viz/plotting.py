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
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from ..core.constants import COL_DATE, FLOOD_CLASSES, FLOOD_COLORS, LOWESS_FRAC
from ..core.utils import significance_stars
from ..core.validation import ArrayLike
from ..stats.frequency import (
    COL_FDC_EXCEEDANCE,
    COL_FDC_VALUE,
    exceedance_probability,
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
    "flood_heatmap",
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


def _flood_annotation(
    label: str, threshold: float, data: pd.Series | np.ndarray
) -> str:
    """Flood-line label with the threshold and its exceedance probability (%)."""
    exceed = exceedance_probability(data, threshold)
    return f"{label} ({threshold:,.0f}) — {exceed:.1f}%"


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
    is drawn as a horizontal line annotated with its exceedance probability.
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
                _flood_annotation(label, threshold, data),
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
                annotation_text=_flood_annotation(label, threshold, data),
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
    data_by_period: Mapping[object, ArrayLike],
    *,
    order: list[object] | None = None,
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
