"""Plot builders: LOWESS parity, bounds plots, and the Phase-4 (Section
11/12/13) distribution/box-whisker/flood/day-grid/whole-series-trend plot
functions.

The time-series/ITA-scatter functions still have no test coverage in this
codebase; this file covers what was newly ported here rather than attempting
to backfill the whole module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as GoFigure

from hydrotrends.core.constants import FLOOD_LIMITS_1000CUSECS
from hydrotrends.stats.trends import SenSlope, sens_slope
from hydrotrends.viz.plotting import (
    anomaly_bar_chart,
    boxwhisker_grid,
    build_day_grid,
    compute_decadal_summary,
    compute_lowess,
    compute_sliding_window_summary,
    daily_duration_curve_interactive,
    daily_duration_curve_static,
    daily_flood_heatmap,
    daily_flood_heatmap_interactive,
    daily_flood_overlay,
    daily_trend_heatmap,
    day_grid_layout,
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
    hist_mode,
    monthly_trend_heatmap,
    parametric_bounds_plot,
    robust_bounds_plot,
    seasonal_divergence_lowess_chart,
    seasonal_trend_heatmap,
    single_histogram,
    sliding_windows_bar_chart,
)

from . import reference_v26_core as ref


def _legend_labels(fig: Figure) -> list[str]:
    ax = fig.axes[0]
    return [t.get_text() for t in ax.get_legend().get_texts()]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_lowess_matches_v26_reference(seed):
    rng = np.random.default_rng(seed)
    x = np.arange(30, dtype="float64")
    y = 50.0 + 0.5 * x + 10.0 * np.sin(x / 3) + rng.normal(0, 2.0, x.size)

    got = compute_lowess(x, y, frac=0.3)
    expected = ref.compute_lowess(x, y, frac=0.3)

    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_compute_lowess_handles_degenerate_neighbourhood():
    # Every point coincident in x (d_max == 0 branch in the source).
    x = np.zeros(10)
    y = np.arange(10, dtype="float64")
    got = compute_lowess(x, y, frac=0.3)
    expected = ref.compute_lowess(x, y, frac=0.3)
    assert np.allclose(got, expected)


def test_parametric_bounds_plot_basic():
    years = np.arange(2000, 2020, dtype="float64")
    values = 100.0 + 2.0 * (years - 2000) + np.sin(years)

    fig = parametric_bounds_plot(
        years,
        values,
        unit_label="Cusecs",
        sen_slope=SenSlope(slope=2.0, intercept=100.0),
        change_point_year=2010,
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert "Parametric Bounds" in ax.get_title()
    labels = _legend_labels(fig)
    assert any("Mean" in lbl for lbl in labels)
    assert any("± 3 Std Dev" in lbl for lbl in labels)
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Bai-Perron Break" in lbl for lbl in labels)


def test_robust_bounds_plot_basic():
    years = np.arange(2000, 2020, dtype="float64")
    values = 100.0 + 2.0 * (years - 2000) + np.sin(years)

    fig = robust_bounds_plot(
        years,
        values,
        unit_label="MAF",
        sen_slope=SenSlope(slope=2.0, intercept=100.0),
        change_point_year=2012,
    )
    ax = fig.axes[0]
    assert "Robust Bounds" in ax.get_title()
    labels = _legend_labels(fig)
    assert any("Median" in lbl for lbl in labels)
    assert any("IQR" in lbl for lbl in labels)
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Pettitt Break" in lbl for lbl in labels)


def test_bounds_plots_omit_overlays_when_not_provided():
    years = np.arange(2000, 2010, dtype="float64")
    values = np.linspace(100, 110, years.size)

    fig = parametric_bounds_plot(years, values, unit_label="Cusecs")
    labels = _legend_labels(fig)
    assert not any("Sen's Slope" in lbl for lbl in labels)
    assert not any("Break" in lbl for lbl in labels)


def test_bounds_plots_ignore_nan_change_point():
    years = np.arange(2000, 2010, dtype="float64")
    values = np.linspace(100, 110, years.size)

    fig = robust_bounds_plot(years, values, change_point_year=float("nan"))
    labels = _legend_labels(fig)
    assert not any("Break" in lbl for lbl in labels)


def _sample_data_by_period(rng, periods, n=20):
    return {p: rng.uniform(20_000, 60_000, n) for p in periods}


def test_distribution_grid_sets_suptitle_and_marks_insufficient_data():
    rng = np.random.default_rng(0)
    periods = ["Apr-01", "Apr-02", "Apr-03"]
    data = _sample_data_by_period(rng, periods)
    data["Apr-03"] = np.array([1.0, 2.0])  # below min_points=3

    fig = distribution_grid(data, order=periods, title="Daily Distribution", ncols=2)
    assert fig._suptitle.get_text() == "Daily Distribution"
    assert "insufficient data" in fig.axes[2].get_title()


def test_boxwhisker_grid_applies_shared_ylim():
    rng = np.random.default_rng(1)
    periods = ["Apr-01", "Apr-02"]
    data = _sample_data_by_period(rng, periods)

    fig = boxwhisker_grid(data, order=periods, ylim=(0.0, 100_000.0), ncols=2)
    for ax in fig.axes[: len(periods)]:
        assert ax.get_ylim() == (0.0, 100_000.0)


def test_single_histogram_draws_mean_median_lines():
    rng = np.random.default_rng(2)
    fig = single_histogram(rng.uniform(30, 40, 50), title="Annual Distribution")
    labels = _legend_labels(fig)
    assert any(lbl.startswith("Mean:") for lbl in labels)
    assert any(lbl.startswith("Median:") for lbl in labels)


def test_duration_curve_flood_annotation_has_no_probability():
    """Section 12's plain duration curve shows just the threshold -- the
    exceedance-probability readout is Section 13's Daily-only enhancement."""
    rng = np.random.default_rng(3)
    values = rng.uniform(20_000, 900_000, 200)
    flood_limits = {k: v * 1000 for k, v in FLOOD_LIMITS_1000CUSECS.items()}

    fig = flow_duration_curve_static(
        values, unit_label="Cusecs", flood_limits=flood_limits
    )
    ax = fig.axes[0]
    annotations = [a.get_text() for a in ax.texts]
    assert any(a == "LF (250,000)" for a in annotations)
    assert not any("%" in a for a in annotations)

    fig_i = flow_duration_curve_interactive(
        values, unit_label="Cusecs", flood_limits=flood_limits
    )
    assert isinstance(fig_i, GoFigure)
    hline_texts = [ann.text or "" for ann in fig_i.layout.annotations or []]
    assert any(t == "LF (250,000)" for t in hline_texts)
    assert not any("%" in t for t in hline_texts)


def test_flood_heatmap_and_interactive_counterpart_run():
    periods = ["Apr-01", "Apr-02", "Apr-03"]
    counts = pd.DataFrame(
        {
            "LF": [5, 0, 2],
            "MF": [1, 0, 0],
            "HF": [0, 0, 0],
            "VHF": [0, 0, 0],
            "EHF": [0, 0, 0],
        },
        index=periods,
    )
    fig = flood_heatmap(counts, title="Daily Flood Exceedance Count")
    assert isinstance(fig, Figure)

    fig_i = flood_heatmap_interactive(counts, title="Daily Flood Exceedance Count")
    assert isinstance(fig_i, GoFigure)


def test_flood_overlay_static_and_interactive_run():
    periods = ["Apr-01", "Apr-02", "Apr-03"]
    mean_s = pd.Series([30_000.0, 32_000.0, 31_000.0], index=periods)
    q1_s = pd.Series([25_000.0, 27_000.0, 26_000.0], index=periods)
    q3_s = pd.Series([35_000.0, 37_000.0, 36_000.0], index=periods)
    recent_s = pd.Series([31_000.0, 33_000.0, 30_000.0], index=periods)
    flood_limits = {k: v * 1000 for k, v in FLOOD_LIMITS_1000CUSECS.items()}

    fig = flood_overlay_static(
        periods,
        mean_s,
        recent_s,
        "2023-24",
        q1_s,
        q3_s,
        unit_label="Cusecs",
        flood_limits=flood_limits,
    )
    assert isinstance(fig, Figure)
    labels = _legend_labels(fig)
    assert any("Historical Mean" in lbl for lbl in labels)
    assert any("Recent Year (2023-24)" in lbl for lbl in labels)

    fig_i = flood_overlay_interactive(
        periods,
        mean_s,
        recent_s,
        "2023-24",
        q1_s,
        q3_s,
        unit_label="Cusecs",
        flood_limits=flood_limits,
    )
    assert isinstance(fig_i, GoFigure)


# ── Section 13: day-grid engine + cell functions ────────────────────────────
@pytest.mark.parametrize(
    "n_days,expect_shape", [(28, (3, 10)), (30, (3, 10)), (31, (3, 11))]
)
def test_day_grid_layout_shapes(n_days, expect_shape):
    positions, nrows, ncols = day_grid_layout(n_days)
    assert (nrows, ncols) == expect_shape
    assert len(positions) == n_days
    assert positions[0] == (0, 0)
    assert len(set(positions)) == n_days  # no two days share a cell


def test_build_day_grid_sets_suptitle_and_axis_count():
    def cell(ax, day_num):
        ax.set_title(str(day_num))

    fig = build_day_grid(31, title="Test Grid", cell_draw_fn=cell)
    assert fig._suptitle.get_text() == "Test Grid"
    assert len(fig.axes) == 33  # 3x11 grid, all 33 cells exist (2 blank)


def test_hist_mode_finds_the_tallest_bin_center():
    values = np.concatenate([np.full(50, 10.0), np.full(5, 100.0)])
    assert hist_mode(values, bins=10) < 50.0


@pytest.mark.parametrize(
    "draw_fn",
    [draw_boxwhisker_cell, draw_histogram_kde_cell, draw_violin_cell],
)
def test_distribution_cells_mark_insufficient_data(draw_fn):
    def cell(ax, day_num):
        draw_fn(ax, day_num, [1.0, 2.0], unit_label="Cusecs")

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    assert "insufficient data" in fig.axes[0].get_title()


def test_draw_boxwhisker_cell_shows_five_number_summary():
    rng = np.random.default_rng(0)
    values = rng.uniform(20_000, 60_000, 30)

    def cell(ax, day_num):
        draw_boxwhisker_cell(ax, day_num, values, unit_label="Cusecs")

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(lbl.startswith("Median:") for lbl in labels)
    assert any(lbl.startswith("Max:") for lbl in labels)
    assert any(lbl.startswith("Min:") for lbl in labels)


def test_draw_violin_cell_shows_mean_marker():
    rng = np.random.default_rng(1)
    values = rng.uniform(20_000, 60_000, 30)

    def cell(ax, day_num):
        draw_violin_cell(ax, day_num, values, unit_label="Cusecs")

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(lbl.startswith("Mean:") for lbl in labels)


@pytest.mark.parametrize(
    "draw_fn",
    [
        draw_parametric_bounds_cell,
        draw_robust_bounds_cell,
        draw_anomaly_cell,
        draw_decadal_blocks_cell,
        draw_recent_vs_longterm_cell,
    ],
)
def test_trend_cells_mark_insufficient_data(draw_fn):
    def cell(ax, day_num):
        draw_fn(ax, day_num, [2000, 2001], [1.0, 2.0], unit_label="Cusecs")

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    assert "insufficient data" in fig.axes[0].get_title()


def test_draw_parametric_bounds_cell_draws_sen_slope_and_change_point():
    years = np.arange(2000, 2020, dtype="float64")
    values = 100.0 + 2.0 * (years - 2000) + np.sin(years)
    sen = sens_slope(values)

    def cell(ax, day_num):
        draw_parametric_bounds_cell(
            ax,
            day_num,
            years,
            values,
            unit_label="Cusecs",
            sen_slope=sen,
            change_point_year=2010,
        )

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Bai-Perron" in lbl for lbl in labels)


def test_draw_robust_bounds_cell_draws_sen_slope_and_change_point():
    years = np.arange(2000, 2020, dtype="float64")
    values = 100.0 + 2.0 * (years - 2000) + np.sin(years)
    sen = sens_slope(values)

    def cell(ax, day_num):
        draw_robust_bounds_cell(
            ax,
            day_num,
            years,
            values,
            unit_label="Cusecs",
            sen_slope=sen,
            change_point_year=2012,
        )

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Pettitt" in lbl for lbl in labels)


def test_draw_ita_scatter_cell_marks_insufficient_data_below_six_points():
    def cell(ax, day_num):
        draw_ita_scatter_cell(ax, day_num, [1.0, 2.0, 3.0])

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    assert "insufficient data" in fig.axes[0].get_title()


def test_draw_ita_scatter_cell_runs_with_enough_points():
    rng = np.random.default_rng(2)
    values = rng.uniform(20_000, 60_000, 20)

    def cell(ax, day_num):
        draw_ita_scatter_cell(ax, day_num, values)

    fig = build_day_grid(1, title="Test", cell_draw_fn=cell)
    assert fig.axes[0].get_title() == "01"


def test_daily_duration_curve_annotation_includes_exceedance_probability():
    """The Section-13 enhancement over flow_duration_curve_static: an
    exceedance-probability readout alongside the threshold."""
    rng = np.random.default_rng(3)
    values = rng.uniform(20_000, 900_000, 200)
    flood_limits = {k: v * 1000 for k, v in FLOOD_LIMITS_1000CUSECS.items()}

    fig = daily_duration_curve_static(
        values, unit_label="Cusecs", flood_limits=flood_limits
    )
    annotations = [a.get_text() for a in fig.axes[0].texts]
    assert any(
        "LF (250,000)" in a and "Exceedance Probability" in a for a in annotations
    )

    fig_i = daily_duration_curve_interactive(
        values, unit_label="Cusecs", flood_limits=flood_limits
    )
    assert isinstance(fig_i, GoFigure)
    hline_texts = [ann.text or "" for ann in fig_i.layout.annotations or []]
    assert any(
        "LF (250,000)" in t and "Exceedance Probability" in t for t in hline_texts
    )


def test_daily_flood_heatmap_and_interactive_counterpart_run():
    periods = [f"Apr-{d:02d}" for d in range(1, 31)]
    rng = np.random.default_rng(4)
    counts = pd.DataFrame(
        {
            "LF": rng.integers(0, 10, len(periods)),
            "MF": rng.integers(0, 5, len(periods)),
            "HF": np.zeros(len(periods), dtype=int),
            "VHF": np.zeros(len(periods), dtype=int),
            "EHF": np.zeros(len(periods), dtype=int),
        },
        index=periods,
    )
    fig = daily_flood_heatmap(counts, periods)
    assert isinstance(fig, Figure)

    fig_i = daily_flood_heatmap_interactive(counts, periods)
    assert isinstance(fig_i, GoFigure)


def test_daily_flood_overlay_runs_and_labels_hydro_year():
    periods = [f"Apr-{d:02d}" for d in range(1, 31)]
    rng = np.random.default_rng(5)
    mean_s = pd.Series(rng.uniform(30_000, 40_000, len(periods)), index=periods)
    q1_s = pd.Series(rng.uniform(20_000, 25_000, len(periods)), index=periods)
    q3_s = pd.Series(rng.uniform(45_000, 50_000, len(periods)), index=periods)
    min_s = pd.Series(rng.uniform(10_000, 15_000, len(periods)), index=periods)
    max_s = pd.Series(rng.uniform(60_000, 70_000, len(periods)), index=periods)
    std_s = pd.Series(rng.uniform(1_000, 5_000, len(periods)), index=periods)
    recent_s = pd.Series(rng.uniform(30_000, 40_000, len(periods)), index=periods)
    flood_limits = {k: v * 1000 for k, v in FLOOD_LIMITS_1000CUSECS.items()}

    fig = daily_flood_overlay(
        periods,
        mean_s,
        recent_s,
        "2023-24",
        q1_s,
        q3_s,
        min_s,
        max_s,
        std_s,
        unit_label="Cusecs",
        flood_limits=flood_limits,
    )
    assert isinstance(fig, Figure)
    labels = _legend_labels(fig)
    assert any("HydroYear 2023-24" in lbl for lbl in labels)
    assert any("3 Std Dev" in lbl for lbl in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: whole-series trend suite (anomalies / decadal blocks / sliding
# windows) -- distinct from the Section-13 per-day cells above.
# ─────────────────────────────────────────────────────────────────────────────
def test_anomaly_bar_chart_colors_bars_by_sign():
    years = np.arange(2000, 2010, dtype="float64")
    values = np.array([90, 95, 100, 80, 120, 100, 100, 60, 140, 100], dtype="float64")
    mean_val = values.mean()

    fig = anomaly_bar_chart(years, values, title="Annual Anomalies", unit_label="MAF")
    ax = fig.axes[0]
    assert ax.get_title() == "Annual Anomalies"
    assert "Deviation from Mean (MAF)" in ax.get_ylabel()

    bars = ax.patches
    assert len(bars) == len(values)
    for bar, val in zip(bars, values, strict=True):
        expected = "#2ca02c" if val >= mean_val else "#d62728"
        assert mcolors.to_hex(bar.get_facecolor()) == expected


def test_compute_decadal_summary_matches_expected_boundaries():
    # Spans four decade buckets: 1975-1980, 1981-1990, 1991-2000, 2001-2005
    # (the open-ended trailing bucket keeps the fixed "2001-2010" label even
    # though the data stops at 2005 -- matching the source's fixed-string
    # buckets for every boundary but the first/last).
    years = np.arange(1975, 2006)
    values = np.arange(len(years), dtype="float64")

    summary = compute_decadal_summary(years, values)

    assert list(summary["Decade"].astype(str)) == [
        "1975–1980",
        "1981–1990",
        "1991–2000",
        "2001–2010",
    ]
    assert list(summary["Years"]) == [6, 10, 10, 5]
    df = pd.DataFrame({"Year": years, "Val": values})
    expected_means = [
        df.loc[df["Year"].between(1975, 1980), "Val"].mean(),
        df.loc[df["Year"].between(1981, 1990), "Val"].mean(),
        df.loc[df["Year"].between(1991, 2000), "Val"].mean(),
        df.loc[df["Year"].between(2001, 2005), "Val"].mean(),
    ]
    assert np.allclose(summary["Mean"].to_numpy(), expected_means)


def test_decadal_blocks_bar_chart_annotates_bars_and_overall_mean_line():
    years = np.arange(1975, 2006)
    values = np.arange(len(years), dtype="float64")
    summary = compute_decadal_summary(years, values)
    overall_mean = float(values.mean())

    fig = decadal_blocks_bar_chart(
        summary, overall_mean=overall_mean, title="Decadal (Annual)", unit_label="MAF"
    )
    ax = fig.axes[0]
    assert ax.get_title() == "Decadal (Annual)"
    assert len(ax.patches) == len(summary)
    labels = _legend_labels(fig)
    assert any("Overall Mean" in lbl for lbl in labels)
    assert ax.get_ylim()[1] == pytest.approx(float(summary["Mean"].max()) * 1.15)


def test_compute_sliding_window_summary_windows_and_spans():
    years = np.arange(1970, 2020)  # 50 years: 1970..2019
    values = years.astype("float64")

    summary = compute_sliding_window_summary(years, values)

    assert list(summary["Period"]) == [
        "Overall Climatological",
        "Last 40 Years",
        "Last 20 Years",
        "Last 10 Years",
        "Last 5 Years",
    ]
    assert list(summary["Years_Included"]) == [50, 40, 20, 10, 5]
    assert list(summary["Span"]) == [
        "1970–2019",
        "1980–2019",
        "2000–2019",
        "2010–2019",
        "2015–2019",
    ]
    # values == years here, so each window's mean is the midpoint of its span.
    assert summary.loc[summary["Period"] == "Last 5 Years", "Mean"].iloc[0] == 2017.0


def test_sliding_windows_bar_chart_highlights_overall_climatological():
    years = np.arange(1970, 2020)
    values = years.astype("float64")
    summary = compute_sliding_window_summary(years, values)

    fig = sliding_windows_bar_chart(summary, title="Sliding (Annual)", unit_label="MAF")
    ax = fig.axes[0]
    assert ax.get_title() == "Sliding (Annual)"
    assert len(ax.patches) == len(summary)
    assert mcolors.to_hex(ax.patches[0].get_facecolor()) == "#e74c3c"
    for bar in ax.patches[1:]:
        assert mcolors.to_hex(bar.get_facecolor()) == "#3498db"
    labels = _legend_labels(fig)
    assert any("Long-term Historical Mean" in lbl for lbl in labels)
    assert any("Recent Sliding Means" in lbl for lbl in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11's 5 summary overview plots (7A-7E).
# ─────────────────────────────────────────────────────────────────────────────
def test_seasonal_divergence_lowess_chart_skips_seasons_with_insufficient_data():
    years = np.arange(2000, 2021)  # 21 years
    rng = np.random.default_rng(6)
    series_by_season = {
        "Early_Kharif": pd.Series(rng.uniform(1, 2, len(years)), index=years),
        # Only 2 valid years -- below the source's "> 3" threshold.
        "Late_Kharif": pd.Series([1.0, 1.1], index=[2000, 2001]),
        "Rabi": pd.Series(rng.uniform(0.5, 1.0, len(years)), index=years),
    }

    fig = seasonal_divergence_lowess_chart(years, series_by_season, unit_label="MAF")
    ax = fig.axes[0]
    assert "Multi-Decadal Smoothing" in ax.get_title()
    labels = _legend_labels(fig)
    assert any("Early Kharif (LOWESS)" in lbl for lbl in labels)
    assert any("Rabi (LOWESS)" in lbl for lbl in labels)
    assert not any("Late Kharif" in lbl for lbl in labels)


def test_seasonal_trend_heatmap_blanks_ns_and_missing_season():
    trend_by_season = {
        "Early_Kharif": ("increasing", "***"),
        "Rabi": ("decreasing", "ns"),
        # "Kharif"/"Late_Kharif"/"Annual" omitted -- left blank.
    }
    fig = seasonal_trend_heatmap(trend_by_season, unit_label="MAF")
    ax = fig.axes[0]
    assert "Seasonal & Annual Monotonic Trends" in ax.get_title()
    texts = [t.get_text() for t in ax.texts]
    # Only the 2 populated cells get annotated at all -- seaborn masks NaN
    # cells (the 3 omitted columns) entirely, with no text object emitted.
    assert texts == ["***", ""]


def test_monthly_trend_heatmap_shows_literal_ns():
    monthly_results = pd.DataFrame(
        {"trend": ["increasing", "no trend"], "significance": ["**", "ns"]},
        index=["Apr", "May"],
    )
    fig = monthly_trend_heatmap(monthly_results, unit_label="MAF")
    ax = fig.axes[0]
    assert "Monthly Monotonic Trends" in ax.get_title()
    texts = [t.get_text() for t in ax.texts]
    assert "**" in texts
    # Unlike seasonal/dekadal/daily, "ns" is shown literally, not blanked.
    assert "ns" in texts


def test_dekadal_trend_heatmap_places_periods_by_month_and_dekad():
    dekadal_results = pd.DataFrame(
        {"trend": ["increasing", "decreasing"], "significance": ["*", "ns"]},
        index=["Apr1", "May2"],
    )
    fig = dekadal_trend_heatmap(dekadal_results, unit_label="Cusecs")
    ax = fig.axes[0]
    assert "10-Daily Monotonic Trends" in ax.get_title()
    assert ax.get_xlabel() == "Dekad (1, 2, 3)"
    texts = [t.get_text() for t in ax.texts]
    assert "*" in texts


def test_daily_trend_heatmap_places_periods_by_month_and_day():
    daily_results = pd.DataFrame(
        {"trend": ["increasing", "no trend"], "significance": ["***", "ns"]},
        index=["Apr-01", "Mar-31"],
    )
    fig = daily_trend_heatmap(daily_results, unit_label="Cusecs")
    ax = fig.axes[0]
    assert "Daily Monotonic Trends" in ax.get_title()
    assert ax.get_xlabel() == "Day of Month"
    texts = [t.get_text() for t in ax.texts]
    assert "***" in texts
