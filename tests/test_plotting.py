"""Plot builders: LOWESS parity, bounds plots, and the Phase-4 (Section 12)
distribution/box-whisker/flood plot functions.

The time-series/ITA-scatter functions still have no test coverage in this
codebase; this file covers what was newly ported here rather than attempting
to backfill the whole module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as GoFigure

from hydrotrends.core.constants import FLOOD_LIMITS_1000CUSECS
from hydrotrends.stats.trends import SenSlope
from hydrotrends.viz.plotting import (
    boxwhisker_grid,
    compute_lowess,
    distribution_grid,
    flood_heatmap,
    flood_heatmap_interactive,
    flood_overlay_interactive,
    flood_overlay_static,
    flow_duration_curve_interactive,
    flow_duration_curve_static,
    parametric_bounds_plot,
    robust_bounds_plot,
    single_histogram,
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
