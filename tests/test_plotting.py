"""Plot builders: LOWESS parity and the parametric/robust bounds plots.

The rest of :mod:`hydrotrends.viz.plotting` (timeseries, ITA scatter, flow-
duration curves, distribution grid, flood heatmap) has no test coverage yet
in this codebase; this file covers what was newly ported here rather than
attempting to backfill the whole module.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from hydrotrends.stats.trends import SenSlope
from hydrotrends.viz.plotting import (
    compute_lowess,
    parametric_bounds_plot,
    robust_bounds_plot,
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
