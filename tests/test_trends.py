"""Trend tests: cross-validated against reference implementations."""

import math

import numpy as np
import pytest
from scipy import stats

import hydrotrends as ht

try:
    import pymannkendall as pmk
except ImportError:
    pmk = None

requires_pmk = pytest.mark.skipif(pmk is None, reason="pymannkendall not installed")


@requires_pmk
def test_mann_kendall_matches_pymannkendall(trend_series):
    mk = ht.mann_kendall(trend_series)
    ref = pmk.original_test(trend_series)
    assert math.isclose(mk.s, ref.s)
    assert math.isclose(mk.z_score, ref.z, rel_tol=1e-9)
    assert math.isclose(mk.p_value, ref.p, rel_tol=1e-9)
    assert str(mk.trend) == ref.trend


@requires_pmk
def test_sens_slope_matches(trend_series):
    assert math.isclose(
        ht.sens_slope(trend_series).slope,
        pmk.original_test(trend_series).slope,
        rel_tol=1e-9,
    )


@requires_pmk
def test_hamed_rao_matches(trend_series):
    mmk = ht.mann_kendall_modified(trend_series)
    ref = pmk.hamed_rao_modification_test(trend_series)
    assert math.isclose(mmk.z_score, ref.z, rel_tol=1e-9)
    assert math.isclose(mmk.p_value, ref.p, rel_tol=1e-9)


def test_ols_matches_scipy(trend_series):
    lf = ht.linear_regression(trend_series)
    lr = stats.linregress(np.arange(len(trend_series)), trend_series)
    assert math.isclose(lf.slope, lr.slope, rel_tol=1e-9)
    assert math.isclose(lf.p_value, lr.pvalue, rel_tol=1e-7)
    assert math.isclose(lf.r2, lr.rvalue**2, rel_tol=1e-9)


def test_modified_falls_back_below_10():
    x = np.arange(8.0)
    assert ht.mann_kendall_modified(x) == ht.mann_kendall(x)


def test_directions():
    assert ht.mann_kendall(np.arange(20)).trend == ht.TrendDirection.INCREASING
    assert ht.mann_kendall(np.arange(20)[::-1]).trend == ht.TrendDirection.DECREASING


def test_soft_guards():
    assert np.isnan(ht.sens_slope([5.0]).slope)
    assert np.isnan(ht.linear_regression([3.0]).slope)
    assert ht.moving_average_trend([1.0, 2.0]) is None
    assert not np.isfinite(ht.log_linear_regression([-1.0, -2.0, 3.0]).slope)


def test_percent_slope():
    assert math.isclose(ht.percent_slope(2.0, 50.0), 4.0)
    assert np.isnan(ht.percent_slope(2.0, 0.0))
