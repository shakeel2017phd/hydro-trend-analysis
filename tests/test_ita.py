"""Innovative Trend Analysis: aggregate faithful to source, plus sub-trends."""
import math

import numpy as np

import hydrotrends as ht


def test_aggregate_slope_matches_source():
    x = np.array([10., 12., 11., 13., 20., 22., 21., 23.])  # n=8
    r = ht.innovative_trend_analysis(x)
    expect = 2.0 * (x[4:].mean() - x[:4].mean()) / 8
    assert math.isclose(r.slope, expect)
    assert r.trend == ht.TrendDirection.INCREASING and r.n_used == 8


def test_odd_length_drops_first():
    x = np.array([999., 10., 12., 11., 13., 20., 22., 21., 23.])  # 9 pts
    r = ht.innovative_trend_analysis(x)
    assert r.n_used == 8


def test_sub_trends_reveal_opposite_low_high():
    # low flows increase, high flows decrease
    s = np.array([1., 2., 3., 90., 91., 92., 5., 6., 7., 80., 81., 82.])
    r = ht.innovative_trend_analysis(s)
    assert r.low_slope > 0 and r.high_slope < 0
    assert math.isclose(r.low_slope, 2 * (5.5 - 1.5) / 12)
    assert math.isclose(r.high_slope, 2 * (81.5 - 91.5) / 12)


def test_halves_sorted_ascending():
    r = ht.innovative_trend_analysis(np.array([5., 1., 9., 3., 8., 2., 7., 4.]))
    assert list(r.first_half) == sorted(r.first_half)
    assert list(r.second_half) == sorted(r.second_half)


def test_soft_guards():
    assert ht.innovative_trend_analysis([1., 2., 3.]).n_used == 0     # n<4
    assert math.isnan(ht.innovative_trend_analysis([1., 2., 3., 4.]).low_slope)
