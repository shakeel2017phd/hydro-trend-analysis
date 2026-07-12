"""Innovative Trend Analysis: aggregate faithful to source, plus sub-trends."""

import math

import numpy as np

import hydrotrends as ht


def test_aggregate_slope_matches_source():
    x = np.array([10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0])  # n=8
    r = ht.innovative_trend_analysis(x)
    expect = 2.0 * (x[4:].mean() - x[:4].mean()) / 8
    assert math.isclose(r.slope, expect)
    assert r.trend == ht.TrendDirection.INCREASING and r.n_used == 8


def test_odd_length_drops_first():
    x = np.array([999.0, 10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0])  # 9 pts
    r = ht.innovative_trend_analysis(x)
    assert r.n_used == 8


def test_sub_trends_reveal_opposite_low_high():
    # low flows increase, high flows decrease
    s = np.array([1.0, 2.0, 3.0, 90.0, 91.0, 92.0, 5.0, 6.0, 7.0, 80.0, 81.0, 82.0])
    r = ht.innovative_trend_analysis(s)
    assert r.low_slope > 0 and r.high_slope < 0
    assert math.isclose(r.low_slope, 2 * (5.5 - 1.5) / 12)
    assert math.isclose(r.high_slope, 2 * (81.5 - 91.5) / 12)


def test_halves_sorted_ascending():
    r = ht.innovative_trend_analysis(np.array([5.0, 1.0, 9.0, 3.0, 8.0, 2.0, 7.0, 4.0]))
    assert list(r.first_half) == sorted(r.first_half)
    assert list(r.second_half) == sorted(r.second_half)


def test_soft_guards():
    assert ht.innovative_trend_analysis([1.0, 2.0, 3.0]).n_used == 0  # n<4
    assert math.isnan(ht.innovative_trend_analysis([1.0, 2.0, 3.0, 4.0]).low_slope)
