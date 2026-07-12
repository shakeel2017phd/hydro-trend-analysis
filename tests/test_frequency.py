"""Flow-duration / exceedance and Q-percentile indices."""

import math

import numpy as np

import hydrotrends as ht
from hydrotrends.stats.frequency import (
    COL_FDC_EXCEEDANCE,
    COL_FDC_VALUE,
)


def test_fdc_weibull():
    x = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
    fdc = ht.flow_duration_curve(x)
    assert list(fdc[COL_FDC_VALUE]) == [50, 40, 30, 20, 10]  # descending
    expected = [i / 6 * 100 for i in range(1, 6)]  # rank/(n+1)*100
    assert np.allclose(fdc[COL_FDC_EXCEEDANCE], expected)


def test_exceedance_counts():
    s = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    assert ht.exceedance_counts(s, {"A": 250.0, "B": 450.0}) == {"A": 3, "B": 1}


def test_exceedance_probability():
    s = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    assert math.isclose(ht.exceedance_probability(s, 300.0), 60.0)


def test_flow_percentiles_ordering():
    q = ht.flow_percentiles(np.arange(1, 101))
    assert q["Q90"] < q["Q50"] < q["Q10"]  # low-flow < median < high-flow
    assert math.isclose(q["Q50"], np.percentile(np.arange(1, 101), 50))


def test_empty():
    assert ht.flow_duration_curve([]).empty
    assert math.isnan(ht.exceedance_probability([], 5.0))
    assert math.isnan(ht.flow_percentiles([])["Q50"])
