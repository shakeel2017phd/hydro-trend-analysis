"""Descriptive statistics match the source formulas and degrade gracefully."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import hydrotrends as ht


def test_matches_source_formulas():
    x = np.array([12.0, 7.0, 22.0, 9.0, 15.0, 30.0, 5.0])
    r = ht.describe(x)
    assert math.isclose(r.mean, x.mean())
    assert math.isclose(r.std, x.std(ddof=1))  # sample std
    assert math.isclose(r.cv_pct, x.std(ddof=1) / x.mean() * 100)
    assert math.isclose(r.skewness, sp.skew(x, bias=False))
    assert math.isclose(r.kurtosis, sp.kurtosis(x, bias=False))
    assert math.isclose(r.p90, np.percentile(x, 90))


def test_nan_dropped():
    assert ht.describe([1.0, np.nan, 3.0]).n == 2


@pytest.mark.parametrize(
    "data,std_nan,skew_nan,kurt_nan",
    [
        ([], True, True, True),
        ([5.0], True, True, True),
        ([5.0, 9.0], False, True, True),
        ([1.0, 2.0, 4.0], False, False, True),
    ],
)
def test_small_sample_degradation(data, std_nan, skew_nan, kurt_nan):
    r = ht.describe(data)
    assert math.isnan(r.std) is std_nan
    assert math.isnan(r.skewness) is skew_nan
    assert math.isnan(r.kurtosis) is kurt_nan


def test_cv_nan_on_zero_mean():
    assert math.isnan(ht.describe([-3.0, 0.0, 3.0]).cv_pct)


def test_describe_by(daily_pre):
    from hydrotrends.core.constants import COL_FLOW_CUSECS, COL_MONTH

    out = ht.describe_by(daily_pre.hydro, value_col=COL_FLOW_CUSECS, by=COL_MONTH)
    assert len(out) == 12 and "mean" in out.columns
