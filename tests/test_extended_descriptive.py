"""Phase-3 Descriptive Statistics Summary: formulas and degenerate-input handling."""

import math

import numpy as np
import pytest
from scipy import stats as sp

import hydrotrends as ht


def test_matches_known_formulas():
    x = np.array([12.0, 7.0, 22.0, 9.0, 15.0, 30.0, 5.0])
    r = ht.describe_extended(x)
    assert r.n == 7 and r.missing == 0
    assert math.isclose(r.mean, x.mean())
    assert math.isclose(r.std, x.std(ddof=1))
    assert math.isclose(r.variance, x.var(ddof=1))
    assert math.isclose(r.cv_pct, x.std(ddof=1) / x.mean() * 100)
    assert math.isclose(r.mad, np.mean(np.abs(x - x.mean())))
    assert math.isclose(r.se, x.std(ddof=1) / math.sqrt(len(x)))
    assert math.isclose(r.ci_lower, x.mean() - 1.96 * r.se)
    assert math.isclose(r.ci_upper, x.mean() + 1.96 * r.se)
    assert math.isclose(r.skewness, sp.skew(x, bias=False))
    assert math.isclose(r.kurtosis, sp.kurtosis(x, bias=False))
    assert math.isclose(r.pearson_skew, 3 * (r.mean - r.median) / r.std)
    assert math.isclose(r.bowley_skew, (r.p75 + r.p25 - 2 * r.median) / r.iqr)
    assert math.isclose(r.q90, np.percentile(x, 10))
    assert math.isclose(r.q95, np.percentile(x, 5))
    assert math.isclose(r.q99, np.percentile(x, 1))
    assert math.isclose(r.total, x.sum())


def test_missing_counted_separately_from_n():
    r = ht.describe_extended([1.0, np.nan, 3.0, np.nan])
    assert r.n == 2 and r.missing == 2


def test_no_flow_and_negative_counted_separately():
    r = ht.describe_extended([0.0, 0.0, -5.0, 3.0, 7.0])
    assert r.no_flow_count == 2
    assert r.negative_count == 1
    assert r.n == 5


def test_geometric_and_harmonic_mean_positive_only():
    x = [0.0, -3.0, 2.0, 4.0, 8.0]
    r = ht.describe_extended(x)
    positive = np.array([2.0, 4.0, 8.0])
    assert math.isclose(r.geometric_mean, sp.gmean(positive))
    assert math.isclose(r.harmonic_mean, sp.hmean(positive))


def test_geometric_and_harmonic_mean_nan_when_no_positive_values():
    r = ht.describe_extended([0.0, -1.0, -2.0])
    assert math.isnan(r.geometric_mean)
    assert math.isnan(r.harmonic_mean)


def test_record_completeness_requires_expected_n():
    r = ht.describe_extended([1.0, 2.0, 3.0])
    assert math.isnan(r.completeness_pct)
    r2 = ht.describe_extended([1.0, 2.0, 3.0], expected_n=6)
    assert math.isclose(r2.completeness_pct, 50.0)


def test_last_n_mean_uses_tail_of_given_order():
    values = list(range(1, 11))  # 1..10, already "chronological"
    r = ht.describe_extended(values, last_n_window=5)
    assert math.isclose(r.last_n_mean, np.mean([6, 7, 8, 9, 10]))


def test_last_n_mean_nan_without_window():
    r = ht.describe_extended([1.0, 2.0, 3.0])
    assert math.isnan(r.last_n_mean)


def test_mode_most_frequent_rounded_value():
    x = [1.0, 1.0, 1.0, 2.0, 3.0, 4.0]
    r = ht.describe_extended(x)
    assert r.mode == 1.0


def test_mode_kde_fallback_when_all_unique():
    rng = np.random.default_rng(3)
    x = rng.normal(loc=50, scale=5, size=200)
    r = ht.describe_extended(x)
    assert x.min() <= r.mode <= x.max()


@pytest.mark.parametrize(
    "data,expect_nan_fields",
    [
        ([], ["mean", "std", "skewness", "shapiro_stat"]),
        ([5.0], ["std", "se", "skewness", "shapiro_stat"]),
        ([5.0, 9.0], ["skewness", "l_skewness"]),
    ],
)
def test_small_sample_degrades_gracefully(data, expect_nan_fields):
    r = ht.describe_extended(data)
    for field in expect_nan_fields:
        assert math.isnan(getattr(r, field)), field


def test_all_identical_values_no_exception():
    r = ht.describe_extended([3.0] * 10)
    assert r.std == 0.0
    assert r.mode == 3.0
    assert math.isnan(r.skewness)  # scipy degrades on zero-variance input


def test_to_series_has_pretty_labels():
    s = ht.describe_extended([1.0, 2.0, 3.0]).to_series()
    assert "Std Dev" in s.index and "Q90" in s.index and "Sum (Volume)" in s.index
