"""Change-point detection locates a clean step and guards small samples."""

import pytest

import hydrotrends as ht


def test_pettitt_detects_step(step_series):
    r = ht.pettitt_test(step_series)
    assert r.index is not None and 8 <= r.index <= 12
    assert r.p_value < 0.01


def test_cusum_detects_step(step_series):
    assert 8 <= ht.cusum_change_point(step_series) <= 12


def test_bai_perron_detects_step(step_series):
    assert ht.bai_perron_change_point(step_series) == 10


def test_small_sample_guards():
    assert ht.pettitt_test([1.0, 2.0, 3.0]).index is None
    assert ht.cusum_change_point([1.0, 2.0, 3.0]) is None
    assert ht.bai_perron_change_point([1.0] * 8) is None


def test_bai_perron_only_single_break(step_series):
    with pytest.raises(NotImplementedError):
        ht.bai_perron_change_point(step_series, n_breaks=2)
