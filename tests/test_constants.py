"""Domain constants stay faithful to the source values."""

import math
from collections import Counter

import hydrotrends as ht
from hydrotrends.core import constants as c


def test_conversion_factors_exact():
    assert math.isclose(c.CUSEC_TO_M3S, 0.3048**3, abs_tol=1e-15)
    assert c.M3_PER_MAF == 1233.4818375475238e6
    assert c.M3_PER_BCM == 1.0e9


def test_flood_limits_ascending():
    vals = [c.FLOOD_LIMITS_1000CUSECS[k] for k in c.FLOOD_CLASSES]
    assert vals == sorted(vals)
    assert dict(c.FLOOD_LIMITS_1000CUSECS) == {
        "LF": 250,
        "MF": 375,
        "HF": 500,
        "VHF": 650,
        "EHF": 800,
    }


def test_met_seasons_partition_year():
    months = Counter(m for grp in c.MET_SEASONS.values() for m in grp)
    assert set(months) == set(range(1, 13))
    assert all(count == 1 for count in months.values())


def test_mappings_read_only():
    for name in ("FLOOD_LIMITS_1000CUSECS", "MET_SEASONS", "RESOLUTION_INFO"):
        import pytest

        with pytest.raises(TypeError):
            getattr(c, name)["x"] = 1  # type: ignore[index]


def test_resolution_info_wired():
    di = c.RESOLUTION_INFO[c.TimeResolution.DAILY]
    ti = c.RESOLUTION_INFO[c.TimeResolution.TEN_DAILY]
    assert di.n_period_labels == 366 and ti.n_period_labels == 36
    assert ti.is_pre_aggregated and not di.is_pre_aggregated


def test_period_labels_include_leap_day():
    assert "Feb-29" in c.HYDRO_PERIODS and "Feb-29" in c.CAL_PERIODS
    assert len(c.HYDRO_DEKADS) == 36 and c.HYDRO_DEKADS[0] == "Apr1"


def test_met_periods_start_december():
    assert c.MET_MONTHS[0] == "Dec" and c.MET_MONTHS[-1] == "Nov"
    assert c.MET_PERIODS[0] == "Dec-01" and "Feb-29" in c.MET_PERIODS
    assert len(c.MET_DEKADS) == 36 and c.MET_DEKADS[0] == "Dec1"
    assert set(c.MET_MONTHS) == set(c.HYDRO_MONTHS) == set(c.CAL_MONTHS)


def test_strenum_string_equality():
    assert ht.TrendDirection.INCREASING == "increasing"
    assert ht.FlowUnit.CUSECS == "Cusecs"
    assert ht.TimeResolution.TEN_DAILY == "10daily"
