"""Preprocessing: derived columns, volumes, seasons, year filtering."""
import math

import numpy as np
import pandas as pd

import hydrotrends as ht
from hydrotrends.core import constants as k


def _daily_year(start="2000-04-01", end="2001-03-31", flow=100_000.0):
    dates = pd.date_range(start, end)
    df = pd.DataFrame({k.COL_DATE: dates, k.COL_FLOW_CUSECS: flow})
    return df


def test_daily_volume_matches_formula():
    pre = ht.preprocess(_daily_year(), k.TimeResolution.DAILY)
    row = pre.hydro.iloc[0]
    expect = 100_000 * k.CUSEC_TO_M3S * k.SEC_PER_DAY / k.M3_PER_MAF
    assert math.isclose(row[k.COL_VOL_MAF], expect)
    assert row[k.COL_N_DAYS] == 1
    assert row[k.COL_PERIOD] == "Apr-01"


def test_seasons_both_schemes():
    pre = ht.preprocess(_daily_year(), k.TimeResolution.DAILY)
    h = pre.hydro.set_index(k.COL_DATE)
    assert h.loc["2000-06-05", k.COL_SEASON] == "Early_Kharif"
    assert h.loc["2000-06-20", k.COL_SEASON] == "Late_Kharif"
    assert h.loc["2001-01-15", k.COL_SEASON] == "Rabi"
    assert h.loc["2001-01-15", k.COL_MET_SEASON] == "Winter"
    assert h.loc["2000-07-15", k.COL_MET_SEASON] == "Monsoon"


def test_incomplete_year_dropped():
    full = _daily_year()
    partial = pd.DataFrame({
        k.COL_DATE: pd.date_range("2001-05-01", "2001-05-31"),
        k.COL_FLOW_CUSECS: 100_000.0,
    })
    pre = ht.preprocess(pd.concat([full, partial], ignore_index=True), k.TimeResolution.DAILY)
    assert set(pre.hydro[k.COL_HYDRO_YEAR].unique()) == {2000}


def test_tendaily_ndays_leap_aware():
    # dekad-end dates for one hydro year
    rows = []
    for yr, mo in [(2000, m) for m in (4,5,6,7,8,9,10,11,12)] + [(2001, m) for m in (1,2,3)]:
        dim = pd.Timestamp(yr, mo, 1).days_in_month
        for d in (10, 20, dim):
            rows.append(pd.Timestamp(yr, mo, d))
    df = pd.DataFrame({k.COL_DATE: rows, k.COL_FLOW_CUSECS: 100_000.0})
    pre = ht.preprocess(df, k.TimeResolution.TEN_DAILY)
    h = pre.hydro
    feb3 = h[(h[k.COL_MONTH_NUM] == 2) & (h[k.COL_DEKAD] == 3)][k.COL_N_DAYS].iloc[0]
    assert feb3 == 8  # 2001 non-leap February: 28 - 20


def test_daily_and_tendaily_annual_volume_agree(daily_pre, tendaily_pre):
    dd = daily_pre.hydro.groupby(k.COL_HYDRO_YEAR)[k.COL_VOL_MAF].sum()
    tt = tendaily_pre.hydro.groupby(k.COL_HYDRO_YEAR)[k.COL_VOL_MAF].sum()
    for yr in dd.index:
        assert abs(dd[yr] - tt[yr]) / dd[yr] < 0.02
