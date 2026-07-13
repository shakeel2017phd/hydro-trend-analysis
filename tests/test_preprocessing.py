"""Preprocessing: derived columns, volumes, seasons, year filtering."""

import math

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
    partial = pd.DataFrame(
        {
            k.COL_DATE: pd.date_range("2001-05-01", "2001-05-31"),
            k.COL_FLOW_CUSECS: 100_000.0,
        }
    )
    pre = ht.preprocess(
        pd.concat([full, partial], ignore_index=True), k.TimeResolution.DAILY
    )
    assert set(pre.hydro[k.COL_HYDRO_YEAR].unique()) == {2000}


def test_tendaily_ndays_leap_aware():
    # dekad-end dates for one hydro year
    rows = []
    for yr, mo in [(2000, m) for m in (4, 5, 6, 7, 8, 9, 10, 11, 12)] + [
        (2001, m) for m in (1, 2, 3)
    ]:
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


def test_monthly_volumes_matches_manual_groupby(daily_pre):
    monthly = ht.monthly_volumes(daily_pre.hydro)
    assert set(monthly.columns) == {
        k.COL_HYDRO_YEAR,
        k.COL_MONTH,
        k.COL_MONTH_NUM,
        k.COL_VOL_MAF,
        k.COL_VOL_BCM,
    }
    expect = (
        daily_pre.hydro.groupby([k.COL_HYDRO_YEAR, k.COL_MONTH])[k.COL_VOL_MAF]
        .sum()
        .rename("expect")
    )
    got = monthly.set_index([k.COL_HYDRO_YEAR, k.COL_MONTH])[k.COL_VOL_MAF]
    pd.testing.assert_series_equal(
        got.sort_index(), expect.sort_index(), check_names=False
    )


def test_monthly_volumes_sorted_by_year_then_calendar_month():
    # Matches the source's `monthly_vol` sort exactly: within a hydro year,
    # rows are ordered by calendar MonthNum (Jan..Dec), not hydro-month order.
    # analyze_monthly_volumes() is what reorders to Apr..Mar for reporting.
    pre = ht.preprocess(_daily_year(), k.TimeResolution.DAILY)
    monthly = ht.monthly_volumes(pre.hydro)
    assert list(monthly[k.COL_MONTH_NUM]) == sorted(monthly[k.COL_MONTH_NUM])


def test_hydro_seasonal_volumes_kharif_is_early_plus_late(daily_pre):
    seasonal = ht.hydro_seasonal_volumes(daily_pre.hydro)
    assert set(seasonal) == {
        "Early_Kharif",
        "Late_Kharif",
        "Kharif",
        "Rabi",
        "Annual",
    }
    combined = (
        seasonal["Early_Kharif"][k.COL_VOL_MAF] + seasonal["Late_Kharif"][k.COL_VOL_MAF]
    )
    pd.testing.assert_series_equal(
        seasonal["Kharif"][k.COL_VOL_MAF], combined, check_names=False
    )


def test_hydro_seasonal_volumes_annual_is_everything(daily_pre):
    seasonal = ht.hydro_seasonal_volumes(daily_pre.hydro)
    whole_year = daily_pre.hydro.groupby(k.COL_HYDRO_YEAR)[k.COL_VOL_MAF].sum()
    pd.testing.assert_series_equal(
        seasonal["Annual"][k.COL_VOL_MAF].sort_index(),
        whole_year.sort_index(),
        check_names=False,
    )


def test_met_seasonal_volumes_keys_and_calendar_seasons_match_manual_groupby(
    daily_pre,
):
    met = ht.met_seasonal_volumes(daily_pre.hydro)
    assert set(met) == {"Winter", "Spring", "Summer", "Monsoon", "Autumn"}

    # Seasons that don't cross the Dec/Jan boundary must match a plain
    # calendar-year groupby exactly.
    hydro = daily_pre.hydro
    for season in ("Spring", "Summer", "Monsoon", "Autumn"):
        expect = (
            hydro[hydro[k.COL_MET_SEASON] == season]
            .groupby(k.COL_YEAR)[k.COL_VOL_MAF]
            .sum()
        )
        pd.testing.assert_series_equal(
            met[season][k.COL_VOL_MAF].sort_index(),
            expect.sort_index(),
            check_names=False,
            check_index_type=False,
        )


def test_met_seasonal_volumes_winter_rolls_december_into_next_year():
    # Two consecutive hydro years so a complete Dec(Y0)+Jan(Y0+1)+Feb(Y0+1)
    # winter triple exists; each month gets a distinct constant flow so the
    # expected Winter total is unambiguous.
    frames = []
    for start_year, flow in ((2000, 100_000.0), (2001, 200_000.0)):
        dates = pd.date_range(f"{start_year}-04-01", f"{start_year + 1}-03-31")
        frames.append(pd.DataFrame({k.COL_DATE: dates, k.COL_FLOW_CUSECS: flow}))
    df = pd.concat(frames, ignore_index=True)
    pre = ht.preprocess(df, k.TimeResolution.DAILY)

    met = ht.met_seasonal_volumes(pre.hydro)
    winter = met["Winter"][k.COL_VOL_MAF]

    # Dec 2000 (flow 100_000) + Jan/Feb 2001 (flow 200_000) -> labelled 2001.
    hydro = pre.hydro
    dec_2000 = hydro[(hydro[k.COL_YEAR] == 2000) & (hydro[k.COL_MONTH_NUM] == 12)][
        k.COL_VOL_MAF
    ].sum()
    jan_feb_2001 = hydro[
        (hydro[k.COL_YEAR] == 2001) & (hydro[k.COL_MONTH_NUM].isin([1, 2]))
    ][k.COL_VOL_MAF].sum()
    assert math.isclose(winter.loc[2001], dec_2000 + jan_feb_2001)
