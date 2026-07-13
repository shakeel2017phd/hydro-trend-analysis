"""Analysis orchestration and end-to-end report generation."""

import math

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import hydrotrends as ht
from hydrotrends.api import analyze_by_period, analyze_series
from hydrotrends.core.constants import (
    COL_DATE,
    COL_FLOW_CUSECS,
    COL_HYDRO_YEAR,
    COL_PERIOD,
    COL_VOL_MAF,
    HYDRO_MONTHS,
    TimeResolution,
)
from hydrotrends.viz.reports import STAT_GROUPS


def _synthetic_hydro_years(
    n_years=10, start_year=2010, base_flow=50_000.0, step=1000.0
):
    """``n_years`` complete hydrological years with a clear upward flow trend."""
    rng = np.random.default_rng(7)
    frames = []
    for i, year in enumerate(range(start_year, start_year + n_years)):
        dates = pd.date_range(f"{year}-04-01", f"{year + 1}-03-31")
        flow = base_flow + step * i + rng.normal(0, 50, len(dates))
        frames.append(pd.DataFrame({COL_DATE: dates, COL_FLOW_CUSECS: flow}))
    return ht.preprocess(pd.concat(frames, ignore_index=True), TimeResolution.DAILY)


def test_analyze_series_matches_individual_tests(trend_series):
    row = analyze_series(trend_series)
    assert row["trend"] == "increasing"
    assert math.isclose(row["p_value"], ht.mann_kendall(trend_series).p_value)
    assert math.isclose(row["sens_slope"], ht.sens_slope(trend_series).slope)


def test_all_row_keys_are_valid_schema_keys(step_series):
    row = analyze_series(step_series, years=range(2000, 2020))
    schema_keys = {c.key for g in STAT_GROUPS for c in g.columns}
    assert set(row) <= schema_keys


def test_change_point_years_populated(step_series):
    years = np.arange(2000, 2020)
    row = analyze_series(step_series, years=years)
    assert row["pettitt_cp_year"] == 2010
    assert row["cusum_cp_year"] == 2010 and row["bpcp_year"] == 2010


def test_n_below_4_descriptive_only():
    row = analyze_series([1.0, 2.0, 3.0])
    assert "trend" not in row and row["n"] == 3


def test_analyze_by_period_reindexes():
    recs = [
        {
            COL_PERIOD: p,
            COL_HYDRO_YEAR: 2000 + i,
            "flow": 100 + (3 * i if p == "Apr-01" else -2 * i),
        }
        for p in ("Apr-02", "Apr-01")
        for i in range(16)
    ]
    res = analyze_by_period(
        pd.DataFrame(recs), value_col="flow", order=["Apr-01", "Apr-02"]
    )
    assert list(res.index) == ["Apr-01", "Apr-02"]
    assert res.loc["Apr-01", "trend"] == "increasing"
    assert res.loc["Apr-02", "trend"] == "decreasing"


def test_analyze_preprocessed_ordering(daily_pre):
    res = ht.analyze_preprocessed(daily_pre, value_col=COL_FLOW_CUSECS)
    assert len(res) == 366 and res.index[0] == "Apr-01"


def test_generate_report_end_to_end(daily_pre, tmp_path):
    out = ht.generate_report(
        daily_pre,
        tmp_path / "report.xlsx",
        columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")],
        title="Test Report",
    )
    assert out.exists()
    wb = load_workbook(out)
    assert "Cover" in wb.sheetnames and "Trends (Cusecs)" in wb.sheetnames
    assert wb["Trends (Cusecs)"].cell(4, 1).value == "Apr-01"  # hydro order


# ── monthly / seasonal volume aggregation + trend analysis ──────────────────
def test_analyze_monthly_volumes_hydro_month_order_and_trend():
    pre = _synthetic_hydro_years()
    result = ht.analyze_monthly_volumes(pre.hydro, value_col=COL_VOL_MAF)
    assert list(result.index) == list(HYDRO_MONTHS)
    assert (result["n"] == 10).all()
    assert (result["trend"] == "increasing").all()
    assert (result["p_value"] < 0.05).all()


def test_analyze_seasonal_volumes_order_and_kharif_consistency():
    pre = _synthetic_hydro_years()
    result = ht.analyze_seasonal_volumes(pre.hydro, value_col=COL_VOL_MAF)
    assert [s.split()[0] for s in result.index] == [
        "Early",
        "Late",
        "Kharif",
        "Rabi",
        "Annual",
    ]
    assert (result["n"] == 10).all()
    assert (result["trend"] == "increasing").all()


def test_analyze_monthly_volumes_below_min_is_descriptive_only():
    pre = _synthetic_hydro_years(n_years=3)
    result = ht.analyze_monthly_volumes(pre.hydro, value_col=COL_VOL_MAF)
    assert (result["n"] == 3).all()
    assert "trend" not in result.columns


def test_describe_monthly_and_seasonal_volumes_agree_with_analyze():
    pre = _synthetic_hydro_years()
    trend = ht.analyze_monthly_volumes(pre.hydro, value_col=COL_VOL_MAF)
    desc = ht.describe_monthly_volumes(pre.hydro, value_col=COL_VOL_MAF)
    assert list(desc.index) == list(HYDRO_MONTHS)
    pd.testing.assert_series_equal(desc["mean"], trend["mean"])

    seas_trend = ht.analyze_seasonal_volumes(pre.hydro, value_col=COL_VOL_MAF)
    seas_desc = ht.describe_seasonal_volumes(pre.hydro, value_col=COL_VOL_MAF)
    pd.testing.assert_series_equal(seas_desc["mean"], seas_trend["mean"])


def test_generate_report_writes_volume_sheets_when_paired(tmp_path):
    pre = _synthetic_hydro_years()
    out = ht.generate_report(
        pre,
        tmp_path / "report.xlsx",
        columns=[
            ht.ReportColumn(
                COL_FLOW_CUSECS,
                "Cusecs",
                volume_column=COL_VOL_MAF,
                volume_unit_label="MAF",
            )
        ],
        title="Test Report",
    )
    wb = load_workbook(out)
    for name in (
        "Monthly Trends (MAF)",
        "Seasonal Trends (MAF)",
        "Monthly Descriptive (MAF)",
        "Seasonal Descriptive (MAF)",
    ):
        assert name in wb.sheetnames
    assert wb["Monthly Trends (MAF)"].cell(4, 1).value == "Apr"


def test_generate_report_omits_volume_sheets_without_pairing(daily_pre, tmp_path):
    out = ht.generate_report(
        daily_pre,
        tmp_path / "report.xlsx",
        columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")],
        title="Test Report",
    )
    wb = load_workbook(out)
    assert not any(
        "Monthly Trends" in s or "Seasonal Trends" in s for s in wb.sheetnames
    )
