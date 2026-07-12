"""Analysis orchestration and end-to-end report generation."""
import math

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import hydrotrends as ht
from hydrotrends.api import analyze_by_period, analyze_series
from hydrotrends.core.constants import COL_FLOW_CUSECS, COL_HYDRO_YEAR, COL_PERIOD
from hydrotrends.viz.reports import STAT_GROUPS


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
    recs = [{COL_PERIOD: p, COL_HYDRO_YEAR: 2000 + i, "flow": 100 + (3 * i if p == "Apr-01" else -2 * i)}
            for p in ("Apr-02", "Apr-01") for i in range(16)]
    res = analyze_by_period(pd.DataFrame(recs), value_col="flow",
                            order=["Apr-01", "Apr-02"])
    assert list(res.index) == ["Apr-01", "Apr-02"]
    assert res.loc["Apr-01", "trend"] == "increasing"
    assert res.loc["Apr-02", "trend"] == "decreasing"


def test_analyze_preprocessed_ordering(daily_pre):
    res = ht.analyze_preprocessed(daily_pre, value_col=COL_FLOW_CUSECS)
    assert len(res) == 366 and res.index[0] == "Apr-01"


def test_generate_report_end_to_end(daily_pre, tmp_path):
    out = ht.generate_report(
        daily_pre, tmp_path / "report.xlsx",
        columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")],
        title="Test Report",
    )
    assert out.exists()
    wb = load_workbook(out)
    assert "Cover" in wb.sheetnames and "Trends (Cusecs)" in wb.sheetnames
    assert wb["Trends (Cusecs)"].cell(4, 1).value == "Apr-01"  # hydro order
