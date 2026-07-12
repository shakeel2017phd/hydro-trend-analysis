"""Reader format handling, scaling, and validation."""
import numpy as np
import pandas as pd
import pytest

import hydrotrends as ht
from hydrotrends.core.constants import COL_DATE, COL_FLOW_CUSECS
from hydrotrends.core import exceptions as e


def _write_csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_reads_and_normalises_headers(tmp_path):
    p = _write_csv(tmp_path, "d.csv", pd.DataFrame(
        {"obs": ["02/04/2000", "01/04/2000"], "flow": [200.0, 100.0]}))
    df = ht.read_input(ht.InputSpec(p, "daily", date_column="obs", value_column="flow"))
    assert list(df.columns) == [COL_DATE, COL_FLOW_CUSECS]
    assert df[COL_DATE].is_monotonic_increasing  # sorted


def test_value_scale(tmp_path):
    p = _write_csv(tmp_path, "k.csv", pd.DataFrame(
        {"Date": ["01/04/2000"], "v": [100.0]}))
    df = ht.read_input(ht.InputSpec(p, "10daily", value_column="v", value_scale=1000))
    assert df[COL_FLOW_CUSECS].iloc[0] == 100_000.0


@pytest.mark.parametrize("fmt,text,expected", [
    ("iso", "2000-01-05", pd.Timestamp("2000-01-05")),
    ("month_first", "03/04/2000", pd.Timestamp("2000-03-04")),
    ("auto", "03/04/2000", pd.Timestamp("2000-04-03")),
])
def test_daily_date_formats(tmp_path, fmt, text, expected):
    p = _write_csv(tmp_path, "f.csv", pd.DataFrame({"Date": [text], "inflow_cusec": [1]}))
    df = ht.read_input(ht.InputSpec(p, "daily", date_format=fmt))
    assert df[COL_DATE].iloc[0] == expected


def test_dekad_compact(tmp_path):
    p = _write_csv(tmp_path, "c.csv", pd.DataFrame(
        {"Date": ["2000Apr1", "2000Jun2"], "inflow_cusec": [1, 2]}))
    df = ht.read_input(ht.InputSpec(p, "10daily", date_format="dekad_compact"))
    assert set(df[COL_DATE]) == {pd.Timestamp("2000-04-01"), pd.Timestamp("2000-06-11")}


def test_missing_column_error(tmp_path):
    p = _write_csv(tmp_path, "m.csv", pd.DataFrame({"Date": ["01/04/2000"], "x": [1]}))
    with pytest.raises(e.MissingColumnError):
        ht.read_input(ht.InputSpec(p, "daily", value_column="flow"))


def test_missing_file_error():
    with pytest.raises(e.ReaderError):
        ht.read_input(ht.InputSpec("/no/such/file.csv", "daily"))


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("x")
    with pytest.raises(e.ReaderError):
        ht.read_input(ht.InputSpec(p, "daily"))
