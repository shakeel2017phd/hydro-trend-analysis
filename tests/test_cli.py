"""CLI: analyze command and error handling."""

import pytest
from openpyxl import load_workbook

import hydrotrends as ht
from hydrotrends import cli


def _daily_csv():
    return str(ht.datasets.sample_path("tarbela_daily"))


def test_analyze_single_input(tmp_path):
    out = tmp_path / "r.xlsx"
    rc = cli.main(
        ["analyze", _daily_csv(), "-o", str(out), "--value-column", "inflow_cusec"]
    )
    assert rc == 0 and out.exists()
    assert "Cover" in load_workbook(out).sheetnames


def test_analyze_writes_monthly_and_seasonal_volume_sheets(tmp_path):
    """Every flow unit is paired with its volume unit (UNIT_PAIRS) by default,
    so a plain run gets the monthly/seasonal/annual volume trend sheets too."""
    out = tmp_path / "r.xlsx"
    cli.main(
        ["analyze", _daily_csv(), "-o", str(out), "--value-column", "inflow_cusec"]
    )
    sheets = load_workbook(out).sheetnames
    assert "Monthly Trends (MAF)" in sheets
    assert "Seasonal Trends (MAF)" in sheets
    assert "Monthly Trends (BCM)" in sheets
    assert "Seasonal Trends (BCM)" in sheets


def test_no_descriptive(tmp_path):
    out = tmp_path / "r.xlsx"
    cli.main(
        [
            "analyze",
            _daily_csv(),
            "-o",
            str(out),
            "--value-column",
            "inflow_cusec",
            "--no-descriptive",
        ]
    )
    assert not any("Descriptive" in s for s in load_workbook(out).sheetnames)


def test_10daily_compact(tmp_path):
    out = tmp_path / "t.xlsx"
    rc = cli.main(
        [
            "analyze",
            str(ht.datasets.sample_path("tarbela_10daily")),
            "-o",
            str(out),
            "-r",
            "10daily",
            "--value-column",
            "Inflow_1000Cusecs",
            "--value-scale",
            "1000",
            "--date-format",
            "dekad_compact",
        ]
    )
    assert rc == 0 and out.exists()


def test_missing_file_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze", "/no/such.csv", "-o", str(tmp_path / "x.xlsx")])
    assert exc.value.code == 1


def test_no_input_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze", "-o", str(tmp_path / "x.xlsx")])
    assert exc.value.code == 1
