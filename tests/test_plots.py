"""generate_plots() (Section 12: distribution/duration/flood-exceedance
plots; Section 13: Daily-only per-calendar-month day-grids; Section 11:
whole-series trend suites + summary heatmaps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hydrotrends as ht
from hydrotrends.core.constants import COL_FLOW_CUSECS, COL_VOL_MAF
from hydrotrends.plots import (
    _write_all_trend_plots,
    _write_mean_shifts_summary,
    _write_parametric_bounds_month_grids,
    _write_robust_bounds_month_grids,
    _write_trend_suite,
    generate_plots,
)


def _relative_files(out) -> set[str]:
    return {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}


def test_generate_plots_10daily_input_skips_daily_scale(tendaily_pre, tmp_path):
    out = generate_plots(
        tendaily_pre,
        tmp_path,
        columns=[
            ht.ReportColumn(
                COL_FLOW_CUSECS,
                "Cusecs",
                volume_column=COL_VOL_MAF,
                volume_unit_label="MAF",
            )
        ],
    )
    files = _relative_files(out)
    root = "Distribution_and_Flood_Plots/Cusecs_MAF"

    # No Daily scale -- there's no day-level data to derive it from.
    assert not any(f.startswith(f"{root}/Daily/") for f in files)

    for name in (
        f"{root}/10Daily/10Daily_Distribution_Grid.png",
        f"{root}/10Daily/10Daily_BoxWhisker_Grid.png",
        f"{root}/10Daily/10Daily_Duration_Curve.png",
        f"{root}/10Daily/10Daily_Duration_Curve.html",
        f"{root}/10Daily/10Daily_Flood_Overlay.png",
        f"{root}/10Daily/10Daily_Flood_Overlay.html",
        f"{root}/10Daily/10Daily_Flood_Heatmap.png",
        f"{root}/10Daily/10Daily_Flood_Heatmap.html",
        f"{root}/Monthly/Monthly_Distribution_Grid.png",
        f"{root}/Monthly/Monthly_BoxWhisker_Grid.png",
        f"{root}/Monthly/Monthly_Duration_Curve.png",
        f"{root}/Seasonal/Seasonal_Distribution_Grid.png",
        f"{root}/Seasonal/Seasonal_BoxWhisker_Grid.png",
        f"{root}/Seasonal/Seasonal_Duration_Curve.png",
        f"{root}/Annual/Annual_Distribution.png",
        f"{root}/Annual/Annual_Duration_Curve.png",
    ):
        assert name in files, name

    # Monthly/Seasonal/Annual have no flood exceedance (volume-based). Check
    # the filename only, not the full path -- the root dir itself is named
    # "Distribution_and_Flood_Plots".
    def _filenames_under(scale_dir: str) -> list[str]:
        return [f.rsplit("/", 1)[-1] for f in files if f"/{scale_dir}/" in f]

    assert not any("Flood" in name for name in _filenames_under("Monthly"))
    assert not any("Flood" in name for name in _filenames_under("Seasonal"))
    assert not any("Flood" in name for name in _filenames_under("Annual"))


def test_generate_plots_omits_volume_scales_without_pairing(tendaily_pre, tmp_path):
    out = generate_plots(
        tendaily_pre,
        tmp_path,
        columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")],
    )
    files = _relative_files(out)
    root = "Distribution_and_Flood_Plots/Cusecs"
    assert any(f.startswith(f"{root}/10Daily/") for f in files)
    assert not any(f.startswith(f"{root}/Monthly/") for f in files)
    assert not any(f.startswith(f"{root}/Seasonal/") for f in files)
    assert not any(f.startswith(f"{root}/Annual/") for f in files)


def test_generate_plots_daily_input_writes_per_month_grids(daily_pre, tmp_path):
    out = generate_plots(
        daily_pre,
        tmp_path,
        columns=[
            ht.ReportColumn(
                COL_FLOW_CUSECS,
                "Cusecs",
                volume_column=COL_VOL_MAF,
                volume_unit_label="MAF",
            )
        ],
    )
    files = _relative_files(out)
    root = "Distribution_and_Flood_Plots/Cusecs_MAF"

    for month in ("Apr", "Jan", "Mar"):
        assert f"{root}/Daily/Daily_Distribution_{month}.png" in files
        assert f"{root}/Daily/Daily_BoxWhisker_{month}.png" in files
    assert f"{root}/Daily/Daily_Duration_Curve.png" in files
    assert f"{root}/Daily/Daily_Flood_Overlay.png" in files
    assert f"{root}/Daily/Daily_Flood_Heatmap.png" in files
    # 10-Daily is also produced -- derived from the daily record.
    assert f"{root}/10Daily/10Daily_Distribution_Grid.png" in files

    # Section 13: the Daily-only per-calendar-month day-grid system.
    s13_root = "Daily_Detailed_Plots_v24/Cusecs"
    for cell_type in (
        "BoxWhisker",
        "HistogramKDE",
        "Violin",
        "ParametricBounds",
        "RobustBounds",
        "Anomalies",
        "ITAScatter",
        "DecadalBlocks",
        "RecentVsLongTerm",
    ):
        for month in ("Apr", "Jan", "Feb"):  # Feb: exercises the <=30-day layout
            assert f"{s13_root}/{cell_type}/Daily_{cell_type}_{month}.png" in files
    assert f"{s13_root}/DurationCurve/Daily_Duration_Curve.png" in files
    assert f"{s13_root}/DurationCurve/Daily_Duration_Curve.html" in files
    assert f"{s13_root}/FloodHeatmap/Daily_Flood_Heatmap.png" in files
    assert f"{s13_root}/FloodHeatmap/Daily_Flood_Heatmap.html" in files
    assert f"{s13_root}/FloodOverlay/Daily_Flood_Overlay.png" in files
    # No interactive counterpart for the Daily-specific flood overlay -- the
    # source only produces a static PNG here (unlike Section 12's version).
    assert f"{s13_root}/FloodOverlay/Daily_Flood_Overlay.html" not in files

    # Section 11: the whole-series trend suites + summary heatmaps. The
    # bundled sample only spans 3 hydrological years -- below the suites'
    # own 10-year minimum -- so every per-series 6-plot suite (Ann_*/Seas_*/
    # Mon_*/Dek_*/Daily_*) is a no-op; only the aggregated summary artifacts
    # that don't need 10 years should exist.
    s11_root = "Plots_Volume_MAF"
    assert f"{s11_root}/Mean_Shifts_Summary_MAF.xlsx" in files
    assert f"{s11_root}/Summary_Heatmaps/Summary_Seasonal_Shifts_MAF.png" in files
    assert f"{s11_root}/Summary_Heatmaps/Summary_Seasonal_Heatmap_MAF.png" in files
    assert f"{s11_root}/Summary_Heatmaps/Summary_Dekadal_Heatmap_Cusecs.png" in files
    assert f"{s11_root}/Summary_Heatmaps/Summary_Daily_Heatmap_Cusecs.png" in files
    # Monthly heatmap needs at least one month with >=4 years of data; the
    # 3-year sample has none, so it's correctly skipped rather than written
    # with an entirely blank panel.
    assert f"{s11_root}/Summary_Heatmaps/Summary_Monthly_Heatmap_MAF.png" not in files
    assert not any(f.startswith(f"{s11_root}/Annual_Plots/") for f in files)
    assert not any(f.startswith(f"{s11_root}/Daily_Plots/") for f in files)


def test_generate_plots_10daily_input_skips_section13(tendaily_pre, tmp_path):
    out = generate_plots(
        tendaily_pre, tmp_path, columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")]
    )
    files = _relative_files(out)
    assert not any(f.startswith("Daily_Detailed_Plots_v24/") for f in files)


def test_generate_plots_requires_at_least_one_column(tendaily_pre, tmp_path):
    with pytest.raises(ValueError, match="ReportColumn"):
        generate_plots(tendaily_pre, tmp_path, columns=[])


def _one_day_by_period(rng, n_years=20, trend=800.0):
    """A single fabricated "Apr-01" period, values trending upward across
    years -- enough for Sen's slope and a change point to both be non-NaN.
    Everything else in HYDRO_MONTHS is absent, so the month-grid writers
    (which loop over all 12 months but skip absent ones) only touch "Apr",
    keeping this fast instead of paying generate_plots()'s full-daily-input
    cost just to check the Sen's-slope/change-point wiring.
    """
    years = np.arange(2000, 2000 + n_years)
    values = 30_000.0 + trend * (years - years[0]) + rng.normal(0, 50, n_years)
    values[n_years // 2 :] += 5000  # a clear step, for a real change point
    return {"Apr-01": pd.Series(values, index=years)}


def test_write_parametric_bounds_month_grids_wires_sen_slope_and_change_point(
    tmp_path, monkeypatch
):
    """Direct test of the plots.py wiring (not just the cell-drawing function,
    already covered in test_plotting.py): stats functions are actually
    called per day and their results reach the cell's legend.
    """
    rng = np.random.default_rng(0)
    by_period = _one_day_by_period(rng)
    saved: dict[str, object] = {}
    monkeypatch.setattr(
        "hydrotrends.plots._save_static",
        lambda fig, path, *, dpi: saved.setdefault("fig", fig),
    )
    _write_parametric_bounds_month_grids(
        tmp_path, by_period, {"Apr": ["Apr-01"]}, "Cusecs"
    )
    labels = [t.get_text() for t in saved["fig"].axes[0].get_legend().get_texts()]
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Bai-Perron" in lbl for lbl in labels)


def test_write_robust_bounds_month_grids_wires_sen_slope_and_change_point(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(1)
    by_period = _one_day_by_period(rng)
    saved: dict[str, object] = {}
    monkeypatch.setattr(
        "hydrotrends.plots._save_static",
        lambda fig, path, *, dpi: saved.setdefault("fig", fig),
    )
    _write_robust_bounds_month_grids(tmp_path, by_period, {"Apr": ["Apr-01"]}, "Cusecs")
    labels = [t.get_text() for t in saved["fig"].axes[0].get_legend().get_texts()]
    assert any("Sen's Slope" in lbl for lbl in labels)
    assert any("Pettitt" in lbl for lbl in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: whole-series trend suites + summary heatmaps.
# ─────────────────────────────────────────────────────────────────────────────
def _trending_series(rng, n_years=20, trend=800.0):
    """A single fabricated whole series, long enough (>=10 years) for the
    trend suite's own minimum, with a clear trend and a mid-series step (so
    Sen's slope and a change point are both well-defined, and the two
    sorted-and-paired ITA halves aren't degenerate).
    """
    years = np.arange(2000, 2000 + n_years)
    values = 30_000.0 + trend * (years - years[0]) + rng.normal(0, 50, n_years)
    values[n_years // 2 :] += 5_000
    return years, values


def test_write_trend_suite_writes_all_six_plots_and_summary_rows(tmp_path):
    rng = np.random.default_rng(2)
    years, values = _trending_series(rng)

    dec_rows, win_rows = _write_trend_suite(
        tmp_path, "Ann", "Annual", years, values, "MAF"
    )

    for suffix in (
        "1_Parametric",
        "2_Robust",
        "3_Anomalies",
        "4_ITA_Scatter",
        "5_Decadal_Blocks",
        "6_Sliding_Windows",
    ):
        assert (tmp_path / f"Ann_{suffix}.png").exists(), suffix

    assert dec_rows and all(row["Scale"] == "Annual" for row in dec_rows)
    assert {"Decade_Block", "Years_Count", "Mean_Value_MAF"} <= dec_rows[0].keys()
    assert win_rows and all(row["Scale"] == "Annual" for row in win_rows)
    assert {"Period", "Span", "Years_Included", "Mean_Value_MAF"} <= win_rows[0].keys()
    assert [row["Period"] for row in win_rows] == [
        "Overall Climatological",
        "Last 40 Years",
        "Last 20 Years",
        "Last 10 Years",
        "Last 5 Years",
    ]


def test_write_trend_suite_skips_series_shorter_than_ten_years(tmp_path):
    years = np.arange(2000, 2008, dtype="float64")  # 8 years
    values = np.linspace(100.0, 110.0, years.size)

    dec_rows, win_rows = _write_trend_suite(
        tmp_path, "Mon_Apr", "Monthly (Apr)", years, values, "Cusecs"
    )

    assert dec_rows == []
    assert win_rows == []
    assert list(tmp_path.iterdir()) == []


def test_write_mean_shifts_summary_writes_two_sheets(tmp_path):
    dec_rows = [
        {
            "Scale": "Annual",
            "Decade_Block": "1991–2000",
            "Years_Count": 10,
            "Mean_Value_MAF": 12.5,
        }
    ]
    win_rows = [
        {
            "Scale": "Annual",
            "Period": "Overall Climatological",
            "Span": "1991–2020",
            "Years_Included": 30,
            "Mean_Value_MAF": 12.5,
        }
    ]

    _write_mean_shifts_summary(tmp_path, "MAF", dec_rows, win_rows)
    path = tmp_path / "Mean_Shifts_Summary_MAF.xlsx"
    assert path.exists()

    decadal = pd.read_excel(path, sheet_name="Decadal_Blocks")
    sliding = pd.read_excel(path, sheet_name="Sliding_Windows")
    assert list(decadal["Scale"]) == ["Annual"]
    assert list(sliding["Period"]) == ["Overall Climatological"]


def test_write_all_trend_plots_noop_without_volume_column(tendaily_pre, tmp_path):
    rc = ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")  # no volume pairing
    _write_all_trend_plots(
        tendaily_pre,
        rc,
        tmp_path,
        is_daily=False,
        include_daily_period_suites=True,
    )
    assert list(tmp_path.iterdir()) == []


def test_write_all_trend_plots_respects_include_daily_period_suites_flag(
    daily_pre, tmp_path, monkeypatch
):
    rc = ht.ReportColumn(
        COL_FLOW_CUSECS, "Cusecs", volume_column=COL_VOL_MAF, volume_unit_label="MAF"
    )
    calls: list[int] = []
    monkeypatch.setattr(
        "hydrotrends.plots._write_daily_trend_suites",
        lambda *a, **k: (calls.append(1), ([], []))[1],
    )

    _write_all_trend_plots(
        daily_pre, rc, tmp_path, is_daily=True, include_daily_period_suites=False
    )
    assert calls == []

    _write_all_trend_plots(
        daily_pre, rc, tmp_path, is_daily=True, include_daily_period_suites=True
    )
    assert calls == [1]


def test_generate_plots_include_daily_period_suites_defaults_to_true():
    import inspect

    default = (
        inspect.signature(generate_plots)
        .parameters["include_daily_period_suites"]
        .default
    )
    assert default is True
