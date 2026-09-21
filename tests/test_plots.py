"""generate_plots() (Section 12: distribution/duration/flood-exceedance
plots; Section 13: Daily-only per-calendar-month day-grids)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hydrotrends as ht
from hydrotrends.core.constants import COL_FLOW_CUSECS, COL_VOL_MAF
from hydrotrends.plots import (
    _write_parametric_bounds_month_grids,
    _write_robust_bounds_month_grids,
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
        columns=[ht.ReportColumn(COL_FLOW_CUSECS, "Cusecs")],
    )
    files = _relative_files(out)
    root = "Distribution_and_Flood_Plots/Cusecs"

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
