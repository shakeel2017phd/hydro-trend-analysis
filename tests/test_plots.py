"""generate_plots() (Section 12: distribution/duration/flood-exceedance plots)."""

from __future__ import annotations

import pytest

import hydrotrends as ht
from hydrotrends.core.constants import COL_FLOW_CUSECS, COL_VOL_MAF
from hydrotrends.plots import generate_plots


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


def test_generate_plots_requires_at_least_one_column(tendaily_pre, tmp_path):
    with pytest.raises(ValueError, match="ReportColumn"):
        generate_plots(tendaily_pre, tmp_path, columns=[])
