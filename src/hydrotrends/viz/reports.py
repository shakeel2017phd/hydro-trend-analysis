"""Excel report generation for :mod:`hydrotrends`.

Writes the formatted results workbook from the original script — the big
grouped analysis table (descriptive stats + every trend test side by side),
with coloured group headers, trend-coloured cells, and significance-coloured
cells. No formulas are written (a results report, not a recalculating model),
so the values are the computed statistics themselves.

The layout that was a loose list of tuples in the source is now a **typed**
schema (:class:`StatGroup` / :class:`StatColumn`): a mistyped result key or
column kind is caught by the type checker instead of silently mis-rendering.
Cells whose key is absent from the results frame render blank, so the schema can
include columns (e.g. change-point detection) before every producer exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import Cell
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

__all__ = [
    "ColumnKind",
    "StatColumn",
    "StatGroup",
    "STAT_GROUPS",
    "ResultSheet",
    "write_results_sheet",
    "write_descriptive_sheet",
    "CoverRow",
    "DEFAULT_COVER_ROWS",
    "write_cover_sheet",
    "write_workbook",
]

# ── palette (report-presentation; distinct from the analysis constants) ───────
_HDR, _SUB, _SUB2 = "1F4E79", "2E75B6", "4472C4"
_ALT, _WHT, _YEL = "D6E4F0", "FFFFFF", "FFF2CC"
_GRN, _RED = "C6EFCE", "FFC7CE"
_CP_CLR = "FCE4D6"
_FONT = "Arial"

_TREND_BG = {"increasing": _GRN, "decreasing": _RED, "no trend": _YEL}
_SIG_CLR = {"***": "CC0000", "**": "FF6600", "*": "FF9900", ".": "CCAA00", "ns": "808080"}


# ── typed table schema ────────────────────────────────────────────────────────
class ColumnKind(StrEnum):
    NUM = "num"
    INT = "int"
    TREND = "trend"
    SIG = "sig"
    CP_YEAR = "cp_year"


class StatColumn(NamedTuple):
    header: str
    key: str
    kind: ColumnKind


class StatGroup(NamedTuple):
    name: str
    color: str
    columns: tuple[StatColumn, ...]


def _num(header: str, key: str) -> StatColumn:
    return StatColumn(header, key, ColumnKind.NUM)


STAT_GROUPS: tuple[StatGroup, ...] = (
    StatGroup("DESCRIPTIVE STATISTICS", "1F4E79", (
        StatColumn("N", "n", ColumnKind.INT),
        _num("Mean", "mean"), _num("Median", "median"), _num("Std Dev", "std"),
        _num("CV (%)", "cv_pct"), _num("Min", "min"),
        _num("P10", "p10"), _num("P25", "p25"), _num("P75", "p75"), _num("P90", "p90"),
        _num("Max", "max"), _num("Skewness", "skew"), _num("Kurtosis", "kurt"),
    )),
    StatGroup("ORIGINAL MK + SEN'S SLOPE", "375623", (
        StatColumn("Trend", "trend", ColumnKind.TREND),
        _num("p-value", "p_value"), _num("Z-score", "z_score"),
        _num("Sen's Slope", "sens_slope"), _num("Sen's Slope (%/yr)", "sens_slope_pct"),
        StatColumn("Significance", "significance", ColumnKind.SIG),
    )),
    StatGroup("MODIFIED MK (Hamed-Rao)", "833C00", (
        StatColumn("MMK Trend", "mmk_trend", ColumnKind.TREND),
        _num("MMK p-value", "mmk_p"), _num("MMK Z-score", "mmk_z"),
        StatColumn("MMK Sig.", "mmk_sig", ColumnKind.SIG),
    )),
    StatGroup("LINEAR REGRESSION", "31538A", (
        _num("Slope", "lin_slope"), _num("Intercept", "lin_intercept"),
        _num("R\u00b2", "lin_r2"), _num("p-value", "lin_p"),
    )),
    StatGroup("LOG-LINEAR REGRESSION", "7030A0", (
        _num("Log Slope", "log_slope"), _num("R\u00b2", "log_r2"), _num("p-value", "log_p"),
    )),
    StatGroup("INNOVATIVE TREND (ITA)", "4B0082", (
        StatColumn("ITA Trend", "ita_trend", ColumnKind.TREND),
        _num("ITA Slope", "ita_slope"),
    )),
    StatGroup("5-YR BACKWARD MA", "215868", (
        _num("MA5 Mean", "ma5_mean"), _num("MA5 Std", "ma5_std"),
        StatColumn("MA5 Trend", "ma5_trend", ColumnKind.TREND),
    )),
    StatGroup("CHANGE-POINT DETECTION", "843C0C", (
        StatColumn("Pettitt CP Year", "pettitt_cp_year", ColumnKind.CP_YEAR),
        _num("Pettitt K", "pettitt_K"), _num("Pettitt p-value", "pettitt_p"),
        StatColumn("Pettitt Sig.", "pettitt_sig", ColumnKind.SIG),
        StatColumn("CUSUM CP Year", "cusum_cp_year", ColumnKind.CP_YEAR),
        StatColumn("Bai-Perron CP", "bpcp_year", ColumnKind.CP_YEAR),
    )),
)

# ── number-format key sets (from v26 smart_fmt) ───────────────────────────────
_INT_KEYS = {"n", "pettitt_cp_year", "cusum_cp_year", "bpcp_year"}
_PVAL_KEYS = {"p_value", "mmk_p", "lin_p", "log_p", "pettitt_p"}
_SLOPE_KEYS = {"sens_slope", "lin_slope", "ita_slope"}
_CUSEC_INT_KEYS = {
    "mean", "median", "std", "min", "max",
    "p10", "p25", "p75", "p90", "ma5_mean", "ma5_std",
}


def _smart_fmt(value: Any, key: str, unit: str) -> tuple[Any, str | None]:
    """Return (cell value, number format) matching the source's formatting rules."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "", None
    if key in _INT_KEYS:
        return int(round(float(value))), "0"
    v = float(value)
    if key in _PVAL_KEYS or key == "log_slope":
        return round(v, 3), "0.000"
    if key in _SLOPE_KEYS and unit in ("MAF", "BCM"):
        return round(v, 3), "0.000"
    if unit == "Cusecs" and key in _CUSEC_INT_KEYS:
        return int(round(v)), "0"
    return round(v, 2), "0.00"


# ── low-level styling ─────────────────────────────────────────────────────────
def _center() -> Alignment:
    return Alignment(horizontal="center", vertical="center", wrap_text=True)


def _left() -> Alignment:
    return Alignment(horizontal="left", vertical="center")


def _border() -> Border:
    edge = Side(style="thin")
    return Border(left=edge, right=edge, top=edge, bottom=edge)


def _cell(
    ws: Worksheet, row: int, col: int, value: Any, *,
    bg: str = _WHT, bold: bool = False, align: str = "center",
    number_format: str | None = None, size: int = 10, color: str = "000000",
) -> Cell:
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.font = Font(name=_FONT, bold=bold, size=size, color=color)
    cell.alignment = _center() if align == "center" else _left()
    cell.border = _border()
    if number_format:
        cell.number_format = number_format
    return cell


def _header(
    ws: Worksheet, row: int, col: int, value: Any, *,
    bg: str = _HDR, size: int = 10,
) -> Cell:
    return _cell(ws, row, col, value, bg=bg, bold=True, size=size, color=_WHT)


def _merged_header(
    ws: Worksheet, row: int, c1: int, c2: int, value: Any, *,
    bg: str = _HDR, size: int = 10,
) -> None:
    ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)
    _header(ws, row, c1, value, bg=bg, size=size)  # write the anchor cell only


def _trend_cell(ws: Worksheet, row: int, col: int, value: Any) -> None:
    label = str(value)
    _cell(ws, row, col, label, bg=_TREND_BG.get(label, _WHT), bold=True)


def _sig_cell(ws: Worksheet, row: int, col: int, value: Any) -> None:
    symbol = str(value).strip()
    starred = symbol in ("*", "**", "***")
    bg = _YEL if symbol in ("*", "**", "***", ".") else "F2F2F2"
    _cell(ws, row, col, symbol, bg=bg, bold=starred, color=_SIG_CLR.get(symbol, "808080"))


# ── sheet + workbook writers ──────────────────────────────────────────────────
def write_results_sheet(
    ws: Worksheet,
    table: pd.DataFrame,
    *,
    title: str,
    index_label: str = "Period",
    unit: str = "",
) -> None:
    """Write the grouped analysis table (one row per period) to a worksheet."""
    ws.sheet_view.showGridLines = False
    n_stat_cols = sum(len(g.columns) for g in STAT_GROUPS)
    total_cols = 1 + n_stat_cols

    _merged_header(ws, 1, 1, total_cols, title, bg=_HDR, size=12)

    _header(ws, 2, 1, index_label, bg=_SUB)
    col = 2
    for group in STAT_GROUPS:
        span = len(group.columns)
        _merged_header(ws, 2, col, col + span - 1, group.name, bg=group.color, size=9)
        col += span

    _header(ws, 3, 1, f"{index_label}  [{unit}]", bg=_SUB2, size=9)
    col = 2
    for group in STAT_GROUPS:
        for column in group.columns:
            _header(ws, 3, col, column.header, bg=group.color, size=9)
            col += 1

    for i, (idx, row) in enumerate(table.iterrows()):
        r = 4 + i
        row_bg = _ALT if i % 2 == 0 else _WHT
        _cell(ws, r, 1, str(idx), bg=row_bg, bold=True, align="left")
        col = 2
        for group in STAT_GROUPS:
            for column in group.columns:
                value = row.get(column.key, math.nan)
                if column.kind is ColumnKind.TREND:
                    _trend_cell(ws, r, col, value)
                elif column.kind is ColumnKind.SIG:
                    _sig_cell(ws, r, col, value)
                elif column.kind is ColumnKind.CP_YEAR:
                    v, fmt = _smart_fmt(value, column.key, unit)
                    _cell(ws, r, col, v, bg=_CP_CLR if v != "" else row_bg,
                          bold=v != "", number_format=fmt)
                else:
                    v, fmt = _smart_fmt(value, column.key, unit)
                    _cell(ws, r, col, v, bg=row_bg, number_format=fmt)
                col += 1

    ws.freeze_panes = "B4"

    widths = [15.0]
    for group in STAT_GROUPS:
        for column in group.columns:
            if column.kind is ColumnKind.CP_YEAR or column.key == "n":
                widths.append(9.0)
            elif column.kind in (ColumnKind.TREND, ColumnKind.SIG):
                widths.append(13.0)
            else:
                widths.append(11.0)
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width


@dataclass(frozen=True)
class ResultSheet:
    """One results table plus its tab name and labels."""

    name: str
    table: pd.DataFrame
    title: str
    index_label: str = "Period"
    unit: str = ""


_INVALID_SHEET_CHARS = str.maketrans({c: "_" for c in r"[]:*?/\\"})


def _safe_sheet_name(name: str) -> str:
    return name.translate(_INVALID_SHEET_CHARS)[:31] or "Sheet"


def write_workbook(path: str | Path, sheets: Sequence[ResultSheet]) -> Path:
    """Write one or more result sheets to an .xlsx workbook. Returns the path."""
    if not sheets:
        raise ValueError("a workbook needs at least one sheet")
    wb = Workbook()
    default = wb.active
    for spec in sheets:
        ws = wb.create_sheet(title=_safe_sheet_name(spec.name))
        write_results_sheet(
            ws, spec.table, title=spec.title,
            index_label=spec.index_label, unit=spec.unit,
        )
    wb.remove(default)  # drop the auto-created empty sheet

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive-statistics sheet
# ─────────────────────────────────────────────────────────────────────────────
# Field name (as produced by stats.descriptive.describe_by) -> column header.
_DESCRIPTIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("n", "N"), ("mean", "Mean"), ("median", "Median"), ("std", "Std Dev"),
    ("cv_pct", "CV (%)"), ("minimum", "Min"), ("p10", "P10"), ("p25", "P25"),
    ("p75", "P75"), ("p90", "P90"), ("maximum", "Max"), ("iqr", "IQR"),
    ("value_range", "Range"), ("skewness", "Skewness"), ("kurtosis", "Kurtosis"),
    ("total", "Sum"),
)
_ALWAYS_TWO_DP = {"cv_pct", "skewness", "kurtosis"}


def _descriptive_fmt(field: str, value: Any, unit: str) -> tuple[Any, str | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "", None
    if field == "n":
        return int(round(float(value))), "0"
    v = float(value)
    if field in _ALWAYS_TWO_DP:
        return round(v, 2), "0.00"
    if unit == "Cusecs":
        return int(round(v)), "0"
    return round(v, 2), "0.00"


def write_descriptive_sheet(
    ws: Worksheet,
    stats: pd.DataFrame,
    *,
    title: str,
    index_label: str = "Period",
    unit: str = "",
    columns: tuple[tuple[str, str], ...] = _DESCRIPTIVE_COLUMNS,
) -> None:
    """Write a per-period descriptive-statistics table.

    ``stats`` is a frame indexed by period with the fields produced by
    :func:`hydrotrends.stats.descriptive.describe_by`.
    """
    ws.sheet_view.showGridLines = False
    total_cols = 1 + len(columns)
    _merged_header(ws, 1, 1, total_cols, title, bg=_HDR, size=12)

    _header(ws, 2, 1, f"{index_label}  [{unit}]", bg=_SUB, size=9)
    for j, (_field, header) in enumerate(columns, start=2):
        _header(ws, 2, j, header, bg=_SUB, size=9)

    for i, (idx, row) in enumerate(stats.iterrows()):
        r = 3 + i
        row_bg = _ALT if i % 2 == 0 else _WHT
        _cell(ws, r, 1, str(idx), bg=row_bg, bold=True, align="left")
        for j, (field, _header_text) in enumerate(columns, start=2):
            v, fmt = _descriptive_fmt(field, row.get(field, math.nan), unit)
            _cell(ws, r, j, v, bg=row_bg, number_format=fmt)

    ws.freeze_panes = "B3"
    widths = [15.0] + [10.0] * len(columns)
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width


# ─────────────────────────────────────────────────────────────────────────────
# Cover sheet
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CoverRow:
    """One cover-sheet row: a section header, or a label + description."""

    label: str
    description: str = ""
    is_section: bool = False


DEFAULT_COVER_ROWS: tuple[CoverRow, ...] = (
    CoverRow("SHEET INDEX", is_section=True),
    CoverRow("*_Trend_* sheets",
             "Descriptive statistics + all trend tests, one row per period"),
    CoverRow("*_Descriptive_* sheets", "Per-period descriptive statistics"),
    CoverRow(""),
    CoverRow("NUMBER FORMAT RULES", is_section=True),
    CoverRow("Cusecs (mean, min, max, ...)", "0 decimal places (whole numbers)"),
    CoverRow("Cumecs / Volumes / R\u00b2 / Z", "2 decimal places"),
    CoverRow("p-values", "3 decimal places"),
    CoverRow("Sen's / Linear / ITA slope", "3 decimal places"),
)

_COVER_WIDTH = 11  # columns A..K


def write_cover_sheet(
    ws: Worksheet,
    *,
    title: str,
    subtitle: str = "",
    rows: Sequence[CoverRow] = DEFAULT_COVER_ROWS,
) -> None:
    """Write a cover sheet with a title, optional subtitle, and index rows."""
    ws.sheet_view.showGridLines = False
    for c in range(1, _COVER_WIDTH + 1):
        ws.column_dimensions[get_column_letter(c)].width = 22

    _merged_header(ws, 1, 1, _COVER_WIDTH, title, bg=_HDR, size=15)
    r = 3
    if subtitle:
        _merged_header(ws, 2, 1, _COVER_WIDTH, subtitle, bg=_SUB, size=11)
        r = 4

    for entry in rows:
        if entry.is_section:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=_COVER_WIDTH)
            _cell(ws, r, 1, f"\u25b6  {entry.label}", bg=_SUB, bold=True,
                  color=_WHT, align="left", size=11)
        elif entry.label:
            _cell(ws, r, 1, entry.label, bg=_ALT if r % 2 == 0 else _WHT,
                  bold=True, align="left")
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=_COVER_WIDTH)
            _cell(ws, r, 2, entry.description, bg=_WHT, align="left")
        r += 1