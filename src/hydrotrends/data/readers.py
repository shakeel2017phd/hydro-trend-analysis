"""Reading source files into the package's canonical schema.

Each input is described by an :class:`~hydrotrends.core.config.InputSpec` that
carries the file's path, its temporal resolution, and *its own* header names.
:func:`read_input` reads one such file (CSV or ``.xlsx``/``.xlsm`` Excel) and
returns a tidy frame with canonical columns — :data:`~hydrotrends.core.constants.COL_DATE` (datetime) and
:data:`~hydrotrends.core.constants.COL_FLOW_CUSECS` (mean flow, Cusecs) — so
nothing downstream needs to know what the original headers were, and a daily and
a 10-daily file with completely different headers can be loaded side by side.

The reader validates as it goes (missing columns, unparseable dates, duplicate
dates, empty data) and raises the typed errors from
:mod:`hydrotrends.core.exceptions`. It sorts by date and records the resolution
and source path in ``DataFrame.attrs`` so later stages can recover them.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ..core.config import Config, InputSpec
from ..core.constants import COL_DATE, COL_FLOW_CUSECS, MONTH_ABBR_TO_NUM, DateFormat
from ..core.exceptions import ReaderError
from ..core.logging_config import get_logger
from ..core.validation import require_columns, require_min_observations

__all__ = ["read_input", "read_all"]

logger = get_logger(__name__)

_CSV_SUFFIXES = {".csv"}
_EXCEL_SUFFIXES = {".xlsx", ".xlsm"}  # openpyxl engine (a project dependency)


def _read_table(spec: InputSpec) -> pd.DataFrame:
    """Read the raw table (CSV or Excel), dispatching on file suffix.

    Excel reads use the first sheet. I/O and parse failures — plus unsupported
    file types — are translated into :class:`ReaderError`.
    """
    suffix = spec.path.suffix.lower()
    reader: Callable[[], pd.DataFrame]
    if suffix in _CSV_SUFFIXES:
        reader = lambda: pd.read_csv(spec.path)  # noqa: E731
    elif suffix in _EXCEL_SUFFIXES:
        reader = lambda: pd.read_excel(  # noqa: E731
            spec.path, sheet_name=0, engine="openpyxl"
        )
    elif suffix == ".xls":
        raise ReaderError(
            "legacy .xls is not supported; save as .xlsx or .csv",
            path=str(spec.path),
        )
    else:
        raise ReaderError(
            f"unsupported file type {suffix or '(none)'!r}; use .csv, .xlsx or .xlsm",
            path=str(spec.path),
        )

    try:
        return reader()
    except FileNotFoundError as err:
        raise ReaderError("input file not found", path=str(spec.path)) from err
    except pd.errors.EmptyDataError as err:
        raise ReaderError("input file is empty", path=str(spec.path)) from err
    except (pd.errors.ParserError, ValueError, OSError, ImportError) as err:
        raise ReaderError(
            f"could not read {suffix} file: {err}", path=str(spec.path)
        ) from err


_DEKAD_START_DAY = {1: 1, 2: 11, 3: 21}  # dekad digit -> representative (start) day


def _parse_dekad_compact(raw: pd.Series) -> pd.Series:
    """Parse compact 10-daily strings like ``"2020Apr1"`` to real dates.

    Layout: year(4) + month-abbr(3) + dekad digit(1). The dekad digit maps to
    its start day (1/11/21) — a within-dekad, month-length-independent date.
    """
    s = raw.astype(str).str.strip()
    year = s.str[:4].astype(int)
    month = s.str[4:7].map(MONTH_ABBR_TO_NUM)
    dekad = s.str[7:].astype(int)
    if month.isna().any():
        raise ValueError("unrecognised month abbreviation")
    if not dekad.isin([1, 2, 3]).all():
        raise ValueError("dekad digit must be 1, 2 or 3")
    day = dekad.map(_DEKAD_START_DAY)
    return pd.to_datetime(
        {"year": year, "month": month.astype(int), "day": day}
    )


def _parse_dates(raw: pd.Series, spec: InputSpec) -> pd.Series:
    """Parse the source date column according to ``spec.date_format``."""
    fmt = spec.date_format
    try:
        if fmt is DateFormat.DEKAD_COMPACT:
            return _parse_dekad_compact(raw)
        if fmt is DateFormat.ISO:
            return pd.to_datetime(raw, format="ISO8601")
        # AUTO (day-first) or MONTH_FIRST: mixed real-date formats.
        day_first = fmt is not DateFormat.MONTH_FIRST
        return pd.to_datetime(raw, format="mixed", dayfirst=day_first)
    except (ValueError, TypeError, KeyError) as err:
        raise ReaderError(
            f"could not parse dates in column {spec.date_column!r} "
            f"as {fmt.value}",
            path=str(spec.path),
        ) from err


def read_input(spec: InputSpec) -> pd.DataFrame:
    """Read one input file into a canonical, date-sorted frame.

    Returns
    -------
    DataFrame
        Columns ``COL_DATE`` (datetime64) and ``COL_FLOW_CUSECS`` (float),
        sorted ascending by date. ``df.attrs`` carries ``resolution`` and
        ``source_path``.

    Raises
    ------
    ReaderError
        File missing/empty/unreadable, or dates unparseable.
    MissingColumnError
        The declared date/value column is not in the file.
    InsufficientDataError
        No usable (non-NaN) values in the value column.
    NonMonotonicIndexError
        Duplicate dates remain after sorting.
    """
    logger.info("reading %s (%s)", spec.path, spec.resolution.value)
    raw = _read_table(spec)
    require_columns(raw, (spec.date_column, spec.value_column))

    dates = _parse_dates(raw[spec.date_column], spec)
    values = pd.to_numeric(raw[spec.value_column], errors="coerce")

    # Surface silent data problems as warnings rather than swallowing them.
    n_coerced = int((raw[spec.value_column].notna() & values.isna()).sum())
    if n_coerced:
        logger.warning(
            "%d value(s) in column %r were not numeric and became NaN",
            n_coerced,
            spec.value_column,
        )
    if spec.value_scale != 1.0:
        values = values * spec.value_scale
        logger.info("scaled %r by %g", spec.value_column, spec.value_scale)
    n_negative = int((values < 0).sum())
    if n_negative:
        logger.warning(
            "%d negative flow value(s) in %s", n_negative, spec.path.name
        )

    out = pd.DataFrame({COL_DATE: dates, COL_FLOW_CUSECS: values})
    out = out.sort_values(COL_DATE, kind="stable").reset_index(drop=True)

    require_min_observations(
        out[COL_FLOW_CUSECS], 1, operation=f"reading {spec.path.name}"
    )
    # Duplicate dates are a data problem, but reading shouldn't fail on them —
    # data.cleaning resolves them under a policy. Warn so they're not silent.
    n_dup_dates = int(out[COL_DATE].duplicated().sum())
    if n_dup_dates:
        logger.warning(
            "%d duplicate date(s) in %s; resolve with data.cleaning",
            n_dup_dates,
            spec.path.name,
        )

    out.attrs["resolution"] = spec.resolution.value
    out.attrs["source_path"] = str(spec.path)
    logger.info("  -> %d rows, %s to %s", len(out), out[COL_DATE].iloc[0].date(),
                out[COL_DATE].iloc[-1].date())
    return out


def read_all(config: Config) -> list[tuple[InputSpec, pd.DataFrame]]:
    """Read every input declared in ``config`` (daily, 10-daily, or both).

    Returns a list of ``(spec, frame)`` pairs in the order the inputs were
    declared, so the caller keeps each frame's resolution alongside its data.
    """
    return [(spec, read_input(spec)) for spec in config.inputs]