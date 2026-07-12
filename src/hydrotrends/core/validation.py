"""Data-quality checks for :mod:`hydrotrends`.

These are the cross-cutting, *raising* validators — the ones that should abort a
run when the input is unusable (a missing column, an out-of-order date index,
too few observations to analyse at all). They raise the typed errors from
:mod:`hydrotrends.core.exceptions`, so callers get actionable, catchable
failures instead of a downstream ``KeyError`` or a silent ``NaN``.

Two distinct notions of "not enough data" live in the package, and they are
deliberately kept separate:

* **Hard guard (here).** :func:`require_min_observations` *raises* when a series
  is too short to bother with — used at the top of a workflow / reader.
* **Soft guard (in :mod:`hydrotrends.stats`).** Each individual test keeps its
  own intrinsic threshold (modified Mann-Kendall needs 10, ITA needs 4, ...) and
  returns ``NaN`` rather than raising, because those functions run inside loops
  over many short per-period sub-series where skipping is the right behaviour.

The year-completeness predicate mirrors the original script's rule exactly: a
period (hydro or calendar year) is "complete" only if its earliest observation
falls on the expected start date. This is a start-of-period presence check, not
a full 365-day audit — faithful to the source, and documented as such.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

from .exceptions import (
    InsufficientDataError,
    MissingColumnError,
    NonMonotonicIndexError,
    ValidationError,
)

__all__ = [
    "require_columns",
    "count_valid",
    "require_min_observations",
    "require_monotonic_increasing",
    "find_complete_periods",
]

# Type alias for "something numeric we can count / order".
ArrayLike: TypeAlias = pd.Series | np.ndarray | Sequence[float]


def require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
) -> None:
    """Raise :class:`MissingColumnError` if any ``required`` column is absent.

    The error carries both the missing names and the columns that *were* present,
    which is what makes it useful against the source data's inconsistent headers
    (e.g. ``inflow_cusec`` vs ``Inflow_Cumecs``).
    """
    present = set(df.columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise MissingColumnError(missing, available=list(df.columns))


def count_valid(data: ArrayLike) -> int:
    """Return the number of non-null (non-NaN) observations in ``data``."""
    return int(pd.Series(data).count())


def require_min_observations(
    data: ArrayLike,
    required: int,
    *,
    operation: str | None = None,
) -> int:
    """Ensure ``data`` has at least ``required`` valid observations.

    Returns the valid count on success; raises :class:`InsufficientDataError`
    (carrying ``n`` and ``required``) otherwise.
    """
    n = count_valid(data)
    if n < required:
        raise InsufficientDataError(n, required, operation=operation)
    return n


def require_monotonic_increasing(
    index: pd.Index | ArrayLike,
    *,
    allow_duplicates: bool = False,
) -> None:
    """Ensure a (time) index is strictly/weakly increasing.

    Parameters
    ----------
    index:
        The index or ordered values to check (e.g. a ``DatetimeIndex``).
    allow_duplicates:
        If ``False`` (default), equal consecutive values are treated as an error
        as well, i.e. the index must be *strictly* increasing.

    Raises
    ------
    NonMonotonicIndexError
        With ``position`` set to the first offending element.
    """
    idx = pd.Index(index)
    is_ordered = idx.is_monotonic_increasing and (allow_duplicates or idx.is_unique)
    if is_ordered:
        return

    values = idx.to_numpy()
    for i in range(1, len(values)):
        out_of_order = values[i] < values[i - 1]
        duplicate = (not allow_duplicates) and (values[i] == values[i - 1])
        if out_of_order or duplicate:
            raise NonMonotonicIndexError(position=i)
    # Should be unreachable given the checks above, but stay defensive.
    raise NonMonotonicIndexError()


def find_complete_periods(
    df: pd.DataFrame,
    *,
    period_col: str,
    date_col: str,
    start_month: int,
    start_day: int,
) -> tuple[set[int], set[int]]:
    """Partition period labels into complete vs. incomplete.

    A period (e.g. a hydrological or calendar year identified by ``period_col``)
    is *complete* when its earliest ``date_col`` value lands on
    ``start_month``/``start_day``. This reproduces the source script's
    incomplete-year filter; ``preprocessing`` uses it to drop and log partial
    years.

    Returns
    -------
    (complete, incomplete):
        Two sets of period labels. Their union is every label in ``df``.

    Raises
    ------
    MissingColumnError
        If ``period_col`` or ``date_col`` is not present.
    ValidationError
        If ``date_col`` is not datetime-typed.
    """
    require_columns(df, (period_col, date_col))
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValidationError(
            f"column {date_col!r} must be datetime-typed to assess completeness"
        )

    first_dates = df.groupby(period_col)[date_col].min()
    starts_on_boundary = (first_dates.dt.month == start_month) & (
        first_dates.dt.day == start_day
    )
    complete = {int(p) for p in first_dates.index[starts_on_boundary]}
    all_periods = {int(p) for p in df[period_col].unique()}
    incomplete = all_periods - complete
    return complete, incomplete
