"""Exception hierarchy for :mod:`hydrotrends`.

Every error the package raises deliberately derives from
:class:`HydroTrendsError`, so downstream code can catch *all* library-specific
failures with a single ``except HydroTrendsError`` while still being able to
target narrower cases (a missing column, too few data points, a bad config)
when it wants to.

Hierarchy::

    HydroTrendsError
    ├── ConfigError
    └── DataError
        ├── ReaderError
        ├── MissingColumnError
        └── ValidationError
            ├── InsufficientDataError
            └── NonMonotonicIndexError

The richer subclasses store the offending values as attributes (not just inside
the message string) so callers can inspect them programmatically, e.g.::

    try:
        stats.mann_kendall(series)
    except InsufficientDataError as err:
        logger.warning("skipping %s: need %d, got %d", name, err.required, err.n)
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "HydroTrendsError",
    "ConfigError",
    "DataError",
    "ReaderError",
    "MissingColumnError",
    "ValidationError",
    "InsufficientDataError",
    "NonMonotonicIndexError",
]


class HydroTrendsError(Exception):
    """Base class for every error raised by :mod:`hydrotrends`."""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class ConfigError(HydroTrendsError):
    """Raised for invalid, missing, or contradictory configuration values."""


# ─────────────────────────────────────────────────────────────────────────────
# Data: reading, shape, and validation
# ─────────────────────────────────────────────────────────────────────────────
class DataError(HydroTrendsError):
    """Base class for problems with the input data (I/O, schema, contents)."""


class ReaderError(DataError):
    """Raised when a source file cannot be located, opened, or parsed.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    path:
        The file path involved, if known. Stored on the instance as ``path``.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        if path is not None:
            message = f"{message} (path: {path})"
        super().__init__(message)


class MissingColumnError(DataError):
    """Raised when required column(s) are absent from the input table.

    Attributes
    ----------
    missing:
        The columns that were required but not found.
    available:
        The columns that *were* present, to make the message actionable.
    """

    def __init__(
        self,
        missing: Iterable[str],
        *,
        available: Iterable[str] | None = None,
    ) -> None:
        self.missing: tuple[str, ...] = tuple(missing)
        self.available: tuple[str, ...] = tuple(available) if available is not None else ()
        message = f"missing required column(s): {', '.join(self.missing) or '<none>'}"
        if self.available:
            message += f"; available columns: {', '.join(self.available)}"
        super().__init__(message)


class ValidationError(DataError):
    """Base class for values that are the right shape but semantically invalid."""


class InsufficientDataError(ValidationError):
    """Raised when a computation needs more observations than were supplied.

    Attributes
    ----------
    n:
        The number of (valid) observations actually available.
    required:
        The minimum number the operation needs.
    """

    def __init__(
        self,
        n: int,
        required: int,
        *,
        operation: str | None = None,
    ) -> None:
        self.n = n
        self.required = required
        self.operation = operation
        what = f"{operation} " if operation else ""
        super().__init__(
            f"{what}needs at least {required} observation(s), got {n}"
        )


class NonMonotonicIndexError(ValidationError):
    """Raised when a time index is expected to be sorted/increasing but is not.

    Attributes
    ----------
    position:
        Index position of the first out-of-order (or duplicate) entry, if known.
    """

    def __init__(
        self,
        message: str = "time index is not monotonically increasing",
        *,
        position: int | None = None,
    ) -> None:
        self.position = position
        if position is not None:
            message = f"{message} (first offending position: {position})"
        super().__init__(message)