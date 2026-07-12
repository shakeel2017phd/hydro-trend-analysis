"""Logging setup for :mod:`hydrotrends`.

This replaces the ad-hoc ``print(...)`` status lines (and the blanket
``warnings.filterwarnings("ignore")``) of the original script with the standard
:mod:`logging` machinery. Severity is carried by the log level instead of emoji
markers::

    print("Step 1/4 — ...")          -> logger.info("...")
    print("  [!] Dropped ...")       -> logger.warning("...")
    print("  ⚠️  WARNING: ...")       -> logger.warning("...")
    print("  ❌  Error ...")          -> logger.error("...")
    print("  ✅  Saved -> ...")       -> logger.info("saved -> %s", path)

Library-friendly by design
--------------------------
Importing :mod:`hydrotrends` attaches only a :class:`logging.NullHandler`, so
the library stays **silent** until the embedding application configures logging
itself — a package must never configure the root logger on import.

Applications, the CLI, and example scripts can opt in to formatted console
output with a single call::

    from hydrotrends.core.logging_config import configure_logging
    configure_logging("INFO")        # or logging.INFO

Use :func:`get_logger` everywhere inside the package to obtain a correctly
namespaced logger (``hydrotrends.<module>``).
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

__all__ = ["LOGGER_NAME", "get_logger", "configure_logging"]

#: Root logger name for the whole package.
LOGGER_NAME = "hydrotrends"

#: Handler names, so (re)configuration is idempotent instead of stacking up.
_CONSOLE_HANDLER_NAME = "hydrotrends-console"
_WARNINGS_LOGGER = "py.warnings"

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

# A NullHandler on the package logger prevents "No handlers could be found"
# noise and keeps the library quiet by default. Installed once, at import time.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under :data:`LOGGER_NAME`.

    Parameters
    ----------
    name:
        A module-local name, typically ``__name__``. The result is placed under
        the package logger, e.g. ``get_logger("hydrotrends.stats.trends")`` and
        ``get_logger("stats.trends")`` both return the
        ``hydrotrends.stats.trends`` logger. ``None`` returns the package root
        logger.
    """
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _coerce_level(level: int | str) -> int:
    """Accept either a numeric level or its name ('INFO', 'debug', ...)."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def _ensure_console_handler(
    logger: logging.Logger,
    *,
    level: int,
    formatter: logging.Formatter,
    stream: TextIO,
) -> None:
    """Attach exactly one named console handler to ``logger`` (idempotent)."""
    for existing in list(logger.handlers):
        if getattr(existing, "name", None) == _CONSOLE_HANDLER_NAME:
            logger.removeHandler(existing)
    handler = logging.StreamHandler(stream)
    handler.name = _CONSOLE_HANDLER_NAME
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Opt-in console logging for the package. Returns the package logger.

    Safe to call more than once: it replaces its own handler rather than adding
    duplicates, so repeated calls (e.g. from tests) won't multiply output.

    Parameters
    ----------
    level:
        Threshold as an int (``logging.INFO``) or name (``"INFO"``).
    stream:
        Target stream; defaults to :data:`sys.stderr` so log output stays off
        stdout and won't corrupt piped data.
    fmt, datefmt:
        Override the default log-line and timestamp formats.
    capture_warnings:
        When true, route :mod:`warnings` through logging (as ``py.warnings``)
        and surface them on the same console — a targeted replacement for the
        original blanket ``filterwarnings("ignore")``.
    """
    resolved_level = _coerce_level(level)
    target_stream: TextIO = stream if stream is not None else sys.stderr
    formatter = logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt or _DEFAULT_DATEFMT)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved_level)
    _ensure_console_handler(
        logger, level=resolved_level, formatter=formatter, stream=target_stream
    )
    # Don't double-emit through an application's root handler.
    logger.propagate = False

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger(_WARNINGS_LOGGER)
        warnings_logger.setLevel(resolved_level)
        _ensure_console_handler(
            warnings_logger,
            level=resolved_level,
            formatter=formatter,
            stream=target_stream,
        )
        warnings_logger.propagate = False

    return logger
