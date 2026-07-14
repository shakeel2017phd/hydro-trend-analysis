"""Small shared helpers for :mod:`hydrotrends`.

Used across ``stats`` and ``viz``: the p-value -> significance-symbol formatter,
the flow/volume unit converters, and safe scalar rounding. Everything here is a
pure function with no side effects.

Resolution-aware volume
-----------------------
Volume from a *mean* flow depends on how many days the observation represents:

    volume = mean_flow[m3/s] * n_days * 86_400[s/day]

For a **daily** series each row spans one day (``n_days = 1``). For a
**10-daily** series each value is the *average* inflow over a dekad, so
``n_days`` is that dekad's day-count (10, 10, or 8-11 for the final dekad) —
see :func:`days_in_dekad`. Passing ``n_days`` explicitly keeps a single code
path correct for both resolutions instead of hardcoding a daily assumption.
"""

from __future__ import annotations

import calendar
import math
from typing import TypeAlias

import numpy as np
import pandas as pd

from .constants import (
    CUSEC_TO_M3S,
    DEKAD_UPPER_DAYS,
    M3_PER_BCM,
    M3_PER_MAF,
    SEC_PER_DAY,
    SIGNIFICANCE_NS,
    SIGNIFICANCE_STARS,
    VolumeUnit,
)
from .validation import ArrayLike

__all__ = [
    "significance_stars",
    "days_in_dekad",
    "cusecs_to_cumecs",
    "mean_flow_to_volume_m3",
    "m3_to_volume",
    "mean_flow_to_volume",
    "safe_round",
    "to_float_array",
    "hydro_year_label",
    "met_year_label",
]

# Anything the vectorisable converters accept: a scalar or an array/Series.
Numeric: TypeAlias = float | np.ndarray | pd.Series


# ─────────────────────────────────────────────────────────────────────────────
# Significance formatting
# ─────────────────────────────────────────────────────────────────────────────
def significance_stars(p_value: float | None) -> str:
    """Map a p-value to a significance symbol (``***``/``**``/``*``/``.``/``ns``).

    ``None`` or NaN yields ``ns``. Thresholds come from
    :data:`~hydrotrends.core.constants.SIGNIFICANCE_STARS`; the first threshold
    the p-value falls below wins.
    """
    if p_value is None:
        return SIGNIFICANCE_NS
    try:
        if math.isnan(p_value):
            return SIGNIFICANCE_NS
    except (TypeError, ValueError):
        return SIGNIFICANCE_NS
    for threshold, symbol in SIGNIFICANCE_STARS:
        if p_value < threshold:
            return symbol
    return SIGNIFICANCE_NS


# ─────────────────────────────────────────────────────────────────────────────
# Calendar helper for dekad volumes
# ─────────────────────────────────────────────────────────────────────────────
def days_in_dekad(year: int, month: int, dekad: int) -> int:
    """Number of days in a given dekad.

    Dekads 1 and 2 are fixed-length (days 1-10 and 11-20); dekad 3 runs from day
    21 to month end, so its length depends on the month (and on leap years for
    February). Boundaries come from
    :data:`~hydrotrends.core.constants.DEKAD_UPPER_DAYS`.
    """
    if dekad not in (1, 2, 3):
        raise ValueError(f"dekad must be 1, 2 or 3, got {dekad}")
    upper1, upper2 = DEKAD_UPPER_DAYS  # (10, 20)
    if dekad == 1:
        return upper1
    if dekad == 2:
        return upper2 - upper1
    return calendar.monthrange(year, month)[1] - upper2


# ─────────────────────────────────────────────────────────────────────────────
# Flow / volume unit conversions
# ─────────────────────────────────────────────────────────────────────────────
def cusecs_to_cumecs(flow_cusecs: Numeric) -> Numeric:
    """Convert flow from Cusecs (ft3/s) to Cumecs (m3/s). Scalar or array."""
    result: Numeric = flow_cusecs * CUSEC_TO_M3S
    return result


def mean_flow_to_volume_m3(mean_flow_cusecs: Numeric, n_days: Numeric) -> Numeric:
    """Volume in cubic metres from a mean flow (Cusecs) over ``n_days`` days.

    ``n_days`` is 1 for a daily observation and the dekad's day-count for a
    10-daily average (see :func:`days_in_dekad`). Scalar or array.
    """
    result: Numeric = mean_flow_cusecs * CUSEC_TO_M3S * SEC_PER_DAY * n_days
    return result


def m3_to_volume(volume_m3: Numeric, unit: VolumeUnit | str) -> Numeric:
    """Convert a volume in cubic metres to MAF or BCM. Scalar or array."""
    resolved = VolumeUnit(unit)
    if resolved is VolumeUnit.MAF:
        result: Numeric = volume_m3 / M3_PER_MAF
    else:  # VolumeUnit.BCM
        result = volume_m3 / M3_PER_BCM
    return result


def mean_flow_to_volume(
    mean_flow_cusecs: Numeric,
    n_days: Numeric,
    unit: VolumeUnit | str,
) -> Numeric:
    """Mean flow (Cusecs) over ``n_days`` -> volume in the requested unit.

    Convenience wrapper composing :func:`mean_flow_to_volume_m3` and
    :func:`m3_to_volume`.
    """
    return m3_to_volume(mean_flow_to_volume_m3(mean_flow_cusecs, n_days), unit)


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────
def safe_round(value: float | None, ndigits: int = 4) -> float:
    """Round a scalar, passing ``None``/NaN through as NaN instead of erroring."""
    if value is None:
        return math.nan
    try:
        if math.isnan(value):
            return math.nan
    except (TypeError, ValueError):
        pass
    return round(float(value), ndigits)


def to_float_array(data: ArrayLike) -> np.ndarray:
    """Coerce ``data`` to a 1-D float array, dropping NaNs."""
    arr: np.ndarray = (
        pd.to_numeric(pd.Series(data), errors="coerce")
        .dropna()
        .to_numpy(dtype="float64")
    )
    return arr


def hydro_year_label(hydro_year: int) -> str:
    """YYYY-YY label for a hydrological year (Apr -> Mar), e.g. 2020 -> 2020-21."""
    return f"{hydro_year}-{str(hydro_year + 1)[2:]}"


def met_year_label(met_year: int) -> str:
    """YYYY-YY label for a meteorological year (Dec -> Nov).

    A met year is *labelled* by the year its Jan-Nov fall in (see
    :func:`~hydrotrends.data.preprocessing.enrich`), so it *starts* the
    previous calendar year's December, e.g. met year 2020 -> 2019-20.
    """
    return f"{met_year - 1}-{str(met_year)[2:]}"
