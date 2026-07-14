"""Fixed domain constants for :mod:`hydrotrends`.

This module holds *values only* — unit-conversion factors, the water-year and
season definitions, flood thresholds, and the canonical statistical defaults.
Anything that computes or that a user may override at runtime lives elsewhere
(logic in ``stats``/``data``; runtime-tunable settings in ``core.config``).

All values are lifted directly from the original Tarbela daily-inflow analysis
script so results stay bit-for-bit comparable with the pre-refactor output.

Mappings are exposed as read-only (:class:`types.MappingProxyType`) so a caller
can't accidentally mutate shared module state.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

__all__ = [
    "CUSEC_TO_M3S",
    "SEC_PER_DAY",
    "M3_PER_MAF",
    "M3_PER_BCM",
    "FlowUnit",
    "VolumeUnit",
    "UNIT_PAIRS",
    "TimeScale",
    "Season",
    "MetSeason",
    "MET_SEASONS",
    "HYDRO_YEAR_START_MONTH",
    "EARLY_KHARIF_MONTHS",
    "LATE_KHARIF_MONTHS",
    "KHARIF_JUNE_SPLIT_DAY",
    "RABI_MONTHS",
    "DEKAD_UPPER_DAYS",
    "HYDRO_MONTHS",
    "CAL_MONTHS",
    "MET_MONTHS",
    "FULL_MONTHS",
    "MONTH_ABBR_TO_NUM",
    "HYDRO_PERIODS",
    "CAL_PERIODS",
    "MET_PERIODS",
    "HYDRO_DEKADS",
    "CAL_DEKADS",
    "MET_DEKADS",
    "TimeResolution",
    "OnIssue",
    "TrendDirection",
    "FillMethod",
    "DateFormat",
    "ResolutionInfo",
    "RESOLUTION_INFO",
    "COL_DATE",
    "COL_FLOW_CUSECS",
    "COL_FLOW_CUMECS",
    "COL_YEAR",
    "COL_HYDRO_YEAR",
    "COL_MET_YEAR",
    "COL_MONTH_NUM",
    "COL_MONTH",
    "COL_DAY",
    "COL_DEKAD",
    "COL_PERIOD",
    "COL_N_DAYS",
    "COL_VOL_M3",
    "COL_VOL_MAF",
    "COL_VOL_BCM",
    "COL_SEASON",
    "COL_MET_SEASON",
    "FLOOD_CLASSES",
    "FLOOD_LIMITS_1000CUSECS",
    "FLOOD_COLORS",
    "DEFAULT_ALPHA",
    "ITA_SLOPE_EPS",
    "MOVING_AVERAGE_WINDOW",
    "LOWESS_FRAC",
    "REPORTED_PERCENTILES",
    "BAI_PERRON_N_BREAKS",
    "BAI_PERRON_MIN_SIZE",
    "SIGNIFICANCE_STARS",
    "SIGNIFICANCE_NS",
]

# ─────────────────────────────────────────────────────────────────────────────
# Unit conversions
# ─────────────────────────────────────────────────────────────────────────────
# Flow: 1 cubic foot per second -> cubic metres per second.
# Exact by definition: (0.3048 m/ft) ** 3 = 0.028316846592.
CUSEC_TO_M3S: Final = 0.028316846592

SEC_PER_DAY: Final = 86_400

# Volume: cubic metres per unit of the two reporting systems.
#   1 acre-foot = 1233.4818375475238 m3  ->  MAF (million acre-feet) = x 1e6.
M3_PER_MAF: Final = 1.2334818375475238e9
#   BCM (billion cubic metres) = 1e9 m3.
M3_PER_BCM: Final = 1.0e9


class FlowUnit(StrEnum):
    """Instantaneous-flow reporting units."""

    CUSECS = "Cusecs"
    CUMECS = "Cumecs"


class VolumeUnit(StrEnum):
    """Accumulated-volume reporting units."""

    MAF = "MAF"
    BCM = "BCM"


# The two unit systems the original script builds, one workbook each:
#   Cusecs pairs with MAF, Cumecs pairs with BCM.
UNIT_PAIRS: Final[Mapping[FlowUnit, VolumeUnit]] = MappingProxyType(
    {
        FlowUnit.CUSECS: VolumeUnit.MAF,
        FlowUnit.CUMECS: VolumeUnit.BCM,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal aggregation scales
# ─────────────────────────────────────────────────────────────────────────────
class TimeScale(StrEnum):
    """The five aggregation scales analysed across the package."""

    DAILY = "daily"
    TEN_DAILY = "10daily"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    ANNUAL = "annual"


# ─────────────────────────────────────────────────────────────────────────────
# Hydrological (water) year and cropping seasons — Indus Basin convention
# ─────────────────────────────────────────────────────────────────────────────
# Water year runs 1 April -> 31 March. Jan/Feb/Mar belong to the *previous*
# water year (i.e. HydroYear = calendar Year - 1 for those three months).
HYDRO_YEAR_START_MONTH: Final = 4


class Season(StrEnum):
    """Cropping seasons and the two aggregate periods derived from them.

    ``EARLY_KHARIF`` / ``LATE_KHARIF`` / ``RABI`` are assigned per-day; ``KHARIF``
    (early + late) and ``ANNUAL`` are aggregates used only in summary tables.
    """

    EARLY_KHARIF = "Early_Kharif"
    LATE_KHARIF = "Late_Kharif"
    KHARIF = "Kharif"
    RABI = "Rabi"
    ANNUAL = "Annual"


# Season boundaries (month numbers; day rule for the June split).
#   Early Kharif : April, May, and 1-10 June
#   Late  Kharif : 11-30 June, July, August, September
#   Rabi         : October -> March (everything else)
EARLY_KHARIF_MONTHS: Final[tuple[int, ...]] = (4, 5)
LATE_KHARIF_MONTHS: Final[tuple[int, ...]] = (7, 8, 9)
KHARIF_JUNE_SPLIT_DAY: Final = 10
RABI_MONTHS: Final[tuple[int, ...]] = (10, 11, 12, 1, 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Meteorological seasons — an alternative, calendar-based grouping
# ─────────────────────────────────────────────────────────────────────────────
# Independent of the Kharif/Rabi cropping seasons above; useful for climatic
# summaries. Note Winter spans a calendar-year boundary (Dec + Jan + Feb).
class MetSeason(StrEnum):
    """Calendar-based meteorological seasons for the Indus region."""

    WINTER = "Winter"
    SPRING = "Spring"
    SUMMER = "Summer"
    MONSOON = "Monsoon"
    AUTUMN = "Autumn"


MET_SEASONS: Final[Mapping[MetSeason, tuple[int, ...]]] = MappingProxyType(
    {
        MetSeason.WINTER: (12, 1, 2),
        MetSeason.SPRING: (3, 4),
        MetSeason.SUMMER: (5, 6),
        MetSeason.MONSOON: (7, 8, 9),
        MetSeason.AUTUMN: (10, 11),
    }
)

# Dekad (10-daily) boundaries: days 1-10 -> 1, 11-20 -> 2, 21-end -> 3.
DEKAD_UPPER_DAYS: Final[tuple[int, int]] = (10, 20)

# ─────────────────────────────────────────────────────────────────────────────
# Month label orderings
# ─────────────────────────────────────────────────────────────────────────────
HYDRO_MONTHS: Final[tuple[str, ...]] = (
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
)
CAL_MONTHS: Final[tuple[str, ...]] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
# Meteorological year: Dec 1 -> Nov 30 (see COL_MET_YEAR).
MET_MONTHS: Final[tuple[str, ...]] = (
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
)
FULL_MONTHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Jan": "January",
        "Feb": "February",
        "Mar": "March",
        "Apr": "April",
        "May": "May",
        "Jun": "June",
        "Jul": "July",
        "Aug": "August",
        "Sep": "September",
        "Oct": "October",
        "Nov": "November",
        "Dec": "December",
    }
)

# Month abbreviation -> number (1-12); reverse of CAL_MONTHS.
MONTH_ABBR_TO_NUM: Final[Mapping[str, int]] = MappingProxyType(
    {abbr: num for num, abbr in enumerate(CAL_MONTHS, start=1)}
)

# ─────────────────────────────────────────────────────────────────────────────
# Day-of-year ("Period") and dekad label orderings
# ─────────────────────────────────────────────────────────────────────────────
# Ordered categorical keys the daily and 10-daily scales sort by. Periods are
# "%b-%d" labels ("Apr-01" ... "Mar-31"); a leap year is used as the reference
# so 29 February ("Feb-29") is present and leap-year data orders correctly.
_HYDRO_MONTH_NUMS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3)
_CAL_MONTH_NUMS: Final[tuple[int, ...]] = tuple(range(1, 13))
_MET_MONTH_NUMS: Final[tuple[int, ...]] = (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_LEAP_REF_YEAR: Final = 2020  # any leap year; only used for per-month day counts


def _period_labels(month_numbers: tuple[int, ...]) -> tuple[str, ...]:
    """Build ordered '%b-%d' labels for a month ordering (leap day included)."""
    labels: list[str] = []
    for month in month_numbers:
        n_days = calendar.monthrange(_LEAP_REF_YEAR, month)[1]
        abbr = CAL_MONTHS[month - 1]  # canonical English abbr (locale-independent)
        labels.extend(f"{abbr}-{day:02d}" for day in range(1, n_days + 1))
    return tuple(labels)


HYDRO_PERIODS: Final[tuple[str, ...]] = _period_labels(_HYDRO_MONTH_NUMS)
CAL_PERIODS: Final[tuple[str, ...]] = _period_labels(_CAL_MONTH_NUMS)
MET_PERIODS: Final[tuple[str, ...]] = _period_labels(_MET_MONTH_NUMS)

# Dekad labels: month abbr + dekad number, e.g. "Apr1", "Apr2", "Apr3".
HYDRO_DEKADS: Final[tuple[str, ...]] = tuple(
    f"{m}{d}" for m in HYDRO_MONTHS for d in (1, 2, 3)
)
CAL_DEKADS: Final[tuple[str, ...]] = tuple(
    f"{m}{d}" for m in CAL_MONTHS for d in (1, 2, 3)
)
MET_DEKADS: Final[tuple[str, ...]] = tuple(
    f"{m}{d}" for m in MET_MONTHS for d in (1, 2, 3)
)


# ─────────────────────────────────────────────────────────────────────────────
# Input temporal resolution
# ─────────────────────────────────────────────────────────────────────────────
# The resolution of a *source file* (as opposed to TimeScale, which is an
# analysis aggregation level). A run may process a daily file, a 10-daily file,
# or both at once; each input carries its resolution so readers/preprocessing
# pick the right period-label set and volume day-count without a hardcoded
# daily assumption.
class TimeResolution(StrEnum):
    """Temporal resolution of an input series."""

    DAILY = "daily"
    TEN_DAILY = "10daily"


class OnIssue(StrEnum):
    """Policy for how a cleaning step handles a detected data-quality issue."""

    KEEP = "keep"  # leave rows in place, just report
    DROP = "drop"  # remove the offending rows
    ERROR = "error"  # raise instead of proceeding


class TrendDirection(StrEnum):
    """Direction reported by a trend test."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    NO_TREND = "no trend"


class FillMethod(StrEnum):
    """How to impute missing values before the missing-row policy is applied."""

    NONE = "none"  # no filling (missing rows handled by the OnIssue policy)
    LINEAR = "linear"  # linear interpolation across the gap
    FFILL = "ffill"  # carry the last valid value forward
    BFILL = "bfill"  # carry the next valid value backward


class DateFormat(StrEnum):
    """How a source file encodes its date column.

    ``AUTO``/``MONTH_FIRST``/``ISO`` are for real dates (any resolution);
    ``DEKAD_COMPACT`` is the 10-daily string encoding.
    """

    AUTO = "auto"  # mixed real dates, day-first (e.g. 31/12/2020)
    MONTH_FIRST = "month_first"  # mixed real dates, US month-first (12/31/2020)
    ISO = "iso"  # ISO-8601 (2020-12-31), unambiguous
    DEKAD_COMPACT = "dekad_compact"  # "2020Apr1" = year + month-abbr + dekad digit


class ResolutionInfo(NamedTuple):
    """Per-resolution metadata: which scale, ordered period labels, and size."""

    scale: TimeScale
    hydro_periods: tuple[str, ...]
    cal_periods: tuple[str, ...]
    met_periods: tuple[str, ...]
    n_period_labels: int  # size of the ordered category (366 daily / 36 dekad)
    is_pre_aggregated: bool  # True for 10-daily (values are dekad averages)


RESOLUTION_INFO: Final[Mapping[TimeResolution, ResolutionInfo]] = MappingProxyType(
    {
        TimeResolution.DAILY: ResolutionInfo(
            scale=TimeScale.DAILY,
            hydro_periods=HYDRO_PERIODS,
            cal_periods=CAL_PERIODS,
            met_periods=MET_PERIODS,
            n_period_labels=len(CAL_PERIODS),
            is_pre_aggregated=False,
        ),
        TimeResolution.TEN_DAILY: ResolutionInfo(
            scale=TimeScale.TEN_DAILY,
            hydro_periods=HYDRO_DEKADS,
            cal_periods=CAL_DEKADS,
            met_periods=MET_DEKADS,
            n_period_labels=len(CAL_DEKADS),
            is_pre_aggregated=True,
        ),
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical column names
# ─────────────────────────────────────────────────────────────────────────────
# Readers normalise each source file's own headers to these internal names, so
# everything downstream (preprocessing, stats, viz) is header-agnostic. The raw
# input is a mean flow in Cusecs (daily value, or dekad average for 10-daily).
COL_DATE: Final = "Date"
COL_FLOW_CUSECS: Final = "Inflow_Cusecs"
COL_FLOW_CUMECS: Final = "Inflow_Cumecs"
COL_YEAR: Final = "Year"
COL_HYDRO_YEAR: Final = "HydroYear"
COL_MET_YEAR: Final = "MetYear"  # meteorological year (Dec 1 -> Nov 30)
COL_MONTH_NUM: Final = "MonthNum"
COL_MONTH: Final = "Month"
COL_DAY: Final = "Day"
COL_DEKAD: Final = "Dekad"
COL_PERIOD: Final = "Period"  # resolution-dependent label ("Apr-01" or "Apr1")
COL_N_DAYS: Final = "NDays"  # days the row represents (1 daily; dekad length)
COL_VOL_M3: Final = "Vol_m3"
COL_VOL_MAF: Final = "Vol_MAF"
COL_VOL_BCM: Final = "Vol_BCM"
COL_SEASON: Final = "Season"  # cropping season (Kharif/Rabi)
COL_MET_SEASON: Final = "MetSeason"  # meteorological season

# ─────────────────────────────────────────────────────────────────────────────
# Flood limits — canonical Indus Basin thresholds, in units of 1000 Cusecs
# ─────────────────────────────────────────────────────────────────────────────
#   LF = Low, MF = Medium, HF = High, VHF = Very High, EHF = Exceptionally High
# Listed in ascending severity; convert to an active flow unit downstream
# (Cusecs: x 1000; Cumecs: x 1000 x CUSEC_TO_M3S).
FLOOD_CLASSES: Final[tuple[str, ...]] = ("LF", "MF", "HF", "VHF", "EHF")

FLOOD_LIMITS_1000CUSECS: Final[Mapping[str, int]] = MappingProxyType(
    {"LF": 250, "MF": 375, "HF": 500, "VHF": 650, "EHF": 800}
)

FLOOD_COLORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LF": "#90EE90",  # light green
        "MF": "#ADD8E6",  # light blue
        "HF": "#FFA07A",  # light salmon
        "VHF": "#FF0000",  # red
        "EHF": "#8B0000",  # dark red
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical defaults
# ─────────────────────────────────────────────────────────────────────────────
# These are the *default* values; core.config may override the tunable ones
# (e.g. alpha) at runtime.
DEFAULT_ALPHA: Final = 0.05  # significance level for MK / Pettitt
ITA_SLOPE_EPS: Final = 1e-6  # |slope| below this -> "no trend" in ITA
MOVING_AVERAGE_WINDOW: Final = 5  # MA(5) smoothing window (min_periods = 5)
LOWESS_FRAC: Final = 0.3  # smoothing span for LOWESS overlays
REPORTED_PERCENTILES: Final[tuple[int, ...]] = (10, 25, 75, 90)

# Bai-Perron structural-break search.
BAI_PERRON_N_BREAKS: Final = 1
BAI_PERRON_MIN_SIZE: Final = 5

# p-value -> significance symbol. Checked in ascending order; first match wins,
# with SIGNIFICANCE_NS as the fallback when p >= the largest threshold.
SIGNIFICANCE_STARS: Final[tuple[tuple[float, str], ...]] = (
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
    (0.10, "."),
)
SIGNIFICANCE_NS: Final = "ns"
