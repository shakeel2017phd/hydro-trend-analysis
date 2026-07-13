"""Row-level preprocessing for :mod:`hydrotrends`.

Takes a canonical frame from :mod:`hydrotrends.data.readers` (columns
``Date`` + ``Inflow_Cusecs``) plus its
:class:`~hydrotrends.core.constants.TimeResolution`
and produces the enriched, quality-filtered frames the analysis runs on.

What it adds
------------
* Calendar parts: ``Year``, ``MonthNum``, ``Month``, ``Day``, ``Dekad``.
* ``HydroYear`` — water year (Jan-Mar belong to the previous year).
* ``Period`` — the ordered category key, **resolution-dependent**: a day label
  (``"Apr-01"``) for daily input, a dekad label (``"Apr1"``) for 10-daily.
* ``NDays`` — days the row represents: ``1`` for daily, the dekad's length
  (10/10/8-11) for a 10-daily *average*. This is what makes volumes correct for
  both resolutions.
* ``Inflow_Cumecs`` and volumes ``Vol_m3`` / ``Vol_MAF`` / ``Vol_BCM``.
* **Both** season columns: ``Season`` (Kharif/Rabi, with the day-level June-10
  split) and ``MetSeason`` (calendar meteorological seasons via ``MET_SEASONS``).

It then splits into two quality-filtered frames — one keyed on complete
hydrological years, one on complete calendar years — dropping (and logging) any
partial years at the start/end of the record, resolution-aware: a year is
complete when its earliest row is the first sub-period of the start month
(day 1 for daily, dekad 1 for 10-daily).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import utils
from ..core.config import InputSpec
from ..core.constants import (
    COL_DATE,
    COL_DAY,
    COL_DEKAD,
    COL_FLOW_CUMECS,
    COL_FLOW_CUSECS,
    COL_HYDRO_YEAR,
    COL_MET_SEASON,
    COL_MONTH,
    COL_MONTH_NUM,
    COL_N_DAYS,
    COL_PERIOD,
    COL_SEASON,
    COL_VOL_BCM,
    COL_VOL_M3,
    COL_VOL_MAF,
    COL_YEAR,
    DEKAD_UPPER_DAYS,
    EARLY_KHARIF_MONTHS,
    HYDRO_YEAR_START_MONTH,
    KHARIF_JUNE_SPLIT_DAY,
    LATE_KHARIF_MONTHS,
    MET_SEASONS,
    MetSeason,
    Season,
    TimeResolution,
    VolumeUnit,
)
from ..core.logging_config import get_logger
from ..core.validation import find_complete_periods

__all__ = [
    "PreprocessedData",
    "enrich",
    "preprocess",
    "preprocess_all",
    "monthly_volumes",
    "hydro_seasonal_volumes",
    "met_seasonal_volumes",
]

logger = get_logger(__name__)

_CALENDAR_YEAR_START_MONTH = 1
_JUNE = 6  # the month the Kharif season splits within
_DECEMBER = 12  # the month Winter's met-year label rolls forward at

# Month number -> meteorological season label (inverted from MET_SEASONS).
_MONTH_TO_MET_SEASON: dict[int, str] = {
    month: season.value for season, months in MET_SEASONS.items() for month in months
}

# Which per-day (cropping) Season values roll up into each named hydrological
# seasonal/annual total. Distinct from the *meteorological* seasons
# (``MetSeason``) that ``met_seasonal_volumes`` aggregates below — the source
# script and this package's day-level ``Season`` column only ever mean the
# Kharif/Rabi cropping calendar, so the aggregation and analysis functions for
# it are named ``hydro_season*`` to keep that unambiguous.
# ``Kharif`` is Early + Late combined; ``Annual`` is every season (the source's
# ``season_dfs`` builds these from ``df_hy`` the same way: an unfiltered
# ``groupby("HydroYear")`` for Annual, a Season-filtered one for the rest).
_HYDRO_SEASON_MEMBERSHIP: dict[str, tuple[str, ...]] = {
    Season.EARLY_KHARIF.value: (Season.EARLY_KHARIF.value,),
    Season.LATE_KHARIF.value: (Season.LATE_KHARIF.value,),
    Season.KHARIF.value: (Season.EARLY_KHARIF.value, Season.LATE_KHARIF.value),
    Season.RABI.value: (Season.RABI.value,),
    Season.ANNUAL.value: (
        Season.EARLY_KHARIF.value,
        Season.LATE_KHARIF.value,
        Season.RABI.value,
    ),
}

# Meteorological seasons in calendar order (Winter first, since it starts the
# meteorological year at December).
_MET_SEASON_ORDER: tuple[str, ...] = (
    MetSeason.WINTER.value,
    MetSeason.SPRING.value,
    MetSeason.SUMMER.value,
    MetSeason.MONSOON.value,
    MetSeason.AUTUMN.value,
)


@dataclass(frozen=True)
class PreprocessedData:
    """Enriched, quality-filtered frames for one input at one resolution."""

    resolution: TimeResolution
    hydro: pd.DataFrame  # rows within complete hydrological years
    calendar: pd.DataFrame  # rows within complete calendar years


def _cropping_season(df: pd.DataFrame) -> np.ndarray:
    """Assign Kharif/Rabi seasons with the day-level June-10 split (v26 rule)."""
    is_early = df[COL_MONTH_NUM].isin(EARLY_KHARIF_MONTHS) | (
        (df[COL_MONTH_NUM] == _JUNE) & (df[COL_DAY] <= KHARIF_JUNE_SPLIT_DAY)
    )
    is_late = (
        (df[COL_MONTH_NUM] == _JUNE) & (df[COL_DAY] > KHARIF_JUNE_SPLIT_DAY)
    ) | df[COL_MONTH_NUM].isin(LATE_KHARIF_MONTHS)
    result: np.ndarray = np.select(
        [is_early, is_late],
        [Season.EARLY_KHARIF.value, Season.LATE_KHARIF.value],
        default=Season.RABI.value,
    )
    return result


def enrich(df: pd.DataFrame, resolution: TimeResolution) -> pd.DataFrame:
    """Add all derived columns (calendar parts, volumes, both season schemes)."""
    out = df.copy()
    dt = out[COL_DATE].dt

    out[COL_YEAR] = dt.year
    out[COL_MONTH_NUM] = dt.month
    out[COL_MONTH] = dt.strftime("%b")
    out[COL_DAY] = dt.day

    upper1, upper2 = DEKAD_UPPER_DAYS  # (10, 20)
    out[COL_DEKAD] = np.where(
        out[COL_DAY] <= upper1, 1, np.where(out[COL_DAY] <= upper2, 2, 3)
    )
    out[COL_HYDRO_YEAR] = np.where(
        out[COL_MONTH_NUM].isin([1, 2, 3]), out[COL_YEAR] - 1, out[COL_YEAR]
    )

    if resolution is TimeResolution.DAILY:
        out[COL_PERIOD] = dt.strftime("%b-%d")
        out[COL_N_DAYS] = 1
    else:  # TEN_DAILY: dekad label, and dekad-length day count
        out[COL_PERIOD] = out[COL_MONTH] + out[COL_DEKAD].astype(str)
        days_in_month = dt.days_in_month
        out[COL_N_DAYS] = np.select(
            [out[COL_DEKAD] == 1, out[COL_DEKAD] == 2],
            [upper1, upper2 - upper1],
            default=days_in_month - upper2,
        )

    # Flow in Cumecs and resolution-correct volumes (mean flow x n_days).
    out[COL_FLOW_CUMECS] = utils.cusecs_to_cumecs(out[COL_FLOW_CUSECS])
    out[COL_VOL_M3] = utils.mean_flow_to_volume_m3(
        out[COL_FLOW_CUSECS], out[COL_N_DAYS]
    )
    out[COL_VOL_MAF] = utils.m3_to_volume(out[COL_VOL_M3], VolumeUnit.MAF)
    out[COL_VOL_BCM] = utils.m3_to_volume(out[COL_VOL_M3], VolumeUnit.BCM)

    # Both season classifications, always.
    out[COL_SEASON] = _cropping_season(out)
    out[COL_MET_SEASON] = out[COL_MONTH_NUM].map(_MONTH_TO_MET_SEASON)

    out.attrs["resolution"] = resolution.value
    return out


def _filter_complete_years(
    df: pd.DataFrame,
    *,
    period_col: str,
    start_month: int,
    resolution: TimeResolution,
    kind: str,
) -> pd.DataFrame:
    """Keep only years whose record starts at the first sub-period (logged)."""
    if resolution is TimeResolution.DAILY:
        complete, incomplete = find_complete_periods(
            df,
            period_col=period_col,
            date_col=COL_DATE,
            start_month=start_month,
            start_day=1,
        )
    else:  # TEN_DAILY: earliest row of the year must be dekad 1 of start_month
        first_rows = df.loc[df.groupby(period_col)[COL_DATE].idxmin()]
        starts_ok = (first_rows[COL_MONTH_NUM] == start_month) & (
            first_rows[COL_DEKAD] == 1
        )
        complete = {int(p) for p in first_rows[period_col][starts_ok]}
        incomplete = {int(p) for p in df[period_col].unique()} - complete

    if incomplete:
        logger.warning("dropped incomplete %s year(s): %s", kind, sorted(incomplete))
    kept: pd.DataFrame = df[df[period_col].isin(complete)].copy()
    return kept


def preprocess(df: pd.DataFrame, resolution: TimeResolution) -> PreprocessedData:
    """Enrich a canonical frame and split it into complete hydro/calendar years."""
    enriched = enrich(df, resolution)
    hydro = _filter_complete_years(
        enriched,
        period_col=COL_HYDRO_YEAR,
        start_month=HYDRO_YEAR_START_MONTH,
        resolution=resolution,
        kind="hydro",
    )
    calendar = _filter_complete_years(
        enriched,
        period_col=COL_YEAR,
        start_month=_CALENDAR_YEAR_START_MONTH,
        resolution=resolution,
        kind="calendar",
    )
    logger.info(
        "preprocessed %s: %d hydro-year rows, %d calendar-year rows",
        resolution.value,
        len(hydro),
        len(calendar),
    )
    return PreprocessedData(resolution=resolution, hydro=hydro, calendar=calendar)


def preprocess_all(
    loaded: list[tuple[InputSpec, pd.DataFrame]],
) -> list[tuple[InputSpec, PreprocessedData]]:
    """Preprocess every ``(spec, frame)`` pair from :func:`readers.read_all`."""
    return [(spec, preprocess(df, spec.resolution)) for spec, df in loaded]


def monthly_volumes(hydro: pd.DataFrame) -> pd.DataFrame:
    """Per-hydrological-year, per-calendar-month total volumes.

    One row per ``(HydroYear, Month)``, summing ``Vol_MAF``/``Vol_BCM`` across
    every row that falls in it. Each row already carries the correct
    day-count-scaled volume via :func:`enrich`, so this is correct for daily and
    10-daily input alike. Mirrors the source's ``monthly_vol`` grouping.
    """
    return (
        hydro.groupby([COL_HYDRO_YEAR, COL_MONTH, COL_MONTH_NUM])[
            [COL_VOL_MAF, COL_VOL_BCM]
        ]
        .sum()
        .reset_index()
        .sort_values([COL_HYDRO_YEAR, COL_MONTH_NUM])
        .reset_index(drop=True)
    )


def hydro_seasonal_volumes(hydro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-hydrological-year cropping-seasonal and annual total volumes.

    Keyed ``Early_Kharif``/``Late_Kharif``/``Kharif``/``Rabi``/``Annual``
    (mirrors the source's ``season_dfs``); each value is a frame indexed by
    ``HydroYear`` with summed ``Vol_MAF``/``Vol_BCM``. ``Kharif`` is Early +
    Late Kharif combined; ``Annual`` is every season combined. For the
    calendar-based meteorological seasons instead, see
    :func:`met_seasonal_volumes`.
    """
    return {
        name: hydro[hydro[COL_SEASON].isin(members)]
        .groupby(COL_HYDRO_YEAR)[[COL_VOL_MAF, COL_VOL_BCM]]
        .sum()
        for name, members in _HYDRO_SEASON_MEMBERSHIP.items()
    }


def met_seasonal_volumes(hydro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-meteorological-year total volume for each meteorological season.

    Keyed ``Winter``/``Spring``/``Summer``/``Monsoon``/``Autumn``
    (:class:`~hydrotrends.core.constants.MetSeason`); each value is a frame
    indexed by a **meteorological year** with summed ``Vol_MAF``/``Vol_BCM``.

    Winter (Dec + Jan + Feb) spans a calendar-year boundary, so it's labelled
    by the year its Jan/Feb fall in — December 2019 counts toward
    "Winter 2020" — the standard climatological convention. That triple
    always falls entirely inside one hydrological-year block (Dec is the 9th
    hydro-month, Jan/Feb the 10th/11th), so it is never split at the edges of
    ``hydro`` the way a plain calendar-year grouping would be. The other four
    seasons don't cross a year boundary and use the plain calendar year;
    Spring (Mar + Apr) does straddle a *hydrological*-year boundary, so its
    first/last occurrence in the record can be partial if either neighbouring
    hydro year was dropped as incomplete.
    """
    met_year = np.where(
        hydro[COL_MONTH_NUM] == _DECEMBER, hydro[COL_YEAR] + 1, hydro[COL_YEAR]
    )
    working = hydro.assign(_MetYear=met_year)
    return {
        season: working[working[COL_MET_SEASON] == season]
        .groupby("_MetYear")[[COL_VOL_MAF, COL_VOL_BCM]]
        .sum()
        for season in _MET_SEASON_ORDER
    }
