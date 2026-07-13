"""Parity check against the source script's statistical core.

Runs :func:`hydrotrends.analyze_preprocessed` against
``tests/reference_v26_core.py`` (a frozen, verbatim copy of the source
script's formulas -- see that module's docstring) on the same synthetic data,
and asserts the two agree on every shared field, for both daily and 10-daily
resolutions and both flow columns (Cusecs/Cumecs).

Scope note: this checks the **statistical engine** only, not the full
2000+-line source script end to end -- that script hardcodes a Windows CSV
path and runs matplotlib/plotly plotting at import time, so it isn't
something an automated test can literally ``import`` and execute. What's
checked here is exactly what could regress silently: the trend-test and
descriptive-stat formulas per period, across years, for the daily-period and
10-daily-dekad tables the source script actually produces
(``RESULTS["daily_<col>"]`` / ``RESULTS["10daily_<col>"]``). The synthetic
data (not the tiny 3-hydro-year bundled sample) is used deliberately, so
every period has enough points to exercise the full computation path --
trend tests (n >= 4), the Hamed-Rao autocorrelation correction (n >= 10), and
change-point detection (n >= 5) -- rather than hitting the ``n < 4``
descriptive-only guard everywhere.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import hydrotrends as ht
from hydrotrends.core.constants import (
    COL_DATE,
    COL_FLOW_CUMECS,
    COL_FLOW_CUSECS,
    COL_PERIOD,
    COL_YEAR,
    TimeResolution,
)

from . import reference_v26_core as ref

_STR_KEYS = {
    "trend",
    "mmk_trend",
    "significance",
    "mmk_sig",
    "ita_trend",
    "pettitt_sig",
    "ma5_trend",
}
_INT_KEYS = {"n", "pettitt_cp_year", "cusum_cp_year", "bpcp_year"}

# Decimal precision the source actually rounds each field to (see
# reference_v26_core.full_analysis / ols_fit) -- lin_slope is 8dp, sens_slope/
# ita_slope/log_slope/lin_intercept are 6dp, pettitt_K is 2dp, everything else
# numeric is 4dp. Comparisons below use a tolerance of *one rounding step* at
# this precision (not exact equality after rounding), because two independent
# implementations computing the same value via a different operation order
# can legitimately land on opposite sides of a rounding boundary in the last
# digit (e.g. 0.068950045... vs a pre-rounded 0.0689) without that being a
# real discrepancy -- only a difference far bigger than one rounding step
# indicates an actual formula/logic mismatch.
_PRECISION = {
    "lin_slope": 8,
    "sens_slope": 6,
    "ita_slope": 6,
    "log_slope": 6,
    "lin_intercept": 6,
    "pettitt_K": 2,
}
_DEFAULT_PRECISION = 4

_SHARED_KEYS = [
    "n",
    "mean",
    "median",
    "std",
    "cv_pct",
    "min",
    "max",
    "p10",
    "p25",
    "p75",
    "p90",
    "skew",
    "kurt",
    "trend",
    "p_value",
    "z_score",
    "sens_slope",
    "sens_slope_pct",
    "significance",
    "mmk_trend",
    "mmk_p",
    "mmk_z",
    "mmk_sig",
    "lin_slope",
    "lin_intercept",
    "lin_r2",
    "lin_p",
    "log_slope",
    "log_r2",
    "log_p",
    "ma5_mean",
    "ma5_std",
    "ma5_trend",
    "ita_slope",
    "ita_trend",
    "pettitt_cp_year",
    "pettitt_K",
    "pettitt_p",
    "pettitt_sig",
    "cusum_cp_year",
    "bpcp_year",
]


def _is_nan(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _assert_field_matches(hydro_row: dict, ref_row: dict, key: str, where: str) -> None:
    h, r = hydro_row.get(key), ref_row.get(key)
    if _is_nan(r) or _is_nan(h):
        assert _is_nan(r) and _is_nan(h), f"{where} {key}: hydro={h!r} ref={r!r}"
        return
    if key in _STR_KEYS:
        # The source pads some significance codes with a trailing space
        # ("* ", ". "); hydrotrends doesn't. Compare stripped.
        assert str(h).strip() == str(r).strip(), f"{where} {key}: hydro={h!r} ref={r!r}"
        return
    if key in _INT_KEYS:
        assert int(h) == int(r), f"{where} {key}: hydro={h!r} ref={r!r}"
        return
    ndigits = _PRECISION.get(key, _DEFAULT_PRECISION)
    tolerance = 1.5 * 10**-ndigits
    assert abs(float(h) - float(r)) <= tolerance, (
        f"{where} {key}: hydro={h!r} ref={r!r} (tolerance {tolerance:g})"
    )


def _reference_table(hydro: pd.DataFrame, value_col: str) -> dict[str, dict]:
    """Reproduce the source script's per-period loop exactly (its Section 5):
    group by Period, sort each group by calendar Year, run full_analysis."""
    rows = {}
    for period, group in hydro.groupby(COL_PERIOD, sort=False):
        sub = group.sort_values(COL_YEAR)
        rows[str(period)] = ref.full_analysis(sub[value_col], sub[COL_YEAR].values)
    return rows


def _assert_tables_match(pre, value_col: str, expected_periods: int) -> None:
    hydro_table = ht.analyze_preprocessed(pre, value_col=value_col)
    ref_table = _reference_table(pre.hydro, value_col)

    assert set(hydro_table.index) >= set(ref_table)
    for period, ref_row in ref_table.items():
        hydro_row = hydro_table.loc[period].to_dict()
        for key in _SHARED_KEYS:
            _assert_field_matches(hydro_row, ref_row, key, where=f"[{period}]")
    assert len(ref_table) == expected_periods


def _synthetic_daily(n_years: int = 20, seed: int = 42) -> pd.DataFrame:
    """``n_years`` complete hydrological years of daily flow: a mild trend
    plus a seasonal cycle plus noise, kept positive (needed for the source's
    log-linear regression, which requires >= 4 positive values)."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, start_year in enumerate(range(2000, 2000 + n_years)):
        dates = pd.date_range(f"{start_year}-04-01", f"{start_year + 1}-03-31")
        doy = np.arange(len(dates))
        base = 50_000 + 300 * i
        seasonal = 20_000 * np.sin(2 * np.pi * doy / 365.25)
        noise = rng.normal(0, 2_000, len(dates))
        flow = np.clip(base + seasonal + noise, 1_000, None)
        frames.append(pd.DataFrame({COL_DATE: dates, COL_FLOW_CUSECS: flow}))
    return pd.concat(frames, ignore_index=True)


def _synthetic_10daily(n_years: int = 20, seed: int = 43) -> pd.DataFrame:
    """``n_years`` of one row per dekad (day 1/11/21 representative dates,
    matching how the dekad-compact reader parses a 10-daily source file)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, start_year in enumerate(range(2000, 2000 + n_years)):
        months = [(start_year, m) for m in (4, 5, 6, 7, 8, 9, 10, 11, 12)] + [
            (start_year + 1, m) for m in (1, 2, 3)
        ]
        base = 50_000 + 300 * i
        for month_idx, (yr, mo) in enumerate(months):
            seasonal = 20_000 * np.sin(2 * np.pi * month_idx / 12)
            for day in (1, 11, 21):
                flow = max(base + seasonal + rng.normal(0, 1_500), 1_000)
                date = pd.Timestamp(yr, mo, day)
                rows.append({COL_DATE: date, COL_FLOW_CUSECS: flow})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("value_col", [COL_FLOW_CUSECS, COL_FLOW_CUMECS])
def test_daily_parity_against_v26_reference(value_col):
    pre = ht.preprocess(_synthetic_daily(), TimeResolution.DAILY)
    _assert_tables_match(pre, value_col, expected_periods=366)


@pytest.mark.parametrize("value_col", [COL_FLOW_CUSECS, COL_FLOW_CUMECS])
def test_10daily_parity_against_v26_reference(value_col):
    pre = ht.preprocess(_synthetic_10daily(), TimeResolution.TEN_DAILY)
    _assert_tables_match(pre, value_col, expected_periods=36)


def test_reference_module_matches_hydrotrends_on_a_single_series():
    """Sanity check the reference fixture itself against the ported
    functions directly (independent of the per-period grouping machinery),
    on a plain step-change series."""
    x = np.array([10.0] * 10 + [30.0] * 10 + list(range(40, 50)))

    ref_row = ref.full_analysis(pd.Series(x))
    hydro_row = ht.analyze_series(x)

    for key in _SHARED_KEYS:
        if key in ("pettitt_cp_year", "cusum_cp_year", "bpcp_year"):
            continue  # no explicit `years` passed here; index-based, checked above
        assert key in hydro_row or math.isnan(ref_row.get(key, float("nan"))), key
        _assert_field_matches(hydro_row, ref_row, key, where="[single-series]")
