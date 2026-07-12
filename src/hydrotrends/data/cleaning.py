"""Policy-driven data cleaning for :mod:`hydrotrends`.

The library equivalent of the notebook's interactive "delete these rows? (y/n)"
prompts — but reproducible and side-effect-free. :func:`clean` inspects a
canonical frame for the three classic problems (missing values, duplicate
dates, negative flows), applies a declarative :class:`CleaningPolicy`, and
returns the cleaned frame together with a :class:`CleaningReport` of exactly
what it found and did. No console prompts, no hidden mutation.

Pipeline position: ``readers.read_input`` -> ``clean`` -> ``preprocessing``.
Cleaning is optional; skip it and the reader's warnings still tell you what's
in the data.

Non-numeric values are folded into "missing": they became NaN at read time (the
reader coerces and warns), and :func:`clean` re-coerces defensively, so a bad
string is handled by the ``missing`` policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.constants import COL_DATE, COL_FLOW_CUSECS, FillMethod, OnIssue
from ..core.exceptions import ValidationError
from ..core.logging_config import get_logger

__all__ = ["OnIssue", "FillMethod", "CleaningPolicy", "CleaningReport", "clean"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleaningPolicy:
    """How to handle each data-quality issue.

    Defaults are conservative-but-safe: drop unusable rows (missing, duplicate
    dates), but *keep* negatives — a reservoir's net inflow can legitimately go
    negative, so they're reported rather than silently deleted.
    """

    missing: OnIssue = OnIssue.DROP
    duplicate_dates: OnIssue = OnIssue.DROP
    negatives: OnIssue = OnIssue.KEEP
    missing_fill: FillMethod = FillMethod.NONE   # impute value gaps before `missing`

    def __post_init__(self) -> None:
        for field_name in ("missing", "duplicate_dates", "negatives"):
            value = getattr(self, field_name)
            if not isinstance(value, OnIssue):
                try:
                    object.__setattr__(self, field_name, OnIssue(value))
                except ValueError as err:
                    valid = ", ".join(o.value for o in OnIssue)
                    raise ValidationError(
                        f"invalid policy for {field_name!r}: {value!r}; "
                        f"choose from: {valid}"
                    ) from err
        if not isinstance(self.missing_fill, FillMethod):
            try:
                object.__setattr__(self, "missing_fill", FillMethod(self.missing_fill))
            except ValueError as err:
                valid = ", ".join(f.value for f in FillMethod)
                raise ValidationError(
                    f"invalid missing_fill {self.missing_fill!r}; choose from: {valid}"
                ) from err


@dataclass(frozen=True)
class CleaningReport:
    """What cleaning found and removed. All counts are pre-removal detections."""

    n_input: int
    n_missing: int
    n_filled: int
    n_duplicate_dates: int
    n_negative: int
    n_removed: int
    n_output: int

    def summary(self) -> str:
        return (
            f"{self.n_input} rows in -> {self.n_output} out "
            f"(removed {self.n_removed}; filled {self.n_filled}; detected: "
            f"missing={self.n_missing}, duplicate_dates={self.n_duplicate_dates}, "
            f"negatives={self.n_negative})"
        )


def _apply(
    issue: OnIssue,
    mask: pd.Series,
    drop_mask: pd.Series,
    *,
    label: str,
) -> pd.Series:
    """Fold one issue's rows into the running drop-mask per its policy."""
    count = int(mask.sum())
    if not count:
        return drop_mask
    if issue is OnIssue.ERROR:
        raise ValidationError(f"{count} {label} value(s) found (policy=error)")
    if issue is OnIssue.DROP:
        return drop_mask | mask
    logger.warning("%d %s value(s) kept (policy=keep)", count, label)  # KEEP
    return drop_mask


def clean(
    df: pd.DataFrame,
    policy: CleaningPolicy | None = None,
    *,
    date_col: str = COL_DATE,
    value_col: str = COL_FLOW_CUSECS,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a canonical frame per ``policy``; return ``(cleaned, report)``.

    Duplicate detection is on the *date* column (each timestamp should be
    unique); the first occurrence is kept. Missing covers NaN in either the date
    or value column (including non-numeric values coerced to NaN).
    """
    policy = policy or CleaningPolicy()
    n_input = len(df)
    work = df.copy()

    # Defensive re-coercion so a stray non-numeric value counts as missing.
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    # Optionally impute value gaps (on rows that have a valid date) before the
    # missing-row policy runs. Interpolation is opt-in because it can bias trends.
    n_filled = _fill_missing(work, policy.missing_fill, date_col, value_col)

    missing_mask = work[date_col].isna() | work[value_col].isna()
    dup_mask = work[date_col].notna() & work[date_col].duplicated(keep="first")
    negative_mask = work[value_col] < 0  # NaN < 0 is False, so excludes missing

    n_missing = int(missing_mask.sum())
    n_duplicate = int(dup_mask.sum())
    n_negative = int(negative_mask.sum())

    drop_mask = pd.Series(False, index=work.index)
    drop_mask = _apply(policy.missing, missing_mask, drop_mask, label="missing")
    drop_mask = _apply(
        policy.duplicate_dates, dup_mask, drop_mask, label="duplicate-date"
    )
    drop_mask = _apply(policy.negatives, negative_mask, drop_mask, label="negative")

    cleaned = work[~drop_mask].reset_index(drop=True)
    cleaned.attrs.update(df.attrs)  # preserve resolution/source_path
    report = CleaningReport(
        n_input=n_input,
        n_missing=n_missing,
        n_filled=n_filled,
        n_duplicate_dates=n_duplicate,
        n_negative=n_negative,
        n_removed=int(drop_mask.sum()),
        n_output=len(cleaned),
    )
    logger.info("cleaning: %s", report.summary())
    return cleaned, report


def _fill_missing(
    work: pd.DataFrame,
    method: FillMethod,
    date_col: str,
    value_col: str,
) -> int:
    """Impute missing values in ``value_col`` (rows with a valid date). Returns count filled."""
    if method is FillMethod.NONE:
        return 0
    fillable = work[value_col].isna() & work[date_col].notna()
    if not fillable.any():
        return 0
    if method is FillMethod.LINEAR:
        work[value_col] = work[value_col].interpolate(
            method="linear", limit_area="inside"
        )
    elif method is FillMethod.FFILL:
        work[value_col] = work[value_col].ffill()
    else:  # FillMethod.BFILL
        work[value_col] = work[value_col].bfill()
    n_filled = int((fillable & work[value_col].notna()).sum())
    if n_filled:
        logger.info("filled %d missing value(s) via %s", n_filled, method.value)
    return n_filled