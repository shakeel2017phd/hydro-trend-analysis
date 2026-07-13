"""hydrotrends — hydrological time-series trend analysis.

A curated public surface for the common workflow::

    import hydrotrends as ht

    # bundled sample, or read your own
    pre = ht.datasets.load_preprocessed("tarbela_daily")
    ht.generate_report(pre, "report.xlsx",
                       columns=[ht.ReportColumn(ht.COL_FLOW_CUSECS, "Cusecs")])

    # or work with the pieces directly
    df   = ht.read_input(ht.InputSpec("inflow.csv", "daily"))
    mk   = ht.mann_kendall(series)
    ita  = ht.innovative_trend_analysis(series)

Plotting lives in :mod:`hydrotrends.viz.plotting` and is *not* imported here, so
``import hydrotrends`` does not pull in Matplotlib/Plotly/Seaborn.
"""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:  # not built yet (setuptools_scm generates _version.py)
    __version__ = "0.0.0+unknown"

from .api import (
    ReportColumn,
    analyze_by_period,
    analyze_monthly_volumes,
    analyze_preprocessed,
    analyze_seasonal_volumes,
    analyze_series,
    describe_monthly_volumes,
    describe_seasonal_volumes,
    generate_report,
)
from .core.config import Config, InputSpec, SeasonScheme
from .core.constants import (
    COL_DATE,
    COL_FLOW_CUMECS,
    COL_FLOW_CUSECS,
    COL_VOL_BCM,
    COL_VOL_MAF,
    DateFormat,
    FillMethod,
    FlowUnit,
    MetSeason,
    OnIssue,
    Season,
    TimeResolution,
    TrendDirection,
    VolumeUnit,
)
from .core.exceptions import (
    ConfigError,
    DataError,
    HydroTrendsError,
    InsufficientDataError,
    MissingColumnError,
    NonMonotonicIndexError,
    ReaderError,
    ValidationError,
)
from .data import datasets
from .data.cleaning import CleaningPolicy, CleaningReport, clean
from .data.preprocessing import (
    PreprocessedData,
    monthly_volumes,
    preprocess,
    seasonal_volumes,
)
from .data.readers import read_all, read_input
from .stats.changepoint import (
    PettittResult,
    bai_perron_change_point,
    cusum_change_point,
    pettitt_test,
)
from .stats.descriptive import DescriptiveStats, describe, describe_by
from .stats.frequency import (
    exceedance_counts,
    exceedance_probability,
    flow_duration_curve,
    flow_percentiles,
)
from .stats.ita import ITAResult, innovative_trend_analysis
from .stats.trends import (
    LinearFit,
    MannKendallResult,
    SenSlope,
    linear_regression,
    log_linear_regression,
    mann_kendall,
    mann_kendall_modified,
    moving_average_trend,
    percent_slope,
    sens_slope,
)

__all__ = [
    "__version__",
    # data access
    "datasets",
    "read_input",
    "read_all",
    "clean",
    "CleaningPolicy",
    "CleaningReport",
    "preprocess",
    "PreprocessedData",
    "monthly_volumes",
    "seasonal_volumes",
    # configuration
    "Config",
    "InputSpec",
    "SeasonScheme",
    # analysis + reporting
    "analyze_series",
    "analyze_by_period",
    "analyze_preprocessed",
    "analyze_monthly_volumes",
    "analyze_seasonal_volumes",
    "generate_report",
    "ReportColumn",
    # descriptive
    "describe",
    "describe_by",
    "describe_monthly_volumes",
    "describe_seasonal_volumes",
    "DescriptiveStats",
    # trends
    "mann_kendall",
    "mann_kendall_modified",
    "sens_slope",
    "linear_regression",
    "log_linear_regression",
    "moving_average_trend",
    "percent_slope",
    "MannKendallResult",
    "SenSlope",
    "LinearFit",
    # innovative trend analysis
    "innovative_trend_analysis",
    "ITAResult",
    # change-point
    "pettitt_test",
    "cusum_change_point",
    "bai_perron_change_point",
    "PettittResult",
    # frequency
    "flow_duration_curve",
    "exceedance_counts",
    "exceedance_probability",
    "flow_percentiles",
    # enums
    "TimeResolution",
    "FlowUnit",
    "VolumeUnit",
    "DateFormat",
    "OnIssue",
    "FillMethod",
    "TrendDirection",
    "Season",
    "MetSeason",
    # canonical column names
    "COL_DATE",
    "COL_FLOW_CUSECS",
    "COL_FLOW_CUMECS",
    "COL_VOL_MAF",
    "COL_VOL_BCM",
    # exceptions
    "HydroTrendsError",
    "ConfigError",
    "DataError",
    "ReaderError",
    "MissingColumnError",
    "ValidationError",
    "InsufficientDataError",
    "NonMonotonicIndexError",
]
