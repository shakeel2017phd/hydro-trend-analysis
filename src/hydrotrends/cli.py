"""Command-line interface for :mod:`hydrotrends`.

Exposes the ``hydro-trend`` command, which runs the full pipeline —
read -> clean -> preprocess -> analyse -> report — and writes an ``.xlsx``
workbook per input.

Two ways to specify inputs:

* **Flags** for a single settings profile shared by all positional files::

      hydro-trend analyze inflow.csv -o report.xlsx -r daily \\
          --value-column inflow_cusec

* **A YAML config** for heterogeneous inputs (different headers, scales, date
  encodings), which mirrors :class:`hydrotrends.core.config.Config`::

      hydro-trend analyze --config hydrotrends.yaml -o ./reports
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .api import ReportColumn, generate_report
from .core.config import Config, InputSpec
from .core.constants import (
    COL_FLOW_CUMECS,
    COL_FLOW_CUSECS,
    COL_VOL_BCM,
    COL_VOL_MAF,
    DEFAULT_ALPHA,
    UNIT_PAIRS,
    FlowUnit,
    VolumeUnit,
)
from .core.exceptions import HydroTrendsError
from .core.logging_config import configure_logging, get_logger
from .data.cleaning import clean
from .data.preprocessing import preprocess
from .data.readers import read_all

__all__ = ["main", "build_parser"]

logger = get_logger(__name__)

try:
    _VERSION = version("hydro-trend-analysis")
except PackageNotFoundError:  # not installed (e.g. run from a source checkout)
    _VERSION = "0.0.0"

_FLOW_COLUMN = {FlowUnit.CUSECS: COL_FLOW_CUSECS, FlowUnit.CUMECS: COL_FLOW_CUMECS}
_VOLUME_COLUMN = {VolumeUnit.MAF: COL_VOL_MAF, VolumeUnit.BCM: COL_VOL_BCM}
_SUBTITLE = "Mann-Kendall | Sen's Slope | ITA | Change-Point"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hydro-trend",
        description="Hydrological time-series trend analysis and reporting.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="run trend analysis and write a report")
    a.add_argument("input", nargs="*", help="input CSV/XLSX file(s)")
    a.add_argument(
        "-o",
        "--output",
        required=True,
        help="output .xlsx path (single input) or directory (multiple)",
    )
    a.add_argument("-r", "--resolution", choices=["daily", "10daily"], default="daily")
    a.add_argument("--date-column", default="Date")
    a.add_argument("--value-column", default="inflow_cusec")
    a.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="multiply the value column (e.g. 1000 for 1000-Cusecs)",
    )
    a.add_argument(
        "--date-format",
        choices=["auto", "month_first", "iso", "dekad_compact"],
        default="auto",
    )
    a.add_argument(
        "--calendar",
        action="store_true",
        help="use calendar-year framing (default: hydrological)",
    )
    a.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="significance level for the trend tests",
    )
    a.add_argument("--no-clean", action="store_true", help="skip the cleaning step")
    a.add_argument(
        "--no-descriptive",
        action="store_true",
        help="omit the descriptive-statistics sheets",
    )
    a.add_argument("--config", help="YAML config file (overrides the input flags)")
    a.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO, -vv for DEBUG logging",
    )
    a.set_defaults(func=_cmd_analyze)
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    if args.config:
        return Config.from_yaml(args.config)
    if not args.input:
        raise HydroTrendsError("no input files given (or use --config)")
    specs = [
        InputSpec(
            path=path,
            resolution=args.resolution,
            date_column=args.date_column,
            value_column=args.value_column,
            value_scale=args.value_scale,
            date_format=args.date_format,
        )
        for path in args.input
    ]
    return Config(inputs=tuple(specs), alpha=args.alpha)


def _output_path(output: str, spec: InputSpec, *, single: bool) -> Path:
    out = Path(output)
    if single and out.suffix.lower() == ".xlsx":
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{spec.path.stem}_report.xlsx"


def _cmd_analyze(args: argparse.Namespace) -> int:
    level = ("WARNING", "INFO", "DEBUG")[min(args.verbose, 2)]
    configure_logging(level)

    config = _config_from_args(args)
    if not config.inputs:
        raise HydroTrendsError("configuration lists no inputs")

    columns = [
        ReportColumn(
            _FLOW_COLUMN[unit],
            unit.value,
            volume_column=_VOLUME_COLUMN[UNIT_PAIRS[unit]],
            volume_unit_label=UNIT_PAIRS[unit].value,
        )
        for unit in config.flow_units
    ]
    single = len(config.inputs) == 1

    for spec, raw in read_all(config):
        frame = raw
        if not args.no_clean:
            frame, report = clean(raw)
            logger.info("cleaning %s: %s", spec.path.name, report.summary())
        pre = preprocess(frame, spec.resolution)
        out = _output_path(args.output, spec, single=single)
        generate_report(
            pre,
            out,
            columns=columns,
            title=f"{spec.path.stem} — Trend Analysis",
            subtitle=_SUBTITLE,
            calendar=args.calendar,
            include_descriptive=not args.no_descriptive,
            alpha=config.alpha,
            season_schemes=config.season_schemes,
        )
        logger.warning("wrote %s", out)  # always visible (default level)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``hydro-trend`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except HydroTrendsError as err:
        parser.exit(status=1, message=f"error: {err}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
