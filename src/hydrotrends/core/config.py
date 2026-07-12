"""Runtime configuration for :mod:`hydrotrends`.

Where :mod:`hydrotrends.core.constants` holds fixed domain facts, this module
holds the **overridable** settings for a run: input/output paths (replacing the
original script's hardcoded ``D:\\Indus_Analysis\\...`` literals), which flow
units to process, the significance level, figure DPI, and which seasonal
classifications to build.

By default a run produces **both** seasonal breakdowns — the Kharif/Rabi
cropping seasons *and* the calendar-based meteorological seasons
(:data:`~hydrotrends.core.constants.MET_SEASONS`) — mirroring how the original
script emits both unit workbooks. Either can be switched off via
``season_schemes``.

:class:`Config` is a frozen, self-validating dataclass. Construct it directly,
from a dict, or from a YAML file::

    from hydrotrends.core.config import Config, InputSpec

    cfg = Config(
        inputs=[
            InputSpec("~/data/tarbela_daily.csv", "daily"),
            InputSpec(
                "~/data/tarbela_10daily.csv", "10daily",
                date_column="dekad_end", value_column="avg_inflow",
            ),
        ],
        output_dir="./outputs",
    )
    cfg = Config.from_yaml("hydrotrends.yaml")

Invalid values raise :class:`hydrotrends.core.exceptions.ConfigError` at
construction time, so a bad config fails fast and loudly rather than surfacing
as a confusing error deep in the analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .constants import (
    DEFAULT_ALPHA,
    LOWESS_FRAC,
    MET_SEASONS,
    MOVING_AVERAGE_WINDOW,
    DateFormat,
    FlowUnit,
    MetSeason,
    TimeResolution,
)
from .exceptions import ConfigError
from .logging_config import get_logger

__all__ = ["SeasonScheme", "InputSpec", "Config"]

logger = get_logger(__name__)

_DEFAULT_DPI = 300
_DEFAULT_LOG_LEVEL = "INFO"


class SeasonScheme(StrEnum):
    """A seasonal classification an analysis run can build.

    ``CROPPING`` uses the Kharif/Rabi seasons (with the day-level June-10 split,
    applied in ``preprocessing``). ``METEOROLOGICAL`` uses the calendar-based
    :data:`~hydrotrends.core.constants.MET_SEASONS` month map. A run may enable
    one or both.
    """

    CROPPING = "cropping"
    METEOROLOGICAL = "meteorological"


@dataclass(frozen=True, slots=True)
class InputSpec:
    """A single input file paired with its temporal resolution and its headers.

    Constructing a run from a list of these is what lets the package process a
    daily file, a 10-daily file, or both at once — the resolution travels with
    the path so readers/preprocessing stay resolution-aware. ``date_column`` and
    ``value_column`` are the header names *in this file* (they may differ
    between files); readers normalise them to the canonical internal schema.
    """

    path: Path
    resolution: TimeResolution
    date_column: str = "Date"
    value_column: str = "inflow_cusec"
    value_scale: float = (
        1.0  # multiply the value column by this (e.g. 1000 for 1000-Cusecs)
    )
    date_format: DateFormat = DateFormat.AUTO

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path).expanduser())
        object.__setattr__(self, "value_scale", float(self.value_scale))
        if self.value_scale <= 0:
            raise ConfigError(f"value_scale must be > 0, got {self.value_scale}")
        if not isinstance(self.resolution, TimeResolution):
            try:
                object.__setattr__(self, "resolution", TimeResolution(self.resolution))
            except ValueError as err:
                valid = ", ".join(r.value for r in TimeResolution)
                raise ConfigError(
                    f"invalid resolution {self.resolution!r}; choose from: {valid}"
                ) from err
        if not isinstance(self.date_format, DateFormat):
            try:
                object.__setattr__(self, "date_format", DateFormat(self.date_format))
            except ValueError as err:
                valid = ", ".join(d.value for d in DateFormat)
                raise ConfigError(
                    f"invalid date_format {self.date_format!r}; choose from: {valid}"
                ) from err
        if (
            self.date_format is DateFormat.DEKAD_COMPACT
            and self.resolution is TimeResolution.DAILY
        ):
            raise ConfigError(
                "date_format='dekad_compact' only applies to 10-daily input, not daily"
            )


_T = TypeVar("_T")


def _dedupe_preserving_order(items: Iterable[_T]) -> tuple[_T, ...]:
    seen: list[_T] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class Config:
    """Validated, immutable settings for a single analysis run."""

    inputs: tuple[InputSpec, ...] = ()
    output_dir: Path = Path("outputs")

    # Which flow units to build (each pairs with a volume unit via UNIT_PAIRS).
    # The original script builds both workbooks, so both are on by default.
    flow_units: tuple[FlowUnit, ...] = (FlowUnit.CUSECS, FlowUnit.CUMECS)

    # Which seasonal classifications to build. BOTH by default.
    season_schemes: tuple[SeasonScheme, ...] = (
        SeasonScheme.CROPPING,
        SeasonScheme.METEOROLOGICAL,
    )

    # Statistical knobs (defaults sourced from constants).
    alpha: float = DEFAULT_ALPHA
    lowess_frac: float = LOWESS_FRAC
    moving_average_window: int = MOVING_AVERAGE_WINDOW

    # Output / behaviour.
    dpi: int = _DEFAULT_DPI
    make_interactive: bool = True  # export Plotly HTML alongside PNGs
    log_level: str = _DEFAULT_LOG_LEVEL

    # ── validation & coercion ────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._coerce_paths()
        self._coerce_inputs()
        self._coerce_flow_units()
        self._coerce_season_schemes()
        self._validate_numeric()

    def _coerce_paths(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())

    def _coerce_inputs(self) -> None:
        specs: list[InputSpec] = []
        for item in self.inputs:
            if isinstance(item, InputSpec):
                specs.append(item)
            elif isinstance(item, Mapping):
                allowed = {
                    "path",
                    "resolution",
                    "date_column",
                    "value_column",
                    "value_scale",
                    "date_format",
                }
                unknown = set(item) - allowed
                if unknown:
                    raise ConfigError(
                        f"unknown input key(s): {', '.join(sorted(unknown))}; "
                        f"allowed: {', '.join(sorted(allowed))}"
                    )
                if "path" not in item or "resolution" not in item:
                    raise ConfigError(
                        "input spec needs at least 'path' and 'resolution'"
                    )
                specs.append(InputSpec(**item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                specs.append(InputSpec(path=item[0], resolution=item[1]))
            else:
                raise ConfigError(
                    f"invalid input spec {item!r}; expected InputSpec, "
                    "{'path':..., 'resolution':...}, or (path, resolution)"
                )
        object.__setattr__(self, "inputs", tuple(specs))

    def _coerce_flow_units(self) -> None:
        try:
            units = _dedupe_preserving_order(FlowUnit(u) for u in self.flow_units)
        except ValueError as err:
            valid = ", ".join(u.value for u in FlowUnit)
            raise ConfigError(
                f"invalid flow unit ({err}); choose from: {valid}"
            ) from err
        if not units:
            raise ConfigError("flow_units must contain at least one unit")
        object.__setattr__(self, "flow_units", units)

    def _coerce_season_schemes(self) -> None:
        try:
            schemes = _dedupe_preserving_order(
                SeasonScheme(s) for s in self.season_schemes
            )
        except ValueError as err:
            valid = ", ".join(s.value for s in SeasonScheme)
            raise ConfigError(
                f"invalid season scheme ({err}); choose from: {valid}"
            ) from err
        if not schemes:
            raise ConfigError(
                "season_schemes must contain at least one of: "
                + ", ".join(s.value for s in SeasonScheme)
            )
        object.__setattr__(self, "season_schemes", schemes)

    def _validate_numeric(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0.0 < self.lowess_frac <= 1.0:
            raise ConfigError(f"lowess_frac must be in (0, 1], got {self.lowess_frac}")
        if self.moving_average_window < 1:
            raise ConfigError(
                f"moving_average_window must be >= 1, got {self.moving_average_window}"
            )
        if self.dpi < 1:
            raise ConfigError(f"dpi must be >= 1, got {self.dpi}")

    # ── seasonal helpers ─────────────────────────────────────────────────────
    @property
    def includes_cropping(self) -> bool:
        """True when the Kharif/Rabi cropping scheme should be built."""
        return SeasonScheme.CROPPING in self.season_schemes

    @property
    def includes_meteorological(self) -> bool:
        """True when the meteorological scheme should be built."""
        return SeasonScheme.METEOROLOGICAL in self.season_schemes

    @property
    def meteorological_seasons(self) -> Mapping[MetSeason, tuple[int, ...]]:
        """The month-map ``preprocessing`` applies for meteorological seasons.

        Always returns the constant view; it is only *applied* when
        :attr:`includes_meteorological` is true. Cropping seasons have no
        equivalent pure month-map (the June-10 split is day-level) and are
        handled directly in ``preprocessing``.
        """
        return MET_SEASONS

    # ── (de)serialisation ────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        """Build a Config from a mapping, rejecting unknown keys (catches typos)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown config key(s): {', '.join(sorted(unknown))}; "
                f"valid keys: {', '.join(sorted(known))}"
            )
        return cls(**dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a Config from a YAML file."""
        p = Path(path).expanduser()
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except FileNotFoundError as err:
            raise ConfigError(f"config file not found: {p}") from err
        except yaml.YAMLError as err:
            raise ConfigError(f"could not parse YAML config {p}: {err}") from err

        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ConfigError(
                f"config file {p} must contain a mapping at the top level"
            )
        logger.debug("loaded config from %s", p)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain, YAML-friendly Python types."""
        return {
            "inputs": [
                {
                    "path": str(spec.path),
                    "resolution": spec.resolution.value,
                    "date_column": spec.date_column,
                    "value_column": spec.value_column,
                    "value_scale": spec.value_scale,
                    "date_format": spec.date_format.value,
                }
                for spec in self.inputs
            ],
            "output_dir": str(self.output_dir),
            "flow_units": [u.value for u in self.flow_units],
            "season_schemes": [s.value for s in self.season_schemes],
            "alpha": self.alpha,
            "lowess_frac": self.lowess_frac,
            "moving_average_window": self.moving_average_window,
            "dpi": self.dpi,
            "make_interactive": self.make_interactive,
            "log_level": self.log_level,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        logger.debug("wrote config to %s", p)
