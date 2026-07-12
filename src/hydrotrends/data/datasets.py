"""Bundled sample datasets for :mod:`hydrotrends`.

Small, ready-to-use Tarbela inflow samples shipped inside the package so
examples, tests, and quick experiments work with zero setup::

    import hydrotrends.data.datasets as ds

    ds.available_datasets()          # ['tarbela_10daily', 'tarbela_daily']
    df   = ds.load("tarbela_daily")           # canonical frame
    pre  = ds.load_preprocessed("tarbela_daily")
    spec = ds.dataset_spec("tarbela_10daily") # the InputSpec (scale + date format wired in)

Each dataset already knows its resolution, value scaling, and date encoding, so
the 10-daily sample transparently exercises ``value_scale=1000`` and
``date_format="dekad_compact"``.

Files are located via :mod:`importlib.resources`, so this works from an
installed wheel — provided the package is installed normally (unzipped), which
is the case for a standard ``pip install``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from ..core.config import InputSpec
from ..core.constants import DateFormat, TimeResolution
from ..core.exceptions import ReaderError
from .preprocessing import PreprocessedData, preprocess
from .readers import read_input

__all__ = [
    "available_datasets",
    "sample_path",
    "dataset_spec",
    "load",
    "load_preprocessed",
]

_SAMPLE_PACKAGE = "hydrotrends.data"
_SAMPLE_DIR = "sample_data"


@dataclass(frozen=True)
class _SampleMeta:
    filename: str
    resolution: TimeResolution
    value_column: str = "inflow_cusec"
    value_scale: float = 1.0
    date_format: DateFormat = DateFormat.AUTO


_SAMPLES: dict[str, _SampleMeta] = {
    "tarbela_daily": _SampleMeta(
        filename="tarbela_daily_sample.csv",
        resolution=TimeResolution.DAILY,
    ),
    "tarbela_10daily": _SampleMeta(
        filename="tarbela_10daily_sample.csv",
        resolution=TimeResolution.TEN_DAILY,
        value_column="Inflow_1000Cusecs",
        value_scale=1000.0,
        date_format=DateFormat.DEKAD_COMPACT,
    ),
}


def available_datasets() -> list[str]:
    """Names of the bundled sample datasets."""
    return sorted(_SAMPLES)


def _meta(name: str) -> _SampleMeta:
    try:
        return _SAMPLES[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; available: {', '.join(available_datasets())}"
        ) from None


def sample_path(name: str) -> Path:
    """Filesystem path to a bundled sample file."""
    meta = _meta(name)
    resource = resources.files(_SAMPLE_PACKAGE).joinpath(_SAMPLE_DIR, meta.filename)
    with resources.as_file(resource) as located:
        path = Path(located)
    if not path.exists():
        raise ReaderError("bundled sample data not found", path=str(path))
    return path


def dataset_spec(name: str) -> InputSpec:
    """The :class:`InputSpec` for a bundled dataset (scale + date format wired in)."""
    meta = _meta(name)
    return InputSpec(
        path=sample_path(name),
        resolution=meta.resolution,
        value_column=meta.value_column,
        value_scale=meta.value_scale,
        date_format=meta.date_format,
    )


def load(name: str) -> pd.DataFrame:
    """Read a bundled dataset into a canonical frame."""
    return read_input(dataset_spec(name))


def load_preprocessed(name: str) -> PreprocessedData:
    """Read and preprocess a bundled dataset in one step."""
    spec = dataset_spec(name)
    return preprocess(read_input(spec), spec.resolution)