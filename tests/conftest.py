"""Shared fixtures for the hydrotrends test suite."""
from __future__ import annotations

import numpy as np
import pytest

import hydrotrends as ht


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def trend_series(rng: np.random.Generator) -> np.ndarray:
    """A 30-point series with a clear upward trend."""
    return 50 + 2.0 * np.arange(30) + rng.normal(0, 3, 30)


@pytest.fixture
def step_series() -> np.ndarray:
    """A step change at position 10 (n=20)."""
    return np.array([10.0] * 10 + [30.0] * 10)


@pytest.fixture(scope="session")
def daily_pre() -> ht.PreprocessedData:
    return ht.datasets.load_preprocessed("tarbela_daily")


@pytest.fixture(scope="session")
def tendaily_pre() -> ht.PreprocessedData:
    return ht.datasets.load_preprocessed("tarbela_10daily")
