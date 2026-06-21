"""Shared fixtures for the qedge test suite.

Bare-environment layering: heavy ML deps (hmmlearn, river, ruptures, shap,
lightgbm, scikit-learn) are imported lazily inside the tests that need them via
``pytest.importorskip``, so the numpy/pandas-only tests (Protocols, fracdiff,
synthetic determinism, no-lookahead canaries) run before the full stack is
installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from qedge.config import QedgeConfig


@pytest.fixture()
def cfg() -> QedgeConfig:
    """A default config instance (no env overrides applied in tests)."""
    return QedgeConfig()


@pytest.fixture()
def seeded_rng() -> np.random.Generator:
    """A deterministic NumPy generator for reproducible test data."""
    return np.random.default_rng(7)
