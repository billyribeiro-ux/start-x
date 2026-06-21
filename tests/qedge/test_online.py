"""Tests for online learning and concept-drift detection (`qedge.modeling.online`).

Three properties are pinned down:

* an :class:`OnlineClassifier` actually *learns* — prequential accuracy on a
  linearly separable stream beats chance;
* a :class:`DriftMonitor` is quiet on a stationary stream and trips shortly
  after an abrupt mean shift, with ``n_drifts`` rising;
* everything that uses randomness is seeded, so the suite is deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from qedge.config import QedgeConfig
from qedge.modeling.online import (
    DriftMonitor,
    OnlineClassifier,
    prequential_accuracy,
)

# A learnable stream needs enough samples for SGD logistic regression to warm
# up before its running accuracy is judged.
_N_LEARNABLE_SAMPLES = 2000
# Comfortably above the 0.5 coin-flip floor for a separable linear problem.
_ABOVE_CHANCE = 0.65

# Drift-stream geometry: a long stationary pre-segment, then a large mean jump.
_N_STATIONARY = 1500
_N_PRE_SHIFT = 1000
_N_POST_SHIFT = 1000
_SHIFT_MAGNITUDE = 10.0
# Detectors are allowed a short settling window after the change point.
_DETECTION_GRACE = 200
# A stationary stream should stay (near) silent.
_MAX_STATIONARY_DRIFTS = 2


def _linearly_separable_stream(
    n: int, rng: np.random.Generator
) -> list[tuple[dict[str, float], int]]:
    """Build ``n`` samples whose label is the sign of a fixed linear combo.

    ``y = 1`` iff ``2*x0 - 3*x1 + 0.5*x2 > 0``. A tiny amount of noise keeps the
    boundary non-trivial without making it unlearnable.
    """
    weights = np.array([2.0, -3.0, 0.5])
    features = rng.standard_normal((n, 3))
    noise = 0.1 * rng.standard_normal(n)
    scores = features @ weights + noise
    stream: list[tuple[dict[str, float], int]] = []
    for row, score in zip(features, scores, strict=True):
        x = {"x0": float(row[0]), "x1": float(row[1]), "x2": float(row[2])}
        y = int(score > 0.0)
        stream.append((x, y))
    return stream


def test_online_classifier_learns_above_chance() -> None:
    """Prequential (test-then-train) accuracy must beat a coin flip."""
    rng = np.random.default_rng(7)
    stream = _linearly_separable_stream(_N_LEARNABLE_SAMPLES, rng)
    accuracy = prequential_accuracy(stream)
    assert accuracy > _ABOVE_CHANCE


def test_online_classifier_predict_methods() -> None:
    """After training, the API surface returns coherent, in-domain outputs."""
    rng = np.random.default_rng(7)
    stream = _linearly_separable_stream(_N_LEARNABLE_SAMPLES, rng)
    clf = OnlineClassifier()
    for x, y in stream:
        clf.learn_one(x, y)

    probe = {"x0": 1.0, "x1": -1.0, "x2": 0.0}
    proba = clf.predict_proba_one(probe)
    assert pytest.approx(sum(proba.values()), abs=1e-9) == 1.0
    assert all(0.0 <= p <= 1.0 for p in proba.values())
    assert clf.predict_one(probe) in (0, 1)


def test_drift_monitor_quiet_on_stationary_stream() -> None:
    """A stationary Gaussian stream should produce few/no drift flags."""
    cfg = QedgeConfig()
    rng = np.random.default_rng(7)
    monitor = DriftMonitor(cfg.online)
    for _ in range(_N_STATIONARY):
        monitor.update(float(rng.normal(0.0, 1.0)))
    assert monitor.n_drifts <= _MAX_STATIONARY_DRIFTS


def test_drift_monitor_flags_abrupt_shift() -> None:
    """An abrupt mean shift must be flagged shortly after the change point."""
    cfg = QedgeConfig()
    rng = np.random.default_rng(7)
    monitor = DriftMonitor(cfg.online)

    detected_index: int | None = None
    drifts_before_shift = 0
    for i in range(_N_PRE_SHIFT + _N_POST_SHIFT):
        mean = 0.0 if i < _N_PRE_SHIFT else _SHIFT_MAGNITUDE
        fired = monitor.update(float(rng.normal(mean, 1.0)))
        if i < _N_PRE_SHIFT and fired:
            drifts_before_shift += 1
        if i >= _N_PRE_SHIFT and fired and detected_index is None:
            detected_index = i

    assert monitor.n_drifts > 0
    assert detected_index is not None
    assert detected_index - _N_PRE_SHIFT <= _DETECTION_GRACE
    assert monitor.last_detector is not None
    assert monitor.last_detector in {
        "ADWIN",
        "PageHinkley",
        "ADWIN+PageHinkley",
    }


def test_drift_monitor_is_deterministic() -> None:
    """Same seed -> identical drift trajectory (no hidden state divergence)."""
    cfg = QedgeConfig()

    def run() -> int:
        rng = np.random.default_rng(7)
        monitor = DriftMonitor(cfg.online)
        for i in range(_N_PRE_SHIFT + _N_POST_SHIFT):
            mean = 0.0 if i < _N_PRE_SHIFT else _SHIFT_MAGNITUDE
            monitor.update(float(rng.normal(mean, 1.0)))
        return monitor.n_drifts

    assert run() == run()


def test_prequential_accuracy_empty_stream() -> None:
    """An empty stream scores 0.0 rather than dividing by zero."""
    assert prequential_accuracy([]) == 0.0
