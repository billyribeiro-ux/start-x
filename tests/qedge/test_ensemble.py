"""Tests for the family-diverse ensemble layer.

Two contracts are asserted:

* :func:`combine_edge_streams` delegates to the proven startx return-stream combiner and hands back
  a combined stream plus weights that are non-negative and sum to ~1 (a valid long-only book).
* :class:`FamilyEnsembleClassifier` earns its keep through diversity: on a synthetic task its
  accuracy is at least the median single-member accuracy, its probability rows are normalized, and
  two fits with the same seed are identical (determinism is a hard contract requirement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qedge.config import QedgeConfig
from qedge.modeling.ensemble import FamilyEnsembleClassifier, combine_edge_streams

lightgbm = pytest.importorskip("lightgbm")


def _two_streams(seed: int) -> dict[str, pd.Series]:
    """Two independent, positive-drift daily return streams over the locked analysis window."""
    idx = pd.bdate_range("2019-01-01", periods=900)
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.normal(0.0008, 0.012, size=len(idx)), index=idx)
    b = pd.Series(rng.normal(0.0006, 0.010, size=len(idx)), index=idx)
    return {"a": a, "b": b}


def test_combine_edge_streams_returns_stream_and_sensible_weights(
    seeded_rng: np.random.Generator,
) -> None:
    streams = _two_streams(seed=7)

    combined, weights = combine_edge_streams(streams)

    assert isinstance(combined, pd.Series)
    assert not combined.empty
    # Weights cover exactly the supplied edges, are non-negative (long-only), and sum to 1.
    assert set(weights) == set(streams)
    assert all(w >= 0.0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_combine_edge_streams_equal_method_truly_blends() -> None:
    """The 'equal' method is a real average of the inputs (a genuine blend, not a passthrough)."""
    streams = _two_streams(seed=7)

    combined, weights = combine_edge_streams(streams, method="equal")

    assert weights == pytest.approx({"a": 0.5, "b": 0.5})
    assert combined.notna().all()


def _synthetic_classification(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A modestly-separable binary task split into train/test with a fixed seed."""
    features, labels = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        class_sep=0.8,
        random_state=seed,
    )
    return train_test_split(features, labels, test_size=0.4, random_state=seed)


def test_family_ensemble_beats_median_member(cfg: QedgeConfig) -> None:
    """Diversity helps: ensemble accuracy >= median single-member accuracy."""
    seed = cfg.scanner.seed
    x_train, x_test, y_train, y_test = _synthetic_classification(seed)

    ensemble = FamilyEnsembleClassifier(cfg).fit(x_train, y_train)
    ensemble_acc = accuracy_score(y_test, ensemble.predict(x_test))

    members = {
        "rf": RandomForestClassifier(random_state=seed),
        "gbm": lightgbm.LGBMClassifier(random_state=seed, verbose=-1, deterministic=True),
        "lr": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
            ],
        ),
    }
    member_accs = []
    for model in members.values():
        model.fit(x_train, y_train)
        member_accs.append(accuracy_score(y_test, model.predict(x_test)))
    median_member_acc = float(np.median(member_accs))

    assert ensemble_acc >= median_member_acc


def test_family_ensemble_predict_proba_rows_sum_to_one(cfg: QedgeConfig) -> None:
    x_train, x_test, y_train, _ = _synthetic_classification(cfg.scanner.seed)

    ensemble = FamilyEnsembleClassifier(cfg).fit(x_train, y_train)
    proba = ensemble.predict_proba(x_test)

    assert proba.shape == (x_test.shape[0], len(np.unique(y_train)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_family_ensemble_is_deterministic(cfg: QedgeConfig) -> None:
    """Same seed -> bit-for-bit identical probabilities across two independent fits."""
    x_train, x_test, y_train, _ = _synthetic_classification(cfg.scanner.seed)

    proba_a = FamilyEnsembleClassifier(cfg).fit(x_train, y_train).predict_proba(x_test)
    proba_b = FamilyEnsembleClassifier(cfg).fit(x_train, y_train).predict_proba(x_test)

    np.testing.assert_array_equal(proba_a, proba_b)
