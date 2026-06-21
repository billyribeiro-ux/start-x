"""Tests for out-of-sample feature importance (clustered MDA and clustered SHAP).

What these pin down:

* On synthetic data where ``y`` is driven by ONE informative feature plus pure
  noise features, MDA ranks the informative feature (or its cluster) top — and it
  does so on OUT-OF-SAMPLE folds.
* Clustered SHAP agrees: the informative feature/cluster is top.
* A shuffled-``y`` control collapses every importance toward zero / flat — no
  feature can dominate when the labels carry no signal. This is the OOS canary:
  in-sample MDI would still happily "find" structure in shuffled labels.
* Anti-MDI guard: the public functions never touch ``feature_importances_`` and
  remain correct even when that in-sample attribute is poisoned.

LightGBM/scikit-learn/shap are imported lazily via ``importorskip`` so the
numpy-only tiers of the suite still collect on a bare environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qedge.modeling.importance import clustered_shap_oos, mda_importance_oos
from startx.validation.purged_cv import PurgedKFold

_SEED = 7
_N = 600
_N_SPLITS = 5


def _synthetic(
    rng: np.random.Generator, *, signal: bool
) -> tuple[pd.DataFrame, pd.Series]:
    """One informative feature ``x_signal`` plus four noise features.

    When ``signal`` is True the label is a thresholded function of ``x_signal``
    (plus a correlated twin ``x_signal_dup`` to exercise clustering); when False
    the label is independent noise — the shuffled-y control.
    """
    idx = pd.date_range("2019-01-01", periods=_N, freq="B")
    x_signal = rng.standard_normal(_N)
    frame = pd.DataFrame(
        {
            "x_signal": x_signal,
            "x_signal_dup": x_signal + 0.01 * rng.standard_normal(_N),
            "noise0": rng.standard_normal(_N),
            "noise1": rng.standard_normal(_N),
            "noise2": rng.standard_normal(_N),
        },
        index=idx,
    )
    if signal:
        # Label depends only on the informative feature (+ a little label noise).
        logits = 2.5 * x_signal + 0.25 * rng.standard_normal(_N)
        y = pd.Series((logits > 0).astype(int), index=idx, name="y")
    else:
        y = pd.Series(rng.integers(0, 2, size=_N), index=idx, name="y")
    return frame, y


def _t1(index: pd.Index) -> pd.Series:
    """Single-bar label-end times (no overlap) for the purged splitter."""
    return pd.Series(index, index=index)


def _rf_factory() -> object:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=80, max_depth=4, random_state=_SEED, n_jobs=1
    )


def _cv(index: pd.Index) -> PurgedKFold:
    return PurgedKFold(n_splits=_N_SPLITS, t1=_t1(index), embargo_pct=0.0)


# --------------------------------------------------------------------------- #
# MDA — per feature
# --------------------------------------------------------------------------- #
def test_mda_ranks_informative_feature_top_oos() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=True)
    imp = mda_importance_oos(
        _rf_factory, X, y, cv=_cv(X.index), scoring="accuracy", seed=_SEED
    )
    # Per-feature clusters -> index aligns 1:1 with the feature order.
    ranked = imp.sort_values(ascending=False)
    top_feature = X.columns[ranked.index[0]]
    assert top_feature in ("x_signal", "x_signal_dup")
    # The informative feature beats every pure-noise feature.
    noise_ids = [i for i, c in enumerate(X.columns) if c.startswith("noise")]
    signal_ids = [i for i, c in enumerate(X.columns) if c.startswith("x_signal")]
    assert imp.loc[signal_ids].max() > imp.loc[noise_ids].max()


def test_mda_deterministic() -> None:
    pytest.importorskip("sklearn")
    rng_a = np.random.default_rng(_SEED)
    X, y = _synthetic(rng_a, signal=True)
    a = mda_importance_oos(_rf_factory, X, y, cv=_cv(X.index), seed=_SEED)
    b = mda_importance_oos(_rf_factory, X, y, cv=_cv(X.index), seed=_SEED)
    pd.testing.assert_series_equal(a, b)


def test_mda_shuffled_y_is_flat() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=False)
    imp = mda_importance_oos(
        _rf_factory, X, y, cv=_cv(X.index), scoring="accuracy", seed=_SEED
    )
    # With no signal, no feature meaningfully beats the noise floor: the spread of
    # importances is tiny and the max is near zero (permuting noise barely moves
    # an OOS score). This is exactly where in-sample MDI would lie.
    assert imp.abs().max() < 0.05
    assert imp.std() < 0.05


# --------------------------------------------------------------------------- #
# MDA — clustered
# --------------------------------------------------------------------------- #
def test_mda_clustered_ranks_signal_cluster_top() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=True)
    # Cluster the two correlated signal columns together; noise each alone.
    clusters = {
        0: ["x_signal", "x_signal_dup"],
        1: ["noise0"],
        2: ["noise1"],
        3: ["noise2"],
    }
    imp = mda_importance_oos(
        _rf_factory,
        X,
        y,
        cv=_cv(X.index),
        clusters=clusters,
        scoring="accuracy",
        seed=_SEED,
    )
    assert imp.idxmax() == 0  # the signal cluster wins
    assert imp.loc[0] > imp.drop(index=0).max()


# --------------------------------------------------------------------------- #
# Clustered SHAP
# --------------------------------------------------------------------------- #
def test_shap_agrees_with_mda_oos() -> None:
    pytest.importorskip("shap")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=True)
    shap_imp = clustered_shap_oos(_rf_factory, X, y, cv=_cv(X.index), seed=_SEED)
    ranked = shap_imp.sort_values(ascending=False)
    top_feature = X.columns[ranked.index[0]]
    assert top_feature in ("x_signal", "x_signal_dup")
    signal_ids = [i for i, c in enumerate(X.columns) if c.startswith("x_signal")]
    noise_ids = [i for i, c in enumerate(X.columns) if c.startswith("noise")]
    assert shap_imp.loc[signal_ids].max() > shap_imp.loc[noise_ids].max()


def test_shap_clustered_ranks_signal_cluster_top() -> None:
    pytest.importorskip("shap")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=True)
    clusters = {
        0: ["x_signal", "x_signal_dup"],
        1: ["noise0", "noise1", "noise2"],
    }
    imp = clustered_shap_oos(
        _rf_factory, X, y, cv=_cv(X.index), clusters=clusters, seed=_SEED
    )
    assert imp.idxmax() == 0
    assert imp.loc[0] > imp.loc[1]


def test_shap_shuffled_y_is_flat() -> None:
    pytest.importorskip("shap")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=False)
    imp = clustered_shap_oos(_rf_factory, X, y, cv=_cv(X.index), seed=_SEED)
    # No feature dominates: the largest contribution is at most a modest multiple
    # of the smallest (no single feature carries the model the way x_signal does
    # in the signal case, where the ratio is an order of magnitude or more).
    ordered = imp.sort_values(ascending=False).to_numpy()
    assert ordered[0] < 3.0 * ordered[-1] + 1e-9


# --------------------------------------------------------------------------- #
# Anti-MDI guard
# --------------------------------------------------------------------------- #
def test_importance_does_not_use_in_sample_mdi() -> None:
    """Poisoning ``feature_importances_`` must not change the OOS result.

    If either function secretly consulted scikit-learn's in-sample MDI, this
    poisoned attribute would corrupt the ranking. Because importance comes only
    from OOS permutation / OOS SHAP, the result is byte-identical.
    """
    pytest.importorskip("shap")
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    class _MDIPoisonedRF(RandomForestClassifier):  # type: ignore[misc]
        """A forest whose in-sample MDI is forced to lie (all weight on noise2)."""

        @property
        def feature_importances_(self) -> np.ndarray:  # type: ignore[override]
            n = self.n_features_in_
            poisoned = np.zeros(n, dtype=float)
            poisoned[-1] = 1.0  # claim the LAST (noise) feature is everything
            return poisoned

    def _poison_factory() -> object:
        return _MDIPoisonedRF(
            n_estimators=80, max_depth=4, random_state=_SEED, n_jobs=1
        )

    rng = np.random.default_rng(_SEED)
    X, y = _synthetic(rng, signal=True)

    clean = mda_importance_oos(
        _rf_factory, X, y, cv=_cv(X.index), scoring="accuracy", seed=_SEED
    )
    poisoned = mda_importance_oos(
        _poison_factory, X, y, cv=_cv(X.index), scoring="accuracy", seed=_SEED
    )
    # Identical structure / random_state -> the poisoned MDI is simply ignored.
    pd.testing.assert_series_equal(clean, poisoned)
    # And the informative feature still tops the ranking despite the lie.
    signal_ids = [i for i, c in enumerate(X.columns) if c.startswith("x_signal")]
    noise_ids = [i for i, c in enumerate(X.columns) if c.startswith("noise")]
    assert poisoned.loc[signal_ids].max() > poisoned.loc[noise_ids].max()
    assert sklearn is not None
