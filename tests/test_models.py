"""Rigorous, fast tests for the MODELS / self-learning layer.

The crown jewel is the MODEL-LEVEL LEAKAGE CANARY: with a feature that genuinely predicts the
label, walk-forward OOS AUC must clear 0.7; when the label is *shuffled* (signal destroyed),
the same pipeline must collapse to ~0.5. A pipeline that scores high on shuffled labels is
leaking. Everything here is synthetic and runs in seconds — no network required, except an
optional, skipped live ``build_dataset`` smoke.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from startx.models import (
    Dataset,
    fit_model,
    importances,
    make_baseline,
    make_model,
    model_factory,
    shap_summary,
    tune,
)
from startx.models.registry import latest, load_model, save_model
from startx.validation.walkforward import walk_forward_predict


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _make_xy(n: int = 600, n_features: int = 8, *, signal: bool = True, seed: int = 0):
    """Return (X, y, t1, w) where, if ``signal``, y depends monotonically on feature 0.

    ``t1`` spans a few bars ahead (overlapping labels, like triple-barrier output) so the
    purged/walk-forward machinery is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    Xv = rng.normal(size=(n, n_features))
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(Xv, columns=cols, index=idx)

    if signal:
        # Logit driven by feature 0 (+ mild feature 1) plus noise -> learnable but not trivial.
        logit = 2.5 * Xv[:, 0] + 0.8 * Xv[:, 1] + rng.normal(scale=0.5, size=n)
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.uniform(size=n) < prob).astype(int), index=idx)
    else:
        y = pd.Series(rng.integers(0, 2, size=n), index=idx)

    span = 4
    end_pos = np.minimum(np.arange(n) + span, n - 1)
    t1 = pd.Series(end_pos.astype(float), index=idx)  # positional t1 (monotone, overlapping)
    w = pd.Series(1.0, index=idx)
    return X, y, t1, w


# --------------------------------------------------------------------------- #
# 1. Dataset dataclass contract
# --------------------------------------------------------------------------- #
def test_dataset_shared_index_and_validation():
    X, y, t1, w = _make_xy(n=50)
    meta = pd.DataFrame(
        {"date": pd.RangeIndex(50), "symbol": "T", "ret": 0.0, "label": 1, "t1": t1.values}
    )
    ds = Dataset(X=X, y=y, t1=t1, w=w, meta=meta)
    assert len(ds) == 50
    assert ds.feature_names == list(X.columns)

    with pytest.raises(ValueError):
        Dataset(X=X, y=y.iloc[:10], t1=t1, w=w, meta=meta)


# --------------------------------------------------------------------------- #
# 2. make_model / make_baseline / fit_model
# --------------------------------------------------------------------------- #
def test_make_model_fits_and_scores():
    X, y, _, w = _make_xy(n=400, signal=True)
    model = fit_model(make_model(n_estimators=120), X, y, w)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert auc > 0.75  # in-sample sanity; OOS is checked by the canary below


def test_baseline_fits_with_weights():
    X, y, _, w = _make_xy(n=300, signal=True)
    base = fit_model(make_baseline(), X, y, w)
    auc = roc_auc_score(y, base.predict_proba(X)[:, 1])
    assert auc > 0.7


# --------------------------------------------------------------------------- #
# 3. MODEL-LEVEL LEAKAGE CANARY (the key test)
# --------------------------------------------------------------------------- #
def test_walk_forward_leakage_canary():
    """OOS AUC must beat 0.7 on real signal AND collapse to ~0.5 on shuffled labels."""
    factory = model_factory({"n_estimators": 150})

    # (a) Learnable signal -> strong OOS AUC.
    X, y, t1, _ = _make_xy(n=800, signal=True, seed=1)
    preds = walk_forward_predict(
        X, y, t1, factory, train_min=300, test_span=100, embargo_pct=0.01, proba=True
    )
    auc_signal = roc_auc_score(preds["y_true"], preds["y_prob"])
    assert auc_signal > 0.7, f"expected strong OOS AUC, got {auc_signal:.3f}"

    # (b) Shuffled labels -> no learnable structure -> AUC near 0.5.
    rng = np.random.default_rng(7)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    preds_shuf = walk_forward_predict(
        X, y_shuf, t1, factory, train_min=300, test_span=100, embargo_pct=0.01, proba=True
    )
    auc_shuf = roc_auc_score(preds_shuf["y_true"], preds_shuf["y_prob"])
    assert 0.40 <= auc_shuf <= 0.60, f"shuffled AUC should be ~0.5, got {auc_shuf:.3f}"


# --------------------------------------------------------------------------- #
# 4. importances
# --------------------------------------------------------------------------- #
def test_importances_sorted_and_signal_on_top():
    X, y, _, w = _make_xy(n=500, signal=True)
    model = fit_model(make_model(n_estimators=200), X, y, w)
    imp = importances(model, X.columns)
    assert isinstance(imp, pd.Series)
    assert list(imp.values) == sorted(imp.values, reverse=True)  # descending
    # The driving feature should carry meaningful gain (top-3 of 8).
    assert "f0" in imp.head(3).index


# --------------------------------------------------------------------------- #
# 5. shap_summary (non-empty, with importance fallback guarantee)
# --------------------------------------------------------------------------- #
def test_shap_summary_non_empty():
    X, y, _, w = _make_xy(n=400, signal=True)
    model = fit_model(make_model(n_estimators=150), X, y, w)
    summary = shap_summary(model, X, max_display=10)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["feature", "mean_abs_shap"]
    assert not summary.empty
    assert len(summary) <= 10
    assert (summary["mean_abs_shap"] >= 0).all()


# --------------------------------------------------------------------------- #
# 6. tune (robust on tiny data; real search on larger)
# --------------------------------------------------------------------------- #
def test_tune_tiny_data_returns_defaults():
    X, y, t1, w = _make_xy(n=60, signal=True)
    assert tune(X, y, t1, w, n_trials=3) == {}  # below _MIN_ROWS -> {}


def test_tune_returns_params_on_signal():
    X, y, t1, w = _make_xy(n=400, signal=True)
    best = tune(X, y, t1, w, n_trials=4, n_splits=3)
    assert isinstance(best, dict)
    assert "learning_rate" in best  # a real search ran and produced params


# --------------------------------------------------------------------------- #
# 7. registry round-trip
# --------------------------------------------------------------------------- #
def test_registry_round_trip(tmp_path):
    X, y, _, w = _make_xy(n=200, signal=True)
    model = fit_model(make_model(n_estimators=80), X, y, w)
    path = save_model(model, {"horizon": "short", "auc": 0.81}, "canary", dir=tmp_path)
    assert path.exists()

    loaded, meta = load_model("canary", dir=tmp_path)
    assert meta["horizon"] == "short"
    assert meta["model_class"] == "LGBMClassifier"
    np.testing.assert_allclose(
        loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1]
    )
    assert latest("canary", dir=tmp_path) == path


# --------------------------------------------------------------------------- #
# 8. Optional live smoke (skipped without an FMP key)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="FMP_API_KEY unset")
def test_live_build_dataset_smoke():
    from startx.data.cache import ParquetCache
    from startx.fmp.client import FMPClient
    from startx.models import build_dataset
    from startx.settings import get_settings

    settings = get_settings()
    cache = ParquetCache(settings.cache_dir)
    client = FMPClient(settings)
    try:
        ds = build_dataset(
            ["AAPL"], "2023-01-01", "2024-01-01", horizon="short",
            client=client, cache=cache, settings=settings, pooled=True,
        )
    finally:
        client.close()

    assert len(ds) > 50
    assert ds.X.select_dtypes(include="number").shape[1] == ds.X.shape[1]  # all numeric
    assert set(ds.y.unique()) <= {0, 1}
    assert list(ds.X.index) == list(range(len(ds)))  # RangeIndex
    assert {"date", "symbol", "ret", "label", "t1"} <= set(ds.meta.columns)
    assert "symbol_code" in ds.X.columns and "sector_code" in ds.X.columns
