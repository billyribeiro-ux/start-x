"""Anchored (expanding-window) walk-forward out-of-sample prediction.

Walk-forward is the most honest backtest harness: train on the past, predict a
contiguous future block, advance, repeat. The window is *anchored* (expanding):
every fold trains on all data from the start up to the embargo boundary before
the test block. An embargo gap of ``embargo_pct`` of the dataset is inserted
between the train end and the test start to neutralise serial-correlation
leakage across the boundary (cf. purged CV).

Only out-of-sample predictions are returned — there is no peeking.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def _predict_proba(model: Any, X_test) -> np.ndarray:
    """Return P(class==1) if available, else fall back to ``predict``."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_test))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                # Locate the positive class column (label 1 / max label).
                pos = np.argmax(classes)
                return proba[:, pos]
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test)).ravel()
    return np.asarray(model.predict(X_test), dtype=float).ravel()


def walk_forward_predict(
    X,
    y: pd.Series,
    t1: pd.Series,
    model_factory: Callable[[], Any],
    *,
    train_min: int,
    test_span: int,
    embargo_pct: float = 0.01,
    proba: bool = True,
) -> pd.DataFrame:
    """Expanding-window walk-forward OOS predictions.

    Parameters
    ----------
    X, y, t1:
        Features, labels and label-end times. All share one time-ordered index.
        ``t1`` is accepted for API consistency and embargo sizing; the embargo
        gap is applied positionally between train end and test start.
    model_factory:
        Zero-arg callable returning a *fresh* sklearn-like estimator each fold
        (must implement ``fit`` and ``predict``; ``predict_proba`` optional).
    train_min:
        Minimum number of training samples before the first test block.
    test_span:
        Number of samples in each contiguous test block.
    embargo_pct:
        Embargo gap between train end and test start, as a fraction of dataset
        length (rounded down, minimum 0).
    proba:
        If ``True`` populate ``y_prob`` with the positive-class probability
        (or decision score); otherwise ``y_prob`` mirrors the hard prediction.

    Returns
    -------
    pd.DataFrame
        Indexed by the OOS sample index with columns
        ``["y_true", "y_prob", "y_pred"]`` — stitched out-of-sample only.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))
    index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else y.index
    n = len(index)
    if len(y) != n:
        raise ValueError("X and y must have the same length")
    if train_min < 1 or test_span < 1:
        raise ValueError("train_min and test_span must be >= 1")

    embargo = int(n * embargo_pct)
    X_arr = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = y.to_numpy()

    rows_idx: list[int] = []
    y_true: list[float] = []
    y_prob: list[float] = []
    y_pred: list[float] = []

    test_start = train_min + embargo
    while test_start < n:
        test_end = min(test_start + test_span, n)
        train_end = test_start - embargo  # exclusive; embargo gap before test
        if train_end < 1:
            test_start = test_end
            continue
        tr = slice(0, train_end)
        te = slice(test_start, test_end)

        y_tr = y_arr[tr]
        # A classifier needs >= 2 classes; skip degenerate folds gracefully.
        if np.unique(y_tr[~_isnan(y_tr)]).size < 2:
            test_start = test_end
            continue

        model = model_factory()
        model.fit(X_arr[tr], y_tr)
        preds = np.asarray(model.predict(X_arr[te])).ravel()
        if proba:
            scores = _predict_proba(model, X_arr[te])
        else:
            scores = preds.astype(float)

        for j, pos in enumerate(range(test_start, test_end)):
            rows_idx.append(pos)
            y_true.append(y_arr[pos])
            y_prob.append(float(scores[j]))
            y_pred.append(float(preds[j]))

        test_start = test_end

    out = pd.DataFrame(
        {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred},
        index=index[rows_idx] if rows_idx else index[:0],
    )
    return out


def _isnan(a: np.ndarray) -> np.ndarray:
    """NaN mask that is safe for non-float (object/int) arrays."""
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.zeros(len(a), dtype=bool)
