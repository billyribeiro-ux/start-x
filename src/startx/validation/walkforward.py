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
        ``t1`` is the *end* of each sample's label interval; the sample's *start*
        is its own entry time. In addition to the positional embargo, every train
        row whose label interval ``[entry, t1]`` overlaps the test block's interval
        is **purged** so multi-bar labels (e.g. 63d/252d triple-barrier) cannot
        bleed across the boundary.
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

    # Label interval bounds for purging, positionally aligned to `index`.
    # `starts` = each sample's entry time, `ends` = its label-end time (t1).
    # Prefer real entry timestamps (the index, when it carries time) so multi-bar
    # overlap is measured in time, not position; fall back to positions otherwise.
    starts, ends = _label_bounds(index, t1, n)

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

        # PURGE label-span overlap: drop train rows whose label interval
        # [entry, t1] overlaps the test block's interval. This is what the
        # positional embargo alone misses for multi-bar labels — the last ~span
        # train labels otherwise close *inside* the test window and leak.
        tr_pos = np.arange(0, train_end)
        test_lo = starts[test_start:test_end].min()
        test_hi = ends[test_start:test_end].max()
        overlap = (starts[tr_pos] <= test_hi) & (ends[tr_pos] >= test_lo)
        tr_pos = tr_pos[~overlap]
        if tr_pos.size < 1:
            test_start = test_end
            continue
        te = slice(test_start, test_end)

        y_tr = y_arr[tr_pos]
        # A classifier needs >= 2 classes; skip degenerate folds gracefully.
        if np.unique(y_tr[~_isnan(y_tr)]).size < 2:
            test_start = test_end
            continue

        model = model_factory()
        model.fit(X_arr[tr_pos], y_tr)
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


def _is_datetime_like(values: np.ndarray) -> bool:
    """True if ``values`` is a datetime64 array (real timestamps)."""
    return np.issubdtype(np.asarray(values).dtype, np.datetime64)


def _label_bounds(
    index: pd.Index, t1: pd.Series | None, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return positionally-aligned ``(starts, ends)`` label-interval bounds.

    ``starts`` is each sample's entry time, ``ends`` its label-end time (t1),
    both as comparable values aligned to row position. Overlap is measured in
    *time* whenever real timestamps are available — either on the index (the
    DatetimeIndex shape) or on ``t1``'s own values + entry index (the production
    Dataset shape, where ``X``/``t1`` carry a RangeIndex but ``t1`` is indexed by
    entry date). When no timestamps are available, or ``t1`` is unusable, every
    label collapses to a single positional point so a single-bar label never
    spuriously overlaps and the purge degrades to a no-op rather than misfiring.
    NaN/NaT ``t1`` entries collapse to the sample's own start (conservative).
    """
    idx_vals = index.to_numpy()
    t1_vals = (
        t1.to_numpy() if isinstance(t1, pd.Series)
        else (np.asarray(t1) if t1 is not None else None)
    )
    t1_index_vals = np.asarray(t1.index) if isinstance(t1, pd.Series) else None

    # Case A — the index itself carries entry timestamps (DatetimeIndex shape).
    if _is_datetime_like(idx_vals):
        starts = idx_vals
        if t1_vals is not None and len(t1_vals) == n and _is_datetime_like(t1_vals):
            ends = pd.Series(t1_vals).fillna(pd.Series(starts)).to_numpy()
            return starts, ends
        return starts, starts.copy()

    # Case B — positional index, but t1 carries time on its values AND its index
    # (production Dataset: t1 reindexed by entry date). Overlap in real time.
    if (
        t1_vals is not None and len(t1_vals) == n and _is_datetime_like(t1_vals)
        and t1_index_vals is not None and _is_datetime_like(t1_index_vals)
    ):
        starts = t1_index_vals
        ends = pd.Series(t1_vals).fillna(pd.Series(starts)).to_numpy()
        return starts, ends

    # Case C — no usable timestamps anywhere: single-point positional labels.
    pos = np.arange(n)
    return pos, pos.copy()


def _isnan(a: np.ndarray) -> np.ndarray:
    """NaN mask that is safe for non-float (object/int) arrays."""
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.zeros(len(a), dtype=bool)
