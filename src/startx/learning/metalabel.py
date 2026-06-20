"""Walk-forward meta-labeling with a strict point-in-time training firewall.

A *meta-label* is a second-stage model that decides take/skip on the primary strategy's trades:
given a candidate trade's pre-trade forensics, predict P(win) and act only when it is high enough.
Done naively this is a leakage trap — and this file's whole reason to exist is to NOT leak.

THE FIREWALL (read this before touching :func:`walk_forward_metalabel`):
  The meta-model that decides take/skip for trade ``i`` may train ONLY on trades that fully
  **resolved before trade i was entered** — i.e. training trades whose label-end time
  ``t1 <= entry_time_i``. A trade that was still open (or hadn't even started) when we entered
  ``i`` had no known outcome at decision time, so using it to score ``i`` is hindsight.

  This is NOT the same as a positional / "first 70% of rows" split. Two trades can be ordered by
  entry time yet the earlier-entered one can resolve LATER (a slow, wide-stop trade entered on
  Monday that closes in three weeks vs. a scalp entered Tuesday that closes Tuesday). A positional
  split would happily train on that slow trade when scoring the scalp — even though, at the
  scalp's entry, the slow trade's win/loss was still unknown. We gate on ``t1`` precisely to
  forbid that. See the per-block gate below, which is commented step by step.

API:
  * :func:`walk_forward_metalabel` — produce OOS ``meta_prob_win`` for every trade, training each
    forward block under the ``t1 <= entry`` gate.
  * :func:`apply_meta_filter` — keep only trades whose ``meta_prob_win >= threshold``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def _predict_proba_win(model: Any, X_test: np.ndarray) -> np.ndarray:
    """Return P(win) for ``X_test``, locating the positive (label==1) column robustly.

    Falls back to ``decision_function`` / ``predict`` for estimators without ``predict_proba``.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_test))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                # Column whose class label is 1 (the "win" class); argmax handles {0,1}.
                pos = int(np.argmax(np.asarray(classes)))
                return proba[:, pos]
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test)).ravel()
    return np.asarray(model.predict(X_test), dtype=float).ravel()


def walk_forward_metalabel(
    X: pd.DataFrame,
    y: pd.Series,
    entry_times: pd.Series,
    t1: pd.Series,
    model_factory: Callable[[], Any],
    *,
    train_min: int,
    test_span: int,
) -> pd.DataFrame:
    """Strictly point-in-time walk-forward meta-label predictions.

    For each forward test block of (up to) ``test_span`` trades, the training set is EVERY trade
    whose label-end ``t1 <= the first entry_time of the test block`` — that is, every trade that
    had fully resolved strictly before the earliest decision in the block — provided there are at
    least ``train_min`` of them. A fresh ``model_factory()`` is fit on those resolved trades and
    used to predict P(win) for the whole block.

    Parameters
    ----------
    X:
        Forensic feature matrix (one row per trade). Non-numeric columns are coerced; NaNs are
        passed straight to the estimator (LightGBM handles them natively).
    y:
        Binary win label (1 == win) aligned to ``X``.
    entry_times:
        Decision time of each trade (when we must decide take/skip). Aligned to ``X``.
    t1:
        Label-end time of each trade (when its win/loss becomes known). Aligned to ``X``. This is
        the firewall key — training is gated on ``t1 <= entry`` of the block being scored.
    model_factory:
        Zero-arg callable returning a fresh estimator each block (fit/predict; predict_proba
        optional) — so no state leaks across the train/test boundary.
    train_min, test_span:
        Minimum resolved-trade count before a block may be scored, and block size.

    Returns
    -------
    pd.DataFrame
        Columns ``[entry_time, meta_prob_win, y_true, y_pred]``, one row per trade that received
        an OOS prediction (a leading warm-up block has no eligible training set and is omitted),
        sorted by ``entry_time``. The original trade index is preserved so callers can align the
        predictions back onto their trades frame.
    """
    if train_min < 1 or test_span < 1:
        raise ValueError("train_min and test_span must be >= 1")

    # Align everything on the X index, then order strictly by entry time. Trades are decided in
    # entry-time order; the firewall reasons in that same timeline.
    idx = X.index
    y = pd.Series(np.asarray(y), index=idx) if not isinstance(y, pd.Series) else y.reindex(idx)
    entry_times = pd.to_datetime(pd.Series(np.asarray(entry_times), index=idx))
    t1 = pd.to_datetime(pd.Series(np.asarray(t1), index=idx))

    order = entry_times.sort_values(kind="mergesort").index
    X_ord = X.loc[order]
    y_ord = y.loc[order]
    entry_ord = entry_times.loc[order]
    t1_ord = t1.loc[order]

    X_arr = X_ord.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_arr = y_ord.to_numpy()
    entry_arr = entry_ord.to_numpy()  # datetime64[ns], ascending
    t1_arr = t1_ord.to_numpy()
    n = len(order)

    rec_index: list = []
    rec_entry: list = []
    rec_prob: list = []
    rec_true: list = []
    rec_pred: list = []

    test_start = 0
    while test_start < n:
        test_end = min(test_start + test_span, n)
        block_entry = entry_arr[test_start]  # earliest decision time in this test block

        # ---- THE POINT-IN-TIME FIREWALL ----------------------------------------------------
        # Eligible training trades = those that fully RESOLVED at or before this block's first
        # entry, i.e. t1 <= block_entry. We gate on t1 (NOT on position): an earlier-entered but
        # late-resolving trade is correctly EXCLUDED because its outcome was unknown at decision
        # time. We scan the whole prefix [0, test_start) — entry-ordered, but a trade's t1 can
        # exceed block_entry even though its entry precedes it, so position alone is insufficient.
        prefix = np.arange(0, test_start)
        resolved = prefix[t1_arr[prefix] <= block_entry] if prefix.size else prefix
        # ------------------------------------------------------------------------------------

        if resolved.size < train_min:
            # Not enough fully-resolved history yet to honestly train — skip scoring this block.
            test_start = test_end
            continue

        y_tr = y_arr[resolved]
        valid_tr = ~_isnan(y_tr)
        # A classifier needs >= 2 classes among resolved trades; skip degenerate windows.
        if np.unique(y_tr[valid_tr]).size < 2:
            test_start = test_end
            continue

        model = model_factory()
        model.fit(X_arr[resolved], y_tr)

        te = np.arange(test_start, test_end)
        probs = _predict_proba_win(model, X_arr[te])
        preds = np.asarray(model.predict(X_arr[te])).ravel()
        for k, pos in enumerate(te):
            rec_index.append(order[pos])
            rec_entry.append(entry_arr[pos])
            rec_prob.append(float(probs[k]))
            rec_true.append(y_arr[pos])
            rec_pred.append(float(preds[k]))

        test_start = test_end

    out = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(rec_entry),
            "meta_prob_win": rec_prob,
            "y_true": rec_true,
            "y_pred": rec_pred,
        },
        index=pd.Index(rec_index, name=X.index.name),
    )
    if not out.empty:
        out = out.sort_values("entry_time", kind="mergesort")
    return out


def apply_meta_filter(
    trades: pd.DataFrame, meta_prob_win: pd.Series, threshold: float = 0.5
) -> pd.DataFrame:
    """Keep only trades the meta-model would TAKE — those with ``meta_prob_win >= threshold``.

    ``meta_prob_win`` is aligned to ``trades`` by index. Trades without a meta prediction (e.g.
    the leading warm-up block that had no eligible training set) are dropped — we never had a
    point-in-time-clean opinion on them, so we cannot claim to have taken or skipped them.
    """
    probs = pd.Series(meta_prob_win).reindex(trades.index)
    keep = probs >= threshold
    keep = keep.fillna(False)
    return trades.loc[keep].copy()


def _isnan(a: np.ndarray) -> np.ndarray:
    """NaN mask safe for non-float (object/int) label arrays."""
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.zeros(len(a), dtype=bool)
