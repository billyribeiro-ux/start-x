"""The self-learning driver: run the walk-forward meta-label, filter, and measure the gain.

This is where the loss-autopsy pays off. We take an autopsied trade ledger (entry-day forensics
+ ``is_win`` label), run the strictly point-in-time :func:`walk_forward_metalabel`, drop the
setups that look like past losers, and compare the FILTERED trade economics against the
UNFILTERED ones. If the meta-label has learned the fingerprint of losing trades, the filtered
book has a higher win rate / expectancy / profit factor — that gap is the loss-avoidance value.

It also exposes :func:`drift_monitor`: a rolling out-of-sample accuracy / win-rate that flags
when the learned fingerprint is decaying (regime change) — the self-adjusting retrain trigger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.train import model_factory
from ..validation.metrics import expectancy, hit_rate, profit_factor
from .autopsy import win_loss_signature
from .metalabel import apply_meta_filter, walk_forward_metalabel


def trade_metrics(rets: pd.Series | np.ndarray) -> dict:
    """Economics of a set of trade returns.

    Returns ``{n_trades, win_rate, expectancy, profit_factor, avg_ret, total_return}``. Empty
    input yields zeros / NaNs so filtered-vs-unfiltered comparisons never raise.
    """
    r = pd.Series(rets, dtype=float).dropna()
    if r.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "profit_factor": float("nan"),
            "avg_ret": float("nan"),
            "total_return": 0.0,
        }
    return {
        "n_trades": int(r.size),
        "win_rate": float(hit_rate(r)),
        "expectancy": float(expectancy(r)),
        "profit_factor": float(profit_factor(r)),
        "avg_ret": float(r.mean()),
        # Compounded total return over the (filtered) book — what the equity curve actually does.
        "total_return": float(np.prod(1.0 + r.to_numpy()) - 1.0),
    }


@dataclass
class SelfLearningResult:
    """Outcome of one self-learning pass.

    Attributes
    ----------
    unfiltered, filtered:
        :func:`trade_metrics` dicts for, respectively, every trade that received an OOS meta
        prediction and only the trades the meta-label would have TAKEN.
    signature:
        The :func:`win_loss_signature` table — the automated "why losses happen".
    meta_auc:
        Out-of-sample AUC of ``meta_prob_win`` vs ``y_true``. ~0.5 == the meta-label learned
        nothing (no exploitable loser fingerprint); high (>~0.7) == it discriminates winners.
    n_kept, n_dropped:
        How many predicted trades survived / were filtered by the meta gate.
    """

    unfiltered: dict
    filtered: dict
    signature: pd.DataFrame
    meta_auc: float
    n_kept: int
    n_dropped: int


def _oos_auc(prob: np.ndarray, y_true: np.ndarray) -> float:
    """Rank-based AUC of P(win) vs the realised win label (ties == 0.5). NaN-safe."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y_true, dtype=float)
    valid = ~np.isnan(prob) & ~np.isnan(y)
    prob, y = prob[valid], y[valid] > 0
    pos, neg = prob[y], prob[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(prob, kind="mergesort")
    ranks = np.empty(prob.size, dtype=float)
    ranks[order] = np.arange(1, prob.size + 1, dtype=float)
    s = np.sort(prob)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            mask = prob == s[i]
            ranks[mask] = ranks[mask].mean()
        i = j + 1
    return float((ranks[y].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def run_self_learning(
    autopsy: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold: float = 0.5,
    train_min: int = 50,
    test_span: int = 20,
    factory=None,
) -> SelfLearningResult:
    """Run the full self-learning loop on an autopsied trade ledger.

    Builds ``X`` from ``feature_cols`` (plus any carried primary-model probability column, which
    is a legitimate pre-trade signal), uses ``is_win`` as the label, runs the point-in-time
    :func:`walk_forward_metalabel`, applies :func:`apply_meta_filter`, and compares FILTERED vs
    UNFILTERED trade economics — demonstrating the loss-avoidance value of the meta gate.

    ``autopsy`` must contain ``entry_date``, ``ret_net`` and ``is_win`` (from
    :func:`startx.learning.autopsy.autopsy_trades`). A ``t1`` (label-end) column is used as the
    leakage gate; if absent it falls back to ``exit_date``, and if that is also absent to
    ``entry_date`` (degenerate same-day resolution).

    Returns a :class:`SelfLearningResult`.
    """
    signature = win_loss_signature(autopsy)
    empty = trade_metrics(pd.Series(dtype=float))
    if autopsy is None or autopsy.empty:
        return SelfLearningResult(empty, empty, signature, float("nan"), 0, 0)

    factory = factory or model_factory()

    # Primary-model probability (if the strategy produced one) is a valid pre-trade feature.
    cols = list(feature_cols)
    for extra in ("primary_prob", "prob_win", "p_primary"):
        if extra in autopsy.columns and extra not in cols:
            cols.append(extra)
    cols = [c for c in cols if c in autopsy.columns]

    X = autopsy[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(autopsy["is_win"], errors="coerce")
    entry_times = pd.to_datetime(autopsy["entry_date"])
    # t1 == when the trade's win/loss is KNOWN. Prefer an explicit t1; else exit_date; else entry.
    if "t1" in autopsy.columns:
        t1 = pd.to_datetime(autopsy["t1"])
    elif "exit_date" in autopsy.columns:
        t1 = pd.to_datetime(autopsy["exit_date"])
    else:
        t1 = entry_times

    preds = walk_forward_metalabel(
        X, y, entry_times, t1, factory, train_min=train_min, test_span=test_span
    )
    if preds.empty:
        return SelfLearningResult(empty, empty, signature, float("nan"), 0, 0)

    # Only trades that received a point-in-time-clean prediction count toward the comparison —
    # the unfiltered baseline is "take every predicted trade", the filtered book applies the gate.
    predicted = autopsy.loc[preds.index]
    meta_prob = preds["meta_prob_win"]
    kept = apply_meta_filter(predicted, meta_prob, threshold=threshold)

    unfiltered = trade_metrics(pd.to_numeric(predicted["ret_net"], errors="coerce"))
    filtered = trade_metrics(pd.to_numeric(kept["ret_net"], errors="coerce"))
    meta_auc = _oos_auc(preds["meta_prob_win"].to_numpy(), preds["y_true"].to_numpy())

    return SelfLearningResult(
        unfiltered=unfiltered,
        filtered=filtered,
        signature=signature,
        meta_auc=meta_auc,
        n_kept=int(len(kept)),
        n_dropped=int(len(predicted) - len(kept)),
    )


def drift_monitor(meta_results: pd.DataFrame, window: int = 50) -> pd.Series:
    """Rolling out-of-sample meta accuracy — the self-adjusting retrain trigger.

    ``meta_results`` is the frame from :func:`walk_forward_metalabel` (``y_true``, ``y_pred``,
    ``entry_time``). We order by ``entry_time`` and return a rolling-``window`` mean of the OOS
    hit indicator (``y_pred == y_true``): the fraction of recent take/skip calls the meta-label
    got right. A sustained decay below chance flags a regime change in the loser fingerprint and
    is the cue to retrain (which, by construction, the walk-forward already does block by block).
    """
    if meta_results is None or meta_results.empty:
        return pd.Series(dtype=float)
    df = meta_results
    if "entry_time" in df.columns:
        df = df.sort_values("entry_time", kind="mergesort")
    correct = (pd.to_numeric(df["y_pred"], errors="coerce")
               == pd.to_numeric(df["y_true"], errors="coerce")).astype(float)
    roll = correct.rolling(window=window, min_periods=max(1, window // 2)).mean()
    if "entry_time" in df.columns:
        roll.index = pd.to_datetime(df["entry_time"]).to_numpy()
    return roll
