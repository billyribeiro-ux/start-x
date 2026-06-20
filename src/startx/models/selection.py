"""Feature importance / selection helpers.

:func:`importances` reads the model's native gain importances (fast, no extra data).
:func:`permutation_importance_purged` offers a heavier, leakage-aware permutation score that
shuffles one feature at a time on a held-out, purge-respecting test set and measures the drop
in ROC-AUC. It is kept light (single split, capped repeats) and is opt-in.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..validation.purged_cv import PurgedKFold


def importances(model: Any, feature_names: list[str] | pd.Index) -> pd.Series:
    """Native (gain) feature importances as a descending-sorted Series.

    Works with LightGBM (``booster_.feature_importance("gain")``) and any estimator exposing
    ``feature_importances_``. Falls back to a zero Series if neither is available.
    """
    names = list(feature_names)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        gain = booster.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        gain = np.asarray(model.feature_importances_, dtype=float)
    else:
        gain = np.zeros(len(names), dtype=float)

    if len(gain) != len(names):
        # Defensive: align lengths so we never raise on a surprising model surface.
        m = min(len(gain), len(names))
        gain, names = gain[:m], names[:m]

    return pd.Series(gain, index=names, name="gain").sort_values(ascending=False)


def permutation_importance_purged(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    w: pd.Series | None = None,
    *,
    n_splits: int = 3,
    embargo_pct: float = 0.01,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.Series:
    """Purged permutation importance: mean AUC drop when each feature is shuffled.

    Trains a fresh model on each purged training fold, then on the matching test fold measures
    the decrease in ROC-AUC caused by independently permuting each feature ``n_repeats`` times.
    Returns a descending-sorted Series (higher = more important). Light by design.
    """
    rng = np.random.default_rng(seed)
    names = list(X.columns)
    X_arr = X.to_numpy()
    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=float) if w is not None else None

    drops: dict[str, list[float]] = {c: [] for c in names}
    splitter = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    for tr, te in splitter.split(X):
        if np.unique(y_arr[tr]).size < 2 or np.unique(y_arr[te]).size < 2:
            continue
        model = model_factory()
        if w_arr is not None:
            model.fit(X_arr[tr], y_arr[tr], sample_weight=w_arr[tr])
        else:
            model.fit(X_arr[tr], y_arr[tr])

        base = roc_auc_score(y_arr[te], model.predict_proba(X_arr[te])[:, 1])
        X_te = X_arr[te].copy()
        for j, col in enumerate(names):
            col_drops: list[float] = []
            saved = X_te[:, j].copy()
            for _ in range(n_repeats):
                X_te[:, j] = rng.permutation(saved)
                auc = roc_auc_score(y_arr[te], model.predict_proba(X_te)[:, 1])
                col_drops.append(base - auc)
            X_te[:, j] = saved
            drops[col].append(float(np.mean(col_drops)))

    agg = {c: (float(np.mean(v)) if v else 0.0) for c, v in drops.items()}
    return pd.Series(agg, name="perm_auc_drop").sort_values(ascending=False)
