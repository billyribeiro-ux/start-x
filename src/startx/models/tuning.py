"""Leakage-free hyper-parameter tuning via Optuna over purged cross-validation.

:func:`tune` maximizes the mean out-of-sample ROC-AUC across
:class:`startx.validation.purged_cv.PurgedKFold` folds. Because scoring happens only on purged,
embargoed test folds, the search itself cannot reward look-ahead leakage. On tiny datasets
(``< _MIN_ROWS`` rows, or fewer than two classes) tuning is skipped and the swing defaults are
returned, so the pipeline never crashes on thin data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..validation.purged_cv import PurgedKFold
from .train import make_model

#: Below this many rows tuning is unreliable; return defaults instead.
_MIN_ROWS = 200


def _suggest(trial: optuna.Trial) -> dict[str, Any]:
    """Sample a LightGBM hyper-parameter configuration for one trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _cv_auc(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    w: pd.Series | None,
    *,
    n_splits: int,
    embargo_pct: float,
    seed: int,
) -> float:
    """Mean OOS ROC-AUC of ``params`` across purged folds (NaN if no scorable fold)."""
    splitter = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    X_arr = X.to_numpy()
    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=float) if w is not None else None

    scores: list[float] = []
    for tr, te in splitter.split(X):
        if len(tr) == 0 or len(te) == 0:
            continue
        if np.unique(y_arr[tr]).size < 2 or np.unique(y_arr[te]).size < 2:
            continue  # AUC undefined without both classes
        model = make_model(random_state=seed, **params)
        if w_arr is not None:
            model.fit(X_arr[tr], y_arr[tr], sample_weight=w_arr[tr])
        else:
            model.fit(X_arr[tr], y_arr[tr])
        prob = model.predict_proba(X_arr[te])[:, 1]
        scores.append(roc_auc_score(y_arr[te], prob))

    return float(np.mean(scores)) if scores else float("nan")


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    w: pd.Series | None = None,
    *,
    n_trials: int = 30,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Return best LightGBM ``params`` maximizing mean purged-CV OOS ROC-AUC.

    Robust to tiny data: with ``< _MIN_ROWS`` rows or a single class present, returns ``{}``
    (callers fall back to :func:`startx.models.train.make_model` defaults). Optuna logging is
    suppressed.
    """
    if len(X) < _MIN_ROWS or pd.Series(y).nunique() < 2:
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        score = _cv_auc(
            params, X, y, t1, w,
            n_splits=n_splits, embargo_pct=embargo_pct, seed=seed,
        )
        if not np.isfinite(score):
            # No scorable fold for this config: discourage without crashing the study.
            raise optuna.TrialPruned()
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        return dict(study.best_params)
    except ValueError:
        # All trials pruned (degenerate data): fall back to defaults.
        return {}
