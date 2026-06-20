"""Model constructors and fitting helpers for the directional swing classifier.

Two estimators are offered:

* :func:`make_model` — a gradient-boosted ``LGBMClassifier`` with swing-trading defaults;
* :func:`make_baseline` — a ``StandardScaler`` + ``LogisticRegression`` pipeline (sanity floor).

:func:`model_factory` returns the zero-arg callable that
:func:`startx.validation.walkforward.walk_forward_predict` and
:class:`startx.validation.purged_cv.PurgedKFold` consume — a *fresh* estimator per fold so no
state ever leaks across the train/test boundary.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: LightGBM defaults tuned for noisy, modestly-sized daily swing datasets: shallow-ish trees,
#: slow learning, strong subsampling + L2 to fight overfitting on overlapping labels.
_LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": -1,
}


def make_model(**overrides: Any) -> lgb.LGBMClassifier:
    """A binary ``LGBMClassifier`` with swing defaults; ``overrides`` win over defaults."""
    params = {**_LGBM_DEFAULTS, **overrides}
    return lgb.LGBMClassifier(**params)


def make_baseline() -> Pipeline:
    """Standardized logistic-regression baseline (linear sanity floor)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def fit_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    w: pd.Series | np.ndarray | None = None,
) -> Any:
    """Fit ``model`` on ``(X, y)`` with optional sample weights ``w``; returns ``model``.

    Works for both the LightGBM classifier and the sklearn ``Pipeline`` baseline (the
    pipeline routes ``sample_weight`` to its final ``clf`` step).
    """
    if w is None:
        model.fit(X, y)
        return model

    sw = np.asarray(w, dtype=float)
    if isinstance(model, Pipeline):
        final = model.steps[-1][0]
        model.fit(X, y, **{f"{final}__sample_weight": sw})
    else:
        model.fit(X, y, sample_weight=sw)
    return model


def model_factory(params: dict | None = None) -> Callable[[], lgb.LGBMClassifier]:
    """Return a zero-arg factory producing fresh ``LGBMClassifier`` instances.

    Plug directly into :func:`walk_forward_predict` / purged-CV loops. ``params`` (e.g. the
    output of :func:`startx.models.tuning.tune`) override the swing defaults.
    """
    overrides = dict(params or {})

    def factory() -> lgb.LGBMClassifier:
        return make_model(**overrides)

    return factory


def baseline_factory() -> Callable[[], Pipeline]:
    """Zero-arg factory for the logistic-regression baseline (for walk-forward comparisons)."""
    return make_baseline
