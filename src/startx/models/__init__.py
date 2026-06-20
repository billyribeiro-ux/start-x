"""Models / self-learning layer for the swing-trading engine.

This subsystem turns PIT features + triple-barrier labels into a leakage-free training
:class:`Dataset`, fits gradient-boosted (and baseline) classifiers, tunes them over purged
cross-validation, explains them, and persists them:

* :mod:`~startx.models.dataset`   — :class:`Dataset`, :func:`build_dataset`, per-symbol builds;
* :mod:`~startx.models.train`     — :func:`make_model`, :func:`make_baseline`, :func:`model_factory`;
* :mod:`~startx.models.tuning`    — :func:`tune` (Optuna over PurgedKFold OOS AUC);
* :mod:`~startx.models.selection` — :func:`importances`, purged permutation importance;
* :mod:`~startx.models.explain`   — :func:`shap_summary` (SHAP with importance fallback);
* :mod:`~startx.models.registry`  — :func:`save_model` / :func:`load_model` / :func:`latest`.
"""
from __future__ import annotations

from .dataset import Dataset, build_dataset, per_symbol_datasets
from .explain import shap_summary
from .registry import latest, load_model, save_model
from .selection import importances, permutation_importance_purged
from .train import baseline_factory, fit_model, make_baseline, make_model, model_factory
from .tuning import tune

__all__ = [
    "Dataset",
    "build_dataset",
    "per_symbol_datasets",
    "make_model",
    "make_baseline",
    "fit_model",
    "model_factory",
    "baseline_factory",
    "tune",
    "importances",
    "permutation_importance_purged",
    "shap_summary",
    "save_model",
    "load_model",
    "latest",
]
