"""SHAP-based model explanation with a safe importance fallback.

:func:`shap_summary` returns the mean absolute SHAP value per feature (global importance) using
``shap.TreeExplainer``. SHAP can be brittle across library/model versions, so *any* failure
falls back to native gain importances (:func:`startx.models.selection.importances`) rather than
crashing the pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .selection import importances


def _mean_abs_shap(model: Any, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature via ``shap.TreeExplainer`` (may raise)."""
    import shap  # local import: optional, and keeps import cost off the hot path

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    # Normalize across SHAP versions: list (per-class) or 3-D array (n, features, classes).
    if isinstance(values, list):
        arr = np.abs(values[-1] if len(values) > 1 else values[0])
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = np.abs(arr[:, :, -1])
        else:
            arr = np.abs(arr)

    mean_abs = arr.mean(axis=0)
    return pd.Series(np.asarray(mean_abs, dtype=float), index=list(X.columns))


def shap_summary(model: Any, X: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
    """Global feature importance as a ``DataFrame[feature, mean_abs_shap]`` (descending).

    Uses SHAP when possible; on *any* error falls back to native gain importances (relabeled
    into the same schema) so callers always get a non-empty frame. Returns at most
    ``max_display`` rows.
    """
    try:
        series = _mean_abs_shap(model, X).sort_values(ascending=False)
        out = series.rename("mean_abs_shap").reset_index()
        out.columns = ["feature", "mean_abs_shap"]
    except Exception:
        # Fallback: native gain importances, mapped into the SHAP schema.
        gain = importances(model, list(X.columns))
        out = gain.reset_index()
        out.columns = ["feature", "mean_abs_shap"]

    return out.head(max_display).reset_index(drop=True)
