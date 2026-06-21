"""Ensemble layer — diversity across model FAMILIES, not single-model tuning.

Two complementary combiners live here, both built on the contract's central thesis: the edge is
in *assembling* weakly-correlated pieces, never in over-fitting one piece.

1. :func:`combine_edge_streams` — a thin, typed wrapper that delegates straight to the proven
   return-stream combiner ``startx.portfolio.ensemble.combine`` (max-Sharpe tangency / risk-parity /
   equal weight over resampled P&L streams). We re-expose the per-edge weights so callers get the
   combined stream and the weighting in one call without reaching into the ``EnsembleResult``.

2. :class:`FamilyEnsembleClassifier` — a soft-voting classifier across DIVERSE estimator families
   (a bagged tree ensemble, a gradient-boosted tree ensemble, and a linear model on scaled inputs).
   Each member is calibrated to emit honest probabilities; the ensemble averages those calibrated
   probabilities. Diversity across families — not tuning any single family — is what lifts OOS
   accuracy above the median member. Every estimator is seeded from ``cfg.scanner.seed`` so two fits
   on the same data are bit-for-bit identical.

We import the startx combiner via its submodule path (``startx.portfolio.ensemble``) directly; the
package ``__init__`` re-exports a heavier ``engine`` module, so importing the submodule keeps the
dependency surface minimal even though the package import is also currently clean.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qedge.config import QedgeConfig, get_config

# Direct submodule import: avoids pulling in the package __init__'s heavier `engine` re-exports.
from startx.portfolio.ensemble import combine

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.base import ClassifierMixin

CombineMethod = Literal["max_sharpe", "risk_parity", "equal"]

# Soft-voting averages member probabilities with equal family weight by default. Calibration uses
# a small internal CV; these are structural choices of the diversity design, not tunable thresholds.
_N_CALIBRATION_FOLDS = 3


def combine_edge_streams(
    streams: dict[str, pd.Series],
    *,
    method: CombineMethod = "max_sharpe",
) -> tuple[pd.Series, dict[str, float]]:
    """Combine named edge return ``streams`` into one book via the proven startx combiner.

    Delegates to :func:`startx.portfolio.ensemble.combine` (max-Sharpe tangency by default) and
    re-exposes its outputs as a simple pair.

    Parameters
    ----------
    streams:
        Mapping of edge name -> P&L return :class:`~pandas.Series` (any frequency; the combiner
        resamples to a common rebalance frequency internally).
    method:
        ``"max_sharpe"`` (tangency), ``"risk_parity"`` (inverse-vol), or ``"equal"``.

    Returns
    -------
    tuple
        ``(combined, weights)`` where ``combined`` is the combined return stream (at the combiner's
        rebalance frequency) and ``weights`` maps each edge to its non-negative weight summing to 1.
    """
    result = combine(streams, method=method)
    return result.combined, dict(result.weights)


def _build_members(seed: int) -> list[tuple[str, ClassifierMixin]]:
    """Construct the diverse, deterministically-seeded family estimators.

    Three distinct learning families so their errors decorrelate:

    * ``rf``  — bagged axis-aligned decision trees (low bias, high variance, averaged).
    * ``gbm`` — gradient-boosted trees (sequential bias reduction).
    * ``lr``  — a linear model on standardized features (a smooth, low-variance prior).
    """
    rf: ClassifierMixin = RandomForestClassifier(random_state=seed)
    gbm: ClassifierMixin = LGBMClassifier(random_state=seed, verbose=-1, deterministic=True)
    lr: ClassifierMixin = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
        ],
    )
    return [("rf", rf), ("gbm", gbm), ("lr", lr)]


class FamilyEnsembleClassifier:
    """Soft-voting ensemble across DIVERSE model families with calibrated probabilities.

    Each member is wrapped in :class:`~sklearn.calibration.CalibratedClassifierCV` so it emits
    honest, comparable probabilities; the ensemble prediction is the unweighted average of those
    calibrated probabilities (equal weight per family — diversity, not single-model tuning, is the
    source of lift). Every estimator is seeded from ``cfg.scanner.seed`` for full determinism.
    """

    def __init__(self, config: QedgeConfig | None = None) -> None:
        """Build the ensemble; ``config`` defaults to the process-wide qedge config singleton."""
        self._config = config if config is not None else get_config()
        self._seed = self._config.scanner.seed
        self._members: list[tuple[str, CalibratedClassifierCV]] = [
            (
                name,
                CalibratedClassifierCV(estimator, cv=_N_CALIBRATION_FOLDS, method="sigmoid"),
            )
            for name, estimator in _build_members(self._seed)
        ]
        self.classes_: NDArray[np.int_] | None = None

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.int_],
    ) -> FamilyEnsembleClassifier:
        """Fit every calibrated family member on ``(X, y)``; returns ``self``."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        for _, member in self._members:
            member.fit(X_arr, y_arr)
        # All members share the label space; take it from the first fitted member.
        self.classes_ = np.asarray(self._members[0][1].classes_)
        return self

    def predict_proba(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        """Average calibrated class probabilities across families; rows sum to 1."""
        if self.classes_ is None:
            raise RuntimeError("FamilyEnsembleClassifier must be fitted before predict_proba")
        X_arr = np.asarray(X, dtype=np.float64)
        proba_stack = np.stack([member.predict_proba(X_arr) for _, member in self._members])
        return np.asarray(proba_stack.mean(axis=0), dtype=np.float64)

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.int_]:
        """Predict the class with the highest averaged calibrated probability."""
        if self.classes_ is None:
            raise RuntimeError("FamilyEnsembleClassifier must be fitted before predict")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)])
