"""Out-of-sample feature importance — clustered MDA and clustered SHAP.

THE CONTRACT (non-negotiable). Importance here is measured **only out of sample**:
every estimator is fit on a train fold and interrogated on the *held-out* test
fold. In-sample impurity importance (MDI — scikit-learn's
``feature_importances_``) is **banned**: it is computed on the training data the
tree already memorised, it inflates high-cardinality / noisy features, and it
says nothing about generalisation. Neither function in this module ever reads
``feature_importances_``; importance comes exclusively from (a) permuting the
held-out fold and watching the score fall, or (b) SHAP values evaluated on the
held-out fold.

MULTICOLLINEARITY. Two correlated features each look unimportant under a naive
single-feature permutation because the model leans on the surviving twin. We
therefore permute / aggregate whole **clusters** (from
:func:`qedge.features.cluster.feature_clusters`, fit on TRAIN data only). When no
clustering is supplied each feature is its own singleton cluster, recovering the
classic per-feature reading.

Two estimators of importance, both OOS:

* :func:`mda_importance_oos` — Mean-Decrease-Accuracy. For each test fold and each
  cluster, shuffle that cluster's columns *within the test fold* and measure how
  much the score degrades. A bigger drop ⇒ the cluster carried more signal.
  Higher return value = more important.
* :func:`clustered_shap_oos` — mean absolute SHAP attribution on the test fold,
  summed within each cluster. A model-faithful complement to MDA.

Both are deterministic: every estimator is seeded and every permutation draws
from an explicit :class:`numpy.random.Generator`.

Only numpy, pandas, sklearn, shap and ``qedge.features.cluster`` are imported.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss

__all__ = ["clustered_shap_oos", "mda_importance_oos"]

#: Supported scoring rules for :func:`mda_importance_oos`.
_SCORING_NEG_LOG_LOSS = "neg_log_loss"
_SCORING_ACCURACY = "accuracy"
_VALID_SCORINGS = (_SCORING_NEG_LOG_LOSS, _SCORING_ACCURACY)

#: A clustering with a single group means "permute every feature together"; we
#: instead default to per-feature singleton clusters when none is supplied.
_NO_CLUSTERS: dict[int, list[str]] | None = None


class _SplitCV(Protocol):
    """Minimal cross-validator surface: yields positional ``(train, test)``."""

    def split(
        self, X: pd.DataFrame, y: pd.Series | None = ...
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]: ...


def _resolve_clusters(
    columns: list[str], clusters: dict[int, list[str]] | None
) -> dict[int, list[str]]:
    """Return ``clusters`` or, if ``None``, one singleton cluster per column.

    When clusters are supplied they are filtered to the columns actually present
    in ``X`` (and emptied groups dropped) so a clustering fit on a superset of
    features stays valid.
    """
    present = set(columns)
    if clusters is None:
        return {i: [col] for i, col in enumerate(columns)}
    filtered: dict[int, list[str]] = {}
    next_id = 0
    for members in clusters.values():
        kept = [c for c in members if c in present]
        if kept:
            filtered[next_id] = kept
            next_id += 1
    if not filtered:
        raise ValueError("no cluster members are present in X's columns")
    return filtered


def _all_class_labels(y: pd.Series) -> npt.NDArray[np.int_ | np.object_]:
    """Sorted unique labels across the whole dataset (stable column order)."""
    return np.sort(pd.unique(y))


def _score(
    estimator: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    scoring: str,
    labels: npt.NDArray[np.int_ | np.object_],
) -> float:
    """Score a fitted estimator on the test fold (higher = better).

    ``neg_log_loss`` returns the negated log-loss so that, like accuracy, larger
    is better; importance is then ``base_score - permuted_score``.
    """
    if scoring == _SCORING_ACCURACY:
        preds = estimator.predict(X_test)
        return float(accuracy_score(y_test, preds))
    proba = estimator.predict_proba(X_test)
    # Pin the label set so folds missing a class still align probability columns.
    return float(-log_loss(y_test, proba, labels=list(labels)))


def _fit_clone(
    estimator_factory: Callable[[], ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ClassifierMixin:
    """Build a fresh estimator from the factory and fit it on the train fold."""
    estimator = estimator_factory()
    estimator.fit(X_train, y_train)
    return estimator


def mda_importance_oos(
    estimator_factory: Callable[[], ClassifierMixin],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: _SplitCV,
    clusters: dict[int, list[str]] | None = _NO_CLUSTERS,
    scoring: str = _SCORING_NEG_LOG_LOSS,
    seed: int,
) -> pd.Series:
    """Clustered Mean-Decrease-Accuracy importance, measured out of sample.

    For every ``(train, test)`` fold from ``cv`` a fresh estimator is fit on the
    train fold and scored on the test fold (the *base* score). Then, for each
    cluster, that cluster's columns are jointly permuted **within the test fold**
    and the estimator re-scored; the drop ``base - permuted`` is the cluster's
    importance contribution for that fold. Contributions are averaged across folds.

    The estimator is fit on train and the *permutation* happens on the held-out
    test fold — never on training data — so this is a true OOS measurement and
    uses no in-sample MDI.

    Args:
        estimator_factory: Zero-arg callable returning an *unfitted* classifier.
            It must already encode its own ``random_state`` for determinism.
        X: Features-in-columns frame, time-ordered to match ``cv``.
        y: Aligned class labels.
        cv: A cross-validator whose ``split(X, y)`` yields positional
            ``(train_idx, test_idx)`` arrays (e.g. ``CombinatorialPurgedCV`` or
            ``PurgedKFold``).
        clusters: ``cluster_id -> [feature names]`` from
            :func:`qedge.features.cluster.feature_clusters` (fit on TRAIN data).
            When ``None`` each feature is permuted alone.
        scoring: ``"neg_log_loss"`` (default) or ``"accuracy"``.
        seed: Seeds the permutation RNG so shuffles are reproducible.

    Returns:
        A :class:`pandas.Series` indexed by ``cluster_id``, the mean score drop
        per cluster across folds. Higher = more important.

    Raises:
        ValueError: If ``scoring`` is unsupported or ``X``/``y`` lengths differ.
    """
    if scoring not in _VALID_SCORINGS:
        raise ValueError(f"scoring must be one of {_VALID_SCORINGS}, got {scoring!r}")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    groups = _resolve_clusters(list(X.columns), clusters)
    labels = _all_class_labels(y)
    rng = np.random.default_rng(seed)

    # Accumulate per-cluster importance and a fold counter to average at the end.
    totals: dict[int, float] = {cid: 0.0 for cid in groups}
    n_folds = 0

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if len(X_test) == 0 or len(X_train) == 0:
            continue

        estimator = _fit_clone(estimator_factory, X_train, y_train)
        base = _score(estimator, X_test, y_test, scoring=scoring, labels=labels)

        for cid, members in groups.items():
            permuted = X_test.copy()
            # Joint permutation: one shuffle order applied to ALL columns of the
            # cluster so their internal correlation structure is preserved while
            # the cluster's link to y is destroyed.
            order = rng.permutation(len(permuted))
            for col in members:
                permuted[col] = permuted[col].to_numpy()[order]
            permuted_score = _score(
                estimator, permuted, y_test, scoring=scoring, labels=labels
            )
            totals[cid] += base - permuted_score
        n_folds += 1

    if n_folds == 0:
        raise ValueError("cv produced no usable folds")

    importance = {cid: total / n_folds for cid, total in totals.items()}
    return pd.Series(importance, name="mda_importance").sort_index()


def _shap_abs_per_feature(
    estimator: ClassifierMixin, X_test: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Mean ``|SHAP|`` per feature on the test fold, multiclass-collapsed.

    ``shap.TreeExplainer`` returns either a 2-D ``(n_samples, n_features)`` array
    (binary / regression), a 3-D ``(n_samples, n_features, n_classes)`` array, or
    a list of per-class 2-D arrays (older multiclass API). In every case we take
    the absolute value, sum any class axis (a feature's total attribution across
    all classes), then average over samples to get one number per feature.
    """
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(X_test)

    if isinstance(raw, list):
        # List of per-class (n_samples, n_features) arrays -> stack on a class axis.
        stacked = np.stack([np.abs(np.asarray(c, dtype=np.float64)) for c in raw], axis=-1)
        per_sample_feature = stacked.sum(axis=-1)
    else:
        arr = np.abs(np.asarray(raw, dtype=np.float64))
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            per_sample_feature = arr.sum(axis=-1)
        else:  # (n_samples, n_features)
            per_sample_feature = arr
    return np.asarray(per_sample_feature.mean(axis=0), dtype=np.float64)


def clustered_shap_oos(
    estimator_factory: Callable[[], ClassifierMixin],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: _SplitCV,
    clusters: dict[int, list[str]] | None = _NO_CLUSTERS,
    seed: int,
) -> pd.Series:
    """Clustered mean-|SHAP| importance, measured out of sample.

    Per ``(train, test)`` fold a fresh estimator is fit on train and SHAP values
    are computed on the **held-out test fold** via :class:`shap.TreeExplainer`
    (tree models). Per-feature mean ``|SHAP|`` (multiclass attributions summed
    across classes) is summed within each cluster, then averaged across folds.
    SHAP is only ever evaluated OOS; no in-sample MDI is touched.

    Args:
        estimator_factory: Zero-arg callable returning an unfitted *tree*
            classifier (RandomForest / GradientBoosting / LightGBM), seeded for
            determinism.
        X: Features-in-columns frame, time-ordered to match ``cv``.
        y: Aligned class labels.
        cv: Cross-validator yielding positional ``(train_idx, test_idx)``.
        clusters: ``cluster_id -> [feature names]`` (fit on TRAIN data). When
            ``None`` each feature is its own singleton cluster.
        seed: Accepted for API symmetry and reproducibility; the SHAP path is
            deterministic given a seeded estimator, but the seed also reseeds a
            local RNG so any future stochastic step stays reproducible.

    Returns:
        A :class:`pandas.Series` indexed by ``cluster_id``, the mean
        within-cluster total mean-|SHAP| across folds. Higher = more important.

    Raises:
        ValueError: If ``X``/``y`` lengths differ or ``cv`` yields no folds.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    groups = _resolve_clusters(list(X.columns), clusters)
    columns = list(X.columns)
    # Reseed for reproducibility even though the TreeExplainer path is exact.
    _ = np.random.default_rng(seed)

    totals: dict[int, float] = {cid: 0.0 for cid in groups}
    n_folds = 0

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        if len(X_test) == 0 or len(X_train) == 0:
            continue

        estimator = _fit_clone(estimator_factory, X_train, y_train)
        per_feature = _shap_abs_per_feature(estimator, X_test)
        feature_to_value = dict(zip(columns, per_feature, strict=True))

        for cid, members in groups.items():
            totals[cid] += float(sum(feature_to_value[col] for col in members))
        n_folds += 1

    if n_folds == 0:
        raise ValueError("cv produced no usable folds")

    importance = {cid: total / n_folds for cid, total in totals.items()}
    return pd.Series(importance, name="clustered_shap").sort_index()
