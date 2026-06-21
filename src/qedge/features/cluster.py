"""Hierarchical feature clustering on a correlation-distance metric.

Feature importance in :mod:`qedge.modeling.importance` is measured by permuting
*clusters* of features rather than raw columns. The reason is multicollinearity:
two highly correlated features each look unimportant under a single-feature
permutation (permuting one leaves its near-duplicate intact, so the model barely
suffers), which masks a genuine joint edge. Grouping correlated features and
permuting the whole group restores an honest reading.

This module builds those groups. Pairwise correlations are turned into a proper
**distance metric** ``sqrt(0.5 * (1 - corr))`` — zero for perfectly correlated
features, ``1`` for anti-correlated ones — which satisfies the triangle
inequality and is the standard choice for clustering return/feature series
(López de Prado, *Advances in Financial ML*, ch. 4). Average-linkage hierarchical
clustering then cuts the dendrogram at ``cfg.features.cluster_distance_threshold``.

POINT-IN-TIME CONTRACT: clusters MUST be fit on TRAIN data only. The correlation
structure is itself a learned quantity; fitting it on the full sample leaks
test-fold information into the grouping that the importance permutation then uses.
:func:`feature_clusters` clusters whatever ``X`` it is handed and performs no
splitting — callers are responsible for passing the *training* slice only.

Only numpy, pandas, scipy and ``qedge.config`` are imported.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from qedge.config import get_config

__all__ = ["correlation_distance", "feature_clusters"]

#: Average linkage — robust to chaining and the natural pairing for a distance
#: defined from correlations (no Ward/centroid Euclidean assumptions).
_LINKAGE_METHOD = "average"
#: ``fcluster`` criterion: cut the dendrogram wherever the cophenetic distance
#: between merged observations exceeds the threshold.
_FCLUSTER_CRITERION = "distance"
#: A single column cannot be correlated with anything — it is its own cluster.
_MIN_FEATURES_TO_CLUSTER = 2
#: Correlation-distance bounds: 0 for corr=+1, 1 for corr=-1.
_DISTANCE_MIN = 0.0
_DISTANCE_MAX = 1.0


def correlation_distance(X: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return the ``sqrt(0.5 * (1 - corr))`` distance matrix for ``X``'s columns.

    The Pearson correlation matrix is mapped to a metric distance: identical
    columns are distance ``0``, perfectly anti-correlated columns distance ``1``.
    Any NaN correlation (e.g. a constant column has undefined correlation) is
    treated as maximally distant (``1``) so a degenerate feature is never glued
    to another. The diagonal is forced to exactly ``0``.

    Args:
        X: A features-in-columns :class:`pandas.DataFrame`.

    Returns:
        A symmetric ``(n_features, n_features)`` float distance matrix aligned to
        ``X.columns`` order.
    """
    corr = X.corr().to_numpy(dtype=np.float64)
    # A constant column yields NaN correlations; treat those pairs as maximally
    # distant rather than letting NaNs propagate into the linkage.
    corr = np.asarray(np.nan_to_num(corr, nan=-1.0), dtype=np.float64)
    corr = np.clip(corr, -1.0, 1.0)
    distance = np.sqrt(0.5 * (1.0 - corr))
    distance = np.clip(distance, _DISTANCE_MIN, _DISTANCE_MAX)
    # Enforce exact symmetry and a zero diagonal against float round-off so the
    # matrix is a valid input to scipy.spatial.distance.squareform.
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, _DISTANCE_MIN)
    return np.asarray(distance, dtype=np.float64)


def feature_clusters(
    X: pd.DataFrame, *, threshold: float | None = None
) -> dict[int, list[str]]:
    """Group ``X``'s columns into clusters of correlated features.

    Average-linkage hierarchical clustering is run on the
    :func:`correlation_distance` matrix and the dendrogram is cut at
    ``threshold`` (cophenetic distance). Features whose mutual distance stays
    below the cut land in the same cluster; weakly correlated features separate.

    MUST be fit on TRAIN data only — see the module docstring. This function does
    no train/test splitting itself; it clusters exactly the rows it is given, so
    pass the training slice.

    Args:
        X: Features-in-columns frame. Column order defines membership ordering.
        threshold: Correlation-distance cut height. Defaults to
            ``cfg.features.cluster_distance_threshold``. With the
            ``sqrt(0.5*(1-corr))`` metric a threshold of ``0.5`` cuts at
            ``corr = 0.5``.

    Returns:
        A mapping ``cluster_id -> [feature names]``. Cluster ids are contiguous
        integers starting at ``0``, assigned in order of first appearance scanning
        ``X.columns`` left to right; the feature lists preserve column order. The
        union of all lists is exactly ``X.columns`` with no duplicates.

    Raises:
        ValueError: If ``X`` has no columns.
    """
    columns = list(X.columns)
    if len(columns) == 0:
        raise ValueError("X must have at least one column to cluster")

    cut = (
        get_config().features.cluster_distance_threshold
        if threshold is None
        else threshold
    )

    if len(columns) < _MIN_FEATURES_TO_CLUSTER:
        # A lone feature is trivially its own cluster.
        return {0: columns}

    distance = correlation_distance(X)
    # squareform needs the condensed upper triangle of the (symmetric, zero-diag)
    # distance matrix; average linkage then builds the dendrogram.
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method=_LINKAGE_METHOD)
    raw_labels = fcluster(linkage_matrix, t=cut, criterion=_FCLUSTER_CRITERION)

    # Relabel to contiguous ids in order of first appearance so the mapping is
    # deterministic and independent of scipy's internal label numbering.
    remap: dict[int, int] = {}
    clusters: dict[int, list[str]] = {}
    for column, raw_label in zip(columns, raw_labels, strict=True):
        label = int(raw_label)
        if label not in remap:
            new_id = len(remap)
            remap[label] = new_id
            clusters[new_id] = []
        clusters[remap[label]].append(column)
    return clusters
