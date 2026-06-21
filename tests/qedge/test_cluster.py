"""Tests for hierarchical feature clustering on correlation distance.

The contract these tests pin down:

* Highly correlated columns land in the SAME cluster (so the importance layer
  permutes them together and multicollinearity does not mask a joint edge).
* Independent columns land in SEPARATE clusters.
* The mapping is a partition of ``X.columns`` (no lost/duplicated features) with
  deterministic, contiguous, first-appearance-ordered cluster ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qedge.config import QedgeConfig
from qedge.features.cluster import correlation_distance, feature_clusters


def _two_correlated_blocks(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    """Build two correlated blocks (a*/b*) plus an independent column.

    ``a0,a1`` are near-duplicates; ``b0,b1`` are a second near-duplicate pair
    uncorrelated with the a-block; ``c`` is independent noise.
    """
    a_base = rng.standard_normal(n)
    b_base = rng.standard_normal(n)
    eps = 0.01
    frame = pd.DataFrame(
        {
            "a0": a_base + eps * rng.standard_normal(n),
            "a1": a_base + eps * rng.standard_normal(n),
            "b0": b_base + eps * rng.standard_normal(n),
            "b1": b_base + eps * rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    return frame


def _cluster_of(clusters: dict[int, list[str]], feature: str) -> int:
    """Return the cluster id that contains ``feature``."""
    for cid, members in clusters.items():
        if feature in members:
            return cid
    raise AssertionError(f"{feature} not found in any cluster")


def test_correlated_columns_share_a_cluster() -> None:
    rng = np.random.default_rng(7)
    X = _two_correlated_blocks(rng)
    clusters = feature_clusters(X)

    # The two near-duplicate pairs each stay together...
    assert _cluster_of(clusters, "a0") == _cluster_of(clusters, "a1")
    assert _cluster_of(clusters, "b0") == _cluster_of(clusters, "b1")
    # ...the two independent blocks separate...
    assert _cluster_of(clusters, "a0") != _cluster_of(clusters, "b0")
    # ...and the lone noise column is on its own.
    assert _cluster_of(clusters, "c") != _cluster_of(clusters, "a0")
    assert _cluster_of(clusters, "c") != _cluster_of(clusters, "b0")


def test_independent_columns_separate() -> None:
    rng = np.random.default_rng(11)
    n = 800
    X = pd.DataFrame(
        {name: rng.standard_normal(n) for name in ("f0", "f1", "f2", "f3")}
    )
    clusters = feature_clusters(X)
    # Four independent columns -> four distinct singleton clusters.
    assert len(clusters) == X.shape[1]
    for members in clusters.values():
        assert len(members) == 1


def test_clusters_partition_all_columns() -> None:
    rng = np.random.default_rng(3)
    X = _two_correlated_blocks(rng)
    clusters = feature_clusters(X)
    flat = [col for members in clusters.values() for col in members]
    # Every column appears exactly once.
    assert sorted(flat) == sorted(X.columns)
    assert len(flat) == len(set(flat))
    # Cluster ids are contiguous from zero.
    assert sorted(clusters.keys()) == list(range(len(clusters)))


def test_threshold_controls_granularity() -> None:
    rng = np.random.default_rng(5)
    X = _two_correlated_blocks(rng)
    # A very small cut (everything must be near-identical to merge) fragments
    # into more clusters than a generous cut.
    tight = feature_clusters(X, threshold=0.05)
    loose = feature_clusters(X, threshold=0.9)
    assert len(tight) >= len(loose)
    # The generous cut collapses the whole frame into a single cluster.
    assert len(loose) == 1


def test_default_threshold_is_config() -> None:
    rng = np.random.default_rng(9)
    X = _two_correlated_blocks(rng)
    explicit = feature_clusters(
        X, threshold=QedgeConfig().features.cluster_distance_threshold
    )
    default = feature_clusters(X)
    assert explicit == default


def test_correlation_distance_is_a_metric() -> None:
    rng = np.random.default_rng(13)
    X = _two_correlated_blocks(rng)
    d = correlation_distance(X)
    # Symmetric, zero diagonal, bounded in [0, 1].
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)
    assert d.min() >= 0.0
    assert d.max() <= 1.0
    # Correlated pair is much closer than an independent pair.
    cols = list(X.columns)
    i_a0, i_a1 = cols.index("a0"), cols.index("a1")
    i_c = cols.index("c")
    assert d[i_a0, i_a1] < d[i_a0, i_c]


def test_single_column_is_its_own_cluster() -> None:
    X = pd.DataFrame({"only": np.arange(10.0)})
    clusters = feature_clusters(X)
    assert clusters == {0: ["only"]}
