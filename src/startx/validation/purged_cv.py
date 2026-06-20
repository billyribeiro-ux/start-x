"""Purged K-Fold cross-validation — the leakage firewall.

In financial ML the samples are *not* IID: a label observed at time ``t`` is
built from information spanning ``[t, t1]`` (e.g. a triple-barrier outcome).
If a train sample's label interval overlaps a test sample's label interval,
information leaks across the train/test boundary and the backtest becomes
optimistic — the classic "100% win-rate" overfitting trap.

This module removes that leakage in two steps (López de Prado, *Advances in
Financial ML*, Ch. 7):

1. **Purging** — drop from the training set every sample whose label interval
   ``[start, t1]`` *overlaps* the test set's label interval. Overlap is the
   condition ``start <= test_end`` AND ``t1 >= test_start``.
2. **Embargo** — additionally drop training samples that fall in a small window
   immediately *after* the test set. Serial correlation means a test label can
   still influence features of nearby subsequent samples even without interval
   overlap; the embargo (a fraction ``embargo_pct`` of the dataset length)
   neutralises that.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def _as_t1(t1: pd.Series, index: pd.Index) -> pd.Series:
    """Return ``t1`` aligned to ``index`` with NaN entries filled by start time.

    A NaN ``t1`` means the label never closed (or spans a single bar); treating
    it as ``start`` makes its interval a single point, which is the safe,
    conservative choice for overlap tests.
    """
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pandas Series mapping start_time -> end_time")
    t1 = t1.reindex(index)
    start = pd.Series(index, index=index)
    return t1.fillna(start)


def purge_embargo_train(
    train_idx: np.ndarray,
    test_positions: np.ndarray,
    *,
    starts: pd.Index | np.ndarray,
    ends: np.ndarray,
    embargo: int,
    n_samples: int,
) -> np.ndarray:
    """Apply purge + embargo to ``train_idx`` given a block of test positions.

    Parameters use *positional* indices (0..n_samples-1) throughout, with
    ``starts``/``ends`` providing the label interval bounds (as comparable
    values, e.g. timestamps) for every sample by position.

    Returns the surviving train positions as an ``ndarray``.
    """
    if test_positions.size == 0:
        return np.asarray(train_idx)

    test_start = starts[test_positions].min()
    test_end = ends[test_positions].max()

    tr_starts = starts[train_idx]
    tr_ends = ends[train_idx]

    # PURGE: a train label interval [start, t1] overlaps the test interval
    # [test_start, test_end] iff start <= test_end and t1 >= test_start.
    overlaps = (tr_starts <= test_end) & (tr_ends >= test_start)
    keep = ~overlaps

    # EMBARGO: remove train samples positioned just after the test block.
    if embargo > 0:
        test_max_pos = int(test_positions.max())
        embargo_hi = test_max_pos + embargo  # inclusive upper bound
        in_embargo = (train_idx > test_max_pos) & (train_idx <= embargo_hi)
        keep &= ~in_embargo

    return np.asarray(train_idx)[keep]


class PurgedKFold:
    """K-Fold CV with purging and embargo for overlapping financial labels.

    Parameters
    ----------
    n_splits:
        Number of contiguous (time-ordered) test folds.
    t1:
        ``pd.Series`` mapping each sample's start time (its index) to the end
        time of its label. Must share its index with ``X`` passed to
        :meth:`split`. NaN values are treated as the start time.
    embargo_pct:
        Fraction of the dataset length used as the post-test embargo. With
        ``embargo_pct=0.01`` and 1000 samples the embargo spans 10 samples.

    Notes
    -----
    ``X`` is assumed time-ordered (row ``i`` is no later than row ``i+1``); the
    folds are contiguous slices, mirroring López de Prado's implementation.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 embargo_pct: float = 0.01) -> None:
        if t1 is None:
            raise ValueError("PurgedKFold requires t1 (label end times)")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        """Return the number of splitting iterations (sklearn-compatible)."""
        return self.n_splits

    def split(
        self, X, y=None, groups=None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` positional arrays for each fold.

        ``X`` must be time-ordered and share its index with ``t1``.
        """
        index = _infer_index(X, self.t1)
        n = len(index)
        if n != len(self.t1):
            raise ValueError("X and t1 must have the same length / index")

        t1 = _as_t1(self.t1, index)
        starts = index.to_numpy()
        ends = t1.to_numpy()
        embargo = int(n * self.embargo_pct)

        positions = np.arange(n)
        fold_bounds = np.array_split(positions, self.n_splits)

        for test_positions in fold_bounds:
            if test_positions.size == 0:
                continue
            test_idx = test_positions
            lo, hi = int(test_positions.min()), int(test_positions.max())
            train_idx = positions[(positions < lo) | (positions > hi)]
            train_idx = purge_embargo_train(
                train_idx, test_idx,
                starts=starts, ends=ends, embargo=embargo, n_samples=n,
            )
            yield train_idx, test_idx


def _infer_index(X, t1: pd.Series) -> pd.Index:
    """Return the index to use, preferring ``X``'s index, falling back to t1's."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.index
    if X is not None:
        n = len(X)
        if n != len(t1):
            raise ValueError("X and t1 must have the same length")
        return t1.index
    return t1.index
