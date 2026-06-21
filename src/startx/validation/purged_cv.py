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


def _is_datetime_like(values: np.ndarray) -> bool:
    """True if ``values`` is a datetime64 array (real timestamps)."""
    return np.issubdtype(np.asarray(values).dtype, np.datetime64)


def label_start_end(t1: pd.Series, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return positionally-aligned ``(starts, ends)`` label-interval bounds.

    ``starts`` is each sample's entry time and ``ends`` its label-end time, as a
    **type-consistent, comparable** pair — the firewall's overlap test
    ``start <= test_end and t1 >= test_start`` is only meaningful when both sides
    share a dtype. Two production shapes are supported:

    * **DatetimeIndex shape** — ``index`` carries entry timestamps and ``t1``'s
      *values* are end timestamps (the legacy/portfolio path). Both bounds are
      timestamps.
    * **RangeIndex Dataset shape** — ``X``/``t1`` carry a RangeIndex but ``t1``
      is indexed by entry date and its values are end dates (the model path).
      Both bounds are taken from ``t1`` (its index = starts, its values = ends),
      so neither side is the meaningless RangeIndex.

    The previous implementation took ``starts = index.to_numpy()`` unconditionally
    and compared int positions against datetime ``ends`` -> ``UFuncTypeError``.
    """
    idx_vals = index.to_numpy()
    t1_vals = t1.to_numpy() if isinstance(t1, pd.Series) else np.asarray(t1)
    t1_index_vals = np.asarray(t1.index) if isinstance(t1, pd.Series) else None

    # DatetimeIndex shape: starts come from the index, ends from t1's values.
    if _is_datetime_like(idx_vals):
        aligned = _as_t1(t1, index)
        return idx_vals, aligned.to_numpy()

    # RangeIndex Dataset shape: t1 is indexed by entry date -> use t1 for both.
    if (
        t1_index_vals is not None and _is_datetime_like(t1_index_vals)
        and _is_datetime_like(t1_vals)
    ):
        starts = t1_index_vals
        ends = pd.Series(t1_vals).fillna(pd.Series(starts)).to_numpy()
        return starts, ends

    # Fallback: positions for starts, ends aligned to index (already comparable
    # if both are positional/numeric). Keeps single-bar labels as points.
    aligned = _as_t1(t1, index)
    ends = aligned.to_numpy()
    if _is_datetime_like(ends):
        # No usable entry timestamps anywhere -> collapse to single positional points.
        pos = np.arange(len(index))
        return pos, pos.copy()
    return idx_vals, ends


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

        # Type-consistent label bounds (robust to the RangeIndex Dataset shape,
        # where index positions are ints but t1 carries datetimes -> the old
        # ``starts = index.to_numpy()`` raised UFuncTypeError on the overlap test).
        starts, ends = label_start_end(self.t1, index)
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
