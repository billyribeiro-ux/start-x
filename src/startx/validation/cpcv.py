"""Combinatorial Purged Cross-Validation (CPCV).

Standard k-fold gives a *single* train/test partition per fold and therefore a
single backtest path. CPCV (López de Prado, *Advances in Financial ML*, Ch. 12)
splits the time-ordered data into ``N`` contiguous groups and, for every
combination of ``k`` groups held out as test, trains on the remaining
``N - k`` groups (purged + embargoed). This yields ``C(N, k)`` train/test
splits and, crucially, a *large number of distinct backtest paths*, which is
what the Probability of Backtest Overfitting (PBO) needs.

Number of paths
---------------
Each of the ``N`` groups appears in ``C(N - 1, k - 1)`` of the test
combinations (choose the other ``k - 1`` test groups from the remaining
``N - 1``). Reassembling one out-of-sample prediction per group draw gives::

    n_paths = C(N - 1, k - 1) = C(N, k) * k / N
"""
from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from startx.validation.purged_cv import _as_t1, _infer_index, purge_embargo_train


class CombinatorialPurgedCV:
    """Combinatorial Purged CV producing ``C(n_groups, n_test_groups)`` splits.

    Parameters
    ----------
    n_groups:
        Number ``N`` of contiguous time-ordered groups.
    n_test_groups:
        Number ``k`` of groups held out as test in each combination.
    t1:
        Label end times (see :class:`~startx.validation.purged_cv.PurgedKFold`).
    embargo_pct:
        Post-test embargo as a fraction of dataset length.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 t1: pd.Series | None = None, embargo_pct: float = 0.01) -> None:
        if t1 is None:
            raise ValueError("CombinatorialPurgedCV requires t1 (label end times)")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("require 1 <= n_test_groups < n_groups")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.t1 = t1
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        """Number of train/test combinations ``C(N, k)``."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of distinct OOS backtest paths, ``C(N - 1, k - 1)``."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def _group_positions(self, n: int) -> list[np.ndarray]:
        """Split ``range(n)`` into ``n_groups`` contiguous positional blocks."""
        return [g for g in np.array_split(np.arange(n), self.n_groups) if g.size]

    def split(
        self, X, y=None, groups=None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for every test-group combination.

        For each combination the test set is the union of the chosen groups'
        positions; the train set is everything else, purged and embargoed
        against *each* contiguous test block independently.
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

        group_pos = self._group_positions(n)
        for combo in combinations(range(len(group_pos)), self.n_test_groups):
            test_idx = np.sort(np.concatenate([group_pos[g] for g in combo]))
            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_idx] = True
            train_idx = positions[~test_mask]
            # Purge/embargo against each contiguous test block independently so
            # that non-adjacent test groups each get their own protection band.
            for g in combo:
                block = group_pos[g]
                train_idx = purge_embargo_train(
                    train_idx, block,
                    starts=starts, ends=ends, embargo=embargo, n_samples=n,
                )
            yield train_idx, test_idx

    def build_paths(
        self, X, predictions: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> list[list[tuple[int, int]]]:
        """Map split index -> which path each test group contributes to.

        Returns a list of length :attr:`n_paths`; entry ``p`` is the ordered
        list of ``(split_idx, group_id)`` pairs describing which
        (combination, group) draws form path ``p``. This reassembles one
        OOS prediction per group into a coherent full-sample path, as required
        by PBO. ``predictions`` is accepted for API symmetry but the assignment
        depends only on the combinatorial structure.
        """
        index = _infer_index(X, self.t1)
        n = len(index)
        group_pos = self._group_positions(n)
        n_grp = len(group_pos)

        combos = list(combinations(range(n_grp), self.n_test_groups))
        # paths[group][j] = split index where this group plays its j-th test role
        per_group_splits: list[list[int]] = [[] for _ in range(n_grp)]
        for s_idx, combo in enumerate(combos):
            for g in combo:
                per_group_splits[g].append(s_idx)

        n_paths = self.n_paths
        # Build n_paths paths, each picking, for every group, a distinct split.
        paths: list[list[int]] = []
        for p in range(n_paths):
            path = []
            for g in range(n_grp):
                split_for_group = per_group_splits[g][p]
                path.append((split_for_group, g))
            paths.append(path)
        return paths

    def assemble_oos(
        self,
        X,
        split_test_preds: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[pd.Series]:
        """Reassemble per-split OOS predictions into :attr:`n_paths` full paths.

        Parameters
        ----------
        X:
            The dataset (for index/length); must match what was passed to
            :meth:`split`.
        split_test_preds:
            One entry per yielded split, in the *same order* as :meth:`split`,
            each a ``(test_idx, preds)`` pair of equal-length arrays.

        Returns
        -------
        list[pd.Series]
            ``n_paths`` series indexed like ``X``, each a complete OOS
            prediction path covering the whole sample exactly once.
        """
        index = _infer_index(X, self.t1)
        n = len(index)
        group_pos = self._group_positions(n)
        n_grp = len(group_pos)
        combos = list(combinations(range(n_grp), self.n_test_groups))
        if len(split_test_preds) != len(combos):
            raise ValueError("split_test_preds must have one entry per split")

        # Map (split_idx, group) -> predictions for that group's positions.
        # Each test block in a split corresponds to one group.
        group_of_pos = np.empty(n, dtype=int)
        for gid, gp in enumerate(group_pos):
            group_of_pos[gp] = gid

        # split_group_pred[s][g] = array of preds aligned to group_pos[g]
        split_group_pred: dict[tuple[int, int], np.ndarray] = {}
        for s_idx, (combo, (test_idx, preds)) in enumerate(zip(combos, split_test_preds)):
            test_idx = np.asarray(test_idx)
            preds = np.asarray(preds, dtype=float)
            for g in combo:
                gp = group_pos[g]
                mask = np.isin(test_idx, gp)
                split_group_pred[(s_idx, g)] = preds[mask]

        per_group_splits: list[list[int]] = [[] for _ in range(n_grp)]
        for s_idx, combo in enumerate(combos):
            for g in combo:
                per_group_splits[g].append(s_idx)

        paths: list[pd.Series] = []
        for p in range(self.n_paths):
            values = np.full(n, np.nan)
            for g in range(n_grp):
                s_idx = per_group_splits[g][p]
                gp = group_pos[g]
                values[gp] = split_group_pred[(s_idx, g)]
            paths.append(pd.Series(values, index=index))
        return paths
