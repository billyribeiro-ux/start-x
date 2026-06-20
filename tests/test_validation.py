"""Rigorous tests for the validation (anti-overfitting) layer.

The crown jewel is the PurgedKFold LEAKAGE FIREWALL test: with overlapping
labels, no training sample's label interval may overlap the test interval, and
the post-test embargo region must be excluded.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from startx.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe,
    max_drawdown,
    probability_of_backtest_overfitting,
    scoreboard,
    sharpe,
    walk_forward_predict,
)
from startx.validation.metrics import (
    cagr,
    expectancy,
    hit_rate,
    profit_factor,
    sortino,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_overlapping(n: int = 100, span: int = 5, freq: str = "D"):
    """Return (X, y, t1) where each label spans ``span`` bars (overlapping)."""
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    y = pd.Series(np.tile([0, 1], n // 2 + 1)[:n], index=idx)
    # label observed at idx[i] closes at idx[min(i+span, n-1)] -> overlaps.
    end_pos = np.minimum(np.arange(n) + span, n - 1)
    t1 = pd.Series(idx[end_pos], index=idx)
    return X, y, t1


# --------------------------------------------------------------------------- #
# 1. PurgedKFold — the leakage firewall (CRITICAL)
# --------------------------------------------------------------------------- #
def test_purged_kfold_leakage_firewall():
    """No train label interval may overlap the test interval; embargo excluded."""
    n, span, embargo_pct = 120, 6, 0.02
    X, y, t1 = make_overlapping(n=n, span=span)
    starts = t1.index.to_numpy()
    ends = t1.to_numpy()
    embargo = int(n * embargo_pct)

    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=embargo_pct)
    n_folds = 0
    for train_idx, test_idx in pkf.split(X):
        n_folds += 1
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test disjoint"

        test_start = starts[test_idx].min()
        test_end = ends[test_idx].max()
        tr_starts = starts[train_idx]
        tr_ends = ends[train_idx]

        # FIREWALL: no overlap of [start, t1] with [test_start, test_end].
        overlap = (tr_starts <= test_end) & (tr_ends >= test_start)
        assert not overlap.any(), (
            f"LEAKAGE: {overlap.sum()} train samples overlap the test interval"
        )

        # EMBARGO: positions in (max_test, max_test+embargo] must be purged.
        max_test = int(test_idx.max())
        embargo_zone = set(range(max_test + 1, min(max_test + embargo + 1, n)))
        assert embargo_zone.isdisjoint(set(train_idx.tolist())), "embargo not applied"

    assert n_folds == 5


def test_purged_kfold_handles_nan_t1():
    """NaN t1 is treated as the sample's start time (single-point interval)."""
    n = 60
    X, y, t1 = make_overlapping(n=n, span=4)
    t1 = t1.copy()
    t1.iloc[10:20] = pd.NaT  # inject NaNs
    pkf = PurgedKFold(n_splits=4, t1=t1, embargo_pct=0.0)
    starts = t1.index.to_numpy()
    ends = t1.fillna(pd.Series(t1.index, index=t1.index)).to_numpy()
    for train_idx, test_idx in pkf.split(X):
        test_start = starts[test_idx].min()
        test_end = ends[test_idx].max()
        overlap = (starts[train_idx] <= test_end) & (ends[train_idx] >= test_start)
        assert not overlap.any()


def test_purged_kfold_requires_t1():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, t1=None)


# --------------------------------------------------------------------------- #
# 2. CPCV — combination count and path count
# --------------------------------------------------------------------------- #
def test_cpcv_split_and_path_counts():
    n, n_groups, k = 120, 6, 2
    X, y, t1 = make_overlapping(n=n, span=4)
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=k, t1=t1,
                               embargo_pct=0.01)

    splits = list(cv.split(X))
    assert len(splits) == comb(n_groups, k) == 15
    assert cv.get_n_splits(X) == 15
    # n_paths = C(N-1, k-1)
    assert cv.n_paths == comb(n_groups - 1, k - 1) == 5

    starts = t1.index.to_numpy()
    ends = t1.to_numpy()
    # group boundaries (contiguous) to evaluate purge per test BLOCK
    group_pos = [g for g in np.array_split(np.arange(n), n_groups) if g.size]
    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        test_set = set(test_idx.tolist())
        # Firewall must hold against EACH contiguous test block independently;
        # the global union interval would span gaps between non-adjacent groups.
        for blk in group_pos:
            if not set(blk.tolist()) <= test_set:
                continue
            blk_start = starts[blk].min()
            blk_end = ends[blk].max()
            overlap = (starts[train_idx] <= blk_end) & (ends[train_idx] >= blk_start)
            assert not overlap.any(), "LEAKAGE: train overlaps a CPCV test block"


def test_cpcv_assemble_oos_covers_sample():
    """Each reassembled path covers every sample exactly once."""
    n, n_groups, k = 60, 5, 2
    X, y, t1 = make_overlapping(n=n, span=3)
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=k, t1=t1,
                               embargo_pct=0.0)
    # produce trivial predictions = test position, per split
    split_preds = []
    for _train_idx, test_idx in cv.split(X):
        split_preds.append((test_idx, test_idx.astype(float)))
    paths = cv.assemble_oos(X, split_preds)
    assert len(paths) == cv.n_paths
    for path in paths:
        # every position filled exactly once (no NaN), values equal positions
        assert path.notna().all()
        np.testing.assert_array_equal(path.to_numpy(), np.arange(n, dtype=float))


# --------------------------------------------------------------------------- #
# 3. Deflated Sharpe — white noise + many trials => low DSR => overfit verdict
# --------------------------------------------------------------------------- #
def test_deflated_sharpe_white_noise_low_probability():
    """Pure white noise selected from many trials should NOT survive."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, 500)  # zero true edge
    dsr_prob = deflated_sharpe(returns, n_trials=100)
    assert 0.0 <= dsr_prob <= 1.0
    assert dsr_prob < 0.95, "noise must not clear the 95% DSR bar"

    sb = scoreboard(pd.Series(returns), n_trials=100)
    assert sb["verdict"] == "LIKELY OVERFIT"
    assert sb["deflated_sharpe"] <= 0.0  # centred DSR below threshold


def test_deflated_sharpe_strong_edge_high_probability():
    """A genuinely strong, consistent edge should clear the bar with few trials."""
    rng = np.random.default_rng(3)
    # high mean relative to vol => high Sharpe, long sample
    returns = rng.normal(0.004, 0.01, 800)
    dsr_prob = deflated_sharpe(returns, n_trials=1)
    assert dsr_prob > 0.95
    sb = scoreboard(pd.Series(returns), n_trials=1)
    assert sb["verdict"] == "EDGE SURVIVES"


def test_deflated_sharpe_more_trials_lowers_probability():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0015, 0.01, 500)
    p_few = deflated_sharpe(returns, n_trials=1)
    p_many = deflated_sharpe(returns, n_trials=5000)
    assert p_many <= p_few  # more trials => harder to clear


# --------------------------------------------------------------------------- #
# 4. PBO on random IS/OOS matrices => ~0.5
# --------------------------------------------------------------------------- #
def test_pbo_random_matrices_near_half():
    rng = np.random.default_rng(0)
    n_splits, n_configs = 400, 20
    perf_is = rng.normal(size=(n_splits, n_configs))
    perf_oos = rng.normal(size=(n_splits, n_configs))  # independent of IS
    pbo = probability_of_backtest_overfitting(perf_is, perf_oos)
    assert 0.40 <= pbo <= 0.60, f"random PBO should be ~0.5, got {pbo}"


def test_pbo_perfect_persistence_near_zero():
    """If OOS ranking == IS ranking, the IS-best is always OOS-best => PBO≈0."""
    rng = np.random.default_rng(1)
    n_splits, n_configs = 200, 15
    perf_is = rng.normal(size=(n_splits, n_configs))
    perf_oos = perf_is + rng.normal(scale=1e-6, size=perf_is.shape)
    pbo = probability_of_backtest_overfitting(perf_is, perf_oos)
    assert pbo < 0.05


def test_pbo_anti_persistence_near_one():
    """If OOS is the inverse of IS, the IS-best is always OOS-worst => PBO≈1."""
    rng = np.random.default_rng(2)
    n_splits, n_configs = 200, 15
    perf_is = rng.normal(size=(n_splits, n_configs))
    perf_oos = -perf_is
    pbo = probability_of_backtest_overfitting(perf_is, perf_oos)
    assert pbo > 0.95


# --------------------------------------------------------------------------- #
# 5. Walk-forward — learnable vs shuffled (leakage canary)
# --------------------------------------------------------------------------- #
def _learnable_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f": f}, index=idx)
    y = pd.Series((f > 0).astype(int), index=idx)  # y = sign of feature
    t1 = pd.Series(idx, index=idx)
    return X, y, t1


def test_walk_forward_learns_easy_signal():
    X, y, t1 = _learnable_dataset()
    out = walk_forward_predict(
        X, y, t1, lambda: LogisticRegression(),
        train_min=80, test_span=40, embargo_pct=0.0, proba=True,
    )
    assert {"y_true", "y_prob", "y_pred"} == set(out.columns)
    assert len(out) > 0
    acc = (out["y_pred"].astype(int) == out["y_true"].astype(int)).mean()
    assert acc > 0.9, f"learnable signal OOS accuracy too low: {acc}"
    assert ((out["y_prob"] >= 0.0) & (out["y_prob"] <= 1.0)).all()


def test_walk_forward_shuffled_is_chance():
    """Leakage canary: shuffled labels => OOS accuracy ≈ 0.5."""
    X, y, t1 = _learnable_dataset(seed=5)
    rng = np.random.default_rng(99)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    out = walk_forward_predict(
        X, y_shuf, t1, lambda: LogisticRegression(),
        train_min=80, test_span=40, embargo_pct=0.0, proba=True,
    )
    acc = (out["y_pred"].astype(int) == out["y_true"].astype(int)).mean()
    assert 0.4 <= acc <= 0.6, f"shuffled OOS accuracy should be ~0.5, got {acc}"


def test_walk_forward_returns_only_oos():
    X, y, t1 = _learnable_dataset(n=200)
    out = walk_forward_predict(
        X, y, t1, lambda: LogisticRegression(),
        train_min=60, test_span=30, embargo_pct=0.05,
    )
    # first OOS index must be at/after train_min + embargo
    embargo = int(200 * 0.05)
    first_pos = X.index.get_loc(out.index[0])
    assert first_pos >= 60 + embargo


# --------------------------------------------------------------------------- #
# 6. Classic metrics — sanity on known series
# --------------------------------------------------------------------------- #
def test_sharpe_known_series():
    # constant positive returns with tiny noise => high, positive Sharpe
    rng = np.random.default_rng(0)
    r = 0.001 + rng.normal(0, 1e-6, 252)
    sr = sharpe(r, periods=252)
    assert sr > 0
    # zero-vol series => nan
    assert np.isnan(sharpe(np.zeros(100)))
    # symmetric zero-mean noise => Sharpe near 0
    big = rng.normal(0, 0.01, 100_000)
    assert abs(sharpe(big)) < 0.5


def test_max_drawdown_known():
    # equity goes 1 -> 1.2 -> 0.6 -> 0.9 ; max DD = 0.6/1.2 - 1 = -0.5
    equity = pd.Series([1.0, 1.2, 0.6, 0.9])
    dd = max_drawdown(equity, is_returns=False)
    assert dd == pytest.approx(-0.5)
    # returns path: -0.5 drop after +0 => DD <= -0.5
    rets = pd.Series([0.0, -0.5, 0.5])
    assert max_drawdown(rets, is_returns=True) == pytest.approx(-0.5)
    # monotonically rising equity => zero drawdown
    assert max_drawdown(pd.Series([1, 2, 3, 4.0]), is_returns=False) == pytest.approx(0.0)


def test_profit_factor_expectancy_hitrate():
    r = pd.Series([1.0, -0.5, 2.0, -0.5])
    assert profit_factor(r) == pytest.approx(3.0)  # gains 3 / losses 1
    assert expectancy(r) == pytest.approx(0.5)
    assert hit_rate(r) == pytest.approx(0.5)
    assert np.isinf(profit_factor(pd.Series([1.0, 2.0])))  # no losses


def test_sortino_and_cagr():
    rng = np.random.default_rng(0)
    # positive drift WITH real downside days so downside deviation is defined
    r = 0.001 + rng.normal(0, 0.01, 252)
    assert (r < 0).any(), "fixture must contain losing days"
    assert sortino(r) > 0
    # all-positive returns => no downside => nan (documented behaviour)
    assert np.isnan(sortino(np.full(50, 0.001)))
    # cagr of constant 0 returns is 0
    assert cagr(np.zeros(252)) == pytest.approx(0.0)
    # 0.1% daily for a year compounds to a positive CAGR
    assert cagr(np.full(252, 0.001), periods=252) > 0
