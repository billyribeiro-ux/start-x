"""The methodology gate: noise must FAIL, a planted edge must PASS.

These tests assert the qedge survival gate behaves the way the contract demands:
white-noise returns produce a low Deflated Sharpe and a Probability of Backtest
Overfitting near 0.5 (selection is no better than chance) -> ``passed is False``;
a strong planted signal produces a high DSR and low PBO -> ``passed is True``. We
also check the Combinatorial Purged CV wired from config has the right path count
and never leaks (no train/test index overlap).
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from qedge.config import QedgeConfig
from qedge.validation import GateResult, make_cpcv, survival_gate


def _daily_t1(n: int) -> pd.Series:
    """A daily-frequency t1 (each label closes one trading day later)."""
    entries = pd.to_datetime(pd.bdate_range("2019-01-02", periods=n))
    return pd.Series(entries.shift(1, freq="B"), index=entries)


def test_make_cpcv_path_count_matches_combinatorics(cfg: QedgeConfig) -> None:
    t1 = _daily_t1(600)
    cv = make_cpcv(t1)
    assert cv.n_groups == cfg.validation.cpcv_n_groups
    assert cv.n_test_groups == cfg.validation.cpcv_n_test_groups
    expected = comb(
        cfg.validation.cpcv_n_groups - 1, cfg.validation.cpcv_n_test_groups - 1
    )
    assert cv.n_paths == expected


def test_make_cpcv_folds_have_no_train_test_overlap() -> None:
    t1 = _daily_t1(600)
    cv = make_cpcv(t1)
    features = pd.DataFrame({"f": np.arange(len(t1))}, index=t1.index)
    n_splits = 0
    for train_idx, test_idx in cv.split(features):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert test_idx.size > 0
        assert train_idx.size > 0
        n_splits += 1
    assert n_splits == cv.get_n_splits()


def test_white_noise_fails_the_gate(seeded_rng: np.random.Generator) -> None:
    """Many competing random configs -> low DSR, PBO ~0.5 -> LIKELY OVERFIT."""
    n_obs, n_configs = 750, 20
    configs = [seeded_rng.normal(0.0, 0.01, size=n_obs) for _ in range(n_configs)]
    selected, competitors = configs[0], configs[1:]

    result = survival_gate(selected, n_trials=n_configs, trial_returns=competitors)

    assert isinstance(result, GateResult)
    assert result.deflated_sharpe < 0.5  # no real edge
    assert 0.35 <= result.pbo <= 0.65  # selection no better than chance
    assert result.passed is False
    assert result.verdict == "LIKELY OVERFIT"


def test_strong_signal_passes_the_gate(seeded_rng: np.random.Generator) -> None:
    """A high-Sharpe planted edge -> high DSR, low PBO -> ROBUST."""
    n_obs, n_configs = 750, 20
    noise = [seeded_rng.normal(0.0, 0.01, size=n_obs) for _ in range(n_configs)]
    # Strong positive drift with modest vol => high, persistent Sharpe.
    signal = seeded_rng.normal(0.0015, 0.005, size=n_obs)

    result = survival_gate(signal, n_trials=n_configs, trial_returns=noise)

    assert result.deflated_sharpe >= 0.95  # clears dsr_min
    assert result.pbo <= 0.5  # IS winner stays an OOS winner
    assert result.passed is True
    assert result.verdict == "ROBUST"


def test_gate_uses_config_thresholds(cfg: QedgeConfig) -> None:
    """passed must be exactly (dsr >= dsr_min) AND (pbo <= pbo_max)."""
    signal = np.full(500, 0.001) + np.tile([1e-4, -1e-4], 250)
    result = survival_gate(signal, n_trials=1)
    expected = (
        result.deflated_sharpe >= cfg.validation.dsr_min
        and result.pbo <= cfg.validation.pbo_max
    )
    assert result.passed is expected


def test_gate_handles_degenerate_short_input() -> None:
    """Too few observations -> defined, failing result (never crashes)."""
    result = survival_gate(np.array([0.01]), n_trials=1)
    assert result.passed is False
    assert result.n_obs == 1
    assert np.isnan(result.deflated_sharpe)
