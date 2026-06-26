"""Tests for the 'better long model' — vol-targeted risk-parity combiner.

These pin the three properties the model's honesty depends on:
  1. NO LOOKAHEAD — every weight/leverage uses only trailing info (``.shift(1)``); truncating
     future bars must not change any past combined return.
  2. RISK PARITY — weights are inverse-vol and sum to 1; the lower-vol stream gets the bigger weight.
  3. LEVERAGE CAP + DIVERSIFICATION — leverage never exceeds the cap, and combining two genuinely
     uncorrelated streams lifts the blend's Sharpe above the better single stream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.portfolio.long_model import vol_target_risk_parity


def _streams(n: int = 3000, seed: int = 7) -> dict[str, pd.Series]:
    """Two near-uncorrelated, EQUAL-Sharpe daily return streams with DIFFERENT vols.

    Equal Sharpe (mean scales with vol: both ≈ 1.3) is the clean diversification case — risk parity
    equalises their risk contribution and an uncorrelated blend strictly beats either single. The
    differing vols (0.6% vs 1.2%) keep the 'risk parity favours the low-vol book' property testable.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    a = pd.Series(rng.normal(0.0005, 0.006, n), index=idx)   # calm book (low vol)
    b = pd.Series(rng.normal(0.0010, 0.012, n), index=idx)   # punchy book (2× vol, 2× drift)
    return {"a": a, "b": b}


def _sharpe(r: pd.Series) -> float:
    r = r.fillna(0.0)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def test_no_lookahead_truncation_invariance():
    """Truncating the tail of the inputs must not change any earlier combined return."""
    s = _streams()
    full = vol_target_risk_parity(s)
    cut = vol_target_risk_parity({k: v.iloc[:900] for k, v in s.items()})
    common = full.combined.index.intersection(cut.combined.index)
    assert len(common) >= 800
    np.testing.assert_allclose(
        full.combined.loc[common].to_numpy(), cut.combined.loc[common].to_numpy(), atol=1e-12)


def test_risk_parity_weights_sum_to_one_and_favour_low_vol():
    """Each row's weights sum to 1; the calmer (lower-vol) stream carries the larger weight."""
    s = _streams()
    res = vol_target_risk_parity(s)
    w = res.weights.dropna()
    row_sums = w.sum(axis=1)
    settled = row_sums[row_sums > 0]            # ignore the warm-up rows (weights NaN -> 0)
    np.testing.assert_allclose(settled.to_numpy(), 1.0, atol=1e-9)
    # 'a' is the low-vol stream, so risk parity should give it the heavier average weight
    assert res.weights["a"].mean() > res.weights["b"].mean()


def test_leverage_never_exceeds_cap():
    s = _streams()
    cap = 2.5
    res = vol_target_risk_parity(s, lev_cap=cap)
    assert float(res.leverage.max()) <= cap + 1e-9
    assert (res.leverage >= 0).all()


def test_diversification_lifts_sharpe_over_best_single():
    """Two uncorrelated streams, vol-targeted-risk-parity → blend Sharpe ≥ the better single."""
    s = _streams()
    res = vol_target_risk_parity(s)
    best_single = max(_sharpe(s["a"]), _sharpe(s["b"]))
    assert res.stats["sharpe"] > best_single
    # vol-targeting should land realised vol within a sane band of the 10% target
    assert 0.04 < res.stats["vol_ann"] < 0.20


def test_single_stream_is_handled():
    """A one-book 'blend' is just that book vol-targeted — must run and stay leverage-capped."""
    s = _streams()
    res = vol_target_risk_parity({"a": s["a"]}, lev_cap=3.0)
    assert res.combined.notna().all()
    assert float(res.leverage.max()) <= 3.0 + 1e-9
    settled = res.weights["a"][res.weights["a"] > 0]   # post warm-up (warm-up weights are 0)
    np.testing.assert_allclose(settled.to_numpy(), 1.0, atol=1e-9)
