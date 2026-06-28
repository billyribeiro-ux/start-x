"""Tests for the swing scanner P1: PIT feature store + Layer-0 tradeability gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startx.scanner.feature_store import (
    LEAD_SWING,
    Feature,
    FeatureSet,
)
from startx.scanner.tradeability import MIN_DOLLAR_VOL, score_symbol


# ----------------------------------------------------------------------------- feature store
def test_feature_seam_is_flagged_not_faked():
    f = Feature.seam("dealer_gex", "2024-01-02", LEAD_SWING, "L0_tradeability", "OPRA")
    assert f.available is False
    assert np.isnan(f.value)
    assert f.source.startswith("seam:")


def test_featureset_rejects_leakage():
    """A feature timestamped AFTER the decision bar is leakage and must raise."""
    fs = FeatureSet(symbol="X", as_of=pd.Timestamp("2024-01-02"))
    fs.add(Feature("ok", 1.0, pd.Timestamp("2024-01-02"), LEAD_SWING, "L0_tradeability", "fmp_eod"))
    with pytest.raises(ValueError, match="LEAKAGE"):
        fs.add(Feature("bad", 1.0, pd.Timestamp("2024-01-03"), LEAD_SWING, "L0_tradeability", "fmp_eod"))


def test_feature_unknown_layer_rejected():
    with pytest.raises(ValueError):
        Feature("x", 1.0, pd.Timestamp("2024-01-02"), LEAD_SWING, "L9_bogus", "fmp_eod")


# ----------------------------------------------------------------------------- tradeability gate
def _frame(n=400, px=100.0, vol=1e7, seed=0, beta_to=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    if beta_to is not None:
        r = 0.8 * beta_to["close"].pct_change().reindex(idx).fillna(0).to_numpy() + rng.normal(0, 0.003, n)
    else:
        r = rng.normal(0, 0.012, n)                         # idiosyncratic
    close = px * np.cumprod(1 + r)
    return pd.DataFrame({"date": idx, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": vol / close})


def _factors():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=400)
    spy = 400 * np.cumprod(1 + rng.normal(0.0003, 0.009, 400))
    f = {"SPY": pd.DataFrame({"date": idx, "close": spy, "high": spy, "low": spy, "open": spy,
                              "volume": 1e6})}
    for k in ["TLT", "HYG", "UUP"]:
        s = 100 * np.cumprod(1 + rng.normal(0, 0.006, 400))
        f[k] = pd.DataFrame({"date": idx, "close": s, "high": s, "low": s, "open": s, "volume": 1e6})
    return f, idx[-1]


def test_liquid_member_is_tradeable():
    factors, asof = _factors()
    px = _frame(vol=5e8, beta_to=factors["SPY"])             # liquid + tracks the market
    t = score_symbol("BIGCO", px, factors, asof, is_member=True, is_etf=False)
    assert t.tradeable and t.dollar_vol >= MIN_DOLLAR_VOL


def test_illiquid_nonmember_idiosyncratic_rejected():
    factors, asof = _factors()
    px = _frame(vol=5e6, seed=3)                              # thin + idiosyncratic, not a member
    t = score_symbol("PENNY", px, factors, asof, is_member=False, is_etf=False)
    assert not t.tradeable
    assert any("illiquid" in r or "no front" in r for r in t.reasons)


def test_member_with_low_macro_r2_still_tradeable():
    """A liquid S&P member must NOT be rejected just for low macro-R² (index flow IS the hand)."""
    factors, asof = _factors()
    px = _frame(vol=3e8, seed=9)                              # liquid, idiosyncratic, but a MEMBER
    t = score_symbol("DEFENSIVE", px, factors, asof, is_member=True, is_etf=False)
    assert t.tradeable                                        # passes via index/ETF flow


def test_gex_seam_present_and_unavailable():
    factors, asof = _factors()
    px = _frame(vol=3e8, beta_to=factors["SPY"])
    t = score_symbol("X", px, factors, asof, is_member=True, is_etf=False)
    seams = {f.name for f in t.features.seams()}
    assert {"dealer_gex", "options_liquidity"} <= seams      # surfaced, not faked
