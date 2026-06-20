"""Feature-layer unit tests: PIT (no-lookahead) guarantee, technical sanity, leakage guard.

These are fast and synthetic. One optional network smoke test builds a tiny real matrix and is
skipped when no FMP key is configured.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from startx.data.pit import in_window, trailing
from startx.data.prices import _enrich
from startx.events.detect import compute_signals
from startx.features.flow import flow_features
from startx.features.regime import regime_features
from startx.features.technical import technical_features


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _uptrend(n: int = 320, seed: int = 0) -> pd.DataFrame:
    """A steady upward drift with mild noise -> positive momentum, RSI > 50."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0015, 0.005, n)  # positive drift
    close = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": 1_000_000.0, "vwap": close, "changePercent": 0.0,
    })
    return _enrich(df)


# --------------------------------------------------------------------------------------------
# PIT helper unit tests
# --------------------------------------------------------------------------------------------
def test_pit_helpers_tolerate_garbage():
    assert in_window(None, "2024-01-01", "2024-02-01").empty
    assert trailing(None, "2024-01-01", 30).empty
    assert in_window(pd.DataFrame(), "2024-01-01", "2024-02-01").empty
    # missing ts column -> empty
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert trailing(df, "2024-01-01", 30).empty


def test_trailing_is_inclusive_and_bounded():
    ts = pd.to_datetime(["2023-12-25", "2024-01-01", "2024-01-31", "2024-02-05"])
    df = pd.DataFrame({"ts": ts, "v": [1, 2, 3, 4]})
    win = trailing(df, "2024-01-31", 30)
    # window is [Jan-01, Jan-31] inclusive: Dec-25 is >30d before (out), Feb-05 future (out),
    # Jan-01 is exactly 30d before (in), Jan-31 == asof (in).
    assert set(win["v"]) == {2, 3}


# --------------------------------------------------------------------------------------------
# (a) PIT GUARANTEE: a catalyst stamped AFTER t must not appear; a before-row must.
# --------------------------------------------------------------------------------------------
def test_flow_features_no_lookahead():
    asof = pd.Timestamp("2024-03-15")
    before = pd.Timestamp("2024-03-10")   # visible at asof
    after = pd.Timestamp("2024-03-20")    # future -> must NOT be visible at asof

    # Two insider BUYS: one before, one after the as-of date.
    insider = pd.DataFrame({
        "ts": [before, after],
        "acquisitionOrDisposition": ["A", "A"],
        "securitiesTransacted": [1000, 5000],
    })
    catalysts = {"insider": insider}

    feats = flow_features([asof], catalysts)
    row = feats.iloc[0]
    # Only the BEFORE buy (1000 shares, 1 buy) may count; the AFTER buy is invisible.
    assert row["insider_net_30"] == 1000.0
    assert row["insider_buy_cnt_30"] == 1
    assert row["insider_buy_cnt_90"] == 1

    # If we move the as-of date past the second trade, it becomes visible.
    feats_later = flow_features([pd.Timestamp("2024-03-25")], catalysts)
    assert feats_later.iloc[0]["insider_net_30"] == 6000.0
    assert feats_later.iloc[0]["insider_buy_cnt_30"] == 2


def test_flow_news_and_earnings_pit():
    asof = pd.Timestamp("2024-06-10")
    news = pd.DataFrame({
        "ts": pd.to_datetime(["2024-06-08", "2024-06-09", "2024-06-12"]),  # last is future
        "title": ["a", "b", "c"],
    })
    earnings = pd.DataFrame({
        "ts": pd.to_datetime(["2024-05-01", "2024-08-01"]),  # past reported, future scheduled
        "epsActual": [1.20, np.nan],
        "epsEstimated": [1.00, 1.30],
    })
    feats = flow_features([asof], {"news": news, "earnings": earnings}).iloc[0]
    # only the two pre-asof headlines within 5d count
    assert feats["news_cnt_5"] == 2
    assert feats["news_cnt_20"] == 2
    # last reported earnings was 2024-05-01 -> 40 days before; surprise = (1.2-1.0)/1.0 = 0.2
    assert feats["days_since_last_earnings"] == 40.0
    assert feats["last_eps_surprise"] == pytest.approx(0.2)
    # next scheduled earnings 2024-08-01 is future-but-public -> PIT-OK
    assert feats["days_to_next_earnings"] == pytest.approx((pd.Timestamp("2024-08-01") - asof).days)


# --------------------------------------------------------------------------------------------
# (b) TECHNICAL: uptrend -> momentum > 0, RSI > 50, no close/fwd columns.
# --------------------------------------------------------------------------------------------
def test_technical_features_uptrend():
    signals = compute_signals(_uptrend(), market=None)
    tech = technical_features(signals)
    last = tech.iloc[-1]
    for h in (5, 21, 63, 126, 252):
        assert last[f"mom_{h}"] > 0, f"momentum {h} should be positive in an uptrend"
    assert last["rsi_14"] > 50, "RSI should be above 50 in an uptrend"
    # in an uptrend price sits above its SMAs and near the 52w high
    assert last["dist_sma_50"] > 0
    assert last["dist_52w_high"] > -0.10
    # leakage guards
    assert "close" not in tech.columns
    assert not any(c.startswith("fwd_") for c in tech.columns)
    assert not any(c in tech.columns for c in ("open", "high", "low", "volume"))


def test_technical_passthrough_signals():
    signals = compute_signals(_uptrend(), market=None)
    tech = technical_features(signals)
    for col in ("ar_z", "vol_z", "gap"):
        assert col in tech.columns


# --------------------------------------------------------------------------------------------
# (c) NO feature column may start with 'fwd_'.
# --------------------------------------------------------------------------------------------
def test_no_forward_columns_anywhere():
    signals = compute_signals(_uptrend(), market=None)
    tech = technical_features(signals)
    flow = flow_features(signals["date"], {})
    bench = _uptrend(seed=2)
    vix = _uptrend(seed=3)
    regime = regime_features(signals["date"], bench, vix)
    for frame in (tech, flow, regime):
        assert not any(c.startswith("fwd_") for c in frame.columns)


def test_regime_features_shape_and_pit():
    signals = compute_signals(_uptrend(), market=None)
    bench = _uptrend(seed=5)
    vix = _uptrend(seed=6)
    regime = regime_features(signals["date"], bench, vix)
    assert len(regime) == signals["date"].nunique()
    for c in ("vix_level", "vix_chg_5d", "vix_pctrank_252", "bench_ret_20",
              "bench_ret_63", "bench_above_sma200", "bench_vol_21"):
        assert c in regime.columns
    # pct-rank stays within [0, 1]
    pr = regime["vix_pctrank_252"].dropna()
    assert ((pr >= 0) & (pr <= 1)).all()


# --------------------------------------------------------------------------------------------
# optional network smoke test
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="no FMP_API_KEY set")
def test_build_matrix_smoke():
    from startx.features.assemble import build_feature_matrix
    from startx.fmp.client import FMPClient

    with FMPClient() as client:
        m = build_feature_matrix("AAPL", "2024-01-01", "2024-03-31", client=client)
    assert not m.empty
    assert not any(c.startswith("fwd_") for c in m.columns)
    assert "symbol" in m.columns and (m["symbol"] == "AAPL").all()
    assert "sector" in m.columns
