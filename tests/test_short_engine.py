"""Tests for the short-the-index engine (mirror exit mechanics + bear-regime gating)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.short_engine import (
    overbought_short,
    short_backtest,
)


def _frame(closes: np.ndarray, start: str = "2015-01-01") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": closes, "high": closes * 1.002, "low": closes * 0.998,
        "close": closes, "volume": np.full(n, 1e6),
    })


def _downtrend(seed: int = 0) -> np.ndarray:
    """Build the 200-SMA up, then a sustained sawtooth DOWNtrend with sharp 2-day pops — the pops
    push RSI(2) > 90 *below* the SMA, i.e. the 'sell the overbought bounce in a bear' setup."""
    rng = np.random.default_rng(seed)
    up = np.linspace(100, 200, 240) + rng.normal(0, 0.3, 240)       # build the 200-SMA up to ~150
    base = np.linspace(200, 95, 240)                                # steady decline through the SMA
    bounce = np.zeros(240)
    for k in range(8, 240, 14):
        bounce[k:k + 2] += base[k] * 0.05                           # sharp 2-day +5% pops → RSI2>90
    down = base + bounce + rng.normal(0, 0.3, 240)
    return np.concatenate([up, down]).astype(float)


def test_overbought_short_only_fires_below_the_trend():
    """The signal must NEVER fire while price is above its 200-SMA (no shorting an uptrend)."""
    df = _frame(_downtrend())
    sig = overbought_short(df, rsi_threshold=90, trend_ma=200)
    sma = df["close"].rolling(200).mean()
    fired = df.index[sig.to_numpy()]
    assert len(fired) > 0, "expected some signals in the downtrend leg"
    assert (df.loc[fired, "close"].to_numpy() < sma.loc[fired].to_numpy()).all(), \
        "a short fired while above the 200-SMA (uptrend) — forbidden"


def test_short_pnl_sign_and_stop_above_entry():
    """A short profits when price FALLS; its hard stop sits ABOVE entry; outcome matches ret sign."""
    df = _frame(_downtrend())
    sig = overbought_short(df, rsi_threshold=85, trend_ma=200)
    res = short_backtest(df, sig, stop_mult=1.0, trail_mult=3.0, max_days=10)
    led = res.ledger
    assert not led.empty
    # hard stop is ABOVE entry for a short (the mirror of a long's stop-below)
    assert (led["stop_price"] > led["entry_price"]).all()
    # a short "win" means it covered BELOW entry (price fell); pnl sign matches outcome
    for _, r in led.iterrows():
        assert (r["pnl_per_share"] > 0) == (r["outcome"] == "WIN") or r["outcome"] == "SCRATCH"
        if r["outcome"] == "WIN":
            assert r["exit_price"] < r["entry_price"]
    assert (led["bars_held"] > 0).all()
    assert {"n", "win_rate", "profit_factor", "expectancy_pct"} <= set(res.stats)


def test_short_is_point_in_time():
    """Truncating future bars must not change a past trade (entries use only trailing data)."""
    df = _frame(_downtrend())
    sig_full = overbought_short(df, rsi_threshold=90, trend_ma=200)
    sig_cut = overbought_short(df.iloc[:430].reset_index(drop=True), rsi_threshold=90, trend_ma=200)
    assert (sig_full.iloc[:430].to_numpy() == sig_cut.to_numpy()).all(), "signal changed when future removed"


def test_no_overlapping_shorts():
    """One position at a time — a new entry cannot open before the prior trade's exit."""
    df = _frame(_downtrend())
    sig = overbought_short(df, rsi_threshold=80, trend_ma=200)
    led = short_backtest(df, sig, max_days=10).ledger
    if len(led) >= 2:
        e = pd.to_datetime(led["entry_date"]).to_numpy()
        x = pd.to_datetime(led["exit_date"]).to_numpy()
        assert (e[1:] > x[:-1]).all(), "a short opened before the previous one closed"
