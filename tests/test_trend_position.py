"""Tests for System #3 — the 200-SMA ±band position core (`strategy/trend_position.py`).

Covers: point-in-time stability of the regime state (no lookahead), the hysteresis band, and the
position_book ledger / buy-and-hold benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.trend_position import position_book, position_state


def _frame(closes: np.ndarray, start: str = "2015-01-01") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": np.full(n, 1e6),
    })


def _trend_then_crash(seed: int = 0) -> np.ndarray:
    """A long uptrend, a sharp drawdown, then a recovery — enough to force ≥1 full long stint."""
    rng = np.random.default_rng(seed)
    up = np.linspace(100, 200, 250) + rng.normal(0, 0.5, 250)
    down = np.linspace(200, 120, 120) + rng.normal(0, 0.5, 120)
    up2 = np.linspace(120, 190, 180) + rng.normal(0, 0.5, 180)
    return np.concatenate([up, down, up2]).astype(float)


def test_position_state_is_point_in_time():
    """Truncating future bars must not change any past state value (the loop is causal + shifted)."""
    df = _frame(_trend_then_crash())
    full = position_state(df, sma_len=200, band=0.03)
    cut = 420
    trunc = position_state(df.iloc[:cut].reset_index(drop=True), sma_len=200, band=0.03)
    assert (full.iloc[:cut].to_numpy() == trunc.to_numpy()).all(), "past state changed when future removed → lookahead"
    # state is first-day-False (shifted) and never NaN
    assert full.dtype == bool and not full.isna().any() and not bool(full.iloc[0])


def test_hysteresis_band_holds_inside_the_band():
    """Long is entered above SMA·(1+band), kept inside the band, dropped below SMA·(1-band)."""
    # 220 flat bars to seed the SMA at ~100, then up through the upper band, a dip that stays
    # *inside* the band, then a break below the lower band.
    base = np.full(220, 100.0)
    up = np.linspace(100, 130, 40)        # well above 100·1.03 = 103
    dip = np.full(20, 121.0)              # SMA is ~rising but 121 stays above the lower band → hold long
    crash = np.linspace(121, 70, 60)      # breaks below SMA·(1-band) → flat
    df = _frame(np.concatenate([base, up, dip, crash]))
    st = position_state(df, sma_len=200, band=0.03)
    # during the clear uptrend (after warm-up + cross) we are long
    assert bool(st.iloc[270]), "should be long in the established uptrend"
    # the small dip does not flip us flat (hysteresis)
    assert bool(st.iloc[285]), "a dip that stays inside the band must NOT flip to flat"
    # the deep crash flips us flat
    assert not bool(st.iloc[-1]), "a break below the lower band must go flat"


def test_position_book_ledger_and_benchmark():
    df = _frame(_trend_then_crash())
    res = position_book(df, sma_len=200, band=0.03, start="2015-01-01", end="2017-12-31")
    led, s, b = res.ledger, res.stats, res.benchmark
    assert not led.empty
    # every booked trade is a real long stint: positive hold, entry before exit, outcome matches ret sign
    assert (led["bars_held"] > 0).all()
    assert (pd.to_datetime(led["entry_date"]) <= pd.to_datetime(led["exit_date"])).all()
    for _, r in led.iterrows():
        sign = "WIN" if r["ret"] > 0 else ("SCRATCH" if r["ret"] == 0 else "LOSS")
        assert r["outcome"] == sign
    # the book steps to cash (exposure strictly between 0 and 1) and reports a B&H benchmark
    assert 0.0 < s["exposure"] < 1.0
    assert {"cagr", "max_drawdown", "ann_sharpe", "calmar"} <= set(b)
    # drawdown defense: the position core's drawdown is no worse than buy-and-hold's
    assert s["max_drawdown"] >= b["max_drawdown"]  # less negative (shallower) or equal


def test_band_reduces_turnover_vs_no_band():
    """A wider hysteresis band must not INCREASE the number of long stints (it cuts whipsaw)."""
    df = _frame(_trend_then_crash())
    n_no_band = position_book(df, band=0.0, start="2015-01-01", end="2017-12-31").ledger.shape[0]
    n_band = position_book(df, band=0.03, start="2015-01-01", end="2017-12-31").ledger.shape[0]
    assert n_band <= n_no_band
