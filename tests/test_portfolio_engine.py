"""Regression tests for the portfolio engine's stats windowing.

Pins the fix for the in-window Sharpe bug (found by the long_swing firewall drill): headline
stats (Sharpe / total_return / maxDD) must be computed over the STUDY WINDOW [start, end], not
over the full price index. Folding the flat, pre-window days (the book is flat before its first
in-window entry) into the daily series dilutes the annualised Sharpe by ~sqrt(total/in_window)
days — e.g. a true 1.07 Sharpe was being reported as 0.50 when the full 1993→2026 SPY index was
passed with start=2019.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.portfolio import run_portfolio

TRADING_DAYS = 252


def _synthetic_spy(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A deterministic OHLCV frame with enough variation to generate trades."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "date": pd.bdate_range("2018-01-01", periods=n),
        "open": openp, "high": high, "low": low, "close": close,
        "volume": np.full(n, 1e6),
    })


def _ann_sharpe(eq: pd.Series) -> float:
    dr = eq.pct_change().fillna(0.0)
    sd = dr.std(ddof=1)
    return float(dr.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def test_stats_are_window_scoped_not_full_index():
    """The reported Sharpe must equal the in-window Sharpe of the returned equity curve, and
    must be materially higher than the full-index (flat-pre-window-diluted) Sharpe."""
    spy = _synthetic_spy()
    spy.attrs["symbol"] = "SPY"
    # one open position per sleeve; a "down day" dip sleeve fires often enough to trade.
    sleeves = {"dip": lambda s, a: s["close"] < s["close"].shift(1)}

    # Window covers only the LAST ~40% of the frame -> ~60% of days are pre-window flat.
    start = spy["date"].iloc[300]
    end = spy["date"].iloc[-1]
    res = run_portfolio(spy, {}, sleeves, exits={"dip": (1.0, 3.0, 10)},
                        start=start, end=end, gold_calm=False)

    assert res.stats["n"] > 0, "test needs at least one trade to be meaningful"

    # In-window Sharpe recomputed independently from the returned (full) equity curve.
    eqw = res.equity[(res.equity.index >= start) & (res.equity.index <= end)]
    eqw = eqw / eqw.iloc[0]
    manual_window = _ann_sharpe(eqw)
    assert np.isfinite(res.stats["ann_sharpe"])
    assert abs(res.stats["ann_sharpe"] - manual_window) < 1e-6

    # The naive full-index Sharpe (what the bug reported) is materially LOWER because of the
    # flat pre-window days. The fix must not regress back to it.
    full_index = _ann_sharpe(res.equity)
    assert res.stats["ann_sharpe"] > full_index * 1.2


def test_leading_flat_history_does_not_change_total_return():
    """total_return is the in-window compounded return — independent of how the curve is rebased."""
    spy = _synthetic_spy()
    spy.attrs["symbol"] = "SPY"
    sleeves = {"dip": lambda s, a: s["close"] < s["close"].shift(1)}
    start, end = spy["date"].iloc[300], spy["date"].iloc[-1]
    res = run_portfolio(spy, {}, sleeves, exits={"dip": (1.0, 3.0, 10)},
                        start=start, end=end, gold_calm=False)
    eqw = res.equity[(res.equity.index >= start) & (res.equity.index <= end)]
    expected = float(eqw.iloc[-1] / eqw.iloc[0] - 1.0)
    assert abs(res.stats["total_return"] - expected) < 1e-9
