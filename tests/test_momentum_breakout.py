"""Unit tests for the momentum-breakout swing system (``strategy/momentum_breakout.py``).

This is the strongest standalone edge on the 2019->2026 relearn (per the module docstring) yet had
ZERO dedicated coverage. Every case is deterministic and runs on hand-built synthetic OHLCV --
no network, no FMP.

The documented contract we pin (module docstring + ``breakout_signals``):
  ``breakout_signals`` fires when the close makes a *fresh* ``lookback``-day high (a new cross, not
  still-extended) while above the ``trend_ma``-day SMA. It must NOT fire when price is below trend,
  nor on a bar that was already at the rolling high yesterday (no double-fire), and the output is a
  bool Series aligned 1:1 to the (sorted) input.

The ``backtest`` smoke pins only DOCUMENTED, sleeve-agnostic behavior: a coherent ledger (entry
before exit, exit reason from the known set, one position at a time) and -- the point of this file --
that every outcome label is in the three-way {WIN, SCRATCH, LOSS} set (the desk's LOCKED rule:
a +1-ATR / breakeven exit is a WIN/SCRATCH, never a LOSS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.momentum_breakout import atr, backtest, breakout_signals


def _uptrend_frame(n: int = 320, start: str = "2021-01-04") -> pd.DataFrame:
    """A steady, low-noise uptrend so ``close > SMA(200)`` holds after warm-up.

    Each bar gets a symmetric +/-0.5 range around the close. The close rises one unit per bar, so
    EVERY bar is technically a fresh 20-day high -- callers that want a *single* clean breakout
    flatten a stretch first (see ``_flat_then_breakout``).
    """
    dates = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 100.0 + n - 1, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _flat_then_breakout(n: int = 320, breakout_at: int = 300) -> pd.DataFrame:
    """A rising base that warms up SMA(200) well below price, then a long FLAT plateau (so no NEW
    20-day highs occur on the plateau), a one-bar dip the bar *before* the breakout (so the cross is
    genuinely *fresh*: yesterday was below its rolling high), then ONE decisive new high.

    ``breakout_signals`` requires (a) close >= rolling 20-day high, (b) the prior close was *below*
    its own rolling high (the cross is new, not still-extended), and (c) close > SMA(200). This frame
    satisfies all three at exactly ``breakout_at`` and nowhere else.
    """
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = np.empty(n)
    close[:220] = np.linspace(100.0, 180.0, 220)   # rising base -> SMA200 warms up below 180
    close[220:] = 180.0                            # flat plateau -> no fresh 20-day highs here
    close[breakout_at - 1] = 178.0                 # a one-bar dip -> prior bar is below its high
    close[breakout_at] = 185.0                     # the single decisive breakout above the plateau
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


# --------------------------------------------------------------------------- ATR sanity


def test_atr_is_positive_and_warmup_is_nan():
    """ATR (Wilder EWM of true range) is NaN through warm-up, then strictly positive on a frame with
    a non-degenerate range."""
    p = _uptrend_frame()
    a = atr(p, window=14)
    assert a.iloc[:13].isna().all()        # min_periods=window -> first 13 are NaN
    assert (a.iloc[14:] > 0).all()
    assert len(a) == len(p)


# --------------------------------------------------------------------------- positive case


def test_breakout_fires_on_a_fresh_new_high_above_trend():
    """POSITIVE: the single decisive new 20-day high above the plateau, with close > SMA(200), is the
    firing bar -- and it is the ONLY fire (the flat plateau bars before it are at, not crossing, the
    rolling high, so they are not *fresh*)."""
    p = _flat_then_breakout(breakout_at=300)
    sig = breakout_signals(p, lookback=20, trend_ma=200)

    assert bool(sig.iloc[300]) is True
    assert int(sig.sum()) == 1
    # Price is genuinely above its 200-SMA at the breakout (trend leg satisfied).
    sma = p["close"].rolling(200).mean()
    assert p.loc[300, "close"] > sma.iloc[300]
    # The flat plateau bars (already sitting at the rolling high) are NOT fresh crosses -> silent.
    assert not sig.iloc[221:300].any()


# --------------------------------------------------------------------------- negative cases


def test_breakout_silent_below_trend_ma():
    """NEGATIVE (trend leg fails): a fresh 20-day high inside a DOWNtrend (close < SMA200) must NOT
    fire -- the system buys strength only while the trend is intact."""
    n = 320
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = np.linspace(200.0, 100.0, n)  # steady DOWNtrend -> close < SMA(200) after warm-up
    i = 300
    # Engineer a fresh local 20-day high at i (a small up-pop) that is still below the falling SMA200.
    close[i - 25 : i] = close[i]          # a flat shelf at the local level
    close[i - 1] = close[i] - 2.0         # dip the bar before -> the cross at i is genuinely fresh
    close[i] = close[i - 25] + 1.0        # tiny pop above the shelf -> a new 20-day high
    p = pd.DataFrame(
        {"date": dates, "high": close + 0.5, "low": close - 0.5, "close": close}
    )
    hh = p["close"].rolling(20).max()
    sma = p["close"].rolling(200).mean()
    # The fresh-high leg is satisfied ...
    assert p.loc[i, "close"] >= hh.iloc[i]
    assert p.loc[i - 1, "close"] < hh.iloc[i - 1]
    # ... but price is below trend, so the signal stays silent (no fire anywhere in the downtrend).
    assert p.loc[i, "close"] < sma.iloc[i]
    assert bool(breakout_signals(p).iloc[i]) is False
    assert int(breakout_signals(p).sum()) == 0


def test_breakout_does_not_refire_while_still_extended():
    """NEGATIVE (freshness leg): the bar AFTER a breakout that merely *holds* at the new-high level
    (no fresh cross) must not fire again -- the rule is 'fresh cross, not still-extended'."""
    p = _flat_then_breakout(breakout_at=300)
    # The bar after the breakout holds flat at the breakout level: it is at the rolling high but did
    # NOT cross to a *new* high, so it is not a fresh breakout.
    p.loc[301, "close"] = p.loc[300, "close"]
    p.loc[301, "high"] = p.loc[301, "close"] + 0.5
    p.loc[301, "low"] = p.loc[301, "close"] - 0.5
    sig = breakout_signals(p, lookback=20, trend_ma=200)
    assert bool(sig.iloc[300]) is True   # the decisive cross fires
    assert bool(sig.iloc[301]) is False  # holding at the high is not a fresh cross -> silent


# --------------------------------------------------------------------------- shape / NaN


def test_signal_alignment_length_and_nan_safety():
    """Output is a bool Series aligned 1:1 with the (sorted) input, NaN-safe (warm-up -> False), and
    invariant under a shuffled input because the function sorts by date internally."""
    p = _flat_then_breakout(breakout_at=300)
    sig = breakout_signals(p)
    assert sig.dtype == bool
    assert not sig.isna().any()
    assert len(sig) == len(p)
    # First 199 bars have no 200-SMA -> all False.
    assert not sig.iloc[:199].any()

    shuffled = p.sample(frac=1.0, random_state=7).reset_index(drop=True)
    sig_s = breakout_signals(shuffled)
    sorted_dates = p.sort_values("date").reset_index(drop=True)["date"]
    assert sorted_dates[sig.values].tolist() == sorted_dates[sig_s.values].tolist()


# --------------------------------------------------------------------------- backtest smoke


def _coherent_ledger_asserts(trades: pd.DataFrame) -> None:
    """Shared invariants for any non-empty momentum ledger."""
    assert not trades.empty
    # Outcome labels obey the desk's THREE-way rule -- never the old binary {WIN, LOSS}.
    assert set(trades["outcome"]).issubset({"WIN", "SCRATCH", "LOSS"})
    # Exit reason is from the documented set.
    assert set(trades["exit_reason"]).issubset({"chandelier", "stop", "target", "time"})
    # Every trade exits on/after entry, and the LOCKED trade-sheet columns are present.
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    assert (trades["bars_held"] >= 0).all()
    for col in ("symbol", "entry_price", "stop_price", "exit_price", "pnl_per_share", "outcome"):
        assert col in trades.columns


def test_backtest_smoke_produces_a_coherent_ledger():
    """A frame with breakouts produces a coherent ledger; stats agree with the rows."""
    p = _flat_then_breakout(breakout_at=300)
    p.attrs["symbol"] = "TEST"
    trades, stats = backtest(p, lookback=20, trend_ma=200, max_days=15)
    _coherent_ledger_asserts(trades)
    assert trades["symbol"].eq("TEST").all()
    assert stats["n"] == len(trades)
    # One position at a time: no entry opens while the prior trade is still live.
    ed = trades.sort_values("entry_date")
    assert (ed["entry_date"].iloc[1:].values >= ed["exit_date"].iloc[:-1].values).all()


def test_backtest_outcome_label_is_three_way_and_matches_sign():
    """The outcome label is consistent with the sign of the net (cost-adjusted) per-share return:
    a strictly profitable net is WIN, a strictly negative net is LOSS -- and a breakeven is SCRATCH,
    never LOSS (the LOCKED rule). We assert the mapping the production engine uses."""
    p = _flat_then_breakout(breakout_at=300)
    trades, _ = backtest(p, max_days=15)
    assert not trades.empty
    cost = 2.0 / 10000.0
    for _, t in trades.iterrows():
        net = (t["exit_price"] / t["entry_price"] - 1) - cost
        expected = "WIN" if net > 0 else "SCRATCH" if net == 0 else "LOSS"
        assert t["outcome"] == expected


def test_backtest_empty_when_no_breakouts():
    """A frame that never makes a fresh new high above trend yields an empty ledger and n==0 stats
    (the no-trade path is clean, not an exception)."""
    n = 320
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = np.linspace(200.0, 100.0, n)  # monotone DOWNtrend: never a fresh high above SMA200
    p = pd.DataFrame(
        {"date": dates, "high": close + 0.5, "low": close - 0.5, "close": close}
    )
    trades, stats = backtest(p)
    assert trades.empty
    assert stats == {"n": 0}
