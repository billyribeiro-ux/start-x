"""Unit tests for the RSI-2 / IBS<0.1 mean-reversion edge (``strategy/mean_reversion.py``).

This is one of the two edges the desk actually trusts (per FINDINGS.md / CLAUDE.md) yet had
ZERO dedicated coverage. Every case here is deterministic and runs on hand-built synthetic
OHLCV — no network, no FMP.

The documented contract we pin (module docstring + ``ibs_signals``):
  IBS = (close - low) / (high - low)  -> *where in the bar's range the close landed*.
  ``ibs_signals`` fires only when  IBS < ibs_threshold (default 0.10)  AND  close > SMA(trend_ma).
  It must NOT fire when either leg fails, must be NaN-safe (zero-range bars -> False, pre-warmup
  SMA -> False), and the output must be a bool Series aligned 1:1 to the (sorted) input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.mean_reversion import (
    ibs,
    ibs_signals,
    mean_reversion_signals,
    rsi,
)


def _uptrend_frame(n: int = 260, start: str = "2022-01-03") -> pd.DataFrame:
    """A steady, low-noise uptrend so that ``close > SMA(200)`` holds after warm-up.

    Each bar gets a symmetric +/-1.0 range around the close (IBS = 0.5 by construction), so any
    single bar we deliberately push to a low/high close is the *only* signal in the frame.
    """
    dates = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 160.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


# --------------------------------------------------------------------------- IBS arithmetic


def test_ibs_arithmetic_matches_definition():
    """IBS = (close - low) / (high - low): close at the low -> 0, at the high -> 1, midpoint 0.5."""
    p = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=3),
            "high": [110.0, 110.0, 110.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 110.0, 105.0],  # at-low, at-high, midpoint
        }
    )
    out = ibs(p)
    assert out.tolist() == [0.0, 1.0, 0.5]


def test_ibs_zero_range_is_nan():
    """A zero-range bar (high == low) must not divide-by-zero; it yields NaN, not inf/error."""
    p = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=1),
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
        }
    )
    assert np.isnan(ibs(p).iloc[0])


# --------------------------------------------------------------------------- positive case


def test_ibs_signal_fires_on_low_close_in_uptrend():
    """POSITIVE: a bar whose close sits in the bottom 10% of its range, with close > SMA(200),
    is the single firing day."""
    p = _uptrend_frame()
    i = 250  # well past the 200-day SMA warm-up, and within an established uptrend
    # Force bar i to close near the low of a wide range: IBS = 0.1/5.1 ~= 0.0196 < 0.10
    p.loc[i, "high"] = p.loc[i, "close"] + 5.0
    p.loc[i, "low"] = p.loc[i, "close"] - 0.1

    sig = ibs_signals(p)

    assert bool(sig.iloc[i]) is True
    # It is the ONLY signal: every other bar is a symmetric IBS=0.5 bar.
    assert int(sig.sum()) == 1
    assert ibs(p).iloc[i] < 0.10


# --------------------------------------------------------------------------- negative cases


def test_ibs_signal_silent_when_close_is_high_in_range():
    """NEGATIVE (IBS leg fails): a close near the *high* of its range -> IBS large -> no signal,
    even in the same uptrend."""
    p = _uptrend_frame()
    i = 250
    p.loc[i, "high"] = p.loc[i, "close"] + 0.1
    p.loc[i, "low"] = p.loc[i, "close"] - 5.0  # IBS = 5.0/5.1 ~= 0.98 -> NOT < 0.10
    sig = ibs_signals(p)
    assert bool(sig.iloc[i]) is False
    assert int(sig.sum()) == 0


def test_ibs_signal_silent_below_trend_ma():
    """NEGATIVE (trend leg fails): a perfect IBS<0.1 bar in a DOWNtrend (close < SMA200) must
    NOT fire -- the strategy only buys dips inside an uptrend."""
    n = 260
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = np.linspace(160.0, 100.0, n)  # steady DOWNtrend -> close < SMA(200) after warm-up
    p = pd.DataFrame(
        {
            "date": dates,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )
    i = 250
    p.loc[i, "high"] = p.loc[i, "close"] + 5.0
    p.loc[i, "low"] = p.loc[i, "close"] - 0.1  # IBS < 0.10 ...
    assert ibs(p).iloc[i] < 0.10  # ... the IBS leg is satisfied,
    sma = p["close"].rolling(200).mean()
    assert p.loc[i, "close"] < sma.iloc[i]  # ... but price is below trend,
    assert bool(ibs_signals(p).iloc[i]) is False  # ... so no signal.


# --------------------------------------------------------------------------- NaN / warm-up / shape


def test_ibs_signal_nan_safe_and_warmup_is_false():
    """Pre-warm-up rows (SMA(200) is NaN for the first 199 bars) and zero-range bars resolve to
    False, never NaN -- the output is a clean boolean mask."""
    p = _uptrend_frame()
    # zero-range bar near the end: IBS NaN -> must be False
    p.loc[255, "high"] = p.loc[255, "close"]
    p.loc[255, "low"] = p.loc[255, "close"]
    sig = ibs_signals(p)

    assert sig.dtype == bool
    assert not sig.isna().any()
    # First 199 bars cannot have a 200-day SMA -> all False there.
    assert not sig.iloc[:199].any()
    # zero-range bar -> NaN IBS -> False
    assert bool(sig.iloc[255]) is False


def test_ibs_signal_threshold_is_honored():
    """The ``ibs_threshold`` argument actually gates: a bar with IBS ~= 0.30 fires at
    threshold 0.5 but not at the default 0.10."""
    p = _uptrend_frame()
    i = 250
    # IBS = 1.5/5.0 = 0.30
    p.loc[i, "high"] = p.loc[i, "close"] + 3.5
    p.loc[i, "low"] = p.loc[i, "close"] - 1.5
    assert abs(ibs(p).iloc[i] - 0.30) < 1e-9
    assert bool(ibs_signals(p, ibs_threshold=0.10).iloc[i]) is False
    assert bool(ibs_signals(p, ibs_threshold=0.50).iloc[i]) is True


def test_ibs_signal_alignment_and_length():
    """Output Series is aligned 1:1 with the (sorted) input frame -- same length, and stable
    under a shuffled input because the function sorts by date internally."""
    p = _uptrend_frame()
    i = 250
    p.loc[i, "high"] = p.loc[i, "close"] + 5.0
    p.loc[i, "low"] = p.loc[i, "close"] - 0.1
    sig = ibs_signals(p)
    assert len(sig) == len(p)

    # Shuffle input rows; the function sorts by date, so the *set* of firing dates is invariant.
    shuffled = p.sample(frac=1.0, random_state=7).reset_index(drop=True)
    sig_s = ibs_signals(shuffled)
    fired_date = p.loc[i, "date"]
    sorted_dates = p.sort_values("date").reset_index(drop=True)["date"]
    assert sorted_dates[sig.values].tolist() == [fired_date]
    assert sorted_dates[sig_s.values].tolist() == [fired_date]


# --------------------------------------------------------------------------- legacy RSI-2 path


def test_rsi2_signal_is_a_fresh_oversold_cross_in_uptrend():
    """The legacy (deprecated) RSI-2 path fires only on a *fresh* cross below the threshold while
    above the trend MA: it is True the bar the cross happens and not the bar after (no double-fire)."""
    # Mild wiggle keeps RSI(2) comfortably above 15 in the normal uptrend, so the engineered cross
    # below is unambiguous and deterministic (fixed seed).
    n = 260
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(1)
    close = np.linspace(100.0, 160.0, n) + rng.normal(0.0, 0.5, n)
    p = pd.DataFrame({"date": dates, "high": close + 1.0, "low": close - 1.0, "close": close})

    i = 250
    # i-1: a clear up day (drives RSI(2) high). i: a sharp down day -> a *fresh* cross below 15.
    p.loc[i - 1, "close"] = p.loc[i - 2, "close"] + 3.0
    p.loc[i, "close"] = p.loc[i - 1, "close"] - 10.0
    for j in (i - 1, i):
        p.loc[j, "low"] = p.loc[j, "close"] - 1.0
        p.loc[j, "high"] = p.loc[j, "close"] + 1.0

    r = rsi(p["close"], 2)
    sig = mean_reversion_signals(p).fillna(False)

    # RSI was well above the threshold the prior bar, then crossed below it on bar i.
    assert r.iloc[i - 1] >= 15.0
    assert r.iloc[i] < 15.0
    # Fresh cross -> fires at i; does NOT re-fire at i+1 (RSI recovers, and even if it did not, the
    # cross is no longer *fresh*). It is the only firing bar in the frame.
    assert bool(sig.iloc[i]) is True
    assert bool(sig.iloc[i + 1]) is False
    assert int(sig.sum()) == 1
    assert len(sig) == len(p)
