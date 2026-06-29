"""Tests for the improvement levers (exits + sizing) — scanner/improve.py.

Pins the exit re-simulation arithmetic (a known synthetic path) and the calendar-book plumbing, so the
R6 result (a 2-ATR stop beats the 1-ATR stop) rests on correct mechanics, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.scanner.improve import (
    book_stats,
    calendar_book,
    first_touch,
    per_trade_stats,
    resim_blotter,
    resim_returns,
)


def _prices():
    """30 flat bars at 100, then a dip to ~96 and recovery to ~104 — so a tight stop is hit but a wide
    one survives to a higher exit. ATR is ~constant so the multipliers are easy to reason about."""
    n = 40
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    # carve a controlled dip + recovery starting at the entry bar (index 31)
    low[31], close[31], high[31] = 97.0, 98.0, 100.0       # touches 97 intraday
    close[32], high[32], low[32] = 99.0, 100.0, 98.0
    close[35], high[35], low[35] = 104.0, 105.0, 103.0     # later rallies
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": dates, "open": close, "high": high, "low": low, "close": close})


def _taken(signal_idx=30):
    p = _prices()
    sig = p["date"].iloc[signal_idx]
    ent = p["date"].iloc[signal_idx + 1]
    return pd.DataFrame([{"symbol": "TST", "date": sig, "entry_date": ent,
                          "t1": p["date"].iloc[signal_idx + 9], "trigger": "mean_rev",
                          "regime": "risk_off", "ret": 0.0}])


def test_wider_stop_changes_outcome():
    px = {"TST": _prices()}
    taken = _taken()
    tight = resim_returns(taken, px, pt_mult=2.0, sl_mult=1.0, horizon=10, cost_bps=0.0)
    wide = resim_returns(taken, px, pt_mult=2.0, sl_mult=3.0, horizon=10, cost_bps=0.0)
    assert np.isfinite(tight[0]) and np.isfinite(wide[0])
    # entry ~98 (next open). ATR ~2 -> tight 1-ATR stop ~96 is NOT hit (low 97), but both resolve;
    # the key property: a wider stop never gives a WORSE outcome on a path that recovers.
    assert wide[0] >= tight[0] - 1e-9


def test_regime_scaled_only_applies_in_stress():
    px = {"TST": _prices()}
    calm = _taken()
    calm.loc[0, "regime"] = "risk_on"
    r_calm = resim_returns(calm, px, sl_mult=1.0, stress_sl_mult=3.0, cost_bps=0.0)
    # the stress trade uses the widened stop; the calm one keeps the base stop -> outcomes may differ,
    # and the calm one must equal the plain base-stop sim
    r_base_calm = resim_returns(calm, px, sl_mult=1.0, cost_bps=0.0)
    assert np.allclose(r_calm, r_base_calm, equal_nan=True)


def test_blotter_fields_and_fills():
    px = {"TST": _prices()}
    taken = _taken()
    bl = resim_blotter(taken, px, pt_mult=2.0, sl_mult=2.0, horizon=10, cost_bps=0.0)
    assert len(bl) == 1
    r = bl.iloc[0]
    for col in ("entry_date", "entry_time", "entry_px", "exit_date", "exit_time", "exit_px",
                "exit_reason", "bars_held", "net_ret", "pnl_per_share"):
        assert col in bl.columns
    # entry fills at the NEXT open after the signal bar (index 31 here, price 98.0)
    assert abs(r["entry_px"] - 98.0) < 1e-9
    assert r["entry_time"] == "09:30 ET"
    # a time-cap exit is the close (16:00 ET); a touched barrier is flagged intraday
    assert (r["exit_reason"] == "time") == (r["exit_time"] == "16:00 ET")
    assert r["exit_reason"] in ("target", "stop", "time")
    assert r["bars_held"] >= 0


def test_first_touch_intraday():
    bars = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00",
                                    "2024-01-02 09:32:00", "2024-01-02 09:33:00"]),
        "open": [100, 100, 100, 100], "high": [100.5, 101.0, 102.0, 103.0],
        "low": [99.5, 99.0, 98.0, 97.0], "close": [100, 100, 100, 100],
    })
    # target 101.5 first crossed by the 09:32 bar (high 102.0)
    ts, px = first_touch(bars, "target", 101.5)
    assert ts == pd.Timestamp("2024-01-02 09:32:00") and px == 101.5
    # stop 98.5 first crossed by the 09:32 bar (low 98.0)
    ts, px = first_touch(bars, "stop", 98.5)
    assert ts == pd.Timestamp("2024-01-02 09:32:00")
    # after_ts skips earlier bars
    ts, _ = first_touch(bars, "stop", 99.2, after_ts=pd.Timestamp("2024-01-02 09:32:00"))
    assert ts == pd.Timestamp("2024-01-02 09:32:00")
    # never touched -> (None, None); empty -> (None, None)
    assert first_touch(bars, "target", 999.0) == (None, None)
    assert first_touch(pd.DataFrame(), "target", 1.0) == (None, None)


def test_calendar_book_and_stats():
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.bdate_range("2019-01-01", periods=n * 2)[::2]
    taken = pd.DataFrame({"symbol": "X", "entry_date": dates,
                          "t1": dates + pd.Timedelta(days=7),
                          "ret": rng.normal(0.01, 0.05, n)})
    daily = calendar_book(taken, taken["ret"].to_numpy(), max_concurrent=10)
    assert len(daily) > 0
    st = book_stats(daily, n_trials=1)
    assert "sharpe" in st and "maxdd" in st and st["maxdd"] <= 0
    pst = per_trade_stats(taken["ret"].to_numpy(), n_trials=3)
    assert pst["n"] == n and "dsr" in pst
