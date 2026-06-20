"""Reversal Lab: deterministic synthetic checks + one network-gated live smoke.

The synthetic path is a clean up-then-down V (rally to a peak, sell-off to a trough) built so
the ATR-zigzag must find exactly the peak and the trough, the legs must alternate up/down with
the arithmetic we can verify by hand, and the microstructure primitives (volume profile POC,
cumulative delta sign) land on known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startx.settings import get_settings

from startx.analytics.volume_profile import (
    cumulative_delta,
    opening_range,
    session_vwap,
    volume_profile,
)
from startx.data.prices import _enrich
from startx.events.reversals import (
    atr,
    characterize_reversal,
    detect_swings,
    trend_legs,
)


# --------------------------------------------------------------------------- builders
def _v_path(warmup=20, rise=40, fall=40, recover=20, base=100.0, step=1.0):
    """Deterministic path: flat warmup -> rise to a peak -> fall to a trough -> partial recover.

    The warmup lets ATR warm up before any swing (so ``atr_extension`` is finite), and the
    final recovery rally (> 3*ATR off the trough) *confirms* the trough as a swing low — so the
    zigzag yields a confirmed up-leg (base->peak) and a confirmed down-leg (peak->trough).

    Returns the enriched daily frame and (peak_date, trough_date, peak_price, trough_price).
    high == low == close so swing-extreme prices equal the closes (exact arithmetic); ATR stays
    positive via the day-to-day true range (|close - prev_close| == step).
    """
    flat = np.full(warmup, base)
    up = base + step * np.arange(1, rise + 1)            # base+step .. base+rise
    peak = base + step * rise
    down = peak - step * np.arange(1, fall + 1)          # peak-step .. peak-fall
    trough = peak - step * fall
    rec = trough + step * np.arange(1, recover + 1)      # trough+step .. trough+recover
    close = np.concatenate([flat, up, down, rec])
    n = len(close)
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000_000.0,
        "vwap": close,
        "changePercent": 0.0,
    })
    peak_idx = warmup + rise - 1        # last bar of the rise
    trough_idx = warmup + rise + fall - 1   # last bar of the fall
    return _enrich(df), dates[peak_idx], dates[trough_idx], float(peak), float(trough)


def _intraday_bars(prices, mix):
    """Build 1-min-ish bars at given price levels with up/down body sign per ``mix``.

    ``mix`` is a list of (price, volume, sign) where sign>0 => close>open (up bar).
    """
    ts0 = pd.Timestamp("2022-03-01 09:30:00")
    rows = []
    for k, (price, vol, sign) in enumerate(mix):
        o = price - 0.5 if sign > 0 else price + 0.5
        c = price + 0.5 if sign > 0 else price - 0.5
        rows.append({
            "ts": ts0 + pd.Timedelta(minutes=k),
            "open": o, "high": max(o, c) + 0.1, "low": min(o, c) - 0.1,
            "close": c, "volume": float(vol),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ATR
def test_atr_is_positive_and_aligned():
    prices, *_ = _v_path()
    a = atr(prices, window=14)
    assert len(a) == len(prices)
    # ATR is non-negative everywhere and strictly positive once the series has moved.
    assert (a.dropna() >= 0).all()
    assert a.iloc[-1] > 0


# --------------------------------------------------------------------------- swings
def test_detect_swings_finds_peak_and_trough():
    prices, peak_date, trough_date, _, _ = _v_path()
    swings = detect_swings(prices, atr_window=14, atr_mult=3.0)
    assert not swings.empty
    highs = swings[swings["kind"] == "high"]
    lows = swings[swings["kind"] == "low"]
    assert peak_date in set(highs["date"]), "the V peak must be a confirmed swing high"
    assert trough_date in set(lows["date"]), "the V trough must be a confirmed swing low"
    # Swings must strictly alternate high/low/high/...
    kinds = list(swings["kind"])
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)), "kinds alternate"


# --------------------------------------------------------------------------- legs
def test_trend_legs_magnitude_bars_and_directions():
    prices, peak_date, trough_date, peak_price, trough_price = _v_path(rise=40, fall=40)
    swings = detect_swings(prices)
    legs = trend_legs(swings, prices)
    assert not legs.empty
    # Directions of consecutive legs must alternate.
    dirs = list(legs["direction"])
    assert all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1)), "leg directions alternate"

    up_legs = legs[legs["direction"] == "up"]
    assert not up_legs.empty
    up = up_legs.iloc[0]
    # Up leg base(100) -> peak(140) => +40% magnitude; ends at the confirmed swing high.
    assert up["magnitude_pct"] == pytest.approx(40.0, abs=1e-6)
    assert up["end_date"] == peak_date

    down_legs = legs[legs["direction"] == "down"]
    assert not down_legs.empty
    down = down_legs.iloc[0]
    # Down leg peak(140) -> trough(100): (100-140)/140 = -28.571% over exactly 40 bars.
    assert down["magnitude_pct"] == pytest.approx((100.0 - 140.0) / 140.0 * 100.0, abs=1e-6)
    assert down["bars"] == 40
    assert down["start_date"] == peak_date and down["end_date"] == trough_date
    # MFE/MAE sanity: favourable excursion is non-negative in the leg's direction.
    assert (legs["mfe_pct"] >= -1e-6).all()


def test_trend_legs_atr_extension_definition():
    prices, *_ = _v_path()
    legs = trend_legs(detect_swings(prices), prices)
    # Use the down leg: it starts at the peak where ATR is fully warmed up (finite extension).
    down = legs[legs["direction"] == "down"].iloc[0]
    # atr_extension = magnitude_pct / (ATR_at_start/price_at_start*100); recompute by hand.
    a = atr(prices, 14)
    i0 = int(prices.index[prices["date"] == down["start_date"]][0])
    start_price = float(prices.iloc[i0]["close"])
    atr_pct = float(a.iloc[i0]) / start_price * 100.0
    assert np.isfinite(down["atr_extension"])
    assert down["atr_extension"] == pytest.approx(down["magnitude_pct"] / atr_pct, rel=1e-6)


# --------------------------------------------------------------------------- volume profile
def test_volume_profile_poc_is_max_volume_bin():
    # Concentrate volume at price 150; thin elsewhere. POC bin must contain 150.
    mix = [(110.0, 100, 1), (130.0, 100, 1), (150.0, 5000, 1), (170.0, 100, -1)]
    bars = _intraday_bars(None, mix)
    vp = volume_profile(bars, bins=50, value_area=0.7)
    assert vp["total_volume"] == pytest.approx(5300.0)
    assert vp["poc_price"] == pytest.approx(150.0, abs=2.0), "POC must sit at the 150 cluster"
    assert vp["val"] <= vp["poc_price"] <= vp["vah"]


def test_volume_profile_empty_is_nan():
    vp = volume_profile(pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]))
    assert np.isnan(vp["poc_price"]) and np.isnan(vp["vah"]) and np.isnan(vp["val"])
    assert vp["profile"].empty


def test_session_vwap_and_opening_range():
    mix = [(100.0, 1000, 1), (200.0, 1000, -1)]
    bars = _intraday_bars(None, mix)
    # Equal volume at ~100 and ~200 => VWAP near the volume-weighted typical midpoint.
    assert 100.0 < session_vwap(bars) < 200.0
    orr = opening_range(bars, minutes=1)
    assert np.isfinite(orr["high"]) and np.isfinite(orr["low"])


# --------------------------------------------------------------------------- delta
def test_cumulative_delta_sign():
    # Net buying: big up bar dominates a small down bar => positive delta.
    up_heavy = _intraday_bars(None, [(100.0, 5000, 1), (100.0, 1000, -1)])
    assert cumulative_delta(up_heavy) == pytest.approx(4000.0)
    # Net selling => negative.
    down_heavy = _intraday_bars(None, [(100.0, 1000, 1), (100.0, 5000, -1)])
    assert cumulative_delta(down_heavy) == pytest.approx(-4000.0)


# --------------------------------------------------------------------------- characterize
def test_characterize_reversal_daily_keys_and_rvol():
    prices, peak_date, _, _, _ = _v_path()
    out = characterize_reversal(prices, peak_date, intraday=False)
    assert "daily" in out and "intraday" not in out
    daily = out["daily"]
    for key in ("volume", "rvol_20", "range_pct", "atr_14", "atr_expansion", "gap_pct",
                "close_location_value", "ar_z", "candle"):
        assert key in daily, f"missing daily key {key}"

    # rvol_20 must equal day volume / trailing-20 mean volume (here all volume equal => ~1.0).
    i = int(prices.index[prices["date"] == peak_date][0])
    vol = float(prices.iloc[i]["volume"])
    avg20 = float(prices["volume"].iloc[max(0, i - 20):i].mean())
    assert daily["rvol_20"] == pytest.approx(vol / avg20, rel=1e-9)
    assert isinstance(daily["candle"], str)


# --------------------------------------------------------------------------- live smoke
@pytest.mark.skipif(
    not get_settings().fmp_api_key,
    reason="FMP_API_KEY unset — skipping network smoke",
)
def test_live_intraday_smoke():
    from startx.data.cache import ParquetCache
    from startx.data.intraday import get_intraday
    from startx.fmp.client import FMPClient

    settings = get_settings()
    cache = ParquetCache(settings.cache_dir)
    with FMPClient(settings) as client:
        bars = get_intraday(client, cache, "NVDA", "2024-03-08")
    assert not bars.empty, "expected 1-min bars for NVDA on a real trading day"
    vp = volume_profile(bars)
    assert vp["poc_price"] > 0
