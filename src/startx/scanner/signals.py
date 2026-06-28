"""Layers 2 & 3 — LOCATION + TRIGGER features and primary swing events (Charter v1, P3).

All features are point-in-time (trailing only) and EOD-buildable — the options/flow seams (GEX walls,
order-flow imbalance, absorption) stay dark. Primary events are the *hypotheses* the meta-label model
then learns to take or skip; each carries a direction and a trigger tag so attribution is explicit.

Feature families:
  LOCATION (L2): distance to 50/200-DMA, position-in-range, distance to prior 20d H/L, Bollinger
                 location, ATR%, anchored-VWAP distance.
  TRIGGER  (L3): Bollinger band-width (squeeze), NR7, RSI(2)/RSI(14), relative strength vs SPY,
                 short/medium momentum, volume thrust, RSI divergence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOCATION_FEATURES = ["dist_sma50", "dist_sma200", "pos_range20", "dist_phigh20", "dist_plow20",
                     "bb_pos", "atr_pct", "dist_avwap"]
TRIGGER_FEATURES = ["bb_width", "bb_width_pct", "nr7", "rsi2", "rsi14", "rs_spy20", "ret5", "ret21",
                    "vol_thrust", "rsi_div"]
FEATURES = LOCATION_FEATURES + TRIGGER_FEATURES


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(p: pd.DataFrame, n: int = 14) -> pd.Series:
    h, lo, c = p["high"], p["low"], p["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - lo), (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def feature_frame(prices: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Per-bar PIT feature frame for one symbol (index = date)."""
    p = prices.sort_values("date").reset_index(drop=True)
    c = p["close"].astype(float)
    h, lo = p["high"].astype(float), p["low"].astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(p["date"]))
    f = pd.DataFrame(index=idx)

    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    sd20 = c.rolling(20).std()
    atr = _atr(p)
    rng20_hi, rng20_lo = h.rolling(20).max(), lo.rolling(20).min()

    # LOCATION
    f["dist_sma50"] = (c / sma50 - 1).values
    f["dist_sma200"] = (c / sma200 - 1).values
    f["pos_range20"] = ((c - rng20_lo) / (rng20_hi - rng20_lo).replace(0, np.nan)).values
    f["dist_phigh20"] = (c / rng20_hi.shift(1) - 1).values
    f["dist_plow20"] = (c / rng20_lo.shift(1) - 1).values
    f["bb_pos"] = ((c - sma20) / (2 * sd20).replace(0, np.nan)).values     # -1=lower band, +1=upper
    f["atr_pct"] = (atr / c).values
    # anchored VWAP from the last 63d (rolling proxy for a pivot-anchored VWAP)
    tp = (h + lo + c) / 3
    avwap = (tp * p["volume"]).rolling(63).sum() / p["volume"].rolling(63).sum()
    f["dist_avwap"] = (c / avwap.values - 1)

    # TRIGGER
    f["bb_width"] = ((4 * sd20) / sma20).values
    f["bb_width_pct"] = f["bb_width"].rolling(252).rank(pct=True).values  # squeeze = low percentile
    rng = h - lo
    f["nr7"] = (rng <= rng.rolling(7).min()).astype(float).values
    f["rsi2"] = _rsi(c, 2).values
    f["rsi14"] = _rsi(c, 14).values
    spy_al = spy.reindex(idx).ffill()
    rs = (c.values / c.shift(20).values) / (spy_al / spy_al.shift(20))
    f["rs_spy20"] = rs.values
    f["ret5"] = (c / c.shift(5) - 1).values
    f["ret21"] = (c / c.shift(21) - 1).values
    f["vol_thrust"] = (p["volume"] / p["volume"].rolling(20).mean()).values
    # RSI divergence: price 10d change vs RSI14 10d change sign mismatch
    f["rsi_div"] = (np.sign(c.diff(10)) - np.sign(_rsi(c, 14).diff(10))).values
    return f


def primary_events(prices: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Candidate LONG swing entries (the hypotheses). Columns: date, trigger, direction."""
    c = prices.sort_values("date").reset_index(drop=True)["close"].astype(float)
    c.index = feats.index
    sma200 = c.rolling(200).mean()
    up = c > sma200                                              # only long in an uptrend
    ev = []

    # 1) mean-reversion: oversold dip in an uptrend (IBS/RSI2 low near lower band)
    mr = up & (feats["rsi2"] < 10) & (feats["bb_pos"] < -0.6)
    # 2) squeeze release: Bollinger squeeze (bottom-decile width) breaking the prior 5d high
    sq = (feats["bb_width_pct"] < 0.10) & (c > c.shift(1).rolling(5).max().reindex(c.index))
    # 3) pullback-to-support: near rising 50-DMA from above, in an uptrend
    pb = up & (feats["dist_sma50"].abs() < 0.02) & (feats["dist_sma50"] >= 0) & (feats["rsi2"] < 30)
    # 4) relative-strength breakout: RS vs SPY rising + new 20d high
    rb = up & (feats["rs_spy20"] > 1.0) & (feats["dist_phigh20"] >= 0)

    for tag, mask in [("mean_rev", mr), ("squeeze", sq), ("pullback", pb), ("rs_break", rb)]:
        d = feats.index[mask.fillna(False).to_numpy()]
        for dt in d:
            ev.append({"date": dt, "trigger": tag, "direction": 1})
    out = pd.DataFrame(ev).sort_values("date").reset_index(drop=True) if ev else \
        pd.DataFrame(columns=["date", "trigger", "direction"])
    return out
