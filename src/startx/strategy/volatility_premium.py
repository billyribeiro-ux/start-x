"""Volatility-risk-premium sleeve — get paid to provide liquidity into fear, on the index.

Two entries, both LONG the index, both honest about *what* they are: harvests of the volatility
risk premium (you are selling insurance into panic), NOT alpha over the index. The cross-desk
stress test was blunt — these are durable *risk premia* whose excess-over-buy-and-hold is thin, so
they earn their place by **diversifying** the book (low correlation to the trend sleeves) and by
being sized small, never as standalone money-printers.

* :func:`vrp_high_signals` — VIX minus 20-day realized vol of the index, in the top 5% of its
  trailing year. Fear is richly over-priced vs how much the index is actually moving → bounce.
* :func:`vvix_spike_signals` — VVIX (vol-of-vol) at/above its trailing-year 90th percentile. The
  market-maker desk found this is **better-sampled (n≈37 vs 9) and partly orthogonal to spot VIX**,
  and the only OOS-consistent member of the fear family — so it *replaces* the spot-VIX
  capitulation trigger (which deflated to noise: DSR 0.10, profit in 5 lucky trades).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized volatility, in VIX-comparable points (%)."""
    return close.pct_change().rolling(window).std() * np.sqrt(252) * 100.0


def vrp_high_signals(index_prices: pd.DataFrame, vix: pd.DataFrame, *, rv_window: int = 20,
                     lookback: int = 252, pct: float = 0.95) -> pd.Series:
    """Fire when VRP = VIX − realized-vol sits in the top ``pct`` of its trailing ``lookback``."""
    p = index_prices.sort_values("date").reset_index(drop=True)
    v = vix.sort_values("date").set_index("date")["close"].reindex(p["date"]).to_numpy()
    vrp = pd.Series(v, index=p.index) - realized_vol(p["close"], rv_window)
    thr = vrp.rolling(lookback, min_periods=lookback // 2).quantile(pct)
    return (vrp >= thr).fillna(False)


def vvix_spike_signals(index_prices: pd.DataFrame, vvix: pd.DataFrame, *, lookback: int = 252,
                       pct: float = 0.90) -> pd.Series:
    """Fire when VVIX (vol-of-vol) is at/above its trailing ``lookback`` ``pct`` percentile."""
    p = index_prices.sort_values("date").reset_index(drop=True)
    vv = vvix.sort_values("date").set_index("date")["close"].reindex(p["date"])
    vv = pd.Series(vv.to_numpy(), index=p.index)
    thr = vv.rolling(lookback, min_periods=lookback // 2).quantile(pct)
    return (vv >= thr).fillna(False)
