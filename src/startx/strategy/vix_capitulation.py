"""VIX-capitulation swing signal — a *separate* edge from the RSI-2 dip-buy.

The thesis, validated on 16y of data (2010-2026): when VIX makes several *consecutive*
closes above its upper volatility band (20-day Bollinger at 2.5 sigma) while already above its
neutral price (~18, the zero-drift mean-reversion level), the market is in persistent, exhausted
fear — i.e. **capitulation**. Capitulation marks bottoms, not the start of falls: the down-move is
what pushed VIX through the band, and by the third close up there the selling is spent and the
snap-back is loading. Forward S&P over the next 5-10 days is strongly positive (≈+0.9% / 5d, ~70%
up at VIX>25) and shorting it loses. So this fires a **long the index** trade, exited on a
chandelier (peak − N×ATR) so winners run to exhaustion rather than being capped.

This module is deliberately independent of ``mean_reversion.py``: different trigger (a VIX vol
breakout, not an RSI dip), different cadence (rare, high-conviction), same honesty bar (report
out-of-sample, size the exit to the move).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIX_NEUTRAL = 18.0  # full-sample zero-drift level; DEPRECATED as a static floor — VIX neutral
                    # drifts by regime (median 10.8 in 2017, 14.6 in 2024, 18.3 in 2026), so the
                    # default below is "adaptive": a trailing-1yr median, recomputed each day.
NEUTRAL_WINDOW = 252  # trailing window (~1yr) for the adaptive neutral baseline


def vix_upper_band(vix_close: pd.Series, window: int = 20, k: float = 2.5) -> pd.Series:
    """Upper Bollinger band on VIX: SMA(window) + k*std(window)."""
    mid = vix_close.rolling(window).mean()
    sd = vix_close.rolling(window).std()
    return mid + k * sd


def adaptive_neutral(vix_close: pd.Series, window: int = NEUTRAL_WINDOW) -> pd.Series:
    """Regime-aware VIX neutral: the trailing-``window`` median, recomputed each day (point-in-time,
    no lookahead). Replaces the static 18 so "above neutral" tracks the *current* regime."""
    return vix_close.rolling(window, min_periods=60).median()


def _atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = prices["high"], prices["low"], prices["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def capitulation_signals(vix: pd.DataFrame, *, neutral="adaptive",
                         band_window: int = 20, band_k: float = 2.5,
                         min_closes: int = 3) -> pd.Series:
    """Fire on the day VIX logs its ``min_closes``-th *consecutive* close above the upper band
    while VIX > its neutral baseline. ``neutral`` is "adaptive" (trailing-1yr median, regime-aware)
    by default, or pass a float to pin it. One trigger per capitulation episode (the run-length
    crossing point), so we don't pile into the same spike. Returns a bool Series aligned to ``vix``.
    """
    v = vix.sort_values("date").reset_index(drop=True)
    band = vix_upper_band(v["close"], band_window, band_k)
    base = adaptive_neutral(v["close"]) if neutral == "adaptive" else float(neutral)
    above = (v["close"] > band) & (v["close"] > base)
    # consecutive run length of `above`
    run = above * (above.groupby((~above).cumsum()).cumcount() + 1)
    return run == min_closes  # exactly the Nth close -> one entry per episode


def backtest(index_prices: pd.DataFrame, vix: pd.DataFrame, *, neutral="adaptive",
             band_window: int = 20, band_k: float = 2.5, min_closes: int = 3,
             exit: str = "chandelier", atr_mult: float = 3.0, atr_window: int = 14,
             max_days: int = 40, cost_bps: float = 2.0, start=None, end=None):
    """Long the index on a VIX-capitulation trigger; manage the exit.

    exit='chandelier'  -> trail stop at (peak high since entry) - atr_mult*ATR(entry).
    exit='fixed'       -> symmetric atr_mult*ATR target/stop (horizon-matched comparison).
    Fills are close-to-close, net of round-trip ``cost_bps``. Returns (trades_df, stats_dict).
    """
    p = index_prices.sort_values("date").reset_index(drop=True)
    vx = vix.sort_values("date").reset_index(drop=True)
    sig = capitulation_signals(vx, neutral=neutral, band_window=band_window, band_k=band_k,
                               min_closes=min_closes)
    sig_dates = set(vx.loc[sig.fillna(False), "date"])
    vlevel = vx.set_index("date")["close"]
    atr = _atr(p, atr_window)
    cost = cost_bps / 10000.0
    rows = []
    for i in p.index:
        d = p.loc[i, "date"]
        if d not in sig_dates:
            continue
        if (start and d < pd.Timestamp(start)) or (end and d > pd.Timestamp(end)):
            continue
        e = float(p.loc[i, "close"])
        a = float(atr.iloc[i])
        if not np.isfinite(a) or i + 1 >= len(p):
            continue
        w = p.iloc[i + 1: i + 1 + max_days]
        ed = ep = reason = None
        if exit == "fixed":
            tgt, stp = e + atr_mult * a, e - atr_mult * a
            for _, b in w.iterrows():
                if b["low"] <= stp:
                    ed, ep, reason = b["date"], stp, "stop"; break
                if b["high"] >= tgt:
                    ed, ep, reason = b["date"], tgt, "target"; break
        else:  # chandelier
            peak = e
            for _, b in w.iterrows():
                peak = max(peak, float(b["high"]))
                chand = peak - atr_mult * a
                if b["low"] <= chand:
                    ed, ep = b["date"], chand
                    reason = "chandelier" if peak > e else "stop"
                    break
        if reason is None and len(w):
            ed, ep, reason = w.iloc[-1]["date"], float(w.iloc[-1]["close"]), "time"
        if reason is None:
            continue
        mfe = float((w["high"].max() / e - 1)) if len(w) else 0.0
        rows.append(dict(
            symbol=index_prices.attrs.get("symbol", "IDX"), entry_date=d, entry_price=round(e, 2),
            stop_price=round(e - atr_mult * a, 2), exit_date=ed, exit_price=round(float(ep), 2),
            exit_reason=reason, bars_held=int((p.index[p["date"] == ed][0] - i)),
            pnl_per_share=round((float(ep) - e) - e * cost, 2),
            vix_level=round(float(vlevel.get(d, np.nan)), 1),
            atr_pct=round(a / e * 100, 2), mfe_pct=round(mfe * 100, 2),
            outcome=("WIN" if (float(ep) / e - 1) - cost > 0 else "LOSS"),
        ))
    trades = pd.DataFrame(rows)
    return trades, _stats(trades)


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    r = (trades["exit_price"] / trades["entry_price"] - 1)
    eq = (1 + r).cumprod(); wins = r[r > 0]; losses = r[r < 0]
    return {
        "n": int(len(r)), "win_rate": float((r > 0).mean()),
        "total_return": float(eq.iloc[-1] - 1), "expectancy": float(r.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "net_usd_per_share": float(trades["pnl_per_share"].sum()),
    }
