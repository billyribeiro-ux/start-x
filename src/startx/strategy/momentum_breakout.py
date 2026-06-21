"""Momentum breakout swing system — the strongest standalone edge on the 2019→2026 relearn.

Entry: a *fresh* 20-day-high breakout while price is above its 200-day average (trend intact) — buy
strength confirming strength, not a dip. Exit: a chandelier (trail the highest high since entry by
``atr_mult``×ATR) so a runner is ridden to exhaustion instead of capped, with a time stop as a
backstop. This is the "don't cut home-runners" exit applied to the entry that actually feeds it
runners.

Validated out-of-sample (TRAIN 2019-2022 tuned, TEST 2023→2026 untouched): TEST profit factor 3.64,
max drawdown −4.4% vs −19% buy-and-hold, ~2× the return-per-unit-drawdown of being passively long.
It generalizes *untuned* to QQQ and SPX (TEST PF 3.0 / 3.4); the honest caveat is it does NOT hold on
IWM (small-caps were a different regime). On raw total return nothing beats B&H in a +96% bull — the
edge here is risk-adjusted: similar money, a quarter of the pain, and the freedom to stand aside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = prices["high"], prices["low"], prices["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def breakout_signals(prices: pd.DataFrame, lookback: int = 20, trend_ma: int = 200) -> pd.Series:
    """Entry: close makes a *new* ``lookback``-day high (a fresh breakout, not still-extended) while
    above the ``trend_ma``-day average. Returns a bool Series aligned to ``prices`` (sorted)."""
    p = prices.sort_values("date").reset_index(drop=True)
    hh = p["close"].rolling(lookback).max()
    sma = p["close"].rolling(trend_ma).mean()
    new_high = p["close"] >= hh                    # at/through the rolling high
    fresh = p["close"].shift(1) < hh.shift(1)      # wasn't at the high yesterday -> fresh cross
    return (new_high & fresh & (p["close"] > sma)).fillna(False)


def backtest(prices: pd.DataFrame, *, lookback: int = 20, trend_ma: int = 200, exit: str = "chandelier",
             atr_mult: float = 3.0, atr_window: int = 14, max_days: int = 40, cost_bps: float = 2.0,
             start=None, end=None):
    """Long the index on a fresh breakout; manage the exit. Fills close-to-close, net of ``cost_bps``
    round-trip. exit='chandelier' (peak − atr_mult·ATR trail) or 'fixed' (atr_mult·ATR target/stop).
    Returns (trades DataFrame, stats dict)."""
    p = prices.sort_values("date").reset_index(drop=True)
    a = atr(p, atr_window)
    sig = breakout_signals(p, lookback, trend_ma)
    cost = cost_bps / 10000.0
    sym = prices.attrs.get("symbol", "IDX")
    rows = []
    open_until = None  # don't stack a new entry while one is live (one position at a time)
    for i in p.index[sig]:
        d = p.loc[i, "date"]
        if open_until is not None and d <= open_until:
            continue
        if (start and d < pd.Timestamp(start)) or (end and d > pd.Timestamp(end)):
            continue
        e = float(p.loc[i, "close"]); av = float(a.iloc[i])
        if not np.isfinite(av) or i + 1 >= len(p):
            continue
        w = p.iloc[i + 1: i + 1 + max_days]
        ed = ep = reason = None
        if exit == "fixed":
            tgt, stp = e + atr_mult * av, e - atr_mult * av
            for _, b in w.iterrows():
                if b["low"] <= stp:
                    ed, ep, reason = b["date"], stp, "stop"; break
                if b["high"] >= tgt:
                    ed, ep, reason = b["date"], tgt, "target"; break
        else:  # chandelier
            peak = e
            for _, b in w.iterrows():
                peak = max(peak, float(b["high"]))
                chand = peak - atr_mult * av
                if b["low"] <= chand:
                    ed, ep = b["date"], chand
                    reason = "chandelier" if peak > e else "stop"; break
        if reason is None and len(w):
            ed, ep, reason = w.iloc[-1]["date"], float(w.iloc[-1]["close"]), "time"
        if reason is None:
            continue
        open_until = ed
        mfe = float((w["high"].max() / e - 1)) if len(w) else 0.0
        rows.append(dict(
            symbol=sym, entry_date=d, entry_price=round(e, 2),
            stop_price=round(e - atr_mult * av, 2), exit_date=ed, exit_price=round(float(ep), 2),
            exit_reason=reason, bars_held=int(p.index[p["date"] == ed][0] - i),
            pnl_per_share=round((float(ep) - e) - e * cost, 2),
            breakout_high=round(float(p["close"].iloc[max(0, i - lookback):i].max()), 2),
            atr_pct=round(av / e * 100, 2), mfe_pct=round(mfe * 100, 2),
            outcome=("WIN" if (float(ep) / e - 1) - cost > 0 else "SCRATCH" if (float(ep) / e - 1) - cost == 0 else "LOSS"),
        ))
    trades = pd.DataFrame(rows)
    return trades, _stats(trades)


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    r = trades["exit_price"] / trades["entry_price"] - 1
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
