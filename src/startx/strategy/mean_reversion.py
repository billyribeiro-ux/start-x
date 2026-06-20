"""Mean-reversion swing strategy: buy oversold dips in an uptrend, exit at an ATR target/stop.

DEFAULT ENTRY = IBS < 0.1 (the 2019->2026 relearn winner). Internal Bar Strength,
IBS = (close - low) / (high - low), measures *where in the day's range the close landed*; < 0.1
means it closed in the bottom 10% (sold into the close). Bought in an uptrend (close > 200d), this
was the ONLY oversold signal with positive drift-adjusted alpha out-of-sample across SPY, SPX, QQQ
and IWM (+0.05% to +0.73%/trade beyond the buy-and-hold drift), 62-69% win.

The legacy Connors RSI(2)<15 entry is kept (``signal="rsi2"``) but is DEPRECATED: on 2019-2026 its
raw return looks positive only because it rides the bull — its drift-adjusted alpha is -0.22%, i.e.
you'd have done better just holding SPY for the same 5 days. The exit is horizon-matched to the
~5-10 day expected move (~1.5 ATR), sized to what the index actually does in a 1-10 day swing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    d = close.diff(); gain = d.clip(lower=0); loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = prices["high"], prices["low"], prices["close"]; pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def ibs(prices) -> pd.Series:
    """Internal Bar Strength = (close - low) / (high - low). Low = closed near the day's low."""
    p = prices.sort_values("date").reset_index(drop=True)
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    return (p["close"] - p["low"]) / rng


def ibs_signals(prices, ibs_threshold=0.10, trend_ma=200) -> pd.Series:
    """Entry signal: close in the bottom ``ibs_threshold`` of its daily range, in an uptrend."""
    p = prices.sort_values("date").reset_index(drop=True)
    sma = p["close"].rolling(trend_ma).mean()
    return ((ibs(p) < ibs_threshold) & (p["close"] > sma)).fillna(False)


def mean_reversion_signals(prices, rsi_period=2, rsi_threshold=15.0, trend_ma=200) -> pd.Series:
    """Legacy (DEPRECATED) entry: a *fresh* RSI-oversold cross while price is above its trend MA."""
    p = prices.sort_values("date").reset_index(drop=True)
    r = rsi(p["close"], rsi_period); sma = p["close"].rolling(trend_ma).mean()
    return (r < rsi_threshold) & (r.shift(1) >= rsi_threshold) & (p["close"] > sma)


def backtest(prices, *, signal="ibs", ibs_threshold=0.10, rsi_period=2, rsi_threshold=15.0,
             trend_ma=200, atr_mult=1.5, atr_window=14, max_days=10, cost_bps=2.0,
             start=None, end=None):
    """Run the strategy on one symbol's price frame. ``signal`` = "ibs" (default, relearn winner)
    or "rsi2" (legacy). Returns (trades DataFrame, stats dict). Each trade carries its entry, ATR
    target/stop, exit and the IBS at entry (for review/charting)."""
    p = prices.sort_values("date").reset_index(drop=True)
    a = atr(p, atr_window) / p["close"]
    ib = ibs(p)
    if signal == "ibs":
        sig = ibs_signals(p, ibs_threshold, trend_ma)
    else:
        sig = mean_reversion_signals(p, rsi_period, rsi_threshold, trend_ma).fillna(False)
    cost = cost_bps / 10000.0
    rows = []
    for i in p.index[sig]:
        d = p.loc[i, "date"]
        if (start and d < pd.Timestamp(start)) or (end and d > pd.Timestamp(end)):
            continue
        e = p.loc[i, "close"]; ap = a.iloc[i]
        if not np.isfinite(ap):
            continue
        tgt = e * (1 + atr_mult * ap); stp = e * (1 - atr_mult * ap)
        w = p.iloc[i + 1: i + 1 + max_days]; ed = ep = reason = None
        for _, b in w.iterrows():
            if b["low"] <= stp:
                ed, ep, reason = b["date"], stp, "stop"; break
            if b["high"] >= tgt:
                ed, ep, reason = b["date"], tgt, "target"; break
        if reason is None and len(w):
            ed, ep, reason = w.iloc[-1]["date"], w.iloc[-1]["close"], "time"
        if reason is None:
            continue
        rows.append(dict(entry_date=d, entry_price=e, target=tgt, stop=stp, exit_date=ed,
                         exit_price=ep, exit_reason=reason, ibs_entry=round(float(ib.iloc[i]), 3),
                         ret=(ep / e - 1) - cost))
    trades = pd.DataFrame(rows)
    return trades, _stats(trades)


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    r = trades["ret"]; eq = (1 + r).cumprod(); wins = r[r > 0]; losses = r[r < 0]
    return {
        "n": int(len(r)), "win_rate": float((r > 0).mean()), "total_return": float(eq.iloc[-1] - 1),
        "expectancy": float(r.mean()), "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }
