"""Short-the-index engine — the mirror of the long books, for BEAR-regime signals only.

Hard, evidence-backed finding (3-agent short study — S1/S2/S3): **shorting the index has NO
deployable standalone edge.** The index's +0.4%/10d upward drift is a permanent headwind that no
regime filter overcomes — forward drift stays POSITIVE even below the 200-SMA (and *violently*
positive in the bear-market rallies that punctuate every downtrend). Specifically:
  * "fade the rally" overbought shorts IN AN UPTREND are uniformly negative-edge (the drift runs
    them over; after several up days the index keeps going UP — momentum, not exhaustion);
  * "sell the breakdown" trend shorts have NO OOS edge (nothing clears t=1; the 1-ATR stop gets run
    over by the counter-trend bounces the breakdown fires into);
  * low VIX ≠ short, breadth-washout ≠ short (those are LONG/capitulation signals).
The single LEAST-BAD survivor — and the engine's default — is **`overbought_short` (RSI2>90 while
close < 200-SMA: sell the overbought *bounce* inside a confirmed downtrend)**: a thin, fragile,
regime-bursty edge (full-history PF ~1.1–1.4) that **LOSES in bull regimes** and concentrates its
P&L in 2000-09. Treat this engine as a confirmed-downtrend **drawdown HEDGE sized small, judged OOS
— never as alpha or a standalone book.** It refuses to short an uptrend by construction.

Exit logic is the exact mirror of the long chandelier:
  * enter SHORT at the signal-day close;
  * HARD STOP = entry + ``stop_mult``·ATR(14) (cover/buy-back if price RISES — cut the loss);
  * TRAILING COVER = (lowest low since entry) + ``trail_mult``·ATR (ratchets DOWN as price falls —
    ride the decline); binding cover = MIN(hard_stop, trailing);
  * also cover at the ``max_days`` time cap.
  * Gap-aware fills: if a bar gaps UP through the cover level (open above it), you cover at the open,
    not the level — the honest worst fill (mirror of the long's ``min(stop, open)``).

P&L for a short is ``entry − cover`` (you profit when price falls). One position at a time (no
overlapping shorts / pyramiding), so the ledger is a clean, chartable trade list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mean_reversion import atr, ibs, rsi

TRADING_DAYS = 252


@dataclass
class ShortResult:
    ledger: pd.DataFrame
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- #
# Bear-regime SHORT signal builders (all gated on close < trend SMA — no shorting an uptrend)
# --------------------------------------------------------------------------------------------- #
def _below_trend(p: pd.DataFrame, trend_ma: int) -> pd.Series:
    return p["close"] < p["close"].rolling(trend_ma).mean()


def overbought_short(prices, *, rsi_threshold=90.0, rsi_period=2, trend_ma=200) -> pd.Series:
    """Sell strength in a downtrend: RSI(2) > threshold AND close < 200-SMA (the S1 winner)."""
    p = prices.sort_values("date").reset_index(drop=True)
    return ((rsi(p["close"], rsi_period) > rsi_threshold) & _below_trend(p, trend_ma)).fillna(False)


def breakdown_short(prices, *, low_window=20, trend_ma=200) -> pd.Series:
    """Sell the breakdown: a NEW ``low_window``-day low while below the 200-SMA (mirror of breakout)."""
    p = prices.sort_values("date").reset_index(drop=True)
    new_low = p["close"] <= p["close"].rolling(low_window).min()
    return (new_low & _below_trend(p, trend_ma)).fillna(False)


def ibs_high_short(prices, *, ibs_threshold=0.90, trend_ma=200) -> pd.Series:
    """Close in the TOP of its range (IBS > threshold) while below the 200-SMA."""
    p = prices.sort_values("date").reset_index(drop=True)
    return ((ibs(p) > ibs_threshold) & _below_trend(p, trend_ma)).fillna(False)


SHORT_SIGNALS = {
    "overbought": overbought_short,     # RSI2>90 & below SMA200  (S1 winner)
    "breakdown": breakdown_short,       # new 20d low & below SMA200
    "ibs_high": ibs_high_short,         # IBS>0.90 & below SMA200
}


# --------------------------------------------------------------------------------------------- #
# The short backtest (mirror chandelier exit)
# --------------------------------------------------------------------------------------------- #
def short_backtest(prices: pd.DataFrame, signal: pd.Series, *, stop_mult: float = 1.0,
                   trail_mult: float = 3.0, atr_window: int = 14, max_days: int = 10,
                   cost_bps: float = 2.0, sleeve: str = "short",
                   start=None, end=None) -> ShortResult:
    """Run a one-at-a-time SHORT book over ``signal`` (a bool mask aligned to ``prices``)."""
    p = prices.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(p["date"])
    o, h, l, c = (p["open"].to_numpy(float), p["high"].to_numpy(float),
                  p["low"].to_numpy(float), p["close"].to_numpy(float))
    a = atr(p, atr_window).to_numpy(float)
    sig = np.asarray(pd.Series(signal).reindex(p.index).fillna(False).to_numpy(), dtype=bool)
    cost = cost_bps / 10000.0
    st = pd.Timestamp(start) if start is not None else None
    en = pd.Timestamp(end) if end is not None else None
    n = len(p)

    rows = []
    i = 0
    while i < n - 1:
        if not sig[i] or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        d = dates.iloc[i]
        if (st is not None and d < st) or (en is not None and d > en):
            i += 1
            continue
        e, av = c[i], a[i]
        hard = e + stop_mult * av                     # cover ABOVE entry (loss)
        trough = e
        cover = reason = None
        last = min(i + max_days, n - 1)
        for j in range(i + 1, last + 1):
            trough = min(trough, l[j])
            chand = trough + trail_mult * av          # trailing cover, ratchets DOWN
            level = min(hard, chand)                  # binding cover = the lower (hit first as price rises)
            if h[j] >= level:                          # price rose into the cover
                cover = max(level, o[j])               # gap-aware: a gap-UP through fills at the open
                reason = "chandelier" if chand <= hard else "stop"
                jx = j
                break
        else:
            cover, reason, jx = c[last], "time", last
        gross = (e - cover) / e                         # SHORT return: profit when cover < entry
        net = gross - cost
        rows.append({
            "sleeve": sleeve, "side": "short",
            "entry_date": dates.iloc[i].date(), "entry_price": round(e, 2),
            "stop_price": round(hard, 2),                # the hard stop sits ABOVE entry for a short
            "exit_date": dates.iloc[jx].date(), "exit_price": round(float(cover), 2),
            "exit_reason": reason, "bars_held": int(jx - i),
            "ret": round(net, 5),
            "pnl_per_share": round((e - cover) - e * cost, 2),
            "outcome": "WIN" if net > 0 else ("SCRATCH" if net == 0 else "LOSS"),
        })
        i = jx + 1                                       # one position at a time
    led = pd.DataFrame(rows)
    return ShortResult(ledger=led, stats=_short_stats(led))


def _short_stats(led: pd.DataFrame) -> dict:
    if led.empty:
        return {"n": 0}
    r = led["ret"]
    wins = led["outcome"].eq("WIN")
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    eq = float((1.0 + r).prod() - 1.0)
    return {
        "n": int(len(led)),
        "win_rate": float(wins.mean()),
        "expectancy_pct": float(r.mean() * 100),
        "total_return": eq,
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "avg_bars": float(led["bars_held"].mean()),
        "net_per_share": float(led["pnl_per_share"].sum()),
    }
