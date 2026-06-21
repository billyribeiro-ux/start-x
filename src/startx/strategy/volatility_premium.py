"""Volatility-risk-premium sleeve — get paid to provide liquidity into fear, on the index.

Two entries, both LONG the index, both honest about *what* they are: harvests of the volatility
risk premium (you are selling insurance into panic), **NOT alpha over the index**. The cross-desk
stress test was blunt — these are durable *risk premia* whose excess-over-buy-and-hold is thin, so
they earn their place by **diversifying** the book (low correlation to the trend sleeves) and by
being **sized small**, never as standalone money-printers. Treat their P&L as beta-robust /
alpha-fragile: the break-even cost that zeroes their drift-adjusted edge is only ~0-6 bps/side, so
they survive on raw return but their *alpha* dies under realistic fills (see ``portfolio/execution``).

* :func:`vrp_high_signals` — VIX minus 20-day realized vol of the index, in the top 5% of its
  trailing year. Fear is richly over-priced vs how much the index is actually moving -> bounce. This
  is the best-sampled vol entry (TEST 2023-26: ~18 trades, ~61% win) and is **fill-insensitive on
  alpha** — the durable member of the family.
* :func:`vvix_spike_signals` — VVIX (vol-of-vol) at/above its trailing-year 90th percentile. The
  market-maker desk found this is **better-sampled (n≈37 vs the spot-VIX capitulation's 9) and
  partly orthogonal to spot VIX** (fwd-5d +1.3%/+1.1% TRAIN/TEST, 76% up both), and the only
  OOS-consistent member of the fear family — so it **replaces** the deflated spot-VIX capitulation
  trigger (which deflated to noise: DSR 0.10, profit concentrated in 5 lucky trades).

Point-in-time safety
--------------------
Every threshold (the trailing-quantile bands) is computed from a *trailing* rolling window that
ends at the signal bar, so no future information leaks into the decision. Exogenous series (VIX,
VVIX) are aligned to the price calendar by date (never by position) and forward-NaN'd rather than
silently mis-aligned. A signal bar with any missing input resolves to ``False`` (no trade), never a
spurious fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------------------
def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized volatility, in VIX-comparable points (%).

    ``std(daily returns) * sqrt(252) * 100`` over a trailing ``window``. Returned aligned to
    ``close``; the first ``window`` observations are ``NaN`` (insufficient history, no lookahead).
    """
    return close.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252) * 100.0


def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR (EWM of true range) — the exit's volatility unit, shared with the other sleeves."""
    h, l, c = prices["high"], prices["low"], prices["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


# --------------------------------------------------------------------------------------------------
# Alignment helper (point-in-time safe)
# --------------------------------------------------------------------------------------------------
def _align_to_prices(index_prices: pd.DataFrame, other: pd.DataFrame,
                     column: str = "close") -> pd.Series:
    """Align ``other[column]`` onto the (sorted) ``index_prices`` calendar by **date**.

    Returns a Series indexed by ``index_prices``'s positional index (0..n-1), so it lines up with
    the rolling indicators below. Alignment is by calendar date — never by row position — so a
    different start date or a missing exogenous day cannot silently shift the series. Exogenous
    levels are forward-filled across at most a small gap (so a single missing VIX print does not
    blank a whole window) but never back-filled (that would leak the future).
    """
    if "date" not in index_prices.columns:
        raise KeyError("index_prices must carry a 'date' column")
    if "date" not in other.columns or column not in other.columns:
        raise KeyError(f"exogenous frame must carry 'date' and {column!r} columns")

    p = index_prices.sort_values("date").reset_index(drop=True)
    o = (other.sort_values("date")
         .drop_duplicates("date", keep="last")
         .set_index("date")[column]
         .astype(float))
    # Reindex onto the price dates; ffill a *short* gap only (no backfill -> no lookahead).
    aligned = o.reindex(p["date"]).ffill(limit=3)
    return pd.Series(aligned.to_numpy(dtype=float), index=p.index, name=column)


def _trailing_quantile(series: pd.Series, lookback: int, pct: float) -> pd.Series:
    """Trailing rolling quantile that *ends* at each bar (point-in-time, no lookahead).

    ``min_periods`` is held at ``lookback // 2`` so a threshold only exists once there is a
    meaningful trailing sample; before that the comparison resolves to ``False`` (no trade).
    """
    return series.rolling(lookback, min_periods=max(2, lookback // 2)).quantile(pct)


# --------------------------------------------------------------------------------------------------
# Signals (public contract — signatures are LOCKED)
# --------------------------------------------------------------------------------------------------
def vrp_high_signals(index_prices: pd.DataFrame, vix: pd.DataFrame, *, rv_window: int = 20,
                     lookback: int = 252, pct: float = 0.95) -> pd.Series:
    """Fire when the volatility risk premium is richly priced.

    ``VRP = VIX - realized_vol(index, rv_window)``: how much more fear is priced into options than
    the index is actually delivering. When VRP sits in the top ``pct`` of its trailing ``lookback``
    (default top 5% of the past year), insurance is over-priced and a mean-reversion bounce is
    loading -> **long the index**.

    Point-in-time: ``realized_vol`` and the quantile band are both trailing-only. A bar with a
    missing VIX print, missing realized vol (warm-up), or no established band resolves to ``False``.

    Returns a boolean Series aligned to the **sorted** ``index_prices`` (positional index 0..n-1).
    """
    p = index_prices.sort_values("date").reset_index(drop=True)
    v = _align_to_prices(p, vix, "close")
    vrp = v - realized_vol(p["close"], rv_window)
    thr = _trailing_quantile(vrp, lookback, pct)
    fire = (vrp >= thr) & vrp.notna() & thr.notna()
    return fire.fillna(False).astype(bool)


def vvix_spike_signals(index_prices: pd.DataFrame, vvix: pd.DataFrame, *, lookback: int = 252,
                       pct: float = 0.90) -> pd.Series:
    """Fire when VVIX (vol-of-vol) is at/above its trailing ``lookback`` ``pct`` percentile.

    VVIX measures the volatility *of* implied volatility — dealer hedging stress / convexity demand.
    The desk found it the only OOS-consistent fear gauge and partly orthogonal to spot VIX, so it
    **replaces** the spot-VIX capitulation trigger. At/above its trailing-year p90 -> **long the
    index** (the panic in vol-of-vol marks an exhaustion bottom, ~76% up over the next 5 days).

    Point-in-time: the percentile band is trailing-only; a missing VVIX print or an unestablished
    band resolves to ``False``.

    Returns a boolean Series aligned to the **sorted** ``index_prices`` (positional index 0..n-1).
    """
    p = index_prices.sort_values("date").reset_index(drop=True)
    vv = _align_to_prices(p, vvix, "close")
    thr = _trailing_quantile(vv, lookback, pct)
    fire = (vv >= thr) & vv.notna() & thr.notna()
    return fire.fillna(False).astype(bool)


# --------------------------------------------------------------------------------------------------
# Backtest (mirrors the sibling sleeves so the portfolio engine / validation can price these trades)
# --------------------------------------------------------------------------------------------------
def backtest(index_prices: pd.DataFrame, vol_frame: pd.DataFrame, *, signal: str = "vrp",
             rv_window: int = 20, lookback: int = 252, pct: float | None = None,
             exit: str = "chandelier", atr_mult: float = 3.0, atr_window: int = 14,
             max_days: int = 40, cost_bps: float = 2.0, start=None, end=None):
    """Long the index on a volatility-premium trigger; ride a chandelier exit.

    ``signal`` selects the entry: ``"vrp"`` (VIX - realized vol, ``vol_frame`` = VIX) or ``"vvix"``
    (vol-of-vol percentile, ``vol_frame`` = VVIX). ``pct`` defaults to 0.95 for VRP and 0.90 for
    VVIX (the validated thresholds). Fills are close-to-close, net of round-trip ``cost_bps``; the
    realistic-fill re-pricing lives in ``portfolio/execution.apply_fills``. One position at a time
    (no stacking into the same episode). Returns ``(trades_df, stats_dict)`` with the LOCKED
    trade-sheet columns shared across sleeves.

    The chandelier (peak high since entry - ``atr_mult``×ATR(entry)) honors the project rule: ride
    winners to exhaustion, never cap a home-runner.
    """
    p = index_prices.sort_values("date").reset_index(drop=True)
    if signal == "vrp":
        sig = vrp_high_signals(p, vol_frame, rv_window=rv_window, lookback=lookback,
                               pct=0.95 if pct is None else pct)
        vol_level_col = "vrp"
    elif signal == "vvix":
        sig = vvix_spike_signals(p, vol_frame, lookback=lookback,
                                 pct=0.90 if pct is None else pct)
        vol_level_col = "vvix"
    else:
        raise ValueError(f"signal must be 'vrp' or 'vvix', got {signal!r}")

    a = atr(p, atr_window)
    vlevel = _align_to_prices(p, vol_frame, "close")
    cost = cost_bps / 10000.0
    sym = index_prices.attrs.get("symbol", "IDX")
    rows: list[dict] = []
    open_until = None  # one position at a time -> don't pile into the same vol episode

    for i in p.index[sig]:
        d = p.loc[i, "date"]
        if open_until is not None and d <= open_until:
            continue
        if (start and d < pd.Timestamp(start)) or (end and d > pd.Timestamp(end)):
            continue
        e = float(p.loc[i, "close"])
        av = float(a.iloc[i])
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
            symbol=sym, signal=signal, entry_date=d, entry_price=round(e, 2),
            stop_price=round(e - atr_mult * av, 2), exit_date=ed, exit_price=round(float(ep), 2),
            exit_reason=reason, bars_held=int(p.index[p["date"] == ed][0] - i),
            pnl_per_share=round((float(ep) - e) - e * cost, 2),
            **{vol_level_col: round(float(vlevel.iloc[i]), 2)},
            atr_pct=round(av / e * 100, 2), mfe_pct=round(mfe * 100, 2),
            outcome=("WIN" if (float(ep) / e - 1) - cost > 0 else "LOSS"),
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
