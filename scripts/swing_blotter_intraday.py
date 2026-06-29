"""Swing scanner — INTRADAY trade blotter: REAL entry/exit date AND clock time per trade.

Upgrades the EOD blotter to true timestamps by pulling FMP 1-minute bars and finding the exact minute each
trade's barrier was touched. Entry is the next-session OPEN (09:30:00 ET, a real time, not a placeholder);
for target/stop exits the exact touch minute is located in the 1-min data on the exit day; a 10-day time-cap
exit is the close (16:00:00 ET). FMP intraday is US/Eastern regular session and reaches back to 2020.

Source entries = the banked config (sharp 1-ATR label selection, stress regime + meta-prob >= 0.5);
realised exit = the validated 2-ATR stop / 2-ATR target / 10-day cap. 1-min days are cached to parquet so
re-runs are cheap and resumable.

    python scripts/swing_blotter_intraday.py [--since 2020-01-01] [--recent 25] [--csv PATH]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.fmp.client import FMPClient
from startx.fmp.endpoints import historical_chart
from startx.scanner.improve import first_touch, resim_blotter
from startx.scanner.scan import STOP_ATR_MULT

PRICE_DIR = "data/cache/prices"
INTRADAY_DIR = "data/cache/intraday"
SOURCE = "data/cache/reval_oos_sl1.0.parquet"
STRESS = ["risk_off", "crisis"]
COST_BPS = 3.0


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _intraday_symbols(sym):
    """Intraday ticker variants: FMP intraday uses the DOT class form (BRK.B) where the EOD cache uses the
    dash form (BRK-B). Try the dash form first, then the dot form."""
    out = [sym]
    if "-" in sym:
        out.append(sym.replace("-", "."))
    return out


def _intraday_day(client, sym, day):
    """1-min bars for one symbol/day, cached to parquet. Only NON-EMPTY results are cached, so a transient
    fetch failure retries on the next run instead of being frozen as a permanent empty (the seam bug)."""
    cache = f"{INTRADAY_DIR}/{sym}_{day}.parquet"
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        if not df.empty:
            return df
    for s in _intraday_symbols(sym):
        df = historical_chart(client, s, "1min", start=str(day), end=str(day))
        if not df.empty:
            os.makedirs(INTRADAY_DIR, exist_ok=True)
            df.to_parquet(cache)
            return df
    return pd.DataFrame()       # never cache empty -> next run refetches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--recent", type=int, default=25)
    ap.add_argument("--csv", default="data/cache/swing_blotter_intraday.csv")
    args = ap.parse_args()

    oos = pd.read_parquet(SOURCE)
    taken = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)].copy()
    taken = taken[pd.to_datetime(taken["entry_date"]) >= pd.Timestamp(args.since)]
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    prices = {s: _load(s) for s in taken["symbol"].unique() if s in cached}

    bl = resim_blotter(taken, prices, pt_mult=2.0, sl_mult=STOP_ATR_MULT, cost_bps=COST_BPS)
    bl = bl.sort_values("entry_date").reset_index(drop=True)
    print(f"resolving intraday timestamps for {len(bl)} trades (fetching/caching 1-min exit days)...")

    def _window_bars(client, sym, d0, d1):
        """1-min bars across all trading days in [d0, d1] (trading days from the daily cache)."""
        px = prices.get(sym)
        if px is None:
            return pd.DataFrame()
        days = px.loc[(px["date"] >= pd.Timestamp(d0)) & (px["date"] <= pd.Timestamp(d1)), "date"]
        parts = [_intraday_day(client, sym, d.date()) for d in days]
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    entry_ts, exit_ts, src = [], [], []
    n_exact = n_artifact = 0
    with FMPClient() as client:
        for _, t in bl.iterrows():
            entry_ts.append(f"{t['entry_date']} 09:30:00 ET")            # next-open fill = real 09:30 ET
            if t["exit_reason"] == "time":
                exit_ts.append(f"{t['exit_date']} 16:00:00 ET")          # time cap = the close
                src.append("close")
                continue
            after = pd.Timestamp(f"{t['entry_date']} 09:30:00")
            level = t["barrier_level"]                                   # UNROUNDED PT/SL (no cent miss)
            # cheap path: the daily-determined exit day; robust path: the full hold window
            ts, _ = first_touch(_intraday_day(client, t["symbol"], t["exit_date"]),
                                t["exit_reason"], level, after_ts=after)
            if ts is None:
                ts, _ = first_touch(_window_bars(client, t["symbol"], t["entry_date"], t["exit_date"]),
                                    t["exit_reason"], level, after_ts=after)
            if ts is not None:
                exit_ts.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} ET")
                src.append("1min")
                n_exact += 1
            else:
                # the EOD-feed low/high that triggered the daily barrier was NOT reached in regular-session
                # 1-min bars -> the exact intraday time can't be honestly assigned (intrabar-fill artifact).
                exit_ts.append(f"{t['exit_date']} (EOD-only {t['exit_reason']}; not in RTH 1-min)")
                src.append("eod_artifact")
                n_artifact += 1
    bl["entry_ts"] = entry_ts
    bl["exit_ts"] = exit_ts
    bl["exit_ts_src"] = src
    bl["net_pnl_per_share"] = (bl["pnl_per_share"] - COST_BPS / 1e4 * bl["entry_px"]).round(2)

    print(f"  exact 1-min exit timestamps: {n_exact}  |  time-cap closes: {(bl['exit_reason']=='time').sum()}"
          f"  |  EOD-only artifacts (barrier not in RTH 1-min): {n_artifact}\n")
    print(f"SWING TRADE BLOTTER (intraday timestamps) — banked 1-ATR label + {STOP_ATR_MULT:.0f}-ATR exit, "
          f"since {args.since}")
    h = (f"  {'SYMBOL':<6} {'TRIG':<8} | {'ENTRY (date time ET)':<23} {'price':>8} | "
         f"{'EXIT (date time ET)':<23} {'price':>8} | {'REASON':<6} {'NET%':>6}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for _, t in bl.tail(args.recent).iterrows():
        print(f"  {t['symbol']:<6} {t['trigger']:<8} | {t['entry_ts']:<23} {t['entry_px']:>8.2f} | "
              f"{t['exit_ts']:<23} {t['exit_px']:>8.2f} | {t['exit_reason']:<6} {t['net_ret']*100:>5.1f}%")

    wins, losses = bl[bl["net_ret"] > 0], bl[bl["net_ret"] <= 0]
    print(f"\n  --- TOTALS ({len(bl)} trades; per-share $ net of {COST_BPS:.0f}bp) ---")
    print(f"  TOTAL WIN $ : {wins['net_pnl_per_share'].clip(lower=0).sum():+10.2f}  ({len(wins)} wins)")
    print(f"  TOTAL LOSS $: {losses['net_pnl_per_share'].clip(upper=0).sum():+10.2f}  ({len(losses)} losses)")
    print(f"  NET TOTAL $ : {bl['net_pnl_per_share'].sum():+10.2f}")
    print(f"  expectancy  : {bl['net_ret'].mean()*100:+.2f}%/trade   win share {(bl['net_ret']>0).mean()*100:.0f}%"
          f"   avg bars held {bl['bars_held'].mean():.1f}")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    bl.to_csv(args.csv, index=False)
    print(f"\nfull intraday blotter ({len(bl)} trades) -> {args.csv}")


if __name__ == "__main__":
    main()
