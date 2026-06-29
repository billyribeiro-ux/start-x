"""Swing scanner — TRADE BLOTTER: explicit entry/exit date, time, price for every taken trade.

Turns the validated, stress-gated swing trades into a full blotter: per trade the ENTRY (date, session
time, modelled fill price) and the EXIT (date, time, price, reason), bars held, and net P&L under the
BANKED exit (sharp 1-ATR label for selection + wide 2-ATR exit for capture). Source = the re-validation's
selective 1-ATR-label entries (stress regime + meta-prob >= 0.5).

EOD-data honesty: bars are daily, so times are the session convention — entries fill at the next OPEN
(09:30 ET); time-cap (10-day) exits at the CLOSE (16:00 ET); target/stop exits are touched intraday, so
the exact clock time is NOT in EOD data and is flagged "intraday" rather than fabricated.

    python scripts/swing_blotter.py [--recent 25] [--since 2020-01-01] [--csv PATH]
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

from startx.scanner.improve import resim_blotter
from startx.scanner.scan import STOP_ATR_MULT

PRICE_DIR = "data/cache/prices"
SOURCE = "data/cache/reval_oos_sl1.0.parquet"     # selective 1-ATR-LABEL entries (banked-config source)
STRESS = ["risk_off", "crisis"]
COST_BPS = 3.0


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=25, help="how many most-recent trades to print")
    ap.add_argument("--since", default="2020-01-01", help="only trades with entry on/after this date")
    ap.add_argument("--csv", default="data/cache/swing_blotter.csv", help="full blotter CSV (gitignored)")
    args = ap.parse_args()

    if not os.path.exists(SOURCE):
        print(f"source {SOURCE} not found — run scripts/swing_revalidate_2atr.py first to build it.")
        return
    oos = pd.read_parquet(SOURCE)
    taken = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)].copy()
    taken = taken[pd.to_datetime(taken["entry_date"]) >= pd.Timestamp(args.since)]
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    prices = {s: _load(s) for s in taken["symbol"].unique() if s in cached}

    bl = resim_blotter(taken, prices, pt_mult=2.0, sl_mult=STOP_ATR_MULT, cost_bps=COST_BPS)
    if bl.empty:
        print("no trades in range.")
        return
    bl = bl.sort_values("entry_date").reset_index(drop=True)
    bl["net_pnl_per_share"] = (bl["pnl_per_share"] - COST_BPS / 1e4 * bl["entry_px"]).round(2)

    print(f"SWING TRADE BLOTTER — banked exit (1-ATR label + {STOP_ATR_MULT:.0f}-ATR exit), "
          f"entries since {args.since}")
    print(f"{len(bl)} trades across {bl['symbol'].nunique()} symbols. Times = EOD session convention "
          "(entry=open 09:30 ET; time-exit=close 16:00 ET; target/stop touched intraday).\n")

    h = (f"  {'SYMBOL':<6} {'TRIGGER':<8} | {'ENTRY date':<10} {'time':<8} {'price':>9} | "
         f"{'EXIT date':<10} {'time':<8} {'price':>9} | {'REASON':<7} {'BARS':>4} {'NET%':>6}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for _, t in bl.tail(args.recent).iterrows():
        print(f"  {t['symbol']:<6} {t['trigger']:<8} | {str(t['entry_date']):<10} {t['entry_time']:<8} "
              f"{t['entry_px']:>9.2f} | {str(t['exit_date']):<10} {t['exit_time']:<8} {t['exit_px']:>9.2f} | "
              f"{t['exit_reason']:<7} {t['bars_held']:>4} {t['net_ret']*100:>5.1f}%")

    # totals block (locked-style: per-share $, net of cost) + aggregate expectancy
    wins = bl[bl["net_ret"] > 0]
    losses = bl[bl["net_ret"] <= 0]
    print(f"\n  --- TOTALS ({len(bl)} trades; per-share $ net of {COST_BPS:.0f}bp) ---")
    print(f"  TOTAL WIN $ : {wins['net_pnl_per_share'].clip(lower=0).sum():+10.2f}  ({len(wins)} wins)")
    print(f"  TOTAL LOSS $: {losses['net_pnl_per_share'].clip(upper=0).sum():+10.2f}  ({len(losses)} losses)")
    print(f"  NET TOTAL $ : {bl['net_pnl_per_share'].sum():+10.2f}")
    print(f"  expectancy  : {bl['net_ret'].mean()*100:+.2f}%/trade   win share {(bl['net_ret']>0).mean()*100:.0f}%"
          f"   avg bars held {bl['bars_held'].mean():.1f}")
    print("  (per-share $ summed across different-priced names is the locked layout's convention; the"
          " economically comparable figure is expectancy %/trade. +1 ATR / breakeven counts as WIN/scratch.)")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    bl.to_csv(args.csv, index=False)
    print(f"\nfull blotter ({len(bl)} trades) written -> {args.csv}")


if __name__ == "__main__":
    main()
