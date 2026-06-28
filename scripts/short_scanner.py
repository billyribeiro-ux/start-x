"""SHORT candidate scanner — surfaces the (RESEARCH-ONLY) weak-tape pullback setup. NOT a buy/sell signal.

HONESTY BANNER (read before using): this scanner exists because the user asked to "wire the short setup
into the scanner." The setup it scans (short an over-extension into a weak tape) was decoded, tested
under the full firewall, and **does NOT clear it** — on liquid tradeable names the edge is NEGATIVE
(-0.17%/trade, Sharpe ~0, DSR 0.49; see DECODE.md and scripts/short_pullback_walkforward.py). There is
no firewall-cleared standalone single-name short edge in this universe. This tool therefore surfaces
CANDIDATES for manual study only — it is NOT a profitable signal and must not be traded as one.

It scans LIQUID names (>$5, >$20M/day ADV) that, within the last ``--lookback`` trading days while
SPY < its 20-day SMA (weak tape), closed >= 10% above their own 20-day SMA (over-extended).

    python scripts/short_scanner.py [--lookback 5] [--max-names N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PRICE_DIR = "data/cache/prices"


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=5, help="trading days back to scan for triggers")
    ap.add_argument("--max-names", type=int, default=None)
    args = ap.parse_args()

    spc = _load("SPY").set_index("date")["close"]
    weak = (spc < spc.rolling(20).mean())              # weak tape = SPY below its 20-SMA
    asof = spc.index.max()
    recent_weak = weak.loc[weak.index > asof - pd.Timedelta(days=args.lookback * 2)]

    print("=" * 78)
    print("SHORT CANDIDATE SCANNER — RESEARCH ONLY, NOT A TRADEABLE SIGNAL")
    print("The scanned setup is NEGATIVE on liquid names (-0.17%/trade, DSR 0.49). No short edge")
    print("clears the firewall on this universe. Candidates are for manual study only. See DECODE.md.")
    print("=" * 78)
    print(f"as-of {asof.date()}  |  weak tape (SPY<20-SMA) right now: {bool(weak.iloc[-1])}")
    if not weak.iloc[-1] and not recent_weak.tail(args.lookback).any():
        print("Gate OFF (tape not weak in the lookback) — the setup would not fire. Nothing to surface.")
        return

    names = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{PRICE_DIR}/*.parquet"))]
    names = [s for s in names if not s.startswith("_") and s not in ("SPY", "GLD", "UUP")]
    if args.max_names:
        names = names[:args.max_names]

    hits = []
    for sym in names:
        try:
            d = _load(sym)
        except Exception:
            continue
        if len(d) < 220:
            continue
        c = d["close"]; v = d["volume"]
        if np.nanmedian(c.to_numpy()) < 5 or np.nanmedian((c * v).to_numpy()) < 2e7:
            continue                                   # liquid, tradeable only
        sma20 = c.rolling(20).mean()
        ext = c / sma20 - 1.0
        tail = d.tail(args.lookback)
        for _, r in tail.iterrows():
            day = r["date"]
            if not bool(weak.get(day, False)):
                continue
            e = float(ext.loc[r.name]) if r.name in ext.index else np.nan
            if np.isfinite(e) and e >= 0.10:
                hits.append((sym, day.date(), round(e * 100, 1), round(float(r["close"]), 2)))

    if not hits:
        print("\nNo liquid names triggered the (research-only) setup in the lookback window.")
        return
    H = pd.DataFrame(hits, columns=["symbol", "date", "pct_above_20sma", "close"]).sort_values(
        "pct_above_20sma", ascending=False)
    print(f"\n{len(H)} candidate(s) — over-extended into a weak tape (RESEARCH ONLY, edge is negative):")
    print(H.to_string(index=False))
    print("\nReminder: this setup loses money net of costs on liquid names. Do not trade it as a signal.")


if __name__ == "__main__":
    main()
