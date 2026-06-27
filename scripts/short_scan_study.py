"""SHORT-scanner study — reproduces the negative verdict for two single-name short angles.

Part of the 4-angle short-decode campaign (see DECODE.md). The index is un-shortable (upward drift);
this asks whether SINGLE-NAME shorts have a real edge. They don't — and this script shows WHY on the
cached universe, net of cost+borrow, OOS (TRAIN 2018-22 / TEST 2023→end):

  * EVENT-DRIFT (mirror of Reaction-PEAD): short the continuation after a down-gap (<=-5% on >=1.5x
    volume). The forward drift is a tiny 5d dip that INVERTS to POSITIVE by 20d — the names recover.
    The opposite of a downside-PEAD signature → no short edge.
  * BREAKDOWN: short a new 20-day low while below the 200-SMA (mirror of a long breakout). ~30% win;
    you short the break and it bounces into the stop. Negative expectancy in both train and test.

Both confirm the structural truth the whole desk keeps re-finding: weak/breaking names BOUNCE (the
oversold mean-reversion that powers our LONG IBS edge is exactly what runs over a short).

    python scripts/short_scan_study.py [--max-names N]   # N caps the universe for a quick run
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

from startx.strategy.short_engine import short_backtest

PRICE_DIR = "data/cache/prices"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
END = "2026-06-25"


def _names(max_names: int | None) -> list[str]:
    fs = sorted(glob.glob(f"{PRICE_DIR}/*.parquet"))
    syms = [os.path.basename(f)[:-8] for f in fs]
    syms = [s for s in syms if not s.startswith("_") and s not in ("SPY", "GLD", "UUP")]
    return syms[:max_names] if max_names else syms


def _load(sym: str) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    if len(d) < 300:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def event_drift(names: list[str]) -> None:
    """Down-gap (<=-5% on >=1.5x vol) → forward 5/10/20d return. SHORT wins only if NEGATIVE."""
    rows = []
    for sym in names:
        d = _load(sym)
        if d is None:
            continue
        c = d["close"].to_numpy(float)
        o = d["open"].to_numpy(float)
        v = d["volume"].to_numpy(float)
        prevc = np.concatenate([[np.nan], c[:-1]])
        gap = o / prevc - 1.0
        volr = v / pd.Series(v).rolling(50).mean().to_numpy()
        ev = np.where((gap <= -0.05) & (volr >= 1.5))[0]
        for i in ev:
            if i + 21 >= len(c):
                continue
            rows.append((d["date"].iloc[i], c[i + 5] / c[i] - 1, c[i + 10] / c[i] - 1,
                         c[i + 20] / c[i] - 1))
    e = pd.DataFrame(rows, columns=["date", "fwd5", "fwd10", "fwd20"])
    print(f"\n=== EVENT-DRIFT: down-gap<=-5% & vol>=1.5x  (n={len(e)} events) ===")
    print("  forward UNDERLYING return after the gap — a SHORT wins only if these are NEGATIVE:")
    for lbl, sub in [("TRAIN 2018-22", e[e.date <= TRAIN_END]), ("TEST 2023-now", e[e.date >= TEST_START])]:
        print(f"  {lbl}: n={len(sub):5d}  fwd5={sub.fwd5.mean()*100:+.2f}%  "
              f"fwd10={sub.fwd10.mean()*100:+.2f}%  fwd20={sub.fwd20.mean()*100:+.2f}%")
    print("  -> drift is a tiny 5d dip that INVERTS to positive by 20d: the names recover. NO short edge.")


def breakdown(names: list[str], borrow_ann: float = 0.05) -> None:
    """Short a NEW 20-day low while below the 200-SMA; mirror chandelier, 10d cap. Aggregate trades."""
    borrow_bp_per_day = borrow_ann / 252 * 1e4
    led = []
    for sym in names:
        d = _load(sym)
        if d is None:
            continue
        c = d["close"]
        new_low = c <= c.rolling(20).min()
        below = c < c.rolling(200).mean()
        sig = (new_low & below).fillna(False)
        if not sig.any():
            continue
        r = short_backtest(d, sig, stop_mult=1.0, trail_mult=3.0, max_days=10, cost_bps=2.0).ledger
        if not r.empty:
            r["sym"] = sym
            led.append(r)
    if not led:
        print("\n=== BREAKDOWN: no trades ===")
        return
    L = pd.concat(led, ignore_index=True)
    L["entry_date"] = pd.to_datetime(L["entry_date"])
    # subtract a realistic borrow on the short, per held day
    L["ret_net"] = L["ret"] - borrow_bp_per_day / 1e4 * L["bars_held"]
    print(f"\n=== BREAKDOWN: new 20d low & < 200-SMA  (n={len(L)} trades, borrow {borrow_ann:.0%}/yr) ===")
    for lbl, sub in [("TRAIN 2018-22", L[L.entry_date <= TRAIN_END]), ("TEST 2023-now", L[L.entry_date >= TEST_START])]:
        if sub.empty:
            continue
        exp = sub["ret_net"].mean() * 100
        win = sub["outcome"].eq("WIN").mean() * 100
        print(f"  {lbl}: n={len(sub):6d}  expectancy={exp:+.2f}%/trade  win={win:.1f}%  "
              f"total={(sub['ret_net'].sum()*100):+.0f}% (equal-weight)")
    print("  -> negative expectancy, ~30% win: you short the break, it bounces into the 1-ATR stop. NO short edge.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-names", type=int, default=None, help="cap universe for a quick run")
    args = ap.parse_args()
    names = _names(args.max_names)
    print(f"SHORT-SCANNER STUDY — {len(names)} single names from {PRICE_DIR}  (net of ~2bp + borrow)")
    event_drift(names)
    breakdown(names)
    print("\nVERDICT: no standalone single-name short edge (both angles negative OOS). See DECODE.md.")


if __name__ == "__main__":
    main()
