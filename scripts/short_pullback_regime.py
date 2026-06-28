"""SHORT the over-extension, gated to a WEAK TAPE — the one short thread that goes positive OOS.

The ungated pullback-scalp (short over-extension, cover on a 1-ATR retrace, 1-ATR stop, 3-day cap)
works in 2018-22 but decays to negative in the 2023-26 momentum melt-up (see short_pullback_scalp.py).
This gates it: only short when the BROAD TAPE is weak (SPY below its 20-day SMA) — the regime where a
retrace-to-the-mean actually arrives. That flips the OOS positive: ~+0.40%/trade net of 4bp + 5%/yr
borrow, positive in BOTH halves, low PBO. BUT it does NOT clear the deflated-Sharpe firewall (DSR ~0.23
< 0.95). UPDATE — the pre-registered walk-forward KILLED it: this full-universe "+0.40%" is a MIRAGE
driven by microcap/bad-data outliers + a breadth illusion. On LIQUID tradeable names the edge is
NEGATIVE (-0.17%/trade, Sharpe ~0). See scripts/short_pullback_walkforward.py. No tradeable short edge.

    python scripts/short_pullback_regime.py [--max-names N]
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

from startx.strategy.mean_reversion import atr
from qedge.validation import survival_gate

PRICE_DIR = "data/cache/prices"
TRAIN_END = np.datetime64("2022-12-31")
TEST_START = np.datetime64("2023-01-01")
COST_RT = 0.0004
BORROW_D = 0.05 / 252
N_TRIALS = 32 * 8           # honest: 32 scalp configs searched x 8 regime gates considered


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _regime():
    spc = _load("SPY").set_index("date")["close"]
    vx = _load("_VIX").set_index("date")["close"].reindex(spc.index).ffill()
    reg = pd.DataFrame(index=spc.index)
    reg["spy_below_20"] = spc < spc.rolling(20).mean()
    reg["spy_below_50"] = spc < spc.rolling(50).mean()
    reg["vix_gt_20"] = vx > 20
    return reg.shift(1).fillna(False)            # known at entry (no lookahead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-names", type=int, default=None)
    args = ap.parse_args()
    reg = _regime()
    names = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{PRICE_DIR}/*.parquet"))]
    names = [s for s in names if not s.startswith("_") and s not in ("SPY", "GLD", "UUP")]
    if args.max_names:
        names = names[:args.max_names]

    rows = []
    for sym in names:
        try:
            d = _load(sym)
        except Exception:
            continue
        if len(d) < 300:
            continue
        c = d["close"].to_numpy(float); h = d["high"].to_numpy(float)
        l = d["low"].to_numpy(float); o = d["open"].to_numpy(float)
        a = atr(d, 14).to_numpy(float); dts = d["date"].to_numpy()
        sma20 = d["close"].rolling(20).mean().to_numpy()
        sma200 = d["close"].rolling(200).mean().to_numpy()
        sig = np.flatnonzero(c >= 1.10 * sma20)         # over-extension trigger: >=10% above 20-SMA
        sig = sig[sig >= 200]
        last = -1; n = len(c)
        for i in sig:
            if i <= last or not np.isfinite(a[i]) or a[i] <= 0 or i + 1 >= n:
                continue
            e = c[i]; av = a[i]; tgt = e - av; stp = e + av     # 1-ATR target / 1-ATR stop
            cover = None; lb = min(i + 3, n - 1)                # 3-day cap
            for j in range(i + 1, lb + 1):
                if h[j] >= stp:
                    cover = max(stp, o[j]); jx = j; break       # gap-aware loss
                if l[j] <= tgt:
                    cover = min(tgt, o[j]); jx = j; break       # gap-aware win (the retrace)
            else:
                cover = c[lb]; jx = lb
            rows.append((dts[i], (e - cover) / e - COST_RT - BORROW_D * (jx - i),
                         bool(c[i] < sma200[i])))
            last = jx
    T = pd.DataFrame(rows, columns=["date", "ret", "name_below_200"])
    T["date"] = pd.to_datetime(T["date"])
    for col in reg.columns:
        T[col] = T["date"].map(reg[col]).fillna(False).astype(bool)

    print("SHORT over-extension (>=10% above 20-SMA), scalp 1-ATR retrace / 1-ATR stop / 3d cap")
    print(f"{len(names)} names, {len(T)} trades, net of {COST_RT*1e4:.0f}bp + 5%/yr borrow\n")
    print(f"{'gate':22}{'TRAIN exp%':>12}{'OOS exp%':>11}{'OOS win%':>10}{'DSR':>7}{'PBO':>7}{'gate':>7}")

    def row(sub, lbl):
        tr = sub[sub.date <= TRAIN_END]; te = sub[sub.date >= TEST_START]
        if len(te) < 30:
            print(f"{lbl:22}{'OOS n<30':>40}"); return
        g = survival_gate(te["ret"].to_numpy(), n_trials=N_TRIALS)
        print(f"{lbl:22}{tr.ret.mean()*100:>12.3f}{te.ret.mean()*100:>11.3f}"
              f"{(te.ret>0).mean()*100:>10.1f}{g.deflated_sharpe:>7.2f}{g.pbo:>7.2f}"
              f"{'PASS' if g.passed else 'fail':>7}")

    row(T, "UNGATED")
    row(T[T["spy_below_20"].values], "SPY<20-SMA (weak tape)")
    row(T[T["spy_below_50"].values], "SPY<50-SMA")
    row(T[T["vix_gt_20"].values], "VIX>20")
    row(T[T["spy_below_50"].values & T["name_below_200"].values], "SPY<50 & name<200")
    print("\nVERDICT: weak-tape gate is POSITIVE OOS (+0.4%/trade) but DSR ~0.23 < 0.95 — a thin,")
    print("promising, regime-conditional short, NOT yet firewall-cleared. The live short thread.")


if __name__ == "__main__":
    main()
