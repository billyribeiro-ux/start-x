"""Swing scanner — P4 entry: end-to-end validation harness + leakage canary (Charter v1).

Builds meta-labeled datasets across the tradeable v1 universe, runs the walk-forward meta-model, and
gates the result on DSR/PBO net of costs — NOTHING is called an edge unless it clears. First runs the
charter's mandatory leakage check (P4 gate): a shuffled-label run must give OOS AUC ~0.5 and a
future-peeking feature must spike to ~1.0; if either fails the harness is broken and we stop.

    python scripts/swing_validate.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.data.membership import SP500Membership
from startx.scanner.harness import (
    build_dataset,
    canary_lookahead,
    canary_shuffle,
    evaluate,
    walk_forward,
)
from startx.scanner.regime import classify, regime_panel

PRICE_DIR = "data/cache/prices"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
THRESHOLDS = tuple(np.round(np.arange(0.45, 0.701, 0.025), 3))   # finer grid -> meaningful CSCV/PBO
N_TRIALS = len(THRESHOLDS) + 4        # thresholds + 4 triggers searched; feeds the DSR bar (LEDGER.md)


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    print("building regime panel (conditioner)...")
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]

    print(f"building meta-labeled datasets across {len(UNIVERSE)} tradeable ETFs (next-open fills, costs)...")
    parts = []
    for s in UNIVERSE:
        try:
            ds = build_dataset(s, _load(s), spy, reg)
        except Exception as e:
            print(f"  {s}: skip ({type(e).__name__})"); continue
        if not ds.empty:
            parts.append(ds)
    data = pd.concat(parts, ignore_index=True)
    # report only the real-data window
    data = data[data["entry_date"] >= "2012-01-01"]
    base_rate = data["label"].mean()
    print(f"  {len(data)} events  base PT-first rate {base_rate:.0%}  "
          f"({data['entry_date'].min().date()}..{data['entry_date'].max().date()})")
    print("  by trigger: " + " ".join(f"{t}={int(n)}" for t, n in data['trigger'].value_counts().items()))

    # ---- P4 GATE: leakage canary FIRST (if this fails, the harness is broken) ----
    print("\n=== LEAKAGE CANARY (charter P4 gate) ===")
    auc_shuf = canary_shuffle(data)
    auc_leak = canary_lookahead(data)
    print(f"  shuffled-label OOS AUC  = {auc_shuf:.3f}  (must be ~0.50 — noise has no edge)")
    print(f"  lookahead-feature AUC   = {auc_leak:.3f}  (must be ~1.00 — leak is caught)")
    ok = (0.43 <= auc_shuf <= 0.57) and (auc_leak >= 0.90)
    print(f"  harness integrity: {'PASS' if ok else 'FAIL — fix before trusting any result'}")
    if not ok:
        return

    # ---- real walk-forward meta-model ----
    print("\n=== WALK-FORWARD META-MODEL (OOS, net of costs) ===")
    oos = walk_forward(data, train_min=400, retrain_every=50)
    by_thr, pbo, _ = evaluate(oos, THRESHOLDS, n_trials=N_TRIALS)
    print(f"{'thr':>6}{'n':>6}{'exp/trade':>11}{'PF':>6}{'CVaR5%':>9}{'Sharpe':>8}{'DSR':>7}")
    for thr in THRESHOLDS[::2]:                       # every other threshold to keep it readable
        s = by_thr.get(thr) or {}
        if not s:
            continue
        print(f"{thr:>6}{s['n']:>6}{s['expectancy']*100:>10.2f}%{s['profit_factor']:>6.2f}"
              f"{s['cvar5']*100:>8.2f}%{s['sharpe']:>8.2f}{s['dsr']:>7.2f}")
    print(f"\n  PBO (CSCV across {len(THRESHOLDS)} thresholds) = {pbo:.2f}  (must be < 0.5)")
    best = max((s for s in by_thr.values() if s), key=lambda x: x.get("dsr", -9), default=None)
    if best:
        promote = best["dsr"] > 0.5 and pbo < 0.5 and best["expectancy"] > 0
        print(f"  best: thr {best['threshold']} DSR {best['dsr']:.2f} exp {best['expectancy']*100:+.2f}% "
              f"PF {best['profit_factor']:.2f} -> {'PROMOTE' if promote else 'does NOT clear the gate'}")

    # ---- regime-conditional breakdown (the charter's core thesis: a signal outside its regime is noise)
    print("\n=== REGIME-CONDITIONAL expectancy (best threshold) — where does the edge live? ===")
    thr = best["threshold"] if best else 0.5
    taken = oos[oos["prob"] >= thr].copy()
    taken["regime"] = taken["entry_date"].map(reg["regime"].to_dict())
    for rg in ["risk_on", "neutral", "risk_off", "crisis"]:
        r = taken.loc[taken["regime"] == rg, "ret"]
        if len(r) >= 12:
            print(f"  {rg:9}: n={len(r):4d}  exp {r.mean()*100:+.2f}%  PF "
                  f"{(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else float('inf'):.2f}  win {(r>0).mean()*100:.0f}%")
    print(f"\n  N_trials fed to DSR = {N_TRIALS} (logged to LEDGER.md). win_rate is BANNED as selection.")


if __name__ == "__main__":
    main()
