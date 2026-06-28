"""Swing scanner — P5: the self-learning layer (Charter v1).

Three adaptive pieces on top of the validated P4 harness:
  1. REGIME-GATED hypothesis (pre-registered, theory-motivated): the desk's standing prior is that
     stress/capitulation is the best LONG entry (high VIX = bounce). So we gate the swing meta-model to
     stress regimes (risk_off + crisis) and test it OOS on the walk-forward stream — ONE added trial
     (N+1), not in-sample cherry-picking. Does the gated edge clear DSR/PBO where the pooled one didn't?
  2. OOS FEATURE PROMOTION/DECAY: MDA permutation importance measured out-of-sample (never in-sample
     gain) — which features carry real skill (promote) vs noise (retire). The feature set is alive.
  3. DRIFT MONITOR: rolling OOS expectancy of the taken book — flag decay.

    python scripts/swing_learn.py
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
    evaluate,
    oos_permutation_importance,
    walk_forward,
)
from startx.scanner.regime import classify, regime_panel

PRICE_DIR = "data/cache/prices"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
THRESHOLDS = tuple(np.round(np.arange(0.45, 0.701, 0.025), 3))
N_TRIALS = len(THRESHOLDS) + 4 + 1        # +1 for the pre-registered regime gate (honest accounting)
STRESS = ["risk_off", "crisis"]


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _gate_stats(oos, thresholds, n_trials, label):
    by_thr, pbo, _ = evaluate(oos, thresholds, n_trials=n_trials)
    best = max((s for s in by_thr.values() if s), key=lambda x: x.get("dsr", -9), default=None)
    if not best:
        print(f"  {label:16}: <12 trades"); return None
    promote = best["dsr"] > 0.5 and pbo < 0.5 and best["expectancy"] > 0
    print(f"  {label:16}: best thr {best['threshold']}  n={best['n']:4d}  exp {best['expectancy']*100:+.2f}%  "
          f"PF {best['profit_factor']:.2f}  Sharpe {best['sharpe']:+.2f}  DSR {best['dsr']:.2f}  PBO {pbo:.2f}  "
          f"-> {'PROMOTE' if promote else 'no'}")
    return best, pbo, promote


def main():
    print("building regime panel + meta-labeled datasets...")
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    parts = [build_dataset(s, _load(s), spy, reg) for s in UNIVERSE]
    data = pd.concat([d for d in parts if not d.empty], ignore_index=True)
    data = data[data["entry_date"] >= "2012-01-01"]
    oos = walk_forward(data, train_min=400, retrain_every=50)
    print(f"  {len(oos)} OOS events  ({oos['entry_date'].min().date()}..{oos['entry_date'].max().date()})\n")

    # 1) REGIME-GATED hypothesis (pre-registered): stress regimes only
    print("=== (1) REGIME-GATED validation — does gating to stress beat the pooled book? ===")
    _gate_stats(oos, THRESHOLDS, N_TRIALS, "ungated")
    gated = oos[oos["regime"].isin(STRESS)] if "regime" in oos else oos.iloc[0:0]
    _gate_stats(gated, THRESHOLDS, N_TRIALS, "stress-gated")

    # 2) OOS feature promotion/decay (MDA permutation importance)
    print("\n=== (2) OOS FEATURE IMPORTANCE (MDA permutation — promote winners, retire noise) ===")
    imp = oos_permutation_importance(data)
    if not imp.empty:
        print(f"  base OOS AUC {imp.attrs.get('base_auc', float('nan')):.3f}")
        print("  PROMOTE (top OOS contributors):")
        for _, r in imp.head(6).iterrows():
            print(f"    {r['feature']:16} auc_drop {r['auc_drop']:+.4f}")
        decay = imp[imp["auc_drop"] <= 0.0]
        print(f"  RETIRE candidates (auc_drop <= 0, no OOS skill): "
              f"{', '.join(decay['feature'].tolist()) if not decay.empty else 'none'}")

    # 3) DRIFT monitor — rolling OOS expectancy of the taken book
    print("\n=== (3) DRIFT MONITOR — rolling OOS expectancy (is the edge decaying?) ===")
    thr = 0.5
    taken = oos[oos["prob"] >= thr].set_index("entry_date")["ret"].sort_index()
    yearly = taken.groupby(taken.index.year).agg(["mean", "count"])
    for y, row in yearly.iterrows():
        bar = "+" if row["mean"] > 0 else "-"
        print(f"    {int(y)}: exp {row['mean']*100:+.2f}%  n={int(row['count'])}  {bar}")
    print(f"\n  N_trials={N_TRIALS} (regime gate counted). The feature set + regime gate ADAPT by OOS")
    print("  contribution — that is the self-learning. Honest verdict logged to LEDGER.md.")


if __name__ == "__main__":
    main()
