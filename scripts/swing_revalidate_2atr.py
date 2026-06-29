"""Bank the 2-ATR exit: re-validate the FULL pipeline under a consistent 2-ATR barrier (R6 follow-up).

R6 applied a 2-ATR EXIT to entries chosen by a meta-model trained on 1-ATR-stop LABELS — entry and exit
disagreed. This retrains the whole pipeline so the LABEL is "PT(2ATR) before SL(2ATR) within 10d" (entry
and exit agree), then re-runs the touch-once holdout firewall + skeptic checks, side-by-side vs the 1-ATR
baseline on the SAME survivorship-free universe. Question: does the stress-gated edge still clear the
firewall under the consistent barrier — i.e. is it safe to bank 2-ATR as the swing pipeline default?

  DEV     = walk-forward OOS events, entry < 2020-01-01  (eyeball)
  HOLDOUT = entry >= 2020-01-01, touched once            (the verdict)
  rule    = stress regime (risk_off+crisis) AND meta-prob >= 0.5  (pre-registered, no threshold search)

    python scripts/swing_revalidate_2atr.py [--stocks 80]
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

from startx.data.membership import REAL_DATA_START, SP500Membership
from startx.scanner.harness import _curve_stats, build_dataset, walk_forward
from startx.scanner.regime import classify, regime_panel
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
STRESS = ["risk_off", "crisis"]
SPLIT = pd.Timestamp("2020-01-01")


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    if len(d) < 500:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _liquid_stocks(mem, n, min_dv=50.0):
    ever = set()
    for d in ["2018-06-01", "2020-06-01", "2022-06-01", "2024-06-01", "2026-01-01"]:
        ever |= set(mem.members_asof(d))
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    rows = []
    for s in sorted((ever & cached) - set(ETFS)):
        px = _load(s)
        if px is None:
            continue
        dv = float((px["close"] * px["volume"]).tail(252).median())
        if dv >= min_dv * 1e6:
            rows.append((s, dv))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def _holdout(oos, spy, label):
    """Touch-once holdout on the pre-registered stress+prob>=0.5 rule. Returns the holdout stat dict."""
    st_oos = oos[oos["regime"].isin(STRESS)] if "regime" in oos else oos.iloc[0:0]
    taken = st_oos.loc[st_oos["prob"] >= 0.5]
    s = taken.set_index("entry_date")["ret"].sort_index()
    dev, hold = s[s.index < SPLIT], s[s.index >= SPLIT]
    out = {"label": label, "dev_n": len(dev), "hold_n": len(hold)}
    print(f"\n--- {label} ---")
    print(f"  {'split':9}{'n':>5}{'exp/trade':>11}{'PF':>6}{'win':>6}{'Sharpe':>8}{'DSR(N=2)':>10}")
    for split, r in [("DEV", dev), ("HOLDOUT", hold)]:
        cs = _curve_stats(r)
        if not cs or cs["n"] < 20:
            print(f"  {split:9}{(cs.get('n', 0) if cs else 0):>5}   (underpowered)")
            continue
        dsr = float(deflated_sharpe(r.values, n_trials=2)) if split == "HOLDOUT" else float("nan")
        print(f"  {split:9}{cs['n']:>5}{cs['expectancy']*100:>10.2f}%{cs['profit_factor']:>6.2f}"
              f"{(r > 0).mean()*100:>5.0f}%{cs['sharpe']:>8.2f}"
              f"{(f'{dsr:.2f}' if split == 'HOLDOUT' else '   -'):>10}")
        if split == "HOLDOUT":
            out.update(exp=cs["expectancy"], pf=cs["profit_factor"], sharpe=cs["sharpe"], dsr=dsr,
                       win=float((r > 0).mean()))
    # skeptic: DSR vs N + beta/alpha
    if len(hold) >= 30:
        print("   DSR vs N: " + "  ".join(f"N={n}:{deflated_sharpe(hold.values, n_trials=n):.2f}"
                                          for n in (2, 8, 16, 30)))
        spyfwd = (spy.shift(-10) / spy - 1.0).reindex(hold.index)
        df = pd.concat([hold.rename("y"), spyfwd.rename("x")], axis=1).dropna()
        beta = float(np.polyfit(df.x, df.y, 1)[0])
        alpha = float(df.y.mean() - beta * df.x.mean())
        out.update(alpha=alpha, beta=beta)
        print(f"   BETA: SPY fwd-10d {df.x.mean()*100:+.2f}% | strat {df.y.mean()*100:+.2f}% | "
              f"beta {beta:+.2f} alpha/trade {alpha*100:+.2f}% "
              f"-> {'real alpha' if alpha > 0 and beta < 0.5 else 'mostly beta'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=80)
    args = ap.parse_args()

    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    universe = ETFS + _liquid_stocks(mem, args.stocks)
    prices = {s: _load(s) for s in universe}
    print(f"re-validation universe: {len(universe)} symbols ({len(ETFS)} ETFs + {len(universe)-len(ETFS)} stocks)")

    results = {}
    for sl in (1.0, 2.0):
        cache = f"data/cache/reval_oos_sl{sl}.parquet"
        if os.path.exists(cache):
            oos = pd.read_parquet(cache)
            print(f"\n[sl_mult={sl}] loaded cached oos ({len(oos)} events)")
        else:
            print(f"\n[sl_mult={sl}] building meta-labeled dataset (consistent barrier) + walk-forward...")
            data = pd.concat([build_dataset(s, prices[s], spy, reg, sl_mult=sl) for s in universe],
                             ignore_index=True)
            data = data[data["entry_date"] >= REAL_DATA_START]
            oos = walk_forward(data, train_min=400, retrain_every=50)
            oos.to_parquet(cache)
            print(f"  {len(oos)} OOS events; cached -> {cache}")
        results[sl] = _holdout(oos, spy, f"sl_mult={sl} ({'1-ATR baseline' if sl == 1.0 else '2-ATR consistent'})")

    a, b = results.get(1.0, {}), results.get(2.0, {})
    print("\n=== BANK-IT VERDICT (consistent 2-ATR barrier vs 1-ATR baseline, HOLDOUT) ===")
    if a and b and "dsr" in a and "dsr" in b:
        print(f"  1-ATR : exp {a.get('exp',0)*100:+.2f}%  PF {a.get('pf',0):.2f}  Sharpe {a.get('sharpe',0):+.2f}  "
              f"DSR {a.get('dsr',0):.2f}  alpha {a.get('alpha',float('nan'))*100:+.2f}% beta {a.get('beta',float('nan')):+.2f}")
        print(f"  2-ATR : exp {b.get('exp',0)*100:+.2f}%  PF {b.get('pf',0):.2f}  Sharpe {b.get('sharpe',0):+.2f}  "
              f"DSR {b.get('dsr',0):.2f}  alpha {b.get('alpha',float('nan'))*100:+.2f}% beta {b.get('beta',float('nan')):+.2f}")
        bank = (b.get("exp", -9) > 0 and b.get("pf", 0) > 1.1 and b.get("dsr", 0) > 0.5
                and b.get("sharpe", -9) >= a.get("sharpe", -9) - 0.05)
        print(f"\n  -> {'BANK the consistent 2-ATR pipeline' if bank else 'DO NOT bank — 2-ATR consistent does not clear the holdout firewall'} "
              f"(2-ATR holdout {'clears' if bank else 'fails'} DSR>0.5 + PF>1.1 + exp>0, Sharpe >= baseline).")
    else:
        print("  insufficient holdout sample to render a verdict.")


if __name__ == "__main__":
    main()
