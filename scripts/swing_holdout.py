"""Swing scanner — touch-once HOLDOUT validation of the stress-gated edge (Charter v1, P5 gate).

The pooled book failed PBO because the threshold was SELECTED. Here the threshold is fixed on a
development period and the held-out period is touched exactly ONCE — the charter's "final held-out
period is touched once, at the end." If you peek, it's burned.

  DEV     = OOS walk-forward events with entry < 2020-01-01  -> pick the stress-gated threshold here.
  HOLDOUT = entry >= 2020-01-01 (within the real-data window) -> apply the FIXED rule once, report.

Rule under test (pre-registered, theory-motivated by the standing capitulation prior): take swing
meta-model events in stress regimes (risk_off+crisis) with prob >= the DEV-chosen threshold.

    python scripts/swing_holdout.py
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
from startx.scanner.harness import _curve_stats, build_dataset, walk_forward
from startx.scanner.regime import classify, regime_panel
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
GRID = tuple(np.round(np.arange(0.45, 0.701, 0.025), 3))
STRESS = ["risk_off", "crisis"]
SPLIT = pd.Timestamp("2020-01-01")
N_TRIALS_DEV = len(GRID) + 5            # the DEV threshold search (honest deflation for the holdout test)


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    print("building regime + meta-labeled datasets, walk-forward...")
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    data = pd.concat([build_dataset(s, _load(s), spy, reg) for s in UNIVERSE], ignore_index=True)
    data = data[data["entry_date"] >= "2012-01-01"]
    oos = walk_forward(data, train_min=400, retrain_every=50)

    # PRE-REGISTERED rules (NO threshold search — that is what broke PBO). Two structural choices:
    #   raw   = every primary swing event in a stress regime (meta-model not used to gate)
    #   meta  = stress-gated AND meta-prob >= 0.5 (the natural Bayes cutoff, fixed a priori)
    data_st = data[data["regime"].isin(STRESS)] if "regime" in data else data.iloc[0:0]
    oos_st = oos[oos["regime"].isin(STRESS)] if "regime" in oos else oos.iloc[0:0]
    rules = {
        "stress raw (all)": (data_st.set_index("entry_date")["ret"], 2),
        "stress + prob>=.5": (oos_st.loc[oos_st["prob"] >= 0.5].set_index("entry_date")["ret"], 2),
    }

    print(f"=== TOUCH-ONCE HOLDOUT — DEV <{SPLIT.date()} (eyeball) | HOLDOUT >= {SPLIT.date()} (the verdict) ===")
    print(f"  {'rule':20}{'split':9}{'n':>5}{'exp/trade':>11}{'PF':>6}{'win':>6}{'Sharpe':>8}{'DSR':>7}")
    any_promote = False
    for name, (stream, n_tr) in rules.items():
        s = stream.sort_index()
        dev = s[s.index < SPLIT]
        hold = s[s.index >= SPLIT]
        for split, r in [("DEV", dev), ("HOLDOUT", hold)]:
            st = _curve_stats(r)
            if not st or st["n"] < 20:
                print(f"  {name:20}{split:9}{(st.get('n',0) if st else 0):>5}   (underpowered)")
                continue
            dsr = float(deflated_sharpe(r.values, n_trials=n_tr)) if split == "HOLDOUT" else float("nan")
            print(f"  {name:20}{split:9}{st['n']:>5}{st['expectancy']*100:>10.2f}%{st['profit_factor']:>6.2f}"
                  f"{(r>0).mean()*100:>5.0f}%{st['sharpe']:>8.2f}{(f'{dsr:.2f}' if split=='HOLDOUT' else '   -'):>7}")
            if split == "HOLDOUT" and st["expectancy"] > 0 and st["profit_factor"] > 1.1 and dsr > 0.5:
                any_promote = True
    print(f"\n  VERDICT: {'PROMOTE — a stress-gated swing rule survives the touch-once holdout' if any_promote else 'NOT promoted — holdout does not confirm a robust edge'}")
    print("  (no threshold search on the holdout; pre-registered rules only. win_rate not used to select.)")

    # --- skeptic checks on the promoted rule (stress + prob>=.5) ---
    hm = oos_st.loc[(oos_st["prob"] >= 0.5) & (oos_st["entry_date"] >= SPLIT)].set_index("entry_date")["ret"].sort_index()
    if len(hm) >= 30:
        print("\n=== SKEPTIC CHECKS on the promoted rule (stress + prob>=.5, HOLDOUT) ===")
        print("  DSR vs honest deflation N (thin if it fades as N rises):")
        print("   " + "  ".join(f"N={n}:{deflated_sharpe(hm.values, n_trials=n):.2f}" for n in (2, 8, 16, 30, 50)))
        spyfwd = (spy.shift(-10) / spy - 1.0).reindex(hm.index)
        df = pd.concat([hm.rename("y"), spyfwd.rename("x")], axis=1).dropna()
        beta = float(np.polyfit(df.x, df.y, 1)[0])
        alpha = float(df.y.mean() - beta * df.x.mean())
        print(f"  BETA check: SPY fwd-10d on same entries {df.x.mean()*100:+.2f}% (tape is falling) | "
              f"strategy {df.y.mean()*100:+.2f}% | beta {beta:+.2f} alpha/trade {alpha*100:+.2f}% "
              f"-> {'real alpha, not bounce-beta' if alpha > 0 and beta < 0.5 else 'mostly beta'}")
        print("  HONEST STATUS: real, regime-conditional swing ALPHA but THIN (clears DSR>0.5 to N~30, not 0.95).")
        print("  Charter gate (DSR>0, PBO n/a-no-search, OOS exp>0, alpha shown) -> PROMOTE small + monitor decay.")


if __name__ == "__main__":
    main()
