"""Swing scanner — P7: horizon generalization (Charter v1, the EOD-feasible part).

Does the validated stress-gated edge survive when the LABEL horizon shrinks from SWING (10d) to
SHORT_SWING (3d)? Same features / triggers / regime gate — only the triple-barrier horizon and ATR
multiples re-weight (the charter's "horizon is a first-class config axis"). Touch-once holdout 2020-26,
the pre-registered rule (stress + prob>=0.5), net of costs. DAY/0DTE/SCALP need intraday + options data
(seams) and are out of scope here.

    python scripts/swing_horizon.py
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
STRESS = ["risk_off", "crisis"]
SPLIT = pd.Timestamp("2020-01-01")
HORIZONS = {                                  # name -> (horizon_days, pt_mult, sl_mult)
    "SHORT_SWING 3d": (3, 1.5, 1.0),
    "SWING 10d (validated)": (10, 2.0, 1.0),
}


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    prices = {s: _load(s) for s in UNIVERSE}
    print("=== P7 HORIZON GENERALIZATION — stress + prob>=0.5, touch-once holdout 2020-26 ===")
    print(f"  {'horizon':24}{'devN':>6}{'holdN':>7}{'exp':>9}{'PF':>6}{'Sharpe':>8}{'DSR16':>7}{'alpha':>8}{'beta':>6}")
    for name, (h, pt, sl) in HORIZONS.items():
        data = pd.concat([build_dataset(s, prices[s], spy, reg, pt_mult=pt, sl_mult=sl, horizon=h)
                          for s in UNIVERSE], ignore_index=True)
        data = data[data["entry_date"] >= "2012-01-01"]
        oos = walk_forward(data, train_min=400, retrain_every=50)
        st = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)]
        dev = st[st["entry_date"] < SPLIT]
        hold = st[st["entry_date"] >= SPLIT].set_index("entry_date")["ret"].sort_index()
        s = _curve_stats(hold)
        if not s or s["n"] < 20:
            print(f"  {name:24}{len(dev):>6}{(s.get('n',0) if s else 0):>7}   underpowered"); continue
        dsr = float(deflated_sharpe(hold.values, n_trials=16))
        # beta/alpha vs SPY forward return over the same horizon
        spyfwd = (spy.shift(-h) / spy - 1.0).reindex(hold.index)
        df = pd.concat([hold.rename("y"), spyfwd.rename("x")], axis=1).dropna()
        beta = float(np.polyfit(df.x, df.y, 1)[0]) if len(df) > 10 else float("nan")
        alpha = float(df.y.mean() - beta * df.x.mean()) if np.isfinite(beta) else float("nan")
        print(f"  {name:24}{len(dev):>6}{s['n']:>7}{s['expectancy']*100:>8.2f}%{s['profit_factor']:>6.2f}"
              f"{s['sharpe']:>8.2f}{dsr:>7.2f}{alpha*100:>7.2f}%{beta:>6.2f}")
    print("\n  Read: if SHORT_SWING holds positive exp + alpha + DSR>0.5, the taxonomy generalizes to 3d.")
    print("  If it collapses, the edge is swing-horizon-specific (also a real finding). win_rate not used.")


if __name__ == "__main__":
    main()
