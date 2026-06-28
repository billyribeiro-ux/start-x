"""Swing scanner — P6 live output: explainable, calibrated setups (Charter v1).

Trains the validated stress-gated meta-model, then surfaces qualifying swing setups with a CALIBRATED
conviction, ranked driver attribution, invalidation level, and regime-matched expectancy/CVaR. If the
CURRENT regime isn't stress (the gate is off), it says so and shows the most recent qualifying cohort as
a worked example of the output card — never a bare score.

    python scripts/swing_scan_live.py [--lookback 10]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.data.membership import SP500Membership
from startx.scanner.harness import build_dataset
from startx.scanner.regime import classify, regime_panel
from startx.scanner.scan import STRESS, fit, surface

PRICE_DIR = "data/cache/prices"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _card(s):
    drv = ", ".join(f"{n}{'+' if v >= 0 else ''}{v:.2f}" for n, v in s.drivers)
    print(f"  {s.symbol:5} {str(s.date.date())}  {s.direction.upper():5} [{s.regime}]  trigger={s.trigger}")
    print(f"        conviction(calibrated) {s.conviction:.0%}  |  entry~{s.entry_ref:.2f}  "
          f"invalidation {s.invalidation:.2f}")
    print(f"        drivers: {drv}")
    print(f"        regime-cohort: exp {s.cohort_exp*100:+.2f}%  CVaR5% {s.cohort_cvar5*100:.2f}%  (n={s.cohort_n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10)
    args = ap.parse_args()

    print("training the validated stress-gated meta-model + calibrating...")
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    prices = {s: _load(s) for s in UNIVERSE}
    data = pd.concat([build_dataset(s, prices[s], spy, reg) for s in UNIVERSE], ignore_index=True)
    data = data[data["entry_date"] >= "2012-01-01"]
    sm = fit(data)

    asof = data["entry_date"].max()
    cur_regime = reg.loc[reg.index <= asof, "regime"].iloc[-1]
    print(f"\nSWING SCANNER — as-of {asof.date()}  |  current regime: {cur_regime}")
    print("(promoted rule: LONG swing setups in a STRESS regime with calibrated conviction >= 50%)\n")

    setups = surface(sm, prices, reg, asof=asof, lookback=args.lookback)
    if setups:
        print(f"=== {len(setups)} live setup(s) in the last {args.lookback} sessions ===")
        for s in setups:
            _card(s)
    else:
        off = cur_regime not in STRESS
        print(f"=== no live setups{' — gate OFF (current regime is not stress)' if off else ''} ===")
        # show the most recent qualifying cohort as a worked example of the output card
        hist = surface(sm, prices, reg, asof=asof, lookback=4000)
        if hist:
            ex = hist[0] if False else max(hist, key=lambda s: s.date)
            print("\nMost recent qualifying setup (worked example of the card format):")
            _card(ex)
    print("\nEvery setup is explained to its drivers + a regime-matched cohort — no bare scores (charter).")


if __name__ == "__main__":
    main()
