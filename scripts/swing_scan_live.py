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
import glob
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.data.membership import SP500Membership
from startx.scanner.harness import build_dataset
from startx.scanner.regime import classify, regime_panel
from startx.scanner.scan import STRESS, fit, surface
from startx.scanner.selflearn import ScannerMemory

PRICE_DIR = "data/cache/prices"
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]   # ETFs + index ETFs


def _liquid_stocks(mem, n, min_dv=50.0):
    """Top-n liquid S&P stocks (point-in-time members ∩ cached, >= min_dv $M/day) — the STOCKS pillar."""
    ever = set()
    for d in ["2020-06-01", "2023-06-01", "2026-01-01"]:
        ever |= set(mem.members_asof(d))
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    rows = []
    for s in sorted((ever & cached) - set(ETFS)):
        try:
            px = pd.read_parquet(f"{PRICE_DIR}/{s}.parquet", columns=["close", "volume"])
        except Exception:
            continue
        if len(px) < 500:
            continue
        dv = float((px["close"] * px["volume"]).tail(252).median())
        if dv >= min_dv * 1e6:
            rows.append((s, dv))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _card(s):
    drv = ", ".join(f"{n}{'+' if v >= 0 else ''}{v:.2f}" for n, v in s.drivers)
    flag = "  ⛔AVOIDED" if s.vetoed else ""
    print(f"  {s.symbol:5} {str(s.date.date())}  {s.direction.upper():5} [{s.regime}]  trigger={s.trigger}{flag}")
    print(f"        conviction(calibrated) {s.conviction:.0%}  |  entry~{s.entry_ref:.2f}  "
          f"invalidation {s.invalidation:.2f}")
    print(f"        drivers: {drv}")
    print(f"        regime-cohort: exp {s.cohort_exp*100:+.2f}%  CVaR5% {s.cohort_cvar5*100:.2f}%  (n={s.cohort_n})")
    if s.vetoed:
        print(f"        self-learned avoid: {s.veto_reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--stocks", type=int, default=40, help="number of liquid stocks to add (the STOCKS pillar)")
    args = ap.parse_args()

    print("training the validated stress-gated meta-model + calibrating...")
    mem = SP500Membership.load()
    reg = classify(regime_panel(membership=mem))
    spy = _load("SPY").set_index("date")["close"]
    universe = ETFS + _liquid_stocks(mem, args.stocks)        # ETFs + indexes + STOCKS (all three pillars)
    print(f"  scanning {len(universe)} symbols: {len(ETFS)} ETFs/indexes + {len(universe)-len(ETFS)} stocks")
    prices = {s: _load(s) for s in universe}
    data = pd.concat([build_dataset(s, prices[s], spy, reg) for s in universe], ignore_index=True)
    data = data[data["entry_date"] >= "2012-01-01"]
    sm = fit(data)

    avoid_rules = ScannerMemory.load().promoted_rules()      # the self-learned loss-avoidance layer
    if avoid_rules:
        print(f"  self-learned avoid layer ACTIVE: {len(avoid_rules)} rule(s) from data/scanner_memory.json")
        for r in avoid_rules:
            print(f"    ⛔ {r.rationale or r.key()}")

    asof = data["entry_date"].max()
    cur_regime = reg.loc[reg.index <= asof, "regime"].iloc[-1]
    print(f"\nSWING SCANNER — as-of {asof.date()}  |  current regime: {cur_regime}")
    print("(promoted rule: LONG swing setups in a STRESS regime with calibrated conviction >= 50%)\n")

    setups = surface(sm, prices, reg, asof=asof, lookback=args.lookback, avoid_rules=avoid_rules)
    if setups:
        live = [s for s in setups if not s.vetoed]
        avoided = [s for s in setups if s.vetoed]
        print(f"=== {len(live)} actionable + {len(avoided)} self-avoided setup(s) in the last {args.lookback} sessions ===")
        for s in setups:
            _card(s)
    else:
        off = cur_regime not in STRESS
        print(f"=== no live setups{' — gate OFF (current regime is not stress)' if off else ''} ===")
        # show the most recent qualifying cohort as a worked example of the output card
        hist = surface(sm, prices, reg, asof=asof, lookback=4000, avoid_rules=avoid_rules)
        if hist:
            ex = max(hist, key=lambda s: s.date)
            print("\nMost recent qualifying setup (worked example of the card format):")
            _card(ex)
    print("\nEvery setup is explained to its drivers + a regime-matched cohort — no bare scores (charter).")


if __name__ == "__main__":
    main()
