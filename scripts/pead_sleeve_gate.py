"""PEAD Round-2 gate: does the long PEAD drift ADD to the core long model, or is it just long beta?

R8 found the long top-quintile PEAD drift replicates OOS (+120bp/trade H21, DSR 1.0) but carries beta ~0.85
with insignificant market-neutral alpha. This gate answers the only question that matters for a NEW sleeve:
does it DIVERSIFY the (already long-biased) core, or double down on the same beta?

Builds the long PEAD sleeve (rule L, H=21) as an INVESTABLE daily stream (equal-weight across concurrent
open positions, uninvested capital = 0 — no active-day cherry-picking), over the full 2018-2026 window,
then:
  1. correlation of the sleeve to the CORE long-model stream and to SPY;
  2. a BETA-HEDGED sleeve (sleeve - beta*SPY) — does any diversifying alpha survive removing market beta?
  3. ensembles core+sleeve (risk-parity, 90/10, 80/20) vs core alone — Sharpe / CAGR / maxDD / DSR.

    python scripts/pead_sleeve_gate.py [--H 21]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.portfolio.long_model import vol_target_risk_parity
from startx.validation.metrics import deflated_sharpe

import importlib.util
spec = importlib.util.spec_from_file_location("study_pead", "scripts/study_pead.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

CORE_CACHE = "data/cache/satellite_core_lm_2018_2026.parquet"
TR = 252
N_PROJECT = 408          # project-cumulative trial count (honest deflation bar)


def _daily_sleeve(ev, rule, H):
    """Investable daily sleeve: equal-weight across concurrent open positions; a continuous calendar
    series (days with no open position = 0, uninvested capital earns 0)."""
    t = sp._trade_returns(ev, rule, H)
    px_cache: dict = {}
    daily: dict = {}
    sign = -1.0 if rule.startswith("S") else 1.0
    for _, tr in t.iterrows():
        sym = tr["symbol"]
        if sym not in px_cache:
            px_cache[sym] = sp._load_price(sym).set_index("date")
        p = px_cache[sym]
        ei = p.index.searchsorted(tr["entry_date"])
        xi = min(ei + H - 1, len(p) - 1)
        entry_open = float(p["open"].iloc[ei])
        closes = p["close"].iloc[ei: xi + 1].astype(float)
        prev = entry_open
        for j, (d, c) in enumerate(closes.items()):
            rr = sign * (c / prev - 1.0)
            if j == 0 or j == len(closes) - 1:
                rr -= sp.COST_PER_SIDE
            daily.setdefault(pd.Timestamp(d), []).append(rr)
            prev = c
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series({d: float(np.mean(v)) for d, v in sorted(daily.items())}, name="sleeve")


def _stats(r, label, n_trials=1):
    r = r.dropna()
    if len(r) < 30:
        return {"label": label, "n": len(r)}
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1).min())
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(r.mean() / sd * np.sqrt(TR)) if sd > 0 else float("nan")
    return {"label": label, "sharpe": sh, "cagr": cagr, "vol": float(sd * np.sqrt(TR)),
            "maxdd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
            "dsr": float(deflated_sharpe(r.values, n_trials=n_trials))}


def _row(s):
    if s.get("n", 99) < 30 and "sharpe" not in s:
        return f"  {s['label']:34}  (only {s.get('n',0)} days — underpowered)"
    return (f"  {s['label']:34}{s['sharpe']:>7.2f}{s['cagr']*100:>7.1f}%{s['vol']*100:>6.1f}%"
            f"{s['maxdd']*100:>8.1f}%{s['calmar']:>7.2f}{s['dsr']:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=21, help="sleeve hold horizon (R8 validated 21d)")
    args = ap.parse_args()

    core = pd.read_parquet(CORE_CACHE)["core"]
    core.index = pd.to_datetime(core.index)
    spy = sp._load_price("SPY").set_index("date")["close"].pct_change().rename("spy")
    ev = pd.read_parquet(sp.EVENTS_F)
    ev = ev[(ev["entry_date"] >= sp.REPORT_START) & (ev["entry_date"] <= sp.REPORT_END)]

    sleeve_active = _daily_sleeve(ev, "L", args.H)                 # active-day stream
    idx = core.index
    sleeve = sleeve_active.reindex(idx).fillna(0.0)               # investable calendar (uninvested = 0)
    spy_d = spy.reindex(idx).fillna(0.0)
    active_frac = float((sleeve != 0).mean())

    M = pd.concat([core.rename("core"), sleeve.rename("sleeve"), spy_d], axis=1).dropna()
    corr_core = float(M["sleeve"].corr(M["core"]))
    corr_spy = float(M["sleeve"].corr(M["spy"]))
    beta = float(np.polyfit(M["spy"], M["sleeve"], 1)[0])
    hedged = (M["sleeve"] - beta * M["spy"]).rename("sleeve_hedged")
    corr_hedged_core = float(hedged.corr(M["core"]))

    print(f"PEAD long sleeve (rule L, H={args.H}) — INVESTABLE daily stream, 2018-2026")
    print(f"  active days: {active_frac:.0%} of the calendar (uninvested capital = 0 on the rest)")
    print(f"  corr(sleeve, CORE) = {corr_core:+.2f}   corr(sleeve, SPY) = {corr_spy:+.2f}   "
          f"beta(sleeve~SPY) = {beta:+.2f}")
    print(f"  corr(beta-HEDGED sleeve, CORE) = {corr_hedged_core:+.2f}\n")

    print(f"  {'stream':34}{'Sharpe':>7}{'CAGR':>8}{'vol':>6}{'maxDD':>8}{'Calmar':>7}{'DSR':>7}  (DSR@N={N_PROJECT})")
    print("  " + "-" * 78)
    rows = [
        _stats(M["core"], "core: better long model (alone)", N_PROJECT),
        _stats(M["sleeve"], f"PEAD sleeve L/H{args.H} (raw)", N_PROJECT),
        _stats(hedged, f"PEAD sleeve L/H{args.H} (beta-hedged)", N_PROJECT),
    ]
    for r in rows:
        print(_row(r))

    # ensembles core + raw sleeve
    rp = vol_target_risk_parity({"core": M["core"], "sleeve": M["sleeve"]}).combined
    e9010 = 0.90 * M["core"] + 0.10 * M["sleeve"]
    e8020 = 0.80 * M["core"] + 0.20 * M["sleeve"]
    # ensemble core + beta-hedged sleeve (the pure-diversifier test)
    rp_h = vol_target_risk_parity({"core": M["core"], "hedged": hedged}).combined
    print()
    ens = [
        _stats(rp, "RP(core, raw sleeve)", N_PROJECT),
        _stats(e9010, "90/10 core/sleeve", N_PROJECT),
        _stats(e8020, "80/20 core/sleeve", N_PROJECT),
        _stats(rp_h, "RP(core, beta-HEDGED sleeve)", N_PROJECT),
    ]
    core_sh = rows[0]["sharpe"]
    for r in ens:
        lift = r["sharpe"] - core_sh
        print(_row(r) + f"   (lift {lift:+.2f})")

    print("\n=== VERDICT ===")
    diversifies = corr_core < 0.35 and rows[1].get("sharpe", -9) > 0
    hedged_alpha = rows[2].get("sharpe", -9) > 0.3 and rows[2].get("dsr", 0) > 0.5
    best_lift = max(e["sharpe"] for e in ens) - core_sh
    if best_lift > 0.03 and (diversifies or hedged_alpha):
        print(f"  PEAD sleeve ADDS: best ensemble lift {best_lift:+.2f} Sharpe over core alone "
              f"(corr {corr_core:+.2f}). PROMOTE small + gate on live decay.")
    elif corr_core >= 0.35:
        print(f"  PEAD sleeve is LONG-BETA (corr to core {corr_core:+.2f}) — it doubles down, not diversifies. "
              f"Beta-hedged residual Sharpe {rows[2].get('sharpe', float('nan')):.2f}, DSR "
              f"{rows[2].get('dsr', float('nan')):.2f} -> {'weak alpha' if hedged_alpha else 'no bankable alpha'}. "
              f"Best ensemble lift {best_lift:+.2f}. DO NOT promote as a diversifier.")
    else:
        print(f"  PEAD sleeve does not clearly add (best lift {best_lift:+.2f}, corr {corr_core:+.2f}). Inconclusive.")


if __name__ == "__main__":
    main()
