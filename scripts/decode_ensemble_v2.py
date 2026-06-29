"""THE DECODED SYSTEM, re-assembled with the VALIDATED 2-ATR exit (R6 follow-up).

The original decode (`decode_ensemble.py`) found the stress-gated swing book was Sharpe ~0.84 / -36% DD and
DRAGGED the better long model (1.22 -> 0.82) — but that used the OLD tight 1-ATR exit. R6 proved a 2-ATR
stop lifts the swing book's standalone risk profile a lot. The question this answers: with the 2-ATR exit,
does the swing book now ADD to the better long model (genuine diversification) instead of dragging?

Apples-to-apples by construction: the SAME cached taken book (the 366 stress-gated trades R6 validated on)
is turned into an investable capped calendar book TWICE — once under the 1-ATR exit (cached returns) and
once under the 2-ATR exit (re-simulated) — then each is risk-parity-combined with the same better long
model. Only the swing exit differs, so the delta is the exit's portfolio effect.

    python scripts/decode_ensemble_v2.py [--start 2018-01-01] [--max-concurrent 10]
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

from startx.portfolio import run_portfolio
from startx.portfolio.long_model import vol_target_risk_parity
from startx.scanner.improve import calendar_book
from startx.scanner.scan import STOP_ATR_MULT
from startx.strategy.market_internals import load_internals
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals

PRICE_DIR = "data/cache/prices"
TAKEN_CACHE = "data/cache/selflearn_taken.parquet"
TR = 252


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _book_stream(spy, aux, sleeves, exits, start, end):
    eq = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=1.5, start=start, end=end).equity
    m = (eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))
    eq = eq[m]
    return (eq / eq.iloc[0]).pct_change().fillna(0.0)


def _stats(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1).min())
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(r.mean() / sd * np.sqrt(TR)) if sd > 0 else float("nan")
    return dict(label=label, sharpe=sh, cagr=cagr, maxdd=mdd, vol=float(sd * np.sqrt(TR)),
                calmar=cagr / abs(mdd) if mdd < 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-06-25")
    ap.add_argument("--max-concurrent", type=int, default=10, dest="maxc")
    args = ap.parse_args()

    # ---- the better long model (unchanged validated core) --------------------------------------------
    spy = _load("SPY")
    spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"), "internals": load_internals()}
    spy_close = spy.set_index("date")["close"]
    print("building the better long model (short IBS + long breakout/fear, vol-targeted RP)...")
    short_r = _book_stream(spy, aux, {"ibs": lambda s, a: ibs_signals(s)},
                           {"ibs": (1, 3, 10)}, args.start, args.end)
    long_r = _book_stream(spy, aux, {"breakout": lambda s, a: breakout_signals(s, 20, 200),
                                     "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"])},
                          {"breakout": (1, 3, 63), "fear": (1, 3, 63)}, args.start, args.end)
    long_model = vol_target_risk_parity({"short": short_r, "long": long_r}).combined

    # ---- the swing book, SAME cached taken events, two exits -----------------------------------------
    taken = pd.read_parquet(TAKEN_CACHE)
    syms = sorted(taken["symbol"].unique())
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    prices = {s: _load(s) for s in syms if s in cached}
    print(f"swing book from cached taken events ({len(taken)} trades, {len(prices)} symbols); "
          f"capped <= {args.maxc} concurrent.")

    swing_1atr = calendar_book(taken, taken["ret"].to_numpy(float), max_concurrent=args.maxc)
    from startx.scanner.improve import resim_trades
    t2 = resim_trades(taken, prices, pt_mult=2.0, sl_mult=STOP_ATR_MULT, horizon=10)
    swing_2atr = calendar_book(t2, t2["ret"].to_numpy(float), max_concurrent=args.maxc)

    def combine(swing, name):
        M = pd.concat([long_model.rename("lm"), swing.rename("sw")], axis=1).dropna()
        m = (M.index >= pd.Timestamp(args.start)) & (M.index <= pd.Timestamp(args.end))
        M = M[m]
        combo = vol_target_risk_parity({"lm": M["lm"], "sw": M["sw"]}).combined
        corr = float(M["lm"].corr(M["sw"]))
        return M, combo, corr

    M1, combo1, corr1 = combine(swing_1atr, "1-ATR")
    M2, combo2, corr2 = combine(swing_2atr, "2-ATR")
    spy_r = spy_close.pct_change().reindex(M2.index).fillna(0.0)

    lm_stats = _stats(M2["lm"], "better long model")
    print(f"\n=== DECODED SYSTEM v2 — does the 2-ATR swing book ADD to the long model? ({args.start}..{args.end}) ===")
    print(f"  corr(long_model, swing): 1-ATR {corr1:+.2f}   2-ATR {corr2:+.2f}\n")
    print(f"  {'stream':28}{'Sharpe':>8}{'CAGR':>8}{'vol':>7}{'maxDD':>8}{'Calmar':>8}")
    rows = [
        lm_stats,
        _stats(M1["sw"], "swing book (1-ATR)"),
        _stats(M2["sw"], "swing book (2-ATR)"),
        _stats(combo1, "ensemble w/ 1-ATR swing"),
        _stats(combo2, "ensemble w/ 2-ATR swing"),
        _stats(spy_r, "SPY buy&hold"),
    ]
    for s in rows:
        print(f"  {s['label']:28}{s['sharpe']:>8.2f}{s['cagr']*100:>7.1f}%{s['vol']*100:>6.1f}%"
              f"{s['maxdd']*100:>7.1f}%{s['calmar']:>8.2f}")

    lift1 = rows[3]["sharpe"] - lm_stats["sharpe"]
    lift2 = rows[4]["sharpe"] - lm_stats["sharpe"]
    print("\n  diversification lift over long-model-alone:")
    print(f"    1-ATR swing: {lift1:+.2f} Sharpe  ({'ADDS' if lift1 > 0.02 else 'drags/neutral'})")
    print(f"    2-ATR swing: {lift2:+.2f} Sharpe  ({'ADDS' if lift2 > 0.02 else 'drags/neutral'})")

    # ---- robustness: is the 2-ATR>1-ATR DIRECTION stable across the concurrency cap? -----------------
    # (the capped-calendar Sharpe can be inflated by averaging correlated stress-day names; if the cap
    #  changes the verdict, the magnitude isn't bankable. We trust the DIRECTION, not the level.)
    print("\n  concurrency-cap sensitivity (ensemble Sharpe; tests the over-diversification caveat):")
    print(f"    {'cap':>4}{'lm+1ATR':>10}{'lm+2ATR':>10}{'delta':>8}")
    for cap in (3, 5, 10, 20):
        s1 = calendar_book(taken, taken["ret"].to_numpy(float), max_concurrent=cap)
        s2 = calendar_book(t2, t2["ret"].to_numpy(float), max_concurrent=cap)
        _, c1, _ = combine(s1, "1")
        _, c2, _ = combine(s2, "2")
        e1, e2 = _stats(c1, "")["sharpe"], _stats(c2, "")["sharpe"]
        print(f"    {cap:>4}{e1:>10.2f}{e2:>10.2f}{e2-e1:>8.2f}")

    if lift2 > 0.02 and lift2 > lift1 + 0.05:
        print("\n  -> the validated 2-ATR exit materially improves the swing book AS A DIVERSIFIER vs 1-ATR.")
        print("     Robust finding = the DIRECTION (2-ATR >> 1-ATR). Treat the absolute ensemble Sharpe as")
        print("     construction-optimistic (capped-calendar can overstate via correlated stress-day names).")
    else:
        print("\n  -> the 2-ATR exit does not clearly improve the swing book's portfolio contribution.")


if __name__ == "__main__":
    main()
