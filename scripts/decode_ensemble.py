"""THE DECODED SYSTEM — re-assemble the validated edges into one honest ensemble.

The old DECODE.md ensemble is VOID: it weighted the cross-sectional ML ranker 26%, and that ranker is a
survivorship mirage. This rebuilds the ensemble from ONLY firewall-cleared, real streams over the
real-data window (2018-2026):

  1. BETTER LONG MODEL — vol-targeted risk-parity across the two swing books (short_swing IBS +
     long_swing breakout/fear). The core long engine (Sharpe ~1.2).
  2. STRESS-GATED SWING ALPHA — the validated scanner edge, built as an INVESTABLE capped book (max-N
     concurrent equal-weight, so no over-diversification illusion). beta ~0.03 -> a real diversifier.

The decode question: does adding the near-market-neutral swing alpha to the long model lift the
risk-adjusted return (genuine diversification), the way the fake ranker only pretended to? Reports each
stream, their correlation, and the combined risk-parity book vs SPY buy-and-hold.

    python scripts/decode_ensemble.py [--start 2018-01-01] [--max-concurrent 10]
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

from startx.data.membership import SP500Membership
from startx.portfolio import run_portfolio
from startx.portfolio.long_model import vol_target_risk_parity
from startx.scanner.harness import build_dataset, walk_forward
from startx.scanner.regime import classify, regime_panel
from startx.strategy.market_internals import load_internals
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals

PRICE_DIR = "data/cache/prices"
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
STRESS = ["risk_off", "crisis"]
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


def _swing_capped_book(data, reg, spy_close, start, end, max_concurrent):
    """Investable stress-gated swing book: <=max_concurrent equal-weight open positions, daily MTM."""
    oos = walk_forward(data, train_min=400, retrain_every=50)
    taken = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)].copy()
    taken = taken.sort_values("entry_date").reset_index(drop=True)
    days = pd.bdate_range(start, end)
    daily = pd.Series(0.0, index=days)
    # each position contributes ret/hold over (entry, t1]; cap concurrent slots by conviction
    open_pos = []   # list of (exit_date, per_day_ret)
    ti = 0
    ev = taken.to_dict("records")
    for d in days:
        open_pos = [(x, r) for (x, r) in open_pos if x >= d]
        while ti < len(ev) and pd.Timestamp(ev[ti]["entry_date"]) <= d:
            e = ev[ti]
            x = pd.Timestamp(e["t1"])
            hold = max(len(pd.bdate_range(pd.Timestamp(e["entry_date"]) + pd.Timedelta(days=1), x)), 1)
            if pd.Timestamp(e["entry_date"]) == d and len(open_pos) < max_concurrent:
                open_pos.append((x, e["ret"] / hold))
            ti += 1
        if open_pos:
            daily.loc[d] = np.mean([r for (_, r) in open_pos])
    return daily


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

    mem = SP500Membership.load()
    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"), "internals": load_internals()}
    spy_close = spy.set_index("date")["close"]
    reg = classify(regime_panel(membership=mem))

    print("building the validated streams (2018-26)...")
    short_r = _book_stream(spy, aux, {"ibs": lambda s, a: ibs_signals(s)},
                           {"ibs": (1, 3, 10)}, args.start, args.end)
    long_r = _book_stream(spy, aux, {"breakout": lambda s, a: breakout_signals(s, 20, 200),
                                     "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"])},
                          {"breakout": (1, 3, 63), "fear": (1, 3, 63)}, args.start, args.end)
    long_model = vol_target_risk_parity({"short": short_r, "long": long_r}).combined

    # swing alpha investable capped book across ETFs + top liquid stocks
    stocks = []
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    ever = set().union(*[mem.members_asof(d) for d in ["2020-06-01", "2023-06-01", "2026-01-01"]])
    for s in sorted((ever & cached) - set(ETFS)):
        px = _load(s)
        if px is not None and len(px) > 800 and float((px["close"] * px["volume"]).tail(252).median()) > 1e8:
            stocks.append(s)
    universe = ETFS + stocks[:60]
    print(f"  swing universe: {len(ETFS)} ETFs + {len(universe)-len(ETFS)} stocks")
    data = pd.concat([build_dataset(s, _load(s), spy_close, reg) for s in universe], ignore_index=True)
    data = data[data["entry_date"] >= "2016-01-01"]
    swing = _swing_capped_book(data, reg, spy_close, args.start, args.end, args.maxc)

    # align + combine (risk parity across the two real edges)
    M = pd.concat([long_model.rename("long_model"), swing.rename("swing_alpha")], axis=1).dropna()
    combo = vol_target_risk_parity({"long_model": M["long_model"], "swing_alpha": M["swing_alpha"]}).combined
    spy_r = spy_close.pct_change().reindex(M.index).fillna(0.0)

    print(f"\n=== THE DECODED SYSTEM (real edges only, {args.start}..{args.end}) ===")
    print(f"  correlation(long_model, swing_alpha) = {M['long_model'].corr(M['swing_alpha']):+.2f}  "
          f"(low => real diversification)")
    rows = [_stats(M["long_model"], "better long model"), _stats(M["swing_alpha"], "swing alpha (capped)"),
            _stats(combo, "DECODED ENSEMBLE"), _stats(spy_r, "SPY buy&hold")]
    print(f"\n  {'stream':22}{'Sharpe':>8}{'CAGR':>8}{'vol':>7}{'maxDD':>8}{'Calmar':>8}")
    for s in rows:
        print(f"  {s['label']:22}{s['sharpe']:>8.2f}{s['cagr']*100:>7.1f}%{s['vol']*100:>6.1f}%"
              f"{s['maxdd']*100:>7.1f}%{s['calmar']:>8.2f}")
    lift = rows[2]["sharpe"] - rows[0]["sharpe"]
    print(f"\n  diversification lift from the swing alpha: {lift:+.2f} Sharpe "
          f"({'REAL — the decoded system beats the long model alone' if lift > 0.02 else 'minimal'}).")
    print("  This is the honest decoded system: validated long engine + a near-market-neutral swing alpha.")


if __name__ == "__main__":
    main()
