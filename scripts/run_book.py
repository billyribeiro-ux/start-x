"""Run the production portfolio book end-to-end and (optionally) export the trade ledger.

Wires the four validated sleeves into the portfolio engine, annotates every trade with a
human-readable PM thesis + conviction + regime (the `thesis` layer), and scores the *real* book
ledger with the validation scorecard (deflated Sharpe / PBO / drift-adjusted alpha).

    python scripts/run_book.py                         # full 2019->now, print stats + scorecard
    python scripts/run_book.py --start 2025-01-01 --out book_2025.csv   # export an annotated ledger
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.portfolio import run_portfolio
from startx.portfolio.thesis import annotate_theses, summarize_book
from startx.portfolio.validate import confidence_scorecard
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals


def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


#: The book: three independent sleeves wrapped to the engine's ``sig(spy, aux)`` contract. The two
#: vol-premium triggers (VRP, VVIX) are MERGED into one ``fear`` sleeve so a vol spike books a single
#: position, not two identical ones (fixes the 2026-03-18 same-bar/same-price stacking).
SLEEVES = {
    "breakout": lambda s, a: breakout_signals(s, 20, 200),   # the trend edge (only real alpha)
    "ibs": lambda s, a: ibs_signals(s),                      # oversold-dip exposure timing
    "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"]),  # VRP ∪ VVIX vol-premium long (one position)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-19")
    ap.add_argument("--n-trials", type=int, default=10, dest="n_trials")
    ap.add_argument("--out", default=None, help="write the annotated ledger to this CSV")
    args = ap.parse_args()

    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD")}

    res = run_portfolio(spy, aux, SLEEVES, start=args.start, end=args.end)
    s = res.stats
    print(f"PRODUCTION BOOK  {args.start} -> {args.end}")
    print(f"  n={s['n']} win={s['win_rate']*100:.0f}% total={s['total_return']*100:+.1f}% "
          f"maxDD={s['max_drawdown']*100:.1f}% PF={s['profit_factor']:.2f} "
          f"Sharpe={s.get('ann_sharpe', float('nan')):.2f} exposure={s.get('exposure', 0)*100:.0f}%")

    led = annotate_theses(res.ledger, spy, aux)
    print("\nBOOK NOW:", summarize_book(led))
    print("\nSCORECARD (real book ledger):")
    print(confidence_scorecard(led, res.equity, n_trials=args.n_trials, spy=spy).to_string())

    if args.out:
        cols = [c for c in ["sleeve", "entry_date", "entry_price", "stop_price", "exit_date",
                            "exit_price", "exit_reason", "bars_held", "weight", "ret",
                            "pnl_contrib", "regime", "conviction", "thesis", "outcome"]
                if c in led.columns]
        led[cols].to_csv(args.out, index=False)
        print(f"\nWrote {len(led)} annotated trades -> {args.out}")


if __name__ == "__main__":
    main()
