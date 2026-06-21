"""Run the production portfolio book end-to-end and (optionally) export the trade ledger.

Wires the validated sleeves into the portfolio engine, annotates every trade with a human-readable
PM thesis + conviction + regime (the `thesis` layer), and scores the *real* book ledger with the
validation scorecard (deflated Sharpe / PBO / drift-adjusted alpha).

TWO MODELS are kept side by side so we can race them (the market-internals breadth guard is a risk
filter whose value shows up in stress, so we judge it head-to-head, not by assertion):
  • **base**    — IBS dip sleeve UNGATED (more trades, more raw return, deeper drawdown).
  • **guarded** — IBS dip sleeve guarded by market internals (skip dips into a broad-breakdown /
                  heavy-down-volume day; higher win-rate, lower drawdown, small raw-return give-up).

    python scripts/run_book.py                                  # both models, full window, compared
    python scripts/run_book.py --start 2023-01-01 --out book.csv --model both
    python scripts/run_book.py --model guarded                  # one model only
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
from startx.strategy.market_internals import load_internals, not_breaking_down


def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


def _breakout(s, a):
    return breakout_signals(s, 20, 200)


def _fear(s, a):
    return fear_signals(s, a["vix"], a["vvix"])


def _ibs_ungated(s, a):
    return ibs_signals(s)


def _ibs_guarded(s, a):
    """IBS dip entry, guarded by market internals: skip dips bought into a broad down-volume
    breakdown (the falling-knife losers; cuts book drawdown ~20%)."""
    return (ibs_signals(s).reset_index(drop=True) & not_breaking_down(a["internals"], s)).astype(bool)


#: Both books kept side by side. Only the IBS sleeve differs (ungated vs breadth-guarded); breakout
#: and the merged VRP∪VVIX `fear` sleeve are identical across models.
MODELS = {
    "base": {"breakout": _breakout, "ibs": _ibs_ungated, "fear": _fear},
    "guarded": {"breakout": _breakout, "ibs": _ibs_guarded, "fear": _fear},
}


def _run_one(name, sleeves, spy, aux, start, end, n_trials, out):
    res = run_portfolio(spy, aux, sleeves, start=start, end=end)
    s = res.stats
    print(f"\n=== MODEL: {name} ===")
    print(f"  n={s['n']} win={s['win_rate']*100:.0f}% total={s['total_return']*100:+.1f}% "
          f"maxDD={s['max_drawdown']*100:.1f}% PF={s['profit_factor']:.2f} "
          f"Sharpe={s.get('ann_sharpe', float('nan')):.2f} exposure={s.get('exposure', 0)*100:.0f}%")
    led = annotate_theses(res.ledger, spy, aux)
    print("  BOOK NOW:", summarize_book(led))
    if out:
        path = out if len(MODELS) == 1 else out.replace(".csv", f"_{name}.csv")
        cols = [c for c in ["sleeve", "entry_date", "entry_price", "stop_price", "exit_date",
                            "exit_price", "exit_reason", "bars_held", "weight", "ret",
                            "pnl_contrib", "regime", "conviction", "thesis", "outcome"]
                if c in led.columns]
        led[cols].to_csv(path, index=False)
        print(f"  wrote {len(led)} trades -> {path}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-19")
    ap.add_argument("--n-trials", type=int, default=10, dest="n_trials")
    ap.add_argument("--model", default="both", choices=["base", "guarded", "both"])
    ap.add_argument("--out", default=None, help="write the annotated ledger(s) to this CSV")
    args = ap.parse_args()

    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"),
           "internals": load_internals()}

    names = ["base", "guarded"] if args.model == "both" else [args.model]
    print(f"PRODUCTION BOOK  {args.start} -> {args.end}  (models: {', '.join(names)})")
    stats = {n: _run_one(n, MODELS[n], spy, aux, args.start, args.end, args.n_trials, args.out)
             for n in names}

    if len(names) == 2:
        print("\n=== HEAD-TO-HEAD ===")
        hdr = f"  {'metric':12}" + "".join(f"{n:>12}" for n in names)
        print(hdr)
        for k, fmt in [("n", "{:d}"), ("win_rate", "{:.0%}"), ("total_return", "{:+.1%}"),
                       ("profit_factor", "{:.2f}"), ("max_drawdown", "{:.1%}"), ("ann_sharpe", "{:.2f}")]:
            row = "".join(f"{fmt.format(stats[n].get(k, 0)):>12}" for n in names)
            print(f"  {k:12}{row}")


if __name__ == "__main__":
    main()
