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

#: Per-sleeve exit = (HARD STOP ATR mult, CHANDELIER ATR mult, MAX hold in trading days).
#: Rule: 1-ATR hard stop cuts the loss short; 3-ATR chandelier rides the winner. Horizon differs by
#: sleeve — a short-term dip vs a long-term trend/capitulation ride. One global exit is wrong.
EXITS = {
    "breakout": (1.0, 3.0, 252),   # long-term trend: 1-ATR stop, 3-ATR trail, ride for months
    "fear":     (1.0, 3.0, 252),   # capitulation: 1-ATR stop, 3-ATR trail, ride to exhaustion (the chandelier exits)
    "ibs":      (1.0, 3.0, 10),    # SHORT-TERM dip (1-10d): 1-ATR stop, 3-ATR trail, ~2-week hard cap
}


#: Exact ENTRY criteria per sleeve, in plain English (the spec — see STRATEGY.md for full detail).
SLEEVE_RULES = {
    "breakout": "ENTRY: SPY closes at a NEW 20-day high while above its 200-day SMA (fresh trend breakout).",
    "ibs": "ENTRY: IBS<0.10 — close in the bottom 10% of the day's range, above the 200-day SMA "
           "(oversold dip in an uptrend); guarded model also requires up-volume >20% (no falling knife).",
    "fear": "ENTRY: VRP (VIX − 20d realized vol) in the top 5% of its trailing year, OR VVIX ≥ its "
            "trailing-year 90th pct (fear over-priced → bounce).",
}

_LEDGER_COLS = ["model", "sleeve", "entry_date", "entry_price", "stop_price", "stop_pct",
                "exit_date", "exit_price", "exit_reason", "bars_held", "ret", "pnl_contrib",
                "entry_rule", "exit_rule", "regime", "conviction", "thesis", "outcome"]


def _enrich_logic(led: pd.DataFrame) -> pd.DataFrame:
    """Make every trade self-documenting: spell out its entry rule, stop %, and exit rule."""
    if led.empty:
        return led
    led = led.copy()
    led["entry_rule"] = led["sleeve"].map(SLEEVE_RULES)
    led["stop_pct"] = ((led["stop_price"] / led["entry_price"] - 1.0) * 100).round(2)

    def _exit_rule(sl: str) -> str:
        sm, cm, md = EXITS.get(sl, (1.0, 3.0, 252))
        return (f"EXIT: {cm:.0f}-ATR chandelier trail (ride the winner) + {sm:.0f}-ATR hard stop "
                f"(cut the loss); hard cap {md} trading days.")
    led["exit_rule"] = led["sleeve"].map(_exit_rule)
    return led


def _run_one(name, sleeves, spy, aux, start, end, out):
    res = run_portfolio(spy, aux, sleeves, exits=EXITS, start=start, end=end)
    s = res.stats
    print(f"\n=== MODEL: {name} ===")
    print(f"  n={s['n']} win={s['win_rate']*100:.0f}% total={s['total_return']*100:+.1f}% "
          f"maxDD={s['max_drawdown']*100:.1f}% PF={s['profit_factor']:.2f} "
          f"Sharpe={s.get('ann_sharpe', float('nan')):.2f} exposure={s.get('exposure', 0)*100:.0f}%")
    led = _enrich_logic(annotate_theses(res.ledger, spy, aux))
    led.insert(0, "model", name)
    print("  BOOK NOW:", summarize_book(led))
    if out:
        path = out.replace(".csv", f"_{name}.csv")
        led[[c for c in _LEDGER_COLS if c in led.columns]].to_csv(path, index=False)
        print(f"  wrote {len(led)} trades -> {path}")
    return s, led


def _stress_grid(spy, aux, names):
    """Both models across regimes, so their regime-dependence is visible at a glance."""
    windows = [("2019-22 (incl. bear)", "2019-01-01", "2022-12-31"),
               ("2023-26 (bull)", "2023-01-01", "2026-06-19"),
               ("FULL 2019-26", "2019-01-01", "2026-06-19")]
    print("\n=== STRESS GRID (models × regimes) ===")
    print(f"  {'window':22}{'model':9}{'n':>4}{'win':>6}{'total':>9}{'PF':>7}{'maxDD':>8}{'Sharpe':>8}")
    for wl, st, en in windows:
        for m in names:
            s = run_portfolio(spy, aux, MODELS[m], exits=EXITS, start=st, end=en).stats
            print(f"  {wl:22}{m:9}{s['n']:4}{s['win_rate']*100:5.0f}%{s['total_return']*100:+8.1f}%"
                  f"{s['profit_factor']:7.2f}{s['max_drawdown']*100:7.1f}%{s.get('ann_sharpe', 0):8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-19")
    ap.add_argument("--n-trials", type=int, default=10, dest="n_trials")
    ap.add_argument("--model", default="both", choices=["base", "guarded", "both"])
    ap.add_argument("--out", default=None, help="write the annotated ledger(s) to this CSV")
    ap.add_argument("--stress", action="store_true", help="print the models × regimes stress grid")
    args = ap.parse_args()

    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"),
           "internals": load_internals()}

    names = ["base", "guarded"] if args.model == "both" else [args.model]
    print(f"PRODUCTION BOOK  {args.start} -> {args.end}  (models: {', '.join(names)})")
    stats, ledgers = {}, []
    for n in names:
        s, led = _run_one(n, MODELS[n], spy, aux, args.start, args.end, args.out)
        stats[n] = s; ledgers.append(led)

    if len(names) == 2:
        print("\n=== HEAD-TO-HEAD ===")
        print(f"  {'metric':12}" + "".join(f"{n:>12}" for n in names))
        for k, fmt in [("n", "{:d}"), ("win_rate", "{:.0%}"), ("total_return", "{:+.1%}"),
                       ("profit_factor", "{:.2f}"), ("max_drawdown", "{:.1%}"), ("ann_sharpe", "{:.2f}")]:
            row = "".join(f"{fmt.format(stats[n].get(k, 0)):>12}" for n in names)
            print(f"  {k:12}{row}")

    # combined "all trades" export (every trade from every model, tagged by `model`)
    if args.out and len(ledgers) > 1:
        allp = args.out.replace(".csv", "_all.csv")
        comb = pd.concat(ledgers, ignore_index=True)
        comb[[c for c in _LEDGER_COLS if c in comb.columns]].to_csv(allp, index=False)
        print(f"\nwrote {len(comb)} trades (all models) -> {allp}")

    if args.stress:
        _stress_grid(spy, aux, names)


if __name__ == "__main__":
    main()
