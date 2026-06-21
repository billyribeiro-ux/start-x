"""Run a production portfolio book end-to-end and (optionally) export the trade ledger.

THREE SEPARATE BOOKS, one per holding horizon — they are NOT mixed in one book (mixing a 1-10 day
dip with a multi-month trend ride under one exit rule is exactly what produced the 134-bar hold
inside a "swing" book). Each book is its own system with its own exit clock:

  • short_swing — SHORT-TERM SWING, 1-10 trading days. Oversold dip-buy in an uptrend (IBS<0.1).
  • long_swing  — LONG-TERM SWING, weeks to ~3 months. Trend breakout + fear capitulation.
  • position    — POSITION / PORTFOLIO, a long hold (months-to-years), 200-SMA trend core.

Within a book TWO MODELS are raced (the market-internals breadth guard is a risk filter judged
head-to-head, not by assertion):
  • base    — sleeves UNGATED (more trades, more raw return, deeper drawdown).
  • guarded — dip/entry breadth-guarded (skip entries into a broad-breakdown / heavy-down-volume day).

    python scripts/run_book.py                                   # short_swing, both models, full window
    python scripts/run_book.py --book long_swing --out book.csv  # the weeks-to-months book
    python scripts/run_book.py --book short_swing --start 2025-01-01 --end 2026-06-19 --out s.csv
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.portfolio import run_portfolio
from startx.portfolio.thesis import annotate_theses, summarize_book
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


# ------------------------------------------------------------------------------------------- #
# THREE horizon-separated books. Each carries its OWN sleeve membership (per model) and its OWN
# per-sleeve exits = (HARD STOP ATR mult, CHANDELIER ATR mult, MAX hold in trading days). The
# rule is the same everywhere — 1-ATR hard stop cuts the loss, 3-ATR chandelier rides the winner
# — but the MAX-HOLD clock is the horizon, and that is what differs by book.
# ------------------------------------------------------------------------------------------- #
BOOKS: dict[str, dict] = {
    "short_swing": {
        "desc": "SHORT-TERM SWING — 1-10 trading days. Oversold dip-buy in an uptrend (IBS<0.1); "
                "1-ATR stop, 3-ATR chandelier, HARD 10-day cap.",
        "models": {"base": {"ibs": _ibs_ungated}, "guarded": {"ibs": _ibs_guarded}},
        "exits": {"ibs": (1.0, 3.0, 10)},      # 1-10 trading days — a true short swing
    },
    "long_swing": {
        # Only `base`: neither breakout nor fear is breadth-gated, so a "guarded" model would be
        # byte-identical (the breadth guard lives in the IBS sleeve, which this book doesn't run).
        "desc": "LONG-TERM SWING — weeks to ~3 months. Trend breakout + fear capitulation; "
                "1-ATR stop, 3-ATR chandelier, ~3-month (63-day) cap.",
        "models": {"base": {"breakout": _breakout, "fear": _fear}},
        "exits": {"breakout": (1.0, 3.0, 63), "fear": (1.0, 3.0, 63)},  # weeks → ~3 months
    },
    "position": {
        "desc": "POSITION / PORTFOLIO — long hold (months to years). 200-SMA trend core; "
                "1-ATR stop, 3-ATR chandelier, multi-year (504-day) backstop.",
        "models": {"base": {"breakout": _breakout}},  # not breadth-gated → single model
        "exits": {"breakout": (1.0, 3.0, 504)},  # ride a long position to exhaustion
    },
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
                "exit_date", "exit_price", "exit_reason", "bars_held", "ret", "pnl_per_share",
                "pnl_contrib", "risk_pct", "R",
                "entry_rule", "exit_rule", "regime", "conviction", "thesis", "outcome"]


def _write_ledger(led: pd.DataFrame, path: str) -> int:
    """Write the trade ledger in the LOCKED layout + the totals block
    (TOTAL WIN $, TOTAL LOSS $, NET TOTAL $) summing per-share P&L net of ~2bp cost."""
    cols = [c for c in _LEDGER_COLS if c in led.columns]
    body = led[cols].copy()
    pps = body["pnl_per_share"] if "pnl_per_share" in body else pd.Series(dtype=float)
    win = float(pps[led["outcome"].values == "WIN"].sum()) if len(pps) else 0.0
    loss = float(pps[led["outcome"].values == "LOSS"].sum()) if len(pps) else 0.0
    totals = pd.DataFrame([
        {"sleeve": "TOTAL WIN $", "pnl_per_share": round(win, 2)},
        {"sleeve": "TOTAL LOSS $", "pnl_per_share": round(loss, 2)},
        {"sleeve": "NET TOTAL $", "pnl_per_share": round(win + loss, 2)},
    ])
    pd.concat([body, totals], ignore_index=True).to_csv(path, index=False)
    return len(body)


def _enrich_logic(led: pd.DataFrame, exits: dict) -> pd.DataFrame:
    """Make every trade self-documenting: spell out its entry rule, stop %, exit rule, and P&L."""
    if led.empty:
        return led
    led = led.copy()
    led["entry_rule"] = led["sleeve"].map(SLEEVE_RULES)
    led["stop_pct"] = ((led["stop_price"] / led["entry_price"] - 1.0) * 100).round(2)
    # per-share P&L, net of ~2bp round-trip cost (the locked-layout dollar column)
    led["pnl_per_share"] = ((led["exit_price"] - led["entry_price"])
                            - led["entry_price"] * 0.0002).round(2)

    # --- R-multiple: read the journal in units of RISK, not just % -----------------------
    # The R unit is the trade's initial 1-ATR risk distance — the fractional gap from entry to
    # the HARD stop (every sleeve uses a 1-ATR hard stop, so stop_price == entry - 1*ATR).
    risk_frac = (led["stop_price"] / led["entry_price"] - 1.0).abs()  # 1-ATR risk distance (fraction)
    led["risk_pct"] = (risk_frac * 100).round(3)                      # the R unit, in %
    led["R"] = (led["ret"] / risk_frac.where(risk_frac > 0)).round(3)  # NaN-safe on a 0-width stop

    def _exit_rule(sl: str) -> str:
        sm, cm, md = exits.get(sl, (1.0, 3.0, 252))
        return (f"EXIT: {cm:.0f}-ATR chandelier trail (ride the winner) + {sm:.0f}-ATR hard stop "
                f"(cut the loss); hard cap {md} trading days.")
    led["exit_rule"] = led["sleeve"].map(_exit_rule)
    return led


def _run_one(name, sleeves, exits, spy, aux, start, end, out, gross_cap, entry_fill):
    res = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=gross_cap,
                        entry_fill=entry_fill, start=start, end=end)
    s = res.stats
    print(f"\n=== MODEL: {name} ===")
    print(f"  n={s['n']} win={s['win_rate']*100:.0f}% total={s['total_return']*100:+.1f}% "
          f"maxDD={s['max_drawdown']*100:.1f}% PF={s['profit_factor']:.2f} "
          f"Sharpe={s.get('ann_sharpe', float('nan')):.2f} exposure={s.get('exposure', 0)*100:.0f}%")
    led = _enrich_logic(annotate_theses(res.ledger, spy, aux), exits)
    led.insert(0, "model", name)
    if not led.empty:
        print(f"  holds: min={int(led['bars_held'].min())}d  "
              f"max={int(led['bars_held'].max())}d  median={led['bars_held'].median():.0f}d")
    print("  BOOK NOW:", summarize_book(led))
    if out:
        path = out.replace(".csv", f"_{name}.csv")
        nrows = _write_ledger(led, path)
        print(f"  wrote {nrows} trades (+ TOTAL WIN $/LOSS $/NET $ block) -> {path}")
    return s, led


def _stress_grid(spy, aux, names, models, exits, gross_cap, entry_fill):
    """Both models across regimes, so their regime-dependence is visible at a glance."""
    windows = [("2019-22 (incl. bear)", "2019-01-01", "2022-12-31"),
               ("2023-26 (bull)", "2023-01-01", "2026-06-19"),
               ("FULL 2019-26", "2019-01-01", "2026-06-19")]
    print("\n=== STRESS GRID (models × regimes) ===")
    print(f"  {'window':22}{'model':9}{'n':>4}{'win':>6}{'total':>9}{'PF':>7}{'maxDD':>8}{'Sharpe':>8}")
    for wl, st, en in windows:
        for m in names:
            s = run_portfolio(spy, aux, models[m], exits=exits, gross_cap=gross_cap,
                              entry_fill=entry_fill, start=st, end=en).stats
            print(f"  {wl:22}{m:9}{s['n']:4}{s['win_rate']*100:5.0f}%{s['total_return']*100:+8.1f}%"
                  f"{s['profit_factor']:7.2f}{s['max_drawdown']*100:7.1f}%{s.get('ann_sharpe', 0):8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="short_swing", choices=list(BOOKS),
                    help="which horizon-separated book to run (default short_swing = 1-10 day)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-19")
    ap.add_argument("--model", default="both", choices=["base", "guarded", "both"])
    ap.add_argument("--out", default=None, help="write the annotated ledger(s) to this CSV")
    ap.add_argument("--stress", action="store_true", help="print the models × regimes stress grid")
    ap.add_argument("--gross-cap", type=float, default=1.5, dest="gross_cap",
                    help="max concurrent gross leverage (leverage drill: 1.5x sweet spot, 1.0x un-levered)")
    ap.add_argument("--entry-fill", default="close", choices=["close", "next_open"], dest="entry_fill",
                    help="fill entries at the signal-day close (default) or the next bar's open")
    args = ap.parse_args()

    book = BOOKS[args.book]
    models, exits = book["models"], book["exits"]

    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"),
           "internals": load_internals()}

    requested = ["base", "guarded"] if args.model == "both" else [args.model]
    names = [n for n in requested if n in models]
    if not names:  # e.g. --model guarded on a book that only has `base`
        names = list(models)[:1]
        print(f"note: book '{args.book}' has no '{args.model}' model "
              f"(its sleeves aren't breadth-gated); running '{names[0]}'.")
    print(f"BOOK: {args.book}  —  {book['desc']}")
    print(f"WINDOW {args.start} -> {args.end}  "
          f"(models: {', '.join(names)} | gross_cap {args.gross_cap}x | entry {args.entry_fill})")
    stats, ledgers = {}, []
    for n in names:
        s, led = _run_one(n, models[n], exits, spy, aux, args.start, args.end, args.out,
                          args.gross_cap, args.entry_fill)
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
        nc = _write_ledger(comb, allp)
        print(f"\nwrote {nc} trades (all models, + TOTAL WIN $/LOSS $/NET $ block) -> {allp}")

    if args.stress:
        _stress_grid(spy, aux, names, models, exits, args.gross_cap, args.entry_fill)


if __name__ == "__main__":
    main()
