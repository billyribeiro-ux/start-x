"""Run the GENUINE paper-forward harness — the three VALIDATED books on a HOLDOUT window.

This is the honest counterpart to the abandoned ``prob_up`` forward harness: instead of a
coin-flip directional model, it drives the three long-only production books
(short_swing / long_swing / position) in a forward / out-of-sample manner — every ENTRY is
clipped to ``[--forward-start, --end]`` so the blotter is genuinely OOS on a slice the book
parameters were not tuned on (the desk calibrated on 2019-22; 2023-26 is held out).

HONEST FRAMING: this is a backtest-driven forward SIMULATION on historical holdout data —
there is NO live broker, no real fills. The value is an OOS track record, not a P&L claim.

    python scripts/forward_books.py
    python scripts/forward_books.py --forward-start 2023-01-01 --end 2026-06-19
    python scripts/forward_books.py --forward-start 2024-01-01 --out blotter.csv
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.forward.book_paper import (
    DEFAULT_END,
    DEFAULT_FORWARD_START,
    run_forward_books,
)


def _print_book(name: str, book) -> None:
    s = book.stats
    print(f"\n=== BOOK: {name}  (OUT-OF-SAMPLE / holdout) ===")
    print(
        f"  n={s['n']}  win={s['win_rate']*100:.0f}%  total={s['total_return']*100:+.1f}%  "
        f"maxDD={s['max_drawdown']*100:.1f}%  Sharpe={s['ann_sharpe']:.2f}"
        + (f"  PF={s['profit_factor']:.2f}" if "profit_factor" in s else "")
        + (f"  exposure={s['exposure']*100:.0f}%" if "exposure" in s else "")
    )
    if "benchmark" in s and s["benchmark"]:
        b = s["benchmark"]
        print(
            f"  vs SPY buy&hold:  total={b.get('total_return', float('nan'))*100:+.1f}%  "
            f"maxDD={b.get('max_drawdown', float('nan'))*100:.1f}%  "
            f"Sharpe={b.get('ann_sharpe', float('nan')):.2f}   "
            f"(position book = drawdown defense, lags B&H in a bull window by design)"
        )
    bl = book.blotter
    if not bl.empty:
        n_open = int((bl["status"] == "open").sum())
        print(
            f"  holds: min={int(bl['bars_held'].min())}d  max={int(bl['bars_held'].max())}d  "
            f"median={bl['bars_held'].median():.0f}d   open-at-end={n_open}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward-start", default=DEFAULT_FORWARD_START, dest="forward_start",
                    help="holdout START — entries before this are never taken (default 2023-01-01)")
    ap.add_argument("--end", default=DEFAULT_END, help="holdout END (default 2026-06-19)")
    ap.add_argument("--gross-cap", type=float, default=1.5, dest="gross_cap",
                    help="concurrent gross-leverage cap for the chandelier books (default 1.5x)")
    ap.add_argument("--out", default=None, help="write the combined tagged blotter to this CSV")
    ap.add_argument("--max-rows", type=int, default=30, dest="max_rows",
                    help="max combined-blotter rows to print (default 30; 0 = all)")
    args = ap.parse_args()

    print("PAPER-FORWARD over the THREE VALIDATED BOOKS — HISTORICAL-HOLDOUT SIMULATION")
    print("(NOT a live-broker track record; the value is an out-of-sample paper blotter)")
    print(f"HOLDOUT WINDOW {args.forward_start} -> {args.end}  "
          f"(entries clipped to the window | gross_cap {args.gross_cap}x)")

    res = run_forward_books(args.forward_start, args.end, gross_cap=args.gross_cap)

    for name, book in res.books.items():
        _print_book(name, book)

    # --- combined OOS blotter ----------------------------------------------------------- #
    comb = res.combined_blotter
    print(f"\n=== COMBINED OOS PAPER BLOTTER ({len(comb)} trades, all books tagged) ===")
    if comb.empty:
        print("  (no trades in window)")
    else:
        show = comb if args.max_rows == 0 else comb.head(args.max_rows)
        with pd.option_context("display.max_rows", None, "display.width", 200):
            cols = ["book", "sleeve", "entry_date", "entry_price", "exit_date", "exit_price",
                    "exit_reason", "bars_held", "ret", "pnl_per_share", "outcome", "status"]
            disp = show[cols].copy()
            disp["entry_date"] = disp["entry_date"].dt.date
            disp["exit_date"] = disp["exit_date"].dt.date
            disp["ret"] = (disp["ret"] * 100).round(2)
            print(disp.to_string(index=False))
        if args.max_rows and len(comb) > args.max_rows:
            print(f"  ... ({len(comb) - args.max_rows} more rows; --max-rows 0 to show all)")

    print("\n=== ONE-LINE HONEST VERDICT ===")
    print("  " + res.summary)

    if args.out:
        res.combined_blotter.to_csv(args.out, index=False)
        print(f"\nwrote {len(res.combined_blotter)} OOS trades (all books, tagged) -> {args.out}")


if __name__ == "__main__":
    main()
