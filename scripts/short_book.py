"""Run the short-the-index engine over a window and report results (+ locked CSV trade sheet).

The desk's settled finding (3-agent short study): shorting the index has NO deployable standalone
edge — the upward drift is a permanent headwind. The least-bad survivor is ``overbought_short``
(RSI2>90 while close < 200-SMA: sell the overbought *bounce* inside a confirmed downtrend), a thin
confirmed-downtrend HEDGE that LOSES in bull regimes. This runner makes that concrete on real data.

    python scripts/short_book.py --start 2019-01-01 --end 2026-06-25            # locked window
    python scripts/short_book.py --symbol _GSPC --start 1990-01-01 --end 2026-06-25  # deep history
    python scripts/short_book.py --signal overbought --out short_overbought_trades.csv

Writes the trade ledger in the LOCKED layout (+ TOTAL WIN $ / TOTAL LOSS $ / NET TOTAL $ block,
per-share P&L net of ~2bp cost). One position at a time, so every row is chartable.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.strategy.short_engine import SHORT_SIGNALS, short_backtest


def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


_LEDGER_COLS = ["sleeve", "side", "entry_date", "entry_price", "stop_price", "exit_date",
                "exit_price", "exit_reason", "bars_held", "ret", "pnl_per_share", "outcome"]


def _write_ledger(led: pd.DataFrame, path: str) -> int:
    """LOCKED layout + totals block. A +1-ATR/breakeven exit is a WIN/SCRATCH (non-loser)."""
    if led.empty:
        pd.DataFrame(columns=_LEDGER_COLS).to_csv(path, index=False)
        return 0
    body = led[[c for c in _LEDGER_COLS if c in led.columns]].copy()
    pps = body["pnl_per_share"]
    win = float(pps[led["outcome"].isin(["WIN", "SCRATCH"]).values].sum())
    loss = float(pps[(led["outcome"] == "LOSS").values].sum())
    totals = pd.DataFrame([
        {"sleeve": "TOTAL WIN $", "pnl_per_share": round(win, 2)},
        {"sleeve": "TOTAL LOSS $", "pnl_per_share": round(loss, 2)},
        {"sleeve": "NET TOTAL $", "pnl_per_share": round(win + loss, 2)},
    ])
    pd.concat([body, totals], ignore_index=True).to_csv(path, index=False)
    return len(body)


def _fmt(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "n=0  (no trades fired in window)"
    return (f"n={s['n']:<3d} win {s['win_rate']:5.0%}  expectancy {s['expectancy_pct']:+.2f}%  "
            f"PF {s['profit_factor']:.2f}  total {s['total_return']:+.1%}  "
            f"avg {s['avg_bars']:.0f}d  net $/sh {s['net_per_share']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-25")
    ap.add_argument("--signal", default="overbought", choices=list(SHORT_SIGNALS) + ["all"])
    ap.add_argument("--stop-mult", type=float, default=1.0, dest="stop_mult")
    ap.add_argument("--trail-mult", type=float, default=3.0, dest="trail_mult")
    ap.add_argument("--max-days", type=int, default=10, dest="max_days")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = _load(args.symbol)
    last = pd.to_datetime(p["date"]).max().date()
    print(f"SHORT-THE-INDEX ENGINE — {args.symbol}  {args.start} -> {args.end}  "
          f"(data through {last}; stop {args.stop_mult}ATR / trail {args.trail_mult}ATR / "
          f"cap {args.max_days}d)\n")
    print("Desk verdict: NO deployable standalone short edge — shipped as a confirmed-downtrend "
          "HEDGE only.\n")

    names = list(SHORT_SIGNALS) if args.signal == "all" else [args.signal]
    survivor = None
    for name in names:
        sig = SHORT_SIGNALS[name](p)
        res = short_backtest(p, sig, stop_mult=args.stop_mult, trail_mult=args.trail_mult,
                             max_days=args.max_days, start=args.start, end=args.end)
        tag = "  <- the only survivor" if name == "overbought" else ""
        print(f"  {name:11} {_fmt(res.stats)}{tag}")
        if name == "overbought":
            survivor = res

    # write the survivor's trade sheet (chartable, locked layout)
    out = args.out
    if out is None and survivor is not None and not survivor.ledger.empty:
        out = f"short_{args.symbol.lower()}_overbought_trades.csv"
    if out is not None and survivor is not None:
        n = _write_ledger(survivor.ledger, out)
        print(f"\nwrote {n} short trades (+ TOTAL WIN $/LOSS $/NET $ block) -> {out}")
        if not survivor.ledger.empty:
            print("\nTrade ledger (overbought_short — chart these):")
            cols = ["entry_date", "entry_price", "stop_price", "exit_date", "exit_price",
                    "exit_reason", "bars_held", "pnl_per_share", "outcome"]
            print(survivor.ledger[cols].to_string(index=False))


if __name__ == "__main__":
    main()
