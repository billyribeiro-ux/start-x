"""Stream the three CLEAN production books through the ensemble — IN-SAMPLE vs WALK-FORWARD.

`ensemble.combine()` fits max-Sharpe tangency weights and scores them on the SAME history — that
is in-sample, and an in-sample diversification "lift" is the easiest number in finance to fake.
This script re-streams the three locked production books (short_swing, long_swing, position) through
BOTH `combine` (in-sample) and the new `combine_walkforward` (genuine OOS, trailing-window weights
applied to the next period only) and prints the brutally honest comparison: does the diversification
benefit SURVIVE out-of-sample weighting, or does it shrink/vanish once the weights can't peek?

These three books are all long-SPY-ish (high pairwise correlation), so the prior is that the
ensemble adds little over just holding the best single book — this script measures exactly how little.

    python scripts/ensemble_books.py
    python scripts/ensemble_books.py --lookback 24 --freq ME
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.portfolio import run_portfolio
from startx.portfolio.ensemble import combine, combine_walkforward
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals
from startx.strategy.trend_position import position_book

START, END = "2019-01-01", "2026-06-19"


def load(s: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{s}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


def build_streams() -> dict[str, pd.Series]:
    """The three CLEAN production books as daily return streams, clipped to the locked window.

    NOTE: ``run_portfolio(...).equity`` is indexed over the FULL SPY history (flat at 1.0 before the
    first in-window trade closes), so we must clip to ``START`` before ``pct_change()`` — otherwise
    the pre-2019 flat segment would flood the monthly resample with zero-return periods and corrupt
    both the in-sample and the walk-forward weight estimation. The position book already starts in
    window, but we clip it identically for a common, aligned index.
    """
    spy = load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": load("_VIX"), "vvix": load("_VVIX"), "gld": load("GLD")}

    short = run_portfolio(
        spy, aux, {"ibs": lambda s, a: ibs_signals(s)},
        exits={"ibs": (1.0, 3.0, 10)}, gross_cap=1.5, start=START, end=END,
    ).equity.loc[START:].pct_change()

    long = run_portfolio(
        spy, aux,
        {"breakout": lambda s, a: breakout_signals(s, 20, 200),
         "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"])},
        exits={"breakout": (1.0, 3.0, 63), "fear": (1.0, 3.0, 63)},
        gross_cap=1.5, start=START, end=END,
    ).equity.loc[START:].pct_change()

    pos = position_book(spy, start=START, end=END).equity.loc[START:].pct_change()

    return {"short_swing": short, "long_swing": long, "position": pos}


def _fmt(x: float, fmt: str = "{:.2f}") -> str:
    try:
        return fmt.format(x) if x == x else "  nan"  # x != x → NaN
    except (TypeError, ValueError):
        return "  nan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=24,
                    help="trailing periods for walk-forward weight estimation (default 24 = 2yr ME)")
    ap.add_argument("--freq", default="ME", help="rebalance frequency (default ME = month-end)")
    ap.add_argument("--n-trials", type=int, default=8, dest="n_trials",
                    help="DSR multiple-testing trials (default 8)")
    args = ap.parse_args()

    streams = build_streams()
    print(f"BOOKS streamed over {START} -> {END}  (freq={args.freq}, lookback={args.lookback})")
    for k, v in streams.items():
        vv = v.dropna()
        print(f"  {k:12} daily obs={len(vv):5}  {vv.index[0].date()} -> {vv.index[-1].date()}")

    insample = combine(streams, method="max_sharpe", long_only=True, freq=args.freq,
                       n_trials=args.n_trials)
    wf = combine_walkforward(streams, lookback=args.lookback, freq=args.freq, long_only=True,
                             method="max_sharpe", n_trials=args.n_trials)

    # --- correlation: why we expect little lift -------------------------------------------- #
    print("\n=== BOOK-vs-BOOK monthly return correlation (all long-SPY-ish) ===")
    print(insample.corr.round(3).to_string())

    # --- in-sample (fixed weights, fit on all history) ------------------------------------- #
    print("\n=== IN-SAMPLE combine() — weights fit & scored on the SAME history ===")
    print("  static weights:", {k: round(v, 3) for k, v in insample.weights.items()})
    print(insample.per_edge.round(3).to_string())

    # --- walk-forward (OOS) ---------------------------------------------------------------- #
    print(f"\n=== WALK-FORWARD combine_walkforward() — OOS, {args.lookback}-period trailing weights ===")
    print(f"  OOS periods: {wf.stats['n_periods']}  "
          f"(first {args.lookback} burned for the initial estimation window)")
    print("  mean OOS weights:", {k: round(v, 3) for k, v in wf.stats["mean_weights"].items()})
    print("\n  time-varying weights (every OOS period):")
    print(wf.weights.round(3).to_string())

    # --- headline comparison --------------------------------------------------------------- #
    iss, wfs = insample.stats, wf.stats
    best_edge = wfs.get("best_single_edge")
    best_sr = wfs.get("best_single_sharpe", float("nan"))
    print("\n=== HEADLINE: IN-SAMPLE vs WALK-FORWARD vs BEST SINGLE BOOK ===")
    print(f"  {'metric':16}{'in-sample':>14}{'walk-forward':>16}{'best-single':>14}")
    rows = [
        ("Sharpe", iss.get("sharpe"), wfs.get("sharpe"), best_sr, "{:.2f}"),
        ("deflated Sharpe", iss.get("deflated_sharpe"), wfs.get("deflated_sharpe"), float("nan"), "{:.3f}"),
        ("maxDD", iss.get("max_drawdown"), wfs.get("max_drawdown"), float("nan"), "{:.1%}"),
        ("Calmar", iss.get("calmar"), wfs.get("calmar"), float("nan"), "{:.2f}"),
        ("CAGR", iss.get("cagr"), wfs.get("cagr"), float("nan"), "{:+.1%}"),
        ("vol (ann)", iss.get("vol_ann"), wfs.get("vol_ann"), float("nan"), "{:.1%}"),
    ]
    for lbl, a, b, c, fmt in rows:
        print(f"  {lbl:16}{_fmt(a, fmt):>14}{_fmt(b, fmt):>16}{_fmt(c, fmt):>14}")
    print(f"  {'best book =':16}{best_edge!s:>14}  (OOS stand-alone Sharpe {_fmt(best_sr)})")

    # --- the verdict ----------------------------------------------------------------------- #
    is_lift = iss.get("sharpe", float("nan")) - best_sr
    wf_lift = wfs.get("sharpe", float("nan")) - best_sr
    print("\n=== DIVERSIFICATION LIFT (ensemble Sharpe − best single book OOS Sharpe) ===")
    print(f"  in-sample lift   : {_fmt(is_lift, '{:+.3f}')}  Sharpe")
    print(f"  walk-forward lift: {_fmt(wf_lift, '{:+.3f}')}  Sharpe")
    if wf_lift == wf_lift:  # not NaN
        if wf_lift <= 0.05:
            verdict = ("VANISHES out-of-sample — the OOS ensemble does NOT beat just holding the "
                       "best single book. The in-sample lift was a weight-fitting artifact.")
        elif wf_lift < 0.5 * max(is_lift, 1e-9):
            verdict = ("SHRINKS sharply out-of-sample — a real but much smaller benefit than the "
                       "in-sample number; treat the in-sample lift as inflated.")
        else:
            verdict = "largely SURVIVES out-of-sample — the diversification benefit is genuine."
        print(f"\n  VERDICT: diversification benefit {verdict}")


if __name__ == "__main__":
    main()
