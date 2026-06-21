"""Walk-forward PARAMETER selection on the three production books (ROADMAP #3).

The books' parameters were picked on the FULL 2019->2026 sample, so they may be quietly
in-sample. This harness re-picks each book's parameters *as you go* using only the past
(López-de-Prado anchored/rolling walk-forward), stitches the unseen TEST blocks into one
out-of-sample return series, and reports its annualised Sharpe + a DEFLATED Sharpe that is
penalised for the number of configs the search tried. The verdict per book: does the edge
SURVIVE honest out-of-sample selection, or was the headline number in-sample?

Parameter sweeps (per ROADMAP #3):
  * short_swing (ibs)    : IBS threshold {0.05,0.10,0.15} x max-hold {5,10,15}
  * long_swing (breakout+fear): max-hold {40,63,90} x chand_mult {2.5,3.0,3.5}
  * position             : sma_len {150,200,250} x band {0.01,0.02,0.03}

Reference (production) configs being judged: ibs thr=0.10/hold=10; long hold=63/chand=3.0;
position sma=200/band=0.03.

    python scripts/walkforward_params.py                       # all three books
    python scripts/walkforward_params.py --book short_swing
    python scripts/walkforward_params.py --train 504 --test 126 --anchored

Window is the LOCKED span 2019-01-01 -> last available bar (data ends 2026-06-18).
"""
from __future__ import annotations

import argparse
import itertools
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.portfolio import run_portfolio
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.trend_position import position_book
from startx.strategy.volatility_premium import fear_signals
from startx.validation.wf_params import (
    BookSpec,
    WalkForwardResult,
    pick_stability,
    walk_forward_select,
)

START = "2019-01-01"
END = "2026-06-19"  # inclusive; SPY data ends 2026-06-18, so this just clamps to the last bar
GROSS_CAP = 1.5


# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


def _daily_returns(equity: pd.Series, start: str, end: str) -> pd.Series:
    """In-window daily returns from a full-span equity curve.

    ``run_portfolio``/``position_book`` return equity over the WHOLE data index (flat at 1.0
    before ``start``). We restrict to [start, end] then diff to returns; a day's realised
    return is the same regardless of how we later slice it (the engine is point-in-time and
    window-restricted), so these returns are safe to re-slice into TRAIN/TEST blocks.
    """
    eq = equity[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
    return eq.pct_change().fillna(0.0)


# --------------------------------------------------------------------------------------- #
# Book specs (grid + per-config runner + production reference)
# --------------------------------------------------------------------------------------- #
def _short_swing_spec(spy, aux) -> BookSpec:
    grid = [{"ibs_threshold": t, "max_hold": h}
            for t, h in itertools.product([0.05, 0.10, 0.15], [5, 10, 15])]

    def run(params):
        thr, hold = params["ibs_threshold"], params["max_hold"]
        res = run_portfolio(
            spy, aux, {"ibs": lambda s, a, thr=thr: ibs_signals(s, thr)},
            exits={"ibs": (1.0, 3.0, hold)}, gross_cap=GROSS_CAP, start=START, end=END,
        )
        return _daily_returns(res.equity, START, END)

    return BookSpec("short_swing", grid, run, {"ibs_threshold": 0.10, "max_hold": 10})


def _long_swing_spec(spy, aux) -> BookSpec:
    grid = [{"max_hold": h, "chand_mult": c}
            for h, c in itertools.product([40, 63, 90], [2.5, 3.0, 3.5])]
    sleeves = {
        "breakout": lambda s, a: breakout_signals(s, 20, 200),
        "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"]),
    }

    def run(params):
        hold, chand = params["max_hold"], params["chand_mult"]
        exits = {"breakout": (1.0, chand, hold), "fear": (1.0, chand, hold)}
        res = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=GROSS_CAP,
                            start=START, end=END)
        return _daily_returns(res.equity, START, END)

    return BookSpec("long_swing", grid, run, {"max_hold": 63, "chand_mult": 3.0})


def _position_spec(spy) -> BookSpec:
    grid = [{"sma_len": s, "band": b}
            for s, b in itertools.product([150, 200, 250], [0.01, 0.02, 0.03])]

    def run(params):
        res = position_book(spy, sma_len=params["sma_len"], band=params["band"],
                            start=START, end=END)
        # position_book equity is already rebased & in-window; diff to daily returns.
        return res.equity.pct_change().fillna(0.0)

    return BookSpec("position", grid, run, {"sma_len": 200, "band": 0.03})


def build_specs(books: list[str]) -> dict[str, BookSpec]:
    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    specs: dict[str, BookSpec] = {}
    aux = None
    if "short_swing" in books or "long_swing" in books:
        aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD")}
    if "short_swing" in books:
        specs["short_swing"] = _short_swing_spec(spy, aux)
    if "long_swing" in books:
        specs["long_swing"] = _long_swing_spec(spy, aux)
    if "position" in books:
        specs["position"] = _position_spec(spy)
    return specs


# --------------------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------------------- #
def _verdict(r: WalkForwardResult) -> str:
    """Brutally-honest one-liner: does the edge survive walk-forward selection?"""
    if r.oos_sharpe >= 0.5 and r.oos_dsr >= 0.90 and r.oos_sharpe >= 0.6 * r.is_sharpe:
        return "SURVIVES — OOS Sharpe holds and deflates clean; the edge is not just in-sample."
    if r.oos_sharpe <= 0.0 or r.oos_dsr < 0.5:
        return "COLLAPSES — out-of-sample selection erases the edge; the headline was in-sample."
    return ("FRAGILE — survives with a haircut (OOS Sharpe below in-sample and/or DSR shy of "
            "0.90); real but small / regime-dependent, not bankable alone.")


def report(name: str, r: WalkForwardResult) -> None:
    print(f"\n{'='*78}\nBOOK: {name}   ({r.n_configs} configs swept, {r.n_folds} walk-forward folds)")
    print(f"{'-'*78}")
    print(f"  {'':22}{'Sharpe':>10}{'DSR':>10}{'maxDD':>10}")
    print(f"  {'in-sample (full span)':22}{r.is_sharpe:>10.2f}{r.is_dsr:>10.2f}{'':>10}")
    print(f"  {'walk-forward OOS':22}{r.oos_sharpe:>10.2f}{r.oos_dsr:>10.2f}"
          f"{r.oos_max_drawdown*100:>9.1f}%")
    haircut = (r.oos_sharpe - r.is_sharpe)
    print(f"  OOS - IS Sharpe gap: {haircut:+.2f}   (DSR penalised for {r.n_configs} trials)")

    stab = pick_stability(r.picks)
    print(f"\n  PARAMETER PICKS over {stab['n_folds']} folds  "
          f"(config switched {stab['n_switches']}x fold-to-fold "
          f"-> {'STABLE' if stab['n_switches'] <= max(1, r.n_folds//3) else 'THRASHING'}):")
    for pname, info in stab["params"].items():
        vals = ", ".join(f"{k}x{v}" for k, v in info["values"].items())
        print(f"    {pname:14} modal={info['modal']} ({info['modal_frac']*100:.0f}% of folds)"
              f"   picks: {vals}")

    print(f"\n  VERDICT: {_verdict(r)}")


# --------------------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all",
                    choices=["all", "short_swing", "long_swing", "position"])
    ap.add_argument("--train", type=int, default=504, help="TRAIN window (trading days, default 2y)")
    ap.add_argument("--test", type=int, default=126, help="TEST block (trading days, default ~6mo)")
    ap.add_argument("--anchored", action="store_true",
                    help="expanding TRAIN (default: rolling fixed-length TRAIN)")
    ap.add_argument("--out", default=None, help="write each book's fold picks to {book}_<out>")
    args = ap.parse_args()

    books = ["short_swing", "long_swing", "position"] if args.book == "all" else [args.book]
    print(f"WALK-FORWARD PARAMETER SELECTION  —  window {START} -> {END}")
    print(f"  TRAIN={args.train}d  TEST={args.test}d  scheme={'anchored' if args.anchored else 'rolling'}"
          f"  objective=Sharpe  (gross_cap {GROSS_CAP}x)")

    specs = build_specs(books)
    results: dict[str, WalkForwardResult] = {}
    for name in books:
        r = walk_forward_select(specs[name], train=args.train, test=args.test,
                                anchored=args.anchored)
        results[name] = r
        report(name, r)
        if args.out:
            path = f"{name}_{args.out}"
            r.picks.to_csv(path, index=False)
            print(f"  wrote fold picks -> {path}")

    # summary table across books
    print(f"\n{'='*78}\nSUMMARY  (is = in-sample full span, oos = walk-forward)")
    print(f"  {'book':14}{'is_Sharpe':>11}{'oos_Sharpe':>12}{'oos_DSR':>10}{'oos_maxDD':>11}")
    for name, r in results.items():
        print(f"  {name:14}{r.is_sharpe:>11.2f}{r.oos_sharpe:>12.2f}{r.oos_dsr:>10.2f}"
              f"{r.oos_max_drawdown*100:>10.1f}%")


if __name__ == "__main__":
    main()
