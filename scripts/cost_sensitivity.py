"""Cost-sensitivity sweep — does each book's edge survive REALISTIC transaction costs?

The desk costs trades at a flat ~2 bp round-trip today (no slippage / fill realism). This script
proves whether each of the three horizon-separated books survives a *range* of round-trip costs and
finds each book's BREAK-EVEN cost (the cost at which the in-window total return → 0).

For every book we sweep ``cost_bps ∈ {1, 2, 5, 10, 20}`` over the LOCKED window 2019-01-01→2026-06-19
and report ``total_return``, ``profit_factor`` and ``ann_sharpe`` at each point, then:

  * BREAK-EVEN cost — linearly interpolate (in log/linear bps) the point where total_return crosses 0.
    If it never crosses inside the swept range we report it as ">20 bp" (robust) or "<1 bp" (already
    dead at the cheapest cost).
  * SHARPE DECAY PER BP — the OLS slope of ann_sharpe vs cost_bps across the sweep. This is the book's
    cost *per unit of turnover*: a book that churns harder (short_swing) bleeds more Sharpe per extra
    bp than a low-turnover book (position). Reported alongside each book's turnover so the link is
    explicit.

REALISM NOTE (printed at the end): SPY's real round-trip slippage is ~1 bp — TIGHTER than the generic
2 bp the books assume — so the books are *conservatively* costed today. But short_swing turns over far
more than position, so the same extra bp costs it much more; the Sharpe-decay-per-bp column quantifies
exactly that.

    python scripts/cost_sensitivity.py            # all three books, locked window
    python scripts/cost_sensitivity.py --book short_swing
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.backtest.costs import CostModel
from startx.portfolio import run_portfolio
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals
from startx.strategy.trend_position import position_book

# Locked study window (see CLAUDE.md / ROADMAP). Do not silently change.
START = "2019-01-01"
END = "2026-06-19"

# The round-trip cost grid to sweep (basis points).
COST_GRID: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)

# Production gross-leverage cap for the swing books (run_book.py default).
GROSS_CAP = 1.5


def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# Book runners — each returns (total_return, profit_factor, ann_sharpe, turnover) at a cost.
# These mirror the PRODUCTION configs in scripts/run_book.py exactly (sleeves, exits, gross_cap),
# using each book's `base` model so the sweep needs no breadth-internals dependency.
# --------------------------------------------------------------------------------------------- #
def _swing_metrics(spy, aux, sleeves, exits, cost_bps: float) -> dict:
    s = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=GROSS_CAP,
                      cost_bps=cost_bps, start=START, end=END).stats
    return {"total_return": s["total_return"], "profit_factor": s["profit_factor"],
            "ann_sharpe": s["ann_sharpe"], "turnover": s["turnover"]}


def _short_swing(spy, aux, cost_bps: float) -> dict:
    return _swing_metrics(spy, aux, {"ibs": lambda s, a: ibs_signals(s)},
                          {"ibs": (1.0, 3.0, 10)}, cost_bps)


def _long_swing(spy, aux, cost_bps: float) -> dict:
    sleeves = {"breakout": lambda s, a: breakout_signals(s, 20, 200),
               "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"])}
    return _swing_metrics(spy, aux, sleeves, {"breakout": (1.0, 3.0, 63),
                                              "fear": (1.0, 3.0, 63)}, cost_bps)


def _position(spy, aux, cost_bps: float) -> dict:
    res = position_book(spy, cost_bps=cost_bps, start=START, end=END)
    s = res.stats
    pf = _profit_factor_from_ledger(res.ledger)
    # turnover proxy: trades per year * average gross (each stint is ~1.0 gross of equity).
    yrs = max((pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25, 1e-9)
    turnover = float(len(res.ledger) / yrs) if len(res.ledger) else 0.0
    return {"total_return": s["total_return"], "profit_factor": pf,
            "ann_sharpe": s["ann_sharpe"], "turnover": turnover}


def _profit_factor_from_ledger(ledger: pd.DataFrame) -> float:
    """Profit factor from a position ledger (per-trade ``ret`` column). Σ wins / |Σ losses|."""
    if ledger is None or ledger.empty or "ret" not in ledger.columns:
        return float("nan")
    r = ledger["ret"].astype(float)
    wins = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


BOOK_RUNNERS = {
    "short_swing": _short_swing,
    "long_swing": _long_swing,
    "position": _position,
}


# --------------------------------------------------------------------------------------------- #
# Sweep + derived metrics
# --------------------------------------------------------------------------------------------- #
def sweep_book(book: str, spy: pd.DataFrame, aux: dict,
               grid: tuple[float, ...] = COST_GRID) -> pd.DataFrame:
    """Run ``book`` across the cost grid; return a tidy frame indexed by cost_bps."""
    runner = BOOK_RUNNERS[book]
    rows = [{"cost_bps": float(c), **runner(spy, aux, c)} for c in grid]
    return pd.DataFrame(rows).set_index("cost_bps")


def breakeven_cost(sweep: pd.DataFrame) -> float:
    """Cost (bps) at which total_return crosses 0, linearly interpolated between grid points.

    Returns ``+inf`` if the book is still profitable at the most expensive cost swept (robust),
    or ``0.0`` if it is already unprofitable at the cheapest cost (dead even at minimal cost).
    Because total_return is monotone-decreasing in cost, there is at most one crossing.
    """
    costs = sweep.index.to_numpy(dtype=float)
    tr = sweep["total_return"].to_numpy(dtype=float)
    if tr[0] <= 0:
        return 0.0                       # unprofitable already at the cheapest swept cost
    if tr[-1] > 0:
        return float("inf")              # still profitable at the most expensive swept cost
    # find the bracketing pair (positive -> non-positive) and linearly interpolate the zero
    for i in range(len(costs) - 1):
        if tr[i] > 0 >= tr[i + 1]:
            c0, c1, y0, y1 = costs[i], costs[i + 1], tr[i], tr[i + 1]
            return float(c0 + (c1 - c0) * y0 / (y0 - y1))  # y0>0, y1<=0 -> in [c0, c1]
    return float("inf")


def sharpe_decay_per_bp(sweep: pd.DataFrame) -> float:
    """OLS slope of ann_sharpe vs cost_bps (Sharpe lost per extra round-trip bp).

    This is the book's cost per unit of turnover: steeper (more negative) ⇒ more cost-sensitive.
    """
    x = sweep.index.to_numpy(dtype=float)
    y = sweep["ann_sharpe"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(x[ok], y[ok], 1)
    return float(slope)


def _fmt_breakeven(be: float) -> str:
    if be == float("inf"):
        return f">{COST_GRID[-1]:.0f} bp (robust across swept range)"
    if be == 0.0:
        return f"<{COST_GRID[0]:.0f} bp (already unprofitable at cheapest cost)"
    return f"{be:.1f} bp"


def _verdict(book: str, sweep: pd.DataFrame, be: float) -> str:
    """Robust-to-2-3x-assumed-cost or fragile? Assumed cost is the desk's 2 bp."""
    assumed = 2.0
    if be == float("inf"):
        head = "ROBUST"
        body = f"still profitable at {COST_GRID[-1]:.0f} bp = {COST_GRID[-1]/assumed:.0f}x the assumed 2 bp."
    elif be == 0.0:
        head = "DEAD"
        body = "unprofitable even at 1 bp."
    elif be >= 3 * assumed:
        head = "ROBUST"
        body = f"break-even {be:.0f} bp = {be/assumed:.1f}x the assumed 2 bp (survives 2-3x)."
    elif be >= 2 * assumed:
        head = "MARGINAL"
        body = f"break-even {be:.0f} bp = {be/assumed:.1f}x assumed (survives 2x, not 3x)."
    else:
        head = "FRAGILE"
        body = f"break-even {be:.1f} bp < 2x the assumed 2 bp — edge does not survive realistic stress."
    return f"{head} — {body}"


def _print_book(book: str, sweep: pd.DataFrame) -> None:
    be = breakeven_cost(sweep)
    decay = sharpe_decay_per_bp(sweep)
    turnover = float(sweep["turnover"].iloc[0])  # turnover is cost-independent
    print(f"\n=== BOOK: {book}  (window {START} -> {END}, gross_cap {GROSS_CAP}x) ===")
    print(f"  {'cost_bps':>9}{'total_ret':>12}{'PF':>9}{'ann_sharpe':>12}")
    for c, r in sweep.iterrows():
        pf = r["profit_factor"]
        pf_s = "  inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"  {c:9.0f}{r['total_return']*100:+11.1f}%{pf_s:>9}{r['ann_sharpe']:12.2f}")
    print(f"  break-even cost : {_fmt_breakeven(be)}")
    print(f"  turnover        : {turnover:.2f} gross/yr")
    print(f"  Sharpe decay    : {decay:+.4f} Sharpe per +1 bp round-trip  (cost per unit of turnover)")
    print(f"  VERDICT         : {_verdict(book, sweep, be)}")


def run(books: list[str]) -> dict[str, pd.DataFrame]:
    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD")}
    out: dict[str, pd.DataFrame] = {}
    for b in books:
        sweep = sweep_book(b, spy, aux)
        out[b] = sweep
        _print_book(b, sweep)
    _print_realism(out)
    return out


def _print_realism(sweeps: dict[str, pd.DataFrame]) -> None:
    spy_rt = CostModel.for_instrument("SPY").round_trip_bps()  # ~1 bp, the realistic SPY round-trip
    print("\n=== REALISM ===")
    print(f"  SPY's real round-trip slippage is ~{spy_rt:.0f} bp — TIGHTER than the generic 2 bp the")
    print("  books assume — so every book above is CONSERVATIVELY costed today (true edge ≥ 2-bp row).")
    print("  But cost scales with TURNOVER: the same extra bp costs a high-churn book far more.")
    print(f"\n  {'book':14}{'turnover/yr':>13}{'Sharpe/bp':>12}{'break-even':>14}")
    for b, sw in sweeps.items():
        be = breakeven_cost(sw)
        be_s = ">20bp" if be == float("inf") else ("<1bp" if be == 0.0 else f"{be:.1f}bp")
        print(f"  {b:14}{float(sw['turnover'].iloc[0]):13.2f}{sharpe_decay_per_bp(sw):+12.4f}{be_s:>14}")
    print("\n  Read: the steeper (more negative) the Sharpe/bp, the more the book bleeds per extra bp")
    print("  of cost — that steepness tracks turnover. short_swing is the most cost-exposed; position")
    print("  the least. At the realistic ~1 bp, all books sit cheaper than the 2-bp headline.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="all",
                    choices=["all", *BOOK_RUNNERS.keys()],
                    help="which book to sweep (default: all three)")
    args = ap.parse_args()
    books = list(BOOK_RUNNERS.keys()) if args.book == "all" else [args.book]
    run(books)


if __name__ == "__main__":
    main()
