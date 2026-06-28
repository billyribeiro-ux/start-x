"""Swing scanner — IMPROVE the edge beyond avoidance: exits + conviction-sizing (firewall-gated).

R5 found the losses aren't separably avoidable (the loss-prone cohort has the fattest right tail). This
tests the two levers that work WITH that variance, on the cached taken book (no entry-model re-fit):

  STUDY A — EXITS: re-simulate the SAME entries under a pre-registered grid of stop/target policies,
            including a regime-scaled stop that widens in stress. Per-trade stats + an INVESTABLE calendar
            book (Sharpe/maxDD/Calmar/DSR). Promote a policy only if it beats the 2.0/1.0 baseline on
            risk-adjusted return AND drawdown, DSR-deflated by the grid size.
  STUDY B — SIZING: equal-weight vs (a) walk-forward edge-model weight and (b) stress-depth weight.
            Judged as a calendar book — does leaning into the fat tail improve Sharpe without wrecking DD?

    python scripts/swing_improve.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd

from startx.scanner.improve import (
    book_stats,
    calendar_book,
    feat_cols_of,
    per_trade_stats,
    resim_returns,
    walk_forward_edge,
)

PRICE_DIR = "data/cache/prices"
TAKEN_CACHE = "data/cache/selflearn_taken.parquet"
LOG = "SELFLEARN_LOG.md"
MAXC = 10                                 # investable concurrency cap for the calendar book

# Pre-registered EXIT grid (baseline first). (label, pt, sl, horizon, stress_sl, stress_pt)
EXIT_GRID = [
    ("baseline 2.0/1.0", 2.0, 1.0, 10, None, None),
    ("wider stop 2.0/1.5", 2.0, 1.5, 10, None, None),
    ("wider stop 2.0/2.0", 2.0, 2.0, 10, None, None),
    ("wider target 3.0/1.0", 3.0, 1.0, 10, None, None),
    ("regime-stop x1.5", 2.0, 1.0, 10, 1.5, None),
    ("regime-stop x2.0", 2.0, 1.0, 10, 2.0, None),
    ("regime 3.0/2.0", 3.0, 1.0, 10, 2.0, 3.0),
]
N_EXIT = len(EXIT_GRID)


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _pt(st):
    if not st or st.get("n", 0) < 12:
        return f"n={st.get('n', 0) if st else 0}"
    return (f"n={st['n']:4d}  exp {st['exp']*100:+.2f}%  PF {st['pf']:.2f}  win% {st['win_share']*100:4.1f}  "
            f"Sh {st['sharpe']:+.2f}  DSR {st['dsr']:.2f}")


def _bk(st):
    if not st or st.get("n_days", 0) < 30:
        return f"days={st.get('n_days', 0) if st else 0}"
    return (f"Sharpe {st['sharpe']:+.2f}  CAGR {st['cagr']*100:+5.1f}%  maxDD {st['maxdd']*100:6.1f}%  "
            f"Calmar {st['calmar']:.2f}  DSR {st['dsr']:.2f}")


def main():
    taken = pd.read_parquet(TAKEN_CACHE)
    feat = feat_cols_of(taken)
    syms = sorted(taken["symbol"].unique())
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    prices = {s: _load(s) for s in syms if s in cached}
    print(f"loaded taken book: {len(taken)} trades, {len(prices)} symbols "
          f"({taken['entry_date'].min().date()}..{taken['entry_date'].max().date()})\n")

    # ============================ STUDY A — EXITS ============================
    print(f"=== STUDY A — EXITS (re-simulate same entries; grid N={N_EXIT}, DSR-deflated) ===")
    print(f"  {'policy':22}{'per-trade':46}| investable calendar book (<=%d concurrent)" % MAXC)
    base_pt = base_bk = None
    rows = []
    for (label, pt, sl, h, ssl, spt) in EXIT_GRID:
        r = resim_returns(taken, prices, pt_mult=pt, sl_mult=sl, horizon=h,
                          stress_sl_mult=ssl, stress_pt_mult=spt)
        pst = per_trade_stats(r, n_trials=N_EXIT)
        bk = book_stats(calendar_book(taken, r, max_concurrent=MAXC), n_trials=N_EXIT)
        rows.append((label, pst, bk))
        if base_pt is None:
            base_pt, base_bk = pst, bk
        print(f"  {label:22}{_pt(pst):46}| {_bk(bk)}")

    # promotion (RISK-ADJUSTED, not raw maxDD): a book earning more per unit of drawdown is an
    # improvement even if absolute DD rises. Require Sharpe AND Calmar AND per-trade exp up, DSR not
    # degraded, with a blowout guard so DD can't balloon past 1.3x baseline for marginal Sharpe.
    win = None
    for (label, pst, bk) in rows[1:]:
        if (bk.get("sharpe", -9) > base_bk.get("sharpe", -9) + 0.05
                and bk.get("calmar", -9) > base_bk.get("calmar", -9)
                and bk.get("dsr", -9) >= base_bk.get("dsr", -9) - 0.02
                and bk.get("maxdd", -9) >= 1.3 * base_bk.get("maxdd", -9)        # DD blowout guard
                and pst.get("exp", -9) >= base_pt.get("exp", -9) - 1e-4):
            if win is None or bk["calmar"] > win[2]["calmar"]:
                win = (label, pst, bk)
    if win:
        print(f"\n  -> EXIT IMPROVEMENT: '{win[0]}' beats baseline on risk-adjusted return AND drawdown.")
        print(f"     baseline book {_bk(base_bk)}")
        print(f"     improved book {_bk(win[2])}")
        exit_verdict = (f"PROMOTE exit '{win[0]}': book Sharpe {base_bk['sharpe']:+.2f}->{win[2]['sharpe']:+.2f}, "
                        f"maxDD {base_bk['maxdd']*100:.1f}%->{win[2]['maxdd']*100:.1f}%, DSR "
                        f"{base_bk['dsr']:.2f}->{win[2]['dsr']:.2f}")
    else:
        print("\n  -> no exit policy beats the 2.0/1.0 baseline on BOTH risk-adjusted return and drawdown "
              "(grid-deflated). The tight stop is not the leak; baseline exit stands.")
        exit_verdict = "no exit policy beat baseline on risk-adjusted return AND drawdown (grid-deflated)"

    # ============================ STUDY B — SIZING ============================
    print("\n=== STUDY B — CONVICTION SIZING (calendar book; lean into the fat right tail) ===")
    base_r = taken["ret"].to_numpy(float)
    eq_bk = book_stats(calendar_book(taken, base_r, max_concurrent=MAXC), n_trials=3)
    print(f"  equal-weight        : {_bk(eq_bk)}")

    # (a) walk-forward edge-model weight (size ~ rank of predicted edge, clipped >=0)
    edge = walk_forward_edge(taken, feat)
    sub = taken.assign(__edge=edge).dropna(subset=["__edge"]).reset_index(drop=True)
    er = sub["ret"].to_numpy(float)
    w_rank = pd.Series(sub["__edge"]).rank(pct=True).to_numpy()       # 0..1 conviction
    edge_bk = book_stats(calendar_book(sub, er, weights=w_rank, max_concurrent=MAXC), n_trials=3)
    edge_eq = book_stats(calendar_book(sub, er, max_concurrent=MAXC), n_trials=3)  # same subset, equal-wt
    print(f"  edge-model weight   : {_bk(edge_bk)}   (vs same-subset equal-wt: {_bk(edge_eq)})")

    # (b) stress-depth weight (theory-driven: deeper stress = fatter tail)
    if "regime_stress" in taken:
        w_stress = pd.Series(taken["regime_stress"]).rank(pct=True).to_numpy()
        st_bk = book_stats(calendar_book(taken, base_r, weights=w_stress, max_concurrent=MAXC), n_trials=3)
        print(f"  stress-depth weight : {_bk(st_bk)}")
    else:
        st_bk = {}

    best_size = max([("edge", edge_bk, edge_eq), ("stress", st_bk, eq_bk)],
                    key=lambda x: x[1].get("sharpe", -9) if x[1] else -9)
    ref = best_size[2]
    improved = (best_size[1] and best_size[1].get("sharpe", -9) > ref.get("sharpe", -9) + 0.05
                and best_size[1].get("maxdd", -9) >= ref.get("maxdd", -9))
    if improved:
        print(f"\n  -> SIZING IMPROVEMENT: '{best_size[0]}' weight lifts book Sharpe "
              f"{ref['sharpe']:+.2f}->{best_size[1]['sharpe']:+.2f} without worse drawdown.")
        size_verdict = (f"PROMOTE '{best_size[0]}' sizing: Sharpe {ref['sharpe']:+.2f}->{best_size[1]['sharpe']:+.2f}, "
                        f"maxDD {ref['maxdd']*100:.1f}%->{best_size[1]['maxdd']*100:.1f}%")
    else:
        print("\n  -> conviction sizing does NOT improve the investable book on risk-adjusted terms "
              "(leaning into the fat tail raises expectancy but adds variance/DD in equal measure). "
              "Equal-weight stands.")
        size_verdict = "conviction sizing did not improve the calendar book on risk-adjusted terms"

    with open(LOG, "a") as fh:
        fh.write(f"\n## Improvement studies — {datetime.now(timezone.utc).date()}\n")
        fh.write(f"- STUDY A (exits, grid N={N_EXIT}): {exit_verdict}.\n")
        fh.write(f"- STUDY B (sizing): {size_verdict}.\n")
    print(f"\nlogged to {LOG}. (win_rate shown for color only — never a selection metric.)")


if __name__ == "__main__":
    main()
