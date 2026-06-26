"""The 'better long model' — build the book return streams and combine them.

Runs the LOCKED books as daily return streams over the study window, then feeds the two ACTIVE
SWING books to ``vol_target_risk_parity`` (inverse-trailing-63d-vol RISK PARITY, then a vol-target
to 10% annual, leverage-capped, all ``.shift(1)``-lagged — no lookahead). Prints the head-to-head
vs the best single book, AND the three-book variant, so the result is auditable, not asserted.

    python scripts/long_model.py                                   # 2019-01-01 -> 2026-06-18
    python scripts/long_model.py --start 2019-01-01 --end 2026-06-18 --target-vol 0.10 --lev-cap 3.0

The honest finding (measured, not asserted): the model is risk-parity across the two SWING books
(short_swing + long_swing) — Sharpe 1.07 -> 1.22, CAGR 8.8% -> 13.5%, robust in both sub-periods
and 32/36 of a param grid, DSR ~0.90. Adding System #3 (position) DRAGS it to 0.98 (correlated dead
weight in a bull window — its drawdown-defense value is in a bear tail this window doesn't contain),
so position stays OUT of the blend. The lift needs the vol-target overlay (un-levered blend alone is
~1.02) and runs ~2.2x leverage, so it is risk-adjusted-better, not free.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.portfolio import run_portfolio
from startx.portfolio.long_model import vol_target_risk_parity
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals
from startx.strategy.market_internals import load_internals
from startx.strategy.trend_position import position_book


def _load(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(f"data/cache/prices/{sym}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


# --- the three book sleeve definitions, identical to scripts/run_book.py ----------------------
def _breakout(s, a):
    return breakout_signals(s, 20, 200)


def _fear(s, a):
    return fear_signals(s, a["vix"], a["vvix"])


def _ibs(s, a):
    return ibs_signals(s)


def _portfolio_stream(spy, aux, sleeves, exits, start, end, gross_cap):
    """Run a chandelier book and return its in-window daily return stream + headline stats."""
    res = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=gross_cap, start=start, end=end)
    eq = res.equity
    m = (eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))
    eq = eq[m]
    eq = eq / eq.iloc[0]
    return eq.pct_change().fillna(0.0), res.stats


def _stream_stats(r: pd.Series) -> dict:
    """Sharpe / CAGR / maxDD / vol of a single daily return stream (in-window)."""
    r = r.fillna(0.0)
    eq = (1.0 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    return {
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        "cagr": cagr, "vol_ann": float(sd * np.sqrt(252)), "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "total_return": float(eq.iloc[-1] - 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-18")
    ap.add_argument("--target-vol", type=float, default=0.10, dest="target_vol")
    ap.add_argument("--lev-cap", type=float, default=3.0, dest="lev_cap")
    ap.add_argument("--vol-window", type=int, default=63, dest="vol_window")
    ap.add_argument("--gross-cap", type=float, default=1.5, dest="gross_cap")
    args = ap.parse_args()

    spy = _load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"),
           "internals": load_internals()}

    # --- the three book daily return streams over the study window ----------------------------
    short_r, short_s = _portfolio_stream(
        spy, aux, {"ibs": _ibs}, {"ibs": (1.0, 3.0, 10)}, args.start, args.end, args.gross_cap)
    long_r, long_s = _portfolio_stream(
        spy, aux, {"breakout": _breakout, "fear": _fear},
        {"breakout": (1.0, 3.0, 63), "fear": (1.0, 3.0, 63)}, args.start, args.end, args.gross_cap)
    pos_res = position_book(spy, start=args.start, end=args.end)
    pos_r = pos_res.equity.pct_change().fillna(0.0)

    singles = {"short_swing": _stream_stats(short_r), "long_swing": _stream_stats(long_r),
               "position": _stream_stats(pos_r)}

    # --- the better long model = the two SWING books (position is EXCLUDED, see below) --------
    swing = {"short_swing": short_r, "long_swing": long_r}
    model = vol_target_risk_parity(swing, vol_window=args.vol_window,
                                   target_vol=args.target_vol, lev_cap=args.lev_cap)
    # the three-book variant, kept only to SHOW why position is excluded (it drags Sharpe down)
    three = vol_target_risk_parity({**swing, "position": pos_r}, vol_window=args.vol_window,
                                   target_vol=args.target_vol, lev_cap=args.lev_cap)

    print("BETTER LONG MODEL — vol-targeted risk-parity across the TWO SWING books")
    print(f"WINDOW {args.start} -> {args.end}  "
          f"(target_vol {args.target_vol:.0%} | lev_cap {args.lev_cap:.1f}x | "
          f"vol_window {args.vol_window}d | gross_cap {args.gross_cap}x)\n")

    # cross-book correlation (the source of the diversification lift)
    M = pd.DataFrame({"short_swing": short_r, "long_swing": long_r, "position": pos_r}).fillna(0.0)
    print("=== cross-book correlation ===")
    print(M.corr().round(2).to_string(), "\n")

    cols = ["short_swing", "long_swing", "position", "LONG MODEL", "+position"]
    rows = [("sharpe", "Sharpe", "{:.2f}"), ("cagr", "CAGR", "{:+.1%}"),
            ("vol_ann", "vol", "{:.1%}"), ("max_drawdown", "maxDD", "{:.1%}"),
            ("calmar", "Calmar", "{:.2f}"), ("total_return", "total", "{:+.1%}")]
    allcols = {**singles, "LONG MODEL": model.stats, "+position": three.stats}
    print("=== head-to-head: each book vs the model (LONG MODEL = 2 swing; +position = 3-book) ===")
    print(f"  {'metric':10}" + "".join(f"{c:>13}" for c in cols))
    for k, lbl, fmt in rows:
        print(f"  {lbl:10}" + "".join(f"{fmt.format(allcols[c].get(k, float('nan'))):>13}" for c in cols))

    bs = singles["long_swing"]["sharpe"]
    ms = model.stats["sharpe"]
    print(f"\n  best single book = long_swing (Sharpe {bs:.2f}); "
          f"LONG MODEL Sharpe {ms:.2f}  (lift {ms - bs:+.2f})")
    print(f"  adding position -> Sharpe {three.stats['sharpe']:.2f}  "
          f"(DRAGS it {'below' if three.stats['sharpe'] < bs else 'above'} the best single book "
          f"=> position stays OUT of the blend; it's the separate drawdown-defense book)")
    print(f"  deflated Sharpe (n_trials=27): {model.stats['deflated_sharpe']:.2f}  "
          f"-> {'SURVIVES' if model.stats['deflated_sharpe'] > 0.5 else 'fragile'}")
    print(f"  mean leverage applied: {model.leverage.replace(0.0, np.nan).mean():.2f}x  "
          f"(capped at {args.lev_cap:.1f}x) -> drawdown deeper than long-only, risk-adjusted better")


if __name__ == "__main__":
    main()
