#!/usr/bin/env python
"""Cross-instrument replication of the three books (ROADMAP #4 — thin samples).

The three books are validated almost entirely on SPY (29–54 trades — thin). This script
re-runs each book on SPY, ^GSPC (``_GSPC``), QQQ and IWM to test whether each edge
REPLICATES across instruments or is SPY-specific luck.

Windows
-------
* **Locked window** (headline): 2019-01-01 → 2026-06-19 — the desk's locked study window.
* **Deep history** (extra sample size, clearly labelled): each instrument's own start
  (SPY 1993, _GSPC 1990, QQQ 1999, IWM 2000) → 2026-06-19.

IMPORTANT data caveat — the ``fear`` sleeve (System #2 long_swing) keys off VVIX, and the
cached ``_VVIX`` series only begins **2006-03**. Deep-history long_swing runs that start
before 2006 therefore have a ``breakout``-only fear-less front section; this is labelled
in the output, not hidden.

Run:  ``.venv/bin/python scripts/cross_instrument.py``
      add ``--deep`` to also print the deep-history matrix,
      add ``--json`` to dump the raw matrix as JSON.

Nothing is written to disk and no secrets are touched — read-only over the price cache.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from startx.portfolio import run_portfolio
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals
from startx.strategy.trend_position import position_book

# --------------------------------------------------------------------------------------
# Config — locked window and the instruments under test.
# --------------------------------------------------------------------------------------
LOCKED_START = "2019-01-01"
LOCKED_END = "2026-06-19"
VVIX_START = pd.Timestamp("2006-03-06")  # _VVIX cache floor — fear sleeve cannot fire before this

INSTRUMENTS = ["SPY", "_GSPC", "QQQ", "IWM"]
CACHE = Path("data/cache/prices")

# Book wiring straight from the task brief (LOCKED exits / horizons).
SHORT_SLEEVES = {"ibs": lambda s, a: ibs_signals(s)}
SHORT_EXITS = {"ibs": (1.0, 3.0, 10)}
LONG_SLEEVES = {
    "breakout": lambda s, a: breakout_signals(s, 20, 200),
    "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"]),
}
LONG_EXITS = {"breakout": (1.0, 3.0, 63), "fear": (1.0, 3.0, 63)}
GROSS_CAP = 1.5  # single-instrument book: one underlying, modest gross


# --------------------------------------------------------------------------------------
# Data loading.
# --------------------------------------------------------------------------------------
def load(symbol: str) -> pd.DataFrame:
    """Load a cached price frame, parse dates, sort ascending. Raises if not cached."""
    path = CACHE / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"price cache missing: {path}")
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


def cached_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split a symbol list into (present, missing) by cache presence."""
    present, missing = [], []
    for s in symbols:
        (present if (CACHE / f"{s}.parquet").exists() else missing).append(s)
    return present, missing


# --------------------------------------------------------------------------------------
# Per book × instrument runners. Each returns a flat metric dict.
# --------------------------------------------------------------------------------------
def _swing_row(und: pd.DataFrame, aux: dict, sleeves, exits, start, end) -> dict:
    res = run_portfolio(und, aux, sleeves, exits=exits, gross_cap=GROSS_CAP,
                        start=start, end=end)
    s = res.stats
    return {
        "n": s["n"],
        "win_pct": s["win_rate"],
        "total_return": s["total_return"],
        "profit_factor": s["profit_factor"],
        "max_drawdown": s["max_drawdown"],
        "ann_sharpe": s["ann_sharpe"],
        "per_sleeve": {k: {"n": v["n"], "win_pct": v["win_rate"], "pf": v["profit_factor"]}
                       for k, v in s["per_sleeve"].items()},
    }


def run_short(symbol: str, aux: dict, start: str, end: str) -> dict:
    und = load(symbol)
    und.attrs["symbol"] = symbol
    return _swing_row(und, aux, SHORT_SLEEVES, SHORT_EXITS, start, end)


def run_long(symbol: str, aux: dict, start: str, end: str) -> dict:
    und = load(symbol)
    und.attrs["symbol"] = symbol
    row = _swing_row(und, aux, LONG_SLEEVES, LONG_EXITS, start, end)
    # Flag fear-blind front-sections (deep history before VVIX exists).
    row["fear_blind"] = pd.Timestamp(start) < VVIX_START
    return row


def run_position(symbol: str, start: str, end: str) -> dict:
    und = load(symbol)
    und.attrs["symbol"] = symbol
    pb = position_book(und, start=start, end=end)
    st, bm = pb.stats, pb.benchmark
    return {
        "n": int(len(pb.ledger)),
        "win_pct": (float((pb.ledger["outcome"] == "WIN").mean())
                    if len(pb.ledger) else float("nan")),
        "total_return": st["total_return"],
        "profit_factor": float("nan"),  # position book is a regime curve, PF not meaningful
        "max_drawdown": st["max_drawdown"],
        "ann_sharpe": st["ann_sharpe"],
        "cagr": st["cagr"],
        "calmar": st["calmar"],
        "exposure": st.get("exposure", float("nan")),
        # buy-and-hold of THIS instrument over the same window:
        "bh_cagr": bm["cagr"],
        "bh_max_drawdown": bm["max_drawdown"],
        "bh_calmar": bm["calmar"],
        "bh_total_return": bm["total_return"],
        "bh_ann_sharpe": bm["ann_sharpe"],
    }


# --------------------------------------------------------------------------------------
# Matrix builder.
# --------------------------------------------------------------------------------------
@dataclass
class WindowSpec:
    label: str
    start: str
    end: str


def build_matrix(symbols: list[str], aux: dict, window: WindowSpec) -> dict:
    """Build {book: {symbol: metrics}} for one window."""
    out: dict[str, dict] = {"short_swing": {}, "long_swing": {}, "position": {}}
    for sym in symbols:
        # Each instrument may have a later inception than the requested start; clamp.
        sym_start = max(pd.Timestamp(window.start), load(sym)["date"].iloc[0])
        start = sym_start.strftime("%Y-%m-%d")
        out["short_swing"][sym] = run_short(sym, aux, start, window.end)
        out["long_swing"][sym] = run_long(sym, aux, start, window.end)
        out["position"][sym] = run_position(sym, start, window.end)
        out["short_swing"][sym]["eff_start"] = start
        out["long_swing"][sym]["eff_start"] = start
        out["position"][sym]["eff_start"] = start
    return out


# --------------------------------------------------------------------------------------
# Pretty printing.
# --------------------------------------------------------------------------------------
def _f(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   n/a"
    if pct:
        return f"{x * 100:6.1f}%"
    return f"{x:7.{nd}f}"


def print_swing(book_name: str, mat: dict, symbols: list[str]) -> None:
    print(f"\n=== {book_name} ===")
    print(f"{'inst':6s} {'n':>4s} {'win%':>7s} {'tot_ret':>9s} {'PF':>7s} "
          f"{'maxDD':>8s} {'sharpe':>8s}   notes")
    for sym in symbols:
        r = mat[sym]
        note = ""
        if r.get("fear_blind"):
            note = "front-section pre-2006 is breakout-only (VVIX n/a)"
        if r.get("eff_start") and r["eff_start"] != "1990-01-02":
            pass
        print(f"{sym:6s} {r['n']:>4d} {_f(r['win_pct'], pct=True)} "
              f"{_f(r['total_return'], pct=True):>9s} {_f(r['profit_factor'])} "
              f"{_f(r['max_drawdown'], pct=True):>8s} {_f(r['ann_sharpe']):>8s}   {note}")
    # per-sleeve breakdown for the long book (breakout vs fear)
    if any("per_sleeve" in mat[s] and len(mat[s]["per_sleeve"]) > 1 for s in symbols):
        print(f"  per-sleeve:")
        for sym in symbols:
            ps = mat[sym].get("per_sleeve", {})
            parts = [f"{k}(n={v['n']},win={_f(v['win_pct'], pct=True).strip()},"
                     f"PF={_f(v['pf']).strip()})" for k, v in ps.items()]
            print(f"    {sym:6s} " + "  ".join(parts))


def print_position(mat: dict, symbols: list[str]) -> None:
    print(f"\n=== position (vs that instrument's buy-and-hold) ===")
    print(f"{'inst':6s} {'n':>4s} {'CAGR':>7s} {'maxDD':>8s} {'Calmar':>7s} "
          f"{'sharpe':>7s} {'expo':>6s} | {'BH_CAGR':>8s} {'BH_maxDD':>9s} {'BH_Calmar':>9s}")
    for sym in symbols:
        r = mat[sym]
        print(f"{sym:6s} {r['n']:>4d} {_f(r['cagr'], pct=True)} "
              f"{_f(r['max_drawdown'], pct=True):>8s} {_f(r['calmar'])} "
              f"{_f(r['ann_sharpe']):>7s} {_f(r['exposure'], pct=True):>6s} | "
              f"{_f(r['bh_cagr'], pct=True):>8s} {_f(r['bh_max_drawdown'], pct=True):>9s} "
              f"{_f(r['bh_calmar']):>9s}")


def print_window(mat: dict, symbols: list[str], window: WindowSpec) -> None:
    eff = {b: {s: mat[b][s].get("eff_start") for s in symbols} for b in mat}
    print("\n" + "#" * 78)
    print(f"# {window.label}  (requested {window.start} -> {window.end})")
    # show any clamped starts
    clamps = {s: eff["short_swing"][s] for s in symbols
              if eff["short_swing"][s] != window.start}
    if clamps:
        print(f"# instrument inception clamps: " +
              ", ".join(f"{s}->{d}" for s, d in clamps.items()))
    print("#" * 78)
    print_swing("short_swing  (IBS<0.1 dip-buy, 10-day cap)", mat["short_swing"], symbols)
    print_swing("long_swing  (breakout 20/200 + VVIX fear, 63-day cap)",
                mat["long_swing"], symbols)
    print_position(mat["position"], symbols)


# --------------------------------------------------------------------------------------
# Public entry — returns the full nested matrix (used by tests too).
# --------------------------------------------------------------------------------------
def run(deep: bool = False) -> dict:
    present, missing = cached_symbols(INSTRUMENTS)
    if missing:
        print(f"WARNING: not cached, skipped (no fabrication): {missing}")
    for req in ["_VIX", "_VVIX", "GLD"]:
        if not (CACHE / f"{req}.parquet").exists():
            raise FileNotFoundError(f"required aux series not cached: {req}")

    aux = {"vix": load("_VIX"), "vvix": load("_VVIX"), "gld": load("GLD")}

    windows = [WindowSpec("LOCKED WINDOW (headline)", LOCKED_START, LOCKED_END)]
    if deep:
        windows.append(WindowSpec("DEEP HISTORY (extra sample, instrument inception -> now)",
                                  "1990-01-01", LOCKED_END))

    result: dict[str, dict] = {}
    for w in windows:
        mat = build_matrix(present, aux, w)
        result[w.label] = mat
        print_window(mat, present, w)
    result["_meta"] = {"present": present, "missing": missing,
                       "vvix_start": str(VVIX_START.date())}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", action="store_true",
                    help="also run the deep-history (instrument-inception) matrix")
    ap.add_argument("--json", action="store_true",
                    help="dump the raw matrix as JSON after the tables")
    args = ap.parse_args()
    result = run(deep=args.deep)
    if args.json:
        # strip nested per_sleeve dicts that hold numpy types
        print("\n--- JSON ---")
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
