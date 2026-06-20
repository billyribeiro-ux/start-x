"""Export every out-of-sample trade with FULL mechanics: entry, target, stop, exit, PnL.

Usage:
    python scripts/export_trades.py --tickers SPY,SPX --horizon long --start 2018-01-01 \
        --mode pooled --out trades.csv

Each trade carries its triple-barrier TARGET (profit-take) and STOP (loss) prices, the
reward/risk distances, and the exit reason (target_hit / stop_hit / time_exit). Fills are
daily close-to-close: entry at entry_date's close, exit at the barrier-touch date's close.
Returns are net of round-trip costs. This is the raw evidence blotter for manual review.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.backtest.engine import BacktestResult
from startx.data.cache import ParquetCache
from startx.data.prices import get_prices
from startx.data.universe import load_universe
from startx.fmp.client import FMPClient
from startx.labeling.config import label_horizon
from startx.settings import get_settings
from startx.strategy.pipeline import run_per_symbol, run_pooled


def _side_str(v) -> str:
    if isinstance(v, str):
        return v.lower()
    return "long" if v > 0 else "short"


def _exit_reason(touch: str, side: str) -> str:
    if touch == "vert":
        return "time_exit"
    if touch == "pt":  # upper barrier hit first
        return "target_hit" if side == "long" else "stop_hit"
    if touch == "sl":  # lower barrier hit first
        return "stop_hit" if side == "long" else "target_hit"
    return "unknown"


def _enrich(symbol: str, bt: BacktestResult, prices: pd.DataFrame, horizon: str) -> pd.DataFrame:
    t = bt.trades.copy()
    if t.empty:
        return t
    closes = prices.set_index("date")["close"].astype(float)
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t["symbol"] = symbol
    t["side"] = t["side"].map(_side_str)
    t["entry_price"] = closes.reindex(t["entry_date"]).to_numpy().round(2)
    t["exit_price"] = closes.reindex(t["exit_date"]).to_numpy().round(2)

    # Pull each trade's triple-barrier levels (target=upper, stop=lower for a long; flipped
    # for a short) and the touch reason, by re-joining the labels on the entry bar.
    labels = label_horizon(prices, horizon).rename(columns={"date": "entry_date"})
    labels["entry_date"] = pd.to_datetime(labels["entry_date"])
    t = t.merge(labels[["entry_date", "upper", "lower", "touch"]], on="entry_date", how="left")

    is_long = t["side"] == "long"
    t["target_price"] = np.where(is_long, t["upper"], t["lower"]).round(2)
    t["stop_price"] = np.where(is_long, t["lower"], t["upper"]).round(2)
    t["reward_pct"] = (np.abs(t["target_price"] / t["entry_price"] - 1.0) * 100).round(2)
    t["risk_pct"] = (np.abs(t["stop_price"] / t["entry_price"] - 1.0) * 100).round(2)
    t["rr"] = (t["reward_pct"] / t["risk_pct"].replace(0, np.nan)).round(2)
    t["exit_reason"] = [_exit_reason(tc, sd) for tc, sd in zip(t["touch"], t["side"])]

    t["price_move_pct"] = ((t["exit_price"] / t["entry_price"] - 1.0) * 100).round(2)
    t["net_return_pct"] = (t.get("ret_net", 0.0) * 100).round(3)
    t["pnl_usd_per_10k"] = (t.get("ret_net", 0.0) * 10_000).round(2)
    t["bars_held"] = t.get("holding_days", t.get("bars_held"))
    t["prob_up"] = t.get("prob_up").round(3)
    t["outcome"] = t.get("ret_net", 0.0).map(lambda r: "WIN" if r > 0 else "LOSS")

    cols = ["symbol", "side", "entry_date", "entry_price", "target_price", "stop_price",
            "reward_pct", "risk_pct", "rr", "exit_date", "exit_price", "exit_reason",
            "bars_held", "price_move_pct", "net_return_pct", "pnl_usd_per_10k", "prob_up",
            "outcome"]
    return t[cols].sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--horizon", default="long", choices=["short", "long"])
    ap.add_argument("--mode", default="pooled", choices=["pooled", "per_symbol"])
    ap.add_argument("--tickers", default="SPY,SPX")
    ap.add_argument("--long-th", type=float, default=0.6, dest="long_th")
    ap.add_argument("--short-th", type=float, default=0.4, dest="short_th")
    ap.add_argument("--out", default="trades.csv")
    args = ap.parse_args()

    settings = get_settings()
    settings.require_key()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()
    tickers = [t.strip() for t in args.tickers.split(",")]

    runner = run_pooled if args.mode == "pooled" else run_per_symbol
    with FMPClient(settings) as client:
        res = runner(tickers, args.start, args.end, args.horizon, client=client, cache=cache,
                     universe=universe, settings=settings, long_th=args.long_th,
                     short_th=args.short_th)
        frames = []
        for sym, bt in res.per_symbol.items():
            prices = get_prices(client, cache, universe.spec(sym).fmp, settings.history_start)
            frames.append(_enrich(sym, bt, prices, args.horizon))

    blotter = pd.concat(frames, ignore_index=True).sort_values(
        ["entry_date", "symbol"]).reset_index(drop=True)
    blotter.to_csv(args.out, index=False)
    n = len(blotter)
    wins = int((blotter["outcome"] == "WIN").sum())
    longs = int((blotter["side"] == "long").sum())
    reasons = blotter["exit_reason"].value_counts().to_dict()
    print(f"Wrote {n} trades -> {args.out}")
    print(f"  long={longs} short={n-longs}  wins={wins} ({wins/n:.1%})  "
          f"avg R:R={blotter['rr'].mean():.2f}  total_pnl_per_10k=${blotter['pnl_usd_per_10k'].sum():,.0f}")
    print(f"  exit reasons: {reasons}")
    print(blotter.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
