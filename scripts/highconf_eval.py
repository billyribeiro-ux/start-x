"""High-conviction evaluation: train on full history, trade only prob_up >= threshold,
score on a recent window with an HONEST multiple-testing penalty.

Usage:
    python scripts/highconf_eval.py --prob 0.80 --eval-start 2023-01-01 --end 2026-06-19
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.data.cache import ParquetCache
from startx.data.prices import get_prices
from startx.data.universe import load_universe
from startx.fmp.client import FMPClient
from startx.labeling.config import label_horizon
from startx.settings import get_settings
from startx.strategy.pipeline import run_pooled
from startx.validation.metrics import max_drawdown, profit_factor, sharpe
from startx.validation.report import scoreboard


def _enrich(symbol, trades, prices, horizon):
    t = trades.copy()
    if t.empty:
        return t
    closes = prices.set_index("date")["close"].astype(float)
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t["symbol"] = symbol
    t["side"] = "long"
    t["entry_price"] = closes.reindex(t["entry_date"]).to_numpy().round(2)
    t["exit_price"] = closes.reindex(t["exit_date"]).to_numpy().round(2)
    lab = label_horizon(prices, horizon).rename(columns={"date": "entry_date"})
    lab["entry_date"] = pd.to_datetime(lab["entry_date"])
    t = t.merge(lab[["entry_date", "upper", "lower", "touch"]], on="entry_date", how="left")
    t["target_price"] = t["upper"].round(2)
    t["stop_price"] = t["lower"].round(2)
    t["reward_pct"] = (np.abs(t["target_price"] / t["entry_price"] - 1) * 100).round(2)
    t["risk_pct"] = (np.abs(t["stop_price"] / t["entry_price"] - 1) * 100).round(2)
    t["exit_reason"] = t["touch"].map({"pt": "target_hit", "sl": "stop_hit", "vert": "time_exit"})
    t["net_return_pct"] = (t.get("ret_net", 0.0) * 100).round(3)
    t["pnl_usd_per_10k"] = (t.get("ret_net", 0.0) * 10_000).round(2)
    t["prob_up"] = t.get("prob_up").round(3)
    t["bars_held"] = t.get("holding_days")
    t["outcome"] = t.get("ret_net", 0.0).map(lambda r: "WIN" if r > 0 else "LOSS")
    cols = ["symbol", "side", "entry_date", "entry_price", "target_price", "stop_price",
            "reward_pct", "risk_pct", "exit_date", "exit_price", "exit_reason", "bars_held",
            "net_return_pct", "pnl_usd_per_10k", "prob_up", "outcome"]
    return t[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,SPX")
    ap.add_argument("--train-start", default="2018-01-01")
    ap.add_argument("--eval-start", default="2023-01-01")
    ap.add_argument("--end", default="2026-06-19")
    ap.add_argument("--horizon", default="long")
    ap.add_argument("--prob", type=float, default=0.80)
    ap.add_argument("--n-trials", type=int, default=12, dest="n_trials",
                    help="honest count of configs tried, for the deflated-Sharpe penalty")
    ap.add_argument("--out", default="/home/user/start-x/spy_spx_highconf_trades.csv")
    args = ap.parse_args()

    settings = get_settings()
    settings.require_key()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()
    tickers = [t.strip() for t in args.tickers.split(",")]

    with FMPClient(settings) as client:
        res = run_pooled(tickers, args.train_start, args.end, args.horizon, client=client,
                         cache=cache, universe=universe, settings=settings,
                         long_th=args.prob, short_th=0.0)
        # Equity restricted to the eval window, renormalized.
        eq = res.portfolio.equity
        eq = eq[eq.index >= pd.Timestamp(args.eval_start)]
        eq = eq / eq.iloc[0]
        rets = eq.pct_change().dropna()
        sb = scoreboard(rets, n_trials=args.n_trials)

        frames, prices_by = [], {}
        for sym, bt in res.per_symbol.items():
            prices = get_prices(client, cache, universe.spec(sym).fmp, settings.history_start)
            prices_by[sym] = prices
            t = bt.trades.copy()
            t["entry_date"] = pd.to_datetime(t["entry_date"])
            t = t[t["entry_date"] >= pd.Timestamp(args.eval_start)]
            frames.append(_enrich(sym, t, prices, args.horizon))
        blotter = pd.concat(frames, ignore_index=True).sort_values("entry_date")
        blotter.to_csv(args.out, index=False)

        spy = prices_by.get("SPY", next(iter(prices_by.values())))
        bh = spy[(spy.date >= args.eval_start) & (spy.date <= args.end)]["close"]
        bh_ret = bh.iloc[-1] / bh.iloc[0] - 1
        bh_sharpe = bh.pct_change().mean() / bh.pct_change().std() * np.sqrt(252)

    n = len(blotter)
    wins = int((blotter["outcome"] == "WIN").sum()) if n else 0
    print(f"\n=== HIGH-CONVICTION (prob_up>={args.prob}) | trades {args.eval_start}->{args.end} ===")
    print(f"  trades={n}  wins={wins} ({wins/max(n,1):.1%})  total_pnl_per_10k="
          f"${blotter['pnl_usd_per_10k'].sum():,.0f}" if n else "  NO TRADES at this threshold")
    print(f"  strategy: Sharpe={sb['sharpe']:.2f}  CAGR={sb['cagr']:.2%}  "
          f"maxDD={sb['max_drawdown']:.1%}  PF={sb['profit_factor']:.2f}")
    print(f"  deflated Sharpe (centred, n_trials={args.n_trials})={sb['deflated_sharpe']:+.3f} "
          f"(prob={sb['deflated_sharpe_prob']:.1%})  VERDICT: {sb['verdict']}")
    print(f"  buy&hold SPY same window: total={bh_ret:.1%}  Sharpe={bh_sharpe:.2f}")
    print(f"  wrote {args.out}")
    if n:
        print(blotter.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
