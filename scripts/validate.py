"""Run the full out-of-sample strategy validation on the seed universe.

Usage:
    python scripts/validate.py [--start 2015-01-01] [--end 2026-06-01]
                               [--horizon short|long] [--mode pooled|per_symbol|both]

Prints, for each model, the out-of-sample scoreboard (Sharpe, deflated Sharpe, CAGR, max
drawdown, profit factor, hit-rate, verdict) and per-symbol backtest metrics. Everything is
walk-forward out-of-sample — no in-sample numbers.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.fmp.client import FMPClient
from startx.settings import get_settings
from startx.strategy.pipeline import StrategyResult, run_per_symbol, run_pooled


def _print(res: StrategyResult) -> None:
    sb, pm = res.scoreboard, res.portfolio.metrics
    print(f"\n{'='*72}\n  MODEL: {res.label.upper()}   "
          f"(OOS samples={res.n_oos}, OOS AUC={res.oos_auc:.3f})\n{'='*72}")
    print(f"  VERDICT: {sb.get('verdict')}")
    print(f"  Sharpe={sb.get('sharpe'):.2f}  DeflatedSharpe(centred)={sb.get('deflated_sharpe'):+.3f}"
          f"  (prob={sb.get('deflated_sharpe_prob'):.2%})")
    print(f"  CAGR={pm.get('cagr'):.2%}  MaxDD={pm.get('max_drawdown'):.2%}  "
          f"ProfitFactor={pm.get('profit_factor'):.2f}  HitRate={pm.get('hit_rate'):.2%}")
    print(f"  Trades={pm.get('n_trades')}  Exposure={pm.get('exposure'):.1%}  "
          f"AvgHold={pm.get('avg_holding_days'):.1f}d  TotalRet={pm.get('total_return'):.2%}")
    print("  per-symbol:")
    for sym, bt in res.per_symbol.items():
        m = bt.metrics
        print(f"    {sym:5s} trades={m['n_trades']:4d}  hit={m['hit_rate']:.0%}  "
              f"CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:+.2f}  "
              f"PF={m['profit_factor']:.2f}  totalRet={m['total_return']:+.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--horizon", default="short", choices=["short", "long"])
    ap.add_argument("--mode", default="both", choices=["pooled", "per_symbol", "both"])
    ap.add_argument("--long-th", type=float, default=0.6, dest="long_th")
    ap.add_argument("--short-th", type=float, default=0.4, dest="short_th")
    args = ap.parse_args()

    settings = get_settings()
    settings.require_key()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()
    tickers = universe.tickers

    print(f"Validating {tickers}\n  window {args.start}→{args.end}  horizon={args.horizon}")
    with FMPClient(settings) as client:
        kw = dict(client=client, cache=cache, universe=universe, settings=settings,
                  long_th=args.long_th, short_th=args.short_th)
        if args.mode in ("pooled", "both"):
            _print(run_pooled(tickers, args.start, args.end, args.horizon, **kw))
        if args.mode in ("per_symbol", "both"):
            _print(run_per_symbol(tickers, args.start, args.end, args.horizon, **kw))
    print("\nReminder: the headline is OOS robustness. A 'LIKELY OVERFIT' verdict or a "
          "deflated Sharpe < 0 means the edge did NOT survive — that is the system working.")


if __name__ == "__main__":
    main()
