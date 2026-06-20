"""Warm the local cache (prices + catalysts) for the seed universe, and print a scan summary.

Usage:
    python scripts/backfill.py [START] [END]      # defaults: last ~2 years
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.events.engine import analyze_symbol
from startx.fmp.client import FMPClient
from startx.settings import get_settings


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=730)).isoformat()
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    settings = get_settings()
    settings.require_key()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()

    print(f"Backfilling {start} → {end} for: {', '.join(universe.tickers)}\n")
    with FMPClient(settings) as client:
        for ticker in universe.tickers:
            try:
                res = analyze_symbol(ticker, start, end, client=client, cache=cache,
                                     universe=universe, settings=settings)
                ev = res.events
                ups = int((ev["direction"] == "up").sum()) if not ev.empty else 0
                downs = int((ev["direction"] == "down").sum()) if not ev.empty else 0
                top = ""
                if not res.reliability.empty:
                    r0 = res.reliability.iloc[0]
                    top = f"  top catalyst: {r0['catalyst']} (avg move {r0['avg_abs_move_pct']}%)"
                print(f"  {ticker:5s} {len(ev):3d} events  (↑{ups} ↓{downs}){top}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {ticker:5s} FAILED: {exc}")
    print("\nDone. Launch the dashboard:  streamlit run src/startx/dashboard/app.py")


if __name__ == "__main__":
    main()
