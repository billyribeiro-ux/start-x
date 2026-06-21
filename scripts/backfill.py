"""Warm the local cache (prices + catalysts) for the seed universe, and print a scan summary.

Usage:
    python scripts/backfill.py [START] [END]      # defaults: last ~2 years
    python scripts/backfill.py --no-breadth       # skip the ~500-symbol S&P 500 breadth warm
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from startx.data.cache import ParquetCache
from startx.data.prices import get_prices
from startx.data.universe import load_universe
from startx.events.engine import analyze_symbol
from startx.fmp.client import FMPClient
from startx.settings import get_settings


def _warm_breadth(client, cache, settings) -> None:
    """Warm the S&P 500 constituent prices and build the market-internals breadth cache.

    The guarded model's IBS sleeve needs ``data/cache/internals.parquet`` (via
    ``market_internals.load_internals``), which ``compute_internals`` builds by reading each current
    constituent's cached price file — it silently skips any that are missing, so on a FRESH checkout
    (where only the 7 seed symbols are warmed) breadth would be empty. This pulls the constituent
    panel and rebuilds the cache so a clean clone runs the guarded book correctly. ~500 symbols; pass
    ``--no-breadth`` to skip.
    """
    from startx.strategy.market_internals import _constituents, load_internals

    members = _constituents(client)
    syms = list(members["symbol"])
    print(f"\nBreadth: warming {len(syms)} S&P 500 constituents (pass --no-breadth to skip)...")
    ok = 0
    for i, sym in enumerate(syms, 1):
        try:
            get_prices(client, cache, sym, settings.history_start)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
        if i % 100 == 0:
            print(f"  ...{i}/{len(syms)} warmed")
    internals = load_internals(refresh=True)  # build data/cache/internals.parquet from the panel
    print(f"  warmed {ok}/{len(syms)} constituents; internals cache built ({len(internals)} days)")


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    warm_breadth = "--no-breadth" not in flags
    start = pos[0] if len(pos) > 0 else (date.today() - timedelta(days=730)).isoformat()
    end = pos[1] if len(pos) > 1 else date.today().isoformat()

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

        # Context series (not traded, no catalysts) — warm the raw price history so a fresh cache
        # has them. get_prices writes key "prices/<fmp>", which the cache sanitizes to
        # data/cache/prices/<fmp-with-^->_>.parquet — the exact filename run_book.py reads
        # (^VIX -> _VIX, ^VVIX -> _VVIX, GLD -> GLD).
        if universe.context:
            print(f"\nContext series: {', '.join(universe.context)}")
            for name, fmp in universe.context.items():
                try:
                    px = get_prices(client, cache, fmp, settings.history_start)
                    print(f"  {name:5s} ({fmp}) {len(px):4d} bars")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:5s} ({fmp}) FAILED: {exc}")

        if warm_breadth:
            try:
                _warm_breadth(client, cache, settings)
            except Exception as exc:  # noqa: BLE001
                print(f"\nBreadth warm FAILED: {exc}  (run again or pass --no-breadth)")
    print("\nDone. Launch the dashboard:  streamlit run src/startx/dashboard/app.py")


if __name__ == "__main__":
    main()
