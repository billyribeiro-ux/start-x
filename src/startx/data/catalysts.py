"""Standalone catalyst loader for the feature layer.

Mirrors ``startx.events.engine._load_catalysts`` but is a reusable public function so the
feature pipeline does not depend on the events engine. Each feed is cached under the same keys
the engine uses (``catalysts/<fmp>/<feed>``) so the two share a warm cache. Company feeds
(earnings/grades/price-target/insider/senate/house) only make sense for single stocks; for an
index/ETF they degrade to empty frames, while news and macro are always loaded.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..fmp import endpoints as ep
from ..fmp.client import FMPClient
from .cache import ParquetCache
from .universe import SymbolSpec

# Company feeds are keyed per-symbol with no date range (the endpoints return a fixed history),
# matching the engine. News/macro are keyed with the date range because they take from/to.
_COMPANY_FEEDS = ("earnings", "grades", "price_target", "insider", "senate", "house")
_ALL_KEYS = (*_COMPANY_FEEDS, "news", "macro")


def load_catalysts(
    client: FMPClient,
    cache: ParquetCache,
    spec: SymbolSpec,
    start: str,
    end: str,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every catalyst feed for ``spec`` over ``[start, end]`` (padded), each cached.

    Returns a dict with keys: earnings, grades, price_target, insider, senate, house, news,
    macro. Non-stock symbols get empty company feeds but still get news + macro. The returned
    frames carry a parsed ``ts`` column (the public/PIT timestamp) per the endpoint wrappers.
    """
    fmp = spec.fmp
    pad_start = (pd.Timestamp(start) - timedelta(days=7)).date().isoformat()
    pad_end = (pd.Timestamp(end) + timedelta(days=1)).date().isoformat()
    cats: dict[str, pd.DataFrame] = {}

    if spec.is_stock:
        cats["earnings"] = cache.get_or_fetch(
            f"catalysts/{fmp}/earnings", lambda: ep.earnings(client, fmp), refresh)
        cats["grades"] = cache.get_or_fetch(
            f"catalysts/{fmp}/grades", lambda: ep.analyst_grades(client, fmp), refresh)
        cats["price_target"] = cache.get_or_fetch(
            f"catalysts/{fmp}/price_target", lambda: ep.price_target_news(client, fmp), refresh)
        cats["insider"] = cache.get_or_fetch(
            f"catalysts/{fmp}/insider", lambda: ep.insider_trades(client, fmp), refresh)
        cats["senate"] = cache.get_or_fetch(
            f"catalysts/{fmp}/senate", lambda: ep.senate_trades(client, fmp), refresh)
        cats["house"] = cache.get_or_fetch(
            f"catalysts/{fmp}/house", lambda: ep.house_trades(client, fmp), refresh)
    else:
        for key in _COMPANY_FEEDS:
            cats[key] = pd.DataFrame()

    cats["news"] = cache.get_or_fetch(
        f"catalysts/{fmp}/news_{start}_{end}",
        lambda: ep.stock_news(client, fmp, start=pad_start, end=pad_end), refresh)
    cats["macro"] = cache.get_or_fetch(
        f"catalysts/macro/econ_{start}_{end}",
        lambda: ep.economic_calendar(client, pad_start, pad_end), refresh)

    # Guarantee every key is present so downstream feature code never KeyErrors.
    for key in _ALL_KEYS:
        cats.setdefault(key, pd.DataFrame())
    return cats
