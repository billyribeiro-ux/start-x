"""Cached cross-asset (intermarket) price context.

The macro/cross-asset set below empirically leads broad equity indices (10y yield ^TNX,
homebuilders ITB/XHB, transports IYT, semis SMH, high-yield credit HYG, defensives XLP vs
discretionary XLY, etc.). We pull full daily history per symbol — cached the same way as
``startx.data.prices`` — and hand back a tidy ``date``/``close`` frame per name. The set is the
input to ``startx.features.intermarket.intermarket_features``.
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from ..fmp.client import FMPClient
from .cache import ParquetCache
from .prices import get_prices

#: Short name -> FMP symbol for the cross-asset leading-indicator set (insertion-ordered).
CONTEXT: "OrderedDict[str, str]" = OrderedDict(
    [
        ("tnx10", "^TNX"),   # 10y Treasury yield
        ("tyx30", "^TYX"),   # 30y Treasury yield
        ("fvx5", "^FVX"),    # 5y Treasury yield
        ("hyg", "HYG"),      # high-yield corporate credit
        ("lqd", "LQD"),      # investment-grade corporate credit
        ("itb", "ITB"),      # homebuilders
        ("xhb", "XHB"),      # homebuilders (alt)
        ("iyt", "IYT"),      # transports
        ("smh", "SMH"),      # semiconductors
        ("xlf", "XLF"),      # financials
        ("xly", "XLY"),      # consumer discretionary
        ("xlp", "XLP"),      # consumer staples (defensives)
        ("gld", "GLD"),      # gold
        ("uup", "UUP"),      # US dollar
        ("tlt", "TLT"),      # long-duration Treasuries
    ]
)


def get_context_prices(
    client: FMPClient,
    cache: ParquetCache,
    history_start: str = "2010-01-01",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Cached daily history for every symbol in :data:`CONTEXT`.

    Returns ``{short_name: DataFrame[date, close, ...]}``. Each frame is cached per symbol under
    ``prices/{symbol}`` (mirroring :func:`startx.data.prices.get_prices`). A symbol that returns
    no rows is simply omitted, so one missing/illiquid feed never sinks the whole context.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, symbol in CONTEXT.items():
        df = get_prices(client, cache, symbol, history_start=history_start, refresh=refresh)
        if df is None or df.empty:
            continue
        out[name] = df
    return out
