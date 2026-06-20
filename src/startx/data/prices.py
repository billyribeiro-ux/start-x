"""Cached daily price history with returns, point-in-time safe."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..fmp.client import FMPClient
from ..fmp.endpoints import historical_price_eod
from .cache import ParquetCache


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    close = df["close"].astype(float)
    df["ret"] = close.pct_change()
    df["log_ret"] = np.log(close / close.shift(1))
    df["prev_close"] = close.shift(1)
    df["gap"] = df["open"].astype(float) / df["prev_close"] - 1.0
    return df


def get_prices(
    client: FMPClient,
    cache: ParquetCache,
    fmp_symbol: str,
    history_start: str = "2010-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Full cached daily history for ``fmp_symbol`` (enriched with returns/gap)."""
    today = date.today().isoformat()
    key = f"prices/{fmp_symbol}"
    raw = cache.get_or_fetch(
        key,
        lambda: historical_price_eod(client, fmp_symbol, start=history_start, end=today),
        refresh=refresh,
    )
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    return _enrich(raw)
