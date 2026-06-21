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


def _last_expected_trading_day(end: pd.Timestamp) -> pd.Timestamp:
    """Last business day at or before ``end``.

    Coverage staleness must be measured against the most recent *trading* day, not the
    raw calendar ``today``: on a weekend, holiday, or before the session has produced a
    bar, an otherwise up-to-date cache (whose last row is the prior trading day) would
    look stale against ``today`` and trigger a pointless re-fetch. ``bdate_range`` (the
    project's business-day convention, used in backtest/engine.py and forward/paper.py)
    skips Sat/Sun; it does not encode market holidays, so this is a conservative lower
    bound — a cache that already reaches this day is treated as fresh-tailed.
    """
    days = pd.bdate_range(end=end, periods=1)
    return days[-1] if len(days) else pd.Timestamp(end).normalize()


def get_prices(
    client: FMPClient,
    cache: ParquetCache,
    fmp_symbol: str,
    history_start: str = "2010-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Full cached daily history for ``fmp_symbol`` (enriched with returns/gap)."""
    today = date.today().isoformat()
    # Compare cache tail against the last expected *trading* day (<= today), never the
    # raw calendar date — otherwise weekends/holidays/pre-market would falsely read as
    # a stale tail and force a re-fetch.
    coverage_end = _last_expected_trading_day(pd.Timestamp(today)).isoformat()
    key = f"prices/{fmp_symbol}"
    raw = cache.get_or_fetch(
        key,
        lambda: historical_price_eod(client, fmp_symbol, start=history_start, end=today),
        refresh=refresh,
        coverage_start=history_start,
        coverage_end=coverage_end,
    )
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    return _enrich(raw)
