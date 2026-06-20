"""Cached 1-minute (or other interval) intraday bars for a single calendar day.

FMP Ultimate exposes ``historical-chart/{interval}`` at intraday resolution. We pull one
calendar day at a time and cache it, so a reversal-day microstructure dissection never
re-pulls. The frame is normalised to a tidy schema (``ts, open, high, low, close, volume``)
and sorted ascending; empty/missing days degrade to an empty frame rather than raising.
"""
from __future__ import annotations

import pandas as pd

from ..fmp.client import FMPClient, FMPError
from .cache import ParquetCache

_COLS = ["ts", "open", "high", "low", "close", "volume"]


def _normalize(rows: object) -> pd.DataFrame:
    """Coerce an FMP intraday payload into ``DataFrame[ts, open, high, low, close, volume]``."""
    df = pd.DataFrame(rows or [])
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=_COLS)
    df = df.rename(columns={"date": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df[_COLS].dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def get_intraday(
    client: FMPClient,
    cache: ParquetCache,
    fmp_symbol: str,
    day: str,
    interval: str = "1min",
    refresh: bool = False,
) -> pd.DataFrame:
    """Cached intraday bars for ``fmp_symbol`` on calendar ``day`` (``YYYY-MM-DD``).

    Returns ``DataFrame[ts(datetime), open, high, low, close, volume]`` sorted ascending,
    restricted to the requested calendar day. An empty/missing day yields an empty frame
    with the canonical columns (never raises on a no-data day).
    """
    key = f"intraday/{fmp_symbol}/{interval}/{day}"

    def _fetch() -> pd.DataFrame:
        try:
            rows = client.get(
                f"historical-chart/{interval}",
                symbol=fmp_symbol,
                **{"from": day, "to": day},
            )
        except FMPError:
            return pd.DataFrame(columns=_COLS)
        return _normalize(rows)

    df = cache.get_or_fetch(key, _fetch, refresh=refresh)
    if df.empty:
        return pd.DataFrame(columns=_COLS)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    # Defensive: keep only the requested calendar day (some payloads spill adjacent days).
    same_day = df["ts"].dt.normalize() == pd.Timestamp(day).normalize()
    df = df[same_day].sort_values("ts").reset_index(drop=True)
    return df[_COLS]
