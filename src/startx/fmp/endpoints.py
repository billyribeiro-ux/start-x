"""Typed wrappers around the FMP stable endpoints used by the engine.

Each returns a tidy pandas DataFrame with parsed datetime columns. Network/JSON failures
for optional catalyst feeds degrade to an empty DataFrame so one missing feed never sinks a
whole scan (e.g. ETFs have no earnings/insider data).
"""
from __future__ import annotations

import pandas as pd

from .client import FMPClient, FMPError


def _to_df(rows: object, ts_cols: dict[str, str] | None = None) -> pd.DataFrame:
    """Build a DataFrame; parse the given columns to tz-naive datetimes (``{src: dst}``)."""
    df = pd.DataFrame(rows or [])
    if ts_cols:
        for src, dst in ts_cols.items():
            if src in df.columns:
                parsed = pd.to_datetime(df[src], errors="coerce", utc=True)
                df[dst] = parsed.dt.tz_localize(None)
            else:
                df[dst] = pd.NaT
    return df


def _safe(fetch):
    try:
        return fetch()
    except FMPError:
        return pd.DataFrame()


# -- prices -----------------------------------------------------------------
def historical_price_eod(
    client: FMPClient, symbol: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    rows = client.get(
        "historical-price-eod/full", symbol=symbol, **{"from": start, "to": end}
    )
    df = _to_df(rows, {"date": "date"})
    if df.empty:
        return df
    keep = ["date", "open", "high", "low", "close", "volume", "vwap", "changePercent"]
    df = df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)
    return df


def profile(client: FMPClient, symbol: str) -> dict:
    data = client.get("profile", symbol=symbol)
    return data[0] if isinstance(data, list) and data else (data or {})


# -- company catalysts ------------------------------------------------------
def earnings(client: FMPClient, symbol: str, limit: int = 80) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("earnings", symbol=symbol, limit=limit), {"date": "ts"}
        )
    )


def analyst_grades(client: FMPClient, symbol: str, limit: int = 1000) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("grades", symbol=symbol, limit=limit), {"date": "ts"}
        )
    )


def price_target_news(client: FMPClient, symbol: str, limit: int = 300) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("price-target-news", symbol=symbol, page=0, limit=limit),
            {"publishedDate": "ts"},
        )
    )


def insider_trades(client: FMPClient, symbol: str, limit: int = 500) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("insider-trading/search", symbol=symbol, limit=limit),
            {"filingDate": "ts", "transactionDate": "transaction_ts"},
        )
    )


def senate_trades(client: FMPClient, symbol: str, limit: int = 200) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("senate-trades", symbol=symbol, limit=limit),
            {"disclosureDate": "ts", "transactionDate": "transaction_ts"},
        )
    )


def house_trades(client: FMPClient, symbol: str, limit: int = 200) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("house-trades", symbol=symbol, limit=limit),
            {"disclosureDate": "ts", "transactionDate": "transaction_ts"},
        )
    )


def stock_news(
    client: FMPClient,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = 250,
) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get(
                "news/stock", symbols=symbol, limit=limit, **{"from": start, "to": end}
            ),
            {"publishedDate": "ts"},
        )
    )


# -- macro ------------------------------------------------------------------
def economic_calendar(client: FMPClient, start: str, end: str) -> pd.DataFrame:
    return _safe(
        lambda: _to_df(
            client.get("economic-calendar", **{"from": start, "to": end}),
            {"date": "ts"},
        )
    )
