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


def historical_chart(
    client: FMPClient, symbol: str, interval: str = "1min",
    start: str | None = None, end: str | None = None,
) -> pd.DataFrame:
    """Intraday OHLCV bars (``interval`` in {1min,5min,15min,30min,1hour,4hour}) for one symbol.

    Timestamps are US/Eastern regular-session (09:30–15:59). Returns tidy OHLCV sorted ascending by
    ``datetime``; empty DataFrame on any API error so a missing intraday day never sinks a blotter run.
    """
    rows = _safe(lambda: client.get(
        f"historical-chart/{interval}", symbol=symbol, **{"from": start, "to": end}))
    if isinstance(rows, pd.DataFrame):           # _safe returned an empty frame (error path)
        return rows
    df = _to_df(rows, {"date": "datetime"})
    if df.empty:
        return df
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].sort_values("datetime").reset_index(drop=True)
    return df


def profile(client: FMPClient, symbol: str) -> dict:
    data = client.get("profile", symbol=symbol)
    return data[0] if isinstance(data, list) and data else (data or {})


# -- company catalysts ------------------------------------------------------
#: Wall-clock time stamped onto an earnings report whose session ("when": bmo/amc) is unknown
#: or after-close. One second past the 16:00 cash close is deliberately *post-close*, so the
#: point-in-time gate in ``events.attribute`` advances it to the NEXT trading day (t+1) rather
#: than crediting the report to day t's move. This is the conservative PIT choice: most US
#: single-stock earnings are released after the close (AMC), and the FMP stable ``earnings``
#: feed carries no intraday time, so absent an explicit "bmo" we must assume after-close.
_AMC_STAMP = "16:00:01"


def earnings(client: FMPClient, symbol: str, limit: int = 80) -> pd.DataFrame:
    """Earnings history with a point-in-time ``ts``.

    The FMP stable ``earnings`` endpoint returns only a ``date`` (no BMO/AMC time). If a future
    feed revision carries a ``time``/``when`` field, a "bmo"/"before"/"premarket" value keeps the
    report on its report date (it was public before that day's open); anything else — and the
    common no-time case — is placed one second after the cash close so it attributes to t+1.
    """
    def _fetch() -> pd.DataFrame:
        df = _to_df(client.get("earnings", symbol=symbol, limit=limit))
        if df.empty or "date" not in df.columns:
            df["ts"] = pd.NaT if not df.empty else df.get("ts")
            return df
        date = pd.to_datetime(df["date"], errors="coerce")
        # Default every report to the post-close (AMC) boundary so it lands on t+1 unless we have
        # explicit evidence it was released before the open.
        stamp = pd.Series(_AMC_STAMP, index=df.index)
        when_col = next((c for c in ("time", "when", "session") if c in df.columns), None)
        if when_col is not None:
            w = df[when_col].astype(str).str.lower()
            is_bmo = w.str.contains("bmo|before|pre|morning|am", regex=True, na=False)
            stamp = stamp.mask(is_bmo, "00:00:00")  # before the open -> stays on the report date
        df["ts"] = pd.to_datetime(
            date.dt.strftime("%Y-%m-%d") + " " + stamp, errors="coerce"
        )
        return df

    return _safe(_fetch)


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
