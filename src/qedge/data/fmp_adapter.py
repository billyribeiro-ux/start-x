"""FMP-backed point-in-time feed adapters.

These adapters implement the data-layer Protocols (:class:`PriceFeed`,
:class:`FundamentalsFeed`, :class:`NewsFeed`) by *wrapping* the reused
``startx`` FMP/prices stack rather than re-implementing HTTP. Two concerns are
kept strictly separated:

* **Fetching** raw frames from FMP — delegated to an injected source object so
  the adapter never imports network code at construction time and unit tests can
  pass a fake in-memory source with no network.
* **Point-in-time enforcement** — every public method filters its result to rows
  whose information timestamp is ``<= asof`` using the same column conventions
  promised by :mod:`qedge.data.protocols`. Future rows are never returned.

The default sources are built lazily from ``startx.settings`` (which reads the
gitignored ``.env``) and are only constructed when a caller does not inject a
source — so importing this module, and the offline tests, never touch the key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from qedge.config import QedgeConfig, get_config
from startx.data.cache import ParquetCache
from startx.data.prices import get_prices
from startx.fmp.client import FMPClient
from startx.fmp.endpoints import earnings as fmp_earnings
from startx.fmp.endpoints import stock_news as fmp_stock_news

# -- protocol return-column contracts (LOCKED to qedge.data.protocols) --------
_PRICE_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")
_PRICE_TS_COL = "date"
_EARNINGS_TS_COL = "release_ts"
_NEWS_TS_COL = "published_ts"


# -- injectable data sources --------------------------------------------------
@runtime_checkable
class PriceSource(Protocol):
    """Fetches a symbol's full daily OHLCV history as a tidy frame.

    The frame must contain at least ``date`` plus the OHLCV columns; extra
    columns (e.g. ``ret``, ``gap`` from the startx enrichment) are tolerated and
    dropped by the adapter. Implementations need not apply any ``asof`` filter —
    that is the adapter's job.
    """

    def __call__(self, symbol: str, *, start: str) -> pd.DataFrame: ...


@runtime_checkable
class FundamentalsSource(Protocol):
    """Fetches a symbol's earnings rows; must expose a ``ts`` release column."""

    def __call__(self, symbol: str) -> pd.DataFrame: ...


@runtime_checkable
class NewsSource(Protocol):
    """Fetches a symbol's news rows; must expose a ``ts`` publication column."""

    def __call__(self, symbol: str) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class _StartxPriceSource:
    """Default :class:`PriceSource`: cached startx FMP daily history.

    The FMP client (which reads the API key from ``.env`` via startx settings)
    and the parquet cache are constructed on first call, never at import time.
    """

    cache_dir: str

    def __call__(self, symbol: str, *, start: str) -> pd.DataFrame:
        client = FMPClient()
        try:
            cache = ParquetCache(self.cache_dir)
            return get_prices(client, cache, symbol, history_start=start)
        finally:
            client.close()


@dataclass(frozen=True, slots=True)
class _StartxFundamentalsSource:
    """Default :class:`FundamentalsSource`: startx ``earnings`` endpoint."""

    def __call__(self, symbol: str) -> pd.DataFrame:
        client = FMPClient()
        try:
            return fmp_earnings(client, symbol)
        finally:
            client.close()


@dataclass(frozen=True, slots=True)
class _StartxNewsSource:
    """Default :class:`NewsSource`: startx ``stock_news`` endpoint."""

    def __call__(self, symbol: str) -> pd.DataFrame:
        client = FMPClient()
        try:
            return fmp_stock_news(client, symbol)
        finally:
            client.close()


# -- shared PIT helper --------------------------------------------------------
def _pit_filter(
    df: pd.DataFrame, ts_col: str, asof: pd.Timestamp, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Project to ``columns`` and keep only rows with ``ts_col <= asof``.

    The timestamp column is coerced to tz-naive datetimes; unparseable / missing
    stamps are treated as not-yet-public and dropped (the conservative choice for
    a no-lookahead guarantee). Output is sorted ascending by ``ts_col`` with a
    fresh range index.
    """
    out = df.loc[:, list(columns)].copy()
    ts = pd.to_datetime(out[ts_col], errors="coerce")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    out[ts_col] = ts
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tz is not None:
        asof_ts = asof_ts.tz_localize(None)
    mask = ts.notna() & (ts <= asof_ts)
    return out.loc[mask].sort_values(ts_col).reset_index(drop=True)


def _resolve_start(start: pd.Timestamp | None, config: QedgeConfig) -> str:
    """Choose the warm-up history start: explicit ``start`` else config default."""
    if start is not None:
        return str(pd.Timestamp(start).strftime("%Y-%m-%d"))
    return config.data.history_start


# -- feeds --------------------------------------------------------------------
class FMPPriceFeed:
    """:class:`~qedge.data.protocols.PriceFeed` backed by FMP daily EOD bars.

    A daily bar for date ``d`` is public at its close, i.e. visible whenever
    ``asof >= d``; rows with ``date > asof`` are dropped.
    """

    def __init__(
        self,
        source: PriceSource | None = None,
        *,
        config: QedgeConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._source: PriceSource = source or _StartxPriceSource(
            self._config.data.cache_dir
        )

    def history(
        self, symbol: str, *, asof: pd.Timestamp, start: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return ``[date, open, high, low, close, volume]`` with ``date <= asof``."""
        raw = self._source(symbol, start=_resolve_start(start, self._config))
        if raw.empty:
            return pd.DataFrame(columns=list(_PRICE_COLUMNS))
        return _pit_filter(raw, _PRICE_TS_COL, asof, _PRICE_COLUMNS)


class FMPFundamentalsFeed:
    """:class:`~qedge.data.protocols.FundamentalsFeed` backed by FMP earnings.

    The startx ``earnings`` endpoint stamps each row with a tz-naive ``ts`` (the
    release date). That stamp is the information timestamp and is surfaced as the
    protocol's ``release_ts`` column.
    """

    def __init__(
        self,
        source: FundamentalsSource | None = None,
        *,
        config: QedgeConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._source: FundamentalsSource = source or _StartxFundamentalsSource()

    def earnings(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return earnings rows with ``release_ts <= asof``."""
        raw = self._source(symbol)
        if raw.empty or "ts" not in raw.columns:
            return pd.DataFrame(columns=[_EARNINGS_TS_COL])
        renamed = raw.rename(columns={"ts": _EARNINGS_TS_COL})
        return _pit_filter(renamed, _EARNINGS_TS_COL, asof, (_EARNINGS_TS_COL,))


class FMPNewsFeed:
    """:class:`~qedge.data.protocols.NewsFeed` backed by FMP stock news.

    The startx ``stock_news`` endpoint stamps each row with a tz-naive ``ts``
    (the publication time), surfaced here as the protocol's ``published_ts``.
    """

    def __init__(
        self,
        source: NewsSource | None = None,
        *,
        config: QedgeConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._source: NewsSource = source or _StartxNewsSource()

    def items(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return news rows with ``published_ts <= asof``."""
        raw = self._source(symbol)
        if raw.empty or "ts" not in raw.columns:
            return pd.DataFrame(columns=[_NEWS_TS_COL])
        renamed = raw.rename(columns={"ts": _NEWS_TS_COL})
        return _pit_filter(renamed, _NEWS_TS_COL, asof, (_NEWS_TS_COL,))
