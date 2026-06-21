"""Point-in-time data feed Protocols.

Each feed is a ``runtime_checkable`` Protocol with a mandatory ``asof`` argument.
The implementation contract is identical across feeds:

    Return only rows whose public/dissemination timestamp is <= ``asof``.

This is the single architectural guarantee against look-ahead at the data layer.
Concrete adapters (FMP, synthetic) implement these; feeds with no available data
source raise :class:`FeedNotAvailable`.

Returned frames use tz-naive ``pandas.Timestamp`` values and a sorted, ascending
time column. Column conventions are documented per method.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


class FeedNotAvailable(RuntimeError):
    """Raised by adapters for data sources that do not exist in this environment.

    The honest-degradation rule: never fabricate options/NBBO or real-time
    internals when no feed exists. Callers catch this, skip the dependent
    strategy, and log that the contract dimension is unpopulated.
    """


@runtime_checkable
class PriceFeed(Protocol):
    """Daily OHLCV prices for a symbol, point-in-time as of ``asof``."""

    def history(
        self, symbol: str, *, asof: pd.Timestamp, start: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return columns ``[date, open, high, low, close, volume]``.

        Only rows with ``date <= asof`` are returned (a daily bar for date ``d``
        is considered public at its close, i.e. visible when ``asof >= d``).
        """
        ...


@runtime_checkable
class FundamentalsFeed(Protocol):
    """As-reported fundamentals / earnings, keyed by release timestamp."""

    def earnings(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return earnings rows with a public ``release_ts`` column <= ``asof``.

        Values are as-reported (never restated). Period-end dates are NOT used as
        the information timestamp — the actual release timestamp is.
        """
        ...


@runtime_checkable
class NewsFeed(Protocol):
    """News / sentiment items, lagged to dissemination time."""

    def items(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return news rows with a ``published_ts`` column <= ``asof``."""
        ...


@runtime_checkable
class OptionsNBBOFeed(Protocol):
    """Options NBBO quotes aligned to the exact quote timestamp.

    No OPRA-grade feed exists in the default environment; the FMP-backed adapter
    raises :class:`FeedNotAvailable`. Greeks must be computed from contemporaneous
    IV surfaces only — never from settlement values used as if intraday-available.
    """

    def nbbo(
        self, symbol: str, *, asof: pd.Timestamp, expiry: pd.Timestamp, strike: float
    ) -> pd.DataFrame:
        """Return NBBO rows ``[quote_ts, bid, ask, bid_size, ask_size, iv]`` <= ``asof``."""
        ...


@runtime_checkable
class InternalsFeed(Protocol):
    """Real-time market internals (TICK / TRIN / ADD / VOLD).

    These are intraday real-time series; no end-of-day value may leak into an
    intraday feature. No feed exists in the default environment — the adapter
    raises :class:`FeedNotAvailable`.
    """

    def internals(self, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return internals rows with a ``ts`` column <= ``asof``."""
        ...
