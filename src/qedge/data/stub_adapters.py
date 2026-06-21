"""Honest-degradation stub feeds for data sources with no provider.

No OPRA-grade options NBBO feed and no real-time market-internals (TICK / TRIN /
ADD / VOLD) feed exists in the default FMP environment. Rather than fabricate
quotes or settlement-derived greeks, these adapters implement the corresponding
Protocols by raising :class:`~qedge.data.protocols.FeedNotAvailable` with a clear
message. Callers catch it, skip the dependent strategy, and log the unpopulated
contract dimension.

Both classes structurally satisfy their Protocols (matching method signatures),
so ``isinstance`` checks against the ``runtime_checkable`` Protocols succeed.
"""
from __future__ import annotations

import pandas as pd

from qedge.data.protocols import FeedNotAvailable

_NO_NBBO_MESSAGE = (
    "Options NBBO is unavailable: no OPRA-grade quote feed exists in the default "
    "FMP environment. Greeks must come from contemporaneous IV surfaces, never "
    "settlement values used as if intraday-available."
)
_NO_INTERNALS_MESSAGE = (
    "Market internals (TICK / TRIN / ADD / VOLD) are unavailable: no real-time "
    "internals feed exists in the default FMP environment."
)


class UnavailableOptionsNBBOFeed:
    """:class:`~qedge.data.protocols.OptionsNBBOFeed` with no backing source."""

    def nbbo(
        self,
        symbol: str,
        *,
        asof: pd.Timestamp,
        expiry: pd.Timestamp,
        strike: float,
    ) -> pd.DataFrame:
        """Always raise :class:`FeedNotAvailable` — no NBBO feed exists."""
        raise FeedNotAvailable(_NO_NBBO_MESSAGE)


class UnavailableInternalsFeed:
    """:class:`~qedge.data.protocols.InternalsFeed` with no backing source."""

    def internals(self, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Always raise :class:`FeedNotAvailable` — no internals feed exists."""
        raise FeedNotAvailable(_NO_INTERNALS_MESSAGE)
