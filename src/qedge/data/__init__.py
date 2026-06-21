"""qedge data layer — point-in-time feeds, information boundaries, adapters.

Every feed takes an explicit ``asof`` timestamp and must return only information
that was public at or before ``asof``. Feeds that have no available data source
(OPRA NBBO, real-time internals) raise :class:`FeedNotAvailable` rather than
fabricating values.
"""
from __future__ import annotations

from qedge.data.boundary import BoundaryKind, InformationBoundary
from qedge.data.protocols import (
    FeedNotAvailable,
    FundamentalsFeed,
    InternalsFeed,
    NewsFeed,
    OptionsNBBOFeed,
    PriceFeed,
)

__all__ = [
    "BoundaryKind",
    "InformationBoundary",
    "FeedNotAvailable",
    "FundamentalsFeed",
    "InternalsFeed",
    "NewsFeed",
    "OptionsNBBOFeed",
    "PriceFeed",
]
