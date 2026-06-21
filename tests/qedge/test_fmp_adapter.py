"""Tests for the FMP-backed feed adapters and the unavailable-feed stubs.

All tests run OFFLINE except a single live smoke test, which is decorated to
SKIP cleanly when ``FMP_API_KEY`` is absent. The PIT logic is exercised with an
injected in-memory fake source, so the no-lookahead guarantee is verified with
no network access.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from qedge.data.fmp_adapter import (
    FMPFundamentalsFeed,
    FMPNewsFeed,
    FMPPriceFeed,
)
from qedge.data.protocols import (
    FeedNotAvailable,
    FundamentalsFeed,
    InternalsFeed,
    NewsFeed,
    OptionsNBBOFeed,
    PriceFeed,
)
from qedge.data.stub_adapters import (
    UnavailableInternalsFeed,
    UnavailableOptionsNBBOFeed,
)

_ASOF = pd.Timestamp("2021-06-15")
_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


# -- fakes (no network) -------------------------------------------------------
def _fake_price_frame() -> pd.DataFrame:
    """Daily bars spanning before and after the as-of date, plus extra columns.

    The trailing ``ret``/``gap`` columns mimic the startx enrichment and must be
    dropped by the adapter so the returned frame matches the protocol exactly.
    """
    dates = pd.to_datetime(
        ["2021-06-13", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17"]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2],
            "volume": [10, 20, 30, 40, 50],
            "ret": [0.0, 0.1, 0.2, 0.3, 0.4],
            "gap": [0.0, 0.01, 0.02, 0.03, 0.04],
        }
    )


def _fake_ts_frame() -> pd.DataFrame:
    """Earnings/news-shaped frame with a tz-naive ``ts`` column straddling asof."""
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2021-06-10", "2021-06-15", "2021-06-20"]),
            "payload": ["past", "today", "future"],
        }
    )


# -- stub feeds raise FeedNotAvailable ---------------------------------------
def test_options_stub_raises_feed_not_available() -> None:
    feed = UnavailableOptionsNBBOFeed()
    with pytest.raises(FeedNotAvailable, match="OPRA"):
        feed.nbbo("SPY", asof=_ASOF, expiry=pd.Timestamp("2021-07-16"), strike=420.0)


def test_internals_stub_raises_feed_not_available() -> None:
    feed = UnavailableInternalsFeed()
    with pytest.raises(FeedNotAvailable, match="internals"):
        feed.internals(asof=_ASOF)


def test_stubs_satisfy_protocols() -> None:
    assert isinstance(UnavailableOptionsNBBOFeed(), OptionsNBBOFeed)
    assert isinstance(UnavailableInternalsFeed(), InternalsFeed)


# -- price feed PIT logic (offline, injected source) --------------------------
def test_price_feed_drops_future_rows() -> None:
    feed = FMPPriceFeed(source=lambda symbol, *, start: _fake_price_frame())
    out = feed.history("SPY", asof=_ASOF)
    assert list(out.columns) == _PRICE_COLUMNS
    assert (out["date"] <= _ASOF).all()
    # 2021-06-16 and 2021-06-17 are strictly after asof and must be dropped.
    assert out["date"].max() == _ASOF
    assert len(out) == 3


def test_price_feed_satisfies_protocol() -> None:
    feed = FMPPriceFeed(source=lambda symbol, *, start: _fake_price_frame())
    assert isinstance(feed, PriceFeed)


def test_price_feed_empty_source_returns_empty_with_columns() -> None:
    feed = FMPPriceFeed(source=lambda symbol, *, start: pd.DataFrame())
    out = feed.history("SPY", asof=_ASOF)
    assert out.empty
    assert list(out.columns) == _PRICE_COLUMNS


def test_price_feed_passes_start_through() -> None:
    captured: dict[str, str] = {}

    def _source(symbol: str, *, start: str) -> pd.DataFrame:
        captured["start"] = start
        return _fake_price_frame()

    feed = FMPPriceFeed(source=_source)
    feed.history("SPY", asof=_ASOF, start=pd.Timestamp("2019-01-01"))
    assert captured["start"] == "2019-01-01"


def test_price_feed_clips_lower_bound_even_when_source_returns_wider() -> None:
    """The window's lower bound is enforced on the OUTPUT, not just forwarded.

    The reused startx cache keys by symbol only, so a source can return history
    earlier than the requested ``start`` (a frame cached on an earlier, wider
    request). The adapter must still clip to ``date >= start`` — this is the
    regression guard for that bug.
    """
    # Source ignores `start` and returns the full 5-bar frame (2021-06-13..17),
    # mimicking a stale, wider cache hit.
    feed = FMPPriceFeed(source=lambda symbol, *, start: _fake_price_frame())
    out = feed.history("SPY", asof=_ASOF, start=pd.Timestamp("2021-06-14"))
    # Only 2021-06-14 and 2021-06-15 satisfy both start <= date <= asof.
    assert out["date"].min() == pd.Timestamp("2021-06-14")
    assert out["date"].max() == _ASOF
    assert len(out) == 2


# -- fundamentals / news feeds PIT logic (offline) ---------------------------
def test_fundamentals_feed_pit_and_rename() -> None:
    feed = FMPFundamentalsFeed(source=lambda symbol: _fake_ts_frame())
    out = feed.earnings("SPY", asof=_ASOF)
    assert "release_ts" in out.columns
    assert (out["release_ts"] <= _ASOF).all()
    assert len(out) == 2  # the 2021-06-20 future row is dropped
    assert isinstance(feed, FundamentalsFeed)


def test_news_feed_pit_and_rename() -> None:
    feed = FMPNewsFeed(source=lambda symbol: _fake_ts_frame())
    out = feed.items("SPY", asof=_ASOF)
    assert "published_ts" in out.columns
    assert (out["published_ts"] <= _ASOF).all()
    assert len(out) == 2
    assert isinstance(feed, NewsFeed)


def test_fundamentals_feed_empty_source() -> None:
    feed = FMPFundamentalsFeed(source=lambda symbol: pd.DataFrame())
    out = feed.earnings("SPY", asof=_ASOF)
    assert out.empty
    assert list(out.columns) == ["release_ts"]


# -- live smoke test (the only network-touching test; skips without a key) ----
@pytest.mark.skipif(
    not os.getenv("FMP_API_KEY"),
    reason="FMP_API_KEY not set; live smoke test skipped (no network in CI).",
)
def test_live_spy_price_smoke() -> None:
    feed = FMPPriceFeed()
    asof = pd.Timestamp("2024-01-31")
    out = feed.history("SPY", asof=asof)
    assert list(out.columns) == _PRICE_COLUMNS
    assert not out.empty
    assert (out["date"] <= asof).all()
