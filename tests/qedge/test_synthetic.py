"""Tests for :class:`qedge.data.synthetic.SyntheticMarket`.

Covers: determinism (same/different seed), point-in-time / no-lookahead, OHLC
sanity, options bid<ask, and runtime Protocol conformance.
"""
from __future__ import annotations

import pandas as pd
import pytest

from qedge.data.protocols import (
    FundamentalsFeed,
    InternalsFeed,
    NewsFeed,
    OptionsNBBOFeed,
    PriceFeed,
)
from qedge.data.synthetic import SyntheticMarket
from qedge.repro import hash_frame

# Analysis-window-aligned fixture dates (well inside Jan 2019 -> Jun 2026).
_START = pd.Timestamp("2019-01-01")
_END = pd.Timestamp("2021-12-31")
_ASOF = pd.Timestamp("2020-06-30")
_FAR_END = pd.Timestamp("2024-12-31")  # generate far past _ASOF for the PIT canary
_SYMBOL = "SPY"
_EXPIRY = pd.Timestamp("2020-09-18")
_STRIKE = 100.0


def _market(seed: int = 7, end: pd.Timestamp = _END) -> SyntheticMarket:
    return SyntheticMarket(_START, end, seed=seed, symbols=("SPY", "QQQ", "IWM"))


# --- (a) determinism ------------------------------------------------------


def test_same_seed_byte_identical_prices() -> None:
    a = _market(seed=7).history(_SYMBOL, asof=_END)
    b = _market(seed=7).history(_SYMBOL, asof=_END)
    assert hash_frame(a) == hash_frame(b)


def test_different_seed_changes_prices() -> None:
    a = _market(seed=7).history(_SYMBOL, asof=_END)
    b = _market(seed=8).history(_SYMBOL, asof=_END)
    assert hash_frame(a) != hash_frame(b)


def test_determinism_across_all_feeds() -> None:
    m1, m2 = _market(seed=7), _market(seed=7)
    assert hash_frame(m1.earnings(_SYMBOL, asof=_END)) == hash_frame(
        m2.earnings(_SYMBOL, asof=_END)
    )
    assert hash_frame(m1.items(_SYMBOL, asof=_END)) == hash_frame(m2.items(_SYMBOL, asof=_END))
    assert hash_frame(
        m1.nbbo(_SYMBOL, asof=_END, expiry=_EXPIRY, strike=_STRIKE)
    ) == hash_frame(m2.nbbo(_SYMBOL, asof=_END, expiry=_EXPIRY, strike=_STRIKE))
    assert hash_frame(m1.internals(asof=_END)) == hash_frame(m2.internals(asof=_END))


def test_distinct_symbols_have_distinct_prices() -> None:
    m = _market(seed=7)
    spy = m.history("SPY", asof=_END)
    qqq = m.history("QQQ", asof=_END)
    assert hash_frame(spy) != hash_frame(qqq)


# --- (b) point-in-time / no-lookahead ------------------------------------


def test_prices_no_lookahead_short_vs_long_generation() -> None:
    short = _market(end=_ASOF).history(_SYMBOL, asof=_ASOF)
    long = _market(end=_FAR_END).history(_SYMBOL, asof=_ASOF)
    assert hash_frame(short) == hash_frame(long)


def test_prices_never_return_future_rows() -> None:
    frame = _market(end=_FAR_END).history(_SYMBOL, asof=_ASOF)
    assert (frame["date"] <= _ASOF).all()
    assert not frame.empty


def test_earnings_pit_release_ts_not_after_asof() -> None:
    short = _market(end=_ASOF).earnings(_SYMBOL, asof=_ASOF)
    long = _market(end=_FAR_END).earnings(_SYMBOL, asof=_ASOF)
    assert hash_frame(short) == hash_frame(long)
    assert (long["release_ts"] <= _ASOF).all()


def test_news_pit_published_ts_not_after_asof() -> None:
    short = _market(end=_ASOF).items(_SYMBOL, asof=_ASOF)
    long = _market(end=_FAR_END).items(_SYMBOL, asof=_ASOF)
    assert hash_frame(short) == hash_frame(long)
    assert (long["published_ts"] <= _ASOF).all()


def test_nbbo_pit_quote_ts_not_after_asof() -> None:
    short = _market(end=_ASOF).nbbo(_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE)
    long = _market(end=_FAR_END).nbbo(_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE)
    assert hash_frame(short) == hash_frame(long)
    assert (long["quote_ts"] <= _ASOF).all()


def test_internals_pit_ts_not_after_asof() -> None:
    short = _market(end=_ASOF).internals(asof=_ASOF)
    long = _market(end=_FAR_END).internals(asof=_ASOF)
    assert hash_frame(short) == hash_frame(long)
    assert (long["ts"] <= _ASOF).all()


# --- (c) OHLC sanity ------------------------------------------------------


def test_ohlc_sanity() -> None:
    frame = _market().history(_SYMBOL, asof=_END)
    assert not frame.empty
    assert (frame["low"] <= frame["open"]).all()
    assert (frame["low"] <= frame["close"]).all()
    assert (frame["open"] <= frame["high"]).all()
    assert (frame["close"] <= frame["high"]).all()
    assert (frame["low"] <= frame["high"]).all()
    assert (frame["low"] > 0).all()
    assert (frame["volume"] > 0).all()


def test_price_columns_match_protocol() -> None:
    frame = _market().history(_SYMBOL, asof=_END)
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]


# --- (d) options bid<ask --------------------------------------------------


def test_options_bid_below_ask_and_sizes_positive() -> None:
    frame = _market().nbbo(_SYMBOL, asof=_END, expiry=_EXPIRY, strike=_STRIKE)
    assert not frame.empty
    assert list(frame.columns) == ["quote_ts", "bid", "ask", "bid_size", "ask_size", "iv"]
    assert (frame["bid"] < frame["ask"]).all()
    assert (frame["bid"] > 0).all()
    assert (frame["bid_size"] > 0).all()
    assert (frame["ask_size"] > 0).all()
    assert (frame["iv"] > 0).all()


def test_internals_columns_match_protocol() -> None:
    frame = _market().internals(asof=_END)
    assert not frame.empty
    assert list(frame.columns) == ["ts", "tick", "trin", "add", "vold"]


def test_earnings_release_after_period_end() -> None:
    frame = _market().earnings(_SYMBOL, asof=_END)
    assert not frame.empty
    assert list(frame.columns) == ["symbol", "period_end", "release_ts", "eps"]
    assert (frame["release_ts"] > frame["period_end"]).all()


def test_news_columns_match_protocol() -> None:
    frame = _market().items(_SYMBOL, asof=_END)
    assert list(frame.columns) == ["symbol", "published_ts", "headline", "sentiment"]


# --- (e) runtime Protocol conformance ------------------------------------


def test_satisfies_all_feed_protocols() -> None:
    market = _market()
    assert isinstance(market, PriceFeed)
    assert isinstance(market, FundamentalsFeed)
    assert isinstance(market, NewsFeed)
    assert isinstance(market, OptionsNBBOFeed)
    assert isinstance(market, InternalsFeed)


# --- misc -----------------------------------------------------------------


def test_unknown_symbol_raises() -> None:
    with pytest.raises(KeyError):
        _market().history("NVDA", asof=_END)


def test_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="end must be >= start"):
        SyntheticMarket(_END, _START, seed=7)
