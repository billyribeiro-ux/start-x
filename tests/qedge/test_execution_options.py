"""Tests for :mod:`qedge.execution.options` — options bid-ask realism.

Covers the contract claims:

* a buy fills at or above the mid and a sell at or below it (the spread is paid
  both ways — the dominant options cost);
* the per-side cross-spread cost scales linearly with the half-spread bps;
* :func:`nbbo_aligned_fill` against a synthetic :class:`OptionsNBBOFeed` fills on
  the correct side (ask for a buy, bid for a sell) AND against the Unavailable
  stub propagates :class:`FeedNotAvailable` (no synthetic fill from a dead feed).
"""
from __future__ import annotations

import pandas as pd
import pytest

from qedge.data.protocols import FeedNotAvailable
from qedge.data.stub_adapters import UnavailableOptionsNBBOFeed
from qedge.data.synthetic import SyntheticMarket
from qedge.execution.options import (
    NBBOFill,
    OptionsBidAskModel,
    nbbo_aligned_fill,
)

# Analysis-window-aligned fixture dates (well inside Jan 2019 -> Jun 2026).
_START = pd.Timestamp("2019-01-01")
_END = pd.Timestamp("2021-12-31")
_ASOF = pd.Timestamp("2020-06-30")
_SYMBOL = "SPY"
_EXPIRY = pd.Timestamp("2020-09-18")
_STRIKE = 100.0


def _market() -> SyntheticMarket:
    return SyntheticMarket(_START, _END, seed=7, symbols=("SPY", "QQQ", "IWM"))


# --- bid-ask model: spread paid both ways ---------------------------------


def test_buy_fills_at_or_above_mid_sell_at_or_below() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    mid = 2.50
    buy = model.fill_price(side="buy", mid=mid)
    sell = model.fill_price(side="sell", mid=mid)
    assert buy >= mid
    assert sell <= mid
    # Symmetric half-spread around the mid.
    assert buy - mid == pytest.approx(mid - sell)


def test_buy_fills_at_ask_sell_at_bid_with_nbbo() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    bid, ask = 2.40, 2.60
    assert model.fill_price(bid=bid, ask=ask, side="buy") == pytest.approx(ask)
    assert model.fill_price(bid=bid, ask=ask, side="sell") == pytest.approx(bid)


def test_cross_spread_cost_scales_with_half_spread_bps() -> None:
    mid = 5.0
    cheap = OptionsBidAskModel(half_spread_bps=25.0).cross_spread_cost(mid, side="buy")
    dear = OptionsBidAskModel(half_spread_bps=50.0).cross_spread_cost(mid, side="buy")
    assert dear == pytest.approx(2.0 * cheap)
    # Exact value: mid * bps / 1e4.
    assert dear == pytest.approx(mid * 50.0 / 1.0e4)


def test_cross_spread_cost_non_negative_both_sides() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    assert model.cross_spread_cost(3.0, side="buy") > 0.0
    assert model.cross_spread_cost(3.0, side="sell") > 0.0
    assert model.cross_spread_cost(3.0, side="buy") == pytest.approx(
        model.cross_spread_cost(3.0, side="sell")
    )


def test_from_config_uses_options_half_spread_bps() -> None:
    from qedge.config import QedgeConfig

    cfg = QedgeConfig()
    model = OptionsBidAskModel.from_config(cfg.execution)
    assert model.half_spread_bps == cfg.execution.options_half_spread_bps


def test_fill_price_rejects_inverted_quote() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    with pytest.raises(ValueError, match="ask must be >= bid"):
        model.fill_price(bid=2.60, ask=2.40, side="buy")


def test_fill_price_requires_quote_or_mid() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    with pytest.raises(ValueError, match="bid"):
        model.fill_price(side="buy")


def test_invalid_side_rejected() -> None:
    model = OptionsBidAskModel(half_spread_bps=50.0)
    with pytest.raises(ValueError, match="side"):
        model.fill_price(side="hold", mid=1.0)  # type: ignore[arg-type]


# --- nbbo_aligned_fill against the synthetic feed -------------------------


def test_nbbo_aligned_fill_buy_uses_ask() -> None:
    market = _market()
    quotes = market.nbbo(_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE)
    expected_ask = float(quotes.iloc[-1]["ask"])
    fill = nbbo_aligned_fill(
        market, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="buy"
    )
    assert isinstance(fill, NBBOFill)
    assert fill.fill_price == pytest.approx(expected_ask)
    assert fill.fill_price >= fill.bid
    assert fill.side == "buy"


def test_nbbo_aligned_fill_sell_uses_bid() -> None:
    market = _market()
    quotes = market.nbbo(_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE)
    expected_bid = float(quotes.iloc[-1]["bid"])
    fill = nbbo_aligned_fill(
        market, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="sell"
    )
    assert fill.fill_price == pytest.approx(expected_bid)
    assert fill.fill_price <= fill.ask
    assert fill.side == "sell"


def test_nbbo_aligned_fill_buy_costs_more_than_sell() -> None:
    market = _market()
    buy = nbbo_aligned_fill(
        market, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="buy"
    )
    sell = nbbo_aligned_fill(
        market, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="sell"
    )
    assert buy.fill_price > sell.fill_price  # spread paid both ways


def test_nbbo_aligned_fill_iv_from_contemporaneous_row() -> None:
    market = _market()
    quotes = market.nbbo(_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE)
    fill = nbbo_aligned_fill(
        market, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="buy"
    )
    assert fill.iv == pytest.approx(float(quotes.iloc[-1]["iv"]))
    assert fill.quote_ts <= _ASOF


def test_nbbo_aligned_fill_propagates_feed_not_available() -> None:
    dead_feed = UnavailableOptionsNBBOFeed()
    with pytest.raises(FeedNotAvailable):
        nbbo_aligned_fill(
            dead_feed, symbol=_SYMBOL, asof=_ASOF, expiry=_EXPIRY, strike=_STRIKE, side="buy"
        )


def test_nbbo_aligned_fill_raises_when_no_quote_before_asof() -> None:
    market = _market()
    # asof before the generated horizon -> empty NBBO -> explicit ValueError.
    early = pd.Timestamp("2018-01-01")
    with pytest.raises(ValueError, match="no NBBO quote"):
        nbbo_aligned_fill(
            market, symbol=_SYMBOL, asof=early, expiry=_EXPIRY, strike=_STRIKE, side="buy"
        )
