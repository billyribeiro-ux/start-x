"""Options execution realism — the bid-ask spread is the dominant cost.

For an options strategy the spread, not commission, dwarfs every other friction:
a 50-bps half-spread crossed on entry and again on exit is a percent of notional
gone before any edge is measured. This module models that explicitly and pulls
fills from a contemporaneous NBBO so greeks/IV never come from a settlement
snapshot used as if it were intraday-available.

Two pieces:

* :class:`OptionsBidAskModel` — half-spread (``cfg.execution.options_half_spread_bps``)
  applied to a mid. Buying **lifts the ask** (pays up); selling **hits the bid**
  (receives less). :meth:`OptionsBidAskModel.cross_spread_cost` is the per-side
  cost; :meth:`OptionsBidAskModel.fill_price` is the synthetic crossed price when
  only a mid is known.
* :func:`nbbo_aligned_fill` — the honest fill. It reads the contemporaneous NBBO
  row at ``asof`` from an :class:`~qedge.data.protocols.OptionsNBBOFeed` and fills
  at the **ask** when buying / the **bid** when selling, sourcing IV from that
  same row. If the feed has no data it raises :class:`FeedNotAvailable`; this
  function propagates it rather than fabricating a fill — no synthetic quote
  stands in for a missing feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from qedge.config import ExecutionConfig, get_config
from qedge.data.protocols import OptionsNBBOFeed

__all__ = [
    "NBBOFill",
    "OptionsBidAskModel",
    "Side",
    "nbbo_aligned_fill",
]

_BPS: float = 1.0e4  # basis points per unit (1.0 = 10_000 bps)

#: Order side. ``"buy"`` lifts the ask; ``"sell"`` hits the bid.
Side = Literal["buy", "sell"]


def _side_sign(side: Side) -> int:
    """+1 for a buy (pays up to the ask), -1 for a sell (receives down to the bid)."""
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


@dataclass(frozen=True)
class OptionsBidAskModel:
    """Half-spread bid-ask cost model — the dominant options friction.

    Parameters
    ----------
    half_spread_bps:
        Half of the quoted bid-ask spread, in basis points of mid
        (``cfg.execution.options_half_spread_bps``). The full quoted spread is
        ``2 * half_spread_bps``; one side of a round trip pays one half-spread.
    """

    half_spread_bps: float

    @classmethod
    def from_config(cls, execution: ExecutionConfig | None = None) -> OptionsBidAskModel:
        """Build from ``cfg.execution`` (the process singleton if not supplied)."""
        exe = execution if execution is not None else get_config().execution
        return cls(half_spread_bps=float(exe.options_half_spread_bps))

    def _half_spread_frac(self) -> float:
        """Half-spread as a fraction of mid (bps / 1e4)."""
        return self.half_spread_bps / _BPS

    def cross_spread_cost(self, mid: float, *, side: Side) -> float:
        """Per-side cost (in price units) of crossing the spread from ``mid``.

        Always non-negative: crossing the spread is a cost whether buying or
        selling. Equals ``mid * half_spread_bps / 1e4`` and so scales linearly
        with the half-spread bps. The ``side`` is validated but does not change
        the magnitude — both directions pay one half-spread.
        """
        _side_sign(side)  # validate side
        return abs(float(mid)) * self._half_spread_frac()

    def fill_price(self, bid: float | None = None, ask: float | None = None, *,
                   side: Side, mid: float | None = None) -> float:
        """Crossed fill price for ``side``.

        With an explicit ``bid``/``ask`` (an NBBO), a buy fills at the ``ask`` and
        a sell at the ``bid`` — the dominant cost, taken directly off the quote.
        With only a ``mid``, the half-spread is applied synthetically: a buy fills
        at ``mid * (1 + half_spread)`` (>= mid) and a sell at
        ``mid * (1 - half_spread)`` (<= mid). Supply either an NBBO pair or a mid.
        """
        sign = _side_sign(side)
        if bid is not None and ask is not None:
            if ask < bid:
                raise ValueError("ask must be >= bid")
            return float(ask) if sign > 0 else float(bid)
        if mid is None:
            raise ValueError("supply either bid+ask or mid")
        return float(mid) * (1.0 + sign * self._half_spread_frac())


@dataclass(frozen=True)
class NBBOFill:
    """A fill aligned to a contemporaneous NBBO row.

    Attributes
    ----------
    fill_price:
        The crossed price: the ask for a buy, the bid for a sell.
    quote_ts:
        Timestamp of the NBBO row used (the latest quote ``<= asof``).
    bid, ask:
        The bid/ask of that NBBO row (the contemporaneous quote).
    iv:
        Implied vol from that same NBBO row — the only honest source for greeks
        (never a settlement value used as if intraday-available).
    side:
        The order side that produced the fill.
    """

    fill_price: float
    quote_ts: pd.Timestamp
    bid: float
    ask: float
    iv: float
    side: Side


def nbbo_aligned_fill(
    feed: OptionsNBBOFeed,
    *,
    symbol: str,
    asof: pd.Timestamp,
    expiry: pd.Timestamp,
    strike: float,
    side: Side,
) -> NBBOFill:
    """Fill an options order against the contemporaneous NBBO at ``asof``.

    Reads the NBBO stream for ``(symbol, expiry, strike)`` truncated to
    ``quote_ts <= asof`` (the point-in-time contract), takes the **latest** such
    quote, and fills at the correct side: the **ask** when buying, the **bid**
    when selling. IV is read from that same row so any downstream greek is built
    from the contemporaneous surface, never settlement.

    Honesty rule: if the feed raises
    :class:`~qedge.data.protocols.FeedNotAvailable` (no OPRA-grade source in this
    environment), the exception propagates unchanged — this function never
    fabricates a synthetic fill to paper over a missing feed.

    Parameters
    ----------
    feed:
        Any :class:`~qedge.data.protocols.OptionsNBBOFeed` implementation.
    symbol, asof, expiry, strike, side:
        Contract coordinates, the as-of timestamp, and the order side.

    Returns
    -------
    NBBOFill
        The crossed price, the source quote timestamp, its bid/ask/iv, and side.

    Raises
    ------
    qedge.data.protocols.FeedNotAvailable
        Propagated from the feed when no NBBO source exists.
    ValueError
        If the feed returns no quote at or before ``asof``.
    """
    _side_sign(side)  # validate side early
    quotes = feed.nbbo(symbol, asof=pd.Timestamp(asof), expiry=pd.Timestamp(expiry),
                       strike=float(strike))
    if quotes.empty:
        raise ValueError(
            f"no NBBO quote at or before asof={pd.Timestamp(asof)!r} for "
            f"{symbol} {pd.Timestamp(expiry).date()} {strike}"
        )
    # The contemporaneous quote is the latest row not after ``asof``. The feed
    # already truncates to ``quote_ts <= asof`` and sorts ascending; take the last.
    row = quotes.iloc[-1]
    bid = float(row["bid"])
    ask = float(row["ask"])
    iv = float(row["iv"])
    model = OptionsBidAskModel.from_config()
    fill = model.fill_price(bid=bid, ask=ask, side=side)
    return NBBOFill(
        fill_price=fill,
        quote_ts=pd.Timestamp(row["quote_ts"]),
        bid=bid,
        ask=ask,
        iv=iv,
        side=side,
    )
