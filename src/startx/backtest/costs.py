"""Transaction-cost model for the backtest engine.

Costs are expressed in basis points (1 bp = 0.0001 = 0.01%). A *round trip* is
one full trade: an entry fill plus the matching exit fill. Commission and
slippage are each paid on both legs, so the round-trip fractional drag is

    2 * (commission_bps + slippage_bps) / 10_000

and it is charged exactly **once per round-trip trade** (subtracted from that
trade's gross return). The engine never charges it per day or per bar.
"""
from __future__ import annotations

from dataclasses import dataclass

_BPS = 1.0e4  # basis points per unit (1.0 = 10_000 bps)

#: Realistic *round-trip* cost (commission + slippage, both legs) per instrument, in basis points.
#: SPY is the deepest, tightest US equity ETF — its real round-trip slippage is ~1 bp, TIGHTER than
#: the generic 2 bp the books assume, so a SPY book costed at 2 bp is conservatively costed. Less
#: liquid names cost more. These are reusable reference points for cost-sensitivity work; the
#: portfolio/position engines still take a raw ``cost_bps`` so callers can sweep freely.
INSTRUMENT_ROUND_TRIP_BPS: dict[str, float] = {
    "SPY": 1.0,    # mega-cap ETF, penny-wide spread — the desk's primary instrument
    "QQQ": 1.0,
    "IWM": 2.0,
    "_DEFAULT": 5.0,  # generic large-cap single name
}


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost model in basis points.

    Parameters
    ----------
    commission_bps:
        Broker commission per leg, in basis points. Default 1.0 bp.
    slippage_bps:
        Expected slippage per leg, in basis points. Default 5.0 bp.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")

    def round_trip_cost(self) -> float:
        """Fractional cost charged once per round-trip trade (entry + exit).

        ``2 * (commission_bps + slippage_bps) / 1e4``. The factor of two covers
        both the entry and the exit fill.
        """
        return 2.0 * (self.commission_bps + self.slippage_bps) / _BPS

    def round_trip_bps(self) -> float:
        """The round-trip cost expressed back in basis points (``round_trip_cost`` × 1e4).

        Convenience for the portfolio/position engines, whose ``cost_bps`` parameter is a
        round-trip figure: ``CostModel.for_instrument('SPY').round_trip_bps()`` is the realistic
        ``cost_bps`` to feed those engines for a SPY book.
        """
        return self.round_trip_cost() * _BPS

    @classmethod
    def for_instrument(cls, symbol: str) -> "CostModel":
        """Build a cost model whose round-trip cost matches a named instrument's realistic level.

        Looks ``symbol`` up in :data:`INSTRUMENT_ROUND_TRIP_BPS` (falling back to ``_DEFAULT``) and
        splits that round-trip figure into per-leg commission/slippage. Commission is held at the
        default 1.0 bp/leg where the round-trip budget allows; the remainder is slippage. For an
        instrument whose entire round-trip budget is ≤ 2× the default commission, the whole budget
        is booked as commission (slippage 0) so the round-trip total is matched exactly.

        ``CostModel.for_instrument('SPY')`` reproduces SPY's ~1 bp round-trip (0.5 bp/leg).
        """
        rt_bps = INSTRUMENT_ROUND_TRIP_BPS.get(symbol.upper(),
                                               INSTRUMENT_ROUND_TRIP_BPS["_DEFAULT"])
        per_leg = rt_bps / 2.0  # round-trip = 2 * (commission + slippage) per leg
        default_comm = cls.__dataclass_fields__["commission_bps"].default
        if per_leg <= default_comm:
            return cls(commission_bps=per_leg, slippage_bps=0.0)
        return cls(commission_bps=default_comm, slippage_bps=per_leg - default_comm)
