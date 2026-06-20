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
