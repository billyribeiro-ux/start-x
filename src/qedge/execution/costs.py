"""Equity execution realism — costs, partial fills, and latency.

The qedge contract is firm: **no Sharpe is reported until execution friction is
modelled.** This module is the equity half of that contract (the options half
lives in :mod:`qedge.execution.options`). It is a thin, typed wrapper around the
proven ``startx.portfolio.execution`` / ``startx.backtest.costs`` layer so the
two packages charge friction identically; qedge merely sources every knob from
:class:`qedge.config.ExecutionConfig` (the contract bans magic numbers).

Three pieces:

* :class:`EquityCostModel` — commission + slippage charged **per side**, applied
  round-trip to a gross return via :meth:`EquityCostModel.net_returns`.
* :func:`apply_partial_fills` — caps each bar's fill at
  ``participation * bar_volume`` and carries the remainder to later bars. It
  never fabricates an instant fill of the full order against a thin bar.
* :class:`LatencyModel` — carries the stated intraday latency. Latency only
  bites intraday: a daily strategy decides on the close and fills next bar, so
  the honest assumption is *next-bar* and the millisecond figure is irrelevant.

All costs are in basis points (1 bp = 0.0001). Per-side commission and slippage
each apply on the entry **and** the exit leg, so the round-trip drag is
``2 * (commission_bps + slippage_bps) / 1e4``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from qedge.config import ExecutionConfig, get_config
from startx.backtest.costs import CostModel
from startx.portfolio.execution import (
    ATRImpact,
    SquareRootImpact,
    drift_adjusted_alpha,
)

__all__ = [
    "EquityCostModel",
    "LatencyModel",
    "PartialFillResult",
    "apply_partial_fills",
    "atr_impact_bps",
    "drift_adjusted_alpha_net",
    "square_root_impact_bps",
]

_BPS: float = 1.0e4  # basis points per unit (1.0 = 10_000 bps)


# ==================================================================================================
# Equity cost model (commission + slippage), per-side, applied round-trip
# ==================================================================================================
@dataclass(frozen=True)
class EquityCostModel:
    """Per-side commission + slippage in bps, charged round-trip on a gross return.

    Wraps :class:`startx.backtest.costs.CostModel` so qedge and startx charge
    friction by the same arithmetic. Build it from :class:`ExecutionConfig` via
    :meth:`from_config`; the defaults read the process-wide config singleton so a
    bare ``EquityCostModel()`` already reflects ``cfg.execution``.

    Parameters
    ----------
    commission_bps:
        Broker commission per leg, in basis points (``cfg.execution.commission_bps``).
    slippage_bps:
        Expected slippage per leg, in basis points (``cfg.execution.slippage_bps``).
    """

    commission_bps: float
    slippage_bps: float

    @classmethod
    def from_config(cls, execution: ExecutionConfig | None = None) -> EquityCostModel:
        """Build from ``cfg.execution`` (the process singleton if not supplied)."""
        exe = execution if execution is not None else get_config().execution
        return cls(
            commission_bps=float(exe.commission_bps),
            slippage_bps=float(exe.slippage_bps),
        )

    def _startx_model(self) -> CostModel:
        """The equivalent startx :class:`CostModel` (single source of the arithmetic)."""
        return CostModel(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )

    def cost_per_side(self) -> float:
        """Fractional cost charged on a **single** leg (entry or exit)."""
        return (self.commission_bps + self.slippage_bps) / _BPS

    def round_trip_cost(self) -> float:
        """Fractional cost charged once per round-trip trade (entry + exit)."""
        return float(self._startx_model().round_trip_cost())

    def net_returns(
        self,
        gross: float | Sequence[float] | npt.NDArray[np.float64] | pd.Series,
        *,
        turnover: float | Sequence[float] | npt.NDArray[np.float64] | pd.Series = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Net a gross return by the round-trip cost, scaled by ``turnover``.

        ``turnover`` is the fraction of a full round trip incurred (``1.0`` =
        enter and exit a full unit). Per-period (e.g. daily) returns whose
        position barely turns over pay proportionally less:

            net = gross - turnover * round_trip_cost.

        Both arguments broadcast; the result is always a 1-D float array (a
        scalar in becomes a length-1 array) so callers get a stable type.
        """
        g = np.atleast_1d(np.asarray(gross, dtype=np.float64))
        t = np.atleast_1d(np.asarray(turnover, dtype=np.float64))
        if np.any(t < 0.0):
            raise ValueError("turnover must be non-negative")
        drag = t * self.round_trip_cost()
        return np.asarray(g - drag, dtype=np.float64)


# ==================================================================================================
# Partial fills — capped at participation * bar volume, remainder carried forward
# ==================================================================================================
@dataclass(frozen=True)
class PartialFillResult:
    """Outcome of working an order across bars under a participation cap.

    Attributes
    ----------
    fills:
        Per-bar filled quantity (same length as ``bar_volume``); each entry is
        ``<= participation * bar_volume[i]`` and ``>= 0``.
    unfilled:
        Quantity still outstanding after the last bar (``0`` if the order
        completed within the supplied bars). An honest carry — never a
        fabricated instant fill.
    fully_filled:
        ``True`` iff ``unfilled == 0``.
    """

    fills: npt.NDArray[np.float64]
    unfilled: float
    fully_filled: bool


def apply_partial_fills(
    orders: float,
    bar_volume: Sequence[float] | npt.NDArray[np.float64] | pd.Series,
    *,
    participation: float | None = None,
) -> PartialFillResult:
    """Work an order of ``orders`` shares across bars, capped per bar by participation.

    Each bar can absorb at most ``participation * bar_volume[i]`` shares; the
    unfilled remainder is carried to the next bar (and the next), so a large
    order against thin bars completes over **multiple** bars rather than
    instantly. If the bars cannot absorb the whole order, the leftover is
    reported in :attr:`PartialFillResult.unfilled` — the layer never fabricates
    an instant fill that the tape could not support.

    Parameters
    ----------
    orders:
        Order size in shares (non-negative; this layer sizes one side at a time).
    bar_volume:
        Per-bar traded volume, in shares, in chronological order.
    participation:
        Maximum share of each bar's volume the order may take. Defaults to
        ``cfg.execution.partial_fill_participation``. Must be in ``(0, 1]``.

    Returns
    -------
    PartialFillResult
        Per-bar fills (each ``<= participation * volume``), the unfilled
        remainder, and whether the order completed. Total filled + unfilled
        equals ``orders`` exactly (quantity is conserved).
    """
    if orders < 0:
        raise ValueError("orders must be non-negative")
    part = (
        float(participation)
        if participation is not None
        else float(get_config().execution.partial_fill_participation)
    )
    if not (0.0 < part <= 1.0):
        raise ValueError("participation must be in (0, 1]")

    volumes = np.asarray(bar_volume, dtype=np.float64)
    fills = np.zeros(volumes.shape, dtype=np.float64)
    remaining = float(orders)
    for i, vol in enumerate(volumes):
        if remaining <= 0.0:
            break
        capacity = max(0.0, part * float(vol))  # cap per bar; never exceeds it
        take = min(remaining, capacity)
        fills[i] = take
        remaining -= take
    remaining = max(0.0, remaining)  # guard tiny negative float drift
    return PartialFillResult(
        fills=fills,
        unfilled=float(remaining),
        fully_filled=bool(remaining <= 0.0),
    )


# ==================================================================================================
# Latency — only bites intraday
# ==================================================================================================
@dataclass(frozen=True)
class LatencyModel:
    """Carries the stated intraday latency assumption (``cfg.execution.intraday_latency_ms``).

    Latency is **only** material intraday. A daily strategy decides on the bar's
    close and cannot transact before the next bar, so its honest fill assumption
    is *next-bar* regardless of any millisecond figure; the latency value is
    documentation that becomes load-bearing only when an intraday strategy ships.
    """

    intraday_latency_ms: float

    @classmethod
    def from_config(cls, execution: ExecutionConfig | None = None) -> LatencyModel:
        """Build from ``cfg.execution`` (the process singleton if not supplied)."""
        exe = execution if execution is not None else get_config().execution
        return cls(intraday_latency_ms=float(exe.intraday_latency_ms))

    def stated_latency(self, *, intraday: bool = False) -> str:
        """Describe the latency assumption for a daily vs. an intraday strategy.

        Daily strategies fill next-bar (latency immaterial); intraday strategies
        state the modelled latency in milliseconds.
        """
        if not intraday:
            return (
                "daily strategy: next-bar fill; intraday latency immaterial "
                f"(stated {self.intraday_latency_ms:g} ms, unused for daily)"
            )
        return f"intraday strategy: {self.intraday_latency_ms:g} ms latency before fill"


# ==================================================================================================
# Thin typed wrappers over startx market-impact + drift-adjusted alpha
# ==================================================================================================
def atr_impact_bps(*, atr_pct: float, coef: float | None = None) -> float:
    """Volatility-scaled (ATR%) market impact in bps, via :class:`startx...ATRImpact`.

    ``coef`` defaults to ``cfg.capacity.impact_coefficient`` (no magic number).
    """
    c = float(coef) if coef is not None else float(get_config().capacity.impact_coefficient)
    return float(ATRImpact(coef=c).bps(atr_pct=atr_pct))


def square_root_impact_bps(
    *,
    atr_pct: float,
    shares: float,
    volume: float,
    coef: float | None = None,
) -> float:
    """Almgren square-root impact in bps, via :class:`startx...SquareRootImpact`.

    ``coef`` defaults to ``cfg.capacity.impact_coefficient`` (no magic number).
    """
    c = float(coef) if coef is not None else float(get_config().capacity.impact_coefficient)
    model = SquareRootImpact(coef=c, shares=float(shares))
    return float(model.bps(atr_pct=atr_pct, volume=volume))


def drift_adjusted_alpha_net(
    trade_returns: Sequence[float] | npt.NDArray[np.float64] | pd.Series,
    drift: Sequence[float] | npt.NDArray[np.float64] | pd.Series,
) -> float:
    """Drift-adjusted alpha (per-trade net return - same-window buy-and-hold drift).

    Thin typed wrapper over :func:`startx.portfolio.execution.drift_adjusted_alpha`
    so qedge code keeps a single import surface for the honest-edge metric.
    """
    return float(drift_adjusted_alpha(trade_returns, drift))
