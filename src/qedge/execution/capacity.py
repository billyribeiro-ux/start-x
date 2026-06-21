"""Capacity / market-impact decay — the AUM at which the edge dies.

The contract is blunt: a backtest's drift-adjusted alpha is a *small-money*
number. As the book scales, every entry and exit demands a larger share of the
day's liquidity, and market impact eats the edge. This module answers the only
capacity question that matters for sizing the desk: **at what AUM does the
strategy's net edge cross zero?** Above that ceiling the alpha is gone — you are
paying the market to take a position that no longer pays you back.

The impact law
--------------
We reuse the Almgren-style **square-root** shape that
:class:`startx.portfolio.execution.SquareRootImpact` is built on:

    impact_bps = coefficient * sqrt(participation) * 100

where ``participation`` is the fraction of average daily volume (ADV) the order
represents. The square-root form is the standard empirical impact law: doubling
participation costs ~1.41* the impact, not 2* — large orders are sub-linear but
relentlessly increasing. The trailing ``* 100`` puts the charge on a basis-point
scale: with the config ``coefficient = 0.1``, being 1% of ADV costs ~1 bp and
being the entire day's volume costs ~10 bps — the same O(10-bps) magnitude the
startx model produces from ``coef * sqrt(participation) * ATR%`` at a typical
~1% ATR (where its ``vol_bps = ATR% * 100`` is O(1)).

Why implement the law here instead of instantiating ``SquareRootImpact``? The
startx class couples impact to a per-bar **ATR%** volatility input
(``coef * sqrt(participation) * ATR%``) and reads participation from a fixed
``shares``/``volume`` pair. At the *capacity-curve* level we model participation
directly as a function of AUM and have no per-bar ATR to feed it, so we keep the
identical square-root-of-participation shape but express it as a clean bps charge
driven only by participation and the config coefficient. The shape — and the
monotone-increasing, sub-linear behaviour the break-even solver relies on — is
the same.

Cost stack
----------
``net_edge_bps`` = gross edge - fixed cost floor - square-root impact. The fixed
floor is ``cfg.execution.spy_cost_bps`` (the round-trip equity friction already
priced into every sleeve); impact is the *additional*, AUM-dependent charge. The
break-even AUM is the root of ``net_edge_bps(aum) == 0``: impact is monotone
increasing in AUM (more dollars -> larger ADV share -> larger sqrt impact, until
participation saturates at 1.0), so net edge is monotone *decreasing* in AUM and
a single sign change exists wherever the strategy starts profitable and ends
unprofitable. We locate it by robust bisection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qedge.config import get_config

#: ADV-participation is a fraction in ``[0, 1]`` — you cannot trade more than the
#: whole day's volume in a single name.
_PARTICIPATION_FLOOR = 0.0
_PARTICIPATION_CEIL = 1.0

#: Bisection controls for :func:`break_even_aum`. The bracket spans
#: well-below-$1 to a planet-sized AUM so the root is always enclosed when one
#: exists; tolerance and iteration cap make the solve deterministic.
_BISECTION_MAX_ITER = 200
_BISECTION_TOL_BPS = 1.0e-9
_BRACKET_LOW_USD = 1.0
_BRACKET_HIGH_USD = 1.0e15


def participation(aum: float, price: float, adv_shares: float) -> float:
    """Fraction of average daily volume an ``aum`` order represents, clipped to ``[0, 1]``.

    The order is ``aum / price`` shares; its participation is that divided by
    ``adv_shares`` (average daily volume in shares). Clipped to ``[0, 1]`` — you
    can demand at most the whole day's liquidity. A non-positive ``price`` or
    ``adv_shares`` is undefined liquidity and returns ``0.0`` (no tradable
    participation rather than a divide-by-zero blow-up).
    """
    if price <= 0.0 or adv_shares <= 0.0 or aum <= 0.0:
        return _PARTICIPATION_FLOOR
    order_shares = float(aum) / float(price)
    frac = order_shares / float(adv_shares)
    return float(np.clip(frac, _PARTICIPATION_FLOOR, _PARTICIPATION_CEIL))


def impact_bps(participation: float, *, coefficient: float | None = None) -> float:
    """Square-root market impact in basis points for a given ADV ``participation``.

    ``impact_bps = coefficient * sqrt(participation) * 100`` (the Almgren-style
    law shared with :class:`startx.portfolio.execution.SquareRootImpact`).
    ``coefficient`` defaults to ``cfg.capacity.impact_coefficient``. Zero
    participation costs nothing; impact rises with the square root of
    participation (sub-linear but monotone increasing), saturating at
    ``coefficient * 100`` bps when the order is the entire day's volume.
    """
    coef = get_config().capacity.impact_coefficient if coefficient is None else coefficient
    p = float(np.clip(participation, _PARTICIPATION_FLOOR, _PARTICIPATION_CEIL))
    if p <= 0.0:
        return 0.0
    return float(coef) * float(np.sqrt(p)) * 100.0


def net_edge_bps(gross_edge_bps: float, aum: float, *, price: float, adv_shares: float,
                 fixed_cost_bps: float | None = None) -> float:
    """Gross edge net of the fixed cost floor and AUM-dependent square-root impact, in bps.

    ``net = gross_edge_bps - fixed_cost_bps - impact_bps(participation(aum))``.
    ``fixed_cost_bps`` defaults to ``cfg.execution.spy_cost_bps`` (the round-trip
    equity friction every sleeve already pays). As AUM grows the participation —
    and therefore the impact term — rises, so the net edge falls monotonically.
    """
    fixed = get_config().execution.spy_cost_bps if fixed_cost_bps is None else fixed_cost_bps
    part = participation(aum, price, adv_shares)
    return float(gross_edge_bps) - float(fixed) - impact_bps(part)


def break_even_aum(gross_edge_bps: float, *, price: float, adv_shares: float,
                   fixed_cost_bps: float | None = None) -> float:
    """The capacity ceiling: the AUM where ``net_edge_bps`` crosses zero.

    Above this AUM the drift-adjusted alpha is gone — impact plus fixed costs
    exceed the gross edge. We solve ``net_edge_bps(aum) == 0`` by bisection over
    a wide bracket, relying on the documented monotonicity: impact is monotone
    increasing in AUM (more dollars -> larger ADV share -> larger sqrt impact)
    until participation saturates at 1.0, so ``net_edge_bps`` is monotone
    *decreasing* in AUM and at most one root exists.

    Edge cases (no interior crossing):

    * If the edge is already negative at infinitesimal AUM (the fixed cost floor
      alone sinks it), the capacity is ``0.0`` — there is no size at which it
      pays.
    * If the edge is *still positive* at the maximum participation (a tiny order
      relative to ADV could never be the whole book, yet even being 100% of ADV
      does not kill it), the strategy never breaks even within the bracket and we
      return ``+inf`` — capacity is effectively unbounded by impact.
    """
    lo, hi = _BRACKET_LOW_USD, _BRACKET_HIGH_USD

    def f(aum: float) -> float:
        return net_edge_bps(gross_edge_bps, aum, price=price, adv_shares=adv_shares,
                            fixed_cost_bps=fixed_cost_bps)

    f_lo = f(lo)
    f_hi = f(hi)

    # Already underwater at the smallest size: the fixed floor (and the
    # infinitesimal impact) outweigh the gross edge -> zero capacity.
    if f_lo <= 0.0:
        return 0.0
    # Still profitable even being the entire day's volume -> impact never bites
    # hard enough; capacity is unbounded by this model.
    if f_hi > 0.0:
        return float("inf")

    for _ in range(_BISECTION_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= _BISECTION_TOL_BPS:
            return float(mid)
        # f is decreasing: positive net edge means we are still below the ceiling.
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def capacity_curve(gross_edge_bps: float, *, price: float, adv_shares: float,
                   aum_grid: tuple[float, ...] | None = None) -> pd.DataFrame:
    """Net edge (bps) across an AUM grid — the reporting view of capacity decay.

    One row per AUM in ``aum_grid`` (default ``cfg.capacity.aum_grid_usd``), with
    the participation, impact, and net edge at that AUM. Because impact rises
    monotonically with AUM, the ``net_edge_bps`` column is monotone decreasing
    down the grid — the visual statement of where the edge dies.
    """
    grid = get_config().capacity.aum_grid_usd if aum_grid is None else aum_grid
    rows: list[dict[str, float]] = []
    for aum in grid:
        part = participation(aum, price, adv_shares)
        rows.append({
            "aum_usd": float(aum),
            "participation": part,
            "impact_bps": impact_bps(part),
            "net_edge_bps": net_edge_bps(gross_edge_bps, aum, price=price,
                                         adv_shares=adv_shares),
        })
    return pd.DataFrame(rows, columns=["aum_usd", "participation", "impact_bps", "net_edge_bps"])
