"""Tests for the capacity / market-impact decay model.

Pins the contract: state the AUM at which the edge dies. We check the four pure
functions individually and then the headline behaviour — a tiny gross edge dies
at low AUM, a large gross edge survives to high AUM — using SPY-scale liquidity
(~$600 price, ~70M ADV).
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from qedge.config import get_config
from qedge.execution import capacity as cap

# SPY-scale liquidity for the realistic-magnitude cases.
_PRICE = 600.0
_ADV_SHARES = 7.0e7


# --------------------------------------------------------------------------------------------------
# participation
# --------------------------------------------------------------------------------------------------
def test_participation_scales_linearly_with_aum() -> None:
    p1 = cap.participation(1.0e8, _PRICE, _ADV_SHARES)
    p2 = cap.participation(2.0e8, _PRICE, _ADV_SHARES)
    p4 = cap.participation(4.0e8, _PRICE, _ADV_SHARES)
    assert np.isclose(p2, 2.0 * p1)
    assert np.isclose(p4, 4.0 * p1)


def test_participation_exact_value() -> None:
    # 1% of ADV: order shares = (price*adv*0.01)/price = adv*0.01 -> aum = price*adv*0.01.
    aum = _PRICE * _ADV_SHARES * 0.01
    assert np.isclose(cap.participation(aum, _PRICE, _ADV_SHARES), 0.01)


def test_participation_clips_at_one() -> None:
    # An AUM that demands 5x the day's volume still clips to a full 1.0 participation.
    aum = _PRICE * _ADV_SHARES * 5.0
    assert cap.participation(aum, _PRICE, _ADV_SHARES) == 1.0


def test_participation_nonpositive_inputs_are_zero() -> None:
    assert cap.participation(0.0, _PRICE, _ADV_SHARES) == 0.0
    assert cap.participation(1.0e8, 0.0, _ADV_SHARES) == 0.0
    assert cap.participation(1.0e8, _PRICE, 0.0) == 0.0


# --------------------------------------------------------------------------------------------------
# impact_bps
# --------------------------------------------------------------------------------------------------
def test_impact_zero_at_zero_participation() -> None:
    assert cap.impact_bps(0.0) == 0.0


def test_impact_increases_with_participation() -> None:
    parts = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    impacts = [cap.impact_bps(p) for p in parts]
    assert all(a < b for a, b in pairwise(impacts))


def test_impact_square_root_law() -> None:
    coef = get_config().capacity.impact_coefficient
    # impact = coef * sqrt(part) * 100; quadrupling participation doubles impact.
    assert np.isclose(cap.impact_bps(0.04), 2.0 * cap.impact_bps(0.01))
    assert np.isclose(cap.impact_bps(1.0), coef * 100.0)


def test_impact_coefficient_override() -> None:
    base = cap.impact_bps(0.25, coefficient=0.1)
    doubled = cap.impact_bps(0.25, coefficient=0.2)
    assert np.isclose(doubled, 2.0 * base)


# --------------------------------------------------------------------------------------------------
# net_edge_bps
# --------------------------------------------------------------------------------------------------
def test_net_edge_decreases_as_aum_grows() -> None:
    gross = 8.0
    aums = [1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10]
    nets = [cap.net_edge_bps(gross, a, price=_PRICE, adv_shares=_ADV_SHARES) for a in aums]
    assert all(a > b for a, b in pairwise(nets))


def test_net_edge_subtracts_fixed_floor_and_impact() -> None:
    fixed = get_config().execution.spy_cost_bps
    gross = 10.0
    aum = 1.0e8
    part = cap.participation(aum, _PRICE, _ADV_SHARES)
    expected = gross - fixed - cap.impact_bps(part)
    assert np.isclose(cap.net_edge_bps(gross, aum, price=_PRICE, adv_shares=_ADV_SHARES), expected)


def test_net_edge_custom_fixed_cost() -> None:
    gross = 10.0
    aum = 1.0e8
    part = cap.participation(aum, _PRICE, _ADV_SHARES)
    expected = gross - 5.0 - cap.impact_bps(part)
    got = cap.net_edge_bps(gross, aum, price=_PRICE, adv_shares=_ADV_SHARES, fixed_cost_bps=5.0)
    assert np.isclose(got, expected)


# --------------------------------------------------------------------------------------------------
# break_even_aum
# --------------------------------------------------------------------------------------------------
def test_break_even_net_edge_is_zero_at_ceiling() -> None:
    gross = 6.0
    be = cap.break_even_aum(gross, price=_PRICE, adv_shares=_ADV_SHARES)
    assert np.isfinite(be) and be > 0.0
    assert abs(cap.net_edge_bps(gross, be, price=_PRICE, adv_shares=_ADV_SHARES)) < 1.0e-6


def test_break_even_sign_change_straddles_ceiling() -> None:
    gross = 6.0
    be = cap.break_even_aum(gross, price=_PRICE, adv_shares=_ADV_SHARES)
    # Slightly below the ceiling the edge still pays; slightly above it is gone.
    assert cap.net_edge_bps(gross, be * 0.9, price=_PRICE, adv_shares=_ADV_SHARES) > 0.0
    assert cap.net_edge_bps(gross, be * 1.1, price=_PRICE, adv_shares=_ADV_SHARES) < 0.0


def test_break_even_zero_when_edge_underwater_at_floor() -> None:
    # Gross edge below the fixed cost floor: no AUM is profitable -> zero capacity.
    fixed = get_config().execution.spy_cost_bps
    be = cap.break_even_aum(fixed - 0.5, price=_PRICE, adv_shares=_ADV_SHARES)
    assert be == 0.0


def test_break_even_infinite_when_impact_never_bites() -> None:
    # A huge gross edge survives even being 100% of ADV -> unbounded capacity.
    be = cap.break_even_aum(500.0, price=_PRICE, adv_shares=_ADV_SHARES)
    assert be == float("inf")


def test_tiny_edge_dies_at_low_aum_large_edge_survives_high() -> None:
    fixed = get_config().execution.spy_cost_bps
    tiny = cap.break_even_aum(fixed + 0.1, price=_PRICE, adv_shares=_ADV_SHARES)
    large = cap.break_even_aum(8.0, price=_PRICE, adv_shares=_ADV_SHARES)
    assert np.isfinite(tiny) and np.isfinite(large)
    # The thin edge has a far smaller capacity ceiling than the fat one.
    assert tiny < large
    # And concretely: tiny dies in the low millions, large survives into the billions.
    assert tiny < 1.0e7
    assert large > 1.0e9


# --------------------------------------------------------------------------------------------------
# capacity_curve
# --------------------------------------------------------------------------------------------------
def test_capacity_curve_monotone_decreasing_in_aum() -> None:
    curve = cap.capacity_curve(8.0, price=_PRICE, adv_shares=_ADV_SHARES)
    assert list(curve.columns) == ["aum_usd", "participation", "impact_bps", "net_edge_bps"]
    assert curve["aum_usd"].is_monotonic_increasing
    # Net edge falls as AUM rises; impact rises.
    assert (curve["net_edge_bps"].diff().dropna() < 0).all()
    assert (curve["impact_bps"].diff().dropna() > 0).all()


def test_capacity_curve_uses_config_grid_by_default() -> None:
    grid = get_config().capacity.aum_grid_usd
    curve = cap.capacity_curve(8.0, price=_PRICE, adv_shares=_ADV_SHARES)
    assert list(curve["aum_usd"]) == [float(a) for a in grid]


def test_capacity_curve_custom_grid() -> None:
    grid = (1.0e6, 1.0e8, 1.0e10)
    curve = cap.capacity_curve(8.0, price=_PRICE, adv_shares=_ADV_SHARES, aum_grid=grid)
    assert list(curve["aum_usd"]) == list(grid)
    assert len(curve) == 3
