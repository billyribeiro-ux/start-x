"""Tests for :mod:`qedge.execution.costs` — equity execution realism.

Covers the contract claims:

* ``net_returns`` reduces a gross return by exactly the round-trip cost in bps;
* partial fills never exceed ``participation * bar_volume`` per bar AND conserve
  total quantity (filled + unfilled == order);
* the latency model reports the stated assumption (next-bar for daily, the
  millisecond figure for intraday).
"""
from __future__ import annotations

import numpy as np
import pytest

from qedge.config import QedgeConfig
from qedge.execution.costs import (
    EquityCostModel,
    LatencyModel,
    apply_partial_fills,
    atr_impact_bps,
    drift_adjusted_alpha_net,
    square_root_impact_bps,
)

_BPS = 1.0e4


@pytest.fixture()
def execution_cfg() -> QedgeConfig:
    return QedgeConfig()


# --- net_returns ----------------------------------------------------------


def test_net_returns_reduces_gross_by_round_trip_bps(execution_cfg: QedgeConfig) -> None:
    model = EquityCostModel.from_config(execution_cfg.execution)
    gross = 0.05
    net = model.net_returns(gross)
    exe = execution_cfg.execution
    expected_drag = 2.0 * (exe.commission_bps + exe.slippage_bps) / _BPS
    assert net.shape == (1,)
    assert net[0] == pytest.approx(gross - expected_drag)


def test_net_returns_matches_round_trip_cost_method() -> None:
    model = EquityCostModel(commission_bps=0.5, slippage_bps=1.0)
    # 2 * (0.5 + 1.0) = 3.0 bps round trip = 0.0003 fractional.
    assert model.round_trip_cost() == pytest.approx(3.0 / _BPS)
    assert model.cost_per_side() == pytest.approx(1.5 / _BPS)
    net = model.net_returns(0.01)
    assert net[0] == pytest.approx(0.01 - 0.0003)


def test_net_returns_scales_with_turnover() -> None:
    model = EquityCostModel(commission_bps=1.0, slippage_bps=2.0)
    rt = model.round_trip_cost()
    # Half turnover pays half the round-trip cost.
    half = model.net_returns(0.0, turnover=0.5)
    full = model.net_returns(0.0, turnover=1.0)
    assert half[0] == pytest.approx(-0.5 * rt)
    assert full[0] == pytest.approx(-rt)


def test_net_returns_vectorised() -> None:
    model = EquityCostModel(commission_bps=1.0, slippage_bps=1.0)
    rt = model.round_trip_cost()
    gross = np.array([0.01, -0.02, 0.0])
    net = model.net_returns(gross, turnover=np.array([1.0, 1.0, 0.0]))
    assert net == pytest.approx(np.array([0.01 - rt, -0.02 - rt, 0.0]))


def test_net_returns_rejects_negative_turnover() -> None:
    model = EquityCostModel(commission_bps=1.0, slippage_bps=1.0)
    with pytest.raises(ValueError, match="turnover"):
        model.net_returns(0.01, turnover=-0.5)


# --- partial fills --------------------------------------------------------


def test_partial_fills_never_exceed_participation_times_volume() -> None:
    volumes = [1_000.0, 2_000.0, 500.0, 4_000.0]
    participation = 0.1
    result = apply_partial_fills(10_000.0, volumes, participation=participation)
    caps = np.asarray(volumes) * participation
    assert np.all(result.fills <= caps + 1e-9)
    assert np.all(result.fills >= 0.0)


def test_partial_fills_conserve_total_quantity() -> None:
    volumes = [1_000.0, 2_000.0, 500.0, 4_000.0]
    order = 10_000.0
    result = apply_partial_fills(order, volumes, participation=0.1)
    assert result.fills.sum() + result.unfilled == pytest.approx(order)


def test_partial_fills_carry_remainder_no_instant_fill() -> None:
    # Tiny bars cannot absorb a big order; the remainder must be carried, not faked.
    volumes = [100.0, 100.0]
    result = apply_partial_fills(1_000.0, volumes, participation=0.1)
    # Each bar absorbs at most 10 shares -> 20 filled, 980 unfilled.
    assert result.fills.sum() == pytest.approx(20.0)
    assert result.unfilled == pytest.approx(980.0)
    assert not result.fully_filled


def test_partial_fills_complete_within_capacity() -> None:
    volumes = [10_000.0, 10_000.0]
    result = apply_partial_fills(500.0, volumes, participation=0.1)
    assert result.fully_filled
    assert result.unfilled == pytest.approx(0.0)
    assert result.fills.sum() == pytest.approx(500.0)
    # First bar can absorb 1000; the 500 order completes on bar 0.
    assert result.fills[0] == pytest.approx(500.0)
    assert result.fills[1] == pytest.approx(0.0)


def test_partial_fills_default_participation_from_config(execution_cfg: QedgeConfig) -> None:
    volumes = [10_000.0]
    result = apply_partial_fills(10_000.0, volumes)  # use cfg default
    cap = execution_cfg.execution.partial_fill_participation * 10_000.0
    assert result.fills[0] == pytest.approx(cap)


def test_partial_fills_reject_bad_participation() -> None:
    with pytest.raises(ValueError, match="participation"):
        apply_partial_fills(100.0, [1000.0], participation=0.0)
    with pytest.raises(ValueError, match="participation"):
        apply_partial_fills(100.0, [1000.0], participation=1.5)


def test_partial_fills_reject_negative_order() -> None:
    with pytest.raises(ValueError, match="orders"):
        apply_partial_fills(-1.0, [1000.0], participation=0.1)


# --- latency --------------------------------------------------------------


def test_latency_model_reports_daily_next_bar_assumption(execution_cfg: QedgeConfig) -> None:
    model = LatencyModel.from_config(execution_cfg.execution)
    stated = model.stated_latency(intraday=False)
    assert "next-bar" in stated
    assert "immaterial" in stated


def test_latency_model_reports_intraday_ms() -> None:
    model = LatencyModel(intraday_latency_ms=250.0)
    stated = model.stated_latency(intraday=True)
    assert "250" in stated
    assert "ms" in stated


def test_latency_model_carries_config_value(execution_cfg: QedgeConfig) -> None:
    model = LatencyModel.from_config(execution_cfg.execution)
    assert model.intraday_latency_ms == execution_cfg.execution.intraday_latency_ms


# --- thin startx wrappers -------------------------------------------------


def test_atr_impact_bps_scales_with_atr() -> None:
    low = atr_impact_bps(atr_pct=0.01, coef=0.5)
    high = atr_impact_bps(atr_pct=0.02, coef=0.5)
    assert high > low > 0.0


def test_square_root_impact_zero_without_shares() -> None:
    assert square_root_impact_bps(atr_pct=0.02, shares=0.0, volume=1_000.0, coef=0.1) == 0.0
    positive = square_root_impact_bps(atr_pct=0.02, shares=100.0, volume=1_000.0, coef=0.1)
    assert positive > 0.0


def test_drift_adjusted_alpha_net_subtracts_drift() -> None:
    returns = [0.02, 0.03]
    drift = [0.01, 0.01]
    # mean(0.02-0.01, 0.03-0.01) = mean(0.01, 0.02) = 0.015
    assert drift_adjusted_alpha_net(returns, drift) == pytest.approx(0.015)
