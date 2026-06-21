"""Tests for the cost-sensitivity sweep (scripts/cost_sensitivity.py).

Core guarantee the task asks for: the sweep RUNS for every book and ``total_return`` is
monotone-decreasing in cost (more cost can never help). Plus unit coverage of the break-even and
Sharpe-decay helpers, and the new per-instrument cost model.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from startx.backtest.costs import CostModel, INSTRUMENT_ROUND_TRIP_BPS

# scripts/ is not an importable package — load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "cost_sensitivity", Path(__file__).resolve().parents[1] / "scripts" / "cost_sensitivity.py")
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)

# Skip the data-backed sweep tests cleanly if the price cache isn't present in this checkout.
_HAVE_DATA = (Path(__file__).resolve().parents[1] / "data" / "cache" / "prices" / "SPY.parquet").exists()


@pytest.fixture(scope="module")
def market():
    spy = cs._load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": cs._load("_VIX"), "vvix": cs._load("_VVIX"), "gld": cs._load("GLD")}
    return spy, aux


# --------------------------------------------------------------------------------------------- #
# The headline contract: the sweep runs and total_return is monotone-decreasing in cost.
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_DATA, reason="price cache not available")
@pytest.mark.parametrize("book", ["short_swing", "long_swing", "position"])
def test_sweep_runs_and_total_return_monotone_decreasing(market, book):
    spy, aux = market
    sweep = cs.sweep_book(book, spy, aux)

    # the sweep produced a row per swept cost, in ascending cost order
    assert list(sweep.index) == [float(c) for c in cs.COST_GRID]
    for col in ("total_return", "profit_factor", "ann_sharpe", "turnover"):
        assert col in sweep.columns

    # total_return must be non-increasing as cost rises (more cost never helps).
    tr = sweep["total_return"].to_numpy(dtype=float)
    diffs = np.diff(tr)
    assert np.all(diffs <= 1e-12), f"{book} total_return not monotone-decreasing in cost: {tr}"
    # and strictly lower at the most expensive cost than the cheapest (cost actually bites).
    assert tr[-1] < tr[0]


@pytest.mark.skipif(not _HAVE_DATA, reason="price cache not available")
@pytest.mark.parametrize("book", ["short_swing", "long_swing", "position"])
def test_sharpe_also_monotone_decreasing(market, book):
    sweep = cs.sweep_book(book, *market)
    sh = sweep["ann_sharpe"].to_numpy(dtype=float)
    assert np.all(np.diff(sh) <= 1e-9), f"{book} ann_sharpe not monotone in cost: {sh}"


@pytest.mark.skipif(not _HAVE_DATA, reason="price cache not available")
def test_turnover_orders_cost_sensitivity(market):
    """The higher-turnover swing books must bleed more Sharpe per bp than the position book."""
    spy, aux = market
    decay = {b: cs.sharpe_decay_per_bp(cs.sweep_book(b, spy, aux))
             for b in ("short_swing", "long_swing", "position")}
    # all decays are non-positive (Sharpe falls with cost)
    assert all(d <= 0 for d in decay.values())
    # the low-turnover position book is the LEAST cost-sensitive (flattest decay)
    assert abs(decay["position"]) < abs(decay["short_swing"])
    assert abs(decay["position"]) < abs(decay["long_swing"])


# --------------------------------------------------------------------------------------------- #
# break-even helper — exercised on synthetic monotone sweeps (no data needed).
# --------------------------------------------------------------------------------------------- #
def _mk_sweep(costs, totals, sharpes=None, turnover=1.0):
    sharpes = sharpes if sharpes is not None else [1.0] * len(costs)
    return pd.DataFrame(
        {"total_return": totals, "profit_factor": [2.0] * len(costs),
         "ann_sharpe": sharpes, "turnover": [turnover] * len(costs)},
        index=pd.Index([float(c) for c in costs], name="cost_bps"))


def test_breakeven_interpolates_within_bracket():
    # total_return crosses 0 between 10 and 20 bp: +0.10 -> -0.10 => zero at 15 bp.
    sweep = _mk_sweep([1, 2, 5, 10, 20], [0.40, 0.35, 0.25, 0.10, -0.10])
    assert cs.breakeven_cost(sweep) == pytest.approx(15.0)


def test_breakeven_robust_when_profitable_at_max_cost():
    sweep = _mk_sweep([1, 2, 5, 10, 20], [0.40, 0.35, 0.25, 0.18, 0.08])
    assert cs.breakeven_cost(sweep) == float("inf")


def test_breakeven_dead_when_unprofitable_at_min_cost():
    sweep = _mk_sweep([1, 2, 5, 10, 20], [-0.01, -0.05, -0.10, -0.20, -0.40])
    assert cs.breakeven_cost(sweep) == 0.0


def test_breakeven_exact_grid_crossing():
    # total_return hits exactly 0 at 10 bp -> break-even is 10 bp.
    sweep = _mk_sweep([1, 2, 5, 10, 20], [0.30, 0.20, 0.10, 0.00, -0.20])
    assert cs.breakeven_cost(sweep) == pytest.approx(10.0)


def test_sharpe_decay_slope_sign_and_magnitude():
    # ann_sharpe falls linearly 1.0 -> 0.62 over 1..20 bp: slope ~ -0.02 / bp.
    sharpes = [1.0, 0.98, 0.92, 0.82, 0.62]
    sweep = _mk_sweep([1, 2, 5, 10, 20], [0.4, 0.35, 0.25, 0.1, -0.1], sharpes=sharpes)
    slope = cs.sharpe_decay_per_bp(sweep)
    assert slope < 0
    assert slope == pytest.approx(-0.02, abs=2e-3)


# --------------------------------------------------------------------------------------------- #
# new reusable per-instrument cost model.
# --------------------------------------------------------------------------------------------- #
def test_for_instrument_spy_is_one_bp_round_trip():
    m = CostModel.for_instrument("SPY")
    assert m.round_trip_bps() == pytest.approx(INSTRUMENT_ROUND_TRIP_BPS["SPY"])
    assert m.round_trip_bps() == pytest.approx(1.0)


def test_for_instrument_case_insensitive_and_default():
    assert CostModel.for_instrument("spy").round_trip_bps() == pytest.approx(1.0)
    # unknown symbol falls back to the generic default
    assert (CostModel.for_instrument("ZZZZ").round_trip_bps()
            == pytest.approx(INSTRUMENT_ROUND_TRIP_BPS["_DEFAULT"]))


def test_for_instrument_default_splits_into_commission_plus_slippage():
    m = CostModel.for_instrument("_DEFAULT")  # 5 bp round-trip -> 2.5 bp/leg
    assert m.commission_bps == pytest.approx(1.0)   # default commission held
    assert m.slippage_bps == pytest.approx(1.5)     # remainder is slippage
    assert m.round_trip_bps() == pytest.approx(5.0)


def test_existing_costmodel_api_intact():
    # the original API/behaviour must be unchanged.
    assert np.isclose(CostModel(commission_bps=1.0, slippage_bps=5.0).round_trip_cost(), 0.0012)
    assert np.isclose(CostModel(0.0, 0.0).round_trip_cost(), 0.0)
