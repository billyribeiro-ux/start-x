"""Backtest-engine tests on deterministic synthetic signals & labels.

Every case is hand-computable: outcomes are read straight from the label rows
``(ret, t1)`` and never from future bars, so these tests also pin the
no-lookahead / one-position-at-a-time contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.backtest.costs import CostModel
from startx.backtest.engine import (
    BacktestResult,
    backtest_signals,
    portfolio_backtest,
)
from startx.backtest.sizing import fixed_fraction, vol_target


def _labels(entries, t1s, rets, labels=None) -> pd.DataFrame:
    """Triple-barrier-shaped label frame."""
    entries = pd.to_datetime(list(entries))
    t1s = pd.to_datetime(list(t1s))
    if labels is None:
        labels = [int(np.sign(r)) for r in rets]
    return pd.DataFrame(
        {"date": entries, "t1": t1s, "ret": list(rets), "label": list(labels)}
    )


def _preds(entries, probs) -> pd.DataFrame:
    """Predictions frame indexed by entry date with a ``prob_up`` column."""
    return pd.DataFrame({"prob_up": list(probs)}, index=pd.to_datetime(list(entries)))


def _nonoverlap_dates(n: int, hold: int = 3, start: str = "2022-01-03"):
    """``n`` entry dates spaced so that each trade's t1 precedes the next entry."""
    base = pd.bdate_range(start, periods=n * (hold + 2))
    entries = [base[i * (hold + 2)] for i in range(n)]
    t1s = [base[i * (hold + 2) + hold] for i in range(n)]
    return entries, t1s, base


# --------------------------------------------------------------------------- #
# Cost model & sizing                                                         #
# --------------------------------------------------------------------------- #
def test_cost_model_round_trip():
    c = CostModel(commission_bps=1.0, slippage_bps=5.0)
    # 2 * (1 + 5) / 1e4 = 12 bps = 0.0012
    assert np.isclose(c.round_trip_cost(), 0.0012)
    assert np.isclose(CostModel(0.0, 0.0).round_trip_cost(), 0.0)


def test_sizing_helpers():
    assert fixed_fraction(1000.0, 1.0) == 1000.0
    assert fixed_fraction(1000.0, 0.25) == 250.0
    assert fixed_fraction(1000.0, 2.0) == 1000.0  # clamped, no leverage
    assert np.isclose(vol_target(0.10, 0.20), 0.5)
    assert vol_target(0.10, 0.05, cap=1.0) == 1.0  # capped
    assert vol_target(0.10, 0.0) == 1.0  # zero realized vol -> cap


# --------------------------------------------------------------------------- #
# All-correct LONG signals                                                     #
# --------------------------------------------------------------------------- #
def test_all_correct_long_trades_are_profitable():
    entries, t1s, base = _nonoverlap_dates(4, hold=3)
    rets = [0.05, 0.04, 0.06, 0.03]  # all positive
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [1.0, 1.0, 1.0, 1.0])  # prob_up = 1 -> always long
    prices = pd.DataFrame({"date": base, "close": 100.0})

    res = backtest_signals(labels=labels, predictions=preds, prices=prices)
    assert isinstance(res, BacktestResult)
    assert res.metrics["n_trades"] == 4
    assert (res.trades["side"] == 1).all()
    assert res.metrics["hit_rate"] == 1.0
    assert res.metrics["total_return"] > 0
    # Net return per trade is gross minus the round-trip cost, still positive here.
    rt = CostModel().round_trip_cost()
    assert np.allclose(res.trades["ret_net"], np.array(rets) - rt)
    assert (res.trades["ret_net"] > 0).all()


# --------------------------------------------------------------------------- #
# SHORT signals                                                                #
# --------------------------------------------------------------------------- #
def test_short_trades_on_positive_ret_lose():
    entries, t1s, base = _nonoverlap_dates(3, hold=3)
    rets = [0.05, 0.04, 0.06]  # positive bars
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [0.0, 0.0, 0.0])  # prob_up = 0 -> always short
    res = backtest_signals(labels=labels, predictions=preds)

    assert res.metrics["n_trades"] == 3
    assert (res.trades["side"] == -1).all()
    # Short of a rising market: gross = -ret < 0, net even worse.
    assert np.allclose(res.trades["ret_gross"], -np.array(rets))
    assert (res.trades["ret_net"] < 0).all()
    assert res.metrics["hit_rate"] == 0.0
    assert res.metrics["total_return"] < 0


def test_short_trades_on_negative_ret_profit():
    entries, t1s, _ = _nonoverlap_dates(2, hold=3)
    rets = [-0.05, -0.03]  # falling market
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [0.0, 0.05])  # both <= short_th -> short
    res = backtest_signals(labels=labels, predictions=preds)
    assert (res.trades["side"] == -1).all()
    assert (res.trades["ret_gross"] > 0).all()  # short a decline -> profit
    assert res.metrics["total_return"] > 0


# --------------------------------------------------------------------------- #
# Mid-band probabilities -> no trades                                          #
# --------------------------------------------------------------------------- #
def test_mid_band_probs_produce_no_trades():
    entries, t1s, _ = _nonoverlap_dates(4, hold=3)
    rets = [0.05, -0.04, 0.06, -0.03]
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [0.5, 0.55, 0.45, 0.5])  # strictly inside (0.4, 0.6)
    res = backtest_signals(labels=labels, predictions=preds)
    assert res.metrics["n_trades"] == 0
    assert res.trades.empty
    assert res.metrics["exposure"] == 0.0
    # Flat account: equity constant at starting capital.
    assert np.allclose(res.equity.to_numpy(), 1.0)
    assert res.metrics["total_return"] == 0.0


# --------------------------------------------------------------------------- #
# Costs strictly reduce net vs gross                                           #
# --------------------------------------------------------------------------- #
def test_costs_strictly_reduce_net_returns():
    entries, t1s, _ = _nonoverlap_dates(3, hold=3)
    rets = [0.05, 0.04, 0.06]
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [1.0, 1.0, 1.0])

    with_costs = backtest_signals(
        labels=labels, predictions=preds, costs=CostModel(1.0, 5.0)
    )
    free = backtest_signals(labels=labels, predictions=preds, costs=CostModel(0.0, 0.0))

    assert (with_costs.trades["ret_net"] < with_costs.trades["ret_gross"]).all()
    assert np.allclose(free.trades["ret_net"], free.trades["ret_gross"])
    # Costs drag down total compounded return.
    assert with_costs.metrics["total_return"] < free.metrics["total_return"]
    rt = CostModel(1.0, 5.0).round_trip_cost()
    drag = with_costs.trades["ret_gross"] - with_costs.trades["ret_net"]
    assert np.allclose(drag, rt)


# --------------------------------------------------------------------------- #
# one_at_a_time prevents overlapping entries                                   #
# --------------------------------------------------------------------------- #
def test_one_at_a_time_prevents_overlap():
    # Dense daily entries whose t1 reaches several bars ahead -> overlapping if
    # not gated. Each trade holds 5 bdays; entries are every single bday.
    base = pd.bdate_range("2022-01-03", periods=20)
    entries = list(base[:15])
    t1s = [base[i + 5] for i in range(15)]
    rets = [0.01] * 15
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [1.0] * 15)  # all want to go long

    res = backtest_signals(labels=labels, predictions=preds, one_at_a_time=True)
    tr = res.trades.sort_values("entry_date").reset_index(drop=True)
    # Each subsequent entry must be strictly after the previous trade's exit (t1).
    for i in range(1, len(tr)):
        assert tr.loc[i, "entry_date"] > tr.loc[i - 1, "exit_date"]
    # Without gating, every candidate entry trades (overlap allowed).
    res_overlap = backtest_signals(labels=labels, predictions=preds, one_at_a_time=False)
    assert res_overlap.metrics["n_trades"] == 15
    assert res.metrics["n_trades"] < res_overlap.metrics["n_trades"]


# --------------------------------------------------------------------------- #
# Metric sanity                                                                #
# --------------------------------------------------------------------------- #
def test_metrics_are_sane():
    entries, t1s, base = _nonoverlap_dates(5, hold=3)
    rets = [0.05, -0.03, 0.04, -0.02, 0.06]
    labels = _labels(entries, t1s, rets)
    preds = _preds(entries, [1.0, 0.0, 1.0, 0.0, 1.0])
    prices = pd.DataFrame({"date": base, "close": 100.0})
    res = backtest_signals(labels=labels, predictions=preds, prices=prices)

    m = res.metrics
    # All required keys present.
    expected = {
        "n_trades", "hit_rate", "avg_ret_net", "total_return", "cagr", "sharpe",
        "sortino", "max_drawdown", "profit_factor", "exposure", "avg_holding_days",
    }
    assert set(m.keys()) == expected
    assert m["max_drawdown"] <= 0.0
    assert 0.0 <= m["exposure"] <= 1.0
    assert 0.0 <= m["hit_rate"] <= 1.0
    assert m["n_trades"] == 5
    assert m["avg_holding_days"] > 0
    # Equity curve aligns to the price calendar and starts at capital.
    assert len(res.equity) == len(prices)
    assert res.equity.iloc[0] == 1.0


def test_equity_steps_at_exit_not_entry():
    # A single trade: equity must stay flat until the exit date, then step.
    entries = [pd.Timestamp("2022-01-03")]
    t1s = [pd.Timestamp("2022-01-10")]
    base = pd.bdate_range("2022-01-03", "2022-01-14")
    labels = _labels(entries, t1s, [0.10])
    preds = _preds(entries, [1.0])
    prices = pd.DataFrame({"date": base, "close": 100.0})
    res = backtest_signals(labels=labels, predictions=preds, prices=prices)
    eq = res.equity
    # Flat at 1.0 up to (and including) the day before exit.
    assert np.allclose(eq.loc[: pd.Timestamp("2022-01-07")].to_numpy(), 1.0)
    # Stepped up on/after the exit date.
    rt = CostModel().round_trip_cost()
    assert np.isclose(eq.loc[pd.Timestamp("2022-01-10")], 1.0 * (1 + 0.10 - rt))
    assert np.isclose(eq.iloc[-1], 1.0 * (1 + 0.10 - rt))


def test_y_prob_fallback_column():
    entries, t1s, _ = _nonoverlap_dates(2, hold=3)
    labels = _labels(entries, t1s, [0.05, 0.04])
    preds = pd.DataFrame({"y_prob": [1.0, 1.0]}, index=pd.to_datetime(entries))
    res = backtest_signals(labels=labels, predictions=preds)
    assert res.metrics["n_trades"] == 2
    assert (res.trades["side"] == 1).all()


# --------------------------------------------------------------------------- #
# Portfolio                                                                    #
# --------------------------------------------------------------------------- #
def test_portfolio_equal_weight_blend():
    entries, t1s, base = _nonoverlap_dates(3, hold=3)
    prices = pd.DataFrame({"date": base, "close": 100.0})
    a = backtest_signals(
        labels=_labels(entries, t1s, [0.05, 0.04, 0.06]),
        predictions=_preds(entries, [1.0, 1.0, 1.0]),
        prices=prices,
    )
    b = backtest_signals(
        labels=_labels(entries, t1s, [-0.05, -0.04, -0.06]),
        predictions=_preds(entries, [0.0, 0.0, 0.0]),  # short a decline -> profit
        prices=prices,
    )
    port = portfolio_backtest({"A": a, "B": b})
    assert isinstance(port, BacktestResult)
    # All trades concatenated.
    assert port.metrics["n_trades"] == a.metrics["n_trades"] + b.metrics["n_trades"]
    assert len(port.trades) == 6
    # Blended curve sits between the two component normalised curves at the end.
    assert port.metrics["max_drawdown"] <= 0.0
    assert 0.0 <= port.metrics["exposure"] <= 1.0


def test_portfolio_empty_input():
    port = portfolio_backtest({})
    assert port.metrics["n_trades"] == 0
    assert port.trades.empty
