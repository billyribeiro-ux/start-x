"""Backtest layer — realistic out-of-sample equity curves for swing signals.

Phase 5 of the pipeline. Turns model probabilities plus triple-barrier label
outcomes into a sequential, no-lookahead, one-position-at-a-time equity curve
with transaction costs, then summarises it with standard performance metrics.

The contract is generic and offline (no network): it consumes the labeling
output schema (``date, t1, ret, label``) and per-entry probabilities, and reuses
:mod:`startx.validation.metrics` for the headline statistics.
"""
from __future__ import annotations

from startx.backtest.costs import CostModel
from startx.backtest.engine import (
    BacktestResult,
    backtest_signals,
    portfolio_backtest,
)
from startx.backtest.sizing import fixed_fraction, vol_target

__all__ = [
    "CostModel",
    "BacktestResult",
    "backtest_signals",
    "portfolio_backtest",
    "fixed_fraction",
    "vol_target",
]
