"""Validation layer — the anti-overfitting firewall.

This subsystem implements leakage-free cross-validation and out-of-sample
performance vetting for financial ML, following López de Prado's *Advances in
Financial Machine Learning* (Ch. 7-8, 11-12) and the Bailey/López de Prado
work on the Deflated Sharpe Ratio and the Probability of Backtest Overfitting.

The contract is generic: everything operates on ``X`` (DataFrame/ndarray),
``y`` (Series) and ``t1`` (a ``pd.Series`` mapping each sample's *start* time
to its label's *end* time). No dependency on feature/label generation.
"""
from __future__ import annotations

from startx.validation.cpcv import CombinatorialPurgedCV
from startx.validation.metrics import (
    cagr,
    deflated_sharpe,
    deflated_sharpe_ratio,
    expectancy,
    hit_rate,
    max_drawdown,
    probability_of_backtest_overfitting,
    profit_factor,
    sharpe,
    sortino,
)
from startx.validation.purged_cv import PurgedKFold
from startx.validation.report import scoreboard
from startx.validation.walkforward import walk_forward_predict

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "walk_forward_predict",
    "scoreboard",
    "sharpe",
    "sortino",
    "max_drawdown",
    "cagr",
    "profit_factor",
    "expectancy",
    "hit_rate",
    "deflated_sharpe_ratio",
    "deflated_sharpe",
    "probability_of_backtest_overfitting",
]
