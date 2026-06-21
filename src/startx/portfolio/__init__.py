"""Portfolio construction layer — vol-sized, risk-managed, concurrent SPY swing book.

Exposes :func:`run_portfolio`, which composes validated single-instrument sleeves
(momentum breakout, IBS mean-reversion, VRP/VVIX volatility-premium, VIX capitulation)
into one shared equity curve with a unified chandelier exit, volatility-target sizing,
concurrent risk/gross caps, and the gold-calm + drawdown-breaker risk overlays.
"""
from .engine import (
    LEDGER_COLUMNS,
    PortfolioResult,
    SleeveSig,
    atr,
    run_portfolio,
)

__all__ = [
    "run_portfolio",
    "PortfolioResult",
    "SleeveSig",
    "LEDGER_COLUMNS",
    "atr",
]
