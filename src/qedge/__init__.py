"""qedge — a leakage-hardened quant-research scanner.

A self-contained package implementing the Quant Research Operating Contract
(Lopez de Prado / Chan standard): point-in-time correctness, triple-barrier +
meta-labeling, fractional differentiation, CPCV + embargo, Deflated Sharpe and
Probability of Backtest Overfitting on every edge, MDA / clustered-SHAP feature
importance, HMM / change-point regime detection, online learning with concept-
drift detection, and realistic execution + capacity modelling.

Governing philosophy: *every strong backtest is guilty of leakage until proven
innocent.* The package reuses proven methodology modules from the sibling
``startx`` package where they already meet the contract, and implements the rest
fresh.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
