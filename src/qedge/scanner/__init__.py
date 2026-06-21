"""qedge scanner — the adversarial edge-discovery harness.

Wires every layer together (data -> features -> regime -> triple-barrier labels
-> sample weights -> cross-validated model -> OOS returns -> execution costs ->
capacity) and refuses to call an edge real until it clears the survival gate
(Deflated Sharpe AND Probability of Backtest Overfitting), with multiple-testing
trials accounted for. Every strong backtest is guilty of leakage until proven
innocent.
"""
from __future__ import annotations
