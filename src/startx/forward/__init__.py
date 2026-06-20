"""Forward / paper-testing harness (Phase 6).

A *forward test* trains a directional model on a closed, strictly past window and then
walks **forward** through an out-of-sample window paper-trading the model's live signals,
printing a blotter of open and closed positions. The model NEVER sees a single bar of the
forward window — entries are taken at the *next* bar's open and exits at the triple-barrier
``t1``, so there is no lookahead anywhere in the loop.

Public surface:

* :class:`PaperTrade`   — one paper position (open or closed) with realised / mark-to-market P&L.
* :func:`forward_test`  — train-on-past, paper-trade-forward; returns blotter / equity / summary.
* :func:`live_signals`  — the "market-open" hook: train on a trailing window, score *today*.
"""
from __future__ import annotations

from .paper import PaperTrade, forward_test, live_signals

__all__ = ["PaperTrade", "forward_test", "live_signals"]
