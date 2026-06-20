"""Self-learning meta-label + loss-autopsy layer.

This package turns *forensics of past trades* into a take/skip gate for new ones. It has three
stages, each in its own module:

* :mod:`startx.learning.autopsy` — attach entry-day microstructure forensics to each trade and
  rank which forensic features most cleanly separate winners from losers (the automated
  "why losses happen").
* :mod:`startx.learning.metalabel` — a *strictly point-in-time* walk-forward meta-model that, for
  each new trade, may only learn from trades that fully **resolved** before that trade's entry,
  then predicts P(win) and filters out setups that look like past losers.
* :mod:`startx.learning.loop` — the self-adjusting driver: run the walk-forward meta-label,
  compare filtered-vs-unfiltered trade economics, and monitor out-of-sample drift to flag when
  the gate needs retraining.

The whole point of the package is leakage discipline. The meta-model that decides take/skip for
trade ``i`` trains ONLY on trades whose label-end time ``t1 <= entry_time_i`` (resolved strictly
before ``i`` was entered); forensic features use ONLY entry-day-or-earlier data and never the
trade's own outcome. See the module docstrings for the firewall implementation.
"""
from __future__ import annotations

from .autopsy import autopsy_trades, win_loss_signature
from .loop import SelfLearningResult, drift_monitor, run_self_learning, trade_metrics
from .metalabel import apply_meta_filter, walk_forward_metalabel

__all__ = [
    "autopsy_trades",
    "win_loss_signature",
    "walk_forward_metalabel",
    "apply_meta_filter",
    "trade_metrics",
    "run_self_learning",
    "drift_monitor",
    "SelfLearningResult",
]
