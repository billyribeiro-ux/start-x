"""The anti-overfitting scoreboard.

A single function, :func:`scoreboard`, condenses an out-of-sample return stream
(plus optional CSCV performance matrices) into a verdict on whether a strategy
has a *real* edge or is most likely an artefact of backtest overfitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.validation.metrics import (
    cagr,
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    probability_of_backtest_overfitting,
    profit_factor,
    sharpe,
)

# DSR is a probability that the true Sharpe is positive. The conventional bar
# for "this edge is unlikely to be a fluke" is 95%. We expose the verdict on a
# centred scale: dsr_centred = DSR_prob - 0.95, so dsr_centred > 0 means the
# strategy clears the 95% confidence bar.
DSR_CONFIDENCE = 0.95
PBO_THRESHOLD = 0.5


def scoreboard(
    strategy_returns_oos: pd.Series,
    n_trials: int,
    perf_is: np.ndarray | None = None,
    perf_oos: np.ndarray | None = None,
    periods: int = 252,
) -> dict:
    """Summarise OOS performance and deliver an overfitting verdict.

    Parameters
    ----------
    strategy_returns_oos:
        Out-of-sample per-period strategy returns (NOT in-sample).
    n_trials:
        Number of independent configurations/strategies tried to find this one.
        Drives the Deflated Sharpe Ratio's selection-bias correction.
    perf_is, perf_oos:
        Optional aligned ``(n_splits, n_configs)`` CSCV performance matrices.
        When both are provided, the Probability of Backtest Overfitting (PBO)
        is computed; otherwise ``pbo`` is ``None``.
    periods:
        Periods per year for annualising the Sharpe (252 = daily).

    Returns
    -------
    dict
        Keys: ``sharpe``, ``deflated_sharpe``, ``pbo``, ``cagr``,
        ``max_drawdown``, ``profit_factor``, ``hit_rate``, ``n_obs``,
        ``verdict``.

    Verdict rule
    ------------
    ``deflated_sharpe`` in the result is *centred*: it equals
    ``DSR_prob - 0.95``, so a positive value means the strategy clears the 95%
    confidence threshold that its true Sharpe is positive. The verdict is::

        "EDGE SURVIVES"  if deflated_sharpe > 0.0  (DSR prob > 0.95)
                         and (pbo is None or pbo < 0.5)
        "LIKELY OVERFIT" otherwise

    In words: an edge survives only when it is statistically distinguishable
    from the best of ``n_trials`` lucky backtests *and* (if measurable) the
    in-sample-best configuration does not systematically collapse out-of-sample.
    """
    r = pd.Series(strategy_returns_oos).astype(float).dropna()
    n_obs = int(r.size)

    sr = sharpe(r, periods=periods)
    dsr_prob = deflated_sharpe(r, n_trials=n_trials)
    dsr_centred = (dsr_prob - DSR_CONFIDENCE) if not np.isnan(dsr_prob) else float("nan")

    pbo: float | None = None
    if perf_is is not None and perf_oos is not None:
        pbo = probability_of_backtest_overfitting(perf_is, perf_oos)

    dsr_ok = (not np.isnan(dsr_centred)) and dsr_centred > 0.0
    pbo_ok = pbo is None or pbo < PBO_THRESHOLD
    verdict = "EDGE SURVIVES" if (dsr_ok and pbo_ok) else "LIKELY OVERFIT"

    return {
        "sharpe": sr,
        "deflated_sharpe": dsr_centred,
        "deflated_sharpe_prob": dsr_prob,
        "pbo": pbo,
        "cagr": cagr(r, periods=periods),
        "max_drawdown": max_drawdown(r, is_returns=True),
        "profit_factor": profit_factor(r),
        "hit_rate": hit_rate(r),
        "n_obs": n_obs,
        "verdict": verdict,
    }
