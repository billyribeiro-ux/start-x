"""Unified, typed validation API — the qedge anti-overfitting survival gate.

This module binds the qedge ``cfg.validation`` presets onto the proven
``startx.validation`` primitives and exposes the firewall callers actually need:

* :func:`make_cpcv` — build a :class:`startx.validation.cpcv.CombinatorialPurgedCV`
  straight from config (``cpcv_n_groups``, ``cpcv_n_test_groups``, ``embargo_pct``);
* :func:`survival_gate` — the ship/no-ship verdict. An edge survives only if its
  Deflated Sharpe Ratio clears ``cfg.validation.dsr_min`` AND its Probability of
  Backtest Overfitting is at or below ``cfg.validation.pbo_max``. DSR is deflated
  by ``n_trials`` (the number of configurations searched, from
  :class:`TrialLedger`) so a wide search is penalised exactly as it should be.

The :class:`TrialLedger` is re-exported here so the validation layer is the single
import surface for the overfitting firewall.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd

from qedge.config import get_config
from qedge.validation.trials import TrialLedger
from startx.validation.cpcv import CombinatorialPurgedCV
from startx.validation.metrics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

__all__ = [
    "GateResult",
    "TrialLedger",
    "make_cpcv",
    "survival_gate",
]

#: Verdict strings surfaced in reports.
_VERDICT_PASS = "ROBUST"
_VERDICT_FAIL = "LIKELY OVERFIT"

#: Minimum observations / candidate configs before a statistic is meaningful.
_MIN_OBS = 2


def make_cpcv(t1: pd.Series) -> CombinatorialPurgedCV:
    """Build a Combinatorial Purged CV splitter from ``cfg.validation``.

    Parameters
    ----------
    t1:
        Label-end times (``pd.Series`` whose index is each sample's start time
        and whose values are the label-end times). Drives purging.

    Returns
    -------
    CombinatorialPurgedCV
        Configured with ``cpcv_n_groups``, ``cpcv_n_test_groups`` and
        ``embargo_pct`` from config; ``.n_paths`` then equals
        ``C(n_groups - 1, n_test_groups - 1)``.
    """
    validation = get_config().validation
    return CombinatorialPurgedCV(
        n_groups=validation.cpcv_n_groups,
        n_test_groups=validation.cpcv_n_test_groups,
        t1=t1,
        embargo_pct=validation.embargo_pct,
    )


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of the DSR/PBO survival gate.

    Attributes
    ----------
    deflated_sharpe:
        Probability the true Sharpe is positive after deflating the observed
        OOS Sharpe by ``n_trials`` (Bailey & López de Prado).
    pbo:
        Probability of Backtest Overfitting (CSCV logit method); ~0.5 means
        in-sample selection is no better than chance.
    n_trials:
        Number of configurations searched, used for the DSR deflation.
    n_obs:
        Number of OOS observations the Sharpe was estimated on.
    passed:
        ``True`` iff ``deflated_sharpe >= cfg.validation.dsr_min`` AND
        ``pbo <= cfg.validation.pbo_max``.
    verdict:
        Human-readable label (``"ROBUST"`` / ``"LIKELY OVERFIT"``).
    """

    deflated_sharpe: float
    pbo: float
    n_trials: int
    n_obs: int
    passed: bool
    verdict: str


def _sample_skew(returns: npt.NDArray[np.float64]) -> float:
    """Bias-corrected sample skewness (the G1 estimator, matching scipy)."""
    n = returns.size
    if n < 3:
        return 0.0
    centred = returns - returns.mean()
    m2 = float(np.mean(centred**2))
    m3 = float(np.mean(centred**3))
    if m2 == 0.0:
        return 0.0
    g1 = m3 / m2**1.5
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def _sample_kurtosis_nonexcess(returns: npt.NDArray[np.float64]) -> float:
    """Bias-corrected, *non-excess* kurtosis (3.0 for a Gaussian; matches scipy)."""
    n = returns.size
    if n < 4:
        return 3.0
    centred = returns - returns.mean()
    m2 = float(np.mean(centred**2))
    m4 = float(np.mean(centred**4))
    if m2 == 0.0:
        return 3.0
    g2 = m4 / m2**2 - 3.0  # excess kurtosis (biased)
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)
    return float(excess + 3.0)


def _per_obs_sharpe_moments(
    returns: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Per-observation Sharpe, skew and (non-excess) kurtosis of ``returns``.

    Matches the convention :func:`startx.validation.metrics.deflated_sharpe_ratio`
    expects: a per-observation Sharpe (no annualisation), bias-corrected sample
    skew, and non-excess kurtosis (3.0 for a Gaussian).
    """
    sd = float(returns.std(ddof=1))
    if sd == 0.0:
        return float("nan"), 0.0, 3.0
    sr_per_obs = float(returns.mean() / sd)
    return sr_per_obs, _sample_skew(returns), _sample_kurtosis_nonexcess(returns)


def _cscv_pbo(returns_matrix: npt.NDArray[np.float64]) -> float:
    """PBO over a candidate matrix via Combinatorially-Symmetric Cross-Validation.

    ``returns_matrix`` has shape ``(n_obs, n_configs)`` — one return column per
    candidate configuration. We split the timeline into ``S`` contiguous blocks,
    and for every way of partitioning those blocks into equal in-sample / out-of-
    sample halves we record each config's mean return on each half. Those aligned
    ``(n_splits, n_configs)`` IS / OOS matrices are fed to
    :func:`startx.validation.metrics.probability_of_backtest_overfitting`, which
    measures how often the in-sample winner lands in the OOS lower half.
    """
    n_obs, _n_configs = returns_matrix.shape
    # Even block count so blocks partition cleanly into IS / OOS halves.
    s_blocks = min(8, n_obs - (n_obs % 2))
    s_blocks -= s_blocks % 2
    if s_blocks < _MIN_OBS:
        s_blocks = _MIN_OBS
    block_pos = [b for b in np.array_split(np.arange(n_obs), s_blocks) if b.size]
    s_blocks = len(block_pos)
    half = s_blocks // 2

    is_rows: list[npt.NDArray[np.float64]] = []
    oos_rows: list[npt.NDArray[np.float64]] = []
    for combo in combinations(range(s_blocks), half):
        is_set = set(combo)
        is_idx = np.concatenate([block_pos[b] for b in range(s_blocks) if b in is_set])
        oos_idx = np.concatenate(
            [block_pos[b] for b in range(s_blocks) if b not in is_set]
        )
        is_rows.append(returns_matrix[is_idx].mean(axis=0))
        oos_rows.append(returns_matrix[oos_idx].mean(axis=0))

    perf_is = np.vstack(is_rows)
    perf_oos = np.vstack(oos_rows)
    return float(probability_of_backtest_overfitting(perf_is, perf_oos))


def survival_gate(
    oos_returns: pd.Series | npt.NDArray[np.float64],
    *,
    n_trials: int,
    trial_returns: list[pd.Series | npt.NDArray[np.float64]] | None = None,
    sr_benchmark: float | None = None,
) -> GateResult:
    """Ship/no-ship verdict for an edge's out-of-sample returns.

    Parameters
    ----------
    oos_returns:
        The selected (best) configuration's out-of-sample per-period returns.
    n_trials:
        Number of configurations searched (e.g. ``TrialLedger.count``). Feeds the
        DSR deflation so a broad search is penalised; coerced to >= 1 by startx.
    trial_returns:
        Optional OOS return series for *every* configuration tried (each aligned
        to ``oos_returns``). They form the candidate matrix for the Probability of
        Backtest Overfitting via CSCV. When omitted, PBO compares the selected
        edge against a flat (zero-return) null comparator — informative only for
        the clear-cut cases (a real edge dominates the null; pure noise does not).
    sr_benchmark:
        Optional explicit deflation benchmark passed through to
        :func:`startx.validation.metrics.deflated_sharpe_ratio`.

    Returns
    -------
    GateResult
        With ``passed = (dsr >= cfg.validation.dsr_min) and
        (pbo <= cfg.validation.pbo_max)``.
    """
    validation = get_config().validation

    r = np.asarray(oos_returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    n_obs = int(r.size)

    if n_obs < _MIN_OBS:
        return GateResult(
            deflated_sharpe=float("nan"),
            pbo=float("nan"),
            n_trials=int(n_trials),
            n_obs=n_obs,
            passed=False,
            verdict=_VERDICT_FAIL,
        )

    sr_per_obs, skew, kurt = _per_obs_sharpe_moments(r)
    dsr = deflated_sharpe_ratio(
        sr_per_obs,
        n_trials=n_trials,
        n_obs=n_obs,
        skew=skew,
        kurt=kurt,
        sr_benchmark=sr_benchmark,
    )

    # Build the candidate matrix (n_obs, n_configs) for PBO.
    columns: list[npt.NDArray[np.float64]] = [r]
    if trial_returns is not None:
        for tr in trial_returns:
            arr = np.asarray(tr, dtype=float).ravel()[:n_obs]
            if arr.size == n_obs:
                columns.append(arr)
    if len(columns) < _MIN_OBS:
        # No competing configs supplied: contrast against a flat null so PBO is
        # defined. A genuine edge beats the null in-sample AND out-of-sample.
        columns.append(np.zeros(n_obs, dtype=float))
    matrix = np.column_stack(columns)
    pbo = _cscv_pbo(matrix)

    dsr_ok = (not np.isnan(dsr)) and dsr >= validation.dsr_min
    pbo_ok = (not np.isnan(pbo)) and pbo <= validation.pbo_max
    passed = bool(dsr_ok and pbo_ok)
    return GateResult(
        deflated_sharpe=float(dsr),
        pbo=float(pbo),
        n_trials=int(n_trials),
        n_obs=n_obs,
        passed=passed,
        verdict=_VERDICT_PASS if passed else _VERDICT_FAIL,
    )
