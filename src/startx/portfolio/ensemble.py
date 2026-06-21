"""Ensemble combiner — assemble decoded edges into one max-Sharpe book.

The decode campaign's central, evidence-backed finding: there is no single oracle. There is a
constellation of small, real, weakly-correlated edges (the SPY swing book, a market-neutral ML
cross-sectional ranker, earnings-reaction drift, index-flow drift, ...), each individually thin
(deflated Sharpe ~0.2-0.6) but cleared by the firewall. The alpha is in *assembling* them, weighted
by evidence — diversification across uncorrelated edges raises the ceiling √(Σ SRᵢ²) far above any
single piece.

This module takes a dict of named return streams (each a strategy's P&L series, any frequency),
resamples to a common rebalance frequency (monthly by default — robust to mixed daily/weekly/period
streams), and combines them by max-Sharpe (tangency), risk-parity, or equal weight. It returns the
combined equity curve, per-edge weights/contributions, the correlation matrix, and honest metrics.

Weighting is computed IN-SAMPLE on the supplied history — for a live book, estimate weights on a
trailing window and apply forward (walk-forward), which the caller can do by slicing. Max-Sharpe is
long-only-clipped by default (no shorting an edge) and normalized to sum to 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EnsembleResult:
    weights: dict[str, float]          # optimal weight per edge (sums to 1)
    combined: pd.Series                # combined return stream (at the rebalance frequency)
    equity: pd.Series                  # combined equity curve (starts at 1.0)
    per_edge: pd.DataFrame             # per-edge sharpe / weight / annualized contribution
    corr: pd.DataFrame                 # edge return correlation matrix
    stats: dict = field(default_factory=dict)


def _resample(r: pd.Series, freq: str) -> pd.Series:
    """Compound a return series to ``freq`` (e.g. 'ME' month-end). Robust to any input frequency."""
    r = r.copy(); r.index = pd.to_datetime(r.index)
    return (1.0 + r.fillna(0.0)).resample(freq).prod() - 1.0


def _sharpe(r: pd.Series, ppy: int) -> float:
    s = r.std()
    return float(r.mean() / s * np.sqrt(ppy)) if s > 0 else 0.0


def _max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _deflated_sharpe(r: pd.Series, ppy: int, n_trials: int = 8) -> float:
    """Probability the true (per-period) Sharpe is > 0 after deflating for ``n_trials``.

    Thin wrapper that delegates to the audited, textbook Bailey/López-de-Prado
    implementation in :func:`startx.validation.metrics.deflated_sharpe`. The DSR
    is a *probability* and is invariant to annualization (the sqrt-ppy scaling
    cancels in the standardisation), so ``ppy`` is accepted for signature
    symmetry with the other helpers but is not used by the formula.
    """
    if len(r) < 12 or r.std() == 0:
        return float("nan")
    try:
        from startx.validation.metrics import deflated_sharpe as _dsr
    except Exception:
        return float("nan")
    return float(_dsr(r, n_trials=n_trials))


def combine(streams: dict[str, pd.Series], *, method: str = "max_sharpe", long_only: bool = True,
            freq: str = "ME", n_trials: int = 8) -> EnsembleResult:
    """Assemble edge return ``streams`` into one book.

    method: 'max_sharpe' (tangency, weights ∝ Σ⁻¹μ), 'risk_parity' (∝ 1/σ), or 'equal'.
    freq:   common rebalance frequency to resample every stream to (default month-end).
    """
    M = pd.DataFrame({k: _resample(v, freq) for k, v in streams.items()}).dropna()
    if M.empty or M.shape[1] == 0:
        raise ValueError("no overlapping history across the supplied streams")
    ppy = {"ME": 12, "W": 52, "QE": 4}.get(freq, 12)
    mu, cov = M.mean().values, M.cov().values

    if method == "max_sharpe":
        # Scale-aware ridge: stabilises the tangency solve when edges are
        # near-collinear (a 1e-10 ridge let weights blow up to e.g. {+249,-248}
        # under long_only=False). Ties the regularisation to the covariance
        # scale (mean diagonal variance) so it works for any return units.
        ridge = 1e-6 * np.trace(cov) / len(cov)
        w = np.linalg.solve(cov + np.eye(len(mu)) * ridge, mu)
    elif method == "risk_parity":
        w = 1.0 / M.std().values
    else:
        w = np.ones(M.shape[1])
    if long_only:
        w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    weights = {c: float(wi) for c, wi in zip(M.columns, w)}

    combined = (M * w).sum(axis=1)
    equity = (1.0 + combined).cumprod()
    per_edge = pd.DataFrame({
        "sharpe": {c: _sharpe(M[c], ppy) for c in M.columns},
        "weight": weights,
        "contrib_ann": {c: float((M[c] * weights[c]).mean() * ppy) for c in M.columns},
    }).sort_values("weight", ascending=False)

    stats = {
        "n_periods": int(len(M)), "freq": freq,
        "sharpe": _sharpe(combined, ppy),
        "deflated_sharpe": _deflated_sharpe(combined, ppy, n_trials),
        "cagr": float((equity.iloc[-1]) ** (ppy / len(M)) - 1.0),
        "vol_ann": float(combined.std() * np.sqrt(ppy)),
        "max_drawdown": _max_drawdown(combined),
        "best_single_sharpe": float(per_edge["sharpe"].max()),
    }
    stats["calmar"] = stats["cagr"] / abs(stats["max_drawdown"]) if stats["max_drawdown"] < 0 else float("nan")
    stats["diversification_lift"] = stats["sharpe"] - stats["best_single_sharpe"]
    return EnsembleResult(weights, combined, equity, per_edge, M.corr(), stats)
