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


@dataclass
class WalkForwardResult:
    """Output of :func:`combine_walkforward` — a genuine OUT-OF-SAMPLE combined book.

    Unlike :class:`EnsembleResult` (weights fit and scored on the SAME history), here every
    combined return is earned by weights that were estimated on a *strictly prior* trailing
    window and held fixed across the next period — no return ever informs the weights that
    are applied to it.

    Attributes
    ----------
    combined:
        The realised OOS combined return stream (at the rebalance ``freq``). Begins only once
        the first full ``lookback`` window has elapsed.
    equity:
        Combined OOS equity curve (starts at 1.0 on the first OOS period).
    weights:
        Time-varying weights — one row per OOS period, columns = edges. ``weights.iloc[t]`` is
        the weight vector estimated on the window *ending before* period ``t`` and applied to it.
    stats:
        Sharpe, deflated Sharpe, CAGR, vol, maxDD, Calmar of the OOS combined stream, plus the
        per-edge OOS-window stand-alone Sharpe and the diversification lift vs the best book.
    """

    combined: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
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


def _max_sharpe_weights(window: pd.DataFrame, *, long_only: bool = True) -> np.ndarray:
    """Tangency (max-Sharpe) weights ∝ Σ⁻¹μ on a single ``window`` of edge returns.

    Identical regularisation/clipping/normalisation logic as the ``max_sharpe`` branch of
    :func:`combine` (scale-aware ridge → optional long-only clip → sum-to-1), factored out so
    the walk-forward driver can re-estimate it per window WITHOUT touching ``combine``. Falls
    back to equal weight if the solve degenerates (zero/negative weight mass, singular cov).
    """
    mu, cov = window.mean().values, window.cov().values
    n = len(mu)
    ridge = 1e-6 * np.trace(cov) / n if n else 0.0
    try:
        w = np.linalg.solve(cov + np.eye(n) * ridge, mu)
    except np.linalg.LinAlgError:
        w = np.ones(n)
    if long_only:
        w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return w


def combine_walkforward(streams: dict[str, pd.Series], *, lookback: int = 24, freq: str = "ME",
                        long_only: bool = True, method: str = "max_sharpe",
                        n_trials: int = 8) -> WalkForwardResult:
    """Walk-forward (genuinely OUT-OF-SAMPLE) ensemble of edge return ``streams``.

    The honest counterpart to :func:`combine`. Every stream is resampled to ``freq``; then a
    rolling estimator marches forward one period at a time:

      for each period ``t`` with at least ``lookback`` prior periods available,
        1. estimate max-Sharpe weights on the TRAILING window ``[t-lookback, t-1]`` (strictly
           prior — period ``t`` is never seen),
        2. apply those fixed weights to period ``t``'s realised returns,
        3. roll forward.

    Because the weights applied to period ``t`` were fit only on data ending at ``t-1``, the
    resulting combined series is a clean OOS track record — there is **no look-ahead**.

    Parameters
    ----------
    lookback:
        Number of trailing ``freq`` periods used to estimate each weight vector (default 24 →
        two years at month-end).
    method:
        Currently ``"max_sharpe"`` (tangency); ``"risk_parity"`` (∝ 1/σ) and ``"equal"`` are
        also supported for comparison. Only ``max_sharpe`` actually *uses* the trailing means.

    Returns a :class:`WalkForwardResult` (OOS combined series, time-varying weights, stats).
    """
    M = pd.DataFrame({k: _resample(v, freq) for k, v in streams.items()}).dropna()
    if M.empty or M.shape[1] == 0:
        raise ValueError("no overlapping history across the supplied streams")
    if lookback < 2:
        raise ValueError("lookback must be >= 2 periods")
    if len(M) <= lookback:
        raise ValueError(
            f"need more than lookback={lookback} overlapping periods to walk forward, "
            f"got {len(M)}")
    ppy = {"ME": 12, "W": 52, "QE": 4}.get(freq, 12)
    cols = list(M.columns)

    oos_rets, oos_idx, wrows = [], [], []
    # March forward: weights from the strictly-prior window are applied to the current period.
    for t in range(lookback, len(M)):
        window = M.iloc[t - lookback:t]            # [t-lookback, t-1] — strictly before t
        if method == "max_sharpe":
            w = _max_sharpe_weights(window, long_only=long_only)
        elif method == "risk_parity":
            sd = window.std().values
            w = np.where(sd > 0, 1.0 / np.where(sd > 0, sd, 1.0), 0.0)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(cols)) / len(cols)
        else:  # equal weight
            w = np.ones(len(cols)) / len(cols)
        row = M.iloc[t]                            # the NEXT (out-of-sample) period
        oos_rets.append(float((row.values * w).sum()))
        oos_idx.append(M.index[t])
        wrows.append(w)

    combined = pd.Series(oos_rets, index=pd.DatetimeIndex(oos_idx), name="oos_combined")
    weights = pd.DataFrame(wrows, index=combined.index, columns=cols)
    equity = (1.0 + combined).cumprod()

    # Per-edge stand-alone Sharpe over the SAME OOS span (apples-to-apples for the lift).
    oos_span = M.loc[combined.index]
    per_edge_sharpe = {c: _sharpe(oos_span[c], ppy) for c in cols}
    best_single = max(per_edge_sharpe.values()) if per_edge_sharpe else float("nan")

    max_dd = _max_drawdown(combined)
    cagr = float(equity.iloc[-1] ** (ppy / len(combined)) - 1.0) if len(combined) else float("nan")
    stats = {
        "n_periods": int(len(combined)), "freq": freq, "lookback": int(lookback),
        "method": method,
        "sharpe": _sharpe(combined, ppy),
        "deflated_sharpe": _deflated_sharpe(combined, ppy, n_trials),
        "cagr": cagr,
        "vol_ann": float(combined.std() * np.sqrt(ppy)),
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "per_edge_oos_sharpe": per_edge_sharpe,
        "best_single_sharpe": float(best_single),
        "best_single_edge": (max(per_edge_sharpe, key=per_edge_sharpe.get)
                             if per_edge_sharpe else None),
        "mean_weights": {c: float(weights[c].mean()) for c in cols},
    }
    stats["diversification_lift"] = stats["sharpe"] - stats["best_single_sharpe"]
    return WalkForwardResult(combined, equity, weights, stats)


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
