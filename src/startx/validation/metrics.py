"""Performance metrics and overfitting statistics.

Two families live here:

* **Classic performance metrics** — Sharpe, Sortino, max drawdown, CAGR,
  profit factor, expectancy, hit rate. Standard but documented for clarity.
* **Overfitting statistics** — the Deflated Sharpe Ratio (Bailey & López de
  Prado, 2014) and the Probability of Backtest Overfitting via the CSCV /
  combinatorially-symmetric logit method (Bailey, Borwein, López de Prado &
  Zhu, 2017). These are what stop a lucky-best-of-N backtest from looking real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _scipy_skew


def _to_returns(x: pd.Series | np.ndarray) -> np.ndarray:
    """Return a 1-D float array of returns with NaNs dropped."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def sharpe(returns: pd.Series | np.ndarray, periods: int = 252,
           risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    ``SR = (mean(r) - rf_per_period) / std(r) * sqrt(periods)``. ``std`` uses
    the sample standard deviation (ddof=1). Returns ``nan`` if volatility is 0.
    """
    r = _to_returns(returns)
    if r.size < 2:
        return float("nan")
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(returns: pd.Series | np.ndarray, periods: int = 252,
            risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio — Sharpe but penalising only downside vol.

    Downside deviation uses returns below the per-period target (``risk_free``),
    with the full sample size in the denominator (ddof=0 convention).
    """
    r = _to_returns(returns)
    if r.size < 2:
        return float("nan")
    target = risk_free / periods
    excess = r - target
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods))


def max_drawdown(equity_or_returns: pd.Series | np.ndarray,
                 is_returns: bool | None = None) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction (e.g. -0.25).

    Accepts either an equity curve or a return series. If ``is_returns`` is
    ``None`` the input is auto-detected: a series whose values are all positive
    and not centred near zero is treated as an equity curve, otherwise as
    returns compounded into an equity curve starting at 1.
    """
    x = np.asarray(equity_or_returns, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    if is_returns is None:
        # Heuristic: returns typically have small magnitude and include
        # negatives or values near zero; equity curves are strictly positive
        # and far from zero. Default to "looks like returns" when any value is
        # <= 0 or the mean magnitude is small.
        is_returns = bool(np.any(x <= 0) or np.mean(np.abs(x)) < 0.5)
    equity = np.cumprod(1.0 + x) if is_returns else x
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def cagr(returns: pd.Series | np.ndarray, periods: int = 252) -> float:
    """Compound annual growth rate implied by a per-period return series.

    ``CAGR = prod(1 + r) ** (periods / n) - 1``.
    """
    r = _to_returns(returns)
    if r.size == 0:
        return float("nan")
    total = np.prod(1.0 + r)
    if total <= 0:
        return -1.0
    return float(total ** (periods / r.size) - 1.0)


def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Gross profit divided by gross loss (absolute).

    ``> 1`` is profitable. ``inf`` when there are no losing returns.
    """
    r = _to_returns(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def expectancy(returns: pd.Series | np.ndarray) -> float:
    """Average return per trade — simply the mean of the return series."""
    r = _to_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(r.mean())


def hit_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of strictly-positive returns (win rate) in ``[0, 1]``."""
    r = _to_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0))


def deflated_sharpe_ratio(sr_hat: float, n_trials: int, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0,
                          sr_benchmark: float | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    The DSR is the probability that the *true* Sharpe ratio is positive, after
    correcting the observed Sharpe ``sr_hat`` for (a) the number of independent
    trials ``n_trials`` that were tried (multiple-testing / selection bias),
    (b) the sample length ``n_obs``, and (c) non-normality of returns (``skew``
    and ``kurt``, where ``kurt`` is the *non-excess* kurtosis, 3 for Gaussian).

    Steps
    -----
    1. Estimate the expected maximum Sharpe under the null (true SR = 0) of
       ``n_trials`` trials. Using the variance of the trial Sharpe estimates
       ``var_sr`` and the Euler-Mascheroni-based extreme-value approximation::

           E[max] ≈ sqrt(var_sr) * ((1-γ) Z⁻¹[1 - 1/N] + γ Z⁻¹[1 - 1/(N e)])

       where ``γ`` is Euler-Mascheroni and ``Z⁻¹`` the inverse normal CDF.
       This is the deflation benchmark ``SR0`` (unless ``sr_benchmark`` given).
    2. The DSR is the probability that ``sr_hat`` exceeds ``SR0`` accounting for
       the standard error of a Sharpe estimate under non-normal returns::

           DSR = Φ( (sr_hat - SR0) * sqrt(n_obs - 1)
                    / sqrt(1 - skew*sr_hat + (kurt-1)/4 * sr_hat²) )

    Returns a probability in ``[0, 1]``. Values above ~0.95 are the usual bar
    for "the edge is unlikely to be a fluke".
    """
    n_trials = max(int(n_trials), 1)
    if n_obs < 2:
        return float("nan")

    if sr_benchmark is None:
        sr0 = _expected_max_sharpe(n_trials, var_sr=1.0 / (n_obs - 1))
    else:
        sr0 = float(sr_benchmark)

    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_hat - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def _expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of ``n_trials`` IID null Sharpe estimates.

    Extreme-value (Gumbel) approximation used in the DSR benchmark ``SR0``.
    With a single trial there is no selection bias, so ``SR0 = 0``.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    n = float(n_trials)
    sqrt_var = np.sqrt(var_sr)
    max_z = ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n)
             + gamma * norm.ppf(1.0 - 1.0 / (n * np.e)))
    return float(sqrt_var * max_z)


def deflated_sharpe(returns: pd.Series | np.ndarray, n_trials: int,
                    periods: int = 252) -> float:
    """Convenience DSR straight from a return series.

    Computes the (non-annualised, per-observation) Sharpe, its skew and
    kurtosis and the sample length, then calls :func:`deflated_sharpe_ratio`.
    The Sharpe is deflated on a *per-observation* basis (no ``sqrt(periods)``
    scaling) because the DSR's standard-error term is expressed per
    observation; annualisation cancels in the standardisation.

    Returns the probability that the true Sharpe is positive, in ``[0, 1]``.
    """
    r = _to_returns(returns)
    n_obs = r.size
    if n_obs < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr_per_obs = r.mean() / sd
    sk = float(_scipy_skew(r, bias=False)) if n_obs > 2 else 0.0
    # scipy kurtosis(fisher=False) returns the *non-excess* kurtosis (3 = normal)
    kt = float(_scipy_kurtosis(r, fisher=False, bias=False)) if n_obs > 3 else 3.0
    return deflated_sharpe_ratio(sr_per_obs, n_trials=n_trials, n_obs=n_obs,
                                 skew=sk, kurt=kt)


def probability_of_backtest_overfitting(
    perf_is: np.ndarray, perf_oos: np.ndarray
) -> float:
    """Probability of Backtest Overfitting (PBO) via the CSCV logit method.

    Following Bailey, Borwein, López de Prado & Zhu (2017). The inputs are two
    aligned matrices of shape ``(n_splits, n_configs)``: for each combinatorial
    train/test split, the in-sample (``perf_is``) and out-of-sample
    (``perf_oos``) performance of every candidate configuration.

    Procedure (per split / row)
    ---------------------------
    1. Pick ``n*`` = the configuration with the best in-sample performance.
    2. Find its *rank* among all configs out-of-sample, and convert to a
       relative rank ``ω`` in ``(0, 1)`` (1 = best OOS, 0 = worst).
    3. Map to a logit ``λ = ln(ω / (1 - ω))``. If the IS-best config is also
       good OOS, ``λ > 0``; if IS performance does not carry over, ``λ`` is
       centred at / below zero.

    PBO is the fraction of splits whose logit is ``<= 0`` — i.e. the
    probability that the strategy selected as best in-sample lands in the lower
    half of the out-of-sample distribution. ``PBO ≈ 0.5`` means selection is no
    better than chance (pure overfitting); ``PBO`` near 0 means the in-sample
    ranking genuinely predicts out-of-sample performance.
    """
    is_mat = np.asarray(perf_is, dtype=float)
    oos_mat = np.asarray(perf_oos, dtype=float)
    if is_mat.shape != oos_mat.shape:
        raise ValueError("perf_is and perf_oos must have the same shape")
    if is_mat.ndim != 2:
        raise ValueError("perf_is/perf_oos must be 2-D (n_splits, n_configs)")
    n_splits, n_configs = is_mat.shape
    if n_configs < 2:
        raise ValueError("need at least 2 configurations to assess overfitting")

    logits = np.empty(n_splits)
    for i in range(n_splits):
        best_is = int(np.argmax(is_mat[i]))
        oos_row = oos_mat[i]
        # Relative rank of the IS-best config in the OOS row, in (0, 1).
        # rank counts how many configs it beats or ties (average-rank style).
        order = np.argsort(oos_row, kind="mergesort")
        ranks = np.empty(n_configs)
        ranks[order] = np.arange(1, n_configs + 1)
        rank_best = ranks[best_is]
        omega = rank_best / (n_configs + 1)  # in (0, 1), avoids 0/1 blow-up
        logits[i] = np.log(omega / (1.0 - omega))

    return float(np.mean(logits <= 0.0))
