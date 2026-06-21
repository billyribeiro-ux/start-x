"""Fractional differentiation — Lopez de Prado fixed-width window (FFD).

Standard integer differencing (``d=1``) makes a price series stationary but
erases almost all memory. Fractional differentiation finds the *smallest*
``d`` that achieves stationarity while preserving the maximum amount of the
series' memory — the property that gives a feature its predictive content.

This module implements the **fixed-width window** variant (FFD) from *Advances
in Financial Machine Learning*, ch. 5. The FFD weight series is truncated once
the magnitude of a weight falls below a threshold, so every output value is the
dot product of the SAME fixed-length backward window of weights against the
trailing inputs. That fixed backward window is the load-bearing point-in-time
guarantee: the value at ``t`` depends ONLY on inputs at ``t`` and earlier, never
on anything in the future. The :data:`FRACDIFF_BOUNDARY` declares this contract.

Public functions:
    * :func:`ffd_weights` — the truncated FFD weight series.
    * :func:`frac_diff_ffd` — apply FFD to a series, NaN-padding warm-up rows.
    * :func:`min_d_adf` — search the smallest ``d`` passing the ADF test.
    * :func:`fracdiff_boundary` — the feature's :class:`InformationBoundary`.

Only numpy, pandas, statsmodels and ``qedge.{config,data.boundary}`` are used.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from qedge.config import FeatureConfig, get_config
from qedge.data.boundary import BoundaryKind, InformationBoundary

__all__ = [
    "FRACDIFF_BOUNDARY",
    "ffd_weights",
    "frac_diff_ffd",
    "fracdiff_boundary",
    "min_d_adf",
]

# Index of the ADF test statistic's p-value in the statsmodels result tuple.
_ADF_PVALUE_INDEX = 1


def ffd_weights(d: float, threshold: float) -> npt.NDArray[np.float64]:
    """Return the fixed-width-window FFD weight series for order ``d``.

    The weights follow the binomial recursion ``w_k = -w_{k-1} * (d - k + 1) / k``
    with ``w_0 = 1``. The series is truncated as soon as ``|w_k| < threshold``;
    the resulting fixed length defines the backward window every output value
    consumes. Weights are returned most-recent-first (``w[0]`` multiplies the
    value at ``t``, ``w[-1]`` the oldest in-window value).

    Args:
        d: Fractional differencing order (``>= 0``). ``0`` reproduces the input,
            ``1`` reproduces the first difference.
        threshold: Strictly positive magnitude cutoff; weights smaller than this
            in absolute value are dropped.

    Returns:
        A 1-D float array of weights, length ``>= 1`` (``w[0]`` is always ``1``).

    Raises:
        ValueError: If ``d`` is negative or ``threshold`` is not positive.
    """
    if d < 0.0:
        raise ValueError("d must be >= 0")
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")

    weights: list[float] = [1.0]
    k = 1
    while True:
        next_weight = -weights[-1] * (d - k + 1.0) / k
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
        k += 1
    return np.asarray(weights, dtype=np.float64)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float) -> pd.Series:
    """Apply fixed-width-window fractional differentiation to ``series``.

    Each output value at ``t`` is the dot product of :func:`ffd_weights` with the
    fixed backward window of inputs ending at ``t``. The first
    ``len(weights) - 1`` rows lack a full window and are returned as ``NaN``. The
    output shares the input's index and order, so it can be aligned directly.

    Args:
        series: The input series (e.g. log prices), ascending time index.
        d: Fractional differencing order passed to :func:`ffd_weights`.
        threshold: Weight-truncation threshold passed to :func:`ffd_weights`.

    Returns:
        A float :class:`pandas.Series` aligned to ``series.index`` with leading
        NaNs for the warm-up window.
    """
    weights = ffd_weights(d, threshold)
    width = weights.size
    values = series.to_numpy(dtype=np.float64)
    n = values.size

    out = np.full(n, np.nan, dtype=np.float64)
    # Weights are most-recent-first; reverse so the oldest in-window value lines
    # up with the start of each sliding window.
    weights_oldest_first = weights[::-1]
    for end in range(width - 1, n):
        window = values[end - width + 1 : end + 1]
        out[end] = float(np.dot(weights_oldest_first, window))
    return pd.Series(out, index=series.index, name=series.name)


def min_d_adf(
    series: pd.Series,
    *,
    d_grid: tuple[float, ...] | None = None,
    adf_pvalue: float | None = None,
    threshold: float | None = None,
) -> tuple[float, float]:
    """Find the smallest ``d`` whose FFD series is ADF-stationary.

    The grid is scanned in ascending order; the first ``d`` whose FFD-transformed
    series passes the Augmented Dickey-Fuller test at ``p <= adf_pvalue`` is
    returned. Smaller ``d`` preserves more memory, so the smallest passing order
    is preferred. If NO grid value passes, the **largest** ``d`` (most aggressive
    differencing, lowest achievable p-value) is returned together with its
    achieved p-value, leaving the caller to decide whether that is acceptable.

    Args:
        series: The input series to render stationary.
        d_grid: Ascending candidate orders. Defaults to
            ``cfg.features.fracdiff_d_grid``.
        adf_pvalue: ADF significance threshold. Defaults to
            ``cfg.features.fracdiff_adf_pvalue``.
        threshold: FFD weight-truncation threshold. Defaults to
            ``cfg.features.fracdiff_weight_threshold``.

    Returns:
        A ``(d, achieved_pvalue)`` tuple. ``achieved_pvalue`` is the ADF p-value
        of the FFD series at the chosen ``d``.

    Raises:
        ValueError: If ``d_grid`` is empty.
    """
    features: FeatureConfig = get_config().features
    grid = features.fracdiff_d_grid if d_grid is None else d_grid
    target_pvalue = features.fracdiff_adf_pvalue if adf_pvalue is None else adf_pvalue
    weight_threshold = (
        features.fracdiff_weight_threshold if threshold is None else threshold
    )

    if len(grid) == 0:
        raise ValueError("d_grid must contain at least one candidate order")

    ascending = sorted(grid)
    best_d = ascending[-1]
    best_pvalue = float("inf")
    for candidate in ascending:
        transformed = frac_diff_ffd(series, candidate, weight_threshold).dropna()
        achieved = float(adfuller(transformed.to_numpy())[_ADF_PVALUE_INDEX])
        if achieved <= target_pvalue:
            return candidate, achieved
        # Track the largest d's p-value as the documented fallback.
        if candidate == ascending[-1]:
            best_d = candidate
            best_pvalue = achieved
    return best_d, best_pvalue


def fracdiff_boundary(d: float, threshold: float) -> InformationBoundary:
    """Return the :class:`InformationBoundary` for an FFD feature.

    The boundary is TRAILING with ``lookback_days`` equal to the FFD window width
    (``len(ffd_weights(d, threshold))``): the value at ``t`` reaches back exactly
    that many bars and never touches data after ``t``.

    Args:
        d: Fractional differencing order used by the feature.
        threshold: Weight-truncation threshold used by the feature.

    Returns:
        A frozen :class:`InformationBoundary` describing the fixed window.
    """
    width = ffd_weights(d, threshold).size
    return InformationBoundary(
        kind=BoundaryKind.TRAILING,
        lookback_days=width,
        latency_days=0,
        description=(
            f"FFD fractional differentiation (d={d}, threshold={threshold}): "
            f"value at t is a fixed {width}-bar backward weighted sum of inputs "
            "at t and earlier only — no future data is read."
        ),
    )


# Default-config boundary for the canonical FFD feature. The fixed backward
# window is what guarantees point-in-time correctness (see module docstring).
_DEFAULT_FEATURES = get_config().features
FRACDIFF_BOUNDARY: InformationBoundary = fracdiff_boundary(
    d=_DEFAULT_FEATURES.fracdiff_d_grid[-1],
    threshold=_DEFAULT_FEATURES.fracdiff_weight_threshold,
)
