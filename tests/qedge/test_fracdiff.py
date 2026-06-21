"""Tests for fixed-width-window fractional differentiation (FFD).

The headline test is the NO-LOOKAHEAD CANARY: poisoning the future of a series
must not change any historical FFD value, proving the fixed backward window is a
true point-in-time guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from qedge.config import QedgeConfig
from qedge.data.boundary import BoundaryKind
from qedge.features.fracdiff import (
    FRACDIFF_BOUNDARY,
    ffd_weights,
    frac_diff_ffd,
    fracdiff_boundary,
    min_d_adf,
)

_ADF_PVALUE_INDEX = 1


def _random_walk(n: int, rng: np.random.Generator) -> pd.Series:
    """A non-stationary random-walk price proxy on a business-day index."""
    steps = rng.standard_normal(n)
    levels = 100.0 + np.cumsum(steps)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.Series(levels, index=idx, name="price")


# --------------------------------------------------------------------------- #
# ffd_weights
# --------------------------------------------------------------------------- #
def test_ffd_weights_first_is_one(cfg: QedgeConfig) -> None:
    w = ffd_weights(0.4, cfg.features.fracdiff_weight_threshold)
    assert w[0] == 1.0


def test_ffd_weights_decay_and_truncation(cfg: QedgeConfig) -> None:
    threshold = cfg.features.fracdiff_weight_threshold
    w = ffd_weights(0.4, threshold)
    # All retained weights (beyond w[0]) are at/above the threshold in magnitude.
    assert np.all(np.abs(w) >= threshold)
    # Magnitudes decay: the tail is strictly smaller than the head.
    assert abs(w[-1]) < abs(w[1])
    # A looser threshold truncates sooner -> shorter window.
    w_loose = ffd_weights(0.4, threshold * 100)
    assert w_loose.size < w.size


def test_ffd_weights_d1_is_first_difference() -> None:
    # For d=1 the exact weights are [1, -1] and the recursion terminates there.
    w = ffd_weights(1.0, 1e-6)
    np.testing.assert_allclose(w, np.array([1.0, -1.0]))


def test_ffd_weights_d0_is_identity() -> None:
    w = ffd_weights(0.0, 1e-6)
    np.testing.assert_allclose(w, np.array([1.0]))


# --------------------------------------------------------------------------- #
# frac_diff_ffd
# --------------------------------------------------------------------------- #
def test_frac_diff_d0_is_original(seeded_rng: np.random.Generator) -> None:
    series = _random_walk(300, seeded_rng)
    out = frac_diff_ffd(series, 0.0, 1e-6)
    # d=0 -> identity transform, no warm-up rows.
    assert not out.isna().any()
    pd.testing.assert_series_equal(out, series, check_names=False)


def test_frac_diff_d1_is_first_difference(seeded_rng: np.random.Generator) -> None:
    series = _random_walk(300, seeded_rng)
    out = frac_diff_ffd(series, 1.0, 1e-6)
    expected = series.diff()
    # First value is NaN warm-up in both; compare the rest.
    pd.testing.assert_series_equal(
        out.iloc[1:], expected.iloc[1:], check_names=False
    )


def test_frac_diff_d1_random_walk_is_stationary(
    seeded_rng: np.random.Generator,
) -> None:
    series = _random_walk(500, seeded_rng)
    # d=1.0 (first difference of a random walk) is stationary white noise.
    diffed = frac_diff_ffd(series, 1.0, 1e-6).dropna()
    pvalue = float(adfuller(diffed.to_numpy())[_ADF_PVALUE_INDEX])
    assert pvalue <= 0.05
    # d=0.0 (the raw random walk) is NOT stationary.
    raw = frac_diff_ffd(series, 0.0, 1e-6).dropna()
    raw_pvalue = float(adfuller(raw.to_numpy())[_ADF_PVALUE_INDEX])
    assert raw_pvalue > 0.05


def test_frac_diff_warmup_nans(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(11)
    series = _random_walk(400, rng)
    d = 0.4
    threshold = cfg.features.fracdiff_weight_threshold
    width = ffd_weights(d, threshold).size
    out = frac_diff_ffd(series, d, threshold)
    # Exactly width-1 leading NaNs, none thereafter.
    assert out.iloc[: width - 1].isna().all()
    assert not out.iloc[width - 1 :].isna().any()


# --------------------------------------------------------------------------- #
# min_d_adf
# --------------------------------------------------------------------------- #
def test_min_d_adf_smallest_passing_on_random_walk(
    cfg: QedgeConfig, seeded_rng: np.random.Generator
) -> None:
    series = _random_walk(800, seeded_rng)
    d, pvalue = min_d_adf(
        series,
        d_grid=cfg.features.fracdiff_d_grid,
        adf_pvalue=cfg.features.fracdiff_adf_pvalue,
        threshold=cfg.features.fracdiff_weight_threshold,
    )
    # A passing order is found...
    assert pvalue <= cfg.features.fracdiff_adf_pvalue
    # ...and it is the smallest such order: the next-smaller grid value fails.
    grid = sorted(cfg.features.fracdiff_d_grid)
    assert d in grid
    if d > grid[0]:
        smaller = grid[grid.index(d) - 1]
        smaller_series = frac_diff_ffd(
            series, smaller, cfg.features.fracdiff_weight_threshold
        ).dropna()
        smaller_p = float(adfuller(smaller_series.to_numpy())[_ADF_PVALUE_INDEX])
        assert smaller_p > cfg.features.fracdiff_adf_pvalue


def test_min_d_adf_uses_config_defaults(seeded_rng: np.random.Generator) -> None:
    # Defaults pulled from cfg.features when kwargs omitted.
    series = _random_walk(800, seeded_rng)
    d, pvalue = min_d_adf(series)
    assert d in QedgeConfig().features.fracdiff_d_grid
    assert pvalue <= QedgeConfig().features.fracdiff_adf_pvalue


def test_min_d_adf_near_stationary_returns_small_d() -> None:
    # White noise is already stationary; the smallest grid value (0.0) suffices.
    rng = np.random.default_rng(3)
    idx = pd.date_range("2019-01-01", periods=600, freq="B")
    noise = pd.Series(rng.standard_normal(600), index=idx, name="noise")
    d, pvalue = min_d_adf(noise)
    assert d == 0.0
    assert pvalue <= QedgeConfig().features.fracdiff_adf_pvalue


def test_min_d_adf_empty_grid_raises(seeded_rng: np.random.Generator) -> None:
    series = _random_walk(100, seeded_rng)
    raised = False
    try:
        min_d_adf(series, d_grid=())
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# Boundary
# --------------------------------------------------------------------------- #
def test_boundary_is_trailing_with_window_lookback(cfg: QedgeConfig) -> None:
    d = 0.4
    threshold = cfg.features.fracdiff_weight_threshold
    boundary = fracdiff_boundary(d, threshold)
    assert boundary.kind is BoundaryKind.TRAILING
    assert boundary.latency_days == 0
    assert boundary.lookback_days == ffd_weights(d, threshold).size


def test_module_boundary_matches_default_config(cfg: QedgeConfig) -> None:
    expected_width = ffd_weights(
        cfg.features.fracdiff_d_grid[-1], cfg.features.fracdiff_weight_threshold
    ).size
    assert FRACDIFF_BOUNDARY.kind is BoundaryKind.TRAILING
    assert FRACDIFF_BOUNDARY.lookback_days == expected_width


# --------------------------------------------------------------------------- #
# NO-LOOKAHEAD CANARY (critical)
# --------------------------------------------------------------------------- #
def test_no_lookahead_canary_future_poison_does_not_leak(
    cfg: QedgeConfig,
) -> None:
    """Poisoning T+1..T+50 must leave every value at index <= T byte-identical.

    FFD only reaches backward over a fixed window, so future values can never
    influence a historical output. Any leak would change the historical slice.
    """
    rng = np.random.default_rng(123)
    d = 0.4
    threshold = cfg.features.fracdiff_weight_threshold

    base = _random_walk(500, rng)
    out_base = frac_diff_ffd(base, d, threshold)

    # Append 50 arbitrary, deliberately extreme "poisoned" future values.
    future_idx = pd.date_range(
        base.index[-1] + pd.offsets.BDay(1), periods=50, freq="B"
    )
    poison = pd.Series(
        rng.standard_normal(50) * 1e6 + 1e9, index=future_idx, name=base.name
    )
    extended = pd.concat([base, poison])
    out_extended = frac_diff_ffd(extended, d, threshold)

    # Every historical index (<= T) must be byte-identical.
    historical = out_extended.loc[base.index]
    pd.testing.assert_series_equal(historical, out_base)
    # Be explicit about byte-equality on the underlying arrays (NaN-aware).
    np.testing.assert_array_equal(
        historical.to_numpy(), out_base.to_numpy()
    )
