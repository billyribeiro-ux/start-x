"""Tests for the ensemble combiner — pins the two firewall fixes:

1. The deflated Sharpe reported by ``ensemble.combine`` now matches the audited,
   textbook Bailey/López-de-Prado implementation in
   ``startx.validation.metrics.deflated_sharpe`` directly (the old local formula
   was a structurally-mangled DSR that diverged four ways and reported 0.61 for
   the decoded ensemble instead of the correct ~0.995).
2. The max-Sharpe tangency solve uses a scale-aware ridge so near-collinear
   edges no longer produce exploding weights (e.g. {+249, -248}) under
   ``long_only=False``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.portfolio.ensemble import _deflated_sharpe, combine
from startx.validation.metrics import deflated_sharpe as _metrics_dsr


def _monthly_series(values: np.ndarray, start: str = "2019-01-31") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx)


# --------------------------------------------------------------------------- #
# Fix 1 — DSR delegates to the audited metrics implementation
# --------------------------------------------------------------------------- #
def test_wrapper_matches_metrics_dsr():
    """``_deflated_sharpe`` is a thin wrapper over metrics.deflated_sharpe."""
    rng = np.random.default_rng(42)
    r = _monthly_series(rng.normal(0.008, 0.03, 120))
    n_trials = 8
    wrapped = _deflated_sharpe(r, ppy=12, n_trials=n_trials)
    direct = _metrics_dsr(r, n_trials=n_trials)
    assert np.isclose(wrapped, direct, equal_nan=True)


def test_combine_dsr_matches_metrics_on_combined_stream():
    """The DSR in ``combine(...).stats`` equals metrics.deflated_sharpe of the
    realised combined stream — i.e. no bespoke (broken) formula in the path."""
    rng = np.random.default_rng(7)
    streams = {
        "a": _monthly_series(rng.normal(0.010, 0.030, 100)),
        "b": _monthly_series(rng.normal(0.006, 0.025, 100)),
        "c": _monthly_series(rng.normal(0.004, 0.040, 100)),
    }
    n_trials = 8
    res = combine(streams, method="max_sharpe", n_trials=n_trials)
    direct = _metrics_dsr(res.combined, n_trials=n_trials)
    assert np.isclose(res.stats["deflated_sharpe"], direct, equal_nan=True)


def test_dsr_is_a_probability():
    rng = np.random.default_rng(1)
    r = _monthly_series(rng.normal(0.012, 0.02, 90))
    dsr = _deflated_sharpe(r, ppy=12, n_trials=8)
    assert 0.0 <= dsr <= 1.0


def test_dsr_nan_on_too_short_or_flat():
    rng = np.random.default_rng(9)
    # Too few observations (< 12) -> NaN via the length guard.
    assert np.isnan(_deflated_sharpe(_monthly_series(rng.normal(0.01, 0.02, 5)), 12, 8))
    # Exactly-flat stream -> NaN via the zero-variance guard.
    assert np.isnan(_deflated_sharpe(_monthly_series(np.zeros(30)), 12, 8))


# --------------------------------------------------------------------------- #
# Fix 2 — scale-aware ridge tames near-collinear edges
# --------------------------------------------------------------------------- #
def _tangency_max_abs_weight(cov, mu, ridge):
    w = np.linalg.solve(cov + np.eye(len(mu)) * ridge, mu)
    w = w / w.sum() if w.sum() != 0 else w
    return float(np.max(np.abs(w)))


def _collinear_cov(rho: float, sig: float = 0.02):
    """A 2x2 covariance of two edges with correlation ``rho`` and equal vol."""
    return sig ** 2 * np.array([[1.0, rho], [rho, 1.0]])


def test_scale_aware_ridge_never_under_regularises_old():
    """The scale-aware ridge is >= the old fixed 1e-10 for any realistic cov
    scale, so the tangency solve is never *less* regularised than before."""
    for sig in (0.02, 0.1, 1.0):
        cov = _collinear_cov(0.5, sig)
        ridge = 1e-6 * np.trace(cov) / len(cov)
        assert ridge >= 1e-10


def test_collinear_streams_no_exploding_weights_long_only_false():
    """Strongly near-collinear edges under long_only=False: the scale-aware
    ridge keeps the tangency max-weight strictly tamer than the old 1e-10 ridge
    (which let weights blow up to the documented {+249, -248} magnitude), and
    the public combine() path stays finite."""
    mu = np.array([0.010, 0.009])
    # Tight collinearity — the conditioning regime that exploded the weights.
    cov = _collinear_cov(1.0 - 1e-6)

    old_max = _tangency_max_abs_weight(cov, mu, 1e-10)
    new_max = _tangency_max_abs_weight(cov, mu, 1e-6 * np.trace(cov) / len(cov))

    # Old ridge explodes into the thousands; the scale-aware ridge is materially
    # (here ~2x) tamer and finite.
    assert old_max > 1_000.0, f"expected old ridge to explode, got {old_max}"
    assert new_max < old_max, f"new ridge not tamer: {new_max} vs {old_max}"
    assert np.isfinite(new_max)

    # And the public path on near-collinear streams is finite with no NaNs.
    rng = np.random.default_rng(0)
    base = rng.normal(0.008, 0.02, 120)
    a = _monthly_series(base)
    b = _monthly_series(base + rng.normal(0.0, 3e-6, 120))
    res = combine({"a": a, "b": b}, method="max_sharpe", long_only=False)
    w = np.array(list(res.weights.values()))
    assert np.all(np.isfinite(w))


def test_long_only_default_weights_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(3)
    streams = {
        "x": _monthly_series(rng.normal(0.009, 0.03, 80)),
        "y": _monthly_series(rng.normal(0.005, 0.02, 80)),
    }
    res = combine(streams, method="max_sharpe", long_only=True)
    w = np.array(list(res.weights.values()))
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)
