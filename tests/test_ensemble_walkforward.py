"""Tests for ``combine_walkforward`` — the genuine OUT-OF-SAMPLE ensemble.

These pin the three properties that make the walk-forward result trustworthy (and that an
in-sample ``combine`` cannot claim):

1. **No look-ahead.** The weights applied to period ``t`` depend ONLY on data strictly before
   ``t``; perturbing a single future return must NOT change any earlier weight.
2. **Correct rolling alignment.** The OOS series starts exactly ``lookback`` periods in, has
   ``n - lookback`` observations, and each weight row is dated to the period it was applied to.
3. **Valid weights.** Every per-period weight row is non-negative (long-only) and sums to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startx.portfolio.ensemble import (
    WalkForwardResult,
    _max_sharpe_weights,
    combine,
    combine_walkforward,
)


def _monthly(values: np.ndarray, start: str = "2015-01-31") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx)


def _streams(n: int = 60, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    return {
        "a": _monthly(rng.normal(0.010, 0.030, n)),
        "b": _monthly(rng.normal(0.006, 0.025, n)),
        "c": _monthly(rng.normal(0.004, 0.040, n)),
    }


# --------------------------------------------------------------------------- #
# Property 2 — correct rolling alignment / shape
# --------------------------------------------------------------------------- #
def test_rolls_with_correct_length_and_index():
    n, lookback = 60, 24
    streams = _streams(n)
    res = combine_walkforward(streams, lookback=lookback, freq="ME")
    assert isinstance(res, WalkForwardResult)
    # n monthly periods, first `lookback` are burned to seed the estimation window.
    assert res.stats["n_periods"] == n - lookback
    assert len(res.combined) == n - lookback
    assert len(res.weights) == n - lookback
    # The OOS series begins exactly at period index `lookback` of the resampled frame.
    full = pd.DataFrame({k: (1 + v.fillna(0)).resample("ME").prod() - 1 for k, v in streams.items()}).dropna()
    assert res.combined.index[0] == full.index[lookback]
    assert res.combined.index[-1] == full.index[-1]
    # equity is the compounded OOS combined stream, starting near 1.
    assert np.isclose(res.equity.iloc[0], 1.0 + res.combined.iloc[0])


def test_combined_return_reconciles_with_weights_times_returns():
    """combined[t] must equal weights[t] · realised returns[t] (the held-fixed weights applied
    to the next period's actual returns)."""
    streams = _streams(50, seed=3)
    lookback = 18
    res = combine_walkforward(streams, lookback=lookback, freq="ME")
    full = pd.DataFrame({k: (1 + v.fillna(0)).resample("ME").prod() - 1 for k, v in streams.items()}).dropna()
    aligned = full.loc[res.combined.index]
    recomputed = (res.weights.values * aligned.values).sum(axis=1)
    assert np.allclose(recomputed, res.combined.values)


# --------------------------------------------------------------------------- #
# Property 3 — valid weights (non-negative, sum to 1)
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_nonnegative_long_only():
    res = combine_walkforward(_streams(72, seed=5), lookback=24, freq="ME", long_only=True)
    w = res.weights.values
    assert np.all(w >= -1e-12), "long-only weights must be non-negative"
    assert np.allclose(w.sum(axis=1), 1.0), "every period's weights must sum to 1"


def test_mean_weights_stats_match_weight_frame():
    res = combine_walkforward(_streams(60, seed=8), lookback=24, freq="ME")
    for c, m in res.stats["mean_weights"].items():
        assert np.isclose(m, res.weights[c].mean())


# --------------------------------------------------------------------------- #
# Property 1 — NO LOOK-AHEAD: a future return cannot move an earlier weight
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_return_does_not_change_earlier_weights():
    lookback = 24
    base = _streams(60, seed=11)
    res0 = combine_walkforward(base, lookback=lookback, freq="ME")

    # Perturb only the VERY LAST period of stream "a" by a large shock.
    perturbed = {k: v.copy() for k, v in base.items()}
    perturbed["a"].iloc[-1] += 5.0
    res1 = combine_walkforward(perturbed, lookback=lookback, freq="ME")

    # Every weight row EXCEPT the last must be byte-identical: the final return is only ever
    # in the *applied* period, never in any estimation window that produced an earlier weight.
    pd.testing.assert_frame_equal(res1.weights.iloc[:-1], res0.weights.iloc[:-1])
    # The last weight row is still produced from a window ending BEFORE the perturbed period,
    # so even it is unchanged — the shock only moves the realised combined return of that period.
    pd.testing.assert_frame_equal(res1.weights, res0.weights)
    assert res1.combined.iloc[-1] != res0.combined.iloc[-1]
    assert np.allclose(res1.combined.iloc[:-1].values, res0.combined.iloc[:-1].values)


def test_weight_at_t_equals_max_sharpe_of_strictly_prior_window():
    """Direct check that period t's weights are the tangency weights of window [t-lookback, t-1]."""
    lookback = 24
    streams = _streams(60, seed=2)
    res = combine_walkforward(streams, lookback=lookback, freq="ME")
    full = pd.DataFrame({k: (1 + v.fillna(0)).resample("ME").prod() - 1 for k, v in streams.items()}).dropna()
    # Check a few interior periods.
    for t in (lookback, lookback + 5, len(full) - 1):
        window = full.iloc[t - lookback:t]
        expected = _max_sharpe_weights(window, long_only=True)
        got = res.weights.loc[full.index[t]].values
        assert np.allclose(got, expected)


# --------------------------------------------------------------------------- #
# Guards & method variants
# --------------------------------------------------------------------------- #
def test_raises_when_not_enough_history():
    streams = _streams(20)
    with pytest.raises(ValueError):
        combine_walkforward(streams, lookback=24, freq="ME")


def test_raises_on_degenerate_lookback():
    with pytest.raises(ValueError):
        combine_walkforward(_streams(40), lookback=1, freq="ME")


def test_equal_and_risk_parity_methods_are_valid_weights():
    for method in ("equal", "risk_parity"):
        res = combine_walkforward(_streams(60, seed=4), lookback=24, freq="ME", method=method)
        w = res.weights.values
        assert np.all(w >= -1e-12)
        assert np.allclose(w.sum(axis=1), 1.0)
    # equal-weight rows are literally 1/N every period.
    eq = combine_walkforward(_streams(60, seed=4), lookback=24, freq="ME", method="equal")
    assert np.allclose(eq.weights.values, 1.0 / eq.weights.shape[1])


def test_combine_untouched_still_works():
    """Sanity: the new code path does not break the existing in-sample combine()."""
    res = combine(_streams(60, seed=6), method="max_sharpe")
    w = np.array(list(res.weights.values()))
    assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)
