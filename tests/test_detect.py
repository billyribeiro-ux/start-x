"""An injected spike must be detected; a calm series must not over-trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.data.prices import _enrich
from startx.events.detect import compute_signals, detect_events


def _synthetic(n=200, spike_idx=150, spike=0.20, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    rets[spike_idx] = spike
    close = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1_000_000.0, "vwap": close, "changePercent": 0.0,
    })
    return _enrich(df), dates[spike_idx]


def test_injected_spike_is_detected():
    prices, spike_date = _synthetic()
    signals = compute_signals(prices, market=None, window=60, min_obs=30)
    events = detect_events(signals, ar_threshold=4.0)
    assert spike_date in set(events["date"]), "the 20% jump should be flagged"
    row = events[events["date"] == spike_date].iloc[0]
    assert row["direction"] == "up"
    assert row["ar_z"] > 4.0


def test_calm_series_does_not_overtrigger():
    rng = np.random.default_rng(1)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close,
                       "close": close, "volume": 1e6, "vwap": close, "changePercent": 0.0})
    signals = compute_signals(_enrich(df), market=None, window=60, min_obs=30)
    events = detect_events(signals, ar_threshold=4.0)
    # With 4-sigma threshold on Gaussian noise, near-zero events expected.
    assert len(events) <= 2
