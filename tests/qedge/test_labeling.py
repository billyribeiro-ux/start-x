"""Tests for the qedge labeling wrappers (config presets over startx primitives)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qedge.config import QedgeConfig
from qedge.labeling import (
    LABEL_COLUMNS,
    label_long,
    label_short,
    uniqueness_weights,
)


def _flat_prices(n: int) -> pd.DataFrame:
    """A perfectly flat OHLC series so no profit/stop barrier is ever touched."""
    dates = pd.bdate_range("2019-01-02", periods=n)
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _trending_prices(n: int, seed: int = 7) -> pd.DataFrame:
    """A noisy OHLC series with intrabar range, for the weight/schema checks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _vert_horizon(labels: pd.DataFrame, prices: pd.DataFrame) -> set[int]:
    """Trading-row gaps (t1 - entry) for every vertical-barrier (timeout) label."""
    pos = {d: i for i, d in enumerate(pd.to_datetime(prices["date"]))}
    return {
        pos[pd.Timestamp(row.t1)] - pos[pd.Timestamp(row.date)]
        for row in labels.itertuples(index=False)
        if row.touch == "vert"
    }


def test_label_short_schema_matches_locked_columns() -> None:
    labels = label_short(_trending_prices(200))
    assert list(labels.columns) == LABEL_COLUMNS
    assert set(labels["label"].unique()) <= {-1, 0, 1}
    assert pd.api.types.is_datetime64_any_dtype(labels["t1"])


def test_label_long_schema_matches_locked_columns() -> None:
    labels = label_long(_trending_prices(300))
    assert list(labels.columns) == LABEL_COLUMNS


def test_short_horizon_cap_equals_config(cfg: QedgeConfig) -> None:
    """Every interior timeout label spans exactly the short horizon (10 days)."""
    n = 200
    prices = _flat_prices(n)
    # A wide explicit vol makes the barriers far from a flat price, so every
    # entry must time out at the vertical barrier -> gap == horizon exactly.
    vol = pd.Series(np.full(n, 0.05))
    labels = label_short(prices, vol=vol)
    horizon = cfg.labeling.short_horizon_days
    assert horizon == 10  # locked short-swing cap
    assert (labels["touch"] == "vert").all()
    interior = _vert_horizon(labels.iloc[: n - horizon], prices)
    assert interior == {horizon}


def test_long_horizon_cap_equals_config(cfg: QedgeConfig) -> None:
    """Every interior timeout label spans exactly the long horizon (63 days)."""
    n = 400
    prices = _flat_prices(n)
    vol = pd.Series(np.full(n, 0.05))
    labels = label_long(prices, vol=vol)
    horizon = cfg.labeling.long_horizon_days
    assert horizon == 63  # locked long-swing cap
    assert (labels["touch"] == "vert").all()
    interior = _vert_horizon(labels.iloc[: n - horizon], prices)
    assert interior == {horizon}


def test_short_and_long_horizons_differ() -> None:
    """The only thing distinguishing the two books is the max-hold clock."""
    prices = _flat_prices(300)
    vol = pd.Series(np.full(300, 0.05))
    short_caps = _vert_horizon(label_short(prices, vol=vol).iloc[:200], prices)
    long_caps = _vert_horizon(label_long(prices, vol=vol).iloc[:200], prices)
    assert short_caps == {10}
    assert long_caps == {63}


def test_uniqueness_weights_normalised_to_mean_one() -> None:
    prices = _trending_prices(250)
    labels = label_short(prices)
    t1 = pd.Series(
        pd.to_datetime(labels["t1"]).to_numpy(),
        index=pd.to_datetime(labels["date"]),
    )
    bar_dates = pd.DatetimeIndex(pd.to_datetime(prices["date"]))
    weights = uniqueness_weights(bar_dates, t1)
    assert len(weights) == len(labels)
    assert (weights > 0).all()
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-9)


def test_metalabel_runs_with_a_trivial_estimator() -> None:
    """metalabel() delegates to startx walk_forward_metalabel (needs sklearn)."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    from qedge.labeling import metalabel

    rng = np.random.default_rng(7)
    n = 120
    entries = pd.to_datetime(pd.bdate_range("2019-01-02", periods=n))
    # Resolve quickly so the point-in-time firewall has training history.
    t1 = pd.Series(entries + pd.Timedelta(days=2), index=range(n))
    entry_times = pd.Series(entries, index=range(n))
    features = pd.DataFrame(
        {"f0": rng.normal(size=n), "f1": rng.normal(size=n)}, index=range(n)
    )
    # A learnable label so both classes are present in every training window.
    labels = pd.Series((features["f0"] > 0).astype(int).to_numpy(), index=range(n))

    out = metalabel(
        features,
        labels,
        entry_times,
        t1,
        lambda: LogisticRegression(max_iter=200),
        train_min=20,
        test_span=10,
    )
    assert list(out.columns) == ["entry_time", "meta_prob_win", "y_true", "y_pred"]
    assert not out.empty
    assert out["meta_prob_win"].between(0.0, 1.0).all()
