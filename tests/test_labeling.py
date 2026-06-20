"""Triple-barrier labeling & sample-weight tests on deterministic synthetic paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.labeling.config import HORIZONS, label_horizon
from startx.labeling.triple_barrier import daily_vol, triple_barrier_labels
from startx.labeling.weights import (
    average_uniqueness,
    num_concurrent_events,
    sample_weights,
)


def _ohlc(closes: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    """Build a minimal prices DataFrame; high/low hug close so touches are exact."""
    close = np.asarray(closes, dtype=float)
    n = len(close)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,  # high == close: an upper touch needs close >= upper
            "low": close,   # low == close: a lower touch needs close <= lower
            "close": close,
            "volume": 1_000_000.0,
            "ret": np.nan,
            "log_ret": np.nan,
        }
    )


# Fixed barrier vol so expected touch bars are computable by hand.
FIXED_VOL = 0.02


def _const_vol(prices: pd.DataFrame, v: float = FIXED_VOL) -> pd.Series:
    return pd.Series(v, index=prices.index, dtype="float64")


def test_monotonic_uptrend_hits_pt_first_at_correct_bar():
    # +1%/bar uptrend; pt_mult=1.5, vol=0.02 -> upper = c0 * 1.03.
    # close[k]/close[0] = 1.01**k; need 1.01**k >= 1.03 -> k = 3 (1.01**3 = 1.0303).
    closes = [100.0 * (1.01 ** k) for k in range(30)]
    prices = _ohlc(closes)
    out = triple_barrier_labels(
        prices, horizon_days=10, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    row = out.iloc[0]
    assert row["label"] == 1
    assert row["touch"] == "pt"
    # First bar crossing the upper barrier is index 3.
    assert row["t1"] == prices["date"].iloc[3]
    assert row["ret"] > 0
    assert row["upper"] > row["lower"] > 0


def test_monotonic_downtrend_hits_sl_first():
    # -1%/bar downtrend; lower = c0 * (1 - 1.5*0.02) = c0 * 0.97.
    # 0.99**k <= 0.97 -> k = 4 (0.99**3 = 0.97030 > 0.97; 0.99**4 = 0.96060 <= 0.97).
    closes = [100.0 * (0.99 ** k) for k in range(30)]
    prices = _ohlc(closes)
    out = triple_barrier_labels(
        prices, horizon_days=10, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    row = out.iloc[0]
    assert row["label"] == -1
    assert row["touch"] == "sl"
    assert row["t1"] == prices["date"].iloc[4]
    assert row["ret"] < 0


def test_flat_within_barriers_times_out_at_vertical():
    # Oscillation of +/-0.5% stays inside +/-3% barriers for the whole horizon.
    closes = [100.0 * (1.0 + 0.005 * (1 if k % 2 else -1)) for k in range(30)]
    closes[0] = 100.0
    prices = _ohlc(closes)
    horizon = 10
    out = triple_barrier_labels(
        prices, horizon_days=horizon, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    row = out.iloc[0]
    assert row["label"] == 0
    assert row["touch"] == "vert"
    # Vertical barrier sits exactly horizon rows ahead of entry.
    assert row["t1"] == prices["date"].iloc[horizon]


def test_same_bar_double_touch_resolves_to_stop():
    # Bar 1 gaps wide: high above upper AND low below lower -> conservative -> sl.
    closes = [100.0, 100.0, 100.0]
    prices = _ohlc(closes)
    prices.loc[1, "high"] = 110.0  # >> upper (103)
    prices.loc[1, "low"] = 90.0    # << lower (97)
    out = triple_barrier_labels(
        prices, horizon_days=2, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    row = out.iloc[0]
    assert row["label"] == -1
    assert row["touch"] == "sl"
    assert row["t1"] == prices["date"].iloc[1]


def test_truncated_horizon_labels_as_timeout_at_last_date():
    # Horizon longer than remaining data -> t1 clamps to last row, label 0.
    closes = [100.0, 100.5, 100.2, 100.1]
    prices = _ohlc(closes)
    out = triple_barrier_labels(
        prices, horizon_days=10, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    row = out.iloc[0]
    assert row["label"] == 0
    assert row["touch"] == "vert"
    assert row["t1"] == prices["date"].iloc[-1]


def test_output_schema_and_index():
    prices = _ohlc([100.0 * (1.01 ** k) for k in range(20)])
    out = triple_barrier_labels(
        prices, horizon_days=5, pt_mult=1.5, sl_mult=1.5, vol=_const_vol(prices)
    )
    assert list(out.columns) == ["date", "t1", "label", "ret", "touch", "upper", "lower"]
    assert list(out.index) == list(range(len(out)))
    assert len(out) == len(prices)
    assert set(out["label"].unique()) <= {-1, 0, 1}
    assert set(out["touch"].unique()) <= {"pt", "sl", "vert"}


def test_daily_vol_positive_and_finite():
    rng = np.random.default_rng(0)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))
    vol = daily_vol(pd.Series(closes), span=21)
    assert len(vol) == len(closes)
    assert np.all(np.isfinite(vol.to_numpy()))
    assert np.all(vol.to_numpy() > 0)


def test_label_horizon_presets_run():
    prices = _ohlc([100.0 * (1.001 ** k) for k in range(120)])
    for name in HORIZONS:
        out = label_horizon(prices, name)
        assert list(out.columns) == [
            "date", "t1", "label", "ret", "touch", "upper", "lower"
        ]
        assert len(out) == len(prices)


def _bars(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_average_uniqueness_fully_overlapping_is_half():
    bars = _bars(5)
    # Two labels live on bars 0..3 simultaneously -> concurrency 2 -> uniqueness 0.5.
    t1 = pd.Series([bars[3], bars[3]], index=[bars[0], bars[0]])
    avg = average_uniqueness(bars, t1)
    assert np.allclose(avg.to_numpy(), 0.5)


def test_average_uniqueness_non_overlapping_is_one():
    bars = _bars(6)
    # Label A on bars 0..2, label B on bars 3..5 — never concurrent -> uniqueness 1.0.
    t1 = pd.Series([bars[2], bars[5]], index=[bars[0], bars[3]])
    avg = average_uniqueness(bars, t1)
    assert np.allclose(avg.to_numpy(), 1.0)


def test_num_concurrent_events_counts():
    bars = _bars(5)
    t1 = pd.Series([bars[3], bars[3]], index=[bars[0], bars[0]])
    conc = num_concurrent_events(bars, t1)
    assert list(conc.loc[bars[0]:bars[3]]) == [2, 2, 2, 2]
    assert conc.loc[bars[4]] == 0


def test_sample_weights_mean_is_one():
    bars = _bars(6)
    t1 = pd.Series([bars[2], bars[5]], index=[bars[0], bars[3]])
    w = sample_weights(bars, t1)
    assert np.isclose(w.mean(), 1.0)

    # With return attribution.
    ret = pd.Series([0.05, -0.02], index=[bars[0], bars[3]])
    w_ret = sample_weights(bars, t1, ret=ret)
    assert np.isclose(w_ret.mean(), 1.0)

    # With time decay.
    w_decay = sample_weights(bars, t1, time_decay=0.5)
    assert np.isclose(w_decay.mean(), 1.0)


def test_sample_weights_time_decay_downweights_old_labels():
    bars = _bars(9)
    t1 = pd.Series(
        [bars[2], bars[5], bars[8]], index=[bars[0], bars[3], bars[6]]
    )
    w = sample_weights(bars, t1, time_decay=0.2)
    # Oldest label should carry less weight than the newest under decay.
    assert w.loc[bars[0]] < w.loc[bars[6]]
    assert np.isclose(w.mean(), 1.0)
