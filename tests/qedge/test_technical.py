"""Value-correctness tests for the trailing technical features.

These pin the *meaning* of each feature on small, hand-constructed frames where
the expected value is computable by inspection: RSI bounds and direction, IBS
bounds and corner cases, the sign of momentum / trailing return on a monotonic
series, realized-vol behaviour, and the lagged overnight gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qedge.features.technical import (
    ibs,
    momentum,
    overnight_gap,
    realized_vol,
    rsi,
    trailing_return,
)


def _frame(
    close: list[float],
    *,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
) -> pd.DataFrame:
    """Build an OHLCV frame on a business-day index from explicit columns.

    Defaults: open == close, high/low bracket the close by +/-1 so high>low.
    """
    n = len(close)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    close_arr = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "open": np.asarray(open_, dtype="float64") if open_ is not None else close_arr,
            "high": (
                np.asarray(high, dtype="float64")
                if high is not None
                else close_arr + 1.0
            ),
            "low": (
                np.asarray(low, dtype="float64")
                if low is not None
                else close_arr - 1.0
            ),
            "close": close_arr,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# trailing_return / momentum
# --------------------------------------------------------------------------- #
def test_trailing_return_known_value() -> None:
    frame = _frame([100.0, 110.0, 121.0])
    out = trailing_return(frame, 1)
    # First row NaN (no prior), then +10%, +10%.
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1], 0.10)
    np.testing.assert_allclose(out.iloc[2], 0.10)


def test_trailing_return_multi_day_window() -> None:
    frame = _frame([100.0, 105.0, 110.0, 120.0])
    out = trailing_return(frame, 2)
    # Row 2 vs row 0: 110/100 - 1; row 3 vs row 1: 120/105 - 1.
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2], 110.0 / 100.0 - 1.0)
    np.testing.assert_allclose(out.iloc[3], 120.0 / 105.0 - 1.0)


def test_trailing_return_window_must_be_positive() -> None:
    frame = _frame([100.0, 101.0])
    raised = False
    try:
        trailing_return(frame, 0)
    except ValueError:
        raised = True
    assert raised


def test_momentum_sign_on_monotonic_series() -> None:
    rising = _frame([float(p) for p in range(100, 130)])
    falling = _frame([float(p) for p in range(130, 100, -1)])
    mom_up = momentum(rising, 10).dropna()
    mom_down = momentum(falling, 10).dropna()
    # Monotonic up -> strictly positive momentum; down -> strictly negative.
    assert (mom_up > 0.0).all()
    assert (mom_down < 0.0).all()


def test_momentum_matches_trailing_return() -> None:
    frame = _frame([float(p) for p in range(100, 140)])
    pd.testing.assert_series_equal(momentum(frame, 7), trailing_return(frame, 7))


# --------------------------------------------------------------------------- #
# realized_vol
# --------------------------------------------------------------------------- #
def test_realized_vol_zero_on_constant_series() -> None:
    frame = _frame([100.0] * 30)
    out = realized_vol(frame, 21)
    # Constant price -> zero log returns -> zero realized vol (after warm-up).
    np.testing.assert_allclose(out.dropna().to_numpy(), 0.0, atol=1e-12)


def test_realized_vol_positive_and_annualised() -> None:
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(100)))
    frame = _frame(prices.tolist())
    out = realized_vol(frame, 21).dropna()
    assert (out > 0.0).all()
    # Cross-check the last point against a direct annualised std of log returns.
    log_ret = np.log(frame["close"] / frame["close"].shift(1))
    tail = log_ret.iloc[-21:]
    expected = float(tail.std(ddof=1) * np.sqrt(252))
    np.testing.assert_allclose(out.iloc[-1], expected)


def test_realized_vol_window_must_be_at_least_two() -> None:
    frame = _frame([100.0, 101.0, 102.0])
    raised = False
    try:
        realized_vol(frame, 1)
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# rsi
# --------------------------------------------------------------------------- #
def test_rsi_within_bounds() -> None:
    rng = np.random.default_rng(11)
    prices = 100.0 + np.cumsum(rng.standard_normal(200))
    frame = _frame(prices.tolist())
    out = rsi(frame, 14).dropna()
    assert (out >= 0.0).all()
    assert (out <= 100.0).all()


def test_rsi_all_gains_is_100() -> None:
    frame = _frame([float(p) for p in range(100, 140)])
    out = rsi(frame, 14).dropna()
    # Strictly rising series: no losses -> RSI pinned at 100.
    np.testing.assert_allclose(out.to_numpy(), 100.0)


def test_rsi_all_losses_is_0() -> None:
    frame = _frame([float(p) for p in range(140, 100, -1)])
    out = rsi(frame, 14).dropna()
    np.testing.assert_allclose(out.to_numpy(), 0.0)


def test_rsi_warmup_is_nan() -> None:
    frame = _frame([float(p) for p in range(100, 140)])
    out = rsi(frame, 14)
    # First `period` rows lack a full Wilder window -> NaN.
    assert out.iloc[:14].isna().all()
    assert not out.iloc[14:].isna().any()


# --------------------------------------------------------------------------- #
# ibs
# --------------------------------------------------------------------------- #
def test_ibs_within_unit_interval() -> None:
    rng = np.random.default_rng(3)
    n = 50
    low = 90.0 + rng.random(n)
    high = low + 2.0 + rng.random(n)
    close = low + (high - low) * rng.random(n)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    out = ibs(frame)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_ibs_corner_values() -> None:
    # close==low -> 0, close==high -> 1, midpoint -> 0.5.
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [12.0, 12.0, 12.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 12.0, 11.0],
            "volume": [1e6, 1e6, 1e6],
        },
        index=pd.date_range("2019-01-01", periods=3, freq="B"),
    )
    out = ibs(frame)
    np.testing.assert_allclose(out.to_numpy(), [0.0, 1.0, 0.5])


def test_ibs_zero_range_bar_is_half() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [1e6],
        },
        index=pd.date_range("2019-01-01", periods=1, freq="B"),
    )
    out = ibs(frame)
    # high == low -> defined as the midpoint 0.5, not NaN.
    np.testing.assert_allclose(out.to_numpy(), [0.5])


# --------------------------------------------------------------------------- #
# overnight_gap
# --------------------------------------------------------------------------- #
def test_overnight_gap_uses_prior_close() -> None:
    frame = _frame([100.0, 110.0, 121.0], open_=[100.0, 105.0, 115.5])
    out = overnight_gap(frame)
    # Row 0 NaN (no prior close). Row 1: open 105 vs prior close 100 -> +5%.
    # Row 2: open 115.5 vs prior close 110 -> +5%.
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1], 105.0 / 100.0 - 1.0)
    np.testing.assert_allclose(out.iloc[2], 115.5 / 110.0 - 1.0)


def test_overnight_gap_is_lagged_not_intraday() -> None:
    # Gap must NOT use today's own close: changing only the last close leaves the
    # whole gap series unchanged (it reads open_t and close_{t-1} only).
    frame_a = _frame([100.0, 110.0, 120.0], open_=[100.0, 108.0, 118.0])
    frame_b = _frame([100.0, 110.0, 999.0], open_=[100.0, 108.0, 118.0])
    out_a = overnight_gap(frame_a)
    out_b = overnight_gap(frame_b)
    pd.testing.assert_series_equal(out_a, out_b)
