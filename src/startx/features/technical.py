"""Technical (price/volume) features — strictly trailing, point-in-time.

Every column is a function of OHLCV up to and including day ``t``: rolling windows are causal
(they end at ``t``), and any indicator that needs yesterday's value uses ``shift`` so nothing
peeks ahead. Raw OHLC levels are intentionally NOT emitted (non-stationary, leak price scale),
nor are the forward-return columns produced by ``compute_signals``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_HORIZONS = (1, 3, 5, 10, 21, 63, 126, 252)
VOL_WINDOWS = (10, 21, 63)
SMA_WINDOWS = (20, 50, 200)
EMA_SPANS = (12, 26)
ATR_WINDOW = 14
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
HIGH_LOW_WINDOW = 252  # ~52 weeks
DRAWDOWN_WINDOW = 252
_ANNUALIZE = float(np.sqrt(252.0))


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI (EMA of gains/losses). Causal: uses only past/current closes."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # All gains (no losses) -> RSI 100; flat (no moves) -> neutral 50.
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range as a fraction of close (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return atr / close


def technical_features(signals: pd.DataFrame) -> pd.DataFrame:
    """Build the trailing technical feature frame from a ``compute_signals`` output.

    ``signals`` must carry at least ``date, open, high, low, close, volume`` (and optionally
    ``ar_z, vol_z, gap`` which are passed through). Returns a frame keyed by ``date`` with one
    row per input row; no raw OHLC and no ``fwd_*`` columns.
    """
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["date"])

    df = signals.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    log_close = np.log(close)

    out = pd.DataFrame({"date": df["date"].values})

    # -- momentum: trailing simple returns over each horizon --------------------------------
    for h in MOMENTUM_HORIZONS:
        out[f"mom_{h}"] = (close / close.shift(h) - 1.0).values

    # -- realized volatility: std of daily log returns, annualized ---------------------------
    daily_log_ret = log_close.diff()
    for w in VOL_WINDOWS:
        out[f"vol_{w}"] = (
            daily_log_ret.rolling(w, min_periods=w).std() * _ANNUALIZE
        ).values

    # -- ATR(14) as fraction of price --------------------------------------------------------
    out[f"atr_{ATR_WINDOW}"] = _atr(high, low, close, ATR_WINDOW).values

    # -- distance from SMAs + SMA slopes -----------------------------------------------------
    for w in SMA_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        out[f"dist_sma_{w}"] = (close / sma - 1.0).values
        # 5-day normalized slope of the SMA (trailing): how fast the average is moving.
        out[f"sma_slope_{w}"] = (sma / sma.shift(5) - 1.0).values

    # -- distance from EMAs ------------------------------------------------------------------
    for span in EMA_SPANS:
        ema = close.ewm(span=span, min_periods=span, adjust=False).mean()
        out[f"dist_ema_{span}"] = (close / ema - 1.0).values

    # -- RSI(14) -----------------------------------------------------------------------------
    out[f"rsi_{RSI_WINDOW}"] = _wilder_rsi(close, RSI_WINDOW).values

    # -- Bollinger %b(20) --------------------------------------------------------------------
    mid = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    sd = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd
    width = (upper - lower)
    pct_b = (close - lower) / width.where(width != 0, np.nan)
    out[f"bollinger_pctb_{BOLLINGER_WINDOW}"] = pct_b.values

    # -- 52w high / low distance (window inclusive of today) ---------------------------------
    roll_high = high.rolling(HIGH_LOW_WINDOW, min_periods=1).max()
    roll_low = low.rolling(HIGH_LOW_WINDOW, min_periods=1).min()
    out["dist_52w_high"] = (close / roll_high - 1.0).values
    out["dist_52w_low"] = (close / roll_low - 1.0).values

    # -- drawdown vs trailing-252 max close --------------------------------------------------
    roll_max_close = close.rolling(DRAWDOWN_WINDOW, min_periods=1).max()
    out["drawdown_252"] = (close / roll_max_close - 1.0).values

    # -- passthrough signal features (already PIT in compute_signals) ------------------------
    for col in ("ar_z", "vol_z", "gap"):
        out[col] = df[col].values if col in df.columns else np.nan

    return out
