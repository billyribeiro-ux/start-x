"""Trailing technical features — pure, lagged, leakage-hardened.

Every function here is a PURE transform of an OHLCV frame returning a
:class:`pandas.Series` aligned to that frame's index, where the value at row
``t`` is built from data at rows ``<= t`` ONLY. That point-in-time discipline is
the entire reason these features are safe to feed a model; it is enforced by the
registry-wide no-lookahead canary in the test-suite.

Two leakage rules are applied consistently:

* **Trailing windows close at ``t``.** A k-day trailing return / volatility /
  momentum uses the bar at ``t`` and the ``k`` bars before it, never anything
  after ``t``. Warm-up rows that lack a full window are ``NaN``.
* **Anything referencing the *previous* bar is explicitly shifted.** The
  overnight gap compares today's open to *yesterday's* close, so the prior close
  is read via :meth:`~pandas.Series.shift` — it is information available at the
  open of ``t`` and carries no look-ahead.

Each feature is registered in :data:`qedge.features.base.REGISTRY` with an
explicit TRAILING :class:`~qedge.data.boundary.InformationBoundary`. Windows come
from :mod:`qedge.config` where a config field maps cleanly; the remaining periods
are documented module-level ``Final`` constants (the contract bans bare magic
numbers, not named constants with a stated rationale).

Only numpy, pandas and ``qedge.{config, data.boundary, features.base}`` are used.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from qedge.config import LabelingConfig, get_config
from qedge.data.boundary import BoundaryKind, InformationBoundary
from qedge.features.base import REGISTRY, FeatureSpec

__all__ = [
    "ibs",
    "momentum",
    "overnight_gap",
    "realized_vol",
    "rsi",
    "trailing_return",
]

# --- Feature periods not owned by a config field --------------------------
#: Short trailing-return horizons (trading days). 1d is the daily return; 5d ~ a
#: trading week; 21d ~ a trading month. These are conventional swing-research
#: lookbacks, kept as named constants because no config field owns them.
_RET_WINDOW_1D: Final[int] = 1
_RET_WINDOW_5D: Final[int] = 5
_RET_WINDOW_21D: Final[int] = 21
#: Wilder's RSI lookback. 14 is the canonical period from Wilder (1978).
_RSI_PERIOD: Final[int] = 14
#: RSI is expressed on a 0..100 scale; this is the constant in 100 - 100/(1+RS).
_RSI_SCALE: Final[float] = 100.0
#: Trading days per year, for annualising realized volatility.
_TRADING_DAYS_PER_YEAR: Final[int] = 252


def _config_labeling() -> LabelingConfig:
    """Return the labeling config section (source of the vol/momentum windows)."""
    return get_config().labeling


def _close(df: pd.DataFrame) -> pd.Series[float]:
    """Return the close column as a float Series aligned to ``df.index``."""
    return df["close"].astype("float64")


def trailing_return(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Trailing simple return over ``window`` bars: ``close_t / close_{t-w} - 1``.

    Uses only the close at ``t`` and the close ``window`` bars earlier, so the
    value at ``t`` reads no future data. The first ``window`` rows are ``NaN``.

    Args:
        df: OHLCV frame with a ``close`` column.
        window: Trailing horizon in bars (``> 0``).

    Returns:
        A float Series aligned to ``df.index``; leading NaNs over the warm-up.

    Raises:
        ValueError: If ``window`` is not strictly positive.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    close = _close(df)
    return close / close.shift(window) - 1.0


def realized_vol(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Annualised realized volatility of trailing daily log returns over ``window``.

    The value at ``t`` is the sample standard deviation of the ``window`` daily
    log returns ending at ``t``, scaled by ``sqrt(252)``. Pure trailing window:
    no row after ``t`` enters the estimate. Warm-up rows are ``NaN``.

    Args:
        df: OHLCV frame with a ``close`` column.
        window: Number of trailing daily returns in the estimate (``>= 2``).

    Returns:
        A float Series aligned to ``df.index``.

    Raises:
        ValueError: If ``window < 2`` (variance needs at least two points).
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a variance estimate")
    close = _close(df)
    log_ret = np.log(close / close.shift(1))
    rolled = log_ret.rolling(window=window).std(ddof=1)
    return rolled * float(np.sqrt(_TRADING_DAYS_PER_YEAR))


def momentum(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Trailing momentum: the simple return over the last ``window`` bars.

    Identical mechanics to :func:`trailing_return` but named for its role as a
    medium-horizon momentum signal (default 63d ~ one quarter). The value at ``t``
    uses only the close at ``t`` and ``window`` bars earlier.

    Args:
        df: OHLCV frame with a ``close`` column.
        window: Momentum lookback in bars (``> 0``).

    Returns:
        A float Series aligned to ``df.index``; leading NaNs over the warm-up.

    Raises:
        ValueError: If ``window`` is not strictly positive.
    """
    return trailing_return(df, window)


def rsi(df: pd.DataFrame, period: int = _RSI_PERIOD) -> pd.Series[float]:
    """Wilder's Relative Strength Index over ``period`` bars, in ``[0, 100]``.

    Average gain and average loss are Wilder-smoothed (an exponential moving
    average with ``alpha = 1/period``) over trailing close-to-close changes, so
    the value at ``t`` depends only on changes at ``<= t``. With no losses the RSI
    is ``100`` (RS is infinite); with no gains it is ``0``. The first ``period``
    rows are ``NaN`` (the first close-change is itself NaN).

    Args:
        df: OHLCV frame with a ``close`` column.
        period: Wilder lookback in bars (``> 0``). Defaults to 14.

    Returns:
        A float Series aligned to ``df.index`` with values in ``[0, 100]``.

    Raises:
        ValueError: If ``period`` is not strictly positive.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    close = _close(df)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWMA with alpha = 1/period. ``min_periods`` so the
    # warm-up rows (before a full first window) stay NaN rather than seeding off
    # one observation.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = _RSI_SCALE - _RSI_SCALE / (1.0 + rs)
    # All-gain windows (avg_loss == 0) yield rs == inf -> out already 100. But
    # all-flat / all-loss windows divide 0/0 -> NaN; resolve them explicitly:
    out = out.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), _RSI_SCALE)
    out = out.where(~((avg_gain == 0.0) & (avg_loss > 0.0)), 0.0)
    return out.astype("float64")


def ibs(df: pd.DataFrame) -> pd.Series[float]:
    """Internal Bar Strength: ``(close - low) / (high - low)`` for the bar at ``t``.

    A purely INTRADAY statistic of the bar at ``t`` — it uses that bar's own
    high/low/close and nothing from any other bar, so it is trivially trailing.
    Values lie in ``[0, 1]``: near 0 the bar closed at its low (a dip), near 1 at
    its high. On a zero-range bar (``high == low``) IBS is defined as ``0.5``
    (the bar's midpoint) rather than ``NaN``.

    Args:
        df: OHLCV frame with ``high``, ``low`` and ``close`` columns.

    Returns:
        A float Series aligned to ``df.index`` with values in ``[0, 1]``.
    """
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = _close(df)
    rng = high - low
    raw = (close - low) / rng
    # Zero-range bars (high == low) give 0/0; define IBS as the midpoint 0.5.
    return raw.where(rng != 0.0, 0.5).astype("float64")


def overnight_gap(df: pd.DataFrame) -> pd.Series[float]:
    """Overnight gap: ``open_t / close_{t-1} - 1`` (today's open vs prior close).

    The prior close is read via an explicit one-bar shift, so the value at ``t``
    uses only information available at the open of ``t`` (today's open and
    yesterday's close) — there is no look-ahead. The first row is ``NaN`` (no
    prior close).

    Args:
        df: OHLCV frame with ``open`` and ``close`` columns.

    Returns:
        A float Series aligned to ``df.index``; first row NaN.
    """
    open_ = df["open"].astype("float64")
    prev_close = _close(df).shift(1)
    return open_ / prev_close - 1.0


# --------------------------------------------------------------------------- #
# Registration. Each spec carries an explicit TRAILING InformationBoundary so
# the registry-wide canary can assert no-lookahead and a non-null boundary.
# --------------------------------------------------------------------------- #
def _trailing(lookback: int, description: str) -> InformationBoundary:
    """Build a TRAILING boundary with zero dissemination latency."""
    return InformationBoundary(
        kind=BoundaryKind.TRAILING,
        lookback_days=lookback,
        latency_days=0,
        description=description,
    )


def _register_all() -> None:
    """Register every technical feature in the package-wide REGISTRY.

    Windows that map to a config field are pulled from ``cfg.labeling``; the rest
    are the documented module-level ``Final`` constants above. Called once at
    import time.
    """
    labeling = _config_labeling()
    vol_window = labeling.short_vol_span
    momentum_window = labeling.long_horizon_days

    specs: list[FeatureSpec] = [
        FeatureSpec(
            name=f"ret_{_RET_WINDOW_1D}d",
            fn=lambda df: trailing_return(df, _RET_WINDOW_1D),
            boundary=_trailing(
                _RET_WINDOW_1D,
                f"Trailing {_RET_WINDOW_1D}-bar simple return; "
                "close_t/close_prev-1, uses bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"ret_{_RET_WINDOW_5D}d",
            fn=lambda df: trailing_return(df, _RET_WINDOW_5D),
            boundary=_trailing(
                _RET_WINDOW_5D,
                f"Trailing {_RET_WINDOW_5D}-bar simple return; uses bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"ret_{_RET_WINDOW_21D}d",
            fn=lambda df: trailing_return(df, _RET_WINDOW_21D),
            boundary=_trailing(
                _RET_WINDOW_21D,
                f"Trailing {_RET_WINDOW_21D}-bar simple return; uses bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"rvol_{vol_window}d",
            fn=lambda df: realized_vol(df, vol_window),
            boundary=_trailing(
                vol_window + 1,
                f"Annualised realized vol of the trailing {vol_window} daily log "
                "returns ending at t (cfg.labeling.short_vol_span); bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"mom_{momentum_window}d",
            fn=lambda df: momentum(df, momentum_window),
            boundary=_trailing(
                momentum_window,
                f"Trailing {momentum_window}-bar momentum "
                "(cfg.labeling.long_horizon_days); uses bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"rsi_{_RSI_PERIOD}",
            fn=lambda df: rsi(df, _RSI_PERIOD),
            boundary=_trailing(
                _RSI_PERIOD + 1,
                f"Wilder RSI({_RSI_PERIOD}) on trailing close changes <= t; "
                "in [0, 100].",
            ),
        ),
        FeatureSpec(
            name="ibs",
            fn=ibs,
            boundary=_trailing(
                0,
                "Internal Bar Strength (close-low)/(high-low) of the bar at t "
                "only; point statistic, in [0, 1].",
            ),
        ),
        FeatureSpec(
            name="overnight_gap",
            fn=overnight_gap,
            boundary=_trailing(
                1,
                "Overnight gap open_t/close_prev-1; prior close is lagged one "
                "bar, available at the open of t.",
            ),
        ),
    ]
    for spec in specs:
        REGISTRY.register(spec)


_register_all()
