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

from qedge.config import FeatureConfig, LabelingConfig, get_config
from qedge.data.boundary import BoundaryKind, InformationBoundary
from qedge.features.base import REGISTRY, FeatureSpec
from qedge.features.fracdiff import frac_diff_ffd, fracdiff_boundary

__all__ = [
    "atr_pct",
    "dist_from_sma",
    "downside_semidev",
    "fracdiff_logclose",
    "ibs",
    "momentum",
    "overnight_gap",
    "realized_vol",
    "return_skew",
    "rsi",
    "trailing_return",
    "vol_of_vol",
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
#: Longer-horizon feature windows (trading days), spanning distinct economic axes
#: from the short lookbacks above: ~6-month momentum, ~quarter vol / semidev / skew,
#: the 200-day trend SMA, and the inner/outer windows of vol-of-vol. Named (not
#: bare) constants with a stated rationale, per the no-magic-numbers rule.
_MOM_LONG_WINDOW: Final[int] = 126
_VOL_LONG_WINDOW: Final[int] = 63
#: Trend SMA window (~5 months). Kept below the 200-day classic so the warm-up is
#: proportionate to a short-swing book and does not strand most of a moderate history.
_SMA_TREND_WINDOW: Final[int] = 100
_VOV_INNER_WINDOW: Final[int] = 21
_VOV_OUTER_WINDOW: Final[int] = 63
_SEMIDEV_WINDOW: Final[int] = 63
_ATR_PERIOD: Final[int] = 14
_SKEW_WINDOW: Final[int] = 63


def _config_labeling() -> LabelingConfig:
    """Return the labeling config section (source of the vol/momentum windows)."""
    return get_config().labeling


def _config_features() -> FeatureConfig:
    """Return the features config section (source of the fracdiff feature params)."""
    return get_config().features


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


def dist_from_sma(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Distance of the close from its trailing ``window``-bar SMA: ``close/SMA - 1``.

    A trend / stretch signal: positive when price is extended above its moving
    average, negative when below. The SMA is a trailing rolling mean (bars ``<= t``
    only); the first ``window-1`` rows are ``NaN``.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    close = _close(df)
    sma = close.rolling(window=window).mean()
    return close / sma - 1.0


def vol_of_vol(df: pd.DataFrame, inner: int, outer: int) -> pd.Series[float]:
    """Volatility-of-volatility: trailing ``outer``-bar std of the rolling ``inner``-bar
    realized vol of daily log returns. Captures vol clustering/instability. Pure
    trailing windows; warm-up rows are ``NaN``.
    """
    if inner < 2 or outer < 2:
        raise ValueError("inner and outer windows must be >= 2")
    close = _close(df)
    log_ret = np.log(close / close.shift(1))
    inner_vol = log_ret.rolling(window=inner).std(ddof=1)
    return inner_vol.rolling(window=outer).std(ddof=1)


def downside_semidev(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Annualised downside semi-deviation of trailing daily log returns over ``window``.

    ``sqrt(mean(min(r, 0)^2) * 252)`` — penalises only adverse moves. Pure trailing
    window (bars ``<= t``); warm-up rows are ``NaN``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    close = _close(df)
    log_ret = np.log(close / close.shift(1))
    downside_sq = log_ret.clip(upper=0.0) ** 2
    mean_sq = downside_sq.rolling(window=window).mean()
    return np.sqrt(mean_sq * float(_TRADING_DAYS_PER_YEAR))


def atr_pct(df: pd.DataFrame, period: int) -> pd.Series[float]:
    """Average True Range over ``period`` bars as a fraction of the close.

    True range uses the *prior* close (read via an explicit one-bar shift), so the
    value at ``t`` uses only bars ``<= t``. ATR is the trailing rolling mean of the
    true range; the result is divided by the close to be scale-free. Warm-up rows
    are ``NaN``.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = _close(df)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return (atr / close).astype("float64")


def return_skew(df: pd.DataFrame, window: int) -> pd.Series[float]:
    """Trailing skewness of daily log returns over ``window`` bars.

    Distributional asymmetry of recent returns (negative => crash-prone tail).
    Pure trailing rolling skew; warm-up rows are ``NaN``.
    """
    if window < 3:
        raise ValueError("window must be >= 3 for a skewness estimate")
    close = _close(df)
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window).skew()


def fracdiff_logclose(df: pd.DataFrame, d: float, threshold: float) -> pd.Series[float]:
    """Fractionally-differenced log close (fixed-width FFD) — stationary with memory.

    Delegates to :func:`qedge.features.fracdiff.frac_diff_ffd`, whose fixed backward
    window is the point-in-time guarantee: the value at ``t`` depends only on log
    closes at ``<= t``. Leading rows (insufficient window) are ``NaN``.
    """
    log_close = np.log(_close(df))
    return frac_diff_ffd(log_close, d, threshold).astype("float64")


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
    features_cfg = _config_features()
    vol_window = labeling.short_vol_span
    momentum_window = labeling.long_horizon_days
    fd_d = features_cfg.fracdiff_feature_d
    fd_thresh = features_cfg.fracdiff_weight_threshold

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
        FeatureSpec(
            name=f"dist_sma_{_SMA_TREND_WINDOW}",
            fn=lambda df: dist_from_sma(df, _SMA_TREND_WINDOW),
            boundary=_trailing(
                _SMA_TREND_WINDOW,
                f"close/SMA({_SMA_TREND_WINDOW})-1 trend stretch; trailing rolling "
                "mean over bars <= t.",
            ),
        ),
        FeatureSpec(
            name=f"mom_{_MOM_LONG_WINDOW}d",
            fn=lambda df: momentum(df, _MOM_LONG_WINDOW),
            boundary=_trailing(
                _MOM_LONG_WINDOW,
                f"Trailing {_MOM_LONG_WINDOW}-bar (~6-month) momentum; bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"rvol_{_VOL_LONG_WINDOW}d",
            fn=lambda df: realized_vol(df, _VOL_LONG_WINDOW),
            boundary=_trailing(
                _VOL_LONG_WINDOW + 1,
                f"Annualised realized vol of the trailing {_VOL_LONG_WINDOW} daily "
                "log returns ending at t; bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"volvol_{_VOV_INNER_WINDOW}_{_VOV_OUTER_WINDOW}",
            fn=lambda df: vol_of_vol(df, _VOV_INNER_WINDOW, _VOV_OUTER_WINDOW),
            boundary=_trailing(
                _VOV_INNER_WINDOW + _VOV_OUTER_WINDOW,
                f"Vol-of-vol: {_VOV_OUTER_WINDOW}-bar std of the {_VOV_INNER_WINDOW}-bar "
                "realized vol; nested trailing windows <= t.",
            ),
        ),
        FeatureSpec(
            name=f"semidev_{_SEMIDEV_WINDOW}d",
            fn=lambda df: downside_semidev(df, _SEMIDEV_WINDOW),
            boundary=_trailing(
                _SEMIDEV_WINDOW + 1,
                f"Annualised downside semi-deviation over the trailing "
                f"{_SEMIDEV_WINDOW} daily log returns; bars <= t only.",
            ),
        ),
        FeatureSpec(
            name=f"atr_pct_{_ATR_PERIOD}",
            fn=lambda df: atr_pct(df, _ATR_PERIOD),
            boundary=_trailing(
                _ATR_PERIOD + 1,
                f"ATR({_ATR_PERIOD})/close; true range uses the lagged prior close, "
                "so the value at t uses bars <= t.",
            ),
        ),
        FeatureSpec(
            name=f"skew_{_SKEW_WINDOW}d",
            fn=lambda df: return_skew(df, _SKEW_WINDOW),
            boundary=_trailing(
                _SKEW_WINDOW + 1,
                f"Trailing {_SKEW_WINDOW}-bar skewness of daily log returns; "
                "bars <= t only.",
            ),
        ),
        FeatureSpec(
            name="fracdiff_logclose",
            fn=lambda df: fracdiff_logclose(df, fd_d, fd_thresh),
            boundary=fracdiff_boundary(fd_d, fd_thresh),
        ),
    ]
    for spec in specs:
        REGISTRY.register(spec)


_register_all()
