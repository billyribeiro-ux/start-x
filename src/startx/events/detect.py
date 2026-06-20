"""Detect statistically significant spikes / sell-offs via abnormal returns.

Abnormal return (AR) = actual return minus the return expected from a market model
(single-stock) or from the trailing mean (index/ETF). All model parameters and the
volatility used for z-scoring are estimated on a *trailing, lagged* window, so detection
at day t never peeks at day t — this is what keeps it point-in-time honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 120
DEFAULT_MIN_OBS = 60
FORWARD_HORIZONS = (1, 3, 5, 10)


def compute_signals(
    prices: pd.DataFrame,
    market: pd.DataFrame | None = None,
    window: int = DEFAULT_WINDOW,
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.DataFrame:
    """Return ``prices`` enriched with abnormal-return and volume signals."""
    if prices.empty:
        return prices
    out = prices.reset_index(drop=True).copy()
    r = out["log_ret"]

    if market is not None and not market.empty:
        mkt = market[["date", "log_ret"]].rename(columns={"log_ret": "mkt_ret"})
        out = out.merge(mkt, on="date", how="left")
        m = out["mkt_ret"]
        cov = r.rolling(window, min_periods=min_obs).cov(m)
        var = m.rolling(window, min_periods=min_obs).var()
        beta = cov / var
        alpha = (
            r.rolling(window, min_periods=min_obs).mean()
            - beta * m.rolling(window, min_periods=min_obs).mean()
        )
        expected = alpha.shift(1) + beta.shift(1) * out["mkt_ret"]
        out["beta"] = beta.shift(1)
        out["ar"] = r - expected
    else:
        out["mkt_ret"] = np.nan
        out["ar"] = r - r.rolling(window, min_periods=min_obs).mean().shift(1)

    out["ar_sigma"] = out["ar"].rolling(window, min_periods=min_obs).std().shift(1)
    out["ar_z"] = out["ar"] / out["ar_sigma"]
    out["ar_pct"] = (np.exp(out["ar"]) - 1.0) * 100.0

    logv = np.log(out["volume"].astype(float).replace(0, np.nan))
    v_mu = logv.rolling(60, min_periods=20).mean().shift(1)
    v_sd = logv.rolling(60, min_periods=20).std().shift(1)
    out["vol_z"] = (logv - v_mu) / v_sd

    # Forward realized moves — DESCRIPTIVE ONLY (uses future data); never a model feature.
    close = out["close"].astype(float)
    for h in FORWARD_HORIZONS:
        out[f"fwd_ret_{h}"] = (close.shift(-h) / close - 1.0) * 100.0
    return out


def detect_events(
    signals: pd.DataFrame, ar_threshold: float = 2.5, vol_threshold: float | None = None
) -> pd.DataFrame:
    """Subset of ``signals`` whose abnormal return is significant (|z| >= threshold)."""
    if signals.empty:
        return signals
    sig = signals["ar_z"].abs() >= ar_threshold
    if vol_threshold is not None:
        sig &= signals["vol_z"].fillna(0) >= vol_threshold
    events = signals[sig & signals["ar_z"].notna()].copy()
    events["direction"] = np.where(events["ar"] >= 0, "up", "down")
    events["ret_pct"] = events["ret"] * 100.0
    events["abs_ar_pct"] = events["ar_pct"].abs()
    cols = [
        "date", "direction", "ret_pct", "ar_pct", "abs_ar_pct", "ar_z", "vol_z", "gap",
        "close", "volume", "beta", *[f"fwd_ret_{h}" for h in FORWARD_HORIZONS],
    ]
    cols = [c for c in cols if c in events.columns]
    return events[cols].sort_values("date").reset_index(drop=True)
