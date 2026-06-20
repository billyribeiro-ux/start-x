"""Intraday volume-profile microstructure: POC / value area, VWAP, opening range, delta.

All functions take a tidy bar frame ``DataFrame[ts, open, high, low, close, volume]`` (the
output of :func:`startx.data.intraday.get_intraday`). They are pure/stateless and tolerant
of empty input, so a no-data session degrades to NaNs rather than raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_profile(
    bars: pd.DataFrame, bins: int = 50, value_area: float = 0.7
) -> dict:
    """Build a price-binned volume profile and derive POC and the value area.

    Each bar's volume is attributed to the price bin containing its *typical price*
    (``(high + low + close) / 3``). The bins span ``[min low, max high]`` across the session.

    Returns a dict:
      * ``poc_price`` — midpoint of the max-volume bin (Point Of Control).
      * ``vah`` / ``val`` — value-area high/low: starting at the POC bin we greedily add the
        richer of the two adjacent bins until ``value_area`` of total volume is covered; VAH
        and VAL are the outer edges of that contiguous region.
      * ``total_volume`` — summed volume.
      * ``profile`` — ``DataFrame[price_low, price_high, volume]`` (one row per bin).

    Empty ``bars`` (or all-zero volume) yields NaN scalars and an empty ``profile``.
    """
    empty_profile = pd.DataFrame(columns=["price_low", "price_high", "volume"])
    nan_result = {
        "poc_price": float("nan"),
        "vah": float("nan"),
        "val": float("nan"),
        "total_volume": 0.0,
        "profile": empty_profile,
    }
    if bars is None or bars.empty:
        return nan_result

    low = pd.to_numeric(bars["low"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    vol = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)

    lo, hi = float(low.min()), float(high.max())
    total = float(vol.sum())
    if not np.isfinite(lo) or not np.isfinite(hi) or total <= 0.0:
        return nan_result

    typical = ((high + low + close) / 3.0).to_numpy()
    n_bins = max(1, int(bins))
    if hi <= lo:  # flat session: single degenerate bin
        edges = np.array([lo, lo + 1e-9])
        n_bins = 1
    else:
        edges = np.linspace(lo, hi, n_bins + 1)

    # Bin index per bar; clip so the max value lands in the last bin.
    idx = np.clip(np.digitize(typical, edges) - 1, 0, n_bins - 1)
    bin_vol = np.zeros(n_bins, dtype=float)
    np.add.at(bin_vol, idx, vol.to_numpy())

    profile = pd.DataFrame(
        {"price_low": edges[:-1], "price_high": edges[1:], "volume": bin_vol}
    )

    poc_i = int(np.argmax(bin_vol))
    poc_price = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)

    # Value area: grow outward from the POC bin, always taking the richer neighbour,
    # until cumulative volume reaches `value_area` of the total.
    target = max(0.0, min(1.0, value_area)) * total
    lo_i = hi_i = poc_i
    covered = bin_vol[poc_i]
    while covered < target and (lo_i > 0 or hi_i < n_bins - 1):
        below = bin_vol[lo_i - 1] if lo_i > 0 else -np.inf
        above = bin_vol[hi_i + 1] if hi_i < n_bins - 1 else -np.inf
        if above >= below:
            hi_i += 1
            covered += bin_vol[hi_i]
        else:
            lo_i -= 1
            covered += bin_vol[lo_i]

    val = float(edges[lo_i])
    vah = float(edges[hi_i + 1])
    return {
        "poc_price": poc_price,
        "vah": vah,
        "val": val,
        "total_volume": total,
        "profile": profile,
    }


def session_vwap(bars: pd.DataFrame) -> float:
    """Volume-weighted average price over the session (typical-price weighted by volume)."""
    if bars is None or bars.empty:
        return float("nan")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    vol = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    denom = float(vol.sum())
    if denom <= 0.0:
        return float(close.mean()) if len(close) else float("nan")
    return float((typical * vol).sum() / denom)


def opening_range(bars: pd.DataFrame, minutes: int = 30) -> dict:
    """High/low of the first ``minutes`` of the session (the opening range)."""
    if bars is None or bars.empty:
        return {"high": float("nan"), "low": float("nan")}
    ts = pd.to_datetime(bars["ts"])
    start = ts.iloc[0]
    cutoff = start + pd.Timedelta(minutes=minutes)
    window = bars[ts < cutoff]
    if window.empty:
        window = bars.iloc[:1]
    return {
        "high": float(pd.to_numeric(window["high"], errors="coerce").max()),
        "low": float(pd.to_numeric(window["low"], errors="coerce").min()),
    }


def cumulative_delta(bars: pd.DataFrame) -> float:
    """Signed-volume sum: ``+volume`` when ``close >= open`` else ``-volume`` (per bar).

    A crude order-flow proxy: positive => net buying pressure, negative => net selling.
    """
    if bars is None or bars.empty:
        return 0.0
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    vol = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    sign = np.where(close.to_numpy() >= open_.to_numpy(), 1.0, -1.0)
    return float(np.nansum(sign * vol.to_numpy()))
