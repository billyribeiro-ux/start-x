"""Reversal characterization ("Reversal Lab").

For every trend reversal (up->down at a swing high, down->up at a swing low) we dissect what
sat underneath it — daily candle anatomy, abnormal return, range/ATR expansion — and, when a
client is supplied, the intraday microstructure of the reversal day (volume profile POC/value
area, session VWAP, opening range, cumulative delta). We then aggregate the *legs that ended
at a reversal* into a statistical "when to be ready to get out" table (percentiles of leg
duration, magnitude and ATR-extension), separately for up-legs and down-legs.

Swings are found with an ATR-scaled zigzag (see :func:`detect_swings`), which adapts the
reversal threshold to each name's volatility instead of using a fixed percentage. All trend
statistics are computed from confirmed swings only, so there is no forward peeking baked into
the structure (the *characterization* of a known reversal day naturally uses that day's data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics.volume_profile import (
    cumulative_delta,
    opening_range,
    session_vwap,
    volume_profile,
)
from ..data.cache import ParquetCache
from ..data.intraday import get_intraday
from ..events.detect import compute_signals
from ..fmp.client import FMPClient

_PERCENTILES = (50, 75, 90)


# --------------------------------------------------------------------------- ATR
def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average True Range.

    True range is ``max(high-low, |high-prev_close|, |low-prev_close|)``; ATR is the Wilder
    (RMA / EWM with ``alpha = 1/window``) smoothing of TR. Returned as a Series aligned to
    ``prices.index``. Empty input yields an empty float Series.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    high = pd.to_numeric(prices["high"], errors="coerce")
    low = pd.to_numeric(prices["low"], errors="coerce")
    close = pd.to_numeric(prices["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    # Wilder smoothing == EWM with alpha = 1/window (adjust=False), seeded on the first TR.
    return tr.ewm(alpha=1.0 / float(window), adjust=False, min_periods=1).mean()


# --------------------------------------------------------------------------- swings
def detect_swings(
    prices: pd.DataFrame,
    *,
    atr_window: int = 14,
    atr_mult: float = 3.0,
) -> pd.DataFrame:
    """ATR-zigzag swing detection — confirmed turning points that alternate high/low.

    Algorithm (single forward pass, point-in-time within the pass):
      1. Compute Wilder ATR (``atr_window``); the reversal threshold at bar ``i`` is
         ``atr_mult * ATR[i]`` in price terms (volatility-adaptive, not a fixed %).
      2. Track a running extreme. While the leg direction is *up* we trail the running **high**
         on each new higher high; the leg flips to *down* (confirming that high as a swing
         **high**) the first time price closes/drops ``threshold`` below the running high.
         Symmetrically for *down* legs and swing **lows**.
      3. The initial direction is bootstrapped from the first move that exceeds the threshold
         from the opening bar.
      4. Swings strictly alternate (high, low, high, ...). If a more extreme same-direction
         extreme appears before confirmation, the pending extreme is simply updated.

    Returns ``DataFrame[date, kind('high'|'low'), price]`` in chronological order. Fewer than
    two bars yields an empty frame.
    """
    cols = ["date", "kind", "price"]
    if prices is None or len(prices) < 2:
        return pd.DataFrame(columns=cols)

    p = prices.reset_index(drop=True)
    dates = pd.to_datetime(p["date"]).to_numpy()
    high = pd.to_numeric(p["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(p["low"], errors="coerce").to_numpy()
    thr = (atr_mult * atr(p, atr_window)).to_numpy()
    n = len(p)

    swings: list[tuple] = []
    direction = 0  # 0 = unknown, +1 = up leg (seeking high), -1 = down leg (seeking low)
    ext_price = high[0]
    ext_low = low[0]
    ext_idx = 0       # index of the running extreme high (for up legs)
    ext_low_idx = 0   # index of the running extreme low (for down legs)

    for i in range(1, n):
        t = thr[i] if np.isfinite(thr[i]) else thr[max(0, i - 1)]
        if not np.isfinite(t) or t <= 0:
            t = 0.0

        if direction == 0:
            # Bootstrap: first decisive move from the opening extremes sets the direction.
            if high[i] > ext_price:
                ext_price, ext_idx = high[i], i
            if low[i] < ext_low:
                ext_low, ext_low_idx = low[i], i
            if t > 0 and ext_price - low[i] >= t and ext_idx <= ext_low_idx:
                # We rose then fell: opening leg was up; confirm the high, now seeking a low.
                swings.append((dates[ext_idx], "high", float(ext_price)))
                direction = -1
                ext_low, ext_low_idx = low[i], i
            elif t > 0 and high[i] - ext_low >= t and ext_low_idx <= ext_idx:
                swings.append((dates[ext_low_idx], "low", float(ext_low)))
                direction = +1
                ext_price, ext_idx = high[i], i
            continue

        if direction == +1:
            if high[i] >= ext_price:
                ext_price, ext_idx = high[i], i
            elif ext_price - low[i] >= t:  # retraced threshold from the running high
                swings.append((dates[ext_idx], "high", float(ext_price)))
                direction = -1
                ext_low, ext_low_idx = low[i], i
        else:  # direction == -1
            if low[i] <= ext_low:
                ext_low, ext_low_idx = low[i], i
            elif high[i] - ext_low >= t:  # rallied threshold from the running low
                swings.append((dates[ext_low_idx], "low", float(ext_low)))
                direction = +1
                ext_price, ext_idx = high[i], i

    return pd.DataFrame(swings, columns=cols)


# --------------------------------------------------------------------------- legs
def trend_legs(swings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Connect consecutive opposite swings into directional legs with rich stats.

    Each leg runs from one swing to the next opposite swing; ``end_date`` is the reversal.
    Columns:
      * ``direction`` — 'up' (low->high) or 'down' (high->low).
      * ``bars`` — trading bars between the two swing dates (inclusive of both endpoints - 1).
      * ``magnitude_pct`` — signed-magnitude as ``(end-start)/start * 100`` (positive for up,
        negative for down).
      * ``atr_extension`` — ``magnitude_pct / (ATR_at_start / price_at_start * 100)``: how many
        "ATR units" the leg travelled (volatility-normalised move size).
      * ``slope_pct_per_bar`` — ``magnitude_pct / bars``.
      * ``mfe_pct`` / ``mae_pct`` — max favourable / adverse excursion within the leg window,
        in percent from the leg's start price (favourable = in the leg's direction).

    Empty/insufficient swings yield an empty frame.
    """
    cols = [
        "start_date", "end_date", "direction", "bars", "magnitude_pct",
        "atr_extension", "slope_pct_per_bar", "mfe_pct", "mae_pct",
    ]
    if swings is None or len(swings) < 2 or prices is None or prices.empty:
        return pd.DataFrame(columns=cols)

    p = prices.reset_index(drop=True).copy()
    p["date"] = pd.to_datetime(p["date"])
    atr_series = atr(p, 14)
    by_date_pos = {d: i for i, d in enumerate(p["date"])}
    high = pd.to_numeric(p["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(p["low"], errors="coerce").to_numpy()

    sw = swings.reset_index(drop=True)
    rows = []
    for a, b in zip(sw.itertuples(index=False), sw.iloc[1:].itertuples(index=False)):
        start_date, start_kind, start_price = a.date, a.kind, float(a.price)
        end_date, end_price = b.date, float(b.price)
        direction = "up" if start_kind == "low" else "down"
        i0 = by_date_pos.get(pd.Timestamp(start_date))
        i1 = by_date_pos.get(pd.Timestamp(end_date))
        if i0 is None or i1 is None or i1 <= i0 or start_price <= 0:
            continue
        bars = i1 - i0
        magnitude_pct = (end_price - start_price) / start_price * 100.0

        atr0 = float(atr_series.iloc[i0]) if np.isfinite(atr_series.iloc[i0]) else np.nan
        atr_pct = (atr0 / start_price * 100.0) if (atr0 and start_price) else np.nan
        atr_extension = magnitude_pct / atr_pct if atr_pct and np.isfinite(atr_pct) else np.nan
        slope = magnitude_pct / bars if bars else np.nan

        seg_high = high[i0 : i1 + 1]
        seg_low = low[i0 : i1 + 1]
        if direction == "up":
            mfe_pct = (np.nanmax(seg_high) - start_price) / start_price * 100.0
            mae_pct = (np.nanmin(seg_low) - start_price) / start_price * 100.0
        else:
            mfe_pct = (start_price - np.nanmin(seg_low)) / start_price * 100.0
            mae_pct = (np.nanmax(seg_high) - start_price) / start_price * 100.0

        rows.append({
            "start_date": pd.Timestamp(start_date),
            "end_date": pd.Timestamp(end_date),
            "direction": direction,
            "bars": int(bars),
            "magnitude_pct": float(magnitude_pct),
            "atr_extension": float(atr_extension) if np.isfinite(atr_extension) else np.nan,
            "slope_pct_per_bar": float(slope) if np.isfinite(slope) else np.nan,
            "mfe_pct": float(mfe_pct),
            "mae_pct": float(mae_pct),
        })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------- candle label
def _candle_label(o: float, h: float, l: float, c: float, prev_c: float | None) -> str:
    """Tiny heuristic candle classifier for the reversal bar."""
    rng = h - l
    if rng <= 0 or not np.isfinite(rng):
        return "flat"
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_frac = body / rng
    bullish = c >= o
    gap_up = prev_c is not None and np.isfinite(prev_c) and o > prev_c
    gap_down = prev_c is not None and np.isfinite(prev_c) and o < prev_c

    if body_frac < 0.1:
        return "doji"
    if lower >= 2 * body and upper <= body:
        return "hammer" if bullish else "hanging_man"
    if upper >= 2 * body and lower <= body:
        return "shooting_star" if not bullish else "inverted_hammer"
    if body_frac >= 0.7:
        if bullish:
            return "bullish_marubozu_gap_up" if gap_up else "strong_bullish"
        return "bearish_marubozu_gap_down" if gap_down else "strong_bearish"
    return "bullish" if bullish else "bearish"


# ------------------------------------------------------------ characterize reversal
def characterize_reversal(
    prices: pd.DataFrame,
    reversal_date,
    *,
    client: FMPClient | None = None,
    cache: ParquetCache | None = None,
    fmp_symbol: str | None = None,
    intraday: bool = True,
) -> dict:
    """Dissect a single reversal day: daily anatomy + (optional) intraday microstructure.

    ``prices`` must be the enriched daily frame (from ``get_prices`` / ``compute_signals``).
    Returns ``{'daily': {...}}`` and, when ``intraday`` and a ``client``/``fmp_symbol`` are
    given and bars exist, ``{'intraday': {...}}``.

    daily keys: ``volume, rvol_20, range_pct, atr_14, atr_expansion, gap_pct,
    close_location_value, ar_z, candle``.
    intraday keys: ``poc, vah, val, session_vwap, reversal_vs_poc, reversal_vs_vwap,
    cumulative_delta, or_high, or_low``.
    """
    ts = pd.Timestamp(reversal_date)
    p = prices.reset_index(drop=True).copy()
    p["date"] = pd.to_datetime(p["date"])

    # Ensure ar_z is available (compute_signals is idempotent/cheap on the daily frame).
    if "ar_z" not in p.columns:
        p = compute_signals(p, market=None)

    pos = p.index[p["date"] == ts]
    if len(pos) == 0:
        return {"daily": {k: float("nan") for k in (
            "volume", "rvol_20", "range_pct", "atr_14", "atr_expansion", "gap_pct",
            "close_location_value", "ar_z")} | {"candle": "unknown"}}
    i = int(pos[0])
    row = p.iloc[i]

    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    vol = float(row["volume"])
    prev_c = float(row["prev_close"]) if "prev_close" in p.columns and pd.notna(
        row.get("prev_close")) else (float(p.iloc[i - 1]["close"]) if i > 0 else np.nan)

    rng = h - l
    range_pct = rng / c * 100.0 if c else np.nan
    clv = ((c - l) - (h - c)) / rng if rng > 0 else np.nan  # close location value in [-1, 1]
    gap_pct = (o / prev_c - 1.0) * 100.0 if prev_c and np.isfinite(prev_c) else np.nan

    atr_series = atr(p, 14)
    atr_14 = float(atr_series.iloc[i]) if np.isfinite(atr_series.iloc[i]) else np.nan
    atr_prev = float(atr_series.iloc[i - 1]) if i > 0 and np.isfinite(
        atr_series.iloc[i - 1]) else np.nan
    atr_expansion = rng / atr_prev if atr_prev and np.isfinite(atr_prev) else np.nan

    vol_series = pd.to_numeric(p["volume"], errors="coerce")
    avg_vol_20 = float(vol_series.iloc[max(0, i - 20):i].mean()) if i > 0 else np.nan
    rvol_20 = vol / avg_vol_20 if avg_vol_20 and np.isfinite(avg_vol_20) else np.nan

    ar_z = float(row["ar_z"]) if "ar_z" in p.columns and pd.notna(row.get("ar_z")) else np.nan

    daily = {
        "volume": vol,
        "rvol_20": float(rvol_20) if np.isfinite(rvol_20) else np.nan,
        "range_pct": float(range_pct) if np.isfinite(range_pct) else np.nan,
        "atr_14": atr_14,
        "atr_expansion": float(atr_expansion) if np.isfinite(atr_expansion) else np.nan,
        "gap_pct": float(gap_pct) if np.isfinite(gap_pct) else np.nan,
        "close_location_value": float(clv) if np.isfinite(clv) else np.nan,
        "ar_z": ar_z,
        "candle": _candle_label(o, h, l, c, prev_c),
    }
    result: dict = {"daily": daily}

    if intraday and client is not None and fmp_symbol is not None:
        cache = cache or ParquetCache("data/cache")
        bars = get_intraday(client, cache, fmp_symbol, ts.date().isoformat())
        if not bars.empty:
            vp = volume_profile(bars)
            vwap = session_vwap(bars)
            poc = vp["poc_price"]
            orr = opening_range(bars)
            result["intraday"] = {
                "poc": poc,
                "vah": vp["vah"],
                "val": vp["val"],
                "session_vwap": vwap,
                "reversal_vs_poc": (c - poc) / poc * 100.0 if poc and np.isfinite(poc)
                else np.nan,
                "reversal_vs_vwap": (c - vwap) / vwap * 100.0 if vwap and np.isfinite(vwap)
                else np.nan,
                "cumulative_delta": cumulative_delta(bars),
                "or_high": orr["high"],
                "or_low": orr["low"],
            }
    return result


# --------------------------------------------------------------------- exit stats
def _exit_stats(legs: pd.DataFrame) -> dict:
    """Percentile table of leg duration / magnitude / ATR-extension, split by direction."""
    out: dict[str, dict] = {}
    for direction in ("up", "down"):
        sub = legs[legs["direction"] == direction] if not legs.empty else legs
        d: dict[str, float] = {"n": int(len(sub))}
        for metric in ("bars", "magnitude_pct", "atr_extension"):
            vals = (pd.to_numeric(sub[metric], errors="coerce").dropna()
                    if metric in sub.columns else pd.Series(dtype=float))
            # magnitude_pct and atr_extension are direction-signed; the exhaustion table is
            # about how *far* a leg runs, so percentiles are taken on the absolute size.
            if metric in ("magnitude_pct", "atr_extension"):
                vals = vals.abs()
            for pct in _PERCENTILES:
                key = f"{metric}_p{pct}"
                d[key] = float(np.percentile(vals, pct)) if len(vals) else float("nan")
        out[direction] = d
    return out


# --------------------------------------------------------------------- full report
def reversal_report(
    ticker: str,
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache,
    universe,
    settings,
    intraday: bool = True,
    refresh: bool = False,
) -> dict:
    """End-to-end reversal lab for ``ticker`` over ``[start, end]``.

    Pulls full daily history (for stable ATR/swing context), detects swings and legs, then for
    every leg whose reversal (``end_date``) falls inside the window emits leg stats merged with
    :func:`characterize_reversal`. Also returns the learned ``exit_stats`` percentile table.

    Returns ``{'ticker', 'spec', 'prices'(windowed), 'swings', 'legs', 'reversals'(list),
    'exit_stats', 'params'}``.
    """
    from ..data.prices import get_prices  # local import avoids a heavy import cycle

    spec = universe.spec(ticker)
    prices_full = get_prices(client, cache, spec.fmp, settings.history_start, refresh)
    if prices_full.empty:
        raise RuntimeError(f"No price data for {ticker} ({spec.fmp}).")

    signals = compute_signals(prices_full, market=None)
    swings = detect_swings(signals)
    legs = trend_legs(swings, signals)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    in_win = legs[(legs["end_date"] >= start_ts) & (legs["end_date"] <= end_ts)] \
        if not legs.empty else legs

    reversals = []
    for leg in in_win.itertuples(index=False):
        char = characterize_reversal(
            signals, leg.end_date, client=client if intraday else None, cache=cache,
            fmp_symbol=spec.fmp if intraday else None, intraday=intraday,
        )
        reversals.append({
            "reversal_date": pd.Timestamp(leg.end_date),
            "reversal_kind": "high" if leg.direction == "up" else "low",
            "leg": {
                "start_date": pd.Timestamp(leg.start_date),
                "end_date": pd.Timestamp(leg.end_date),
                "direction": leg.direction,
                "bars": int(leg.bars),
                "magnitude_pct": float(leg.magnitude_pct),
                "atr_extension": (float(leg.atr_extension)
                                  if pd.notna(leg.atr_extension) else np.nan),
                "slope_pct_per_bar": (float(leg.slope_pct_per_bar)
                                      if pd.notna(leg.slope_pct_per_bar) else np.nan),
                "mfe_pct": float(leg.mfe_pct),
                "mae_pct": float(leg.mae_pct),
            },
            **char,
        })

    win = signals[(signals["date"] >= start_ts) & (signals["date"] <= end_ts)]
    return {
        "ticker": ticker,
        "spec": spec,
        "prices": win.reset_index(drop=True),
        "swings": swings,
        "legs": in_win.reset_index(drop=True),
        "reversals": reversals,
        "exit_stats": _exit_stats(in_win),
        "params": {"start": start, "end": end, "intraday": intraday},
    }
