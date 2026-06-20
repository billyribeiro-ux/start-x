"""Market-regime features — VIX and benchmark context, strictly point-in-time.

For each as-of date ``t`` we use VIX and benchmark history up to and including ``t`` (we
forward-fill onto the target dates and never use a future row). These describe the macro/risk
backdrop the model trades into.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_PCT_RANK_WINDOW = 252
_VIX_CHG_DAYS = 5
_BENCH_RET_WINDOWS = (20, 63)
_BENCH_VOL_WINDOW = 21
_SMA200 = 200
_ANNUALIZE = float(np.sqrt(252.0))


def _pct_rank_last(s: pd.Series) -> float:
    """Percentile rank of the final value within the window (0..1)."""
    arr = s.to_numpy(dtype=float)
    if len(arr) == 0 or np.isnan(arr[-1]):
        return np.nan
    valid = arr[~np.isnan(arr)]
    if len(valid) <= 1:
        return np.nan
    return float((valid <= arr[-1]).sum() - 1) / float(len(valid) - 1)


def _prep(prices: pd.DataFrame | None) -> pd.DataFrame | None:
    if prices is None or prices.empty or "date" not in prices.columns:
        return None
    df = prices[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def regime_features(
    dates: Iterable,
    benchmark_prices: pd.DataFrame | None,
    vix_prices: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per-date regime features from benchmark + VIX price history (all trailing/PIT)."""
    date_index = pd.to_datetime(pd.Index(list(dates))).sort_values().unique()
    out = pd.DataFrame({"date": date_index})

    cols = [
        "vix_level", "vix_chg_5d", "vix_pctrank_252",
        "bench_ret_20", "bench_ret_63", "bench_above_sma200", "bench_vol_21",
    ]
    for c in cols:
        out[c] = np.nan

    bench = _prep(benchmark_prices)
    vix = _prep(vix_prices)

    # -- VIX ---------------------------------------------------------------------------------
    if vix is not None:
        vix = vix.set_index("date")
        vclose = vix["close"].astype(float)
        v_chg = vclose - vclose.shift(_VIX_CHG_DAYS)
        v_rank = vclose.rolling(_PCT_RANK_WINDOW, min_periods=20).apply(_pct_rank_last, raw=False)
        v_level = vclose.reindex(date_index, method="ffill")
        out["vix_level"] = v_level.to_numpy()
        out["vix_chg_5d"] = v_chg.reindex(date_index, method="ffill").to_numpy()
        out["vix_pctrank_252"] = v_rank.reindex(date_index, method="ffill").to_numpy()

    # -- benchmark ---------------------------------------------------------------------------
    if bench is not None:
        bench = bench.set_index("date")
        bclose = bench["close"].astype(float)
        log_ret = np.log(bclose / bclose.shift(1))
        sma200 = bclose.rolling(_SMA200, min_periods=_SMA200).mean()
        above = (bclose > sma200).astype(float).where(sma200.notna(), np.nan)
        vol21 = log_ret.rolling(_BENCH_VOL_WINDOW, min_periods=_BENCH_VOL_WINDOW).std() * _ANNUALIZE
        for w in _BENCH_RET_WINDOWS:
            ret_w = bclose / bclose.shift(w) - 1.0
            out[f"bench_ret_{w}"] = ret_w.reindex(date_index, method="ffill").to_numpy()
        out["bench_above_sma200"] = above.reindex(date_index, method="ffill").to_numpy()
        out["bench_vol_21"] = vol21.reindex(date_index, method="ffill").to_numpy()

    return out
