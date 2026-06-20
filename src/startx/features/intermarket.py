"""Intermarket (cross-asset / macro) features — strictly point-in-time.

For each as-of date ``t`` we use cross-asset history up to and including ``t`` only: every
feature is built from rolling/shift operations on a daily series and then forward-filled onto
the target dates (never a future observation). These encode the macro backdrop that empirically
*leads* broad equity indices:

* the Treasury curve (10y level/change, 30y-5y and 10y-5y slope and its change),
* credit / risk appetite (HYG & LQD trailing returns, the HYG/LQD spread proxy and its change),
* relative strength vs the benchmark for leading groups — homebuilders (ITB/XHB), transports
  (IYT), semis (SMH), financials (XLF), discretionary (XLY), defensives (XLP), plus the
  defensive-rotation tell XLP/XLY,
* safe-haven / macro (gold, the dollar, long Treasuries).

All columns carry an ``im_`` prefix. The output has exactly one row per requested date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Trailing windows (trading days).
_CHG_SHORT = 5
_CHG_LONG = 21
_RS_SHORT = 21
_RS_LONG = 63
_RET_WIN = 21

#: Relative-strength groups: short name in ``context`` -> RS column stem.
_RS_GROUPS = {
    "itb": "itb",
    "xhb": "xhb",
    "iyt": "iyt",
    "smh": "smh",
    "xlf": "xlf",
    "xly": "xly",
    "xlp": "xlp",
}


def _prep(prices: pd.DataFrame | None) -> pd.Series | None:
    """Tidy a price frame into a sorted, de-duplicated ``date``-indexed close series."""
    if prices is None or getattr(prices, "empty", True) or "date" not in prices.columns:
        return None
    if "close" not in prices.columns:
        return None
    df = prices[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    s = pd.Series(df["close"].astype(float).to_numpy(), index=df["date"], name="close")
    return s


def _asof(series: pd.Series, date_index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill ``series`` onto ``date_index`` (PIT: uses only the last value <= each date)."""
    return series.reindex(date_index, method="ffill").to_numpy()


def _ret(series: pd.Series, window: int) -> pd.Series:
    """Simple trailing return over ``window`` trading days."""
    return series / series.shift(window) - 1.0


def intermarket_features(
    dates: "pd.Series | pd.DatetimeIndex",
    context: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Per-date cross-asset/macro features (all trailing/PIT), keyed by ``date``.

    Parameters
    ----------
    dates:
        Target as-of dates. One output row is returned per (sorted, unique) date.
    context:
        ``{short_name: DataFrame[date, close]}`` cross-asset frames (see
        :func:`startx.data.intermarket.get_context_prices`). Missing names are tolerated.
    benchmark_prices:
        Benchmark (e.g. SPY) price frame; relative-strength features need it.
    """
    date_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(list(dates)))).sort_values().unique()
    date_index = pd.DatetimeIndex(date_index)
    out = pd.DataFrame({"date": date_index})

    ctx = {name: _prep(df) for name, df in (context or {}).items()}
    bench = _prep(benchmark_prices)

    def add(col: str, series: pd.Series | None) -> None:
        out[col] = _asof(series, date_index) if series is not None else np.nan

    # -- Treasury yields & curve ------------------------------------------------------------
    tnx = ctx.get("tnx10")
    tyx = ctx.get("tyx30")
    fvx = ctx.get("fvx5")

    if tnx is not None:
        add("im_tnx10_level", tnx)
        add("im_tnx10_chg5", tnx - tnx.shift(_CHG_SHORT))      # abs change in yield
        add("im_tnx10_chg21", tnx - tnx.shift(_CHG_LONG))
    else:
        for c in ("im_tnx10_level", "im_tnx10_chg5", "im_tnx10_chg21"):
            out[c] = np.nan

    # Curve slope = 30y - 5y (and 10y - 5y); both need an aligned join on the union calendar.
    if tyx is not None and fvx is not None:
        idx = tyx.index.union(fvx.index)
        slope = tyx.reindex(idx).ffill() - fvx.reindex(idx).ffill()
        add("im_curve_slope", slope)
        add("im_curve_slope_chg21", slope - slope.shift(_CHG_LONG))
    else:
        out["im_curve_slope"] = np.nan
        out["im_curve_slope_chg21"] = np.nan

    if tnx is not None and fvx is not None:
        idx = tnx.index.union(fvx.index)
        slope_10_5 = tnx.reindex(idx).ffill() - fvx.reindex(idx).ffill()
        add("im_curve_slope_10_5", slope_10_5)
    else:
        out["im_curve_slope_10_5"] = np.nan

    # -- Credit / risk appetite -------------------------------------------------------------
    hyg = ctx.get("hyg")
    lqd = ctx.get("lqd")
    add("im_hyg_ret21", _ret(hyg, _RET_WIN) if hyg is not None else None)
    add("im_lqd_ret21", _ret(lqd, _RET_WIN) if lqd is not None else None)

    if hyg is not None and lqd is not None:
        idx = hyg.index.union(lqd.index)
        ratio = hyg.reindex(idx).ffill() / lqd.reindex(idx).ffill()
        add("im_hyg_lqd_ratio", ratio)
        add("im_hyg_lqd_chg21", ratio - ratio.shift(_CHG_LONG))
    else:
        out["im_hyg_lqd_ratio"] = np.nan
        out["im_hyg_lqd_chg21"] = np.nan

    # -- Relative strength vs benchmark -----------------------------------------------------
    # RS ratio = close_X / close_bench, then trailing change over 21d & 63d.
    for name, stem in _RS_GROUPS.items():
        s = ctx.get(name)
        if s is not None and bench is not None:
            idx = s.index.union(bench.index)
            rs = s.reindex(idx).ffill() / bench.reindex(idx).ffill()
            add(f"im_{stem}_rs21", _ret(rs, _RS_SHORT))
            add(f"im_{stem}_rs63", _ret(rs, _RS_LONG))
        else:
            out[f"im_{stem}_rs21"] = np.nan
            out[f"im_{stem}_rs63"] = np.nan

    # Defensive-rotation tell: staples vs discretionary (XLP/XLY). Rising => risk-off.
    xlp = ctx.get("xlp")
    xly = ctx.get("xly")
    if xlp is not None and xly is not None:
        idx = xlp.index.union(xly.index)
        rot = xlp.reindex(idx).ffill() / xly.reindex(idx).ffill()
        add("im_xlp_xly_rs21", _ret(rot, _RS_SHORT))
        add("im_xlp_xly_rs63", _ret(rot, _RS_LONG))
    else:
        out["im_xlp_xly_rs21"] = np.nan
        out["im_xlp_xly_rs63"] = np.nan

    # -- Safe-haven / macro -----------------------------------------------------------------
    gld = ctx.get("gld")
    uup = ctx.get("uup")
    tlt = ctx.get("tlt")
    add("im_gld_ret21", _ret(gld, _RET_WIN) if gld is not None else None)
    add("im_dollar_chg21", _ret(uup, _RET_WIN) if uup is not None else None)
    add("im_tlt_ret21", _ret(tlt, _RET_WIN) if tlt is not None else None)

    return out
