"""Point-in-time (PIT) windowing helpers.

These are the single chokepoint through which every catalyst feed is filtered before it can
touch a feature. The whole feature layer's no-lookahead guarantee rests on the fact that a row
is only ever visible once its public timestamp (``ts_col``) is at or before the as-of date.
All helpers are defensively tolerant: a ``None``/empty frame, or one missing the timestamp
column, yields an empty frame rather than raising.
"""
from __future__ import annotations

import pandas as pd


def _empty_like(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return an empty frame, preserving columns when we have them."""
    if isinstance(df, pd.DataFrame):
        return df.iloc[0:0]
    return pd.DataFrame()


def in_window(
    df: pd.DataFrame | None,
    start,
    end,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Rows whose ``ts_col`` falls in the half-open interval ``[start, end)``.

    Half-open (end-exclusive) mirrors ``events.attribute._window`` so that an "end" passed as
    ``event_date + 1 day`` includes the whole event day without bleeding into the next one.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or ts_col not in df.columns:
        return _empty_like(df)
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mask = ts.notna() & (ts >= start_ts) & (ts < end_ts)
    return df[mask]


def trailing(
    df: pd.DataFrame | None,
    asof,
    days: int,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Rows with ``ts_col`` in the inclusive window ``[asof - days, asof]``.

    ``asof`` is inclusive on both ends: anything stamped strictly after ``asof`` is future
    information and is dropped — this is the load-bearing no-lookahead filter for flow features.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or ts_col not in df.columns:
        return _empty_like(df)
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    asof_ts = pd.Timestamp(asof)
    start_ts = asof_ts - pd.Timedelta(days=days)
    mask = ts.notna() & (ts >= start_ts) & (ts <= asof_ts)
    return df[mask]
