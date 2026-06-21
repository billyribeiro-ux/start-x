"""Flow / catalyst features — one row per as-of date, strictly point-in-time.

For each date ``t`` we look only at catalyst rows whose public timestamp ``ts <= t`` (via
``pit.trailing``). The single exception is ``days_to_next_earnings``: the *immediate next*
scheduled earnings date is public information today, so counting days to it is PIT-legal — but
ONLY within ~one quarter (``_MAX_SCHED_HORIZON_D``). A date further out was not reliably scheduled
at ``t`` (companies announce the next date weeks-to-a-quarter ahead, not a year), so we drop it
(NaN) rather than leak a not-yet-announced future date.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..data.pit import trailing

INSIDER_WINDOWS = (30, 90)
NEWS_WINDOWS = (5, 20)
CONGRESS_WINDOW = 90
ANALYST_WINDOW = 30
PRICE_TARGET_WINDOW = 30
_MACRO_WINDOW = 1
#: PIT guard for ``days_to_next_earnings``: only the immediate next quarterly date is reliably
#: scheduled/announced at ``t``. A date more than ~one quarter out wasn't knowable, so we NaN it.
_MAX_SCHED_HORIZON_D = 92


def _empty_row(date: pd.Timestamp) -> dict:
    """A feature row of all-NaN/0 defaults; overwritten where data exists."""
    row: dict[str, object] = {"date": date}
    for w in INSIDER_WINDOWS:
        row[f"insider_net_{w}"] = 0.0
        row[f"insider_buy_cnt_{w}"] = 0
        row[f"insider_sell_cnt_{w}"] = 0
    row[f"congress_net_{CONGRESS_WINDOW}"] = 0.0
    row[f"analyst_net_{ANALYST_WINDOW}"] = 0
    row[f"analyst_cnt_{ANALYST_WINDOW}"] = 0
    row[f"price_target_net_{PRICE_TARGET_WINDOW}"] = 0
    row[f"price_target_avg_chg_{PRICE_TARGET_WINDOW}"] = np.nan
    for w in NEWS_WINDOWS:
        row[f"news_cnt_{w}"] = 0
    row["days_since_last_earnings"] = np.nan
    row["last_eps_surprise"] = np.nan
    row["days_to_next_earnings"] = np.nan
    row[f"macro_high_us_{_MACRO_WINDOW}"] = 0
    return row


def _insider(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    for w in INSIDER_WINDOWS:
        win = trailing(df, t, w)
        if win.empty or "acquisitionOrDisposition" not in win.columns:
            continue
        qty = pd.to_numeric(win.get("securitiesTransacted"), errors="coerce").fillna(0.0)
        ad = win["acquisitionOrDisposition"].astype(str)
        bought = float(qty[ad == "A"].sum())
        sold = float(qty[ad == "D"].sum())
        row[f"insider_net_{w}"] = bought - sold
        row[f"insider_buy_cnt_{w}"] = int((ad == "A").sum())
        row[f"insider_sell_cnt_{w}"] = int((ad == "D").sum())


def _congress(row: dict, frames: Iterable[pd.DataFrame], t: pd.Timestamp) -> None:
    net = 0
    for df in frames:
        win = trailing(df, t, CONGRESS_WINDOW)
        if win.empty or "type" not in win.columns:
            continue
        types = win["type"].astype(str).str.lower()
        buys = int(types.str.contains("purchase").sum() + types.str.contains("buy").sum())
        sells = int(types.str.contains("sale").sum() + types.str.contains("sell").sum())
        net += buys - sells
    row[f"congress_net_{CONGRESS_WINDOW}"] = net


def _analyst(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    win = trailing(df, t, ANALYST_WINDOW)
    if win.empty or "action" not in win.columns:
        return
    acts = win["action"].astype(str).str.lower()
    ups = int(acts.str.contains("upgrade").sum())
    downs = int(acts.str.contains("downgrade").sum())
    row[f"analyst_net_{ANALYST_WINDOW}"] = ups - downs
    row[f"analyst_cnt_{ANALYST_WINDOW}"] = int(len(win))


def _price_target(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    win = trailing(df, t, PRICE_TARGET_WINDOW)
    if win.empty or "priceTarget" not in win.columns:
        return
    pt = pd.to_numeric(win.get("priceTarget"), errors="coerce")
    ref = pd.to_numeric(win.get("priceWhenPosted"), errors="coerce")
    valid = pt.notna() & ref.notna() & (ref != 0)
    if not valid.any():
        return
    pt_v, ref_v = pt[valid], ref[valid]
    raises = int((pt_v > ref_v).sum())
    cuts = int((pt_v < ref_v).sum())
    row[f"price_target_net_{PRICE_TARGET_WINDOW}"] = raises - cuts
    row[f"price_target_avg_chg_{PRICE_TARGET_WINDOW}"] = float((pt_v / ref_v - 1.0).mean())


def _news(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    for w in NEWS_WINDOWS:
        win = trailing(df, t, w)
        row[f"news_cnt_{w}"] = int(len(win))


def _earnings(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    if df is None or df.empty or "ts" not in df.columns:
        return
    ts = pd.to_datetime(df["ts"], errors="coerce")
    # Past, already-reported earnings (epsActual present) — PIT requires ts <= t.
    past_mask = ts.notna() & (ts <= t)
    if "epsActual" in df.columns:
        past_mask &= df["epsActual"].notna()
    past = df[past_mask].assign(_ts=ts[past_mask]).sort_values("_ts")
    if not past.empty:
        last = past.iloc[-1]
        row["days_since_last_earnings"] = float((t - last["_ts"]).days)
        est, act = last.get("epsEstimated"), last.get("epsActual")
        if pd.notna(est) and pd.notna(act) and est not in (0, None):
            row["last_eps_surprise"] = float((act - est) / abs(est))

    # Next *scheduled* earnings: the chronologically next earnings *date* after t. We do NOT gate on
    # epsActual being NaN (the feed is a snapshot fetched today, so a date future at t may now carry
    # a backfilled actual — keying on the date alone is what stays PIT-honest). BUT a date more than
    # ~one quarter out was not reliably scheduled/announced at t, so counting days to it would leak a
    # not-yet-known future date: we clip the horizon to _MAX_SCHED_HORIZON_D and NaN anything beyond.
    future_mask = ts.notna() & (ts > t)
    fut = df[future_mask].assign(_ts=ts[future_mask]).sort_values("_ts")
    if not fut.empty:
        days = float((fut.iloc[0]["_ts"] - t).days)
        row["days_to_next_earnings"] = days if days <= _MAX_SCHED_HORIZON_D else np.nan


def _macro(row: dict, df: pd.DataFrame, t: pd.Timestamp) -> None:
    win = trailing(df, t, _MACRO_WINDOW)
    if win.empty or "event" not in win.columns:
        return
    mask = pd.Series(True, index=win.index)
    if "impact" in win.columns:
        mask &= win["impact"].astype(str).str.lower() == "high"
    if "country" in win.columns:
        mask &= win["country"].astype(str).isin(["US", "USA"])
    row[f"macro_high_us_{_MACRO_WINDOW}"] = int(mask.sum())


def flow_features(dates: Iterable, catalysts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-date flow features from PIT-filtered catalyst feeds.

    ``dates`` is the sequence of as-of dates (one feature row each). ``catalysts`` is the dict
    returned by ``data.catalysts.load_catalysts``. Missing/empty feeds yield neutral defaults.
    """
    date_index = pd.to_datetime(pd.Index(list(dates)))
    if len(date_index) == 0:
        return pd.DataFrame(columns=list(_empty_row(pd.Timestamp("2000-01-01")).keys()))

    cats = catalysts or {}
    insider = cats.get("insider")
    senate = cats.get("senate")
    house = cats.get("house")
    grades = cats.get("grades")
    price_target = cats.get("price_target")
    news = cats.get("news")
    earnings = cats.get("earnings")
    macro = cats.get("macro")

    rows: list[dict] = []
    for t in date_index:
        row = _empty_row(t)
        _insider(row, insider, t)
        _congress(row, (senate, house), t)
        _analyst(row, grades, t)
        _price_target(row, price_target, t)
        _news(row, news, t)
        _earnings(row, earnings, t)
        _macro(row, macro, t)
        rows.append(row)

    return pd.DataFrame(rows)
