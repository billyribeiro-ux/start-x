"""Causal attribution: for each event, find the catalysts that plausibly caused it.

CORE SAFETY RULE: only catalysts whose public timestamp falls inside the causal window
``[event_date - lookback, event_date]`` may be attributed. Using a catalyst stamped *after*
the move is the #1 way to fabricate a fake "the model knew" result — it is forbidden here.
Confidence combines corroborating causes with a noisy-OR so multiple aligned signals raise
confidence, while a catalyst pointing the wrong way is discounted.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


def _window(df: pd.DataFrame | None, start, end, ts_col: str = "ts") -> pd.DataFrame:
    if df is None or df.empty or ts_col not in df.columns:
        return pd.DataFrame()
    t = df[ts_col]
    return df[(t >= start) & (t < end)]


def _cause(ctype: str, label: str, direction: str, score: float, **detail: Any) -> dict:
    return {"type": ctype, "label": label, "direction": direction,
            "score": round(float(score), 3), "detail": detail}


def _earnings(win: pd.DataFrame, direction: str) -> dict | None:
    rows = win[win.get("epsActual").notna()] if "epsActual" in win else pd.DataFrame()
    if rows.empty:
        return None
    r = rows.iloc[-1]
    est, act = r.get("epsEstimated"), r.get("epsActual")
    surprise = None
    if est not in (None, 0) and pd.notna(est) and pd.notna(act):
        surprise = (act - est) / abs(est)
    cdir = "up" if (surprise is None or surprise >= 0) else "down"
    beat = "beat" if (surprise is not None and surprise >= 0) else "miss"
    # Normalize against a ~10% EPS surprise (already a large beat/miss), not 100%.
    score = 0.25 + (0.45 * float(np.clip(abs(surprise) / 0.10, 0, 1)) if surprise is not None else 0.0)
    label = f"Earnings {beat} (EPS {act} vs est {est})" if surprise is not None else "Earnings report"
    return _cause("earnings", label, cdir, score, surprise=surprise, eps_actual=act, eps_est=est)


def _grades(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "action" not in win:
        return None
    acts = win.assign(a=win["action"].astype(str).str.lower())
    ups = acts[acts["a"].str.contains("upgrade")]
    downs = acts[acts["a"].str.contains("downgrade")]
    n_up, n_down = len(ups), len(downs)
    if n_up == n_down == 0:
        # initiations / maintains still count weakly
        r = acts.iloc[-1]
        return _cause("analyst_grade", f"Analyst {r['a']}: {r.get('gradingCompany','')}",
                      "neutral", 0.08, action=r["a"])
    cdir = "up" if n_up >= n_down else "down"
    r = (ups if cdir == "up" else downs).iloc[-1]
    label = (f"Analyst {'upgrade' if cdir=='up' else 'downgrade'}: "
             f"{r.get('gradingCompany','')} ({r.get('previousGrade','')}→{r.get('newGrade','')})")
    score = 0.25 + 0.05 * (max(n_up, n_down) - 1)
    return _cause("analyst_grade", label, cdir, min(score, 0.45), n_up=n_up, n_down=n_down)


def _price_target(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "priceTarget" not in win:
        return None
    r = win.iloc[-1]
    pt, ref = r.get("priceTarget"), r.get("priceWhenPosted")
    if pd.isna(pt) or pd.isna(ref) or ref in (0, None):
        return None
    cdir = "up" if pt >= ref else "down"
    verb = "raised" if cdir == "up" else "cut"
    return _cause("price_target", f"Price target {verb} to ${pt} ({r.get('analystCompany','')})",
                  cdir, 0.2, price_target=pt, ref_price=ref)


def _insider(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "acquisitionOrDisposition" not in win:
        return None
    qty = pd.to_numeric(win.get("securitiesTransacted"), errors="coerce").fillna(0)
    ad = win["acquisitionOrDisposition"].astype(str)
    bought = qty[ad == "A"].sum()
    sold = qty[ad == "D"].sum()
    if bought == sold == 0:
        return None
    cdir = "up" if bought >= sold else "down"
    label = f"Insider net {'buy' if cdir=='up' else 'sell'} ({int(max(bought,sold)):,} sh)"
    return _cause("insider", label, cdir, 0.15, bought=int(bought), sold=int(sold))


def _congress(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "type" not in win:
        return None
    r = win.iloc[-1]
    t = str(r.get("type", "")).lower()
    cdir = "up" if "purchase" in t or "buy" in t else "down"
    name = f"{r.get('firstName','')} {r.get('lastName','')}".strip()
    return _cause("congress", f"Congress {r.get('type','trade')}: {name}", cdir, 0.08)


def _news(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "title" not in win:
        return None
    n = len(win)
    titles = win.sort_values("ts")["title"].tail(3).tolist()
    return _cause("news", f"News spike: {n} headline(s)", "neutral",
                  0.1 * float(np.clip(n / 5.0, 0, 1)), count=n, sample=titles)


def _macro(win: pd.DataFrame, direction: str) -> dict | None:
    if win.empty or "event" not in win:
        return None
    hi = win[win.get("impact").astype(str).str.lower() == "high"] if "impact" in win else win
    us = hi[hi.get("country").astype(str).isin(["US", "USA"])] if "country" in hi else hi
    pool = us if not us.empty else hi
    if pool.empty:
        return None
    names = pool["event"].astype(str).tail(3).tolist()
    return _cause("macro", "Macro: " + "; ".join(names), "neutral",
                  0.2 * float(np.clip(len(pool) / 2.0, 0, 1)) + 0.1, events=names)


def _gap_vol(event_row: pd.Series, direction: str) -> dict | None:
    gap = event_row.get("gap")
    vol_z = event_row.get("vol_z")
    if pd.isna(gap) or pd.isna(vol_z):
        return None
    if abs(gap) >= 0.03 and vol_z >= 2.0:
        cdir = "up" if gap > 0 else "down"
        return _cause("gap_volume", f"Gap & volume surge (gap {gap*100:.1f}%, vol z={vol_z:.1f})",
                      cdir, 0.12, gap_pct=gap * 100, vol_z=float(vol_z))
    return None


def attribute_event(
    event_row: pd.Series, catalysts: dict[str, pd.DataFrame], lookback_days: int = 2
) -> tuple[list[dict], dict | None, float]:
    """Return (ranked causes, top cause, confidence in [0,1]) for one event."""
    direction = event_row["direction"]
    ev_date = pd.Timestamp(event_row["date"]).normalize()
    start = ev_date - timedelta(days=lookback_days)
    end = ev_date + timedelta(days=1)  # include all of the event day

    builders = [
        (_earnings, "earnings"), (_grades, "grades"), (_price_target, "price_target"),
        (_insider, "insider"), (_congress, "senate"), (_congress, "house"),
        (_news, "news"), (_macro, "macro"),
    ]
    causes: list[dict] = []
    for fn, key in builders:
        c = fn(_window(catalysts.get(key), start, end), direction)
        if c:
            causes.append(c)
    gv = _gap_vol(event_row, direction)
    if gv:
        causes.append(gv)

    # noisy-OR confidence; wrong-way causes discounted, neutral ones count as context
    prob_none = 1.0
    for c in causes:
        if c["direction"] == direction or c["direction"] == "neutral":
            eff = c["score"]
        else:
            eff = c["score"] * 0.3
        prob_none *= (1.0 - min(max(eff, 0.0), 0.99))
    confidence = round(1.0 - prob_none, 3)

    def _eff(c: dict) -> float:
        return c["score"] if c["direction"] in (direction, "neutral") else c["score"] * 0.3

    causes.sort(key=_eff, reverse=True)
    top = causes[0] if causes else None
    return causes, top, confidence
