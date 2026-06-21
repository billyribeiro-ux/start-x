"""Market internals derived from the S&P 500 constituent panel — the institutional breadth view.

No vendor breadth feed exists on the FMP plan, so we build the internals ourselves from the 503
constituents (the gold-standard, point-in-time way): % above the 50/200-day MA, advance/decline,
up/down volume, new 52-week highs−lows, and the McClellan oscillator. These separate a REAL
capitulation (a broad, exhausted washout) from a narrow shakeout/trap — the dimension the
price+vol sleeves are blind to.

Worked example (2026-03): the fear sleeves fired on BOTH flushes, but only one was real —
  • 2026-03-18 (the trap, −4%):  28% above 50d, 24 new lows  → NOT washed out → veto.
  • 2026-03-27 (the bottom, +16%): 19% above 50d, 46 new lows → genuine climax  → confirm.
:func:`washout_signals` encodes exactly that: only confirm a fear/dip long when breadth shows the
selling is genuinely exhausted.

Point-in-time — what IS and ISN'T enforced (be precise, don't oversell it):
  • ENFORCED (genuine, effective): every threshold is a *trailing* rolling quantile, and the
    "joined" edge is gated by each name's ``dateFirstAdded`` — a stock contributes to breadth only
    on/after the session it entered the index (see the ``mask`` built in :func:`compute_internals`).
    This is a real filter: ~114 of the current 503 joined after 2019-01-01, and their pre-membership
    history is correctly excluded rather than back-projected.
  • NOT ENFORCED (known data limitation, NOT silently hidden): we iterate the *current* constituent
    list only, so names that were **removed/delisted** over the window are absent, and we have no
    ``dateRemoved`` field to gate the "left the index" edge. That is a one-sided survivorship bias:
    breadth is computed over today's survivors. It is mild on the 2019→now S&P (turnover ~20–25
    names/yr, mostly small) but it is real. We do NOT fabricate historical constituents to paper
    over it; :func:`compute_internals` emits a runtime warning so no caller mistakes this for a
    fully point-in-time panel. Fixing it properly needs a vendor historical-membership feed
    (``dateRemoved`` / a dated add/drop log), which the current FMP plan does not provide.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

#: Mirror the cache's per-segment filename sanitizer (data/cache.py `_SAFE`) so a constituent's
#: parquet is read under the SAME name `get_prices` wrote it (e.g. a dotted class share `BRK.B` →
#: `BRK_B.parquet`); otherwise a dotted ticker silently drops out of the breadth panel.
_SAFE_SYM = re.compile(r"[^A-Za-z0-9_-]")

_INTERNALS_CACHE = "data/cache/internals.parquet"
_MEMBERS_CACHE = "data/cache/sp500_members.parquet"


def _constituents(client=None) -> pd.DataFrame:
    """Symbol + dateFirstAdded for the current S&P 500, cached to parquet (one FMP call if absent)."""
    if os.path.exists(_MEMBERS_CACHE):
        return pd.read_parquet(_MEMBERS_CACHE)
    if client is None:
        from ..fmp.client import FMPClient
        from ..settings import get_settings
        client = FMPClient(get_settings())
    rows = client.get("sp500-constituent")
    df = pd.DataFrame([{"symbol": r.get("symbol"),
                        "added": pd.to_datetime(r.get("dateFirstAdded") or "1990-01-01",
                                                errors="coerce")}
                       for r in rows if r.get("symbol")]).dropna(subset=["symbol"])
    df["added"] = df["added"].fillna(pd.Timestamp("1990-01-01"))
    os.makedirs(os.path.dirname(_MEMBERS_CACHE), exist_ok=True)
    df.to_parquet(_MEMBERS_CACHE, index=False)
    return df


#: One ``dateFirstAdded`` add-gate is enforced (see module docstring); the missing
#: ``dateRemoved`` / delisted side is a known, unfixable-here survivorship bias we warn about
#: rather than hide. Set ``STARTX_SILENCE_SURVIVORSHIP=1`` to mute the per-process warning.
_SURVIVORSHIP_WARNING = (
    "market_internals: breadth is built from the CURRENT S&P 500 constituents only. The "
    "`dateFirstAdded` add-gate IS applied, but removed/delisted names are absent (no `dateRemoved` "
    "feed on this FMP plan) -> one-sided survivorship bias. Mild on 2019->now; do NOT treat this "
    "panel as fully point-in-time. Set STARTX_SILENCE_SURVIVORSHIP=1 to silence."
)


def compute_internals(client=None) -> pd.DataFrame:
    """Build the daily internals from the constituent price panel.

    Point-in-time membership is gated on the ``dateFirstAdded`` (joined) edge only; the
    removed/delisted edge is NOT gated (current-survivors panel) — see the module docstring and the
    runtime warning. This is an honest, one-sided survivorship limitation, not a silent one.
    """
    if not os.environ.get("STARTX_SILENCE_SURVIVORSHIP"):
        warnings.warn(_SURVIVORSHIP_WARNING, RuntimeWarning, stacklevel=2)
    members = _constituents(client)
    closes, highs, lows, vols = {}, {}, {}, {}
    for sym in members["symbol"]:
        f = f"data/cache/prices/{_SAFE_SYM.sub('_', sym)}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=["date", "close", "high", "low", "volume"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").drop_duplicates("date").set_index("date")
        closes[sym] = d["close"]; highs[sym] = d["high"]; lows[sym] = d["low"]; vols[sym] = d["volume"]
    C = pd.DataFrame(closes).sort_index()
    H = pd.DataFrame(highs).reindex_like(C); V = pd.DataFrame(vols).reindex_like(C)
    # point-in-time membership mask (a name counts only on/after it joined the index)
    add = members.set_index("symbol")["added"]
    mask = pd.DataFrame(True, index=C.index, columns=C.columns)
    for s in C.columns:
        a = add.get(s)
        if pd.notna(a):
            mask.loc[C.index < a, s] = False
    valid = mask & C.notna()
    sma50 = C.rolling(50).mean(); sma200 = C.rolling(200).mean()
    above50 = (C > sma50) & mask; above200 = (C > sma200) & mask
    up = (C > C.shift(1)) & mask; dn = (C < C.shift(1)) & mask
    hi52 = (C >= C.rolling(252).max()) & mask; lo52 = (C <= C.rolling(252).min()) & mask
    upv = V.where(up).sum(1); dnv = V.where(dn).sum(1)

    n = valid.sum(1)
    I = pd.DataFrame(index=C.index)
    I["n"] = n
    I["pct_above_50"] = (above50.sum(1) / n * 100)
    I["pct_above_200"] = (above200.sum(1) / n * 100)
    I["adv"] = up.sum(1); I["dec"] = dn.sum(1)
    I["ad_pct"] = (I["adv"] - I["dec"]) / (I["adv"] + I["dec"]).replace(0, np.nan) * 100
    I["ud_vol"] = upv / (upv + dnv).replace(0, np.nan) * 100   # up-volume % (low = down-volume washout)
    I["new_hi"] = hi52.sum(1); I["new_lo"] = lo52.sum(1)
    I["nh_nl"] = I["new_hi"] - I["new_lo"]
    rana = ((I["adv"] - I["dec"]) / (I["adv"] + I["dec"]).replace(0, np.nan))
    I["mcclellan"] = ((rana * 1000).ewm(span=19, adjust=False).mean()
                      - (rana * 1000).ewm(span=39, adjust=False).mean())
    return I.reset_index().rename(columns={"index": "date"})


def load_internals(refresh: bool = False, client=None) -> pd.DataFrame:
    """Cached daily internals (computes + persists on first use)."""
    if not refresh and os.path.exists(_INTERNALS_CACHE):
        return pd.read_parquet(_INTERNALS_CACHE)
    I = compute_internals(client)
    os.makedirs(os.path.dirname(_INTERNALS_CACHE), exist_ok=True)
    I.to_parquet(_INTERNALS_CACHE, index=False)
    return I


def washout_signals(internals: pd.DataFrame, prices: pd.DataFrame, *,
                    breadth_pct: float = 0.20, newlow_pct: float = 0.80,
                    lookback: int = 252, window: int = 3) -> pd.Series:
    """A genuine-capitulation confirmation, aligned to ``prices`` (sorted, positional index).

    Confirmed when the selling is broadly exhausted: % above the 50-day is in the bottom
    ``breadth_pct`` of its trailing year **or** new-lows are in the top ``newlow_pct`` (a new-low
    climax). Both are trailing quantiles (regime-aware, no fixed level). ``window`` lets the
    confirmation count if it occurred within the last few days (the fear bar and the breadth climax
    rarely land on the exact same session).
    """
    I = internals.copy(); I["date"] = pd.to_datetime(I["date"])
    I = I.sort_values("date").set_index("date")
    b = I["pct_above_50"]; nl = I["new_lo"]
    b_thr = b.rolling(lookback, min_periods=lookback // 2).quantile(breadth_pct)
    nl_thr = nl.rolling(lookback, min_periods=lookback // 2).quantile(newlow_pct)
    washed = ((b <= b_thr) | (nl >= nl_thr)).fillna(False)
    washed = washed.rolling(window, min_periods=1).max().astype(bool)  # within the last `window` days
    p = prices.sort_values("date").reset_index(drop=True)
    aligned = washed.reindex(pd.to_datetime(p["date"])).ffill(limit=1).fillna(False)
    return pd.Series(aligned.to_numpy(dtype=bool), index=p.index)


def confirm(signal: pd.Series, washout: pd.Series) -> pd.Series:
    """Gate a sleeve's entry signal by breadth confirmation (take only confirmed washouts)."""
    return (signal.reset_index(drop=True) & washout.reset_index(drop=True)).astype(bool)


def not_breaking_down(internals: pd.DataFrame, prices: pd.DataFrame, *,
                      ud_vol_min: float = 20.0) -> pd.Series:
    """IBS dip-buy guard: True when it's safe to buy the dip — i.e. NOT a heavy down-volume day.

    The honest, well-sampled finding (vs the fear-confirmation idea, which did NOT generalize): IBS
    *losers* buy oversold dips on **broad-breakdown** days — up-volume ≤ ~20% (≥80% down-volume),
    advance/decline collapsing — i.e. a falling knife, not a healthy pullback. Up/down volume
    separated IBS winners from losers at AUC 0.78. So we stand aside when up-volume ≤ ``ud_vol_min``.
    This vetoes the 2026-03-18 IBS loss and cuts the book's drawdown ~20% (a risk filter, not an
    alpha booster — it trades a little raw return for higher win-rate and lower drawdown).

    Returns a bool Series aligned to ``prices`` (sorted, positional index); missing internals -> True
    (do not block a trade on absent data).
    """
    I = internals.copy(); I["date"] = pd.to_datetime(I["date"])
    ud = I.sort_values("date").set_index("date")["ud_vol"]
    p = prices.sort_values("date").reset_index(drop=True)
    aligned = ud.reindex(pd.to_datetime(p["date"])).ffill(limit=2).to_numpy()
    # Missing internals -> True (DON'T block a trade on absent data). NaN must be coerced to True
    # BEFORE the comparison: `np.nan > x` is False, which would (wrongly) block — and since the
    # comparison leaves no NaN behind, a trailing .fillna(True) is a dead no-op.
    ok = np.where(np.isnan(aligned), True, aligned > ud_vol_min)
    return pd.Series(ok, index=p.index).astype(bool)


def breadth_tag(internals: pd.DataFrame, date: pd.Timestamp) -> str:
    """A compact human breadth read for a given date, for the trade thesis/regime layer."""
    I = internals.copy(); I["date"] = pd.to_datetime(I["date"])
    row = I[I["date"] <= pd.Timestamp(date)].tail(1)
    if row.empty:
        return "breadth n/a"
    r = row.iloc[0]
    state = "washed-out" if r["pct_above_50"] < 25 else ("broad" if r["pct_above_50"] > 60 else "mixed")
    return (f"breadth {state}: {r['pct_above_50']:.0f}% >50d, {int(r['new_lo'])} new lows, "
            f"up-vol {r['ud_vol']:.0f}%")
