"""Layer 1 — REGIME (Charter v1, P2): the conditioner, on REAL data only.

Regime is never a trigger; it sets which playbook is active (continuation vs fade, risk appetite) and
how big. Built only from series we actually have — vol term structure, vol-of-vol, bond vol, the
volatility risk premium, credit, cross-asset (rates/dollar), and survivorship-free breadth. The charter
items that need data we don't have on this plan are SEAMS, left dark, never faked:
  * dealer gamma / charm / vanna, DIX  -> OPRA options (absent)
  * net liquidity (Fed BS - TGA - RRP) -> FRED (absent)
  * implied correlation / dispersion   -> single-name IV (absent)

Classification is RULE-BASED and point-in-time: every input is z-scored on an EXPANDING window lagged
one bar (standardisation uses only data ≤ t-1), so the regime at t uses no future information — unlike a
full-sample K-means whose cluster centres peek. A composite STRESS score (mean of standardised risk-off
signals) maps to four regimes. The regime layer is a conditioner, not a tradeable edge: it is validated
by eyeball against known events (COVID, 2022 bear), not by the Sharpe firewall.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

PRICE_DIR = "data/cache/prices"
TRADING = 252

# regimes by composite stress score (low stress = risk-on ... high = crisis)
REGIMES = ["risk_on", "neutral", "risk_off", "crisis"]


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet", columns=["date", "close"])
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").set_index("date")["close"].astype(float)


def _zexp(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Expanding z-score, standardisation params lagged one bar -> strictly point-in-time."""
    mu = s.expanding(min_periods=min_periods).mean().shift(1)
    sd = s.expanding(min_periods=min_periods).std().shift(1)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)


def pit_breadth(membership, *, min_history: int = 200) -> pd.DataFrame:
    """Survivorship-free breadth: % of point-in-time S&P 500 members above their 50/200-day SMA.

    Membership is resolved on a monthly grid (it changes ~monthly) and applied to daily bars — PIT and
    tractable. Only names with price data contribute (coverage is ~98-100% in the real-data window).
    """
    above50, above200 = {}, {}
    for f in glob.glob(f"{PRICE_DIR}/*.parquet"):
        sym = os.path.basename(f)[:-8]
        if sym.startswith("_") or sym in ("SPY", "GLD", "UUP", "USO", "CPER", "TLT", "IEF", "HYG", "LQD"):
            continue
        c = _load(sym)
        if c is None or len(c) < min_history:
            continue
        above50[sym] = c > c.rolling(50).mean()
        above200[sym] = c > c.rolling(200).mean()
    A50 = pd.DataFrame(above50)
    A200 = pd.DataFrame(above200)
    idx = A200.index
    months = pd.DatetimeIndex(sorted({pd.Timestamp(y, m, 1) for y, m in
                                      {(d.year, d.month) for d in idx}}))
    out = []
    for i, m0 in enumerate(months):
        m1 = months[i + 1] if i + 1 < len(months) else idx.max() + pd.Timedelta(days=1)
        mem = [s for s in membership.members_asof(m0) if s in A200.columns]
        if not mem:
            continue
        win = (idx >= m0) & (idx < m1)
        if not win.any():
            continue
        sub50 = A50.loc[win, mem]
        sub200 = A200.loc[win, mem]
        out.append(pd.DataFrame({
            "breadth_50": sub50.mean(axis=1, skipna=True),
            "breadth_200": sub200.mean(axis=1, skipna=True),
        }))
    return pd.concat(out).sort_index()


def regime_panel(membership=None) -> pd.DataFrame:
    """Assemble the Layer-1 regime feature panel (real data only), daily, point-in-time."""
    vix, vix9d, vix3m = _load("_VIX"), _load("_VIX9D"), _load("_VIX3M")
    vvix, move = _load("_VVIX"), _load("_MOVE")
    spy, hyg, lqd, uup = _load("SPY"), _load("HYG"), _load("LQD"), _load("UUP")
    idx = spy.index
    df = pd.DataFrame(index=idx)

    spy_ret = spy.pct_change()
    rvol = spy_ret.rolling(20).std() * np.sqrt(TRADING) * 100      # realised vol in vol points

    # vol term structure: >1 = front-month backwardation = stress
    df["ts_9d"] = (vix9d / vix).reindex(idx)
    df["ts_3m"] = (vix / vix3m).reindex(idx)
    # vol-of-vol & bond vol
    df["vvix"] = vvix.reindex(idx)
    df["move"] = move.reindex(idx)
    # volatility risk premium: implied (VIX) - realised; low/negative = danger
    df["vrp"] = (vix.reindex(idx) - rvol)
    # credit appetite: HYG/LQD trend (falling = risk-off)
    cr = (hyg / lqd).reindex(idx)
    df["credit_mom"] = cr / cr.rolling(20).mean() - 1.0
    # dollar: UUP trend (rising = tightening/risk-off)
    df["usd_mom"] = (uup / uup.rolling(20).mean() - 1.0).reindex(idx)
    # trend / realised-vol state
    df["spy_vs_200"] = (spy / spy.rolling(200).mean() - 1.0).reindex(idx)
    df["rvol"] = rvol

    # survivorship-free breadth (the standout — most retail gets this wrong)
    if membership is not None:
        br = pit_breadth(membership)
        df = df.join(br.reindex(idx))

    return df


def classify(panel: pd.DataFrame) -> pd.DataFrame:
    """Rule-based, point-in-time stress score -> four regimes. Returns panel + stress + regime."""
    p = panel.copy()
    # risk-OFF-positive standardised signals (each higher => more stress)
    sig = pd.DataFrame(index=p.index)
    sig["backwardation_9d"] = _zexp(p["ts_9d"])
    sig["backwardation_3m"] = _zexp(p["ts_3m"])
    sig["vvix"] = _zexp(p["vvix"])
    sig["move"] = _zexp(p["move"])
    sig["low_vrp"] = -_zexp(p["vrp"])
    sig["credit_stress"] = -_zexp(p["credit_mom"])
    sig["usd_up"] = _zexp(p["usd_mom"])
    sig["downtrend"] = -_zexp(p["spy_vs_200"])
    sig["high_rvol"] = _zexp(p["rvol"])
    if "breadth_200" in p:
        sig["weak_breadth"] = -_zexp(p["breadth_200"])

    stress = sig.mean(axis=1, skipna=True)
    p["stress"] = stress
    # discrete regimes by the stress score's own expanding quantiles (PIT thresholds)
    q33 = stress.expanding(min_periods=252).quantile(0.40).shift(1)
    q66 = stress.expanding(min_periods=252).quantile(0.75).shift(1)
    q90 = stress.expanding(min_periods=252).quantile(0.925).shift(1)
    reg = pd.Series("neutral", index=p.index, dtype=object)
    reg[stress <= q33] = "risk_on"
    reg[(stress > q66) & (stress <= q90)] = "risk_off"
    reg[stress > q90] = "crisis"
    reg[stress.isna()] = np.nan
    p["regime"] = reg
    return p
