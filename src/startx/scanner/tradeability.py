"""Layer 0 — the TRADEABILITY GATE (Charter v1, P1): the 'tickers that don't care' filter.

Most names have no systematic hand leaning on them; absent a large forced transactor there is no front
to trade, so we score and gate each symbol and only scan the tradeable set. The charter's flow-
sensitivity score draws on four sources — here we implement the ones with real data and SURFACE the
rest as seams (never fake them):

  * dollar liquidity (REAL, fmp_eod) — median trailing 63d $-volume; a thin tape has no resting hand.
  * macro responsiveness (REAL, fmp_eod) — R² of the name's weekly returns on the macro factor set
    [market SPY, rates TLT, credit HYG, dollar UUP]. High R² = a systematic hand transacts it; low R²
    = idiosyncratic, no macro front. This is the charter's 'beta-to-macro' component.
  * index / ETF flow (REAL, membership) — S&P 500 member (point-in-time) or a major index/sector ETF
    (creation-redemption flow vehicle): structurally rebalanced size must transact it.
  * dealer GEX / options liquidity (SEAM — OPRA options data not available on this plan). Flagged,
    not fabricated; when wired it ADDS to the score, it doesn't gate alone.

Gate: tradeable iff liquidity ≥ floor AND has index/ETF flow AND macro-responsive ≥ floor. The score is
the mean of the available z-scored components (the seam is excluded from the mean and reported).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .feature_store import (
    LEAD_SLOW,
    LEAD_STATIC,
    LEAD_SWING,
    Feature,
    FeatureSet,
)

MACRO_FACTORS = ["SPY", "TLT", "HYG", "UUP"]      # market, rates, credit, dollar
MIN_DOLLAR_VOL = 2e7                               # $20M/day floor (tradeable)
MIN_MACRO_R2 = 0.20                                # below this the name is idiosyncratic (no macro front)


@dataclass
class Tradeability:
    symbol: str
    as_of: pd.Timestamp
    tradeable: bool
    score: float                  # mean of available z-scored components (higher = stronger front)
    dollar_vol: float
    macro_r2: float
    index_flow: bool
    reasons: list[str]            # why rejected (empty if tradeable)
    features: FeatureSet


def _macro_r2(sym_px: pd.DataFrame, factors: dict[str, pd.DataFrame], asof: pd.Timestamp,
              lookback: int = 252) -> float:
    """R² of the symbol's weekly returns on the macro factor weekly returns, trailing ``lookback`` days."""
    def wk(px):
        s = px[px["date"] <= asof].set_index("date")["close"].astype(float)
        return s.resample("W-FRI").last().pct_change()
    y = wk(sym_px).tail(lookback // 5)
    X = pd.DataFrame({k: wk(v) for k, v in factors.items() if v is not None}).reindex(y.index)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 30 or df.shape[1] < 2:
        return float("nan")
    A = np.column_stack([np.ones(len(df)), df[[c for c in X.columns]].to_numpy()])
    yy = df["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score_symbol(symbol: str, sym_px: pd.DataFrame, factors: dict, asof, *, is_member: bool,
                 is_etf: bool, z_ctx: dict | None = None) -> Tradeability:
    """Score one symbol's tradeability as of ``asof`` (PIT). ``z_ctx`` carries cross-sectional means/sds."""
    asof = pd.Timestamp(asof)
    fs = FeatureSet(symbol=symbol, as_of=asof)
    hist = sym_px[sym_px["date"] <= asof]
    # 1) dollar liquidity (trailing 63d median)
    if len(hist) >= 63:
        dv = float((hist["close"].astype(float) * hist["volume"].astype(float)).tail(63).median())
    else:
        dv = float("nan")
    fs.add(Feature("dollar_vol_63d", dv, asof, LEAD_SWING, "L0_tradeability", "fmp_eod"))
    # 2) macro responsiveness
    r2 = _macro_r2(sym_px, factors, asof)
    fs.add(Feature("macro_r2", r2, asof, LEAD_SLOW, "L0_tradeability", "fmp_eod"))
    # 3) index / ETF flow
    flow = bool(is_member or is_etf)
    fs.add(Feature("index_etf_flow", float(flow), asof, LEAD_STATIC, "L0_tradeability", "membership"))
    # 4) dealer GEX / options liquidity — SEAM (no OPRA data on this plan)
    fs.add(Feature.seam("dealer_gex", asof, LEAD_SWING, "L0_tradeability", "OPRA"))
    fs.add(Feature.seam("options_liquidity", asof, LEAD_SWING, "L0_tradeability", "OPRA"))

    # GATE: liquid AND has a systematic hand. The 'hand' is index/ETF flow OR strong macro response —
    # either is sufficient (a liquid S&P member is tradeable even if its 1yr macro-R² is low; a liquid
    # non-member is tradeable only if it strongly tracks the macro complex). Macro-R² and the GEX seam
    # REFINE the score, they don't gate alone (per the charter). Idiosyncratic illiquidity is the kill.
    macro_responsive = bool(np.isfinite(r2) and r2 >= MIN_MACRO_R2)
    reasons = []
    if not (np.isfinite(dv) and dv >= MIN_DOLLAR_VOL):
        reasons.append(f"illiquid (${dv/1e6:.1f}M/d < ${MIN_DOLLAR_VOL/1e6:.0f}M)" if np.isfinite(dv)
                       else "no liquidity history")
    if not (flow or macro_responsive):
        reasons.append(f"no front: not an S&P member/ETF and idiosyncratic (macro R²={r2:.2f})"
                       if np.isfinite(r2) else "no front: not a member/ETF and macro R² unavailable")
    tradeable = len(reasons) == 0

    # composite score = mean of available z-scored components (seam excluded)
    z = z_ctx or {}
    parts = []
    if np.isfinite(dv) and "dv" in z:
        parts.append((np.log10(max(dv, 1)) - z["dv"][0]) / z["dv"][1] if z["dv"][1] else 0.0)
    if np.isfinite(r2) and "r2" in z:
        parts.append((r2 - z["r2"][0]) / z["r2"][1] if z["r2"][1] else 0.0)
    parts.append(0.5 if flow else -0.5)
    score = float(np.clip(np.mean(parts), -4, 4)) if parts else float("nan")

    return Tradeability(symbol=symbol, as_of=asof, tradeable=tradeable, score=score, dollar_vol=dv,
                        macro_r2=r2, index_flow=flow, reasons=reasons, features=fs)
