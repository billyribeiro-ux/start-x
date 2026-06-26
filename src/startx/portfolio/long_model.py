"""The 'better long model' — vol-targeted risk-parity across the two SWING books.

The combination that beat the best single book out-of-sample (a 3-agent build-off, then a
measured-not-asserted reproduction): NOT the naive max-Sharpe ensemble (failed walk-forward,
Sharpe 0.36), NOT a discretionary regime tilt (0.92), NOT the ML cross-sectional ranker (a
survivorship mirage, ~0 honest Sharpe), NOT a capitulation-long add (one-trade noise). What works is
the dumb, mechanical rule applied to the two ACTIVE swing books:

    weight the books by inverse trailing-63d vol (RISK PARITY), then scale the whole blend to a
    fixed annual vol target (trailing-63d realized vol estimate), leverage-capped, rebalanced daily.

On 2019-26 it lifts the best single book (long_swing) from **Sharpe 1.07 / CAGR 8.8% to Sharpe ~1.22
/ CAGR ~13.5%**. The lift is robust: it holds in BOTH halves (bear-containing 2019-22 0.75→0.92, bull
2023-26 1.39→1.59) and across 32/36 of a (target_vol × lev_cap × vol_window) grid, DSR ~0.90.

Two honesty caveats that the build-off's first pass got wrong:
  * **Only the two SWING books belong here.** Folding in System #3 (the position core) DRAGS the
    blend to Sharpe 0.98 — BELOW the best single book — because in a bull window the position book is
    correlated dead weight (corr 0.59 w/ long_swing, maxDD -22%); its drawdown-defense value lives in
    the bear tail 2019-26 doesn't contain. Keep it as its own benchmarked book, out of this blend.
  * **The lift needs the vol-target overlay, and it costs leverage.** The *un-levered* static blend
    alone is Sharpe ~1.02 (≤ long-only 1.07) — diversification alone doesn't clear the bar. The jump
    to 1.22 is the time-varying vol-target overlay on a diversified 2-book base, running ~2.2× mean
    leverage; absolute drawdown (-14.8%) is therefore DEEPER than long_swing alone (-10.1%) even as
    Sharpe and Calmar improve. It is risk-adjusted-better, not free.

Everything here uses TRAILING info only (vols and weights are ``.shift(1)``-lagged) — no lookahead.
The function itself is general (combine any ``streams``); the two-swing choice lives in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class LongModelResult:
    combined: pd.Series                 # the model's daily return stream (OOS, causal)
    equity: pd.Series                   # equity curve (starts at 1.0)
    weights: pd.DataFrame               # per-book risk-parity weights through time
    leverage: pd.Series                 # the vol-target leverage applied each day
    stats: dict = field(default_factory=dict)


def _deflated_sharpe(r: pd.Series, n_trials: int) -> float:
    try:
        from startx.validation.metrics import deflated_sharpe
        return float(deflated_sharpe(r.dropna(), n_trials=n_trials))
    except Exception:
        return float("nan")


def _stats(r: pd.Series, *, n_trials: int = 27) -> dict:
    r = r.fillna(0.0)
    eq = (1.0 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    return {
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "deflated_sharpe": _deflated_sharpe(r, n_trials),
        "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
        "max_drawdown": mdd,
        "cagr": cagr,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "total_return": float(eq.iloc[-1] - 1.0),
    }


def vol_target_risk_parity(streams: dict[str, pd.Series], *, vol_window: int = 63,
                           target_vol: float = 0.10, lev_cap: float = 3.0,
                           n_trials: int = 27) -> LongModelResult:
    """Combine book return ``streams`` by inverse-vol risk parity + vol-targeting (causal).

    streams:    {book_name: daily_return_series}. Aligned on the union of dates.
    vol_window: trailing window (trading days) for both the per-book and combined vol estimates.
    target_vol: annualised vol target for the whole blend.
    lev_cap:    max leverage applied by the vol-target step.
    """
    M = pd.DataFrame(streams)
    M = M[~M.index.duplicated(keep="first")].sort_index().fillna(0.0)
    if M.shape[1] == 0:
        raise ValueError("no streams supplied")

    # 1) RISK-PARITY weights from TRAILING per-book vol (lagged one bar — no lookahead).
    vol = M.rolling(vol_window).std().shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)            # each row sums to 1 (or NaN during warm-up)
    base = (M * w).sum(axis=1).fillna(0.0)          # un-levered combined return

    # 2) VOL-TARGET the combined book on its own TRAILING realised vol (lagged).
    cvol = base.rolling(vol_window).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (target_vol / cvol).clip(upper=lev_cap)
    lev = lev.where(np.isfinite(lev), 0.0).fillna(0.0)
    combined = (base * lev).rename("long_model")

    equity = (1.0 + combined).cumprod()
    return LongModelResult(combined=combined, equity=equity, weights=w.fillna(0.0),
                           leverage=lev, stats=_stats(combined, n_trials=n_trials))
