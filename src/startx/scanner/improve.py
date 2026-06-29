"""Improvement levers beyond avoidance — exits and conviction-sizing (Charter v1, firewall-gated).

R5 proved the stress-gated swing losses are not a separably-avoidable subset: the loss-prone cohort has
the fattest right tail. That points at two levers that work WITH the variance instead of against it:

  EXITS    — the losers bleed through a tight 1-ATR stop. Re-simulate the SAME meta-model entries under a
             pre-registered grid of stop/target policies (incl. a regime-scaled stop that widens in stress).
             Only the exit changes, so the meta-model is untouched and every alternative outcome is honest
             OOS. Multiple-testing is paid via the DSR with N = grid size.
  SIZING   — instead of equal-weighting, lean into the fat right tail. Two pre-registered schemes vs
             equal-weight: a walk-forward edge model (LightGBM regressor on realised return, purged) and a
             theory-driven stress-depth weight. Judged as an INVESTABLE calendar book (Sharpe / maxDD /
             Calmar / DSR), not per-trade expectancy — per R4, per-trade alpha is not an investable book.

Nothing here re-fits the entry model; these operate on the cached taken book + the price cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..validation.metrics import deflated_sharpe
from .signals import FEATURES, _atr

STRESS = ["risk_off", "crisis"]
TR = 252


def _resim(taken: pd.DataFrame, prices_by_sym: dict, *, pt_mult, sl_mult, horizon, stress_sl_mult,
           stress_pt_mult, cost_bps):
    """Core barrier re-walk. Returns (rets, exit_dates) aligned to ``taken`` rows (NaN/NaT if unlocatable)."""
    cost = cost_bps / 1e4
    rets = np.full(len(taken), np.nan)
    exits = np.full(len(taken), np.datetime64("NaT"), dtype="datetime64[ns]")
    cache = {}
    for pos, (_, e) in enumerate(taken.iterrows()):
        sym = e["symbol"]
        if sym not in cache:
            p = prices_by_sym.get(sym)
            cache[sym] = None if p is None else (
                p.sort_values("date").reset_index(drop=True),
                pd.DatetimeIndex(pd.to_datetime(p.sort_values("date")["date"])), None)
            if cache[sym] is not None:
                pp = cache[sym][0]
                cache[sym] = (pp, cache[sym][1], _atr(pp).to_numpy(float))
        if cache[sym] is None:
            continue
        p, dates, atr = cache[sym]
        i = int(dates.get_indexer([pd.Timestamp(e["date"])])[0])
        if i < 0 or i + 1 >= len(p) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        o = p["open"].to_numpy(float)
        h = p["high"].to_numpy(float)
        lo = p["low"].to_numpy(float)
        c = p["close"].to_numpy(float)
        ie = i + 1
        entry = o[ie]
        stress = e.get("regime") in STRESS
        slm = stress_sl_mult if (stress and stress_sl_mult is not None) else sl_mult
        ptm = stress_pt_mult if (stress and stress_pt_mult is not None) else pt_mult
        pt = entry + ptm * atr[i]
        sl = entry - slm * atr[i]
        last = min(ie + horizon, len(c) - 1)
        exit_px, jx = c[last], last
        for j in range(ie, last + 1):
            if h[j] >= pt:
                exit_px, jx = pt, j
                break
            if lo[j] <= sl:
                exit_px, jx = sl, j
                break
        rets[pos] = (exit_px / entry - 1.0) - cost
        exits[pos] = dates[jx].to_datetime64()
    return rets, exits


def resim_returns(taken: pd.DataFrame, prices_by_sym: dict, *, pt_mult: float = 2.0, sl_mult: float = 1.0,
                  horizon: int = 10, stress_sl_mult: float | None = None, stress_pt_mult: float | None = None,
                  cost_bps: float = 3.0) -> np.ndarray:
    """Re-walk the triple barrier for each taken entry under an alternative exit policy.

    Reconstructs the exact build_dataset entry (signal bar = ``date``, next-OPEN entry, ATR at the signal
    bar), then walks profit-target / stop / time. ``stress_sl_mult`` / ``stress_pt_mult`` override the
    multipliers when the trade's regime is risk_off/crisis (the regime-scaled exit). Returns realised
    net returns aligned to ``taken`` rows (NaN where the bar can't be located).
    """
    rets, _ = _resim(taken, prices_by_sym, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
                     stress_sl_mult=stress_sl_mult, stress_pt_mult=stress_pt_mult, cost_bps=cost_bps)
    return rets


def resim_trades(taken: pd.DataFrame, prices_by_sym: dict, *, pt_mult: float = 2.0, sl_mult: float = 1.0,
                 horizon: int = 10, stress_sl_mult: float | None = None, stress_pt_mult: float | None = None,
                 cost_bps: float = 3.0) -> pd.DataFrame:
    """Copy of ``taken`` with ``ret`` AND ``t1`` recomputed under an alternative exit policy (so a calendar
    book reflects the true, often longer, hold under a wider stop). Rows that can't be located are dropped."""
    rets, exits = _resim(taken, prices_by_sym, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
                         stress_sl_mult=stress_sl_mult, stress_pt_mult=stress_pt_mult, cost_bps=cost_bps)
    out = taken.copy()
    out["ret"] = rets
    out["t1"] = pd.to_datetime(exits)
    return out.dropna(subset=["ret", "t1"]).reset_index(drop=True)


def per_trade_stats(r: np.ndarray, *, n_trials: int = 1) -> dict:
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std(ddof=1)
    return {"n": int(len(r)), "exp": float(r.mean()),
            "pf": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "win_share": float((r > 0).mean()),
            "sharpe": float(r.mean() / sd * np.sqrt(25)) if sd > 0 else float("nan"),
            "dsr": float(deflated_sharpe(r.values, n_trials=n_trials))}


def calendar_book(taken: pd.DataFrame, rets: np.ndarray, *, weights: np.ndarray | None = None,
                  max_concurrent: int | None = None) -> pd.Series:
    """Daily-MTM calendar book: each trade spreads its return over (entry, t1] as ret/hold, combined as a
    weighted average across concurrent positions. ``max_concurrent`` caps open slots (investable realism).
    """
    df = taken.copy()
    df["__ret"] = rets
    df["__w"] = 1.0 if weights is None else weights
    df = df.dropna(subset=["__ret"]).sort_values("entry_date").reset_index(drop=True)
    if df.empty:
        return pd.Series(dtype=float)
    days = pd.bdate_range(df["entry_date"].min(), pd.to_datetime(df["t1"]).max())
    daily = pd.Series(0.0, index=days)
    open_pos = []                      # (exit_date, per_day_ret, weight)
    ev = df.to_dict("records")
    ti = 0
    for d in days:
        open_pos = [(x, r, w) for (x, r, w) in open_pos if x >= d]
        while ti < len(ev) and pd.Timestamp(ev[ti]["entry_date"]) <= d:
            e = ev[ti]
            x = pd.Timestamp(e["t1"])
            hold = max(len(pd.bdate_range(pd.Timestamp(e["entry_date"]) + pd.Timedelta(days=1), x)), 1)
            if pd.Timestamp(e["entry_date"]) == d and (max_concurrent is None or len(open_pos) < max_concurrent):
                open_pos.append((x, e["__ret"] / hold, e["__w"]))
            ti += 1
        if open_pos:
            wsum = sum(w for (_, _, w) in open_pos) or 1.0
            daily.loc[d] = sum(r * w for (_, r, w) in open_pos) / wsum
    return daily


def book_stats(daily: pd.Series, *, n_trials: int = 1) -> dict:
    r = daily.dropna()
    if len(r) < 30:
        return {"n_days": int(len(r))}
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1).min())
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(r.mean() / sd * np.sqrt(TR)) if sd > 0 else float("nan")
    return {"n_days": int(len(r)), "sharpe": sh, "cagr": cagr, "maxdd": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
            "dsr": float(deflated_sharpe(r.values, n_trials=n_trials))}


def walk_forward_edge(taken: pd.DataFrame, feat_cols: list[str], *, train_min: int = 120,
                      retrain_every: int = 30, seed: int = 42) -> np.ndarray:
    """Expanding-window LightGBM REGRESSOR predicting realised return per taken trade, purged. OOS edge
    estimate used for conviction sizing (size proportional to predicted edge). NaN before train_min.
    """
    import lightgbm as lgb
    d = taken.sort_values("entry_date").reset_index(drop=True)
    X = d[feat_cols].fillna(0.0).to_numpy()
    y = d["ret"].to_numpy(float)
    entry = d["entry_date"].to_numpy()
    t1 = d["t1"].to_numpy()
    pred = np.full(len(d), np.nan)
    model = None
    for k in range(len(d)):
        if k < train_min:
            continue
        if model is None or (k - train_min) % retrain_every == 0:
            mask = t1[:k] < entry[k]
            if mask.sum() < 80:
                continue
            model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                      min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                                      random_state=seed, n_jobs=1, verbosity=-1)
            model.fit(X[:k][mask], y[:k][mask])
        if model is not None:
            pred[k] = float(model.predict(X[k:k + 1])[0])
    return pred


def feat_cols_of(taken: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in taken.columns] + (["regime_stress"] if "regime_stress" in taken else [])
