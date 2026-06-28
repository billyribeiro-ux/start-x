"""P4 — the validation harness (Charter v1): the gate every signal must clear before it promotes.

Meta-labeling: primary events (from signals.primary_events) are the hypotheses; a walk-forward LightGBM
META-model learns P(profit-target hit first) from the regime×location×trigger features, sizing the bet
separately from the direction. A taken event's realised triple-barrier return (next-bar-OPEN entry,
ATR-sized barriers, net of costs) is the trade. Selection metric is NEVER win_rate — we report
expectancy, profit factor, CVaR(5%), and the Deflated Sharpe / PBO. Nothing is called an edge unless it
clears DSR > 0 and PBO < 0.5 cost-adjusted OOS.

Leakage is a unit test, not a hope: ``canary_shuffle`` shuffles the labels and confirms OOS AUC collapses
to ~0.5; ``canary_lookahead`` injects a future-peeking feature and confirms it is caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..validation.metrics import deflated_sharpe
from .signals import FEATURES, feature_frame, primary_events


@dataclass
class HarnessResult:
    stats: dict = field(default_factory=dict)
    oos: pd.DataFrame = field(default_factory=pd.DataFrame)   # per-event OOS prob + return
    n_trials: int = 0


def _atr(p, n=14):
    h, lo, c = p["high"].astype(float), p["low"].astype(float), p["close"].astype(float)
    tr = pd.concat([(h - lo), (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def build_dataset(symbol, prices, spy, regime=None, *, pt_mult=2.0, sl_mult=1.0, horizon=10,
                  cost_bps=3.0):
    """Events + PIT features + triple-barrier meta-label (next-OPEN entry, ATR-sized). One symbol."""
    p = prices.sort_values("date").reset_index(drop=True)
    feats = feature_frame(p, spy)
    ev = primary_events(p, feats)
    if ev.empty:
        return pd.DataFrame()
    o = p["open"].to_numpy(float)
    h = p["high"].to_numpy(float)
    lo = p["low"].to_numpy(float)
    c = p["close"].to_numpy(float)
    atr = _atr(p).to_numpy(float)
    dates = pd.DatetimeIndex(pd.to_datetime(p["date"]))
    pos = {d: i for i, d in enumerate(dates)}
    cost = cost_bps / 1e4
    rows = []
    for _, e in ev.iterrows():
        i = pos.get(e["date"])
        if i is None or i + 1 >= len(c) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        ie = i + 1                                          # next-bar OPEN entry (no signal-close fill)
        entry = o[ie]
        pt = entry + pt_mult * atr[i]
        sl = entry - sl_mult * atr[i]
        last = min(ie + horizon, len(c) - 1)
        label, exit_px, jx = 0, c[last], last
        for j in range(ie, last + 1):
            if h[j] >= pt:                                  # profit target first -> a "take" win
                label, exit_px, jx = 1, pt, j
                break
            if lo[j] <= sl:                                 # stop first
                label, exit_px, jx = 0, sl, j
                break
        ret = (exit_px / entry - 1.0) - cost
        rec = {"symbol": symbol, "date": e["date"], "entry_date": dates[ie], "t1": dates[jx],
               "trigger": e["trigger"], "label": label, "ret": ret}
        frow = feats.loc[e["date"]]
        for k in FEATURES:
            rec[k] = float(frow.get(k, np.nan))
        if regime is not None and e["date"] in regime.index:
            rec["regime_stress"] = float(regime.loc[e["date"], "stress"])
            rec["regime"] = regime.loc[e["date"], "regime"]
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=FEATURES, how="all")
    return df


def walk_forward(ds: pd.DataFrame, *, train_min=200, retrain_every=50, seed=42):
    """Expanding-window meta-model; predict each event OOS, purging labels that overlap the train cut."""
    import lightgbm as lgb
    feat_cols = [c for c in FEATURES if c in ds.columns] + (["regime_stress"] if "regime_stress" in ds else [])
    d = ds.sort_values("entry_date").reset_index(drop=True)
    X = d[feat_cols].fillna(0.0).to_numpy()
    y = d["label"].to_numpy()
    entry = d["entry_date"].to_numpy()
    t1 = d["t1"].to_numpy()
    prob = np.full(len(d), np.nan)
    model = None
    for k in range(len(d)):
        if k < train_min:
            continue
        if model is None or (k - train_min) % retrain_every == 0:
            # PURGE: train only on events whose label fully closed before this event's entry (no overlap)
            mask = t1[:k] < entry[k]
            if mask.sum() < 100 or len(np.unique(y[:k][mask])) < 2:
                continue
            model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                       min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                       random_state=seed, n_jobs=1, verbosity=-1)
            model.fit(X[:k][mask], y[:k][mask])
        if model is not None:
            prob[k] = model.predict_proba(X[k:k + 1])[0, 1]
    d["prob"] = prob
    return d.dropna(subset=["prob"]).reset_index(drop=True)


def _curve_stats(r: pd.Series, ppy=25):
    r = r.dropna()
    if len(r) < 12:
        return {}
    sd = r.std(ddof=1)
    wins, losses = r[r > 0], r[r < 0]
    cvar = float(r[r <= r.quantile(0.05)].mean()) if (r <= r.quantile(0.05)).any() else float("nan")
    return {
        "n": int(len(r)), "expectancy": float(r.mean()), "profit_factor":
        float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan"),
        "cvar5": cvar, "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0, "take_rate": float((r != 0).mean()),
    }


def evaluate(oos: pd.DataFrame, thresholds=(0.5, 0.55, 0.6, 0.65), *, n_trials=1, n_blocks=10):
    """Take events with prob>threshold; OOS net-return stats + DSR per threshold; PBO via proper CSCV."""
    by_thr = {}
    streams = {}
    for thr in thresholds:
        taken = oos[oos["prob"] >= thr]
        r = taken.set_index("entry_date")["ret"].sort_index()
        st = _curve_stats(r)
        if st:
            st["dsr"] = float(deflated_sharpe(r.values, n_trials=n_trials))
            st["threshold"] = thr
        by_thr[thr] = st
        streams[thr] = r
    pbo = _pbo_cscv(oos, thresholds, n_blocks=n_blocks)
    return by_thr, pbo, streams


def _pbo_cscv(oos: pd.DataFrame, thresholds, *, n_blocks=10) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey/Borwein/LdP/Zhu).

    Block the OOS events by time -> performance matrix M[block, config]. Over all combinatorial IS/OOS
    block partitions, PBO = fraction where the IS-best threshold is OOS-below-median. >=0.5 = the
    threshold selection is no better than chance (overfit). Needs >=3 configs to be meaningful.
    """
    from itertools import combinations
    d = oos.sort_values("entry_date").reset_index(drop=True)
    if len(d) < n_blocks * 4 or len(thresholds) < 3:
        return float("nan")
    blocks = np.array_split(np.arange(len(d)), n_blocks)
    M = np.full((n_blocks, len(thresholds)), np.nan)
    for bi, blk in enumerate(blocks):
        sub = d.iloc[blk]
        for ci, thr in enumerate(thresholds):
            taken = sub.loc[sub["prob"] >= thr, "ret"]
            M[bi, ci] = taken.mean() if len(taken) >= 1 else np.nan
    half = n_blocks // 2
    flags = []
    for is_idx in combinations(range(n_blocks), half):
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_perf = np.nanmean(M[list(is_idx)], axis=0)
        oos_perf = np.nanmean(M[list(oos_idx)], axis=0)
        if np.all(np.isnan(is_perf)) or np.all(np.isnan(oos_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        valid = np.isfinite(oos_perf)
        if valid.sum() < 2:
            continue
        rank = (np.sum(oos_perf[valid] <= oos_perf[best]) - 1) / (valid.sum() - 1)
        flags.append(1.0 if rank < 0.5 else 0.0)        # IS-best landed OOS-below-median => overfit
    return float(np.mean(flags)) if flags else float("nan")


# --------------------------------------------------------------------------- LEAKAGE CANARIES
def oos_permutation_importance(ds: pd.DataFrame, *, seed=0) -> pd.DataFrame:
    """MDA feature importance measured OOS (not in-sample gain): train on the first 60%, then permute
    each feature in the held-out 40% and record the AUC drop. Positive = the feature carries real OOS
    skill (promote); ~0 or negative = noise (candidate for decay/retirement). The 'self-learning' signal.
    """
    import lightgbm as lgb
    feat_cols = [c for c in FEATURES if c in ds.columns] + (["regime_stress"] if "regime_stress" in ds else [])
    d = ds.sort_values("entry_date").reset_index(drop=True)
    k = int(len(d) * 0.6)
    if k < 150 or len(d) - k < 80:
        return pd.DataFrame()
    Xtr, ytr = d[feat_cols].fillna(0.0).iloc[:k], d["label"].iloc[:k]
    Xte, yte = d[feat_cols].fillna(0.0).iloc[k:].reset_index(drop=True), d["label"].iloc[k:].to_numpy()
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=-1)
    m.fit(Xtr, ytr)
    base = _auc(yte, m.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for col in feat_cols:
        Xp = Xte.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        drop = base - _auc(yte, m.predict_proba(Xp)[:, 1])
        rows.append({"feature": col, "auc_drop": float(drop)})
    out = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    out.attrs["base_auc"] = float(base)
    return out


def canary_shuffle(ds: pd.DataFrame, *, seed=0) -> float:
    """Shuffle the labels -> a correct harness must give OOS AUC ~0.5 (no edge from noise)."""
    rng = np.random.default_rng(seed)
    d = ds.copy()
    d["label"] = rng.permutation(d["label"].to_numpy())
    oos = walk_forward(d, train_min=min(200, len(d) // 3), retrain_every=50)
    if len(oos) < 30:
        return float("nan")
    return _auc(oos["label"].to_numpy(), oos["prob"].to_numpy())


def canary_lookahead(ds: pd.DataFrame) -> float:
    """Inject a FUTURE-peeking feature (the label itself) -> OOS AUC must spike to ~1.0 (caught)."""
    d = ds.copy()
    d["__leak__"] = d["label"].astype(float)               # blatant leak
    import lightgbm as lgb
    from .signals import FEATURES as F
    cols = [c for c in F if c in d.columns] + ["__leak__"]
    d = d.sort_values("entry_date").reset_index(drop=True)
    k = len(d) // 2
    m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, random_state=0, n_jobs=1, verbosity=-1)
    m.fit(d[cols].fillna(0).iloc[:k], d["label"].iloc[:k])
    p = m.predict_proba(d[cols].fillna(0).iloc[k:])[:, 1]
    return _auc(d["label"].iloc[k:].to_numpy(), p)


def _auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos*n_neg) = AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
