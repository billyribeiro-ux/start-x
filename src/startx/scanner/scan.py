"""P6 — the scanner OUTPUT (Charter v1): explainable, calibrated swing setups.

Turns the validated stress-gated meta-model into the surface the charter mandates — no bare scores.
Per surfaced setup: direction, regime state, location, trigger, a CALIBRATED conviction probability
(isotonic-mapped, not a raw model score), ranked driver attribution (LightGBM SHAP-style contributions),
the invalidation level (the 1-ATR stop), and expectancy + CVaR(5%) from the regime-matched historical
cohort. The GATE is the promoted rule (raw meta-prob >= cutoff in a stress regime); the conviction
DISPLAYED is the calibrated probability (isotonic-mapped to the true ~30% base hit rate, honest not
inflated) — a 33% calibrated conviction is still positive-expectancy on the 2:1 PT/SL payoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .harness import walk_forward
from .signals import FEATURES, _atr

STRESS = ["risk_off", "crisis"]
PROB_CUTOFF = 0.50              # the promoted rule's pre-registered cutoff


@dataclass
class Setup:
    symbol: str
    date: pd.Timestamp
    direction: str
    regime: str
    trigger: str
    conviction: float                          # calibrated probability
    drivers: list[tuple[str, float]]           # ranked (feature, contribution)
    entry_ref: float
    invalidation: float                        # the 1-ATR stop
    cohort_exp: float
    cohort_cvar5: float
    cohort_n: int
    vetoed: bool = False                        # filtered by a self-learned avoid rule
    veto_reason: str = ""                       # which learned loss-pattern vetoed it


@dataclass
class ScanModel:
    model: object
    iso: object                                # isotonic calibration raw_prob -> empirical hit rate
    feat_cols: list[str]
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


def fit(data: pd.DataFrame, *, seed: int = 42) -> ScanModel:
    """Train the meta-model on all history + calibrate its probabilities on walk-forward OOS preds."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    feat_cols = [c for c in FEATURES if c in data.columns] + (["regime_stress"] if "regime_stress" in data else [])
    oos = walk_forward(data, train_min=400, retrain_every=50)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oos["prob"].to_numpy(), oos["label"].to_numpy())     # raw prob -> empirical P(hit) (calibrated)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=-1)
    m.fit(data[feat_cols].fillna(0.0), data["label"])
    return ScanModel(model=m, iso=iso, feat_cols=feat_cols, data=data)


def _cohort(data: pd.DataFrame, regime: str, prob: float, *, band: float = 0.1) -> tuple[float, float, int]:
    """Expectancy + CVaR(5%) from the regime-matched, similar-conviction historical cohort."""
    sub = data[(data.get("regime") == regime)]
    if "cal_prob" in sub:
        sub = sub[(sub["cal_prob"] >= prob - band) & (sub["cal_prob"] <= prob + band)]
    r = sub["ret"]
    if len(r) < 10:
        return float("nan"), float("nan"), len(r)
    cvar = float(r[r <= r.quantile(0.05)].mean())
    return float(r.mean()), cvar, int(len(r))


def surface(sm: ScanModel, prices_by_sym: dict, reg: pd.DataFrame, *, asof=None, lookback: int = 10,
            cutoff: float = PROB_CUTOFF, avoid_rules=None) -> list[Setup]:
    """Surface stress-gated setups whose entry falls in [asof-lookback, asof], calibrated + attributed.

    ``avoid_rules`` (from the self-learning loss memory) do not drop setups silently — each surviving
    setup is flagged ``vetoed`` with the learned loss-pattern that triggered it, so the output shows what
    the system has learned to AVOID and why (transparent, per the charter's no-black-box rule).
    """
    avoid_rules = avoid_rules or []
    data = sm.data.copy()
    # raw prob = the PROMOTED rule's gate (validated at raw>=0.5); calibrated prob = the displayed
    # conviction + the cohort-matching key (isotonic maps raw -> empirical hit rate, base ~30%).
    data["raw_prob"] = sm.model.predict_proba(data[sm.feat_cols].fillna(0.0))[:, 1]
    data["cal_prob"] = sm.iso.predict(data["raw_prob"].to_numpy())
    asof = pd.Timestamp(asof) if asof is not None else data["entry_date"].max()
    lo = asof - pd.Timedelta(days=lookback * 2)
    out = []
    cand = data[(data["entry_date"] >= lo) & (data["entry_date"] <= asof)]
    cand = cand[cand.get("regime").isin(STRESS) & (cand["raw_prob"] >= cutoff)]   # promoted rule
    contrib = sm.model.predict(cand[sm.feat_cols].fillna(0.0), pred_contrib=True) if len(cand) else None
    for ii, (_, row) in enumerate(cand.iterrows()):
        sym = row["symbol"]
        px = prices_by_sym.get(sym)
        atrv = entry = inval = float("nan")
        if px is not None:
            p = px.sort_values("date").reset_index(drop=True)
            j = p.index[pd.to_datetime(p["date"]) == row["entry_date"]]
            if len(j):
                jj = int(j[0])
                entry = float(p["open"].iloc[jj]) if jj < len(p) else float("nan")
                atrv = float(_atr(p).iloc[jj - 1]) if jj >= 1 else float("nan")
                inval = entry - 1.0 * atrv                       # the 1-ATR stop = invalidation
        drivers = []
        if contrib is not None:
            c = contrib[ii][:-1]
            order = np.argsort(-np.abs(c))[:3]
            drivers = [(sm.feat_cols[k], float(c[k])) for k in order]
        cexp, ccvar, cn = _cohort(data, row.get("regime"), float(row["cal_prob"]))
        veto_reason = ""
        for rule in avoid_rules:
            if bool(rule.mask_vetoed(row.to_frame().T.reset_index(drop=True))[0]):
                veto_reason = rule.rationale or rule.key()
                break
        out.append(Setup(symbol=sym, date=row["entry_date"], direction="long",
                         regime=row.get("regime"), trigger=row["trigger"], conviction=float(row["cal_prob"]),
                         drivers=drivers, entry_ref=entry, invalidation=inval,
                         cohort_exp=cexp, cohort_cvar5=ccvar, cohort_n=cn,
                         vetoed=bool(veto_reason), veto_reason=veto_reason))
    # surviving setups first (by conviction), vetoed ones after — both shown, transparently
    return sorted(out, key=lambda s: (s.vetoed, -s.conviction))
