"""Self-learning loss-attribution layer (Charter v1, P5+).

The meta-model decides *take vs skip*. But among the taken, stress-gated swing trades, the majority still
lose (a ~33% base hit rate on a 2:1 payoff is +EV, yet ~2 of 3 are red). This module is the part of the
system that **learns from its own losers**: it dissects WHY the losing trades lost, distills that into
pre-registered AVOID rules, and keeps only the rules that improve the risk-adjusted return on a
touched-once holdout — never on the data they were mined from.

The firewall that keeps this honest (the whole reason it isn't just curve-fitting the losers):

  * every avoid rule is mined from the DEV split (entry < SPLIT) ONLY, then LOCKED;
  * it is judged ONLY on the HOLDOUT split (entry >= SPLIT), which the rule never saw;
  * promotion requires the rule to improve expectancy AND profit factor AND the Deflated Sharpe at the
    *current* trial count N (which rises every iteration, so the bar rises as we search — Bailey/LdP);
  * and the vetoed cohort must itself be worse OOS than the kept cohort (proof the rule removed bad
    trades, not good ones). A rule that merely shrinks the book is rejected.

State persists in a small JSON memory so a LOOP accumulates: each pass re-validates the standing avoid
set against fresh OOS (decay check), proposes the next-strongest untested loss condition, and promotes or
permanently records it. When candidates are exhausted (or K passes promote nothing) the memory is stable
and the system has converged on what to avoid.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..validation.metrics import deflated_sharpe

# ----------------------------------------------------------------------------- pre-registered constants
SPLIT = pd.Timestamp("2020-01-01")        # DEV (mine) < SPLIT <= HOLDOUT (judge once)
MIN_KEEP = 0.50                           # an avoid rule may not veto more than half the book
TAIL_Q = 0.75                             # avoid only the extreme quartile on the loss side (no threshold search)
MIN_DEV_TAKEN = 150                       # need this many DEV taken trades to mine a rule
MEMORY_PATH = Path("scanner_memory.json")  # repo root (committable learned state; /data/ is gitignored)


@dataclass
class AvoidRule:
    """A pre-registered veto: skip a take when this loss-prone condition holds."""
    kind: str                              # "feature_tail" | "trigger"
    feature: str                           # feature name, or "" for a trigger rule
    op: str                                # ">" | "<" | "==" (== for trigger)
    threshold: float                       # DEV-derived cut (feature_tail), or NaN for trigger
    trigger: str = ""                      # trigger tag for kind=="trigger"
    rationale: str = ""                    # human-readable "why we avoid this"
    dev_loss_rate_in: float = float("nan")  # loss rate inside the vetoed region on DEV
    dev_loss_rate_out: float = float("nan")  # loss rate outside it on DEV

    def key(self) -> str:
        return f"{self.kind}:{self.trigger or self.feature}:{self.op}"

    def mask_vetoed(self, df: pd.DataFrame) -> np.ndarray:
        """True where a row is VETOED by this rule (should be skipped)."""
        if self.kind == "trigger":
            return (df["trigger"] == self.trigger).to_numpy()
        v = df[self.feature].to_numpy(float)
        return (v > self.threshold) if self.op == ">" else (v < self.threshold)


@dataclass
class ScannerMemory:
    """Accumulated, persisted self-learning state (survives across loop iterations)."""
    iteration: int = 0
    promoted: list[dict] = field(default_factory=list)    # promoted AvoidRule dicts
    tested: list[str] = field(default_factory=list)       # rule keys already evaluated (promote or reject)
    rejected: list[dict] = field(default_factory=list)    # {key, reason} for the record
    history: list[dict] = field(default_factory=list)     # per-iteration summary rows
    no_promote_streak: int = 0

    @classmethod
    def load(cls, path: Path = MEMORY_PATH) -> "ScannerMemory":
        if Path(path).exists():
            return cls(**json.loads(Path(path).read_text()))
        return cls()

    def save(self, path: Path = MEMORY_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    def promoted_rules(self) -> list[AvoidRule]:
        return [AvoidRule(**{k: v for k, v in d.items()}) for d in self.promoted]


# ----------------------------------------------------------------------------- the "why": loss post-mortem
def loss_postmortem(taken: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Characterise the loss cohort vs the win cohort — the human-readable WHY.

    Returns per-trigger and per-regime loss rates, and a per-feature standardised separation
    (Welch-style effect size) of losers vs winners. A large positive ``effect`` means losers sit at
    HIGHER values of the feature (so the avoid rule is "skip when feature is high"); negative the reverse.
    """
    t = taken.copy()
    t["loss"] = (t["ret"] <= 0).astype(int)
    win, los = t[t["loss"] == 0], t[t["loss"] == 1]
    by_trigger = (t.groupby("trigger")["loss"].agg(["mean", "count"]).rename(columns={"mean": "loss_rate"}))
    by_regime = (t.groupby("regime")["loss"].agg(["mean", "count"]).rename(columns={"mean": "loss_rate"})
                 if "regime" in t else pd.DataFrame())
    rows = []
    for c in feat_cols:
        a, b = los[c].to_numpy(float), win[c].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 20 or len(b) < 20:
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or np.nan
        eff = (a.mean() - b.mean()) / sd if sd and np.isfinite(sd) else np.nan
        rows.append({"feature": c, "loser_mean": float(a.mean()), "winner_mean": float(b.mean()),
                     "effect": float(eff)})
    sep = (pd.DataFrame(rows).assign(abs_eff=lambda d: d["effect"].abs())
           .sort_values("abs_eff", ascending=False).reset_index(drop=True)) if rows else pd.DataFrame()
    return {"overall_loss_rate": float(t["loss"].mean()), "n": int(len(t)),
            "by_trigger": by_trigger, "by_regime": by_regime, "separation": sep}


# ------------------------------------------------------------------ the "what to avoid": candidate rules
def candidate_avoid_rules(dev_taken: pd.DataFrame, feat_cols: list[str]) -> list[AvoidRule]:
    """Deterministically ranked avoid candidates mined from the DEV split only.

    Two families: (a) feature-tail rules — veto the extreme quartile on the loss side of the strongest-
    separating features; (b) trigger rules — veto a trigger whose DEV loss rate is materially worse than
    the book. Ranked by how much the vetoed region's DEV loss rate exceeds the rest (the mined edge).
    Threshold is a fixed DEV quantile (TAIL_Q), NOT searched — that keeps PBO from inflating.
    """
    pm = loss_postmortem(dev_taken, feat_cols)
    cands: list[AvoidRule] = []

    for _, r in pm["separation"].iterrows():
        f, eff = r["feature"], r["effect"]
        if not np.isfinite(eff) or abs(eff) < 0.10:
            continue
        v = dev_taken[f].to_numpy(float)
        if eff > 0:                                    # losers are HIGH -> avoid the high tail
            thr, op, side = float(np.nanquantile(v, TAIL_Q)), ">", "high"
            vetoed = v > thr
        else:                                          # losers are LOW -> avoid the low tail
            thr, op, side = float(np.nanquantile(v, 1 - TAIL_Q)), "<", "low"
            vetoed = v < thr
        lr_in = float(dev_taken.loc[vetoed, "ret"].le(0).mean()) if vetoed.any() else float("nan")
        lr_out = float(dev_taken.loc[~vetoed, "ret"].le(0).mean()) if (~vetoed).any() else float("nan")
        if not (np.isfinite(lr_in) and lr_in > lr_out):     # the tail must actually lose more on DEV
            continue
        cands.append(AvoidRule(kind="feature_tail", feature=f, op=op, threshold=thr,
                               rationale=f"losers cluster at {side} {f} (DEV loss {lr_in:.0%} vs {lr_out:.0%})",
                               dev_loss_rate_in=lr_in, dev_loss_rate_out=lr_out))

    if "trigger" in dev_taken:
        for trig, g in dev_taken.groupby("trigger"):
            if len(g) < 40:
                continue
            lr_in = float(g["ret"].le(0).mean())
            lr_out = float(dev_taken.loc[dev_taken["trigger"] != trig, "ret"].le(0).mean())
            if lr_in > lr_out + 0.05:                    # this trigger is a meaningfully worse loser on DEV
                cands.append(AvoidRule(kind="trigger", feature="", op="==", threshold=float("nan"),
                                       trigger=trig,
                                       rationale=f"trigger '{trig}' loses {lr_in:.0%} vs {lr_out:.0%} on DEV",
                                       dev_loss_rate_in=lr_in, dev_loss_rate_out=lr_out))

    # rank by mined separation strength (loss-rate gap inside vs outside the veto region)
    cands.sort(key=lambda c: -(c.dev_loss_rate_in - c.dev_loss_rate_out))
    return cands


def apply_avoids(df: pd.DataFrame, rules: list[AvoidRule]) -> np.ndarray:
    """Boolean mask of rows that SURVIVE every avoid rule (True = keep / not vetoed)."""
    keep = np.ones(len(df), dtype=bool)
    for rule in rules:
        keep &= ~rule.mask_vetoed(df)
    return keep


# --------------------------------------------------------------------------- OOS judgement (the firewall)
def _stream_stats(r: pd.Series, n_trials: int) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std(ddof=1)
    return {"n": int(len(r)), "exp": float(r.mean()),
            "pf": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "sharpe": float(r.mean() / sd * np.sqrt(25)) if sd > 0 else float("nan"),
            "dsr": float(deflated_sharpe(r.values, n_trials=n_trials))}


def evaluate_rule_oos(holdout_taken: pd.DataFrame, rule: AvoidRule, *, n_trials: int) -> dict:
    """Judge ONE locked avoid rule on the touch-once holdout. Promotion gate is pre-registered here."""
    vetoed = rule.mask_vetoed(holdout_taken)
    kept = holdout_taken.loc[~vetoed]
    before = _stream_stats(holdout_taken.set_index("entry_date")["ret"].sort_index(), n_trials)
    after = _stream_stats(kept.set_index("entry_date")["ret"].sort_index(), n_trials)
    veto_exp = float(holdout_taken.loc[vetoed, "ret"].mean()) if vetoed.any() else float("nan")
    keep_frac = float((~vetoed).mean())
    ok = (after.get("n", 0) >= 30 and keep_frac >= MIN_KEEP
          and np.isfinite(after.get("exp", np.nan)) and np.isfinite(before.get("exp", np.nan))
          and after["exp"] > before["exp"] and after["pf"] > before["pf"]
          and after["dsr"] >= before["dsr"]
          and np.isfinite(veto_exp) and veto_exp < after["exp"])     # removed worse-than-kept trades OOS
    return {"before": before, "after": after, "veto_exp": veto_exp, "keep_frac": keep_frac,
            "n_vetoed": int(vetoed.sum()), "promote": bool(ok)}


def walk_forward_loss_prob(taken: pd.DataFrame, feat_cols: list[str], *, train_min: int = 120,
                           retrain_every: int = 30, veto_q: float = 0.90, seed: int = 42) -> pd.DataFrame:
    """Expanding-window 2nd-stage P(loss) model — every taken trade scored OUT-OF-SAMPLE.

    Stress regimes are rare, so a single 2020 split starves the DEV side (only ~66 pre-2020 stress takes).
    Instead we walk forward: at each retrain step fit a LightGBM P(loss) model on PAST taken trades whose
    label fully closed before this trade's entry (purged, no overlap), and lock the top-``veto_q`` loss-risk
    cut from that same past window. Each future trade then carries an honest OOS ``ploss`` and the locked
    ``ploss_cut`` it would have been judged against — no peeking. Returns the taken frame + those columns.
    """
    import lightgbm as lgb
    d = taken.sort_values("entry_date").reset_index(drop=True)
    X = d[feat_cols].fillna(0.0).to_numpy()
    yl = (d["ret"] <= 0).astype(int).to_numpy()
    entry = d["entry_date"].to_numpy()
    t1 = d["t1"].to_numpy()
    ploss = np.full(len(d), np.nan)
    cut = np.full(len(d), np.nan)
    model = None
    cut_val = np.nan
    for k in range(len(d)):
        if k < train_min:
            continue
        if model is None or (k - train_min) % retrain_every == 0:
            mask = t1[:k] < entry[k]                          # PURGE overlapping labels
            if mask.sum() < 80 or len(np.unique(yl[:k][mask])) < 2:
                continue
            model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                       min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                                       random_state=seed, n_jobs=1, verbosity=-1)
            model.fit(X[:k][mask], yl[:k][mask])
            cut_val = float(np.quantile(model.predict_proba(X[:k][mask])[:, 1], veto_q))
        if model is not None:
            ploss[k] = model.predict_proba(X[k:k + 1])[0, 1]
            cut[k] = cut_val
    d["ploss"] = ploss
    d["ploss_cut"] = cut
    return d.dropna(subset=["ploss"]).reset_index(drop=True)


def walk_forward_veto_eval(taken: pd.DataFrame, feat_cols: list[str], *, veto_q: float = 0.90,
                           n_trials: int = 1, **wf) -> dict:
    """OOS lift from the walk-forward loss-veto: keep trades whose OOS P(loss) is below the locked cut."""
    d = walk_forward_loss_prob(taken, feat_cols, veto_q=veto_q, **wf)
    if len(d) < 60:
        return {"available": False, "n_scored": len(d)}
    keep = d["ploss"] < d["ploss_cut"]
    before = _stream_stats(d.set_index("entry_date")["ret"].sort_index(), n_trials)
    after = _stream_stats(d.loc[keep].set_index("entry_date")["ret"].sort_index(), n_trials)
    veto_exp = float(d.loc[~keep, "ret"].mean()) if (~keep).any() else float("nan")
    lift = (after.get("exp", np.nan) - before.get("exp", np.nan)) if after.get("n", 0) >= 30 else np.nan
    return {"available": True, "veto_q": veto_q, "before": before, "after": after, "veto_exp": veto_exp,
            "keep_frac": float(keep.mean()), "n_vetoed": int((~keep).sum()),
            "exp_lift": float(lift) if np.isfinite(lift) else float("nan")}


def walk_forward_rule_eval(taken: pd.DataFrame, feature: str, direction: str, *, tail_q: float = TAIL_Q,
                           train_min: int = 120, n_trials: int = 1) -> dict:
    """OOS judgement of a single-feature avoid rule with an EXPANDING-window threshold (no fixed-cut peek).

    ``direction`` is the losers' side: ">" (avoid the high tail) or "<" (avoid the low tail). At each trade
    the cut is the ``tail_q`` quantile of that feature over PAST taken trades only; the trade is vetoed if it
    sits in the loss-prone tail. Fully OOS. Returns before/after stats + the vetoed cohort's expectancy.
    """
    d = taken.sort_values("entry_date").reset_index(drop=True)
    v = d[feature].to_numpy(float)
    q = tail_q if direction == ">" else (1 - tail_q)
    vetoed = np.zeros(len(d), dtype=bool)
    scored = np.zeros(len(d), dtype=bool)
    for k in range(len(d)):
        if k < train_min:
            continue
        past = v[:k]
        past = past[np.isfinite(past)]
        if len(past) < 80:
            continue
        cut = float(np.quantile(past, q))
        scored[k] = True
        vetoed[k] = (v[k] > cut) if direction == ">" else (v[k] < cut)
    d = d[scored].copy()
    vt = vetoed[scored]
    if len(d) < 60:
        return {"available": False, "n_scored": int(len(d))}
    keep = ~vt
    before = _stream_stats(d.set_index("entry_date")["ret"].sort_index(), n_trials)
    after = _stream_stats(d.loc[keep].set_index("entry_date")["ret"].sort_index(), n_trials)
    veto_exp = float(d.loc[vt, "ret"].mean()) if vt.any() else float("nan")
    keep_frac = float(keep.mean())
    ok = (after.get("n", 0) >= 40 and keep_frac >= MIN_KEEP
          and np.isfinite(after.get("exp", np.nan)) and np.isfinite(before.get("exp", np.nan))
          and after["exp"] > before["exp"] and after["pf"] > before["pf"]
          and after["dsr"] >= before["dsr"]
          and np.isfinite(veto_exp) and veto_exp < after["exp"])
    return {"available": True, "before": before, "after": after, "veto_exp": veto_exp,
            "keep_frac": keep_frac, "n_vetoed": int(vt.sum()), "promote": bool(ok)}


def walk_forward_avoid_eval(taken: pd.DataFrame, rule: AvoidRule, *, n_trials: int = 1,
                            train_min: int = 120) -> dict:
    """OOS judgement of a candidate AvoidRule via purged walk-forward; returns the pre-registered gate.

    feature_tail rules use an expanding-window threshold (walk_forward_rule_eval). trigger rules veto a
    whole trigger tag (categorical, no threshold) over the same scored window. Promotion requires the rule
    to lift expectancy AND PF AND DSR on the OOS stream and to have removed a worse-than-kept cohort.
    """
    if rule.kind == "feature_tail":
        return walk_forward_rule_eval(taken, rule.feature, rule.op, n_trials=n_trials, train_min=train_min)
    d = taken.sort_values("entry_date").reset_index(drop=True).iloc[train_min:]
    if len(d) < 60:
        return {"available": False, "n_scored": int(len(d))}
    vt = (d["trigger"] == rule.trigger).to_numpy()
    keep = ~vt
    before = _stream_stats(d.set_index("entry_date")["ret"].sort_index(), n_trials)
    after = _stream_stats(d.loc[keep].set_index("entry_date")["ret"].sort_index(), n_trials)
    veto_exp = float(d.loc[vt, "ret"].mean()) if vt.any() else float("nan")
    keep_frac = float(keep.mean())
    ok = (after.get("n", 0) >= 40 and keep_frac >= MIN_KEEP
          and np.isfinite(after.get("exp", np.nan)) and np.isfinite(before.get("exp", np.nan))
          and after["exp"] > before["exp"] and after["pf"] > before["pf"]
          and after["dsr"] >= before["dsr"] and np.isfinite(veto_exp) and veto_exp < after["exp"])
    return {"available": True, "before": before, "after": after, "veto_exp": veto_exp,
            "keep_frac": keep_frac, "n_vetoed": int(vt.sum()), "promote": bool(ok)}


def deploy_threshold(taken: pd.DataFrame, feature: str, op: str, *, tail_q: float = TAIL_Q) -> float:
    """The full-history quantile to LOCK into a promoted feature-tail rule for live deployment."""
    v = taken[feature].to_numpy(float)
    q = tail_q if op == ">" else (1 - tail_q)
    return float(np.nanquantile(v, q))


def loss_veto_model_lift(dev_taken: pd.DataFrame, holdout_taken: pd.DataFrame, feat_cols: list[str],
                         *, n_trials: int, veto_q: float = 0.90, seed: int = 42) -> dict:
    """The ML avoidance layer: a 2nd-stage model trained on DEV taken to predict P(loss), used to veto the
    highest-loss-risk holdout takes (top decile). Judged on the holdout only. This is the model 'learning
    to avoid' beyond the hand-distilled rules — reported alongside, promoted into memory only if it lifts.
    """
    import lightgbm as lgb
    Xd = dev_taken[feat_cols].fillna(0.0)
    yd = (dev_taken["ret"] <= 0).astype(int)              # 1 = loss (what we want to predict & avoid)
    if yd.nunique() < 2 or len(dev_taken) < MIN_DEV_TAKEN:
        return {"available": False}
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=-1)
    m.fit(Xd, yd)
    cut = float(np.quantile(m.predict_proba(Xd)[:, 1], veto_q))      # DEV-derived loss-risk cut (locked)
    ph = m.predict_proba(holdout_taken[feat_cols].fillna(0.0))[:, 1]
    keep = ph < cut
    before = _stream_stats(holdout_taken.set_index("entry_date")["ret"].sort_index(), n_trials)
    after = _stream_stats(holdout_taken.loc[keep].set_index("entry_date")["ret"].sort_index(), n_trials)
    veto_exp = float(holdout_taken.loc[~keep, "ret"].mean()) if (~keep).any() else float("nan")
    lift = (after.get("exp", np.nan) - before.get("exp", np.nan)) if after.get("n", 0) >= 30 else np.nan
    return {"available": True, "before": before, "after": after, "veto_exp": veto_exp,
            "keep_frac": float(keep.mean()), "exp_lift": float(lift) if np.isfinite(lift) else float("nan"),
            "cut": cut}
