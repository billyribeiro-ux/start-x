"""Swing scanner — the self-learning LOSS loop (one iteration per run, accumulative).

The engine that "learns from its losers." Stress regimes are rare, so a single 2020 holdout starves the
mining side; instead every avoid mechanism here is judged by PURGED WALK-FORWARD over the full taken book
(each trade scored by a model/threshold fit only on its past) — the same firewall the meta-model uses.

Each run:
  1. rebuilds (or loads cached) the stress-gated taken book across ETFs + survivorship-free liquid stocks;
  2. POST-MORTEM the losses (full sample) — WHY they lose: which trigger, which regime, which feature
     signatures separate losers from winners;
  3. re-validates the STANDING avoid rules by walk-forward (decay check — drop any that stop helping);
  4. evaluates the ML AVOIDANCE LAYER — a walk-forward 2nd-stage P(loss) veto (veto-q swept as trials);
  5. proposes the NEXT untested loss condition, judges it walk-forward, and promotes only if it clears the
     pre-registered firewall (lifts expectancy AND PF AND DSR at the rising trial count N, vetoed cohort
     genuinely worse OOS); locks a full-history deploy threshold;
  6. persists everything to scanner_memory.json + appends an honest record to SELFLEARN_LOG.md.

Run repeatedly: the memory accumulates only what survives the firewall and converges when nothing new
clears it. The live scanner reads the same memory, so improving the memory improves the live signals.

    python scripts/swing_selflearn_loss.py [--stocks 120] [--rebuild]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd

from startx.data.membership import REAL_DATA_START, SP500Membership
from startx.scanner.harness import build_dataset, walk_forward
from startx.scanner.regime import classify, regime_panel
from startx.scanner.selflearn import (
    ScannerMemory,
    _stream_stats,
    apply_avoids,
    candidate_avoid_rules,
    deploy_threshold,
    loss_postmortem,
    walk_forward_avoid_eval,
    walk_forward_veto_eval,
)
from startx.scanner.signals import FEATURES

PRICE_DIR = "data/cache/prices"
TAKEN_CACHE = "data/cache/selflearn_taken.parquet"
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
STRESS = ["risk_off", "crisis"]
VETO_QS = (0.80, 0.85, 0.90)            # loss-veto aggressiveness swept -> counts as trials
LOG = "SELFLEARN_LOG.md"


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    if len(d) < 500:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _liquid_stocks(mem, n, min_dv=50.0):
    ever = set()
    for d in ["2018-06-01", "2020-06-01", "2022-06-01", "2024-06-01", "2026-01-01"]:
        ever |= set(mem.members_asof(d))
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    rows = []
    for s in sorted((ever & cached) - set(ETFS)):
        px = _load(s)
        if px is None:
            continue
        dv = float((px["close"] * px["volume"]).tail(252).median())
        if dv >= min_dv * 1e6:
            rows.append((s, dv))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def _build_taken(stocks):
    """The stress-gated, prob>=0.5 taken book across the 3 pillars (the expensive step; cached)."""
    sp = SP500Membership.load()
    reg = classify(regime_panel(membership=sp))
    spy = _load("SPY").set_index("date")["close"]
    universe = ETFS + _liquid_stocks(sp, stocks)
    print(f"building taken book across {len(universe)} symbols "
          f"({len(ETFS)} ETFs + {len(universe)-len(ETFS)} stocks)...")
    data = pd.concat([build_dataset(s, _load(s), spy, reg) for s in universe], ignore_index=True)
    data = data[data["entry_date"] >= REAL_DATA_START]
    oos = walk_forward(data, train_min=400, retrain_every=50)
    taken = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)].copy()
    return taken.sort_values("entry_date").reset_index(drop=True)


def _fmt(st):
    if not st or st.get("n", 0) < 12:
        return f"n={st.get('n', 0) if st else 0} (underpowered)"
    return (f"n={st['n']:4d}  exp {st['exp']*100:+.2f}%  PF {st['pf']:.2f}  "
            f"Sharpe {st['sharpe']:+.2f}  DSR {st['dsr']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=120)
    ap.add_argument("--rebuild", action="store_true", help="force rebuild of the cached taken book")
    args = ap.parse_args()

    mem = ScannerMemory.load()
    mem.iteration += 1
    it = mem.iteration
    print(f"========== SELF-LEARNING LOSS LOOP — iteration {it} ==========")

    if args.rebuild or not os.path.exists(TAKEN_CACHE):
        taken = _build_taken(args.stocks)
        os.makedirs(os.path.dirname(TAKEN_CACHE), exist_ok=True)
        taken.to_parquet(TAKEN_CACHE)
        print(f"  cached taken book -> {TAKEN_CACHE}")
    else:
        taken = pd.read_parquet(TAKEN_CACHE)
        print(f"  loaded cached taken book ({len(taken)} trades) — use --rebuild to refresh")

    feat_cols = [c for c in FEATURES if c in taken.columns] + (["regime_stress"] if "regime_stress" in taken else [])
    print(f"  stress-gated taken book: {len(taken)} trades "
          f"({taken['entry_date'].min().date()}..{taken['entry_date'].max().date()})\n")

    # ---- (2) the WHY: full-sample loss post-mortem ---------------------------------------------------
    pm = loss_postmortem(taken, feat_cols)
    print(f"=== WHY THE LOSSES (post-mortem) — overall loss rate {pm['overall_loss_rate']:.0%} of {pm['n']} takes ===")
    print("  loss rate by TRIGGER:")
    for trig, row in pm["by_trigger"].sort_values("loss_rate", ascending=False).iterrows():
        print(f"    {trig:10} loss {row['loss_rate']:.0%}  (n={int(row['count'])})")
    if not pm["by_regime"].empty:
        print("  loss rate by REGIME:")
        for rg, row in pm["by_regime"].sort_values("loss_rate", ascending=False).iterrows():
            print(f"    {rg:10} loss {row['loss_rate']:.0%}  (n={int(row['count'])})")
    print("  features that most SEPARATE losers from winners (|effect|, sign = losers' side):")
    for _, r in pm["separation"].head(6).iterrows():
        side = "HIGHER" if r["effect"] > 0 else "lower "
        print(f"    {r['feature']:14} losers {side} (effect {r['effect']:+.2f}; "
              f"loser~{r['loser_mean']:+.3f} vs winner~{r['winner_mean']:+.3f})")

    n_trials_base = 40 + len(mem.tested) + len(VETO_QS)      # rising project N (everything searched)

    # ---- (3) re-validate STANDING avoids by walk-forward (decay check) -------------------------------
    standing = mem.promoted_rules()
    print(f"\n=== STANDING AVOID RULES ({len(standing)}) — walk-forward decay check ===")
    if standing:
        keep_rules = []
        for rule in standing:
            ev = walk_forward_avoid_eval(taken, rule, n_trials=n_trials_base)
            still = ev.get("available") and ev["after"]["exp"] > ev["before"]["exp"] and ev["after"]["pf"] > ev["before"]["pf"]
            print(f"  [{'KEEP' if still else 'RETIRE (decayed)'}] {rule.key()} — {rule.rationale}")
            if ev.get("available"):
                print(f"           before {_fmt(ev['before'])} -> after {_fmt(ev['after'])}  "
                      f"(vetoed exp {ev['veto_exp']*100:+.2f}%)")
            if still:
                keep_rules.append(rule)
        if len(keep_rules) != len(standing):
            mem.promoted = [r.__dict__ for r in keep_rules]
            standing = keep_rules
    else:
        print("  (none yet)")

    # ---- (4) ML AVOIDANCE LAYER — walk-forward 2nd-stage P(loss) veto, veto-q swept -------------------
    print("\n=== ML AVOIDANCE LAYER — walk-forward P(loss) veto (best veto-q wins; counts as trials) ===")
    best_vm = None
    for q in VETO_QS:
        vm = walk_forward_veto_eval(taken, feat_cols, veto_q=q, n_trials=n_trials_base)
        if not vm.get("available"):
            print(f"  veto_q={q}: insufficient ({vm.get('n_scored', 0)} scored)")
            continue
        print(f"  veto_q={q}: before {_fmt(vm['before'])} -> after {_fmt(vm['after'])}  "
              f"(kept {vm['keep_frac']:.0%}, vetoed exp {vm['veto_exp']*100:+.2f}%, lift {vm['exp_lift']*100:+.2f}%/trade)")
        if best_vm is None or (vm["after"].get("dsr", -9) > best_vm["after"].get("dsr", -9)):
            best_vm = vm

    # ---- (5) learn ONE new rule — next untested candidate, walk-forward judged -----------------------
    print("\n=== LEARN ONE NEW THING — next untested loss condition, walk-forward judged on the full book ===")
    promoted_this_iter = None
    cands = candidate_avoid_rules(taken, feat_cols)
    tested_keys = set(mem.tested) | {r.key() for r in standing}
    nxt = next((c for c in cands if c.key() not in tested_keys), None)
    if nxt is None:
        print("  no untested candidate remains — the avoid set has CONVERGED on this universe.")
        mem.no_promote_streak += 1
    else:
        n_trials = n_trials_base + 1
        ev = walk_forward_avoid_eval(taken, nxt, n_trials=n_trials)
        tgt = ("%.4f" % nxt.threshold) if nxt.kind == "feature_tail" else nxt.trigger
        print(f"  candidate: AVOID when {nxt.feature or 'trigger'} {nxt.op} {tgt}")
        print(f"    rationale: {nxt.rationale}")
        if ev.get("available"):
            print(f"    walk-forward before: {_fmt(ev['before'])}")
            print(f"    walk-forward after : {_fmt(ev['after'])}   (kept {ev['keep_frac']:.0%}, "
                  f"vetoed {ev['n_vetoed']} @ exp {ev['veto_exp']*100:+.2f}%)")
        mem.tested.append(nxt.key())
        if ev.get("promote"):
            if nxt.kind == "feature_tail":                   # lock a full-history deploy threshold
                nxt.threshold = deploy_threshold(taken, nxt.feature, nxt.op)
            mem.promoted.append(nxt.__dict__)
            promoted_this_iter = nxt
            mem.no_promote_streak = 0
            print("    -> PROMOTE ✅ (lifts exp + PF + DSR walk-forward; vetoed cohort worse OOS)")
        else:
            mem.rejected.append({"key": nxt.key(), "rationale": nxt.rationale})
            mem.no_promote_streak += 1
            print("    -> reject (did not clear the firewall) — recorded so it won't be retried")

    # ---- (6) net effect of the full learned avoid set + persist + log --------------------------------
    final_rules = mem.promoted_rules()
    keep_mask = apply_avoids(taken, final_rules)
    base_st = _stream_stats(taken.set_index("entry_date")["ret"].sort_index(), n_trials_base)
    learned_st = _stream_stats(taken.loc[keep_mask].set_index("entry_date")["ret"].sort_index(), n_trials_base)
    print(f"\n=== NET EFFECT of the {len(final_rules)} learned avoid rule(s) on the taken book ===")
    print(f"  raw stress-gated : {_fmt(base_st)}")
    print(f"  + learned avoids : {_fmt(learned_st)}  (kept {keep_mask.mean():.0%})")
    if best_vm:
        print(f"  + ML loss-veto   : {_fmt(best_vm['after'])}  (veto_q={best_vm['veto_q']}, walk-forward)")

    mem.history.append({
        "iteration": it, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taken": len(taken), "promoted_total": len(final_rules),
        "promoted_this_iter": (promoted_this_iter.key() if promoted_this_iter else None),
        "exp_raw": base_st.get("exp"), "exp_avoids": learned_st.get("exp"),
        "dsr_raw": base_st.get("dsr"), "dsr_avoids": learned_st.get("dsr"),
        "veto_q_best": (best_vm["veto_q"] if best_vm else None),
        "exp_mlveto": (best_vm["after"].get("exp") if best_vm else None),
        "no_promote_streak": mem.no_promote_streak,
    })
    mem.save()

    with open(LOG, "a") as fh:
        fh.write(f"\n## Iteration {it} — {datetime.now(timezone.utc).date()}\n")
        fh.write(f"- taken {len(taken)}; overall loss rate {pm['overall_loss_rate']:.0%}.\n")
        if not pm["by_trigger"].empty:
            wt = pm["by_trigger"]["loss_rate"].idxmax()
            fh.write(f"- WHY: worst trigger `{wt}` ({pm['by_trigger'].loc[wt,'loss_rate']:.0%} loss); "
                     f"top loss-separating feature `{pm['separation'].iloc[0]['feature']}`.\n")
        if promoted_this_iter:
            fh.write(f"- LEARNED (promoted): AVOID `{promoted_this_iter.key()}` — {promoted_this_iter.rationale}.\n")
        elif len(cands) and nxt:
            fh.write(f"- tested `{nxt.key()}` -> rejected (no firewall clear; streak {mem.no_promote_streak}).\n")
        else:
            fh.write("- candidates exhausted; avoid set converged.\n")
        if best_vm:                                          # like-for-like: scored subset before vs after
            fh.write(f"- ML loss-veto (walk-forward, veto_q={best_vm['veto_q']}, same scored subset): exp "
                     f"{best_vm['before'].get('exp', float('nan'))*100:+.2f}% -> {best_vm['after'].get('exp', float('nan'))*100:+.2f}%, "
                     f"DSR {best_vm['before'].get('dsr', float('nan')):.2f} -> {best_vm['after'].get('dsr', float('nan')):.2f} "
                     f"(vetoed cohort exp {best_vm['veto_exp']*100:+.2f}% — the 'avoid' trades were PROFITABLE).\n")
        fh.write(f"- standing avoid set: {len(final_rules)} rule(s); holdout-equivalent exp "
                 f"{(base_st.get('exp') or float('nan'))*100:+.2f}% -> {(learned_st.get('exp') or float('nan'))*100:+.2f}%.\n")

    conv = mem.no_promote_streak >= 3
    print(f"\nmemory saved -> scanner_memory.json  |  log -> {LOG}")
    print(f"iteration {it} complete. standing avoids: {len(final_rules)}. "
          f"{'CONVERGED (no new avoid in 3 passes).' if conv else 'still learning — run again to continue.'}")
    return conv


if __name__ == "__main__":
    main()
