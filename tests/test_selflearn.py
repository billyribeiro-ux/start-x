"""Tests for the self-learning loss-attribution layer.

These pin the property the whole "learn from the losers" claim depends on: an avoid rule may be promoted
ONLY if it removes a cohort that is genuinely worse out-of-sample. A rule whose vetoed cohort is actually
PROFITABLE (the real R5 finding — the loss-prone cohort has the fattest right tail) MUST be rejected, even
though it has a higher in-sample loss *rate*. Deterministic, offline, no FMP/price cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.scanner.selflearn import (
    AvoidRule,
    ScannerMemory,
    apply_avoids,
    candidate_avoid_rules,
    loss_postmortem,
    walk_forward_rule_eval,
)


def _book(n=400, seed=0):
    """Synthetic taken book where the HIGH-`bad` tail loses more OFTEN but pays MORE when it wins
    (fat right tail) — so vetoing it lowers expectancy. The firewall must refuse to promote that veto."""
    rng = np.random.default_rng(seed)
    bad = rng.uniform(0, 1, n)
    dates = pd.bdate_range("2017-01-01", periods=n)
    ret = np.empty(n)
    for i, b in enumerate(bad):
        if b > 0.75:                          # the "loss-prone" tail: 60% lose, but winners are huge
            ret[i] = -0.02 if rng.uniform() < 0.6 else +0.12
        else:                                 # the body: 45% lose, modest winners
            ret[i] = -0.02 if rng.uniform() < 0.45 else +0.03
    return pd.DataFrame({"entry_date": dates, "t1": dates + pd.Timedelta(days=5),
                         "ret": ret, "bad": bad, "trigger": "mean_rev", "regime": "risk_off"})


def test_postmortem_separates_losers():
    pm = loss_postmortem(_book(), ["bad"])
    assert 0.0 < pm["overall_loss_rate"] < 1.0
    row = pm["separation"].iloc[0]
    assert row["feature"] == "bad"
    assert row["effect"] > 0          # losers sit at HIGHER `bad` (the high tail loses more often)


def test_firewall_rejects_profitable_vetoed_cohort():
    """The core guarantee: a veto whose removed cohort is net-profitable does NOT promote."""
    book = _book()
    ev = walk_forward_rule_eval(book, "bad", ">", train_min=120, n_trials=10)
    assert ev["available"]
    # the high-`bad` tail is net positive (fat right tail), so vetoing it must be rejected
    assert ev["veto_exp"] > 0
    assert ev["promote"] is False


def test_candidate_rules_are_deterministic_and_keyed():
    book = _book()
    feat = ["bad"]
    c1 = candidate_avoid_rules(book, feat)
    c2 = candidate_avoid_rules(book, feat)
    assert [r.key() for r in c1] == [r.key() for r in c2]      # deterministic ordering
    assert all(":" in r.key() for r in c1)


def test_apply_avoids_and_mask():
    book = _book()
    rule = AvoidRule(kind="feature_tail", feature="bad", op=">", threshold=0.75)
    keep = apply_avoids(book, [rule])
    assert keep.sum() < len(book)                      # some rows vetoed
    assert not book.loc[keep, "bad"].gt(0.75).any()    # nothing kept above the cut
    trig = AvoidRule(kind="trigger", feature="", op="==", threshold=float("nan"), trigger="mean_rev")
    assert apply_avoids(book, [trig]).sum() == 0       # vetoes the whole (single-trigger) book


def test_memory_roundtrip(tmp_path):
    p = tmp_path / "mem.json"
    m = ScannerMemory(iteration=2, tested=["feature_tail:bad:>"])
    m.promoted.append(AvoidRule(kind="feature_tail", feature="bad", op=">", threshold=0.7).__dict__)
    m.save(p)
    loaded = ScannerMemory.load(p)
    assert loaded.iteration == 2
    assert loaded.tested == ["feature_tail:bad:>"]
    rules = loaded.promoted_rules()
    assert len(rules) == 1 and rules[0].feature == "bad" and rules[0].op == ">"
