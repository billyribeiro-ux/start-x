"""Regression guards for two audited issues:

A. ``scripts/highconf_eval.py`` must not hardcode an absolute output path; its default ``--out``
   must resolve repo-relative (and stay identical when run from the repo root).
B. ``startx.strategy.market_internals.compute_internals`` enforces a genuine ``dateFirstAdded``
   add-gate (point-in-time on the "joined" edge), AND is honest about the remaining one-sided
   survivorship limitation by emitting a guarded ``RuntimeWarning`` (silenceable via env var).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from startx.strategy import market_internals as mi

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEMBERS_CACHE = _REPO_ROOT / "data" / "cache" / "sp500_members.parquet"


# ---- A: no hardcoded absolute path in the eval script -------------------------------------------
def test_highconf_eval_has_no_hardcoded_abspath():
    src = (_REPO_ROOT / "scripts" / "highconf_eval.py").read_text()
    assert "/home/user/" not in src and "/Users/" not in src, "hardcoded absolute path present"
    assert "parents[1]" in src, "expected repo-relative path resolution from __file__"


# ---- B: the survivorship limitation is flagged, not hidden --------------------------------------
def test_compute_internals_warns_about_survivorship(monkeypatch):
    if not _MEMBERS_CACHE.exists():
        pytest.skip("members cache absent; warning path needs no network but data is required")
    monkeypatch.delenv("STARTX_SILENCE_SURVIVORSHIP", raising=False)
    monkeypatch.chdir(_REPO_ROOT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mi.compute_internals()
    survivorship = [w for w in caught
                    if issubclass(w.category, RuntimeWarning) and "survivorship" in str(w.message)]
    assert len(survivorship) == 1, "expected exactly one survivorship RuntimeWarning"


def test_survivorship_warning_silenceable(monkeypatch):
    if not _MEMBERS_CACHE.exists():
        pytest.skip("members cache absent")
    monkeypatch.setenv("STARTX_SILENCE_SURVIVORSHIP", "1")
    monkeypatch.chdir(_REPO_ROOT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mi.compute_internals()
    assert not [w for w in caught if "survivorship" in str(w.message)], "warning not silenced"


# ---- C: not_breaking_down must NOT block a trade on absent internals (audit Finding) ----------
def test_not_breaking_down_true_on_missing_internals():
    """Missing internals -> True (do not block). NaN must coerce to True BEFORE the comparison;
    `np.nan > x` is False (would wrongly block) and leaves no NaN for a trailing fillna to catch."""
    prices = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    # internals that don't overlap the price dates -> all-NaN after reindex
    internals = pd.DataFrame({"date": pd.bdate_range("2000-01-01", periods=5),
                              "ud_vol": [50.0, 50.0, 50.0, 50.0, 50.0]})
    out = mi.not_breaking_down(internals, prices)
    assert out.dtype == bool and len(out) == 10
    assert out.all(), "absent internals must resolve to True (don't block the trade)"
    # and a genuine breakdown (low up-volume) still blocks
    internals2 = pd.DataFrame({"date": prices["date"], "ud_vol": [5.0] * 10})  # 5% up-vol = breakdown
    assert not mi.not_breaking_down(internals2, prices, ud_vol_min=20.0).any()


def test_datefirstadded_gate_is_effective(monkeypatch):
    """The add-gate must actually exclude pre-membership history (count rises as names join)."""
    if not _MEMBERS_CACHE.exists():
        pytest.skip("members cache absent")
    monkeypatch.setenv("STARTX_SILENCE_SURVIVORSHIP", "1")
    monkeypatch.chdir(_REPO_ROOT)
    I = mi.compute_internals()
    I = I.set_index(pd.to_datetime(I["date"]))
    n = I["n"].dropna()
    assert n.iloc[: len(n) // 4].mean() < n.iloc[-len(n) // 4:].mean(), (
        "effective constituent count should grow over time if the dateFirstAdded gate is active"
    )
