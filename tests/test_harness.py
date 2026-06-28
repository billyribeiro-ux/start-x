"""Tests for the P4 validation harness — the leakage canary is a unit test, per the charter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.scanner.harness import _auc, _pbo_cscv, canary_lookahead
from startx.scanner.signals import FEATURES


def _synthetic_dataset(n=500, seed=0, signal=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    df = pd.DataFrame({"symbol": "X", "date": idx, "entry_date": idx, "t1": idx,
                       "trigger": "t", "ret": rng.normal(0, 0.02, n)})
    for f in FEATURES:
        df[f] = rng.normal(0, 1, n)
    if signal:
        # a real predictive feature: label depends on rsi2
        df["label"] = (df["rsi2"] + rng.normal(0, 0.5, n) > 0).astype(int)
    else:
        df["label"] = rng.integers(0, 2, n)
    return df


def test_auc_extremes():
    assert _auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0       # perfect separation
    assert _auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0       # perfectly wrong
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    assert abs(_auc(y, p) - 0.5) < 0.06                           # random ~ 0.5


def test_canary_lookahead_is_caught():
    """A feature equal to the label must drive AUC to ~1.0 — the harness MUST catch the leak."""
    ds = _synthetic_dataset(signal=False)
    auc = canary_lookahead(ds)
    assert auc >= 0.95, f"lookahead leak not caught (AUC={auc})"


def test_pbo_cscv_in_range_and_detects_overfit():
    """PBO is a probability in [0,1]; pure-noise configs should NOT look reliably skillful."""
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.bdate_range("2016-01-01", periods=n)
    # build an OOS frame where every 'config' (threshold) is just noise -> PBO should be high (~>=0.4)
    oos = pd.DataFrame({"entry_date": idx, "prob": rng.random(n), "ret": rng.normal(0, 0.02, n)})
    pbo = _pbo_cscv(oos, tuple(np.round(np.arange(0.45, 0.701, 0.05), 3)), n_blocks=8)
    assert 0.0 <= pbo <= 1.0
    assert pbo >= 0.35                                            # noise must not look robustly skillful


def test_pbo_low_when_one_config_dominates():
    """If the same config is best in every block, PBO must be LOW (a real, stable preference)."""
    n = 600
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(2)
    prob = rng.random(n)
    # ret rises monotonically with prob -> the highest threshold is always best in every block
    ret = (prob - 0.5) * 0.05 + rng.normal(0, 0.002, n)
    oos = pd.DataFrame({"entry_date": idx, "prob": prob, "ret": ret})
    pbo = _pbo_cscv(oos, tuple(np.round(np.arange(0.45, 0.701, 0.05), 3)), n_blocks=8)
    assert pbo <= 0.2
