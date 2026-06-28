"""Tests for the Layer-1 regime classifier (PIT z-score + rule-based labeling)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.scanner.regime import REGIMES, _zexp, classify


def test_zexp_is_point_in_time():
    """Expanding z-score must not change a past value when future data is appended."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 800), index=pd.bdate_range("2015-01-01", periods=800))
    full = _zexp(s)
    cut = _zexp(s.iloc[:500])
    common = full.index.intersection(cut.index)
    a, b = full.loc[common], cut.loc[common]
    mask = a.notna() & b.notna()
    np.testing.assert_allclose(a[mask].to_numpy(), b[mask].to_numpy(), atol=1e-9)


def _synthetic_panel(n=700, seed=1):
    """Noisy calm baseline (so expanding z-scores are well-defined) with a stress spike at the end."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    base = pd.DataFrame(index=idx)
    stress = np.zeros(n)
    stress[-40:] = 1.0                                             # spike at the end
    nz = lambda s: rng.normal(0, s, n)                            # noqa: E731 (test helper)
    base["ts_9d"] = 0.9 + 0.25 * stress + nz(0.02)                # backwardation in stress
    base["ts_3m"] = 0.9 + 0.25 * stress + nz(0.02)
    base["vvix"] = 90 + 50 * stress + nz(5)
    base["move"] = 80 + 60 * stress + nz(5)
    base["vrp"] = 5 - 12 * stress + nz(1.0)                        # VRP collapses in stress
    base["credit_mom"] = -0.05 * stress + nz(0.005)               # credit rolls over
    base["usd_mom"] = 0.03 * stress + nz(0.003)                   # dollar bid
    base["spy_vs_200"] = 0.05 - 0.20 * stress + nz(0.01)          # breaks the 200DMA
    base["rvol"] = 12 + 40 * stress + nz(1.5)
    base["breadth_200"] = 0.65 - 0.5 * stress + nz(0.03)          # breadth collapses
    return base


def test_classify_labels_are_valid():
    p = classify(_synthetic_panel())
    labels = set(p["regime"].dropna().unique())
    assert labels <= set(REGIMES)
    assert "stress" in p


def test_stress_spike_is_risk_off_or_crisis():
    """The engineered stress window must classify as risk_off/crisis, and calm as risk_on/neutral."""
    p = classify(_synthetic_panel())
    spike = p.iloc[-20:]                                            # deep in the stress window
    calm = p.iloc[300:400]                                          # baseline
    assert (spike["regime"].isin(["risk_off", "crisis"])).mean() > 0.7
    assert (calm["regime"].isin(["risk_on", "neutral"])).mean() > 0.7
    assert spike["stress"].mean() > calm["stress"].mean()
