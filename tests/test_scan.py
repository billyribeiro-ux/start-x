"""Integration test for the P6 scanner output: fit + calibrate + surface explainable setups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.scanner.scan import STRESS, fit, surface
from startx.scanner.signals import FEATURES


def _dataset(n=750, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-03", periods=n)
    syms = ["SPY", "QQQ", "IWM"]
    rows = []
    for i, dt in enumerate(idx):
        sym = syms[i % 3]
        feats = {f: float(rng.normal(0, 1)) for f in FEATURES}
        stress = float(rng.normal(0, 1))
        # a real, learnable signal: high stress + low rsi2 -> more likely to hit PT
        p_hit = 1 / (1 + np.exp(-(0.8 * stress - 0.5 * feats["rsi2"])))
        label = int(rng.random() < p_hit)
        regime = "crisis" if stress > 1 else ("risk_off" if stress > 0.3 else "neutral")
        rows.append({"symbol": sym, "date": dt, "entry_date": dt, "t1": dt, "trigger": "pullback",
                     "label": label, "ret": (0.02 if label else -0.01) + rng.normal(0, 0.005),
                     "regime": regime, "regime_stress": stress, **feats})
    return pd.DataFrame(rows)


def _prices():
    out = {}
    idx = pd.bdate_range("2016-01-03", periods=760)
    for s in ["SPY", "QQQ", "IWM"]:
        c = 100 * np.cumprod(1 + np.random.default_rng(hash(s) % 99).normal(0, 0.01, 760))
        out[s] = pd.DataFrame({"date": idx, "open": c, "high": c * 1.01, "low": c * 0.99,
                               "close": c, "volume": 1e6})
    return out


def test_fit_and_surface_produces_explainable_setups():
    data = _dataset()
    sm = fit(data)
    setups = surface(sm, _prices(), pd.DataFrame(), asof=data["entry_date"].max(), lookback=4000)
    assert setups, "expected some stress-gated setups in synthetic data"
    for s in setups:
        assert s.regime in STRESS                       # gate: only stress regimes surface
        assert 0.0 <= s.conviction <= 1.0               # calibrated probability
        assert s.drivers and len(s.drivers) <= 3        # ranked driver attribution, no bare score
        assert s.direction == "long"
    # sorted by conviction descending
    convs = [s.conviction for s in setups]
    assert convs == sorted(convs, reverse=True)


def test_calibration_reflects_base_rate():
    """Calibrated conviction should track the true hit rate, not the raw model score."""
    data = _dataset()
    sm = fit(data)
    raw = sm.model.predict_proba(data[sm.feat_cols].fillna(0.0))[:, 1]
    cal = sm.iso.predict(raw)
    # calibrated mean ~ empirical base rate (monotone map, not identity)
    assert abs(cal.mean() - data["label"].mean()) < 0.12
