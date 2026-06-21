"""Self-learning meta-label + loss-autopsy: synthetic, fast, NO network.

The two load-bearing tests are leakage tests, because leakage is the whole risk of this module:

* META LEAKAGE-FILTER CANARY — when one forensic feature cleanly predicts wins, the meta gate
  must lift FILTERED win-rate/expectancy markedly above UNFILTERED and score a high OOS AUC; when
  the labels are SHUFFLED (no learnable signal) the gate must add ~nothing and AUC must fall to
  ~0.5. A real leak would make the shuffled case "work", so this catches it.

* PIT GATE — a crafted case where an early-entered but LATE-resolving trade, if it leaked into
  training, would flip a test prediction. We assert walk_forward_metalabel never trains on any
  trade whose t1 > the test trade's entry_time (the t1<=entry firewall, not a positional split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.learning import (
    apply_meta_filter,
    autopsy_trades,
    run_self_learning,
    walk_forward_metalabel,
    win_loss_signature,
)
from startx.learning.autopsy import forensic_feature_columns
from startx.data.prices import _enrich
from startx.models.train import model_factory


# --------------------------------------------------------------------------- builders
def _synthetic_autopsy(n: int, *, edge: float, shuffle: bool, seed: int) -> pd.DataFrame:
    """Trade ledger already shaped like ``autopsy_trades`` output.

    ``signal`` (named ``rvol_20``) drives the win probability through a logistic with slope
    ``edge``; the realised net return is sign-consistent with the label. With ``shuffle`` the
    labels are permuted so no feature carries information (the control arm of the canary).
    Same-day resolution (t1 == exit == entry+1) keeps the PIT firewall trivially satisfied here.
    """
    rng = np.random.default_rng(seed)
    entry = pd.bdate_range("2021-01-04", periods=n)
    signal = rng.normal(0.0, 1.0, n)
    noise1 = rng.normal(0.0, 1.0, n)
    noise2 = rng.normal(0.0, 1.0, n)
    prob = 1.0 / (1.0 + np.exp(-edge * signal))
    is_win = (rng.uniform(size=n) < prob).astype(int)
    if shuffle:
        is_win = rng.permutation(is_win)
    ret = np.where(is_win == 1, rng.uniform(0.005, 0.05, n), -rng.uniform(0.005, 0.05, n))
    return pd.DataFrame({
        "symbol": "SYN",
        "entry_date": entry,
        "exit_date": entry + pd.Timedelta(days=1),
        "t1": entry + pd.Timedelta(days=1),
        "ret_net": ret,
        "rvol_20": signal,
        "noise_a": noise1,
        "noise_b": noise2,
        "is_win": is_win,
    })


def _price_series(base=100.0, n=80, step=1.0, vol=1_000_000.0):
    """Flat-volume ramp price frame, enriched as ``get_prices`` would (log_ret/gap/prev_close).

    The enrichment matters: ``characterize_reversal`` calls ``compute_signals`` to derive
    ``ar_z``, which needs ``log_ret`` — exactly what real ``get_prices`` output carries.
    """
    close = base + step * np.arange(n, dtype=float)
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": vol,
    })
    return _enrich(df)


# --------------------------------------------------------------------------- autopsy
def test_autopsy_attaches_rvol_and_atr_on_synthetic_prices():
    """autopsy_trades must attach the correct entry-day rvol_20 / atr_14 from the price frame."""
    prices = _price_series(n=80, vol=1_000_000.0)
    entry_date = prices["date"].iloc[60]
    trades = pd.DataFrame({
        "symbol": ["SYN"],
        "entry_date": [entry_date],
        "ret_net": [0.03],
    })
    out = autopsy_trades(trades, {"SYN": prices}, intraday=False)

    assert "is_win" in out.columns and int(out["is_win"].iloc[0]) == 1
    for col in ("rvol_20", "atr_14", "atr_expansion", "gap_pct", "close_location_value", "ar_z"):
        assert col in out.columns, f"missing forensic column {col}"
    # rvol_20 == entry-day volume / trailing-20 mean volume; volume is constant => exactly 1.0.
    assert np.isclose(out["rvol_20"].iloc[0], 1.0, rtol=1e-9)
    # atr_14 must be finite and positive once ATR has warmed up.
    assert np.isfinite(out["atr_14"].iloc[0]) and out["atr_14"].iloc[0] > 0
    # A candle one-hot must exist.
    assert any(c.startswith("candle_") for c in out.columns)


def test_autopsy_robust_to_missing_prices():
    """A trade with no price frame gets NaN forensics, not an exception."""
    trades = pd.DataFrame({"symbol": ["NOPE"], "entry_date": [pd.Timestamp("2022-06-01")],
                           "ret_net": [-0.02]})
    out = autopsy_trades(trades, {}, intraday=False)
    assert int(out["is_win"].iloc[0]) == 0
    assert np.isnan(out["rvol_20"].iloc[0])


def test_forensic_features_exclude_outcome_and_level_columns():
    """Allow-list guard: strategy outcome/price-level columns must NEVER be eligible features.

    A deny-list let ``ret``/``exit_price``/``entry_price``/``target``/``stop`` leak into the
    meta-model (OOS AUC -> 1.0, fake "100% win"). Genuine pre-entry forensics + the carried signal
    must remain eligible.
    """
    df = pd.DataFrame({
        "symbol": ["SPY"], "entry_date": [pd.Timestamp("2025-01-01")],
        "exit_date": [pd.Timestamp("2025-01-03")], "t1": [pd.Timestamp("2025-01-03")],
        "is_win": [1], "ret_net": [0.01], "ret": [0.01], "entry_price": [600.0],
        "target": [610.0], "stop": [590.0], "exit_price": [610.0], "pnl_per_share": [10.0],
        "rvol_20": [1.2], "atr_expansion": [1.3], "gap_pct": [0.002], "ar_z": [-1.1],
        "ibs_entry": [0.03], "candle_doji": [1],
    })
    feats = set(forensic_feature_columns(df))
    assert not feats & {"ret", "entry_price", "target", "stop", "exit_price", "pnl_per_share"}
    assert {"rvol_20", "atr_expansion", "gap_pct", "ar_z", "ibs_entry", "candle_doji"} <= feats


def test_forensic_features_reject_numeric_nonbinary_candle_column():
    """Allow-list guard (defect #3): a numeric ``candle_*`` column that is NOT a genuine 0/1
    one-hot must be REJECTED — the ``candle_`` prefix is not a blank cheque.

    Pre-fix, any numeric ``candle_*`` column passed, so a continuous ``candle_LEAKED_RETURN``
    (or a price level wearing the prefix) leaked into the meta-model. Real one-hots — a known
    archetype name OR strictly 0/1 values — must still be eligible.
    """
    df = pd.DataFrame({
        "symbol": ["SPY", "SPY"], "entry_date": [pd.Timestamp("2025-01-01")] * 2,
        "is_win": [1, 0], "ret_net": [0.01, -0.01],
        "rvol_20": [1.2, 0.8],
        "candle_doji": [1, 0],                        # known-archetype one-hot -> kept
        "candle_custom_flag": [0, 1],                 # unknown name but strictly 0/1 -> kept
        "candle_LEAKED_RETURN": [0.0734, -0.0512],    # continuous leak wearing the prefix -> reject
        "candle_LEAKED_PRICE": [600.5, 590.2],        # price level wearing the prefix -> reject
    })
    feats = set(forensic_feature_columns(df))
    assert not feats & {"candle_LEAKED_RETURN", "candle_LEAKED_PRICE"}, (
        "a numeric, non-0/1 candle_* column leaked through the allow-list")
    assert {"candle_doji", "candle_custom_flag", "rvol_20"} <= feats


def test_win_loss_signature_ranks_separating_feature_first():
    """The feature that actually separates wins from losses must rank first by |AUC-0.5|."""
    autopsy = _synthetic_autopsy(400, edge=2.0, shuffle=False, seed=1)
    sig = win_loss_signature(autopsy)
    assert not sig.empty
    assert list(sig.columns) == [
        "feature", "mean_at_win", "mean_at_loss", "single_feature_auc", "lift"]
    assert sig.iloc[0]["feature"] == "rvol_20", "the separating feature must rank first"
    # It must separate clearly (AUC far from 0.5); noise features must be near 0.5.
    assert abs(sig.iloc[0]["single_feature_auc"] - 0.5) > 0.2
    noise = sig[sig["feature"].isin(["noise_a", "noise_b"])]
    assert (noise["single_feature_auc"].sub(0.5).abs() < 0.15).all()


# --------------------------------------------------------------------------- canary
def test_meta_leakage_filter_canary():
    """FILTERED >> UNFILTERED with a learnable signal; ~equal & AUC~0.5 when labels shuffled."""
    feature_cols = ["rvol_20", "noise_a", "noise_b"]

    # --- learnable: one forensic feature cleanly predicts is_win -----------------------------
    signal = _synthetic_autopsy(360, edge=2.2, shuffle=False, seed=7)
    res = run_self_learning(signal, feature_cols, threshold=0.5, train_min=60, test_span=20)
    assert res.unfiltered["n_trades"] > 0 and res.filtered["n_trades"] > 0
    # The meta-label must discriminate winners out of sample.
    assert res.meta_auc > 0.7, f"expected high meta AUC, got {res.meta_auc:.3f}"
    # Filtering away predicted losers must lift win-rate and expectancy markedly.
    assert res.filtered["win_rate"] > res.unfiltered["win_rate"] + 0.10
    assert res.filtered["expectancy"] > res.unfiltered["expectancy"]

    # --- control: shuffle is_win so nothing is learnable -------------------------------------
    shuffled = _synthetic_autopsy(360, edge=2.2, shuffle=True, seed=7)
    ctrl = run_self_learning(shuffled, feature_cols, threshold=0.5, train_min=60, test_span=20)
    # No signal => AUC ~ 0.5 and filtering adds essentially nothing (could even hurt slightly).
    assert abs(ctrl.meta_auc - 0.5) < 0.12, f"shuffled meta AUC should be ~0.5, got {ctrl.meta_auc:.3f}"
    assert abs(ctrl.filtered["win_rate"] - ctrl.unfiltered["win_rate"]) < 0.10
    # And the signal arm's edge must dominate the control arm's.
    signal_gain = res.filtered["win_rate"] - res.unfiltered["win_rate"]
    ctrl_gain = ctrl.filtered["win_rate"] - ctrl.unfiltered["win_rate"]
    assert signal_gain > ctrl_gain + 0.08


# --------------------------------------------------------------------------- PIT gate
class _SpyModel:
    """A deterministic 1-feature threshold classifier that RECORDS every training set it sees.

    ``fit`` records the exact feature values it was trained on (into the shared ``seen`` list)
    and learns the majority-vote threshold rule ``predict(x) = 1 iff x >= midpoint``. Because it
    records what it trains on, we can assert *directly* which rows the firewall fed it — the most
    rigorous possible leakage check (it inspects the training set, not a noisy model output).
    """

    def __init__(self, seen: list[np.ndarray]):
        self._seen = seen
        self._ones_mean = 0.5

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        self._seen.append(Xa.copy())
        # Threshold = midpoint between the mean feature of wins and the mean feature of losses.
        wins, losses = Xa[ya > 0], Xa[ya <= 0]
        mw = wins.mean() if wins.size else 1.0
        ml = losses.mean() if losses.size else 0.0
        self._thr = (mw + ml) / 2.0
        self._sign = 1.0 if mw >= ml else -1.0  # which side is "win"
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=float).ravel()
        side = (Xa - self._thr) * self._sign
        return (side >= 0).astype(int)

    def predict_proba(self, X):
        p1 = self.predict(X).astype(float)
        return np.column_stack([1.0 - p1, p1])

    @property
    def classes_(self):
        return np.array([0, 1])


def test_pit_gate_excludes_late_resolving_training_trades():
    """The firewall must gate training on ``t1 <= entry``, NOT on positional order.

    Construction: many clean trades (win iff x>0.5) that resolve the NEXT DAY (fast). One POISON
    trade is entered FIRST (earliest entry) but resolves a YEAR LATER — after every test trade's
    entry — and carries the OPPOSITE label rule (a high-x loss). A positional ("first k rows")
    split would put the poison, sitting at position 0, into every later block's training set and
    corrupt the threshold. The t1<=entry firewall must EXCLUDE it from every block because, at
    each test trade's entry, the poison's outcome was still unknown.

    We assert this two ways: (1) directly — the spy records every training set, and the poison's
    feature value must never appear in any of them; (2) behaviourally — predictions follow the
    clean x>0.5 rule, which only holds if the opposite-label poison never trained.
    """
    rng = np.random.default_rng(3)
    n_train = 120
    x = rng.uniform(0.0, 1.0, n_train)
    y = (x > 0.5).astype(int)
    entry = pd.bdate_range("2021-01-04", periods=n_train)
    t1 = entry + pd.Timedelta(days=1)  # fast resolution -> eligible for later blocks

    # POISON: entered first, resolves a year out (after everything), with a high x but label 0
    # (the OPPOSITE of the x>0.5 rule). A unique sentinel feature value makes it traceable.
    poison_x = 0.97531
    poison_entry = entry[0]
    poison_t1 = entry[-1] + pd.Timedelta(days=365)
    poison_y = 0

    X = pd.DataFrame({"x": np.concatenate([[poison_x], x])})
    yy = pd.Series(np.concatenate([[poison_y], y]))
    entries = pd.Series(np.concatenate([[poison_entry], entry.to_numpy()]))
    t1s = pd.Series(np.concatenate([[poison_t1], t1.to_numpy()]))

    seen_training_sets: list[np.ndarray] = []
    factory = lambda: _SpyModel(seen_training_sets)  # noqa: E731 — tiny spy factory

    preds = walk_forward_metalabel(
        X, yy, entries, t1s, factory, train_min=20, test_span=10,
    )
    assert not preds.empty
    assert seen_training_sets, "the firewall must have trained at least one block"

    # (1) DIRECT firewall proof: the poison's feature value must appear in NO training set,
    # because its t1 (a year out) exceeds every test block's entry time.
    for train_x in seen_training_sets:
        assert not np.any(np.isclose(train_x, poison_x)), (
            "poison (entered first but resolving a year later) leaked into a training set — "
            "the gate is positional, not t1<=entry")

    # (2) BEHAVIOURAL proof: every scored trade follows the clean x>0.5 rule. If the poison had
    # trained, the high-x loss would have pulled the threshold and flipped predictions near it.
    x_lookup = X["x"]
    for idx, prow in preds.iterrows():
        xv = float(x_lookup.loc[idx])
        expected = 1 if xv > 0.5 else 0
        assert int(prow["y_pred"]) == expected, (
            f"prediction at x={xv:.3f} was {int(prow['y_pred'])}, expected {expected} — "
            "opposite-label late-t1 poison appears to have leaked into training")

    # (3) Contrast: a NAIVE positional split (train on all earlier-by-position rows) WOULD have
    # included the poison at position 0 — confirming the test would catch a positional leak.
    order = entries.sort_values(kind="mergesort").index
    poison_pos = list(order).index(0)  # original index 0 == poison
    assert poison_pos == 0, "poison is positionally first, so a positional split would train on it"


# --------------------------------------------------------------------------- filter helper
def test_apply_meta_filter_threshold_and_metrics():
    """apply_meta_filter keeps only prob>=threshold; trade_metrics computes the book economics."""
    from startx.learning import trade_metrics

    trades = pd.DataFrame({"ret_net": [0.02, -0.01, 0.03, -0.04]}, index=[10, 11, 12, 13])
    probs = pd.Series([0.9, 0.2, 0.8, 0.1], index=[10, 11, 12, 13])
    kept = apply_meta_filter(trades, probs, threshold=0.5)
    assert list(kept.index) == [10, 12]
    m = trade_metrics(kept["ret_net"])
    assert m["n_trades"] == 2 and m["win_rate"] == 1.0
