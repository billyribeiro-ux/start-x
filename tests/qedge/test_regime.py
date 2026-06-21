"""Tests for the first-class regime layer (:mod:`qedge.modeling.regime`).

The headline test is the NO-LOOKAHEAD CANARY for the HMM: poisoning the future of
the observation matrix must not change any historical *filtered* state. This fails
loudly if anyone swaps the forward (alpha) recursion for hmmlearn's smoothing
(``predict``/``predict_proba``/``score_samples``), which leaks the future — that
is exactly the regression the canary exists to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qedge.config import QedgeConfig

pytest.importorskip("hmmlearn")
pytest.importorskip("ruptures")
pytest.importorskip("sklearn")

from qedge.modeling.regime import (  # noqa: E402  (after importorskip guard)
    ChangePointDetector,
    HMMRegimeDetector,
    RegimeModel,
    VolCorrClusterDetector,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _two_regime_panel(
    rng: np.random.Generator, n_per: int = 200
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a 2-regime feature panel: [return, realized-vol] with distinct means.

    Regime 0 = calm: small positive drift, low vol. Regime 1 = stressed: negative
    drift, high vol. Blocks are concatenated so the true regime path is known.
    """
    calm_ret = rng.normal(0.0008, 0.006, n_per)
    stress_ret = rng.normal(-0.0015, 0.030, n_per)
    returns = np.concatenate([calm_ret, stress_ret, calm_ret])
    truth = np.concatenate(
        [np.zeros(n_per), np.ones(n_per), np.zeros(n_per)]
    ).astype(int)
    # Trailing realized vol as a second feature (causal rolling std).
    s = pd.Series(returns)
    rv = s.rolling(10, min_periods=1).std(ddof=0).to_numpy()
    idx = pd.date_range("2019-01-01", periods=len(returns), freq="B")
    panel = pd.DataFrame({"ret": returns, "rv": rv}, index=idx)
    return panel, truth


def _ambiguous_panel(rng: np.random.Generator, n_per: int = 120) -> pd.DataFrame:
    """A 2-regime panel with *overlapping* means/vols so the latent path is ambiguous.

    The overlap is deliberate: it makes the Viterbi (smoothing) prefix path
    future-sensitive, so the no-lookahead canary catches a ``predict()`` leak as
    well as a ``predict_proba()`` leak. A well-behaved forward filter is invariant.
    """
    calm = rng.normal(0.0, 0.02, n_per)
    stress = rng.normal(0.02, 0.02, n_per)
    returns = np.concatenate([calm, stress, calm])
    idx = pd.date_range("2019-01-01", periods=len(returns), freq="B")
    return pd.DataFrame({"ret": returns}, index=idx)


# --------------------------------------------------------------------------- #
# HMMRegimeDetector — recovery
# --------------------------------------------------------------------------- #
def test_hmm_recovers_two_regimes(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(7)
    panel, truth = _two_regime_panel(rng)
    det = HMMRegimeDetector(n_states=2, config=cfg).fit(panel)
    states = det.predict_pit(panel)

    assert states.index.equals(panel.index)
    # Two distinct states are actually used.
    assert states.nunique() == 2

    # The two recovered states must separate the calm vs stressed blocks: align by
    # majority vote, ignoring an initial warm-up where the filter has little data.
    warm = 30
    aligned = states.to_numpy()[warm:]
    truth_w = truth[warm:]
    # Map each predicted state to the dominant true regime, then score accuracy.
    mapping: dict[int, int] = {}
    for st in np.unique(aligned):
        mask = aligned == st
        mapping[int(st)] = int(np.round(truth_w[mask].mean()))
    mapped = np.array([mapping[int(s)] for s in aligned])
    accuracy = float((mapped == truth_w).mean())
    assert accuracy > 0.8

    # Recovered emission means must differ across states (distinct regimes).
    means = det.model.means_
    assert not np.allclose(means[0], means[1])


def test_hmm_filtered_proba_is_valid_distribution(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(11)
    panel, _ = _two_regime_panel(rng, n_per=120)
    det = HMMRegimeDetector(n_states=2, config=cfg).fit(panel)
    proba = det.filtered_proba(panel)

    assert proba.shape == (len(panel), 2)
    # Every row is a probability distribution.
    np.testing.assert_allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-9)
    assert (proba.to_numpy() >= -1e-12).all()
    # argmax of filtered_proba agrees with predict_pit.
    np.testing.assert_array_equal(
        proba.to_numpy().argmax(axis=1), det.predict_pit(panel).to_numpy()
    )


def test_hmm_deterministic(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(3)
    panel, _ = _two_regime_panel(rng, n_per=100)
    a = HMMRegimeDetector(n_states=2, config=cfg).fit(panel).predict_pit(panel)
    b = HMMRegimeDetector(n_states=2, config=cfg).fit(panel).predict_pit(panel)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# HMMRegimeDetector — NO-LOOKAHEAD CANARY (non-negotiable)
# --------------------------------------------------------------------------- #
def test_hmm_no_lookahead_canary(cfg: QedgeConfig) -> None:
    """Poisoning the future must not change any past filtered state.

    Uses an *ambiguous* (overlapping) 2-regime panel so that smoothing — both
    Viterbi ``predict()`` and forward-backward ``predict_proba()`` — would visibly
    rewrite the prefix when the future changes. A correct forward filter does not.
    """
    rng = np.random.default_rng(7)
    panel = _ambiguous_panel(rng, n_per=120)
    t_cut = len(panel)

    # Fit once on the in-sample prefix; the canary is about INFERENCE causality.
    det = HMMRegimeDetector(n_states=2, config=cfg).fit(panel)
    baseline = det.predict_pit(panel)
    baseline_proba = det.filtered_proba(panel)

    # Append 50 extreme/poisoned future rows (huge magnitude, opposite regime).
    poison = pd.DataFrame(
        {"ret": np.full(50, 50.0)},
        index=pd.date_range(
            panel.index[-1] + pd.tseries.offsets.BDay(1), periods=50, freq="B"
        ),
    )
    extended = pd.concat([panel, poison])

    extended_states = det.predict_pit(extended)
    extended_proba = det.filtered_proba(extended)

    # Every filtered state at t <= T must be UNCHANGED by the poisoned future.
    np.testing.assert_array_equal(
        baseline.to_numpy(),
        extended_states.to_numpy()[:t_cut],
    )
    # And the filtered probabilities must be bit-for-bit identical too.
    np.testing.assert_allclose(
        baseline_proba.to_numpy(),
        extended_proba.to_numpy()[:t_cut],
        atol=0.0,
        rtol=0.0,
    )


# --------------------------------------------------------------------------- #
# ChangePointDetector
# --------------------------------------------------------------------------- #
def test_changepoint_finds_known_break(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(7)
    # Piecewise-constant mean with a single break at index 150.
    left = rng.normal(0.0, 0.5, 150)
    right = rng.normal(8.0, 0.5, 150)
    series = pd.Series(np.concatenate([left, right]))

    det = ChangePointDetector(config=cfg)
    breakpoints = det.detect(series)

    # ruptures convention: last entry is len(series).
    assert breakpoints[-1] == len(series)
    interior = [bp for bp in breakpoints if bp != len(series)]
    assert interior, "no interior breakpoint detected"
    # A detected break should sit close to the true break at 150.
    nearest = min(interior, key=lambda bp: abs(bp - 150))
    assert abs(nearest - 150) <= 10


# --------------------------------------------------------------------------- #
# VolCorrClusterDetector
# --------------------------------------------------------------------------- #
def test_volcorr_separates_high_and_low_vol(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(7)
    n = 250
    low_vol = rng.normal(0.0, 0.004, n)
    high_vol = rng.normal(0.0, 0.040, n)
    returns = pd.Series(
        np.concatenate([low_vol, high_vol]),
        index=pd.date_range("2019-01-01", periods=2 * n, freq="B"),
    )

    det = VolCorrClusterDetector(n_clusters=2, vol_window=10, config=cfg)
    labels = det.fit_predict(returns)

    assert labels.index.equals(returns.index)
    # Focus on rows with a valid label, away from the regime boundary warm-up.
    valid = labels[labels >= 0]
    low_block = valid.iloc[:n].dropna()
    high_block = valid.iloc[n + 30 :]  # skip the transition smear

    # Higher label == higher vol (remapped ascending), so the high-vol block must
    # carry a strictly larger typical label than the low-vol block.
    assert high_block.mode().iloc[0] > low_block.mode().iloc[0]
    # The calm block should be dominated by the lowest cluster (label 0).
    assert low_block.mode().iloc[0] == 0


def test_volcorr_accepts_cross_asset_correlation(cfg: QedgeConfig) -> None:
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    base = rng.normal(0.0, 0.01, n)
    returns = pd.Series(base, index=idx)
    cross = pd.Series(base * 0.5 + rng.normal(0.0, 0.005, n), index=idx)

    det = VolCorrClusterDetector(n_clusters=2, vol_window=10, config=cfg)
    labels = det.fit_predict(returns, cross_returns=cross)
    assert labels.index.equals(idx)
    # Some valid (>= 0) labels are produced once trailing windows fill.
    assert (labels >= 0).any()


# --------------------------------------------------------------------------- #
# RegimeModel bundle
# --------------------------------------------------------------------------- #
def test_regime_model_bundle(cfg: QedgeConfig) -> None:
    model = RegimeModel.from_config(cfg)
    assert isinstance(model.hmm, HMMRegimeDetector)
    assert isinstance(model.changepoint, ChangePointDetector)
    assert isinstance(model.vol_cluster, VolCorrClusterDetector)
    # Detectors share the configured defaults.
    assert model.hmm.n_states == cfg.regime.hmm_n_states
    assert model.changepoint.penalty == cfg.regime.changepoint_penalty
