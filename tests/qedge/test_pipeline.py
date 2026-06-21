"""End-to-end tests for the KEYSTONE scanner pipeline.

These assert the contract the harness must honour:

* a scan on the deterministic synthetic market returns a fully-populated,
  byte-reproducible :class:`EdgeResult` (same ``run_id`` and metrics across two
  runs);
* the methodology is honest under adversarial pressure — a *planted, fold-able*
  edge clears the DSR/PBO survival gate, while a pure-noise price path of the
  same shape does NOT (``passed is False``);
* unpopulated contract dimensions (options NBBO, market internals) are reported
  truthfully because their feeds raise ``FeedNotAvailable``;
* a shared :class:`TrialLedger` accumulates the multiple-testing count across a
  multi-symbol scan, so ``n_trials`` increments symbol by symbol.

The planted-vs-noise pair is built from crafted price frames fed through a tiny
in-memory :class:`PriceFeed`, so the whole real pipeline (PIT features, filtered
HMM regime, triple-barrier labels, purged CPCV, family ensemble, costs, gate)
runs unmodified. Heavy ML deps are imported lazily via ``importorskip`` so the
numpy-only tiers of the suite still collect on a bare environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qedge.config import QedgeConfig
from qedge.validation import TrialLedger

pytest.importorskip("hmmlearn")
pytest.importorskip("lightgbm")
pytest.importorskip("shap")

from qedge.scanner.pipeline import (  # noqa: E402  (after importorskip gate)
    EdgeResult,
    scan_symbol,
)

# A short, momentum-blocked window keeps the real (heavy) ensemble fits tractable
# while still giving the CPCV enough events for a defined DSR/PBO.
_N_BARS = 240
_BLOCK = 20  # length of each consistent-drift block (> the 10-day short horizon)
_DRIFT = 0.004  # per-bar block drift; large vs the noise so the edge is foldable
_NOISE_SD = 0.003
_SEED = 7
_INITIAL_PRICE = 100.0
_FAST_HMM_ITERS = 6  # trim Baum-Welch iterations: tests assert behaviour, not fit quality


def _fast_config() -> QedgeConfig:
    """A default config with the HMM fit trimmed so the end-to-end scan is quick."""
    cfg = QedgeConfig()
    cfg.regime.hmm_n_iter = _FAST_HMM_ITERS
    return cfg


def _make_prices(n: int, seed: int, *, signal: bool) -> pd.DataFrame:
    """Crafted OHLCV frame: momentum blocks (signal) or pure noise.

    When ``signal`` is True the series is built from contiguous blocks of
    consistent up- or down-drift, each longer than the labeling horizon. The
    trailing return at entry (a registered feature) reveals the block's direction
    and the path continues in it, so the triple-barrier meta-label (profit-taking
    barrier first) is learnable and the taken trades realize consistent positive
    returns. When ``signal`` is False the drift is zero — a directionless random
    walk with no foldable edge.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    if signal:
        block_signs = rng.integers(0, 2, size=n // _BLOCK + 1) * 2 - 1
        drift = _DRIFT * np.repeat(block_signs, _BLOCK)[:n]
    else:
        drift = np.zeros(n, dtype=np.float64)
    rets = drift + rng.normal(0.0, _NOISE_SD, size=n)
    close = _INITIAL_PRICE * np.cumprod(1.0 + rets)
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 5.0e6, dtype=np.float64),
        }
    )


class _FrameFeed:
    """A minimal :class:`~qedge.data.protocols.PriceFeed` over one fixed frame.

    Ignores ``asof``/``start`` and returns the whole crafted history so a test can
    drive the real pipeline with a small, deterministic price path.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def history(
        self,
        symbol: str,
        *,
        asof: pd.Timestamp,
        start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return the crafted frame regardless of the requested window."""
        return self._frame.copy()


def _signal_feed() -> _FrameFeed:
    return _FrameFeed(_make_prices(_N_BARS, _SEED, signal=True))


def _noise_feed() -> _FrameFeed:
    return _FrameFeed(_make_prices(_N_BARS, _SEED, signal=False))


def test_scan_symbol_returns_populated_edge_result() -> None:
    """A scan yields a fully-populated EdgeResult with the locked schema."""
    cfg = _fast_config()
    result = scan_symbol("SIG", feed=_signal_feed(), horizon="short", config=cfg)

    assert isinstance(result, EdgeResult)
    assert result.symbol == "SIG"
    assert result.horizon == "short"
    assert result.n_events > 0
    assert result.n_trials == 1
    # Metrics are real numbers (the gate ran on a defined OOS stream).
    assert np.isfinite(result.oos_sharpe)
    assert np.isfinite(result.deflated_sharpe)
    assert np.isfinite(result.pbo)
    assert isinstance(result.passed, bool)
    assert result.verdict in ("ROBUST", "LIKELY OVERFIT")
    assert result.break_even_aum_usd >= 0.0
    # Provenance is populated.
    assert isinstance(result.top_features, list)
    assert result.top_features  # the planted signal yields a non-empty ranking
    assert all(isinstance(name, str) for name, _ in result.top_features)
    assert len(result.feature_snapshot_hash) == 64  # sha256 hex digest
    assert result.run_id


def test_scan_symbol_is_deterministic() -> None:
    """Two identical scans produce identical run_id and metrics (full reproducibility)."""
    cfg = _fast_config()
    a = scan_symbol("SIG", feed=_signal_feed(), horizon="short", config=cfg)
    b = scan_symbol("SIG", feed=_signal_feed(), horizon="short", config=cfg)

    assert a.run_id == b.run_id
    assert a.feature_snapshot_hash == b.feature_snapshot_hash
    assert a.deflated_sharpe == b.deflated_sharpe
    assert a.pbo == b.pbo
    assert a.oos_sharpe == b.oos_sharpe
    assert a.top_features == b.top_features
    assert a.passed == b.passed


def test_planted_signal_passes_gate_and_finds_feature() -> None:
    """A foldable planted edge clears the survival gate (DSR high, PBO low)."""
    cfg = _fast_config()
    result = scan_symbol("SIG", feed=_signal_feed(), horizon="short", config=cfg)

    assert result.passed is True
    assert result.verdict == "ROBUST"
    assert result.deflated_sharpe >= cfg.validation.dsr_min
    assert result.pbo <= cfg.validation.pbo_max
    assert result.oos_sharpe > 0.0
    # The momentum signal lives in the trailing returns; one of them should rank.
    top_names = {name for name, _ in result.top_features}
    assert top_names & {"ret_1d", "ret_5d", "ret_21d"}


def test_pure_noise_does_not_pass_gate() -> None:
    """A directionless random walk of the same shape must FAIL the gate."""
    cfg = _fast_config()
    result = scan_symbol("NOISE", feed=_noise_feed(), horizon="short", config=cfg)

    assert result.passed is False
    assert result.verdict == "LIKELY OVERFIT"
    # No real edge: the deflated Sharpe never clears the ship threshold.
    assert not (result.deflated_sharpe >= cfg.validation.dsr_min)


def test_unpopulated_dimensions_reported() -> None:
    """Options NBBO and internals have no feed -> reported as unpopulated, honestly."""
    cfg = _fast_config()
    result = scan_symbol("NOISE", feed=_noise_feed(), horizon="short", config=cfg)

    assert "options_nbbo" in result.unpopulated_dimensions
    assert "internals" in result.unpopulated_dimensions


def test_shared_ledger_counts_multiple_symbols() -> None:
    """A shared TrialLedger accumulates trials across symbols (multiple testing)."""
    cfg = _fast_config()
    ledger = TrialLedger()
    feed = _noise_feed()

    first = scan_symbol("AAA", feed=feed, horizon="short", trials=ledger, config=cfg)
    second = scan_symbol("BBB", feed=feed, horizon="short", trials=ledger, config=cfg)

    # Each symbol/horizon is one trial; the count rises 1 -> 2 across the scan.
    assert first.n_trials == 1
    assert second.n_trials == 2
    assert ledger.count == 2
    assert ledger.names == ("AAA:short", "BBB:short")
