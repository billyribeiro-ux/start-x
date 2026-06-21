"""End-to-end integration + provenance proof for the qedge scanner.

This module exercises the scanner the way a desk operator does: it drives the
*whole* committed stack through :func:`~qedge.scanner.pipeline.run_scan` over a
single-symbol universe backed by the deterministic
:class:`~qedge.data.synthetic.SyntheticMarket`, then proves the resulting verdict
round-trips through the reproducibility / audit-trail layer
(:mod:`qedge.repro`) without losing its identity or key metrics.

Three guarantees are asserted:

* **Integration.** One synthetic ``run_scan`` returns a populated list of
  :class:`~qedge.scanner.pipeline.EdgeResult`, each carrying a 64-hex
  ``feature_snapshot_hash`` (a sha256 data-snapshot digest) and a non-empty
  ``run_id`` (the provenance stamp).
* **Persistence round-trip.** An :class:`EdgeResult`'s fields rebuild a
  :class:`~qedge.repro.RunRecord` via :func:`~qedge.repro.make_run_record`; once
  written to disk and read back as JSON, the ``run_id`` and the key metrics
  survive byte-for-byte — the audit trail is reproducible.
* **Live FMP smoke (network-gated).** With an exported ``FMP_API_KEY`` a single
  live :func:`~qedge.scanner.pipeline.scan_symbol` on SPY must return a populated
  result. Offline (the gitignored ``.env`` key is *not* exported to the shell)
  this test SKIPS cleanly — that skip is the expected outcome here.

Performance contract: the scanner pipeline is heavy (~15-20s per scan), so the
synthetic scan is performed EXACTLY ONCE behind a module-scoped fixture and
shared by every offline assertion. The HMM Baum-Welch iteration count is trimmed
because these tests assert plumbing and provenance, not regime-fit quality.

Heavy ML deps (hmmlearn / lightgbm / shap) are imported lazily via
``pytest.importorskip`` so this file still *collects* on a bare numpy/pandas
environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

import pytest

from qedge.config import QedgeConfig
from qedge.data.fmp_adapter import FMPPriceFeed
from qedge.data.synthetic import SyntheticMarket
from qedge.repro import RunRecord, make_run_record, write_run_record

pytest.importorskip("hmmlearn")
pytest.importorskip("lightgbm")
pytest.importorskip("shap")

from qedge.scanner.pipeline import (  # noqa: E402  (after the importorskip gate)
    EdgeResult,
    run_scan,
    scan_symbol,
)

#: The single synthetic symbol the offline end-to-end scan runs over. A
#: one-symbol universe keeps the (heavy) scan to a single ensemble pass.
_SYMBOL: Final[str] = "SYN"

#: A trimmed-but-real synthetic window. Long enough that the triple-barrier
#: labeller yields more than ``cpcv_n_groups`` clean events (so the survival gate
#: actually runs), short enough to keep the one heavy scan tractable.
_HISTORY_START: Final[str] = "2021-01-01"
_HISTORY_END: Final[str] = "2022-12-31"

#: Trimmed Baum-Welch iterations: these tests assert plumbing/provenance, not the
#: quality of the HMM fit, so a short fit keeps the single scan fast.
_FAST_HMM_ITERS: Final[int] = 6

#: A sha256 hex digest is exactly this many characters.
_SHA256_HEX_LEN: Final[int] = 64

#: The live SPY smoke test scans the short book (1-10 day cap).
_LIVE_SYMBOL: Final[str] = "SPY"
_SHORT_HORIZON: Final[str] = "short"

#: The data-snapshot key under which the feature hash is logged in a RunRecord.
_FEATURE_HASH_KEY: Final[str] = "features"


def _fast_config() -> QedgeConfig:
    """A default config with the HMM fit trimmed so the single scan is quick."""
    cfg = QedgeConfig()
    cfg.regime.hmm_n_iter = _FAST_HMM_ITERS
    return cfg


@pytest.fixture(scope="module")
def scan_config() -> QedgeConfig:
    """The (module-shared) trimmed config used for the one synthetic scan."""
    return _fast_config()


@pytest.fixture(scope="module")
def synthetic_results(scan_config: QedgeConfig) -> list[EdgeResult]:
    """Run the ONE heavy synthetic end-to-end scan and share it module-wide.

    A deterministic :class:`SyntheticMarket` over a single-symbol universe is
    injected as the feed, so the full pipeline (PIT features, filtered HMM
    regime, triple-barrier labels, purged CPCV, family ensemble, execution costs,
    DSR/PBO gate) runs unmodified against reproducible data. Module scope is
    load-bearing: it guarantees the costly scan happens exactly once.
    """
    feed = SyntheticMarket(
        _HISTORY_START,
        _HISTORY_END,
        seed=scan_config.scanner.seed,
        symbols=(_SYMBOL,),
        config=scan_config,
    )
    return run_scan(
        universe=(_SYMBOL,),
        feed=feed,
        horizon=_SHORT_HORIZON,
        config=scan_config,
    )


def test_run_scan_returns_populated_results_with_provenance(
    synthetic_results: list[EdgeResult],
) -> None:
    """One synthetic ``run_scan`` returns populated EdgeResults with provenance.

    Every result must carry a 64-hex ``feature_snapshot_hash`` (the data-snapshot
    digest) and a non-empty ``run_id`` (the deterministic provenance stamp), and
    the scanned symbol must be the one requested.
    """
    assert isinstance(synthetic_results, list)
    assert synthetic_results, "run_scan returned no EdgeResults"
    assert len(synthetic_results) == 1  # one-symbol universe -> one verdict

    for result in synthetic_results:
        assert isinstance(result, EdgeResult)
        assert result.symbol == _SYMBOL
        assert result.horizon == _SHORT_HORIZON
        # Provenance: a real sha256 hex digest and a non-empty run id.
        assert len(result.feature_snapshot_hash) == _SHA256_HEX_LEN
        assert all(c in "0123456789abcdef" for c in result.feature_snapshot_hash)
        assert isinstance(result.run_id, str)
        assert result.run_id


def test_run_record_round_trip_preserves_run_id_and_metrics(
    synthetic_results: list[EdgeResult],
    tmp_path: Path,
) -> None:
    """A RunRecord built from an EdgeResult survives a JSON write/read unchanged.

    This is the reproducibility / audit-trail proof: the scan's identity
    (``run_id``) and its key metrics persist byte-for-byte across the round-trip,
    so a logged run can be reloaded and trusted.
    """
    result = synthetic_results[0]

    config: dict[str, Any] = {
        "symbol": result.symbol,
        "horizon": result.horizon,
        "n_trials": result.n_trials,
    }
    data_hashes: dict[str, str] = {_FEATURE_HASH_KEY: result.feature_snapshot_hash}
    metrics: dict[str, Any] = {
        "n_events": result.n_events,
        "oos_sharpe": result.oos_sharpe,
        "deflated_sharpe": result.deflated_sharpe,
        "pbo": result.pbo,
        "passed": result.passed,
        "break_even_aum_usd": result.break_even_aum_usd,
    }

    record = make_run_record(config=config, data_hashes=data_hashes, metrics=metrics)

    path = write_run_record(record, tmp_path)
    assert path.exists()
    assert path.name == f"{record.run_id}.json"

    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # Identity survives the round-trip.
    assert loaded["run_id"] == record.run_id
    # The data-snapshot hash survives.
    assert loaded["data_hashes"][_FEATURE_HASH_KEY] == result.feature_snapshot_hash
    # Key metrics survive (JSON renders NaN as the literal "NaN" via default=str;
    # comparing the round-tripped value to the re-serialised source is exact).
    assert loaded["metrics"]["n_events"] == result.n_events
    assert loaded["metrics"]["passed"] == result.passed
    assert json.dumps(loaded["metrics"]["oos_sharpe"]) == json.dumps(
        record.metrics["oos_sharpe"], default=str
    )
    assert json.dumps(loaded["metrics"]["deflated_sharpe"]) == json.dumps(
        record.metrics["deflated_sharpe"], default=str
    )
    assert json.dumps(loaded["metrics"]["pbo"]) == json.dumps(
        record.metrics["pbo"], default=str
    )

    # Re-deriving the record from the same inputs yields the same id (the run_id
    # is a pure function of config + data snapshot, never the wall clock).
    twin: RunRecord = make_run_record(
        config=config, data_hashes=data_hashes, metrics=metrics
    )
    assert twin.run_id == record.run_id


@pytest.mark.skipif(
    not os.getenv("FMP_API_KEY"), reason="no FMP key in env"
)
def test_live_fmp_smoke_scan_spy() -> None:
    """LIVE smoke: a real FMP-backed scan of SPY returns a populated result.

    This is the ONLY network test in the suite. It is gated on an exported
    ``FMP_API_KEY``; the gitignored ``.env`` key is not exported to the shell, so
    offline this SKIPS cleanly (the expected outcome here).
    """
    feed = FMPPriceFeed()
    result = scan_symbol(_LIVE_SYMBOL, feed=feed, horizon=_SHORT_HORIZON)

    assert isinstance(result, EdgeResult)
    assert result.symbol == _LIVE_SYMBOL
    assert result.horizon == _SHORT_HORIZON
    assert result.n_events > 0
    assert len(result.feature_snapshot_hash) == _SHA256_HEX_LEN
    assert result.run_id
