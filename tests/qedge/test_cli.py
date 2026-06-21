"""Tests for the scanner command-line entry point.

The CLI default path runs the FULL (heavy) pipeline per symbol, so this test
keeps cost down two ways: it scans a SINGLE symbol, and it injects a tiny
deterministic in-memory price feed via :func:`qedge.scanner.cli.set_feed_override`
(the documented determinism hook) so no synthetic feed over the locked
multi-year window is generated. The HMM fit on the process config singleton is
trimmed for the same reason. We assert that ``main`` returns ``0`` and that a
run-record JSON lands in the ``--out`` directory.

Heavy ML deps are imported lazily via ``importorskip`` so the numpy-only tiers
of the suite still collect on a bare environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qedge.config import get_config

pytest.importorskip("hmmlearn")
pytest.importorskip("lightgbm")
pytest.importorskip("shap")

from qedge.scanner import cli  # noqa: E402  (after importorskip gate)

# A short momentum-blocked window: enough events for a defined CPCV while keeping
# the single heavy ensemble fit tractable (mirrors the pipeline test fixture).
_N_BARS = 240
_BLOCK = 20
_DRIFT = 0.004
_NOISE_SD = 0.003
_SEED = 7
_INITIAL_PRICE = 100.0
_FAST_HMM_ITERS = 6


def _make_prices(n: int, seed: int) -> pd.DataFrame:
    """Crafted OHLCV frame with foldable momentum blocks (a learnable edge)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    block_signs = rng.integers(0, 2, size=n // _BLOCK + 1) * 2 - 1
    drift = _DRIFT * np.repeat(block_signs, _BLOCK)[:n]
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
    """A minimal PriceFeed over one fixed frame (ignores asof/start)."""

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


def test_cli_main_runs_offline_and_writes_run_record(tmp_path: Path) -> None:
    """main(...) returns 0 and writes a run-record JSON to the --out dir."""
    # Trim the HMM fit on the shared config so the single scan is quick.
    cfg = get_config()
    original_iters = cfg.regime.hmm_n_iter
    cfg.regime.hmm_n_iter = _FAST_HMM_ITERS
    # Inject the deterministic fixture feed (avoids the heavy synthetic default).
    cli.set_feed_override(_FrameFeed(_make_prices(_N_BARS, _SEED)))
    try:
        out_dir = tmp_path / "runs"
        code = cli.main(
            ["--universe", "SIG", "--horizon", "short", "--out", str(out_dir)]
        )
    finally:
        cli.set_feed_override(None)
        cfg.regime.hmm_n_iter = original_iters

    assert code == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1
    # The record is valid JSON carrying the scanned symbol's verdict.
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["config"]["symbol"] == "SIG"
    assert record["config"]["horizon"] == "short"
    assert "verdict" in record["metrics"]
    assert record["run_id"]


def test_cli_empty_universe_is_usage_error() -> None:
    """An all-blank universe is a usage error (argparse exits with code 2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--universe", " , ,"])
    assert exc.value.code == 2
