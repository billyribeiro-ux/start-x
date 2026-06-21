"""Command-line entry point for the qedge adversarial scanner.

Usage::

    python -m qedge.scanner.cli --universe SPY,QQQ,IWM [--horizon short|long]
                                [--live] [--out DIR]

By default (no ``--live``) the scan runs entirely OFFLINE against the
deterministic :class:`~qedge.data.synthetic.SyntheticMarket`, so a run is
byte-reproducible and needs no API key. Passing ``--live`` swaps in the
:class:`~qedge.data.fmp_adapter.FMPPriceFeed`, which reads the FMP key from the
gitignored ``.env`` via the reused ``startx`` settings layer.

The CLI threads the chosen feed and horizon through
:func:`~qedge.scanner.pipeline.run_scan`, prints the
:func:`~qedge.scanner.report.scorecard_table`, and persists one
:class:`~qedge.repro.RunRecord` JSON per symbol to ``--out`` (defaulting to
``cfg.scanner.output_dir``) so every scan leaves an auditable trail.

Determinism hook: tests (and any embedder) may install a custom
:class:`~qedge.data.protocols.PriceFeed` via :func:`set_feed_override`. When set
it takes precedence over both the synthetic default and ``--live``, so a test can
drive the real pipeline with a tiny fixture feed without touching argv parsing.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from qedge.config import QedgeConfig, get_config
from qedge.data.fmp_adapter import FMPPriceFeed
from qedge.data.protocols import PriceFeed
from qedge.repro import make_run_record, write_run_record
from qedge.scanner.pipeline import EdgeResult, Horizon, run_scan
from qedge.scanner.report import scorecard_table

__all__ = ["main", "set_feed_override"]

#: argv separator for the comma-delimited ``--universe`` list.
_UNIVERSE_SEP: str = ","

#: The two horizons the desk runs as separate books (mirrors pipeline.Horizon).
_HORIZON_CHOICES: tuple[str, ...] = ("short", "long")

#: Process exit codes.
_EXIT_OK: int = 0
_EXIT_USAGE: int = 2

#: Optional test/embedder feed override. When non-None it wins over --live and
#: the synthetic default, letting a caller inject a deterministic fixture feed.
_FEED_OVERRIDE: PriceFeed | None = None


def set_feed_override(feed: PriceFeed | None) -> None:
    """Install (or clear, with ``None``) a process-wide feed override for scans.

    This is the deterministic-injection hook documented in the module docstring:
    a test can wire in a tiny in-memory :class:`PriceFeed` so the CLI exercises
    the real pipeline without the synthetic default or any live network.
    """
    global _FEED_OVERRIDE
    _FEED_OVERRIDE = feed


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the scanner CLI."""
    parser = argparse.ArgumentParser(
        prog="qedge.scanner.cli",
        description="Run the qedge adversarial edge-discovery scanner.",
    )
    parser.add_argument(
        "--universe",
        required=True,
        help=(
            "Comma-separated symbols to scan, e.g. 'SPY,QQQ,IWM'. "
            "Each symbol/horizon is one trial in the shared multiple-testing ledger."
        ),
    )
    parser.add_argument(
        "--horizon",
        choices=_HORIZON_CHOICES,
        default="short",
        help="Which book to scan: 'short' (1-10 day cap) or 'long' (~63-day cap).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Use the live FMP price feed instead of the deterministic synthetic "
            "default. Requires the FMP key in .env (read via startx settings)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Directory to write per-symbol run-record JSON files. "
            "Defaults to cfg.scanner.output_dir."
        ),
    )
    return parser


def _parse_universe(raw: str) -> tuple[str, ...]:
    """Split, trim, and upper-case the comma-delimited universe argument."""
    symbols = tuple(
        token.strip().upper()
        for token in raw.split(_UNIVERSE_SEP)
        if token.strip()
    )
    return symbols


def _resolve_feed(*, live: bool, config: QedgeConfig) -> PriceFeed | None:
    """Pick the feed for the scan.

    The test/embedder override wins; then ``--live`` selects the FMP feed; else
    ``None`` lets :func:`run_scan` build the deterministic synthetic default.
    """
    if _FEED_OVERRIDE is not None:
        return _FEED_OVERRIDE
    if live:
        return FMPPriceFeed(config=config)
    return None


def _write_records(
    results: Sequence[EdgeResult],
    *,
    horizon: str,
    output_dir: str,
) -> None:
    """Persist one :class:`~qedge.repro.RunRecord` JSON per scanned symbol.

    Each record mirrors the EdgeResult verdict (the same evidence the scorecard
    shows) and is written under ``output_dir`` keyed by its deterministic run id.
    """
    for result in results:
        record = make_run_record(
            config={
                "symbol": result.symbol,
                "horizon": horizon,
                "n_trials": result.n_trials,
            },
            data_hashes={"features": result.feature_snapshot_hash},
            metrics={
                "n_events": result.n_events,
                "oos_sharpe": result.oos_sharpe,
                "deflated_sharpe": result.deflated_sharpe,
                "pbo": result.pbo,
                "passed": result.passed,
                "verdict": result.verdict,
                "break_even_aum_usd": result.break_even_aum_usd,
                "run_id": result.run_id,
            },
        )
        write_run_record(record, output_dir)


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the scan, print the scorecard, and write run records.

    Returns ``0`` on success and ``2`` on a usage error (empty universe). The
    scan itself is heavy (the full pipeline per symbol); callers wanting a fast,
    deterministic run should scan a single symbol on the synthetic default.
    """
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    universe = _parse_universe(args.universe)
    if not universe:
        parser.error("--universe must contain at least one symbol")
        return _EXIT_USAGE  # pragma: no cover (parser.error exits)

    cfg = get_config()
    horizon: Horizon = args.horizon
    output_dir = args.out if args.out is not None else cfg.scanner.output_dir
    feed = _resolve_feed(live=args.live, config=cfg)

    results = run_scan(universe, feed=feed, horizon=horizon, config=cfg)

    print(scorecard_table(results))
    _write_records(results, horizon=horizon, output_dir=output_dir)
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
