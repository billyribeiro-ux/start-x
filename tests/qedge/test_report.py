"""Unit tests for the scanner report renderers.

These build :class:`~qedge.scanner.pipeline.EdgeResult` instances DIRECTLY (no
scan is ever run — the pipeline is heavy) and assert the three presentation
contracts:

* :func:`scorecard_table` surfaces every symbol and its verdict, with the
  ship/no-ship call visually distinguishable;
* :func:`results_to_frame` exposes the full EdgeResult field set with the right
  shape and list-valued columns carried through losslessly;
* :func:`format_edge` states the explicit honest-degradation caveat exactly when
  the edge has unpopulated contract dimensions, and omits it otherwise.
"""
from __future__ import annotations

import pandas as pd

from qedge.scanner.pipeline import EdgeResult
from qedge.scanner.report import (
    format_edge,
    results_to_frame,
    scorecard_table,
)


def _robust_edge() -> EdgeResult:
    """A passing edge with options/internals UNPOPULATED."""
    return EdgeResult(
        symbol="SPY",
        horizon="short",
        n_events=120,
        n_trials=3,
        oos_sharpe=1.42,
        deflated_sharpe=0.971,
        pbo=0.12,
        passed=True,
        verdict="ROBUST",
        break_even_aum_usd=125_000_000.0,
        top_features=[("ret_5d", 0.031), ("ret_1d", 0.018), ("rsi_2", 0.009)],
        unpopulated_dimensions=["options_nbbo", "internals"],
        feature_snapshot_hash="a" * 64,
        run_id="run_spy_0001",
    )


def _overfit_edge() -> EdgeResult:
    """A failing edge with NO unpopulated dimensions (fully exercised)."""
    return EdgeResult(
        symbol="QQQ",
        horizon="long",
        n_events=64,
        n_trials=3,
        oos_sharpe=0.05,
        deflated_sharpe=0.30,
        pbo=0.71,
        passed=False,
        verdict="LIKELY OVERFIT",
        break_even_aum_usd=0.0,
        top_features=[("macd", 0.002)],
        unpopulated_dimensions=[],
        feature_snapshot_hash="b" * 64,
        run_id="run_qqq_0002",
    )


def _degenerate_edge() -> EdgeResult:
    """An edge with too few events: NaN metrics, empty feature ranking."""
    return EdgeResult(
        symbol="IWM",
        horizon="short",
        n_events=2,
        n_trials=3,
        oos_sharpe=float("nan"),
        deflated_sharpe=float("nan"),
        pbo=float("nan"),
        passed=False,
        verdict="LIKELY OVERFIT",
        break_even_aum_usd=0.0,
        top_features=[],
        unpopulated_dimensions=["options_nbbo", "internals"],
        feature_snapshot_hash="c" * 64,
        run_id="run_iwm_0003",
    )


def test_scorecard_table_contains_symbols_and_verdicts() -> None:
    """Every symbol and verdict appears; pass/fail is visually distinguishable."""
    results = [_robust_edge(), _overfit_edge(), _degenerate_edge()]
    table = scorecard_table(results)

    for symbol in ("SPY", "QQQ", "IWM"):
        assert symbol in table
    assert "ROBUST" in table
    assert "LIKELY OVERFIT" in table
    # The verdict is marked so the ship/no-ship call is obvious at a glance.
    assert "[PASS]" in table
    assert "[FAIL]" in table
    # It is a markdown-style table (header + alignment separator + N body rows).
    lines = table.splitlines()
    assert lines[0].startswith("|") and "Symbol" in lines[0]
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-"}
    assert len(lines) == 2 + len(results)


def test_scorecard_table_renders_nan_and_empty_cells() -> None:
    """A degenerate edge renders 'nan' metrics and a sentinel for empty features."""
    table = scorecard_table([_degenerate_edge()])
    assert "nan" in table
    # The empty top-feature list shows the sentinel rather than an empty cell.
    assert "-" in table


def test_scorecard_table_empty_results_is_header_only() -> None:
    """No symbols scanned yields just the header + separator (still valid)."""
    table = scorecard_table([])
    lines = table.splitlines()
    assert len(lines) == 2
    assert "Symbol" in lines[0]


def test_results_to_frame_columns_and_shape() -> None:
    """The frame has one row per result and every EdgeResult field as a column."""
    results = [_robust_edge(), _overfit_edge(), _degenerate_edge()]
    frame = results_to_frame(results)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == len(results)
    expected_columns = {
        "symbol",
        "horizon",
        "n_events",
        "n_trials",
        "oos_sharpe",
        "deflated_sharpe",
        "pbo",
        "passed",
        "verdict",
        "break_even_aum_usd",
        "top_features",
        "unpopulated_dimensions",
        "feature_snapshot_hash",
        "run_id",
    }
    assert set(frame.columns) == expected_columns
    # List-valued fields carried through losslessly (not stringified).
    spy_row = frame.loc[frame["symbol"] == "SPY"].iloc[0]
    assert spy_row["top_features"] == [
        ("ret_5d", 0.031),
        ("ret_1d", 0.018),
        ("rsi_2", 0.009),
    ]
    assert spy_row["unpopulated_dimensions"] == ["options_nbbo", "internals"]
    assert bool(spy_row["passed"]) is True


def test_results_to_frame_empty_has_full_columns() -> None:
    """An empty result list still yields the full, correctly-ordered column set."""
    frame = results_to_frame([])
    assert frame.shape[0] == 0
    assert "feature_snapshot_hash" in frame.columns
    assert next(iter(frame.columns)) == "symbol"


def test_format_edge_surfaces_unpopulated_caveat() -> None:
    """An edge with unpopulated dims emits an explicit caveat naming them."""
    block = format_edge(_robust_edge())

    assert "SPY" in block
    assert "ROBUST" in block
    assert "[PASS]" in block
    # The honest-degradation caveat is present and names both missing dimensions.
    assert "UNPOPULATED" in block
    assert "options_nbbo" in block
    assert "internals" in block
    # Headline evidence is rendered.
    assert "Deflated Sharpe" in block
    assert "0.971" in block


def test_format_edge_omits_caveat_when_fully_populated() -> None:
    """An edge with no unpopulated dims has NO caveat line."""
    block = format_edge(_overfit_edge())

    assert "QQQ" in block
    assert "LIKELY OVERFIT" in block
    assert "[FAIL]" in block
    assert "UNPOPULATED" not in block
    assert "CAVEAT" not in block
