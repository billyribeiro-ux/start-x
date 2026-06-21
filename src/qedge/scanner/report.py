"""Human- and machine-readable rendering of scanner :class:`EdgeResult` verdicts.

This module is presentation-only: it never runs a scan and never gates a
decision. It consumes the frozen :class:`~qedge.scanner.pipeline.EdgeResult`
objects produced by the pipeline and turns them into three views:

* :func:`scorecard_table` — a monospaced, markdown-friendly scoreboard with one
  row per symbol and the headline evidence columns. The verdict is rendered so
  the ship/no-ship call is impossible to miss (``ROBUST`` vs ``LIKELY OVERFIT``).
* :func:`results_to_frame` — a structured :class:`pandas.DataFrame` of every
  :class:`EdgeResult` field for programmatic / notebook consumption.
* :func:`format_edge` — a per-edge detailed block that, crucially, states the
  explicit honest-degradation caveat whenever a contract dimension (options
  NBBO, market internals) was unpopulated for that scan.

The honest-degradation rule is load-bearing: an edge measured without options
or internals data must SAY SO in its report, so a reader never mistakes a
partially-probed scan for a fully-exercised one.
"""
from __future__ import annotations

import math
from typing import Final

import pandas as pd

from qedge.scanner.pipeline import EdgeResult

__all__ = [
    "scorecard_table",
    "results_to_frame",
    "format_edge",
]

#: Number of top features rendered inline in the scorecard / detail block. The
#: pipeline already truncates ``top_features``; this is a defensive display cap
#: so a wider ranking never blows out a table cell.
_TOP_FEATURES_DISPLAY: Final[int] = 3

#: Float display precision for the headline metrics (Sharpe, DSR, PBO).
_METRIC_PRECISION: Final[int] = 3

#: Glyphs that make the verdict scannable at a glance in the monospaced table.
_VERDICT_ROBUST: Final[str] = "ROBUST"
_VERDICT_MARK_PASS: Final[str] = "[PASS]"
_VERDICT_MARK_FAIL: Final[str] = "[FAIL]"

#: Sentinel rendered for an empty / missing list cell.
_NONE_CELL: Final[str] = "-"

#: Caveat sentence emitted by :func:`format_edge` when dimensions are missing.
_UNPOPULATED_CAVEAT: Final[str] = (
    "CAVEAT: the following contract dimensions were UNPOPULATED for this scan "
    "(no backing feed) and were NOT exercised: {dims}. "
    "This edge was measured without them; treat the verdict accordingly."
)

#: Headline columns of the scorecard, in display order.
_SCORECARD_HEADERS: Final[tuple[str, ...]] = (
    "Symbol",
    "Horizon",
    "Events",
    "OOS Sharpe",
    "Deflated Sharpe",
    "PBO",
    "Verdict",
    "Break-even AUM",
    "Top-3 Features",
    "Unpopulated Dims",
)

#: Full, ordered column set for :func:`results_to_frame` — mirrors EdgeResult.
_FRAME_COLUMNS: Final[tuple[str, ...]] = (
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
)


def _fmt_metric(value: float) -> str:
    """Render a float metric at the headline precision, NaN-safe."""
    if value != value:  # NaN — the only float not equal to itself.
        return "nan"
    return f"{value:.{_METRIC_PRECISION}f}"


def _fmt_aum(value: float) -> str:
    """Render a break-even AUM as a thousands-grouped USD amount.

    ``+inf`` means impact never erodes the edge within the model's bracket, which
    reads as "unbounded" rather than a literal ``$inf``.
    """
    if value != value:  # NaN
        return "nan"
    if math.isinf(value):
        return "unbounded"
    return f"${value:,.0f}"


def _fmt_capacity(result: EdgeResult) -> str:
    """Render the capacity ceiling, but only when there is an edge to size.

    The break-even AUM answers "at what AUM does the edge die" — a question that
    is only meaningful once an edge has cleared the survival gate. For a candidate
    that FAILS the gate we report ``N/A`` rather than a misleading dollar figure
    (or ``unbounded``) for a strategy that was judged non-robust in the first place.
    """
    if not result.passed:
        return _NONE_CELL
    return _fmt_aum(result.break_even_aum_usd)


def _fmt_verdict(result: EdgeResult) -> str:
    """Render the verdict with a leading pass/fail mark for instant scanning."""
    mark = _VERDICT_MARK_PASS if result.passed else _VERDICT_MARK_FAIL
    return f"{mark} {result.verdict}"


def _fmt_top_features(top_features: list[tuple[str, float]]) -> str:
    """Render the top features as ``name (score)`` pairs, capped for display."""
    if not top_features:
        return _NONE_CELL
    parts = [
        f"{name} ({score:.{_METRIC_PRECISION}f})"
        for name, score in top_features[:_TOP_FEATURES_DISPLAY]
    ]
    return ", ".join(parts)


def _fmt_dims(dims: list[str]) -> str:
    """Render the unpopulated-dimension list (or a sentinel when fully populated)."""
    return ", ".join(dims) if dims else _NONE_CELL


def _scorecard_row(result: EdgeResult) -> tuple[str, ...]:
    """Project one :class:`EdgeResult` into its display-string scorecard row."""
    return (
        result.symbol,
        result.horizon,
        str(result.n_events),
        _fmt_metric(result.oos_sharpe),
        _fmt_metric(result.deflated_sharpe),
        _fmt_metric(result.pbo),
        _fmt_verdict(result),
        _fmt_capacity(result),
        _fmt_top_features(result.top_features),
        _fmt_dims(result.unpopulated_dimensions),
    )


def scorecard_table(results: list[EdgeResult]) -> str:
    """Return a monospaced, markdown-style scorecard — one row per symbol.

    The table is a GitHub-flavoured markdown table (pipe-delimited with an
    alignment separator row) whose cells are padded to a fixed column width so it
    also renders cleanly in a plain monospaced terminal. The verdict column is
    prefixed with ``[PASS]`` / ``[FAIL]`` so a reader can find the robust edges
    without parsing the numbers.

    An empty ``results`` list yields just the header (a valid, if empty, table)
    so callers never have to special-case "no symbols scanned".
    """
    rows = [_scorecard_row(r) for r in results]
    # Column widths fit the widest of header / any cell, so the table is aligned
    # in a fixed-width font regardless of symbol length or metric magnitude.
    widths = [len(h) for h in _SCORECARD_HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _render(cells: tuple[str, ...]) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    header_line = _render(_SCORECARD_HEADERS)
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [_render(row) for row in rows]
    return "\n".join([header_line, separator, *body])


def results_to_frame(results: list[EdgeResult]) -> pd.DataFrame:
    """Return a structured DataFrame of every :class:`EdgeResult` field.

    Columns are the EdgeResult fields in declaration order (see
    :data:`_FRAME_COLUMNS`); list-valued fields (``top_features``,
    ``unpopulated_dimensions``) are carried through as Python objects so the
    frame round-trips without lossy stringification. An empty ``results`` list
    yields an empty frame with the full, correct column set.
    """
    records: list[dict[str, object]] = [
        {
            "symbol": r.symbol,
            "horizon": r.horizon,
            "n_events": r.n_events,
            "n_trials": r.n_trials,
            "oos_sharpe": r.oos_sharpe,
            "deflated_sharpe": r.deflated_sharpe,
            "pbo": r.pbo,
            "passed": r.passed,
            "verdict": r.verdict,
            "break_even_aum_usd": r.break_even_aum_usd,
            "top_features": list(r.top_features),
            "unpopulated_dimensions": list(r.unpopulated_dimensions),
            "feature_snapshot_hash": r.feature_snapshot_hash,
            "run_id": r.run_id,
        }
        for r in results
    ]
    return pd.DataFrame.from_records(records, columns=list(_FRAME_COLUMNS))


def format_edge(result: EdgeResult) -> str:
    """Return a detailed, human-readable block for a single edge verdict.

    The block leads with the symbol/horizon and the ship/no-ship verdict, lists
    the headline evidence (events, trials, OOS Sharpe, Deflated Sharpe, PBO,
    break-even AUM), the top features, and provenance (feature snapshot hash and
    run id). When the scan left any contract dimension unpopulated, the block
    ENDS with an explicit caveat naming the missing dimensions — the
    honest-degradation rule made visible to the reader.
    """
    lines = [
        f"Edge: {result.symbol} [{result.horizon}] -- {_fmt_verdict(result)}",
        f"  Events            : {result.n_events}",
        f"  Trials (deflation): {result.n_trials}",
        f"  OOS Sharpe        : {_fmt_metric(result.oos_sharpe)}",
        f"  Deflated Sharpe   : {_fmt_metric(result.deflated_sharpe)}",
        f"  PBO               : {_fmt_metric(result.pbo)}",
        f"  Break-even AUM    : {_fmt_capacity(result)}",
        f"  Top features      : {_fmt_top_features(result.top_features)}",
        f"  Feature hash      : {result.feature_snapshot_hash}",
        f"  Run id            : {result.run_id}",
    ]
    if result.unpopulated_dimensions:
        lines.append(
            "  " + _UNPOPULATED_CAVEAT.format(
                dims=", ".join(result.unpopulated_dimensions)
            )
        )
    return "\n".join(lines)
