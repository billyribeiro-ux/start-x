"""Tests for the GENUINE paper-forward harness (``startx.forward.book_paper``).

This is the honest forward harness — it drives the THREE VALIDATED books (short_swing /
long_swing / position), not the abandoned ``prob_up`` model. The contract these tests pin:

  * it runs end-to-end and produces a non-empty, coherent combined blotter;
  * the blotter is genuinely OUT-OF-SAMPLE — **no trade has an ``entry_date`` strictly
    before ``forward_start``** (the point-in-time / holdout guarantee);
  * per-book stats and the one-line summary exist and are well-formed.

The run reads the local price-parquet cache (the same source ``scripts/run_book.py`` reads),
so it is a real end-to-end exercise of the production books over a holdout window.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from startx.forward.book_paper import (
    BLOTTER_COLUMNS,
    DEFAULT_END,
    DEFAULT_FORWARD_START,
    BookForward,
    ForwardBooksResult,
    run_forward_books,
)

_PRICE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "prices"
_REQUIRED = ["SPY", "_VIX", "_VVIX", "GLD"]
_HAVE_DATA = all((_PRICE_DIR / f"{s}.parquet").exists() for s in _REQUIRED)

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason="local price-parquet cache (SPY/_VIX/_VVIX/GLD) not available"
)

_FORWARD_START = "2023-01-01"
_END = "2026-06-19"


@pytest.fixture(scope="module")
def result() -> ForwardBooksResult:
    """Run the forward harness once over the default holdout window (reused across tests)."""
    return run_forward_books(_FORWARD_START, _END, price_dir=_PRICE_DIR)


def test_runs_and_returns_all_three_books(result: ForwardBooksResult):
    assert isinstance(result, ForwardBooksResult)
    assert set(result.books) == {"short_swing", "long_swing", "position"}
    for name, book in result.books.items():
        assert isinstance(book, BookForward)
        assert book.book == name
        assert not book.equity.empty
    assert result.forward_start == pd.Timestamp(_FORWARD_START)
    assert result.end == pd.Timestamp(_END)


def test_combined_blotter_is_non_empty_and_coherent(result: ForwardBooksResult):
    comb = result.combined_blotter
    assert list(comb.columns) == BLOTTER_COLUMNS
    assert not comb.empty, "the three books should produce trades on a 3.5-year holdout"

    # tagged by every book that traded; combined == sum of per-book blotters
    assert set(comb["book"]) <= {"short_swing", "long_swing", "position"}
    per_book_total = sum(len(b.blotter) for b in result.books.values())
    assert len(comb) == per_book_total

    # coherence: exit on/after entry, finite prices, outcome matches the ret sign
    assert (comb["exit_date"] >= comb["entry_date"]).all()
    assert (comb["entry_price"] > 0).all() and (comb["exit_price"] > 0).all()
    assert (comb["bars_held"] >= 0).all()
    assert set(comb["outcome"]) <= {"WIN", "LOSS", "SCRATCH"}
    assert set(comb["status"]) <= {"open", "closed"}
    # desk rule: a >0 net trade is a WIN/SCRATCH, a <0 trade is a LOSS (never mislabelled)
    assert (comb.loc[comb["ret"] > 0, "outcome"] != "LOSS").all()
    assert (comb.loc[comb["ret"] < 0, "outcome"] == "LOSS").all()


def test_no_trade_enters_before_forward_start(result: ForwardBooksResult):
    """THE out-of-sample guarantee: nothing is traded before the holdout begins."""
    fs = pd.Timestamp(_FORWARD_START)
    comb = result.combined_blotter
    assert (comb["entry_date"] >= fs).all(), "a trade entered before forward_start → not OOS"
    # and per book, individually
    for name, book in result.books.items():
        if not book.blotter.empty:
            assert (book.blotter["entry_date"] >= fs).all(), f"{name} leaked a pre-holdout entry"


def test_per_book_stats_are_well_formed(result: ForwardBooksResult):
    for name, book in result.books.items():
        s = book.stats
        for key in ("n", "win_rate", "total_return", "max_drawdown", "ann_sharpe"):
            assert key in s, f"{name} stats missing {key}"
        assert s["n"] == len(book.blotter)
        # max drawdown is a non-positive fraction; total_return is finite
        assert s["max_drawdown"] <= 0 or pd.isna(s["max_drawdown"])
        assert pd.notna(s["total_return"])
    # the position book carries its honest yardstick (SPY buy-and-hold benchmark)
    assert "benchmark" in result.books["position"].stats


def test_summary_is_honest_and_mentions_holdout(result: ForwardBooksResult):
    summ = result.summary.lower()
    assert "holdout" in summ or "simulation" in summ
    assert "not a live broker" in summ
    # every book is named in the one-liner
    for name in ("short_swing", "long_swing", "position"):
        assert name in result.summary


def test_default_window_constants():
    assert DEFAULT_FORWARD_START == "2023-01-01"
    assert DEFAULT_END == "2026-06-19"


def test_later_forward_start_does_not_leak_earlier_entries(result: ForwardBooksResult):
    """A tighter holdout must not contain any entry before its (later) start."""
    later = run_forward_books("2024-06-01", _END, price_dir=_PRICE_DIR)
    comb = later.combined_blotter
    assert not comb.empty
    assert (comb["entry_date"] >= pd.Timestamp("2024-06-01")).all()
    # a later start can only keep a subset (or fewer) of the full-window trades
    assert len(comb) <= len(result.combined_blotter)
