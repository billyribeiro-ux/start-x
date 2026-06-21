"""Smoke + coherence tests for scripts/cross_instrument.py (ROADMAP #4 cross-instrument).

These don't re-validate the edges (the per-book tests do that) — they pin that the
cross-instrument harness RUNS over the real price cache and returns a *coherent* matrix:
every cached instrument is present in every book with sane, finite headline metrics, and
the position book carries its buy-and-hold benchmark. If the cache or an API shape drifts,
this fails loudly instead of the script silently emitting half a matrix.

Run from the repo root (pytest's configured cwd) so the relative price-cache path resolves.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import scripts.cross_instrument as ci

CACHE = Path("data/cache/prices")
pytestmark = pytest.mark.skipif(
    not (CACHE / "SPY.parquet").exists(),
    reason="price cache not available in this environment",
)


@pytest.fixture(scope="module")
def matrix():
    """Build the locked-window matrix once for the whole module."""
    return ci.run(deep=False)


def _present(matrix) -> list[str]:
    return matrix["_meta"]["present"]


def test_all_cached_instruments_run_in_every_book(matrix):
    present = _present(matrix)
    assert present, "expected at least SPY to be cached"
    locked = matrix["LOCKED WINDOW (headline)"]
    for book in ("short_swing", "long_swing", "position"):
        assert book in locked
        # every cached instrument must appear in every book — no silently dropped rows
        assert set(locked[book].keys()) == set(present)


def test_swing_rows_are_coherent(matrix):
    locked = matrix["LOCKED WINDOW (headline)"]
    for book in ("short_swing", "long_swing"):
        for sym, r in locked[book].items():
            assert r["n"] >= 0
            # a book that produced trades must have finite, in-range headline metrics
            if r["n"] > 0:
                assert 0.0 <= r["win_pct"] <= 1.0, (book, sym, r["win_pct"])
                assert math.isfinite(r["total_return"])
                assert math.isfinite(r["ann_sharpe"])
                assert r["max_drawdown"] <= 0.0  # drawdown is non-positive by construction
                assert r["profit_factor"] >= 0.0 or math.isnan(r["profit_factor"])


def test_long_swing_has_breakout_and_fear_sleeves(matrix):
    locked = matrix["LOCKED WINDOW (headline)"]
    for sym, r in locked["long_swing"].items():
        assert set(r["per_sleeve"].keys()) == {"breakout", "fear"}
        # sleeve counts must reconcile to the book total
        assert sum(v["n"] for v in r["per_sleeve"].values()) == r["n"]


def test_position_carries_buy_and_hold_benchmark(matrix):
    locked = matrix["LOCKED WINDOW (headline)"]
    for sym, r in locked["position"].items():
        for k in ("cagr", "calmar", "max_drawdown", "bh_cagr", "bh_max_drawdown", "bh_calmar"):
            assert k in r, (sym, k)
        assert math.isfinite(r["cagr"])
        assert math.isfinite(r["bh_cagr"])
        # the position book is a drawdown-defense overlay: its drawdown should not be
        # WORSE than buy-and-hold (it sits flat through some declines). Sanity, not strict edge.
        assert r["max_drawdown"] >= r["bh_max_drawdown"] - 1e-9, (sym, r["max_drawdown"], r["bh_max_drawdown"])


def test_known_instruments_present(matrix):
    # SPY must always be cached; the large-cap set should be too in this repo.
    present = set(_present(matrix))
    assert "SPY" in present
    for sym in ("_GSPC", "QQQ", "IWM"):
        assert sym in present, f"expected {sym} cached for the cross-instrument study"
