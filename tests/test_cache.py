"""Tests for ParquetCache coverage-check + TTL (truncated-history bug fix).

All loaders here are in-process fakes; no network is touched.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
import pytest

from startx.data.cache import ParquetCache


def _frame(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


# --- backward compatibility -------------------------------------------------

def test_legacy_signature_still_returns_cached(tmp_path):
    """key/fetch/refresh-only callers keep the original cache-hit behaviour."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return _frame(["2019-01-02", "2019-01-03"], [1.0, 2.0])

    first = cache.get_or_fetch("prices/SPY", fetch)
    second = cache.get_or_fetch("prices/SPY", fetch)
    assert len(first) == len(second) == 2
    assert calls == [1]  # fetched once, served from cache the second time


def test_refresh_forces_refetch(tmp_path):
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return _frame(["2019-01-02"], [1.0])

    cache.get_or_fetch("k", fetch)
    cache.get_or_fetch("k", fetch, refresh=True)
    assert calls == [1, 1]


# --- the bug: wider range must NOT serve truncated cache --------------------

def test_wider_range_request_refetches(tmp_path):
    """A request for a wider range than what's cached re-fetches instead of truncating."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch_short():
        calls.append("short")
        return _frame(["2019-01-02", "2019-01-03"], [1.0, 2.0])

    def fetch_long():
        calls.append("long")
        return _frame(
            ["2010-01-04", "2015-06-01", "2019-01-02", "2019-01-03"],
            [0.1, 0.5, 1.0, 2.0],
        )

    # Populate with the short (2019->) window.
    short = cache.get_or_fetch(
        "prices/SPY", fetch_short,
        coverage_start="2019-01-01", coverage_end="2019-01-03",
    )
    assert len(short) == 2

    # Ask for a wider (2010->) window: must re-fetch, not serve the 2-row frame.
    wide = cache.get_or_fetch(
        "prices/SPY", fetch_long,
        coverage_start="2010-01-01", coverage_end="2019-01-03",
    )
    assert calls == ["short", "long"]
    assert len(wide) == 4
    assert wide["date"].min().year == 2010


def test_contained_range_uses_cache(tmp_path):
    """A request fully inside the cached range is served from cache (no re-fetch)."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return _frame(["2010-01-04", "2019-01-02", "2026-06-01"], [0.1, 1.0, 3.0])

    cache.get_or_fetch(
        "prices/SPY", fetch,
        coverage_start="2010-01-01", coverage_end="2026-06-01",
    )
    # Narrower request -> covered -> cache hit.
    cache.get_or_fetch(
        "prices/SPY", fetch,
        coverage_start="2019-01-01", coverage_end="2026-06-01",
    )
    assert calls == [1]


def test_later_end_request_refetches(tmp_path):
    """A request whose end is past the cached tail re-fetches (stale tail)."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch_old():
        calls.append("old")
        return _frame(["2019-01-02", "2020-01-02"], [1.0, 2.0])

    def fetch_new():
        calls.append("new")
        return _frame(["2019-01-02", "2020-01-02", "2026-06-01"], [1.0, 2.0, 3.0])

    cache.get_or_fetch(
        "prices/SPY", fetch_old,
        coverage_start="2019-01-01", coverage_end="2020-01-02",
    )
    cache.get_or_fetch(
        "prices/SPY", fetch_new,
        coverage_start="2019-01-01", coverage_end="2026-06-01",
    )
    assert calls == ["old", "new"]


# --- inferred coverage for caches written without an explicit range ---------

def test_coverage_inferred_from_date_column(tmp_path):
    """Even with no explicit coverage range at save time, the frame's date axis is
    recorded, so a later wider-range request still re-fetches."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch_short():
        calls.append("short")
        return _frame(["2019-01-02", "2019-01-03"], [1.0, 2.0])

    def fetch_long():
        calls.append("long")
        return _frame(["2010-01-04", "2019-01-03"], [0.1, 2.0])

    # Saved WITHOUT a coverage range -> span inferred from the `date` column.
    cache.get_or_fetch("prices/SPY", fetch_short)
    wide = cache.get_or_fetch(
        "prices/SPY", fetch_long,
        coverage_start="2010-01-01", coverage_end="2019-01-03",
    )
    assert calls == ["short", "long"]
    assert len(wide) == 2 and wide["date"].min().year == 2010


def test_no_date_column_warns_and_serves(tmp_path):
    """A payload with no recognisable time axis cannot be coverage-checked: warn, serve."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"symbol": ["SPY"], "value": [42]})

    cache.get_or_fetch("misc/thing", fetch)  # no date column
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = cache.get_or_fetch(
            "misc/thing", fetch,
            coverage_start="2010-01-01", coverage_end="2026-06-01",
        )
    assert calls == [1]  # served from cache despite the range request
    assert len(out) == 1
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# --- coverage_end records the ACTUAL tail, not an over-claimed request ------

def test_coverage_end_clamped_to_actual_data(tmp_path):
    """If the provider returns fewer bars than the requested end (e.g. asked through
    Friday, only Thursday is published yet), the metadata must record the *actual* last
    bar — not the requested end — so a later request re-fetches once the new bar lands.
    Otherwise the missing fresh bar is masked forever."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch_thru_thu():
        calls.append("thu")
        return _frame(["2026-06-18", "2026-06-19"], [1.0, 2.0])  # Thu/Fri... actually

    def fetch_thru_fri():
        calls.append("fri")
        return _frame(["2026-06-18", "2026-06-19", "2026-06-22"], [1.0, 2.0, 3.0])

    # Requested coverage_end is past the last actual bar (2026-06-19).
    cache.get_or_fetch(
        "prices/SPY", fetch_thru_thu,
        coverage_start="2019-01-01", coverage_end="2026-06-22",
    )
    # A later request whose end matches the now-available newer bar must re-fetch,
    # because the recorded coverage_end was clamped to the actual tail (2026-06-19),
    # not the over-claimed 2026-06-22.
    cache.get_or_fetch(
        "prices/SPY", fetch_thru_fri,
        coverage_start="2019-01-01", coverage_end="2026-06-22",
    )
    assert calls == ["thu", "fri"]


def test_coverage_end_request_equal_to_tail_is_cache_hit(tmp_path):
    """When the requested end equals the cached tail, it's a hit (no spurious re-fetch)."""
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return _frame(["2019-01-02", "2026-06-19"], [1.0, 2.0])

    cache.get_or_fetch(
        "prices/SPY", fetch,
        coverage_start="2019-01-01", coverage_end="2026-06-19",
    )
    cache.get_or_fetch(
        "prices/SPY", fetch,
        coverage_start="2019-01-01", coverage_end="2026-06-19",
    )
    assert calls == [1]


# --- TTL --------------------------------------------------------------------

def test_max_age_expiry_refetches(tmp_path):
    cache = ParquetCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return _frame(["2026-06-01"], [3.0])

    cache.get_or_fetch("k", fetch, max_age=100)
    time.sleep(0.01)
    # Within TTL -> cache hit.
    cache.get_or_fetch("k", fetch, max_age=100)
    assert calls == [1]
    # Expired TTL (max_age=0 -> any age is too old) -> re-fetch.
    cache.get_or_fetch("k", fetch, max_age=0)
    assert calls == [1, 1]
