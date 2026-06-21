"""Prices-level coverage/staleness tests. Fully offline: the FMP loader is replaced by
an in-process fake that records the requested range and returns a synthetic frame. No
network and no real ``FMPClient`` are used.

These exercise the end-to-end wiring of get_prices -> cache.get_or_fetch with the
trading-day-aware coverage_end, which is the defect under repair: previously get_prices
called get_or_fetch WITHOUT coverage args, so a truncated/stale cache was served verbatim.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import startx.data.prices as prices_mod
from startx.data.cache import ParquetCache
from startx.data.prices import _last_expected_trading_day, get_prices


class _FakeFMP:
    """Stand-in for the loader returning a calendar-aware EOD frame for [start, end].

    Records every requested (start, end) so tests can assert re-fetch vs cache-hit.
    """

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, client, symbol, start=None, end=None):
        self.calls.append((start, end))
        # Business-day bars across the requested window — mimics real EOD history.
        idx = pd.bdate_range(start=start, end=end)
        return pd.DataFrame(
            {
                "date": idx,
                "open": range(1, len(idx) + 1),
                "high": range(1, len(idx) + 1),
                "low": range(1, len(idx) + 1),
                "close": range(1, len(idx) + 1),
                "volume": [100] * len(idx),
            }
        )


@pytest.fixture
def fake_loader(monkeypatch):
    fake = _FakeFMP()
    monkeypatch.setattr(prices_mod, "historical_price_eod", fake)
    return fake


# --- (a) wider request must re-fetch, not serve the truncated 2019 series ----

def test_wider_history_start_refetches(fake_loader, tmp_path, monkeypatch):
    """Warm at history_start=2019, then ask for 2010 -> MUST re-fetch the full series."""
    # Freeze "today" so coverage_end is deterministic (a weekday).
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 6, 19)))  # Friday
    cache = ParquetCache(tmp_path)

    warm = get_prices(None, cache, "SPY", history_start="2019-01-01")
    assert warm["date"].min().year == 2019

    wide = get_prices(None, cache, "SPY", history_start="2010-01-01")
    # Re-fetched with the wider start (not served the 2019-truncated cache).
    assert [c[0] for c in fake_loader.calls] == ["2019-01-01", "2010-01-01"]
    assert wide["date"].min().year == 2010


# --- (b) weekend/holiday run against an up-to-date cache must NOT re-fetch ---

def test_weekend_run_does_not_refetch(fake_loader, tmp_path, monkeypatch):
    """Cache built Friday; a Saturday/Sunday run (no new trading day) is a cache hit."""
    # Friday: build the cache. Last bar is Friday 2026-06-19.
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 6, 19)))  # Fri
    cache = ParquetCache(tmp_path)
    get_prices(None, cache, "SPY", history_start="2019-01-01")
    assert len(fake_loader.calls) == 1

    # Saturday run: last expected trading day is still Friday -> no re-fetch.
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 6, 20)))  # Sat
    get_prices(None, cache, "SPY", history_start="2019-01-01")
    # Sunday run: still Friday -> no re-fetch.
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 6, 21)))  # Sun
    get_prices(None, cache, "SPY", history_start="2019-01-01")

    assert len(fake_loader.calls) == 1  # never re-fetched over the weekend


# --- (c) a stale tail (cache ends weeks ago) extends -------------------------

def test_stale_tail_extends(fake_loader, tmp_path, monkeypatch):
    """A cache whose last bar is weeks old re-fetches when a new trading day exists."""
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 5, 1)))  # Fri
    cache = ParquetCache(tmp_path)
    first = get_prices(None, cache, "SPY", history_start="2019-01-01")
    old_tail = first["date"].max()
    assert len(fake_loader.calls) == 1

    # Weeks later, a fresh weekday -> stale tail -> re-fetch and extend.
    monkeypatch.setattr(prices_mod, "date", _FrozenDate(date(2026, 6, 19)))  # Fri
    second = get_prices(None, cache, "SPY", history_start="2019-01-01")
    assert len(fake_loader.calls) == 2
    assert second["date"].max() > old_tail


# --- trading-day staleness helper -------------------------------------------

def test_last_expected_trading_day_skips_weekend():
    # Sat 2026-06-20 and Sun 2026-06-21 both roll back to Fri 2026-06-19.
    assert _last_expected_trading_day(pd.Timestamp("2026-06-20")) == pd.Timestamp("2026-06-19")
    assert _last_expected_trading_day(pd.Timestamp("2026-06-21")) == pd.Timestamp("2026-06-19")
    # A weekday maps to itself.
    assert _last_expected_trading_day(pd.Timestamp("2026-06-19")) == pd.Timestamp("2026-06-19")


class _FrozenDate:
    """Minimal stand-in for ``datetime.date`` exposing the ``.today()`` used by prices."""

    def __init__(self, d: date):
        self._d = d

    def today(self) -> date:
        return self._d
