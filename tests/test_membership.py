"""Tests for point-in-time, survivorship-free S&P 500 membership reconstruction.

These pin the properties the ML ranker's honesty depends on: the cross-section on a past date is the
membership AS OF that date (incl. since-removed names), not today's survivors. Uses a small synthetic
change-history so the test is deterministic and offline (no FMP call).
"""
from __future__ import annotations

import pandas as pd

import pytest

from startx.data.membership import CoverageError, SP500Membership


def _toy() -> SP500Membership:
    # Today's members: AAA, BBB, CCC. History (ascending):
    #  2010: XXX removed, AAA added   -> before 2010, XXX was in, AAA was out
    #  2015: YYY removed, BBB added   -> before 2015, YYY was in, BBB was out
    #  2020: (spinoff) CCC added, no removal
    changes = pd.DataFrame([
        {"date": pd.Timestamp("2010-03-01"), "added": "AAA", "removed": "XXX"},
        {"date": pd.Timestamp("2015-06-01"), "added": "BBB", "removed": "YYY"},
        {"date": pd.Timestamp("2020-09-01"), "added": "CCC", "removed": ""},
    ])
    return SP500Membership(current=frozenset({"AAA", "BBB", "CCC"}), changes=changes,
                           sectors={"AAA": "Tech"})


def test_members_today_vs_past():
    m = _toy()
    # today
    assert m.members_asof("2026-01-01") == {"AAA", "BBB", "CCC"}
    # before the 2020 spinoff add -> CCC not yet a member
    assert m.members_asof("2019-01-01") == {"AAA", "BBB"}
    # before 2015 -> BBB out, YYY (since-removed) restored
    assert m.members_asof("2012-01-01") == {"AAA", "YYY"}
    # before 2010 -> AAA out, XXX restored, YYY still in
    assert m.members_asof("2009-01-01") == {"XXX", "YYY"}


def test_survivorship_restores_removed_names():
    """A since-removed name must reappear in the historical cross-section (the whole point)."""
    m = _toy()
    assert "YYY" in m.members_asof("2012-01-01")     # removed 2015 -> present before
    assert "YYY" not in m.current                      # but not today
    assert "XXX" in m.members_asof("2009-01-01")
    assert m.all_symbols() == {"AAA", "BBB", "CCC", "XXX", "YYY"}


def test_change_on_event_date_is_in_effect():
    """members_asof(event_date) includes that day's change (end-of-day convention)."""
    m = _toy()
    assert "BBB" in m.members_asof("2015-06-01")        # added that day
    assert "YYY" not in m.members_asof("2015-06-01")    # removed that day


def test_tradeable_asof_intersects_available():
    m = _toy()
    avail = {"AAA", "YYY", "ZZZ"}                        # we only have prices for these
    assert m.tradeable_asof("2012-01-01", avail) == {"AAA", "YYY"}   # member ∩ available
    assert m.tradeable_asof("2026-01-01", avail) == {"AAA"}


def test_coverage_report():
    m = _toy()
    rep = m.coverage_report({"AAA", "BBB", "CCC"}, ["2009-01-01", "2026-01-01"])
    assert list(rep["n_members"]) == [2, 3]             # XXX,YYY in 2009 (no prices) -> 2 members
    assert rep.loc[rep.date == "2026-01-01", "coverage"].iloc[0] == 1.0


def test_require_coverage_guard():
    """The survivorship guard must REFUSE (raise) when a date's coverage is below threshold."""
    m = _toy()
    # 2009 cross-section is {XXX, YYY}; we have prices for neither -> 0% coverage -> must raise
    with pytest.raises(CoverageError):
        m.require_coverage({"AAA", "BBB", "CCC"}, ["2009-01-01", "2026-01-01"], min_cov=0.95)
    # a fully-covered set must pass and return the report
    rep = m.require_coverage({"AAA", "BBB", "CCC", "XXX", "YYY"}, ["2009-01-01", "2026-01-01"], min_cov=0.95)
    assert (rep["coverage"] >= 0.95).all()
