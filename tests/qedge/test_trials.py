"""Tests for the multiple-testing trial ledger."""
from __future__ import annotations

from qedge.validation import TrialLedger


def test_register_increments_count() -> None:
    ledger = TrialLedger()
    assert ledger.count == 0
    ledger.register("rsi2_dip")
    ledger.register("vix_capitulation")
    assert ledger.count == 2
    assert ledger.names == ("rsi2_dip", "vix_capitulation")


def test_register_dedupes_identical_configs() -> None:
    """Re-testing the same configuration is not a new independent trial."""
    ledger = TrialLedger()
    ledger.register("breakout")
    ledger.register("breakout")
    ledger.register("breakout")
    assert ledger.count == 1
    assert ledger.names == ("breakout",)


def test_names_preserve_first_seen_order() -> None:
    ledger = TrialLedger()
    for name in ("c", "a", "b", "a", "c"):
        ledger.register(name)
    assert ledger.names == ("c", "a", "b")
    assert ledger.count == 3


def test_membership_and_len() -> None:
    ledger = TrialLedger()
    ledger.register("alpha")
    assert "alpha" in ledger
    assert "beta" not in ledger
    assert len(ledger) == 1


def test_count_feeds_dsr_as_n_trials() -> None:
    """The ledger count is exactly the integer DSR deflation needs."""
    ledger = TrialLedger()
    for i in range(25):
        ledger.register(f"config_{i}")
    assert isinstance(ledger.count, int)
    assert ledger.count == 25
