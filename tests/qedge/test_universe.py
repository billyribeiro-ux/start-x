"""Tests for :mod:`qedge.data.universe` resolution and the membership hook."""
from __future__ import annotations

import pandas as pd
import pytest

from qedge.config import QedgeConfig
from qedge.data.universe import (
    MembershipProvider,
    StaticMembershipProvider,
    resolve_universe,
)

_ASOF_EARLY = pd.Timestamp("2019-01-01")
_ASOF_LATE = pd.Timestamp("2026-06-01")


def test_resolve_universe_matches_config() -> None:
    cfg = QedgeConfig()
    provider = resolve_universe(cfg)
    assert provider.ordered == cfg.data.universe
    assert provider.members(asof=_ASOF_EARLY) == frozenset(cfg.data.universe)


def test_resolve_universe_default_config() -> None:
    provider = resolve_universe()
    assert provider.members(asof=_ASOF_LATE) == frozenset({"SPY", "QQQ", "IWM"})


def test_membership_constant_across_time_for_static_etfs() -> None:
    provider = resolve_universe()
    assert provider.members(asof=_ASOF_EARLY) == provider.members(asof=_ASOF_LATE)


def test_membership_returns_frozenset() -> None:
    provider = resolve_universe()
    members = provider.members(asof=_ASOF_EARLY)
    assert isinstance(members, frozenset)


def test_provider_satisfies_protocol() -> None:
    provider = resolve_universe()
    assert isinstance(provider, MembershipProvider)


def test_ordered_preserves_declaration_order() -> None:
    provider = StaticMembershipProvider(("IWM", "SPY", "QQQ"))
    assert provider.ordered == ("IWM", "SPY", "QQQ")
    assert provider.members(asof=_ASOF_EARLY) == frozenset({"IWM", "SPY", "QQQ"})


def test_empty_universe_raises() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        StaticMembershipProvider(())


def test_duplicate_symbols_raise() -> None:
    with pytest.raises(ValueError, match="duplicate symbols"):
        StaticMembershipProvider(("SPY", "SPY"))
