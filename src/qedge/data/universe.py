"""Trading-universe resolution with a survivorship-aware membership hook.

The seed universe is the static set of liquid ETFs declared in
``cfg.data.universe`` (SPY / QQQ / IWM). Those ETFs have no delisting history, so
membership is constant through time. The :class:`MembershipProvider` Protocol and
the concrete :class:`StaticMembershipProvider` nonetheless expose a
``members(asof)`` hook keyed on a point-in-time timestamp, so that a future
delisting-aware equity universe (which *must* drop names that were not yet listed
or had already delisted as of ``asof`` to avoid survivorship bias) is a drop-in
replacement implementing the same Protocol.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from qedge.config import QedgeConfig, get_config


@runtime_checkable
class MembershipProvider(Protocol):
    """Point-in-time universe membership.

    ``members(asof)`` returns the set of symbols that were legitimately tradeable
    as of ``asof``. For the static ETF seed this is constant; for a real equity
    universe it is survivorship-aware (excludes not-yet-listed and already-delisted
    names) so backtests never see tickers that did not exist at the decision time.
    """

    def members(self, *, asof: pd.Timestamp) -> frozenset[str]:
        """Return the symbols tradeable as of ``asof``."""
        ...


class StaticMembershipProvider:
    """Membership provider for a fixed, survivorship-clean set of symbols.

    Used for the seed ETF universe whose constituents never delist; the
    ``members`` hook ignores ``asof`` and always returns the full set. It exists
    so callers can depend on the :class:`MembershipProvider` Protocol today and
    swap in a delisting-aware equity provider later without code changes.

    Args:
        symbols: The constant universe. Order is preserved in :attr:`ordered`.
    """

    def __init__(self, symbols: tuple[str, ...]) -> None:
        if not symbols:
            raise ValueError("universe must contain at least one symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"duplicate symbols in universe: {symbols}")
        self._ordered: tuple[str, ...] = symbols
        self._members: frozenset[str] = frozenset(symbols)

    @property
    def ordered(self) -> tuple[str, ...]:
        """The universe in declaration order (membership is order-insensitive)."""
        return self._ordered

    def members(self, *, asof: pd.Timestamp) -> frozenset[str]:
        """Return the full static set; ``asof`` is accepted but irrelevant here.

        The argument is consumed (``pd.Timestamp(asof)``) purely to validate that
        callers pass a real timestamp, keeping the call-site identical to a
        survivorship-aware provider.
        """
        _ = pd.Timestamp(asof)
        return self._members


def resolve_universe(config: QedgeConfig | None = None) -> StaticMembershipProvider:
    """Build the :class:`StaticMembershipProvider` from ``cfg.data.universe``.

    Args:
        config: Optional config override; defaults to the process singleton.

    Returns:
        A provider over the configured seed ETF universe (SPY / QQQ / IWM).
    """
    cfg = config if config is not None else get_config()
    return StaticMembershipProvider(tuple(cfg.data.universe))
