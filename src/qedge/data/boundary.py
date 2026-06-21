"""Information-boundary declarations.

The contract demands that EVERY feature state its exact information boundary —
the rule that defines which data is legitimately visible when the feature is
computed at time ``t``. An :class:`InformationBoundary` makes that rule an
explicit, inspectable object attached to each feature in the registry, rather
than a comment that can drift out of sync with the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BoundaryKind(str, Enum):
    """The kind of information boundary a feature respects."""

    #: Uses only bars with timestamp <= t (e.g. trailing OHLCV transforms).
    TRAILING = "trailing"
    #: Uses a public event stamped at its dissemination time <= t (lagged feeds).
    AS_OF_PUBLIC = "as_of_public"
    #: A known-in-advance scheduled value (e.g. days-to-next-earnings) — public
    #: as a schedule but flagged because schedules can be revised.
    SCHEDULED = "scheduled"


@dataclass(frozen=True, slots=True)
class InformationBoundary:
    """An explicit, immutable description of a feature's information boundary.

    Attributes:
        kind: Which boundary rule applies.
        lookback_days: Trailing window length in trading days (0 = point value).
        latency_days: Extra dissemination lag applied beyond ``t`` (for public
            events / cross-asset / news). 0 means available at the close of ``t``.
        description: Human-readable statement of the exact boundary, surfaced in
            reports and documentation.
    """

    kind: BoundaryKind
    lookback_days: int
    latency_days: int
    description: str

    def __post_init__(self) -> None:
        if self.lookback_days < 0:
            raise ValueError("lookback_days must be >= 0")
        if self.latency_days < 0:
            raise ValueError("latency_days must be >= 0")
