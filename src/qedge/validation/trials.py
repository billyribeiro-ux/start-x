"""Multiple-testing accounting — the trial ledger.

Every configuration / strategy variant that is *tried* against the data is a
selection opportunity, and the more variants you try the easier it is to find one
that looks good by luck alone. The Deflated Sharpe Ratio penalises exactly this:
it deflates the observed Sharpe by the number of independent trials. To feed DSR
honestly we must *count* those trials, so this ledger is the running tally of how
many distinct configurations have been evaluated in a search.

The ledger is deterministic and order-stable: it dedupes by name (registering the
same configuration twice does not inflate the count, because re-testing an
identical config is not an additional independent trial) and preserves first-seen
insertion order for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrialLedger:
    """Deterministic tally of distinct configurations tried (for DSR deflation).

    Names are deduped: each unique ``name`` counts once no matter how many times
    it is registered, because re-running the same configuration is not a new
    independent trial. Insertion order of first-seen names is preserved.
    """

    _names: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def register(self, name: str) -> None:
        """Record that configuration ``name`` was tried (idempotent per name)."""
        if name not in self._seen:
            self._seen.add(name)
            self._names.append(name)

    @property
    def count(self) -> int:
        """Number of distinct configurations registered (the DSR ``n_trials``)."""
        return len(self._names)

    @property
    def names(self) -> tuple[str, ...]:
        """Distinct registered names in first-seen order (immutable snapshot)."""
        return tuple(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._seen

    def __len__(self) -> int:
        return len(self._names)
