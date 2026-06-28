"""PIT feature-store skeleton for the swing scanner (Charter v1, P1).

Every feature the engine ever computes carries its provenance so leakage is structurally impossible to
hide: ``as_of`` (the value uses ONLY data timestamped ≤ as_of), a ``lead_time`` tag (so the same
taxonomy re-weights per horizon: swing / slow / static / intraday), the ``layer`` it belongs to, and
its ``source``. A feature that can't state its as-of timestamp is rejected — that is the contract.

``available=False`` marks a SEAM: a charter feature whose data source isn't wired (e.g. dealer GEX from
OPRA, DIX, net-liquidity from FRED, signed tick flow). Seams carry NaN and are surfaced, never faked —
so a downstream model can see the gap instead of training on a fabricated value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# horizon lead-time tags (the first-class config axis the charter mandates)
LEAD_SWING = "swing"        # hours→weeks of lead time (v1)
LEAD_SLOW = "slow"          # months (net liquidity, term structure)
LEAD_STATIC = "static"      # ~constant over the horizon (membership, sector)
LEAD_INTRADAY = "intraday"  # reserved for DAY/SCALP horizons

LAYERS = ("L0_tradeability", "L1_regime", "L2_location", "L3_trigger")


@dataclass(frozen=True)
class Feature:
    """One point-in-time feature value with full provenance."""

    name: str
    value: float
    as_of: pd.Timestamp
    lead_time: str
    layer: str
    source: str
    available: bool = True      # False => SEAM (no data source); value is NaN, surfaced not faked

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer!r}; must be one of {LAYERS}")

    @classmethod
    def seam(cls, name, as_of, lead_time, layer, source) -> "Feature":
        """A charter feature whose data source isn't available — NaN, flagged, never fabricated."""
        return cls(name=name, value=float("nan"), as_of=pd.Timestamp(as_of),
                   lead_time=lead_time, layer=layer, source=f"seam:{source}", available=False)


@dataclass
class FeatureSet:
    """The features computed for one (symbol, as_of) decision point."""

    symbol: str
    as_of: pd.Timestamp
    features: list[Feature] = field(default_factory=list)

    def add(self, f: Feature) -> None:
        if f.as_of > self.as_of:
            raise ValueError(f"LEAKAGE: feature {f.name} as_of {f.as_of} is after decision {self.as_of}")
        self.features.append(f)

    def value(self, name: str) -> float:
        for f in self.features:
            if f.name == name:
                return f.value
        return float("nan")

    def available(self) -> list[Feature]:
        return [f for f in self.features if f.available]

    def seams(self) -> list[Feature]:
        return [f for f in self.features if not f.available]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": self.symbol, "name": f.name, "value": f.value, "as_of": f.as_of,
            "lead_time": f.lead_time, "layer": f.layer, "source": f.source, "available": f.available,
        } for f in self.features])


def winsorized_z(x: pd.Series, clip: float = 4.0) -> pd.Series:
    """Cross-sectional z-score, winsorized — the standard normalization for scoring features."""
    mu, sd = x.mean(), x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return ((x - mu) / sd).clip(-clip, clip)
