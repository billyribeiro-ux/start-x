"""Preset triple-barrier horizons, one per book horizon (see CLAUDE.md / STRATEGY.md)."""
from __future__ import annotations

import pandas as pd

from .triple_barrier import triple_barrier_labels

#: Named barrier presets, aligned to the three production books:
#:   ``short``    — 1-10 day swing (short_swing book; 10-day hard cap).
#:   ``long``     — weeks to ~3 months (long_swing book; 63-day cap).
#:   ``position`` — long hold, months-to-years (position book; 504-day backstop). The labeling
#:                  vertical barrier is a practical ~1yr (252d); the book itself rides to 504.
HORIZONS: dict[str, dict[str, float | int]] = {
    "short": {"horizon_days": 10, "pt_mult": 1.5, "sl_mult": 1.5, "vol_span": 21},
    "long": {"horizon_days": 63, "pt_mult": 2.5, "sl_mult": 2.5, "vol_span": 63},
    "position": {"horizon_days": 252, "pt_mult": 3.0, "sl_mult": 3.0, "vol_span": 63},
}


def label_horizon(prices: pd.DataFrame, name: str) -> pd.DataFrame:
    """Run :func:`triple_barrier_labels` with the preset named ``name``.

    ``name`` must be a key of :data:`HORIZONS` (``"short"`` or ``"long"``).
    """
    if name not in HORIZONS:
        raise KeyError(f"unknown horizon {name!r}; choose from {sorted(HORIZONS)}")
    params = HORIZONS[name]
    return triple_barrier_labels(
        prices,
        horizon_days=int(params["horizon_days"]),
        pt_mult=float(params["pt_mult"]),
        sl_mult=float(params["sl_mult"]),
        vol_span=int(params["vol_span"]),
    )
