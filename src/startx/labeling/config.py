"""Preset triple-barrier horizons for SHORT- and LONG-swing labeling."""
from __future__ import annotations

import pandas as pd

from .triple_barrier import triple_barrier_labels

#: Named barrier presets. ``short`` is a tight ~2-week swing; ``long`` a ~3-month swing.
HORIZONS: dict[str, dict[str, float | int]] = {
    "short": {"horizon_days": 10, "pt_mult": 1.5, "sl_mult": 1.5, "vol_span": 21},
    "long": {"horizon_days": 60, "pt_mult": 2.5, "sl_mult": 2.5, "vol_span": 63},
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
