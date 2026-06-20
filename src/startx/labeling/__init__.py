"""Triple-barrier labeling and overlapping-label sample weighting."""
from __future__ import annotations

from .config import HORIZONS, label_horizon
from .triple_barrier import LABEL_COLUMNS, daily_vol, triple_barrier_labels
from .weights import average_uniqueness, num_concurrent_events, sample_weights

__all__ = [
    "HORIZONS",
    "LABEL_COLUMNS",
    "average_uniqueness",
    "daily_vol",
    "label_horizon",
    "num_concurrent_events",
    "sample_weights",
    "triple_barrier_labels",
]
