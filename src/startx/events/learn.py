"""First self-learning layer: learn which catalysts reliably precede the biggest moves.

This is descriptive learning (empirical base rates over the scanned window), the seed for the
later ML layer. For each catalyst type it answers: how often does it show up around big
moves, how large are those moves on average, and how often does the catalyst point the same
way the stock actually went (direction hit-rate).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

_COLUMNS = ["catalyst", "n_events", "pct_of_events", "avg_abs_move_pct",
            "dir_hit_rate", "avg_confidence"]


def catalyst_reliability(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "causes" not in events.columns:
        return pd.DataFrame(columns=_COLUMNS)

    total = len(events)
    stats: dict[str, list] = defaultdict(list)  # type -> list of (abs_move, match, conf)
    for _, ev in events.iterrows():
        seen: dict[str, dict] = {}
        for c in ev["causes"]:
            seen.setdefault(c["type"], c)  # causes are pre-sorted by effective score
        for ctype, c in seen.items():
            match = c["direction"] == ev["direction"] or c["direction"] == "neutral"
            stats[ctype].append((ev["abs_ar_pct"], match, ev["confidence"]))

    rows = []
    for ctype, lst in stats.items():
        moves = [a for a, _, _ in lst]
        rows.append(
            {
                "catalyst": ctype,
                "n_events": len(lst),
                "pct_of_events": round(len(lst) / total, 3),
                "avg_abs_move_pct": round(float(np.mean(moves)), 2),
                "dir_hit_rate": round(float(np.mean([m for _, m, _ in lst])), 3),
                "avg_confidence": round(float(np.mean([c for _, _, c in lst])), 3),
            }
        )
    out = pd.DataFrame(rows, columns=_COLUMNS)
    return out.sort_values("avg_abs_move_pct", ascending=False).reset_index(drop=True)
