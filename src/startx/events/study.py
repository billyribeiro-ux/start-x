"""Classic event study: average abnormal-return profile around detected events."""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_study_caar(
    signals: pd.DataFrame, events: pd.DataFrame, pre: int = 5, post: int = 10
) -> pd.DataFrame:
    """Average abnormal return (AAR) and cumulative AAR (CAAR) by relative trading day.

    Returned tidy frame has columns: ``group`` (all/up/down), ``rel_day``, ``aar``,
    ``caar``, ``n``. AAR is in percent. This describes the *typical* move shape; it is an
    analysis output, not a predictive feature.
    """
    empty = pd.DataFrame(columns=["group", "rel_day", "aar", "caar", "n"])
    if signals.empty or events.empty:
        return empty

    s = signals.reset_index(drop=True)
    pos_by_date = {d: i for i, d in enumerate(s["date"])}
    ar = s["ar_pct"].to_numpy()
    rel = np.arange(-pre, post + 1)

    def _agg(ev: pd.DataFrame, group: str) -> pd.DataFrame | None:
        mat = []
        for d in ev["date"]:
            i = pos_by_date.get(d)
            if i is None or i - pre < 0 or i + post >= len(s):
                continue
            mat.append(ar[i - pre : i + post + 1])
        if not mat:
            return None
        arr = np.vstack(mat)
        aar = np.nanmean(arr, axis=0)
        return pd.DataFrame(
            {"group": group, "rel_day": rel, "aar": aar, "caar": np.nancumsum(aar),
             "n": arr.shape[0]}
        )

    frames = [
        f
        for f in (
            _agg(events, "all"),
            _agg(events[events["direction"] == "up"], "up"),
            _agg(events[events["direction"] == "down"], "down"),
        )
        if f is not None
    ]
    return pd.concat(frames, ignore_index=True) if frames else empty
