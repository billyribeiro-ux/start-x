"""Point-in-time guard: a catalyst stamped AFTER the move must never be attributed to it."""
from __future__ import annotations

import pandas as pd

from startx.events.attribute import attribute_event


def _event(direction="up", date="2024-02-01"):
    return pd.Series({"date": pd.Timestamp(date), "direction": direction,
                      "gap": 0.0, "vol_z": 0.0})


def test_earnings_before_event_is_attributed():
    earnings = pd.DataFrame({
        "epsActual": [2.0, 3.0],
        "epsEstimated": [1.5, 1.5],
        "ts": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-05")],  # one before, one after
    })
    causes, top, conf = attribute_event(_event(), {"earnings": earnings}, lookback_days=2)
    types = {c["type"] for c in causes}
    assert "earnings" in types
    e = next(c for c in causes if c["type"] == "earnings")
    assert e["detail"]["eps_actual"] == 2.0  # the pre-event report, not the future one
    assert e["direction"] == "up" and conf > 0


def test_future_only_catalyst_is_not_attributed():
    earnings = pd.DataFrame({
        "epsActual": [3.0], "epsEstimated": [1.5],
        "ts": [pd.Timestamp("2024-02-05")],  # strictly after the event
    })
    causes, top, conf = attribute_event(_event(), {"earnings": earnings}, lookback_days=2)
    assert all(c["type"] != "earnings" for c in causes)


def test_wrong_way_catalyst_is_discounted():
    # downgrade (down) on an up move should yield low confidence
    grades = pd.DataFrame({
        "action": ["downgrade"], "gradingCompany": ["X"],
        "previousGrade": ["Buy"], "newGrade": ["Hold"],
        "ts": [pd.Timestamp("2024-01-31")],
    })
    _, _, conf = attribute_event(_event("up"), {"grades": grades}, lookback_days=2)
    assert conf < 0.2
