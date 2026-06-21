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


# --------------------------------------------------------------------------
# Same-day POST-CLOSE gate: a catalyst stamped after day-t's cash close (16:00) was public
# only AFTER the session it follows; it must never be credited to day-t's move. (The previous
# tests only used midnight timestamps and so could not catch the intraday post-close leak.)
# --------------------------------------------------------------------------
def test_same_day_post_close_news_not_attributed_to_t():
    # 19:00 headline on the move's OWN day must not attach to that day's move.
    news = pd.DataFrame({"title": ["AMC headline"], "ts": [pd.Timestamp("2024-02-01 19:00")]})
    causes, _, conf = attribute_event(_event(date="2024-02-01"), {"news": news}, lookback_days=2)
    assert all(c["type"] != "news" for c in causes)
    assert conf == 0.0


def test_post_close_catalyst_is_attributed_to_next_trading_day():
    # The same 19:00 headline IS available before the next trading day's move (t+1).
    news = pd.DataFrame({"title": ["AMC headline"], "ts": [pd.Timestamp("2024-02-01 19:00")]})
    causes, _, _ = attribute_event(_event(date="2024-02-02"), {"news": news}, lookback_days=2)
    assert "news" in {c["type"] for c in causes}


def test_pre_close_same_day_catalyst_still_attributes_to_t():
    # A 10:00 (pre-close) headline on the move day is genuinely pre-move and must attach to t.
    news = pd.DataFrame({"title": ["Morning headline"], "ts": [pd.Timestamp("2024-02-01 10:00")]})
    causes, _, _ = attribute_event(_event(date="2024-02-01"), {"news": news}, lookback_days=2)
    assert "news" in {c["type"] for c in causes}


def test_post_close_earnings_beat_does_not_inflate_confidence_on_t():
    # The documented exploit: a 19:00 post-close beat inflated confidence 0.0 -> 0.70 on day t.
    earnings = pd.DataFrame({
        "epsActual": [3.0], "epsEstimated": [1.5],
        "ts": [pd.Timestamp("2024-02-01 19:00")],  # after the 16:00 close
    })
    causes, _, conf = attribute_event(_event(date="2024-02-01"), {"earnings": earnings}, lookback_days=2)
    assert all(c["type"] != "earnings" for c in causes)
    assert conf == 0.0
    # ...but it correctly credits the NEXT trading day's move.
    causes_t1, _, conf_t1 = attribute_event(
        _event(date="2024-02-02"), {"earnings": earnings}, lookback_days=2)
    assert "earnings" in {c["type"] for c in causes_t1} and conf_t1 > 0


def test_amc_earnings_endpoint_stamp_lands_on_next_trading_day():
    # The endpoint stamps date-only (AMC/unknown) earnings just past the close, so attribution
    # credits t+1, not the report date t. Build the frame the way endpoints.earnings does.
    from startx.fmp.endpoints import _AMC_STAMP, _to_df

    df = _to_df([{"symbol": "X", "date": "2024-02-01", "epsActual": 3.0, "epsEstimated": 1.5}])
    date = pd.to_datetime(df["date"])
    df["ts"] = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + _AMC_STAMP)
    on_t = attribute_event(_event(date="2024-02-01"), {"earnings": df}, lookback_days=2)[0]
    on_t1 = attribute_event(_event(date="2024-02-02"), {"earnings": df}, lookback_days=2)[0]
    assert all(c["type"] != "earnings" for c in on_t)        # not on report day t
    assert "earnings" in {c["type"] for c in on_t1}          # on t+1


def test_friday_catalyst_attributes_to_monday_move():
    # Friday (2024-02-02) pre-close news, Monday (2024-02-05) move: 3 calendar days but only
    # 1 trading day back. The trailing-TRADING-day lookback must keep it in window.
    news = pd.DataFrame({"title": ["Friday news"], "ts": [pd.Timestamp("2024-02-02 10:00")]})
    causes, _, _ = attribute_event(_event(date="2024-02-05"), {"news": news}, lookback_days=2)
    assert "news" in {c["type"] for c in causes}
