"""Streamlit 'Reversal Lab': dissect every trend reversal and learn statistical exit levels.

For a symbol + date range we detect ATR-zigzag swings, build directional legs, mark each
reversal on the price chart, and for every reversal expose its daily anatomy and intraday
microstructure (POC / value area / VWAP / RVOL / cumulative delta). The learned exit-
distribution table answers "how far / how long does a leg typically run before it flips?".

Mirrors the cache/context pattern in ``dashboard/app.py``; it only wires inputs to the
``startx.events.reversals`` services so a FastAPI/Svelte frontend could reuse them unchanged.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.events.reversals import reversal_report
from startx.fmp.client import FMPClient
from startx.settings import get_settings

st.set_page_config(page_title="Start-X — Reversal Lab", layout="wide")


@st.cache_resource
def _context():
    settings = get_settings()
    return settings, ParquetCache(settings.cache_dir), load_universe()


@st.cache_data(show_spinner="Detecting reversals & dissecting microstructure…")
def _run(ticker: str, start: str, end: str, intraday: bool, refresh: bool) -> dict:
    settings, cache, universe = _context()
    with FMPClient(settings) as client:
        return reversal_report(
            ticker, start, end, client=client, cache=cache, universe=universe,
            settings=settings, intraday=intraday, refresh=refresh,
        )


def _price_chart(report: dict) -> go.Figure:
    p = report["prices"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=p["date"], open=p["open"], high=p["high"], low=p["low"], close=p["close"],
        name=report["ticker"], showlegend=False))
    legs = report["legs"]
    if not legs.empty:
        # Draw each leg as a line from start swing to reversal, then mark the reversal.
        for leg in legs.itertuples(index=False):
            color = "#16a34a" if leg.direction == "up" else "#dc2626"
            fig.add_trace(go.Scatter(
                x=[leg.start_date, leg.end_date],
                y=[_close_at(p, leg.start_date), _close_at(p, leg.end_date)],
                mode="lines", line=dict(color=color, width=1, dash="dot"),
                showlegend=False, hoverinfo="skip"))
        highs = legs[legs["direction"] == "up"]   # up-leg ends => swing-high reversal
        lows = legs[legs["direction"] == "down"]  # down-leg ends => swing-low reversal
        if not highs.empty:
            fig.add_trace(go.Scatter(
                x=highs["end_date"], y=[_close_at(p, d) for d in highs["end_date"]],
                mode="markers", name="up→down reversal",
                marker=dict(color="#dc2626", size=12, symbol="triangle-down",
                            line=dict(width=1, color="white"))))
        if not lows.empty:
            fig.add_trace(go.Scatter(
                x=lows["end_date"], y=[_close_at(p, d) for d in lows["end_date"]],
                mode="markers", name="down→up reversal",
                marker=dict(color="#16a34a", size=12, symbol="triangle-up",
                            line=dict(width=1, color="white"))))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False)
    return fig


def _close_at(prices: pd.DataFrame, when) -> float:
    sel = prices[prices["date"] == pd.Timestamp(when)]
    return float(sel["close"].iloc[0]) if not sel.empty else float("nan")


def _fmt(x, pct=False, dp=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dp}f}{'%' if pct else ''}"


def _exit_table(exit_stats: dict) -> pd.DataFrame:
    rows = []
    for direction in ("up", "down"):
        d = exit_stats.get(direction, {})
        rows.append({
            "leg direction": direction,
            "n": d.get("n", 0),
            "bars p50": _fmt(d.get("bars_p50"), dp=0),
            "bars p75": _fmt(d.get("bars_p75"), dp=0),
            "bars p90": _fmt(d.get("bars_p90"), dp=0),
            "|move| % p50": _fmt(d.get("magnitude_pct_p50")),
            "|move| % p75": _fmt(d.get("magnitude_pct_p75")),
            "|move| % p90": _fmt(d.get("magnitude_pct_p90")),
            "ATR-ext p50": _fmt(d.get("atr_extension_p50")),
            "ATR-ext p75": _fmt(d.get("atr_extension_p75")),
            "ATR-ext p90": _fmt(d.get("atr_extension_p90")),
        })
    return pd.DataFrame(rows)


# -- sidebar ---------------------------------------------------------------
settings = get_settings()
if not settings.fmp_api_key:
    st.error("FMP_API_KEY not set. Copy `.env.example` to `.env`, add your key, and reload.")
    st.stop()

universe = load_universe()
st.sidebar.title("Reversal Lab")
st.sidebar.caption("Dissect every trend reversal; learn statistical exit levels.")
ticker = st.sidebar.selectbox("Symbol", universe.tickers, index=universe.tickers.index("NVDA")
                              if "NVDA" in universe.tickers else 0)
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Start", value=date(2024, 1, 1))
end = c2.date_input("End", value=date(2024, 9, 30))
intraday = st.sidebar.checkbox("Pull intraday microstructure (1-min)", value=True)
refresh = st.sidebar.checkbox("Force refresh from FMP", value=False)
st.sidebar.info(
    "Exit percentiles describe *historical* leg duration/size, not a guarantee. p75/p90 are "
    "where trends statistically start to exhaust — 'be ready to get out' zones, not signals.")

# -- main ------------------------------------------------------------------
spec = universe.spec(ticker)
st.title(f"{ticker} — {spec.name} · Reversal Lab")

try:
    report = _run(ticker, start.isoformat(), end.isoformat(), intraday, refresh)
except Exception as exc:  # noqa: BLE001 — surface any data/setup error to the UI
    st.error(f"Reversal scan failed: {exc}")
    st.stop()

reversals = report["reversals"]
legs = report["legs"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reversals", len(reversals))
m2.metric("up→down", int((legs["direction"] == "up").sum()) if not legs.empty else 0)
m3.metric("down→up", int((legs["direction"] == "down").sum()) if not legs.empty else 0)
m4.metric("Swings found", len(report["swings"]))

st.plotly_chart(_price_chart(report), use_container_width=True)

if legs.empty:
    st.warning("No confirmed reversals in this window. Widen the date range — the ATR-zigzag "
               "only confirms a swing once price retraces 3×ATR from the extreme.")
    st.stop()

st.subheader("Trend legs")
st.caption("Each leg runs from one swing to the opposite swing; `end_date` is the reversal.")
legs_show = legs.copy()
legs_show["start_date"] = legs_show["start_date"].dt.date
legs_show["end_date"] = legs_show["end_date"].dt.date
legs_show = legs_show.rename(columns={
    "magnitude_pct": "magnitude %", "atr_extension": "ATR-ext",
    "slope_pct_per_bar": "slope %/bar", "mfe_pct": "MFE %", "mae_pct": "MAE %"})
st.dataframe(legs_show, use_container_width=True, hide_index=True)

st.subheader("Learned exit distribution — 'when to be ready to get out'")
st.caption("Percentiles of leg duration, |move| and ATR-extension, split by leg direction.")
st.dataframe(_exit_table(report["exit_stats"]), use_container_width=True, hide_index=True)

st.subheader("Reversal-by-reversal microstructure")
for rev in sorted(reversals, key=lambda r: r["reversal_date"], reverse=True):
    leg = rev["leg"]
    arrow = "🔴▼ up→down" if rev["reversal_kind"] == "high" else "🟢▲ down→up"
    with st.expander(
        f"{arrow}  ·  {pd.Timestamp(rev['reversal_date']).date()}  ·  "
        f"prior leg {leg['direction']} {_fmt(leg['magnitude_pct'], pct=True)} over "
        f"{leg['bars']} bars  ·  ATR-ext {_fmt(leg['atr_extension'])}"):
        daily = rev.get("daily", {})
        left, right = st.columns(2)
        with left:
            st.markdown("**Daily anatomy**")
            st.write({
                "candle": daily.get("candle"),
                "ar_z": _fmt(daily.get("ar_z")),
                "RVOL(20)": _fmt(daily.get("rvol_20")),
                "range %": _fmt(daily.get("range_pct")),
                "ATR(14)": _fmt(daily.get("atr_14")),
                "ATR expansion ×": _fmt(daily.get("atr_expansion")),
                "gap %": _fmt(daily.get("gap_pct")),
                "close-location-value": _fmt(daily.get("close_location_value")),
            })
        with right:
            st.markdown("**Intraday microstructure**")
            intr = rev.get("intraday")
            if not intr:
                st.caption("No intraday bars (intraday off or unavailable for this day).")
            else:
                st.write({
                    "POC": _fmt(intr.get("poc")),
                    "VAH / VAL": f"{_fmt(intr.get('vah'))} / {_fmt(intr.get('val'))}",
                    "session VWAP": _fmt(intr.get("session_vwap")),
                    "close vs POC %": _fmt(intr.get("reversal_vs_poc"), pct=True),
                    "close vs VWAP %": _fmt(intr.get("reversal_vs_vwap"), pct=True),
                    "cumulative delta": _fmt(intr.get("cumulative_delta"), dp=0),
                    "opening range H/L":
                        f"{_fmt(intr.get('or_high'))} / {_fmt(intr.get('or_low'))}",
                })
