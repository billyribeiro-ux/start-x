"""Streamlit 'Move Explorer': scan a date range and see why each big move happened.

Run with:  streamlit run src/startx/dashboard/app.py
Analytics live in startx.* services; this file only wires inputs and renders outputs, so the
same services can back a FastAPI + Svelte5 frontend later without change.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.events.engine import EngineResult, analyze_symbol
from startx.fmp.client import FMPClient
from startx.settings import get_settings

st.set_page_config(page_title="Start-X — Move Explorer", layout="wide")


@st.cache_resource
def _context():
    settings = get_settings()
    return settings, ParquetCache(settings.cache_dir), load_universe()


@st.cache_data(show_spinner="Scanning & attributing moves…")
def _run(ticker, start, end, ar_threshold, vol_threshold, lookback, refresh) -> EngineResult:
    settings, cache, universe = _context()
    with FMPClient(settings) as client:
        return analyze_symbol(
            ticker, start, end, client=client, cache=cache, universe=universe,
            settings=settings, ar_threshold=ar_threshold, vol_threshold=vol_threshold,
            lookback_days=lookback, refresh=refresh,
        )


def _price_chart(res: EngineResult) -> go.Figure:
    p, ev = res.prices, res.events
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=p["date"], open=p["open"], high=p["high"], low=p["low"], close=p["close"],
        name=res.ticker, showlegend=False))
    for direction, color in (("up", "#16a34a"), ("down", "#dc2626")):
        sel = ev[ev["direction"] == direction]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["date"], y=sel["close"], mode="markers",
                marker=dict(color=color, size=11, symbol="triangle-up" if direction == "up"
                            else "triangle-down", line=dict(width=1, color="white")),
                name=f"{direction} events",
                text=[f"{c} · conf {cf:.0%}" for c, cf in zip(sel["top_cause"], sel["confidence"])],
                hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False)
    return fig


def _caar_chart(caar: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for grp, color in (("up", "#16a34a"), ("down", "#dc2626"), ("all", "#2563eb")):
        sub = caar[caar["group"] == grp]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["rel_day"], y=sub["caar"], mode="lines+markers",
                                     name=f"{grp} (n={int(sub['n'].iloc[0])})",
                                     line=dict(color=color)))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="trading days from event", yaxis_title="cumulative AAR (%)")
    return fig


# -- sidebar ---------------------------------------------------------------
settings = get_settings()
if not settings.fmp_api_key:
    st.error("FMP_API_KEY not set. Copy `.env.example` to `.env`, add your key, and reload.")
    st.stop()

universe = load_universe()
st.sidebar.title("Start-X")
st.sidebar.caption("Why did it move? Causal attribution + event study.")
ticker = st.sidebar.selectbox("Symbol", universe.tickers, index=0)
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Start", value=date(2024, 1, 1))
end = c2.date_input("End", value=date(2024, 12, 31))
ar_threshold = st.sidebar.slider("Significance (|abnormal-return z|)", 1.5, 5.0, 2.5, 0.1)
use_vol = st.sidebar.checkbox("Require volume surge too", value=False)
vol_threshold = st.sidebar.slider("Volume z ≥", 0.0, 4.0, 2.0, 0.5) if use_vol else None
lookback = st.sidebar.slider("Catalyst lookback (days)", 0, 5, 2)
refresh = st.sidebar.checkbox("Force refresh from FMP", value=False)

st.sidebar.info(
    "Headline metric is **out-of-sample robustness**, never in-sample win rate. "
    "A backtest that looks ~100% perfect on history is an overfitting red flag.")

# -- main ------------------------------------------------------------------
spec = universe.spec(ticker)
st.title(f"{ticker} — {spec.name}")

try:
    res = _run(ticker, start.isoformat(), end.isoformat(), ar_threshold, vol_threshold,
               lookback, refresh)
except Exception as exc:  # noqa: BLE001 — surface any data/setup error to the UI
    st.error(f"Scan failed: {exc}")
    st.stop()

ev = res.events
m1, m2, m3, m4 = st.columns(4)
m1.metric("Significant moves", len(ev))
m2.metric("Spikes ↑", int((ev["direction"] == "up").sum()) if not ev.empty else 0)
m3.metric("Sell-offs ↓", int((ev["direction"] == "down").sum()) if not ev.empty else 0)
m4.metric("Attributed", f"{(ev['confidence'] > 0).mean():.0%}" if not ev.empty else "—")

st.plotly_chart(_price_chart(res), use_container_width=True)

if ev.empty:
    st.warning("No statistically significant moves in this window. Lower the significance "
               "threshold or widen the date range.")
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.subheader("Events")
    show = ev[["date", "direction", "ret_pct", "ar_z", "vol_z", "top_cause", "confidence"]].copy()
    show["date"] = show["date"].dt.date
    show = show.rename(columns={"ret_pct": "return %", "top_cause": "primary cause"})
    st.dataframe(show.sort_values("date", ascending=False), use_container_width=True,
                 hide_index=True, height=420)
with right:
    st.subheader("Learned catalyst reliability")
    st.caption("Which catalysts accompany the biggest moves, and how often they point the "
               "right way (direction hit-rate). The seed of the ML layer.")
    st.dataframe(res.reliability, use_container_width=True, hide_index=True)

st.subheader("Average move profile (event study)")
st.plotly_chart(_caar_chart(res.caar), use_container_width=True)

st.subheader("Move-by-move breakdown")
for _, row in ev.sort_values("date", ascending=False).iterrows():
    arrow = "🟢▲" if row["direction"] == "up" else "🔴▼"
    with st.expander(
        f"{arrow}  {row['date'].date()}  ·  {row['ret_pct']:+.2f}%  "
        f"(z={row['ar_z']:+.1f})  ·  {row['top_cause']}  ·  conf {row['confidence']:.0%}"):
        if not row["causes"]:
            st.write("No catalyst found in the lookback window — endogenous / flow-driven move.")
        for c in row["causes"]:
            tag = {"up": "↑", "down": "↓", "neutral": "•"}.get(c["direction"], "•")
            st.markdown(f"**{tag} {c['type']}** — {c['label']}  _(score {c['score']})_")
            if c["type"] == "news" and c["detail"].get("sample"):
                for t in c["detail"]["sample"]:
                    st.caption(f"• {t}")
