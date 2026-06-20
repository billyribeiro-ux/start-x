"""Streamlit 'Evidence Miner': across years of big moves, what preceded up vs down?

Pick symbols + a date range; the scanner snapshots the point-in-time conditions at the prior
trading day (``t-1``) for every statistically significant move and ranks, side by side, which
features and catalysts matched the most for up-moves versus down-moves.

Mirrors the cache/context + key-guard pattern in ``dashboard/app.py``; it only wires inputs to
the ``startx.events.evidence`` services so a FastAPI/Svelte frontend could reuse them unchanged.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.events.evidence import evidence_report
from startx.fmp.client import FMPClient
from startx.settings import get_settings

st.set_page_config(page_title="Start-X — Evidence Miner", layout="wide")


@st.cache_resource
def _context():
    settings = get_settings()
    return settings, ParquetCache(settings.cache_dir), load_universe()


@st.cache_data(show_spinner="Mining evidence across moves…")
def _run(symbols: tuple[str, ...], start: str, end: str, ar_threshold: float) -> dict:
    settings, cache, universe = _context()
    with FMPClient(settings) as client:
        return evidence_report(
            list(symbols), start, end, client=client, cache=cache, universe=universe,
            settings=settings, ar_threshold=ar_threshold,
        )


# -- sidebar ---------------------------------------------------------------
settings = get_settings()
if not settings.fmp_api_key:
    st.error("FMP_API_KEY not set. Copy `.env.example` to `.env`, add your key, and reload.")
    st.stop()

universe = load_universe()
st.sidebar.title("Evidence Miner")
st.sidebar.caption("What conditions PRECEDED big up vs down moves, ranked side by side.")
default = [t for t in ("NVDA", "AAPL", "MSFT") if t in universe.tickers] or universe.tickers[:1]
symbols = st.sidebar.multiselect("Symbols", universe.tickers, default=default)
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Start", value=date(2021, 1, 1))
end = c2.date_input("End", value=date(2024, 12, 31))
ar_threshold = st.sidebar.slider("Significance (|abnormal-return z|)", 1.5, 5.0, 2.5, 0.1)
st.sidebar.info(
    "Strict point-in-time: every condition is the feature snapshot from the PRIOR trading day "
    "(t-1), so day-t's own move never explains itself. Ranking shows single-feature AUC "
    "(0.5 = no signal) and lift — 'what matched the most'.")

# -- main ------------------------------------------------------------------
st.title("Evidence Miner — up-move vs down-move conditions")

if not symbols:
    st.warning("Pick at least one symbol in the sidebar.")
    st.stop()

try:
    report = _run(tuple(symbols), start.isoformat(), end.isoformat(), ar_threshold)
except Exception as exc:  # noqa: BLE001 — surface any data/setup error to the UI
    st.error(f"Evidence scan failed: {exc}")
    st.stop()

evidence = report["evidence"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Moves analyzed", report["n_moves"])
m2.metric("Up-moves ↑", report["n_up"])
m3.metric("Down-moves ↓", report["n_down"])
m4.metric("Symbols", len(symbols))

if evidence.empty:
    st.warning("No statistically significant moves in this window. Lower the significance "
               "threshold or widen the date range / symbol set.")
    st.stop()

st.subheader("Ranked PIT conditions — up vs down, side by side")
st.caption("Each condition is the feature value at the prior trading day (t-1). Sorted by how "
           "far single-feature AUC is from 0.5 (strongest separators first). `mean_at_move` is "
           "that direction's mean; `mean_baseline` is the opposite direction's mean.")
left, right = st.columns(2)
with left:
    st.markdown("**🟢 Up-move conditions**")
    st.dataframe(report["ranking"]["up"], use_container_width=True, hide_index=True, height=460)
with right:
    st.markdown("**🔴 Down-move conditions**")
    st.dataframe(report["ranking"]["down"], use_container_width=True, hide_index=True, height=460)

st.subheader("Catalyst match ranking")
st.caption("Which catalyst types were present before up- vs down-moves, and the direction "
           "hit-rate (fraction of moves with this catalyst present that went up).")
st.dataframe(report["catalysts"], use_container_width=True, hide_index=True)

with st.expander("Raw evidence rows (one per move; features snapshotted at t-1)"):
    show = evidence.copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"]).dt.date
    st.dataframe(show, use_container_width=True, hide_index=True)
