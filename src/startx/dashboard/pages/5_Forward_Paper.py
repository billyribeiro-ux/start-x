"""Streamlit 'Forward / Paper Test': train on the past, paper-trade forward, watch the blotter.

Pick symbols, a training window (closed, past) and a forward window (out-of-sample). The model
is fit ONCE on the training window and never sees the forward data; it then paper-trades the
forward window, taking each signal at the next bar's open and exiting at the triple-barrier
``t1``. The page renders OPEN positions (live mark-to-market), the CLOSED-trade blotter, the
equity curve and the headline summary.

Mirrors the cache/context + key-guard pattern in ``dashboard/app.py``; it only wires inputs to
``startx.forward.paper`` services, so a FastAPI/Svelte frontend could reuse them unchanged.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from startx.backtest.costs import CostModel
from startx.data.cache import ParquetCache
from startx.data.universe import load_universe
from startx.fmp.client import FMPClient
from startx.forward.paper import forward_test
from startx.settings import get_settings

st.set_page_config(page_title="Start-X — Forward / Paper Test", layout="wide")


@st.cache_resource
def _context():
    settings = get_settings()
    return settings, ParquetCache(settings.cache_dir), load_universe()


@st.cache_data(show_spinner="Training on the past & paper-trading forward…")
def _run(
    tickers: tuple[str, ...],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: str,
    long_th: float,
    short_th: float,
    capital: float,
    commission_bps: float,
    slippage_bps: float,
    one_at_a_time: bool,
    tag_reason: bool,
) -> dict:
    settings, cache, universe = _context()
    with FMPClient(settings) as client:
        return forward_test(
            list(tickers), train_start, train_end, test_start, test_end,
            client=client, cache=cache, universe=universe, settings=settings,
            horizon=horizon, long_th=long_th, short_th=short_th,
            capital_per_trade=capital, costs=CostModel(commission_bps, slippage_bps),
            one_at_a_time=one_at_a_time, tag_reason=tag_reason,
        )


def _equity_chart(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.to_numpy(), mode="lines",
                             name="cumulative P&L ($)", line=dict(color="#2563eb")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="date", yaxis_title="cumulative paper P&L ($)")
    return fig


def _fmt_blotter(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy a blotter frame for display (dates -> date, pct as %, cash with sign)."""
    if df.empty:
        return df
    out = df.copy()
    for col in ("entry_date", "exit_date"):
        out[col] = pd.to_datetime(out[col]).dt.date
    out["pnl_%"] = (out["pnl_pct"] * 100).round(2)
    out["pnl_$"] = out["pnl_cash"].round(2)
    out["prob_up"] = out["prob_up"].round(3)
    cols = ["symbol", "side", "trade_type", "entry_date", "entry_price", "exit_date",
            "exit_price", "pnl_%", "pnl_$", "bars_held", "prob_up", "status"]
    return out[[c for c in cols if c in out.columns]]


# -- sidebar ---------------------------------------------------------------
settings = get_settings()
if not settings.fmp_api_key:
    st.error("FMP_API_KEY not set. Copy `.env.example` to `.env`, add your key, and reload.")
    st.stop()

universe = load_universe()
st.sidebar.title("Forward / Paper Test")
st.sidebar.caption("Train on the past, paper-trade forward. Strict no-lookahead.")

default_syms = universe.tickers[: min(3, len(universe.tickers))]
symbols = st.sidebar.multiselect("Symbols", universe.tickers, default=default_syms)

st.sidebar.markdown("**Training window (past, closed)**")
t1, t2 = st.sidebar.columns(2)
train_start = t1.date_input("Train start", value=date(2022, 1, 1))
train_end = t2.date_input("Train end", value=date(2023, 12, 31))

st.sidebar.markdown("**Forward window (out-of-sample)**")
f1, f2 = st.sidebar.columns(2)
test_start = f1.date_input("Forward start", value=date(2024, 1, 1))
test_end = f2.date_input("Forward end", value=date(2024, 6, 30))

horizon = st.sidebar.selectbox("Label horizon", ["short", "long"], index=0)
c1, c2 = st.sidebar.columns(2)
long_th = c1.slider("Long thresh (P↑ ≥)", 0.5, 0.95, 0.6, 0.01)
short_th = c2.slider("Short thresh (P↑ ≤)", 0.05, 0.5, 0.4, 0.01)
capital = st.sidebar.number_input("Capital per trade ($)", 1_000.0, 1_000_000.0, 10_000.0,
                                  step=1_000.0)
cc1, cc2 = st.sidebar.columns(2)
commission_bps = cc1.number_input("Commission (bps)", 0.0, 50.0, 1.0, 0.5)
slippage_bps = cc2.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 0.5)
one_at_a_time = st.sidebar.checkbox("One position at a time / symbol", value=True)
tag_reason = st.sidebar.checkbox("Tag each trade's catalyst (slower)", value=False)

st.sidebar.info(
    "The model is fit ONCE on the training window and never sees the forward data. "
    "Signals fill at the NEXT bar's open and exit at the triple-barrier t1 — no lookahead.")

# -- main ------------------------------------------------------------------
st.title("Forward / Paper-Trading Blotter")

st.warning(
    "Research / illustrative only — NOT the validated production system. This page trades the "
    "ABANDONED `prob_up` directional model (out-of-sample AUC ~0.50) with no meta-label gate; "
    "the validated system is the three-book swing engine in `scripts/run_book.py`.")

if not symbols:
    st.warning("Pick at least one symbol in the sidebar.")
    st.stop()
if not (train_end < test_start):
    st.error("Training window must end strictly before the forward window starts (no overlap).")
    st.stop()

try:
    res = _run(
        tuple(symbols), train_start.isoformat(), train_end.isoformat(),
        test_start.isoformat(), test_end.isoformat(), horizon, long_th, short_th,
        capital, commission_bps, slippage_bps, one_at_a_time, tag_reason,
    )
except Exception as exc:  # noqa: BLE001 — surface any data/setup error to the UI.
    st.error(f"Forward test failed: {exc}")
    st.stop()

blotter = res["blotter"]
open_positions = res["open_positions"]
summary = res["summary"]


def _metric(x, pct: bool = False, money: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if money:
        return f"${x:,.0f}"
    if pct:
        return f"{x:.1%}"
    return f"{x:.2f}"


m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Trades", summary["n_trades"])
m2.metric("Open now", summary["n_open"])
m3.metric("Hit rate", _metric(summary["hit_rate"], pct=True))
m4.metric("Total P&L", _metric(summary["total_pnl_cash"], money=True))
m5.metric("Total return", _metric(summary["total_return"], pct=True))

m6, m7, m8, m9 = st.columns(4)
m6.metric("Sharpe", _metric(summary["sharpe"]))
m7.metric("Max drawdown", _metric(summary["max_drawdown"], pct=True))
m8.metric("Profit factor", _metric(summary["profit_factor"]))
m9.metric("Avg bars held", _metric(summary["avg_bars_held"]))

if blotter.empty:
    st.warning("No signals fired in this window. Loosen the thresholds or widen the dates.")
    st.stop()

st.subheader("Equity curve (cumulative paper P&L)")
st.plotly_chart(_equity_chart(res["equity"]), use_container_width=True)

st.subheader("Open positions — live mark-to-market")
if open_positions.empty:
    st.caption("Flat — no positions still open at the end of the forward window.")
else:
    st.dataframe(_fmt_blotter(open_positions), use_container_width=True, hide_index=True)

st.subheader("Closed trades — blotter")
closed = blotter[blotter["status"] == "closed"]
if closed.empty:
    st.caption("No trades have closed yet in this window.")
else:
    st.dataframe(_fmt_blotter(closed.sort_values("entry_date", ascending=False)),
                 use_container_width=True, hide_index=True, height=460)
