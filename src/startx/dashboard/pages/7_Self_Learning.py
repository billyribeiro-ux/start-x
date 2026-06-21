"""Streamlit 'Self-Learning': the loss-autopsy + meta-label gate, shown end to end.

It runs the real :mod:`startx.learning` services on a trade ledger and renders three things:

  1. the win/loss SIGNATURE — which entry-day forensic features most separate winners from
     losers (the automated "why losses happen");
  2. FILTERED-vs-UNFILTERED trade economics side by side — the loss-avoidance value of the
     strictly point-in-time meta gate;
  3. the DRIFT chart — rolling out-of-sample meta accuracy, the self-adjusting retrain trigger.

Mirrors the cache/context + key-guard pattern in ``dashboard/app.py`` (this file only wires
inputs to the ``startx.learning`` services, so a FastAPI/Svelte frontend could reuse them). When
no live trade ledger is available it falls back to a deterministic synthetic ledger so the page
is always demonstrable offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from startx.learning import (
    autopsy_trades,
    drift_monitor,
    run_self_learning,
    walk_forward_metalabel,
)
from startx.learning.autopsy import forensic_feature_columns
from startx.models.train import model_factory
from startx.settings import get_settings

st.set_page_config(page_title="Start-X — Self-Learning", layout="wide")


@st.cache_data(show_spinner="Building synthetic trade ledger…")
def _synthetic_autopsy(n: int, edge: float, shuffle: bool, seed: int) -> pd.DataFrame:
    """A deterministic synthetic autopsy frame where one forensic feature predicts wins.

    ``edge`` controls how cleanly ``rvol_20`` separates winners from losers (0 == no signal).
    With ``shuffle`` the labels are permuted so the meta-label has nothing to learn (control).
    Returns a frame already shaped like :func:`autopsy_trades` output (forensics + ``is_win``).
    """
    rng = np.random.default_rng(seed)
    entry = pd.bdate_range("2021-01-04", periods=n)
    # A separating forensic feature + several noise features.
    signal = rng.normal(0.0, 1.0, n)
    prob_win = 1.0 / (1.0 + np.exp(-edge * signal))
    is_win = (rng.uniform(size=n) < prob_win).astype(int)
    if shuffle:
        is_win = rng.permutation(is_win)
    # Returns consistent with the label: wins positive, losses negative (with noise).
    ret = np.where(is_win == 1, rng.uniform(0.005, 0.05, n), -rng.uniform(0.005, 0.05, n))
    return pd.DataFrame({
        "symbol": "SYN",
        "entry_date": entry,
        "exit_date": entry + pd.Timedelta(days=1),
        "t1": entry + pd.Timedelta(days=1),
        "ret_net": ret,
        "rvol_20": signal,
        "atr_expansion": rng.normal(1.0, 0.2, n),
        "gap_pct": rng.normal(0.0, 1.0, n),
        "ar_z": rng.normal(0.0, 1.0, n),
        "is_win": is_win,
    })


def _metrics_table(unfiltered: dict, filtered: dict) -> pd.DataFrame:
    rows = []
    for key in ("n_trades", "win_rate", "expectancy", "profit_factor", "avg_ret", "total_return"):
        rows.append({"metric": key, "unfiltered": unfiltered.get(key),
                     "filtered (meta-gated)": filtered.get(key)})
    return pd.DataFrame(rows)


def _drift_chart(drift: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(drift.index), y=drift.to_numpy(), mode="lines",
                             name="rolling OOS accuracy", line=dict(color="#2563eb")))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="chance (retrain below)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="entry time", yaxis_title="rolling meta accuracy")
    return fig


# -- sidebar ---------------------------------------------------------------
settings = get_settings()  # noqa: F841 — keeps context parity with the other pages
st.sidebar.title("Self-Learning")
st.sidebar.caption("Learn the fingerprint of losing trades, then auto-skip setups that match.")
st.sidebar.info(
    "STRICT point-in-time: the meta-model deciding take/skip for a trade trains ONLY on trades "
    "that fully RESOLVED before that trade's entry (t1 <= entry). Forensic features use entry-day "
    "data only — never the trade's exit or outcome.")
n = st.sidebar.slider("Synthetic trades", 200, 1500, 600, 50)
edge = st.sidebar.slider("Loser-fingerprint strength", 0.0, 3.0, 1.5, 0.1)
shuffle = st.sidebar.checkbox("Shuffle labels (control: no learnable signal)", value=False)
threshold = st.sidebar.slider("Meta take threshold (P(win) ≥)", 0.3, 0.8, 0.5, 0.05)
train_min = st.sidebar.slider("Min resolved trades before scoring", 20, 200, 50, 10)
test_span = st.sidebar.slider("Forward block size", 5, 60, 20, 5)

# -- main ------------------------------------------------------------------
st.title("Self-Learning — loss autopsy + point-in-time meta gate")
st.warning(
    "**SYNTHETIC-DATA DEMO.** Every number on this page is computed from a deterministic, "
    "randomly generated trade ledger (`_synthetic_autopsy`), controlled by the sidebar sliders — "
    "**not** live or backtested results. The `startx.learning` analytics are real; only the input "
    "ledger is fake. Swap in `autopsy_trades(live_trades, prices_by)` to run it on a real book.",
    icon="⚠️",
)
st.caption("Autopsy services come from `startx.learning`; this page only wires inputs to them. "
           "Replace the synthetic ledger with `autopsy_trades(live_trades, prices_by)` for live "
           "books — the analytics are identical.")

autopsy = _synthetic_autopsy(n, edge, shuffle, seed=7)
feature_cols = [c for c in forensic_feature_columns(autopsy)]

try:
    result = run_self_learning(
        autopsy, feature_cols, threshold=threshold, train_min=train_min, test_span=test_span,
    )
except Exception as exc:  # noqa: BLE001 — surface any setup error to the UI
    st.error(f"Self-learning run failed: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Meta OOS AUC", f"{result.meta_auc:.3f}" if result.meta_auc == result.meta_auc else "—")
m2.metric("Trades kept", result.n_kept)
m3.metric("Trades dropped", result.n_dropped)
wr_lift = (result.filtered.get("win_rate", float("nan"))
           - result.unfiltered.get("win_rate", float("nan")))
m4.metric("Win-rate lift", f"{wr_lift:+.1%}" if wr_lift == wr_lift else "—")

st.subheader("Why losses happen — win/loss signature")
st.caption("Each entry-day forensic feature ranked by how strongly it separates winners from "
           "losers (single-feature AUC; 0.5 = no signal). The strongest separators are the "
           "fingerprint the meta-gate keys on.")
st.dataframe(result.signature, use_container_width=True, hide_index=True)

st.subheader("Filtered vs unfiltered trade economics")
st.caption("Unfiltered = take every predicted trade. Filtered = take only meta-approved trades. "
           "A higher filtered win-rate / expectancy / profit factor is the loss-avoidance value.")
st.dataframe(_metrics_table(result.unfiltered, result.filtered),
             use_container_width=True, hide_index=True)

st.subheader("Drift monitor — rolling OOS meta accuracy")
st.caption("Rolling fraction of recent take/skip calls the meta-label got right. A sustained "
           "drop toward / below 0.5 flags regime decay — the cue to retrain.")
preds = walk_forward_metalabel(
    autopsy[feature_cols].apply(pd.to_numeric, errors="coerce"),
    pd.to_numeric(autopsy["is_win"], errors="coerce"),
    pd.to_datetime(autopsy["entry_date"]),
    pd.to_datetime(autopsy["t1"]),
    model_factory(),
    train_min=train_min,
    test_span=test_span,
)
drift = drift_monitor(preds, window=max(20, train_min // 2))
if drift.empty:
    st.info("Not enough resolved history to score any forward block — raise the trade count or "
            "lower the minimum.")
else:
    st.plotly_chart(_drift_chart(drift), use_container_width=True)
