"""Intermarket feature tests: PIT (no-lookahead), RS sign, yield change, shape.

All synthetic and fast; no network. One optional network smoke test builds a tiny real
context + feature matrix and is skipped when no FMP key is configured.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from startx.features.intermarket import intermarket_features

_TRADING = pd.bdate_range("2021-01-04", periods=400)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _frame(values, dates=_TRADING) -> pd.DataFrame:
    """Build a date/close frame from a per-day value array (or scalar drift series)."""
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"date": dates[: len(values)], "close": values})


def _drift(rate: float, n: int = 400, start: float = 100.0) -> pd.DataFrame:
    """A smooth compounding series (no noise -> deterministic, easy to reason about)."""
    close = start * np.exp(rate * np.arange(n))
    return _frame(close)


def _yields(level: float, slope: float, n: int = 400) -> pd.DataFrame:
    """Flat-ish yield series with a tiny ramp so changes are non-trivial but deterministic."""
    return _frame(level + slope * np.arange(n))


def _full_context(n: int = 400) -> dict[str, pd.DataFrame]:
    """A complete synthetic context covering every name the feature layer consumes."""
    return {
        "tnx10": _yields(4.0, 0.001, n),
        "tyx30": _yields(4.2, 0.0008, n),
        "fvx5": _yields(3.5, 0.0012, n),
        "hyg": _drift(0.0003, n),
        "lqd": _drift(0.0001, n),
        "itb": _drift(0.0008, n),
        "xhb": _drift(0.0007, n),
        "iyt": _drift(0.0002, n),
        "smh": _drift(0.0010, n),
        "xlf": _drift(0.0004, n),
        "xly": _drift(0.0006, n),
        "xlp": _drift(0.0002, n),
        "gld": _drift(0.0003, n),
        "uup": _drift(0.0001, n),
        "tlt": _drift(0.0001, n),
    }


def _bench(n: int = 400) -> pd.DataFrame:
    return _drift(0.0005, n)


# --------------------------------------------------------------------------------------------
# (a) PIT: a feature at date t is unchanged when future rows are appended or altered.
# --------------------------------------------------------------------------------------------
def test_pit_truncation_invariance():
    ctx = _full_context()
    bench = _bench()
    dates = _TRADING[100:300]

    full = intermarket_features(dates, ctx, bench)

    t = _TRADING[250]  # an as-of date well inside the requested range

    # Truncate every series to <= t, and additionally CORRUPT all future rows to prove the
    # feature at t never reads them.
    def truncate_and_poison(df: pd.DataFrame) -> pd.DataFrame:
        keep = df[df["date"] <= t].copy()
        future = df[df["date"] > t].copy()
        future["close"] = future["close"] * -7.0 + 999.0  # garbage future
        return pd.concat([keep, future], ignore_index=True)

    poisoned_ctx = {k: truncate_and_poison(v) for k, v in ctx.items()}
    poisoned_bench = truncate_and_poison(bench)
    poisoned = intermarket_features(dates, poisoned_ctx, poisoned_bench)

    # Also a hard truncation (future rows simply removed).
    trunc_ctx = {k: v[v["date"] <= t].copy() for k, v in ctx.items()}
    trunc_bench = bench[bench["date"] <= t].copy()
    truncated = intermarket_features(dates[dates <= t], trunc_ctx, trunc_bench)

    feat_cols = [c for c in full.columns if c != "date"]
    row_full = full.loc[full["date"] == t, feat_cols].reset_index(drop=True)
    row_poison = poisoned.loc[poisoned["date"] == t, feat_cols].reset_index(drop=True)
    row_trunc = truncated.loc[truncated["date"] == t, feat_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(row_full, row_poison, check_exact=False, rtol=1e-12)
    pd.testing.assert_frame_equal(row_full, row_trunc, check_exact=False, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# (b) Relative-strength sign: an outperforming series -> positive RS change.
# --------------------------------------------------------------------------------------------
def test_relative_strength_sign():
    bench = _drift(0.0005)
    ctx = {
        "itb": _drift(0.0015),  # outperforms benchmark
        "iyt": _drift(0.0001),  # underperforms benchmark
    }
    dates = _TRADING[100:300]
    feats = intermarket_features(dates, ctx, bench).iloc[-1]

    assert feats["im_itb_rs21"] > 0, "outperformer must have positive 21d RS"
    assert feats["im_itb_rs63"] > 0, "outperformer must have positive 63d RS"
    assert feats["im_iyt_rs21"] < 0, "underperformer must have negative 21d RS"
    assert feats["im_iyt_rs63"] < 0, "underperformer must have negative 63d RS"


def test_xlp_xly_rotation_sign():
    bench = _bench()
    # Defensives (xlp) beating discretionary (xly) -> risk-off -> positive xlp/xly RS.
    ctx = {"xlp": _drift(0.0010), "xly": _drift(0.0002)}
    feats = intermarket_features(_TRADING[100:300], ctx, bench).iloc[-1]
    assert feats["im_xlp_xly_rs21"] > 0
    assert feats["im_xlp_xly_rs63"] > 0


# --------------------------------------------------------------------------------------------
# (c) Yield change computed correctly (absolute change in yield over the window).
# --------------------------------------------------------------------------------------------
def test_yield_change_values():
    # tnx10 ramps +0.01/day from 4.0; over 21 trading days the change is exactly 0.21.
    ctx = {"tnx10": _yields(4.0, 0.01)}
    dates = _TRADING[50:300]
    feats = intermarket_features(dates, ctx, _bench())

    last = feats.iloc[-1]
    assert last["im_tnx10_chg21"] == pytest.approx(0.01 * 21)
    assert last["im_tnx10_chg5"] == pytest.approx(0.01 * 5)
    # level is just the as-of close.
    t = dates[-1]
    expected_level = 4.0 + 0.01 * list(_TRADING).index(t)
    assert last["im_tnx10_level"] == pytest.approx(expected_level)


def test_curve_slope_values():
    # slope = tyx30 - fvx5. Constant series -> constant slope, zero change.
    ctx = {
        "tyx30": _frame(np.full(400, 4.2)),
        "fvx5": _frame(np.full(400, 3.5)),
        "tnx10": _frame(np.full(400, 4.0)),
    }
    feats = intermarket_features(_TRADING[50:300], ctx, _bench()).iloc[-1]
    assert feats["im_curve_slope"] == pytest.approx(0.7)
    assert feats["im_curve_slope_10_5"] == pytest.approx(0.5)
    assert feats["im_curve_slope_chg21"] == pytest.approx(0.0)


def test_credit_spread_proxy():
    # hyg/lqd ratio: with both flat the ratio is constant and its 21d change is 0.
    ctx = {"hyg": _frame(np.full(400, 80.0)), "lqd": _frame(np.full(400, 110.0))}
    feats = intermarket_features(_TRADING[50:300], ctx, _bench()).iloc[-1]
    assert feats["im_hyg_lqd_ratio"] == pytest.approx(80.0 / 110.0)
    assert feats["im_hyg_lqd_chg21"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------
# (d) Shape / leakage guards: one row per requested date, no future column, names prefixed.
# --------------------------------------------------------------------------------------------
def test_shape_and_no_forward_columns():
    ctx = _full_context()
    dates = _TRADING[100:250]
    feats = intermarket_features(dates, ctx, _bench())

    # one row per requested (unique) date
    assert len(feats) == pd.Index(dates).nunique()
    assert "date" in feats.columns
    assert list(feats["date"]) == sorted(pd.Index(dates).unique())

    # every non-date column must carry the im_ prefix; none may look forward
    for c in feats.columns:
        if c == "date":
            continue
        assert c.startswith("im_"), f"{c} missing im_ prefix"
        assert not c.startswith("fwd_")
    assert not any(c.startswith("fwd_") for c in feats.columns)

    # the full advertised feature set is present
    expected = {
        "im_tnx10_level", "im_tnx10_chg5", "im_tnx10_chg21",
        "im_curve_slope", "im_curve_slope_10_5", "im_curve_slope_chg21",
        "im_hyg_ret21", "im_lqd_ret21", "im_hyg_lqd_ratio", "im_hyg_lqd_chg21",
        "im_itb_rs21", "im_itb_rs63", "im_xhb_rs21", "im_xhb_rs63",
        "im_iyt_rs21", "im_iyt_rs63", "im_smh_rs21", "im_smh_rs63",
        "im_xlf_rs21", "im_xlf_rs63", "im_xly_rs21", "im_xly_rs63",
        "im_xlp_rs21", "im_xlp_rs63", "im_xlp_xly_rs21", "im_xlp_xly_rs63",
        "im_gld_ret21", "im_dollar_chg21", "im_tlt_ret21",
    }
    assert expected.issubset(set(feats.columns))


def test_missing_symbols_tolerated():
    # Only a couple of series present; everything else should be NaN columns, no error.
    ctx = {"tnx10": _yields(4.0, 0.001), "hyg": _drift(0.0003), "lqd": _drift(0.0001)}
    feats = intermarket_features(_TRADING[100:200], ctx, _bench())
    assert len(feats) == 100
    assert feats["im_itb_rs21"].isna().all()      # itb absent
    assert feats["im_smh_rs21"].isna().all()      # smh absent
    assert feats["im_tnx10_chg21"].notna().any()  # tnx10 present


def test_empty_context_returns_dates_only_grid():
    dates = _TRADING[100:150]
    feats = intermarket_features(dates, {}, pd.DataFrame())
    assert len(feats) == 50
    # all feature columns exist and are entirely NaN
    feat_cols = [c for c in feats.columns if c != "date"]
    assert feat_cols, "feature columns should still be present"
    assert feats[feat_cols].isna().all().all()


# --------------------------------------------------------------------------------------------
# optional network smoke test
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="no FMP_API_KEY set")
def test_intermarket_smoke():
    from startx.data.cache import ParquetCache
    from startx.data.intermarket import get_context_prices
    from startx.fmp.client import FMPClient

    cache = ParquetCache("data/cache")
    with FMPClient() as client:
        context = get_context_prices(client, cache, history_start="2022-06-01")
        bench = get_context_prices.__globals__["get_prices"](
            client, cache, "SPY", history_start="2022-06-01"
        )

    assert context, "expected at least some context symbols to return data"

    dates = pd.bdate_range("2023-06-01", "2024-12-31", freq="W-WED")
    feats = intermarket_features(dates, context, bench)

    assert len(feats) == len(pd.Index(dates).unique())
    feat_cols = [c for c in feats.columns if c != "date"]
    coverage = {c: float(feats[c].notna().mean()) for c in feat_cols}
    print("\nintermarket smoke — non-null coverage per feature:")
    for c in feat_cols:
        print(f"  {c:24s} {coverage[c]:6.1%}")
    # the core leading signals should have broad coverage on real data
    for c in ("im_tnx10_chg21", "im_hyg_ret21", "im_itb_rs21", "im_gld_ret21"):
        assert coverage[c] > 0.8, f"{c} coverage too low: {coverage[c]:.1%}"
