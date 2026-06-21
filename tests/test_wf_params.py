"""Tests for the walk-forward PARAMETER-selection harness (`validation/wf_params.py`).

Covers the harness mechanics on a fast synthetic book (no lookahead in selection, OOS
stitching, deflation, pick-stability) plus a thin smoke test that wires it to one REAL book
slice so a regression in the production plumbing crashes here, not silently in a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from startx.validation.wf_params import (
    BookSpec,
    pick_stability,
    walk_forward_select,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _synthetic_spec(n: int = 1200, seed: int = 0) -> BookSpec:
    """A 3-config book where ONE config is genuinely best and config returns are deterministic.

    config 'good' has a small positive drift, 'flat' ~0, 'bad' negative. Walk-forward should
    keep picking 'good' on TRAIN and the stitched OOS series should be positive.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    noise = rng.normal(0, 0.005, n)
    # drift/vol = 0.0015/0.005 -> ~4.8 annualised Sharpe in expectation; comfortably > 0.5 OOS
    series = {
        "good": pd.Series(0.0015 + noise, index=idx),
        "flat": pd.Series(noise, index=idx),
        "bad": pd.Series(-0.0015 + noise, index=idx),
    }
    grid = [{"kind": k} for k in series]

    def run(params):
        return series[params["kind"]]

    return BookSpec("synthetic", grid, run, {"kind": "good"})


def test_walk_forward_runs_and_stitches_oos():
    r = walk_forward_select(_synthetic_spec(), train=252, test=63)
    # produced folds and a stitched OOS series of TEST days only
    assert r.n_folds >= 3
    assert len(r.oos_returns) == int(r.picks["n_test"].sum())
    # OOS dates are strictly increasing and disjoint (each TEST block used once)
    assert r.oos_returns.index.is_monotonic_increasing
    assert r.oos_returns.index.is_unique
    # metrics are finite probabilities / numbers
    assert np.isfinite(r.oos_sharpe) and np.isfinite(r.is_sharpe)
    assert 0.0 <= r.oos_dsr <= 1.0 and 0.0 <= r.is_dsr <= 1.0
    assert r.oos_max_drawdown <= 0.0


def test_selection_picks_the_truly_best_config_oos():
    """On a book with a genuinely-best config, WF keeps picking it and OOS Sharpe is healthy."""
    r = walk_forward_select(_synthetic_spec(), train=252, test=63)
    # 'good' should be the modal pick in (nearly) every fold
    stab = pick_stability(r.picks)
    assert stab["params"]["kind"]["modal"] == "good"
    assert stab["params"]["kind"]["modal_frac"] >= 0.7
    assert r.oos_sharpe > 0.5  # the real positive-drift edge survives OOS selection


def test_selection_is_point_in_time_no_test_leak():
    """The pick must depend ONLY on TRAIN: corrupting the TEST blocks cannot change any pick.

    We corrupt the would-be-losing 'bad' config with an absurd +100%/day spike on EXACTLY the
    days that land in some fold's TEST block (computed with the harness's own rolling-window
    arithmetic), leaving every TRAIN region pristine. If selection peeked at TEST it would
    flip to 'bad'; a point-in-time harness must still pick 'good' in every fold.
    """
    n, train, test = 1200, 252, 63
    base = _synthetic_spec(n=n)
    r1 = walk_forward_select(base, train=train, test=test)

    # The only days never used by ANY fold's TRAIN are those in the FINAL TEST block (in rolling
    # mode a fold's TEST block IS the next fold's TRAIN, so only the last block is pure-future).
    last_test_start = train
    while last_test_start + test < n:
        last_test_start += test
    future = np.zeros(n, dtype=bool)
    future[last_test_start:] = True

    idx = _dates(n)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.005, n)
    spike = np.where(future, 1.0, 0.0)  # absurd +100%/day, ONLY on pure-future (never-TRAIN) days
    series = {
        "good": pd.Series(0.0015 + noise, index=idx),
        "flat": pd.Series(noise, index=idx),
        "bad": pd.Series(-0.0015 + noise + spike, index=idx),
    }
    grid = [{"kind": k} for k in series]
    spec2 = BookSpec("leaky", grid, lambda p: series[p["kind"]], {"kind": "good"})
    r2 = walk_forward_select(spec2, train=train, test=test)

    # identical TRAIN data -> identical picks; TEST spike is invisible to selection.
    assert list(r1.picks["param_kind"]) == list(r2.picks["param_kind"])
    assert list(r2.picks["param_kind"]) == ["good"] * r2.n_folds


def test_anchored_vs_rolling_both_run():
    rolling = walk_forward_select(_synthetic_spec(), train=252, test=63, anchored=False)
    anchored = walk_forward_select(_synthetic_spec(), train=252, test=63, anchored=True)
    assert rolling.n_folds == anchored.n_folds  # same TEST schedule
    # anchored fold-0 train starts at the data start; later folds expand
    assert anchored.picks["train_start"].nunique() == 1


def test_requires_at_least_two_configs():
    bad = BookSpec("one", [{"kind": "good"}], lambda p: pd.Series([0.0] * 300, index=_dates(300)),
                   {"kind": "good"})
    with pytest.raises(ValueError):
        walk_forward_select(bad, train=100, test=50)


def test_raises_when_span_too_short_for_train():
    short = _synthetic_spec(n=200)
    with pytest.raises(ValueError):
        walk_forward_select(short, train=504, test=126)  # train longer than the data


def test_real_short_swing_book_slice_runs():
    """Smoke test: wire ONE real book to the harness on a small grid; it must run without crashing.

    Guards the production plumbing (run_portfolio + ibs_signals + equity->returns slicing) that
    scripts/walkforward_params.py depends on. Skipped automatically if the price cache is absent.
    """
    from pathlib import Path

    cache = Path("data/cache/prices")
    if not (cache / "SPY.parquet").exists():
        pytest.skip("price cache not available")

    def load(s):
        p = pd.read_parquet(cache / f"{s}.parquet")
        p["date"] = pd.to_datetime(p["date"])
        return p.sort_values("date").reset_index(drop=True)

    from startx.portfolio import run_portfolio
    from startx.strategy.mean_reversion import ibs_signals

    spy = load("SPY"); spy.attrs["symbol"] = "SPY"
    aux = {"vix": load("_VIX"), "vvix": load("_VVIX"), "gld": load("GLD")}
    start, end = "2019-01-01", "2026-06-19"

    grid = [{"ibs_threshold": t} for t in (0.05, 0.10, 0.15)]

    def run(params):
        res = run_portfolio(
            spy, aux, {"ibs": lambda s, a, t=params["ibs_threshold"]: ibs_signals(s, t)},
            exits={"ibs": (1.0, 3.0, 10)}, gross_cap=1.5, start=start, end=end,
        )
        eq = res.equity[(res.equity.index >= pd.Timestamp(start))
                        & (res.equity.index <= pd.Timestamp(end))]
        return eq.pct_change().fillna(0.0)

    spec = BookSpec("short_swing", grid, run, {"ibs_threshold": 0.10})
    r = walk_forward_select(spec, train=504, test=126)
    assert r.n_folds >= 2
    assert np.isfinite(r.oos_sharpe)
    assert 0.0 <= r.oos_dsr <= 1.0
    assert len(r.oos_returns) > 0
