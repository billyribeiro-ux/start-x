"""Unit tests for the volatility-risk-premium sleeve (``strategy/volatility_premium.py``).

A production sleeve that had ZERO dedicated coverage. Every case is deterministic and runs on
hand-built synthetic OHLCV + exogenous VIX/VVIX frames -- no network, no FMP.

The documented contract we pin (module docstring + the public signal fns):
  - ``vrp_high_signals``: VRP = VIX - realized_vol(index, rv_window); fires when VRP sits in the top
    ``pct`` of its trailing ``lookback`` window. Point-in-time (trailing-only); a missing VIX print
    or unestablished band -> False.
  - ``vvix_spike_signals``: fires when VVIX is at/above its trailing-``lookback`` ``pct`` percentile.
  - ``fear_signals``: the OR of the two, counted once.
  - All three return a bool Series aligned 1:1 to the (sorted) index frame; exogenous series align
    by DATE (never by row position) and a missing input resolves to False.

The ``backtest`` smoke pins documented, sleeve-agnostic behavior plus -- the point of this file --
that every outcome label is in the THREE-way {WIN, SCRATCH, LOSS} set (the desk's LOCKED rule:
a +1-ATR / breakeven exit is a WIN/SCRATCH, never a LOSS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.volatility_premium import (
    backtest,
    fear_signals,
    realized_vol,
    vrp_high_signals,
    vvix_spike_signals,
)


def _index_frame(n: int = 320, start: str = "2021-01-04") -> pd.DataFrame:
    """A calm, low-realized-vol index uptrend so a VIX/VVIX *spike* is what drives every signal
    (not index volatility). Symmetric +/-0.5 bar range; tiny deterministic ripple for a finite ATR."""
    dates = pd.bdate_range(start, periods=n)
    close = np.linspace(300.0, 360.0, n) + 0.2 * np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _vol_frame(dates: pd.Series, level: np.ndarray) -> pd.DataFrame:
    """An exogenous VIX/VVIX-style frame aligned to ``dates`` (carries 'date' + 'close')."""
    level = np.asarray(level, dtype=float)
    return pd.DataFrame({"date": pd.Series(dates).reset_index(drop=True), "close": level})


# --------------------------------------------------------------------------- realized_vol sanity


def test_realized_vol_warmup_nan_then_positive():
    """``realized_vol`` is NaN through warm-up (min_periods=window) and positive once a noisy window
    fills; a perfectly flat price path gives zero realized vol."""
    rng = np.random.default_rng(0)
    noisy = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 100)))
    rv = realized_vol(noisy, window=20)
    assert rv.iloc[:19].isna().all()
    assert (rv.iloc[20:] > 0).all()
    flat = pd.Series(np.full(50, 100.0))
    assert (realized_vol(flat, window=20).iloc[20:] == 0).all()


# --------------------------------------------------------------------------- vrp_high_signals


def test_vrp_fires_on_rich_premium_spike():
    """POSITIVE: a gently DECLINING calm VIX (whose VRP is never in the top of its trailing window)
    with a late sustained spike fires ONLY on the spike -- the rich premium is the trade.

    A declining baseline is the clean negative control: the latest VRP is below older bars, so it
    never tops its trailing p95 on its own. Bolt a spike on top and only the spike fires.
    """
    p = _index_frame()
    vix = _vol_frame(p["date"], np.linspace(28.0, 14.0, len(p)))  # declining -> baseline silent
    vix.loc[300:306, "close"] = 70.0                              # the sustained rich-premium spike
    sig = vrp_high_signals(p, vix, rv_window=20, lookback=252, pct=0.95)
    assert sig.dtype == bool
    fired = list(np.flatnonzero(sig.to_numpy()))
    assert fired, "the spike must fire at least once"
    assert all(300 <= i <= 306 for i in fired)   # ONLY the spike window fires; baseline is silent


def test_vrp_silent_when_premium_never_rich():
    """NEGATIVE: a gently DECLINING calm VIX whose VRP is never in the top 5% of its trailing window
    produces no signal at all (the latest premium is always below older bars)."""
    p = _index_frame()
    declining = _vol_frame(p["date"], np.linspace(30.0, 12.0, len(p)))
    sig = vrp_high_signals(p, declining, rv_window=20, lookback=252, pct=0.95)
    assert int(sig.sum()) == 0


def test_vrp_missing_vix_resolves_to_false():
    """Point-in-time safety: where the VIX frame is missing dates (beyond the short ffill gap) VRP is
    NaN -> False (no spurious fire), and the output stays aligned 1:1 to the index frame."""
    p = _index_frame()
    base_vix = _vol_frame(p["date"], np.linspace(28.0, 14.0, len(p)))
    base_vix.loc[300:308, "close"] = 70.0
    # Drop the VIX rows entirely for a window past the spike -> after ffill(limit=3) the deep gap is
    # NaN, so those bars cannot fire even though the index frame still has those dates.
    truncated = base_vix.drop(index=range(310, 320)).reset_index(drop=True)
    sig = vrp_high_signals(p, truncated, rv_window=20, lookback=252, pct=0.95)
    assert len(sig) == len(p)
    assert not sig.isna().any()
    # The bars whose VIX is missing beyond the 3-day ffill (>=314) cannot fire.
    assert not sig.iloc[314:320].any()


# --------------------------------------------------------------------------- vvix_spike_signals


def test_vvix_fires_at_top_percentile():
    """POSITIVE: a gently DECLINING VVIX (never topping its own band) with a late spike fires ONLY on
    the spike -- VVIX at/above its trailing-year p90 is the trade."""
    p = _index_frame()
    vvix = _vol_frame(p["date"], np.linspace(130.0, 85.0, len(p)))  # declining -> baseline silent
    vvix.loc[300:306, "close"] = 200.0                              # the vol-of-vol spike
    sig = vvix_spike_signals(p, vvix, lookback=252, pct=0.90)
    assert sig.dtype == bool
    fired = list(np.flatnonzero(sig.to_numpy()))
    assert fired, "the spike must fire at least once"
    assert all(300 <= i <= 306 for i in fired)


def test_vvix_silent_in_calm_regime():
    """NEGATIVE: a gently DECLINING VVIX never reaches a high-percentile breakout over its own
    trailing window, so it produces no signal at all."""
    p = _index_frame()
    vvix = _vol_frame(p["date"], np.linspace(130.0, 85.0, len(p)))
    sig = vvix_spike_signals(p, vvix, lookback=252, pct=0.90)
    assert int(sig.sum()) == 0


# --------------------------------------------------------------------------- fear_signals (OR)


def test_fear_is_the_union_of_vrp_and_vvix():
    """``fear_signals`` is exactly the elementwise OR of the two component signals -- the documented
    merge that counts a co-fire once."""
    p = _index_frame()
    vix = _vol_frame(p["date"], np.linspace(28.0, 14.0, len(p)))
    vix.loc[300:305, "close"] = 70.0                        # drives VRP near the end
    vvix = _vol_frame(p["date"], np.linspace(130.0, 85.0, len(p)))
    vvix.loc[200:205, "close"] = 200.0                      # drives VVIX at a DIFFERENT window

    vrp = vrp_high_signals(p, vix)
    vvx = vvix_spike_signals(p, vvix)
    fear = fear_signals(p, vix, vvix)

    pd.testing.assert_series_equal(fear, (vrp | vvx).fillna(False).astype(bool),
                                   check_names=False)
    # Non-trivial: both components contributed at least one fire, and the union covers both windows.
    assert int(vrp.sum()) >= 1
    assert int(vvx.sum()) >= 1
    assert int(fear.sum()) >= max(int(vrp.sum()), int(vvx.sum()))


def test_signals_aligned_length_and_nan_safe():
    """All three public signals are bool Series aligned 1:1 to the (sorted) index frame, NaN-free,
    and stable under a shuffled index input (the functions sort by date internally)."""
    p = _index_frame()
    vix = _vol_frame(p["date"], np.linspace(28.0, 14.0, len(p)))
    vix.loc[300:305, "close"] = 70.0
    vvix = _vol_frame(p["date"], np.linspace(130.0, 85.0, len(p)))
    vvix.loc[300:305, "close"] = 200.0

    for sig in (vrp_high_signals(p, vix), vvix_spike_signals(p, vvix), fear_signals(p, vix, vvix)):
        assert sig.dtype == bool
        assert not sig.isna().any()
        assert len(sig) == len(p)

    shuffled = p.sample(frac=1.0, random_state=3).reset_index(drop=True)
    sig_sorted = vrp_high_signals(p, vix)
    sig_shuf = vrp_high_signals(shuffled, vix)
    sorted_dates = p.sort_values("date").reset_index(drop=True)["date"]
    assert sorted_dates[sig_sorted.values].tolist() == sorted_dates[sig_shuf.values].tolist()


# --------------------------------------------------------------------------- backtest smoke


def _coherent_ledger_asserts(trades: pd.DataFrame, signal: str) -> None:
    assert not trades.empty
    # THREE-way outcome rule -- never the old binary {WIN, LOSS}.
    assert set(trades["outcome"]).issubset({"WIN", "SCRATCH", "LOSS"})
    assert set(trades["exit_reason"]).issubset({"chandelier", "stop", "target", "time"})
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    assert (trades["bars_held"] >= 0).all()
    # LOCKED shared trade-sheet columns + the per-signal vol-level column.
    for col in ("symbol", "signal", "entry_price", "stop_price", "exit_price", "pnl_per_share",
                "outcome"):
        assert col in trades.columns
    assert signal in trades.columns           # vol_level_col == the signal name ('vrp'/'vvix')
    assert trades["signal"].eq(signal).all()


def test_backtest_vrp_smoke_produces_coherent_ledger():
    p = _index_frame()
    p.attrs["symbol"] = "TEST"
    vix = _vol_frame(p["date"], np.linspace(28.0, 14.0, len(p)))
    vix.loc[300:308, "close"] = 70.0
    trades, stats = backtest(p, vix, signal="vrp", max_days=10)
    _coherent_ledger_asserts(trades, "vrp")
    assert trades["symbol"].eq("TEST").all()
    assert stats["n"] == len(trades)
    # One position at a time: no entry opens before the prior trade has exited.
    ed = trades.sort_values("entry_date")
    assert (ed["entry_date"].iloc[1:].values >= ed["exit_date"].iloc[:-1].values).all()


def test_backtest_vvix_smoke_and_outcome_sign_mapping():
    """VVIX backtest produces a coherent ledger AND every outcome label matches the sign of the net
    (cost-adjusted) return -- the three-way rule the production engine uses (breakeven -> SCRATCH)."""
    p = _index_frame()
    vvix = _vol_frame(p["date"], np.linspace(130.0, 85.0, len(p)))
    vvix.loc[300:308, "close"] = 200.0
    trades, _ = backtest(p, vvix, signal="vvix", max_days=10)
    _coherent_ledger_asserts(trades, "vvix")
    cost = 2.0 / 10000.0
    for _, t in trades.iterrows():
        net = (t["exit_price"] / t["entry_price"] - 1) - cost
        expected = "WIN" if net > 0 else "SCRATCH" if net == 0 else "LOSS"
        assert t["outcome"] == expected


def test_backtest_rejects_unknown_signal():
    """``signal`` must be 'vrp' or 'vvix'; anything else raises ValueError (no silent default)."""
    p = _index_frame()
    vix = _vol_frame(p["date"], np.full(len(p), 16.0))
    import pytest

    with pytest.raises(ValueError):
        backtest(p, vix, signal="bogus")


def test_backtest_empty_when_no_fear():
    """A calm vol regime that never triggers yields an empty ledger and n==0 stats (clean no-trade
    path, not an exception)."""
    p = _index_frame()
    vix = _vol_frame(p["date"], np.linspace(30.0, 12.0, len(p)))  # steadily declining -> never rich
    trades, stats = backtest(p, vix, signal="vrp")
    assert trades.empty
    assert stats == {"n": 0}
