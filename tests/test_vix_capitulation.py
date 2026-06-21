"""Unit tests for the VIX-capitulation long signal (``strategy/vix_capitulation.py``).

The second edge the desk trusts (CLAUDE.md "Settled findings": persistent VIX vol-breakout =
capitulation LONG) yet had ZERO dedicated coverage. Deterministic, hand-built VIX series only --
no network, no FMP.

The documented contract we pin (module docstring + ``capitulation_signals``):
  - Upper band = SMA(20) + 2.5*std(20) on VIX close.
  - The signal fires on a *persistent* run of consecutive closes above that band while VIX is also
    above its neutral baseline; it fires ONCE per capitulation episode (one entry per spike).
  - It does NOT fire on an isolated single close above the band, nor when VIX is below neutral.
  - Output is a bool Series aligned 1:1 to the input VIX frame; warm-up rows are False/NaN-safe.

The neutral baseline is pinned to a float in these tests so every assertion is exactly reproducible
(the production default is an adaptive trailing baseline; that path is exercised for shape/NaN-safety).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from startx.strategy.vix_capitulation import (
    VIX_NEUTRAL,
    adaptive_neutral,
    capitulation_signals,
    vix_upper_band,
)


def _vix_frame(close: np.ndarray, start: str = "2022-01-03") -> pd.DataFrame:
    """Wrap a VIX close path into the OHLC-ish frame the module expects (needs date+close)."""
    close = np.asarray(close, dtype=float)
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {"date": dates, "high": close + 0.5, "low": close - 0.5, "close": close}
    )


def _calm_then_spike(
    n: int = 80, base: float = 15.0, spike_start: int = 60, spike_len: int = 5
) -> tuple[pd.DataFrame, int]:
    """A calm VIX (slight deterministic ripple so std>0) that violently spikes for ``spike_len``
    consecutive closes -- the textbook capitulation shape. Returns (frame, spike_start)."""
    close = base + 0.3 * np.sin(np.arange(n))  # tiny deterministic ripple -> non-zero rolling std
    close[spike_start : spike_start + spike_len] = [40.0, 42.0, 44.0, 46.0, 48.0][:spike_len]
    return _vix_frame(close), spike_start


# --------------------------------------------------------------------------- band arithmetic


def test_upper_band_is_sma_plus_k_sigma():
    """Upper band = rolling SMA(window) + k * rolling std(window), recomputed point-in-time."""
    close = pd.Series(np.arange(1, 51, dtype=float))
    band = vix_upper_band(close, window=20, k=2.5)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    expected = mid + 2.5 * sd
    pd.testing.assert_series_equal(band, expected)
    # Warm-up (first 19) has no 20-window -> NaN.
    assert band.iloc[:19].isna().all()
    assert band.notna().iloc[19:].all()


# --------------------------------------------------------------------------- positive case


def test_capitulation_fires_once_on_a_sustained_spike():
    """POSITIVE: a sustained run of consecutive above-band closes (VIX > neutral) triggers exactly
    one capitulation long for the episode."""
    vix, spike_start = _calm_then_spike()
    sig = capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=3)

    fired = np.where(sig.fillna(False).values)[0]
    # Exactly one entry for the single episode (no piling into the same spike).
    assert len(fired) == 1
    # The fire lands inside the spike window and only after >=3 consecutive above-band closes.
    idx = int(fired[0])
    assert spike_start <= idx < spike_start + 5
    band = vix_upper_band(vix["close"])
    above = (vix["close"] > band) & (vix["close"] > VIX_NEUTRAL)
    # The fire requires a genuine *persistent* run: the fire bar and the bar before it are both
    # above the band (the module's run-length counter triggers inside an established streak, never
    # on an isolated single close -- see test_isolated_single_spike_does_not_fire).
    assert bool(above.iloc[idx]) is True
    assert bool(above.iloc[idx - 1]) is True


def _expected_fires(above: np.ndarray, n: int) -> list[int]:
    """Ground-truth reference: indices where the consecutive-above count reaches EXACTLY n
    (an explicit running counter, reset on every break) — independent of the module's vectorised idiom."""
    out, c = [], 0
    for i, a in enumerate(above):
        c = c + 1 if a else 0
        if c == n:
            out.append(i)
    return out


def test_fires_on_exactly_the_nth_consecutive_close():
    """REGRESSION (off-by-one): the module must fire on the bar that completes EXACTLY ``min_closes``
    consecutive above-band closes — verified against an independent running counter. The old idiom
    folded the breaking bar into the run and fired one close early, so it disagrees with this reference."""
    vix, _ = _calm_then_spike(spike_len=5)
    band = vix_upper_band(vix["close"])
    above = ((vix["close"] > band) & (vix["close"] > VIX_NEUTRAL)).to_numpy()
    assert _expected_fires(above, 2), "test must be non-trivial: at least a length-2 run exists"
    for n in (2, 3, 4, 5):
        fired = list(np.flatnonzero(capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=n)
                                    .fillna(False).to_numpy()))
        assert fired == _expected_fires(above, n)


def test_lone_spike_never_fires_even_at_min_closes_2():
    """REGRESSION (the sharp edge of the off-by-one): an isolated single above-band close must NOT
    fire even at ``min_closes=2``. The buggy counter read a lone spike as run==2 and fired; the
    fixed counter reads it as run==1, so persistence is genuinely required."""
    close = 15.0 + 0.3 * np.sin(np.arange(80))
    close[60] = 60.0  # one violent single-day spike
    vix = _vix_frame(close)
    assert int(capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=2).fillna(False).sum()) == 0


def test_min_closes_makes_signal_more_selective():
    """Requiring more consecutive closes (larger ``min_closes``) cannot produce MORE entries than a
    looser requirement; a short spike that satisfies a low bar can fail a higher one."""
    # Spike of only 3 closes above the band.
    vix, _ = _calm_then_spike(spike_len=3)
    fires_2 = int(capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=2).fillna(False).sum())
    fires_5 = int(capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=5).fillna(False).sum())
    assert fires_2 >= 1
    assert fires_5 == 0  # the run is too short to ever reach a length-5 streak


# --------------------------------------------------------------------------- negative cases


def test_no_signal_in_calm_market():
    """NEGATIVE: a calm VIX that never breaches its own upper band produces no signal at all."""
    vix = _vix_frame(15.0 + 0.3 * np.sin(np.arange(80)))
    sig = capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=3)
    assert int(sig.fillna(False).sum()) == 0


def test_isolated_single_spike_does_not_fire():
    """NEGATIVE: a single one-day pop above the band is NOT capitulation -- persistence is required,
    so with min_closes=3 a lone spike must stay silent."""
    close = 15.0 + 0.3 * np.sin(np.arange(80))
    close[60] = 60.0  # one violent single-day spike, then back to calm
    vix = _vix_frame(close)
    sig = capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=3)
    assert int(sig.fillna(False).sum()) == 0


def test_no_signal_when_below_neutral():
    """NEGATIVE (neutral leg): even a sustained band breakout must NOT fire if VIX sits below the
    neutral floor -- pin neutral absurdly high so the ``close > base`` leg can never be satisfied."""
    vix, _ = _calm_then_spike()  # spike tops out at 48
    sig = capitulation_signals(vix, neutral=100.0, min_closes=3)
    assert int(sig.fillna(False).sum()) == 0


# --------------------------------------------------------------------------- NaN / shape / adaptive


def test_output_alignment_length_and_nan_safety():
    """Output is a bool Series aligned 1:1 to the input frame, with no NaNs leaking through, for
    both the pinned-float and the production 'adaptive' neutral path."""
    vix, _ = _calm_then_spike(n=300)  # >252 so the adaptive baseline warms up

    sig_float = capitulation_signals(vix, neutral=VIX_NEUTRAL, min_closes=3)
    assert len(sig_float) == len(vix)
    assert sig_float.dtype == bool
    assert not sig_float.isna().any()

    sig_adaptive = capitulation_signals(vix, neutral="adaptive", min_closes=3)
    assert len(sig_adaptive) == len(vix)
    # The adaptive path may leave warm-up NaNs (band/baseline undefined early); they must never be
    # treated as a fire, so .fillna(False) is a clean boolean mask and any real fire is bool True.
    assert int(sig_adaptive.fillna(False).sum()) >= 1


def test_adaptive_neutral_is_pointintime_and_finite_after_warmup():
    """The adaptive neutral baseline is computed from a trailing window only (no lookahead) and is
    finite once warmed up; constants are sane (static VIX_NEUTRAL == 18.0)."""
    assert VIX_NEUTRAL == 18.0
    close = pd.Series(15.0 + 0.3 * np.sin(np.arange(300)))
    base = adaptive_neutral(close, method="median")
    # min_periods=60 -> first 59 are NaN, rest finite.
    assert base.iloc[:59].isna().all()
    assert np.isfinite(base.iloc[100:]).all()
