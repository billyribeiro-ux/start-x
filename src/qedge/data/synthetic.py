"""Deterministic, seeded synthetic market implementing all five data feeds.

:class:`SyntheticMarket` is the offline fixture used everywhere a real feed is
unavailable or undesirable (unit tests, CI, reproducible demos). It implements
all five :mod:`qedge.data.protocols` Protocols against a single seeded RNG so
that the *same seed yields byte-identical frames* (verified via
:func:`qedge.repro.hash_frame`).

Two invariants are load-bearing and proven in the test-suite:

* **Determinism.** Every series is drawn from :func:`qedge.repro.rng` keyed by a
  ``(seed, symbol, stream)`` triple — never the global RNG — so generation is a
  pure function of construction arguments.
* **Point-in-time / no-lookahead.** Every feed method generates the *full*
  configured horizon and then returns only the rows whose public timestamp is
  ``<= asof``. Because the full horizon is always generated identically, the
  ``<= asof`` slice is identical whether the generator's end date is ``asof`` or
  far beyond it — there is no way for a future row to perturb a past value.

Frames follow the exact column conventions documented in ``protocols.py``.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from qedge.config import QedgeConfig, get_config
from qedge.repro import rng

# --- Calendar -------------------------------------------------------------
#: Trading-day frequency. Business days (Mon-Fri) approximate the NYSE calendar
#: closely enough for synthetic fixtures; holidays are intentionally ignored.
_TRADING_FREQ: Final[str] = "B"

# --- Price process (regime-switching geometric Brownian motion) -----------
#: Number of latent regimes the daily return process switches between.
_N_REGIMES: Final[int] = 2
#: Per-regime annualised drift (calm bull, stressed bear).
_REGIME_DRIFT_ANNUAL: Final[tuple[float, float]] = (0.10, -0.15)
#: Per-regime annualised volatility (calm, stressed).
_REGIME_VOL_ANNUAL: Final[tuple[float, float]] = (0.12, 0.35)
#: Daily probability of *staying* in the current regime (sticky -> persistence).
_REGIME_PERSISTENCE: Final[tuple[float, float]] = (0.98, 0.95)
#: Trading days per year used to convert annual drift/vol to per-bar values.
_TRADING_DAYS_PER_YEAR: Final[int] = 252
#: Starting price for every symbol's first synthetic bar (USD).
_INITIAL_PRICE: Final[float] = 100.0
#: Intrabar high/low are drawn as a fraction of the bar's own volatility; this
#: multiplies the per-bar sigma to size the wick beyond the open/close envelope.
_WICK_VOL_MULT: Final[float] = 0.5
#: Baseline daily share volume; scaled up multiplicatively in stressed regimes.
_BASE_VOLUME: Final[float] = 5.0e6
#: Volume multiplier applied while in the stressed (high-vol) regime.
_STRESS_VOLUME_MULT: Final[float] = 2.5
#: Log-normal noise sigma applied to volume so it is positive and dispersed.
_VOLUME_NOISE_SIGMA: Final[float] = 0.25

# --- Fundamentals (scheduled quarterly earnings) --------------------------
#: Calendar days in a fiscal quarter (period length between earnings periods).
_QUARTER_DAYS: Final[int] = 91
#: Dissemination lag (calendar days) between fiscal-period end and the public
#: ``release_ts`` — earnings are reported weeks after the quarter closes.
_EARNINGS_RELEASE_LAG_DAYS: Final[int] = 30
#: Mean / sigma of synthetic reported EPS (USD) drawn per release.
_EPS_MEAN: Final[float] = 1.50
_EPS_SIGMA: Final[float] = 0.30

# --- News -----------------------------------------------------------------
#: Expected number of news items per symbol per trading day (Poisson rate).
_NEWS_RATE_PER_DAY: Final[float] = 0.4
#: Intraday publication offset bounds (hours after midnight) for ``published_ts``.
_NEWS_HOUR_LOW: Final[float] = 9.0
_NEWS_HOUR_HIGH: Final[float] = 16.0

# --- Options NBBO ---------------------------------------------------------
#: Synthetic implied-vol level (annualised) used to price the chain.
_OPTION_IV: Final[float] = 0.20
#: Half-spread as a fraction of mid price (defines bid<ask around a mid).
_OPTION_HALF_SPREAD_FRAC: Final[float] = 0.02
#: Quote size bounds (contracts) for bid/ask sizes.
_OPTION_SIZE_LOW: Final[int] = 1
_OPTION_SIZE_HIGH: Final[int] = 50
#: Number of intraday NBBO snapshots generated per trading day for a contract.
_OPTION_SNAPSHOTS_PER_DAY: Final[int] = 4

# --- Internals (TICK / TRIN / ADD / VOLD) ---------------------------------
#: Number of intraday internals snapshots generated per trading day.
_INTERNALS_SNAPSHOTS_PER_DAY: Final[int] = 8
#: TICK is symmetric around zero; this bounds its synthetic range.
_TICK_SCALE: Final[float] = 800.0
#: TRIN is log-normal around 1.0; this is the sigma of its log.
_TRIN_LOG_SIGMA: Final[float] = 0.30
#: Advancing/declining issue count drawn around this midpoint.
_ADD_SCALE: Final[float] = 1500.0
#: Up/down volume differential scale (shares).
_VOLD_SCALE: Final[float] = 4.0e8

# --- RNG keying -----------------------------------------------------------
#: Logical streams. Each ``(stream, symbol)`` pair plus an integer ``sub`` index
#: keys an INDEPENDENT generator via a hashed ``SeedSequence`` (see ``_stream_rng``).
_STREAM_PRICE: Final[str] = "price"
_STREAM_FUNDAMENTALS: Final[str] = "fundamentals"
_STREAM_NEWS: Final[str] = "news"
_STREAM_OPTIONS: Final[str] = "options"
_STREAM_INTERNALS: Final[str] = "internals"

# --- Price sub-stream indices (each is an independent generator) ----------
#: Drawing each price quantity from its own generator is what guarantees
#: no-lookahead: a quantity at bar ``i`` is fixed by position ``i`` alone, never
#: by the total bar count, so extending the generator's ``end`` cannot perturb it.
_PRICE_SUB_REGIME: Final[int] = 0
_PRICE_SUB_SHOCK: Final[int] = 1
_PRICE_SUB_WICK: Final[int] = 2
_PRICE_SUB_VOLUME: Final[int] = 3


def _key_to_seed(stream: str, symbol: str | None, sub: int) -> int:
    """Map a ``(stream, symbol, sub)`` key to a stable 64-bit seed.

    Uses a fixed FNV-1a hash over the key bytes (NOT Python's salted ``hash``) so
    the per-key generator is byte-identical across processes and runs. Distinct
    keys map to distinct seeds with negligible collision probability, giving each
    logical stream an independent, non-correlated RNG.
    """
    fnv_offset = 0xCBF29CE484222325
    fnv_prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    payload = f"{stream}\x00{symbol or ''}\x00{sub}".encode()
    acc = fnv_offset
    for byte in payload:
        acc = ((acc ^ byte) * fnv_prime) & mask
    return acc


class SyntheticMarket:
    """Deterministic synthetic implementation of all five qedge data feeds.

    Args:
        start: First trading day of the generated horizon (inclusive).
        end: Last trading day of the generated horizon (inclusive). Generating
            to a later ``end`` never changes the ``<= asof`` slice of any feed.
        seed: RNG seed. Defaults to ``cfg.scanner.seed``.
        symbols: Symbols to generate. Defaults to ``cfg.data.universe``.
        config: Optional config override (defaults to the process singleton).
    """

    def __init__(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        seed: int | None = None,
        symbols: tuple[str, ...] | None = None,
        config: QedgeConfig | None = None,
    ) -> None:
        self._cfg: QedgeConfig = config if config is not None else get_config()
        self._seed: int = seed if seed is not None else self._cfg.scanner.seed
        self._symbols: tuple[str, ...] = (
            symbols if symbols is not None else tuple(self._cfg.data.universe)
        )
        self._start: pd.Timestamp = pd.Timestamp(start).normalize()
        self._end: pd.Timestamp = pd.Timestamp(end).normalize()
        if self._end < self._start:
            raise ValueError("end must be >= start")
        self._calendar: pd.DatetimeIndex = pd.bdate_range(
            self._start, self._end, freq=_TRADING_FREQ
        )

    # -- internals ---------------------------------------------------------

    def _stream_rng(
        self, stream: str, symbol: str | None = None, *, sub: int = 0
    ) -> np.random.Generator:
        """Return an independent Generator for a ``(seed, stream, symbol, sub)`` key.

        The ``sub`` index lets one feed draw several independent quantities, each
        from its own generator. This is what guarantees no-lookahead for the
        vectorised price process: each quantity draws exactly ``n`` values from a
        *dedicated* stream, so the length-``min(n)`` prefix is identical no matter
        how far the generator's ``end`` extends (a longer horizon never shifts the
        RNG position of any other quantity). The base seed and the key hash are
        XOR-mixed so different ``self._seed`` values yield different streams.
        """
        return rng(self._seed ^ _key_to_seed(stream, symbol, sub))

    def _require_symbol(self, symbol: str) -> None:
        if symbol not in self._symbols:
            raise KeyError(f"symbol {symbol!r} not in synthetic universe {self._symbols}")

    def _full_price_frame(self, symbol: str) -> pd.DataFrame:
        """Generate the entire OHLCV history for ``symbol`` (no ``asof`` clip).

        Regime-switching GBM: a sticky 2-state Markov chain selects per-bar drift
        and volatility; close prices compound the regime-conditional log returns.
        High/low envelope each bar with a volatility-scaled wick so that
        ``low <= open, close <= high`` always holds.
        """
        n = len(self._calendar)
        if n == 0:
            return self._empty_price_frame()

        # Each quantity draws from its OWN generator (see ``_stream_rng`` doc): a
        # value at bar ``i`` is fixed by position ``i`` alone, so extending ``end``
        # leaves every ``<= asof`` row byte-identical (the no-lookahead guarantee).
        regime_gen = self._stream_rng(_STREAM_PRICE, symbol, sub=_PRICE_SUB_REGIME)
        shock_gen = self._stream_rng(_STREAM_PRICE, symbol, sub=_PRICE_SUB_SHOCK)
        wick_gen = self._stream_rng(_STREAM_PRICE, symbol, sub=_PRICE_SUB_WICK)
        volume_gen = self._stream_rng(_STREAM_PRICE, symbol, sub=_PRICE_SUB_VOLUME)

        drift = np.asarray(_REGIME_DRIFT_ANNUAL) / _TRADING_DAYS_PER_YEAR
        vol = np.asarray(_REGIME_VOL_ANNUAL) / np.sqrt(_TRADING_DAYS_PER_YEAR)
        stay = np.asarray(_REGIME_PERSISTENCE)

        # Latent regime path (sticky Markov chain over _N_REGIMES states).
        regimes = np.empty(n, dtype=np.int64)
        state = 0
        switch_draws = regime_gen.random(n)
        for i in range(n):
            if switch_draws[i] > stay[state]:
                state = (state + 1) % _N_REGIMES
            regimes[i] = state

        shocks = shock_gen.standard_normal(n)
        log_ret = drift[regimes] + vol[regimes] * shocks
        close = _INITIAL_PRICE * np.exp(np.cumsum(log_ret))
        # Open = prior close (first bar opens at the initial price).
        open_ = np.empty(n, dtype=np.float64)
        open_[0] = _INITIAL_PRICE
        open_[1:] = close[:-1]

        # Wick: a non-negative half-range scaled by the bar's regime volatility.
        wick = _WICK_VOL_MULT * vol[regimes] * close * np.abs(wick_gen.standard_normal(n))
        oc_high = np.maximum(open_, close)
        oc_low = np.minimum(open_, close)
        high = oc_high + wick  # >= max(open, close) since wick >= 0
        low = oc_low - wick  # <= min(open, close) since wick >= 0
        low = np.maximum(low, 1e-6)  # strictly positive price floor

        # Volume: log-normal noise, inflated in the stressed regime.
        regime_mult = np.where(regimes == 1, _STRESS_VOLUME_MULT, 1.0)
        vol_noise = np.exp(_VOLUME_NOISE_SIGMA * volume_gen.standard_normal(n))
        volume = np.rint(_BASE_VOLUME * regime_mult * vol_noise).astype(np.int64)
        volume = np.maximum(volume, 1)  # strictly positive

        return pd.DataFrame(
            {
                "date": self._calendar,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    @staticmethod
    def _empty_price_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.Series([], dtype="datetime64[ns]"),
                "open": pd.Series([], dtype="float64"),
                "high": pd.Series([], dtype="float64"),
                "low": pd.Series([], dtype="float64"),
                "close": pd.Series([], dtype="float64"),
                "volume": pd.Series([], dtype="int64"),
            }
        )

    def _full_earnings_frame(self, symbol: str) -> pd.DataFrame:
        """Generate all scheduled quarterly earnings releases for ``symbol``."""
        gen = self._stream_rng(_STREAM_FUNDAMENTALS, symbol)
        period_ends: list[pd.Timestamp] = []
        cursor = self._start
        # First period end aligned forward one quarter from the start.
        while cursor <= self._end:
            cursor = cursor + pd.Timedelta(days=_QUARTER_DAYS)
            period_ends.append(cursor)
        # Releases land _EARNINGS_RELEASE_LAG_DAYS after each period end.
        release_ts = [pe + pd.Timedelta(days=_EARNINGS_RELEASE_LAG_DAYS) for pe in period_ends]
        n = len(period_ends)
        eps = (_EPS_MEAN + _EPS_SIGMA * gen.standard_normal(n)) if n else np.array([])
        frame = pd.DataFrame(
            {
                "symbol": pd.Series([symbol] * n, dtype="object"),
                "period_end": pd.Series(period_ends, dtype="datetime64[ns]"),
                "release_ts": pd.Series(release_ts, dtype="datetime64[ns]"),
                "eps": pd.Series(np.asarray(eps, dtype="float64"), dtype="float64"),
            }
        )
        return frame

    def _full_news_frame(self, symbol: str) -> pd.DataFrame:
        """Generate all news items for ``symbol`` across the horizon."""
        gen = self._stream_rng(_STREAM_NEWS, symbol)
        published: list[pd.Timestamp] = []
        headlines: list[str] = []
        sentiments: list[float] = []
        for day in self._calendar:
            count = int(gen.poisson(_NEWS_RATE_PER_DAY))
            for k in range(count):
                hour = gen.uniform(_NEWS_HOUR_LOW, _NEWS_HOUR_HIGH)
                ts = day + pd.Timedelta(hours=float(hour))
                published.append(ts)
                headlines.append(f"{symbol} synthetic headline {day.date()}#{k}")
                sentiments.append(float(gen.uniform(-1.0, 1.0)))
        frame = pd.DataFrame(
            {
                "symbol": pd.Series([symbol] * len(published), dtype="object"),
                "published_ts": pd.Series(published, dtype="datetime64[ns]"),
                "headline": pd.Series(headlines, dtype="object"),
                "sentiment": pd.Series(sentiments, dtype="float64"),
            }
        )
        return frame.sort_values("published_ts", kind="stable").reset_index(drop=True)

    def _full_nbbo_frame(
        self, symbol: str, expiry: pd.Timestamp, strike: float
    ) -> pd.DataFrame:
        """Generate the full NBBO quote stream for one ``(symbol, expiry, strike)``."""
        # Sub-stream keyed also by the contract so different strikes differ.
        contract_key = f"{symbol}|{expiry.date()}|{strike:.4f}"
        gen = self._stream_rng(_STREAM_OPTIONS, contract_key)
        quote_ts: list[pd.Timestamp] = []
        bids: list[float] = []
        asks: list[float] = []
        bid_sizes: list[int] = []
        ask_sizes: list[int] = []
        ivs: list[float] = []
        # Only quote on trading days strictly before expiry.
        for day in self._calendar:
            if day >= expiry:
                continue
            for s in range(_OPTION_SNAPSHOTS_PER_DAY):
                hour = _NEWS_HOUR_LOW + (
                    (_NEWS_HOUR_HIGH - _NEWS_HOUR_LOW) * s / _OPTION_SNAPSHOTS_PER_DAY
                )
                ts = day + pd.Timedelta(hours=float(hour))
                # A positive synthetic mid anchored on the strike.
                mid = max(strike * (1.0 + 0.05 * gen.standard_normal()), 0.05)
                half = mid * _OPTION_HALF_SPREAD_FRAC
                bid = mid - half
                ask = mid + half
                iv = _OPTION_IV * (1.0 + 0.10 * abs(gen.standard_normal()))
                quote_ts.append(ts)
                bids.append(bid)
                asks.append(ask)
                bid_sizes.append(int(gen.integers(_OPTION_SIZE_LOW, _OPTION_SIZE_HIGH + 1)))
                ask_sizes.append(int(gen.integers(_OPTION_SIZE_LOW, _OPTION_SIZE_HIGH + 1)))
                ivs.append(float(iv))
        frame = pd.DataFrame(
            {
                "quote_ts": pd.Series(quote_ts, dtype="datetime64[ns]"),
                "bid": pd.Series(bids, dtype="float64"),
                "ask": pd.Series(asks, dtype="float64"),
                "bid_size": pd.Series(bid_sizes, dtype="int64"),
                "ask_size": pd.Series(ask_sizes, dtype="int64"),
                "iv": pd.Series(ivs, dtype="float64"),
            }
        )
        return frame.sort_values("quote_ts", kind="stable").reset_index(drop=True)

    def _full_internals_frame(self) -> pd.DataFrame:
        """Generate the full market-internals stream (not symbol-specific)."""
        gen = self._stream_rng(_STREAM_INTERNALS)
        ts_list: list[pd.Timestamp] = []
        tick: list[float] = []
        trin: list[float] = []
        add: list[float] = []
        vold: list[float] = []
        for day in self._calendar:
            for s in range(_INTERNALS_SNAPSHOTS_PER_DAY):
                hour = _NEWS_HOUR_LOW + (
                    (_NEWS_HOUR_HIGH - _NEWS_HOUR_LOW) * s / _INTERNALS_SNAPSHOTS_PER_DAY
                )
                ts = day + pd.Timedelta(hours=float(hour))
                ts_list.append(ts)
                tick.append(float(_TICK_SCALE * gen.standard_normal()))
                trin.append(float(np.exp(_TRIN_LOG_SIGMA * gen.standard_normal())))
                add.append(float(_ADD_SCALE * gen.standard_normal()))
                vold.append(float(_VOLD_SCALE * gen.standard_normal()))
        return pd.DataFrame(
            {
                "ts": pd.Series(ts_list, dtype="datetime64[ns]"),
                "tick": pd.Series(tick, dtype="float64"),
                "trin": pd.Series(trin, dtype="float64"),
                "add": pd.Series(add, dtype="float64"),
                "vold": pd.Series(vold, dtype="float64"),
            }
        )

    # -- PriceFeed ---------------------------------------------------------

    def history(
        self, symbol: str, *, asof: pd.Timestamp, start: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return ``[date, open, high, low, close, volume]`` with ``date <= asof``."""
        self._require_symbol(symbol)
        frame = self._full_price_frame(symbol)
        asof_ts = pd.Timestamp(asof)
        frame = frame.loc[frame["date"] <= asof_ts]
        if start is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        return frame.reset_index(drop=True)

    # -- FundamentalsFeed --------------------------------------------------

    def earnings(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return earnings rows with ``release_ts <= asof`` (release, not period end)."""
        self._require_symbol(symbol)
        frame = self._full_earnings_frame(symbol)
        asof_ts = pd.Timestamp(asof)
        frame = frame.loc[frame["release_ts"] <= asof_ts]
        return frame.reset_index(drop=True)

    # -- NewsFeed ----------------------------------------------------------

    def items(self, symbol: str, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return news rows with ``published_ts <= asof``."""
        self._require_symbol(symbol)
        frame = self._full_news_frame(symbol)
        asof_ts = pd.Timestamp(asof)
        frame = frame.loc[frame["published_ts"] <= asof_ts]
        return frame.reset_index(drop=True)

    # -- OptionsNBBOFeed ---------------------------------------------------

    def nbbo(
        self, symbol: str, *, asof: pd.Timestamp, expiry: pd.Timestamp, strike: float
    ) -> pd.DataFrame:
        """Return ``[quote_ts, bid, ask, bid_size, ask_size, iv]`` with ``quote_ts <= asof``."""
        self._require_symbol(symbol)
        frame = self._full_nbbo_frame(symbol, pd.Timestamp(expiry), float(strike))
        asof_ts = pd.Timestamp(asof)
        frame = frame.loc[frame["quote_ts"] <= asof_ts]
        return frame.reset_index(drop=True)

    # -- InternalsFeed -----------------------------------------------------

    def internals(self, *, asof: pd.Timestamp) -> pd.DataFrame:
        """Return internals rows with ``ts <= asof``."""
        frame = self._full_internals_frame()
        asof_ts = pd.Timestamp(asof)
        frame = frame.loc[frame["ts"] <= asof_ts]
        return frame.reset_index(drop=True)
