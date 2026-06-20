"""Assemble the full point-in-time feature matrix for the swing-trading model.

``build_feature_matrix`` pulls full price history for the ticker (plus its benchmark and VIX
for context), computes PIT signals, and joins technical + flow + regime features on ``date``,
then clips to the requested window. By computing every rolling feature on the *full* history and
only clipping at the end, the earliest in-window rows still have correctly-warmed trailing
windows. No ``fwd_*`` column is ever propagated.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..data.cache import ParquetCache
from ..data.catalysts import load_catalysts
from ..data.intermarket import get_context_prices
from ..data.prices import get_prices
from ..data.universe import Universe, load_universe
from ..events.detect import compute_signals
from ..fmp import endpoints as ep
from ..fmp.client import FMPClient
from ..settings import Settings, get_settings
from .flow import flow_features
from .intermarket import intermarket_features
from .regime import regime_features
from .technical import technical_features

_FWD_PREFIX = "fwd_"

#: Bump whenever feature definitions change so stale cached matrices are not reused.
FEATURE_VERSION = 3


def _drop_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive guard: strip any forward-looking column before it can leak into features."""
    bad = [c for c in df.columns if c.startswith(_FWD_PREFIX)]
    return df.drop(columns=bad) if bad else df


def build_feature_matrix(
    ticker: str,
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Full PIT feature matrix for ``ticker`` over ``[start, end]`` (one row per trading day)."""
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()
    spec = universe.spec(ticker)

    # Feature compute is heavy (~minutes/ticker over full history); cache the assembled
    # matrix per (symbol, window, feature-version) so repeated builds are near-free.
    cache_key = f"features/{spec.fmp}/{start}__{end}__v{FEATURE_VERSION}"
    if not refresh:
        cached = cache.load(cache_key)
        if cached is not None:
            return cached

    prices = get_prices(client, cache, spec.fmp, settings.history_start, refresh)
    if prices.empty:
        return pd.DataFrame()

    # Benchmark drives the market-model signals for single stocks; regime always wants it.
    bench_prices = get_prices(client, cache, universe.benchmark, settings.history_start, refresh)
    market = bench_prices if spec.is_stock else None

    vix_fmp = universe.context.get("VIX")
    vix_prices = (
        get_prices(client, cache, vix_fmp, settings.history_start, refresh)
        if vix_fmp else pd.DataFrame()
    )

    signals = compute_signals(prices, market)
    signals = _drop_forward(signals)

    # Feature blocks — all keyed on `date`, all trailing/PIT.
    tech = technical_features(signals)
    dates = signals["date"]
    catalysts = load_catalysts(client, cache, spec, start, end, refresh)
    flow = flow_features(dates, catalysts)
    regime = regime_features(dates, bench_prices, vix_prices)

    matrix = tech.merge(flow, on="date", how="left").merge(regime, on="date", how="left")

    # Intermarket macro context (yields/credit/sector-RS/dollar/gold) — drives index direction.
    context = get_context_prices(client, cache, settings.history_start, refresh)
    inter = intermarket_features(dates, context, bench_prices)
    matrix = matrix.merge(inter, on="date", how="left")
    matrix = _drop_forward(matrix)

    # Clip to the requested window only at the end so trailing windows are fully warmed.
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix = matrix[(matrix["date"] >= start_ts) & (matrix["date"] <= end_ts)].reset_index(drop=True)

    matrix.insert(1, "symbol", ticker)
    sector = None
    try:
        prof = ep.profile(client, spec.fmp)
        sector = prof.get("sector") if isinstance(prof, dict) else None
    except Exception:
        sector = None
    matrix.insert(2, "sector", sector)

    result = _drop_forward(matrix)
    cache.save(cache_key, result)
    return result


def build_pooled_matrix(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Concatenate per-ticker feature matrices into one pooled (stacked) frame."""
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        m = build_feature_matrix(
            ticker, start, end, client=client, cache=cache, universe=universe,
            settings=settings, refresh=refresh,
        )
        if not m.empty:
            frames.append(m)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
