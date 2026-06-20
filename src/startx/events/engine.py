"""Orchestrator: turn (ticker, date range) into detected, attributed, learned-from events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from ..data.cache import ParquetCache
from ..data.prices import get_prices
from ..data.universe import SymbolSpec, Universe, load_universe
from ..fmp import endpoints as ep
from ..fmp.client import FMPClient
from ..settings import Settings, get_settings
from .attribute import attribute_event
from .detect import compute_signals, detect_events
from .learn import catalyst_reliability
from .study import event_study_caar


@dataclass
class EngineResult:
    ticker: str
    spec: SymbolSpec
    benchmark: str | None
    prices: pd.DataFrame          # signals over the scanned window
    events: pd.DataFrame          # detected events + attribution columns + `causes`
    caar: pd.DataFrame            # event-study average move profile
    reliability: pd.DataFrame     # learned catalyst reliability
    profile: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def _load_catalysts(
    client: FMPClient, cache: ParquetCache, spec: SymbolSpec, start: str, end: str,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    fmp = spec.fmp
    pad_start = (pd.Timestamp(start) - timedelta(days=7)).date().isoformat()
    pad_end = (pd.Timestamp(end) + timedelta(days=1)).date().isoformat()
    cats: dict[str, pd.DataFrame] = {}

    if spec.is_stock:
        cats["earnings"] = cache.get_or_fetch(
            f"catalysts/{fmp}/earnings", lambda: ep.earnings(client, fmp), refresh)
        cats["grades"] = cache.get_or_fetch(
            f"catalysts/{fmp}/grades", lambda: ep.analyst_grades(client, fmp), refresh)
        cats["price_target"] = cache.get_or_fetch(
            f"catalysts/{fmp}/price_target", lambda: ep.price_target_news(client, fmp), refresh)
        cats["insider"] = cache.get_or_fetch(
            f"catalysts/{fmp}/insider", lambda: ep.insider_trades(client, fmp), refresh)
        cats["senate"] = cache.get_or_fetch(
            f"catalysts/{fmp}/senate", lambda: ep.senate_trades(client, fmp), refresh)
        cats["house"] = cache.get_or_fetch(
            f"catalysts/{fmp}/house", lambda: ep.house_trades(client, fmp), refresh)

    cats["news"] = cache.get_or_fetch(
        f"catalysts/{fmp}/news_{start}_{end}",
        lambda: ep.stock_news(client, fmp, start=pad_start, end=pad_end), refresh)
    cats["macro"] = cache.get_or_fetch(
        f"catalysts/macro/econ_{start}_{end}",
        lambda: ep.economic_calendar(client, pad_start, pad_end), refresh)
    return cats


def _attach_attribution(
    events: pd.DataFrame, catalysts: dict[str, pd.DataFrame], lookback_days: int
) -> pd.DataFrame:
    if events.empty:
        events["causes"] = pd.Series(dtype=object)
        events["top_cause_type"] = None
        events["top_cause"] = None
        events["confidence"] = pd.Series(dtype=float)
        return events
    causes_col, types, labels, confs = [], [], [], []
    for _, row in events.iterrows():
        causes, top, conf = attribute_event(row, catalysts, lookback_days)
        causes_col.append(causes)
        types.append(top["type"] if top else None)
        labels.append(top["label"] if top else "Unattributed")
        confs.append(conf)
    events = events.copy()
    events["causes"] = causes_col
    events["top_cause_type"] = types
    events["top_cause"] = labels
    events["confidence"] = confs
    return events


def analyze_symbol(
    ticker: str,
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    ar_threshold: float = 2.5,
    vol_threshold: float | None = None,
    lookback_days: int = 2,
    refresh: bool = False,
) -> EngineResult:
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()
    spec = universe.spec(ticker)

    prices_full = get_prices(client, cache, spec.fmp, settings.history_start, refresh)
    if prices_full.empty:
        raise RuntimeError(f"No price data for {ticker} ({spec.fmp}).")

    market_full = None
    if spec.is_stock:
        market_full = get_prices(client, cache, universe.benchmark, settings.history_start, refresh)

    signals = compute_signals(prices_full, market_full)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    win = signals[(signals["date"] >= start_ts) & (signals["date"] <= end_ts)]
    events = detect_events(win, ar_threshold, vol_threshold)

    catalysts = _load_catalysts(client, cache, spec, start, end, refresh)
    events = _attach_attribution(events, catalysts, lookback_days)

    caar = event_study_caar(signals, events)
    reliability = catalyst_reliability(events)

    try:
        profile = ep.profile(client, spec.fmp)
    except Exception:
        profile = {}

    return EngineResult(
        ticker=ticker, spec=spec, benchmark=universe.benchmark if spec.is_stock else None,
        prices=win.reset_index(drop=True), events=events, caar=caar, reliability=reliability,
        profile=profile,
        params={"start": start, "end": end, "ar_threshold": ar_threshold,
                "vol_threshold": vol_threshold, "lookback_days": lookback_days},
    )
