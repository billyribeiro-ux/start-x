"""End-to-end strategy runner: dataset → walk-forward OOS predictions → backtest → scoreboard.

Ties the committed layers together for both a POOLED cross-sectional model and PER-SYMBOL
models (long & short). Everything reported here is out-of-sample: predictions come only from
:func:`startx.validation.walkforward.walk_forward_predict`, and the equity curve is built from
realized triple-barrier exits via :mod:`startx.backtest.engine`.

Critical detail: ``build_dataset(pooled=True)`` stacks tickers block-wise, so we re-sort by
``date`` before the positional walk-forward — otherwise folds would not be time-ordered.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import BacktestResult, backtest_signals, portfolio_backtest
from ..data.cache import ParquetCache
from ..data.prices import get_prices
from ..data.universe import Universe, load_universe
from ..models.dataset import Dataset, build_dataset, per_symbol_datasets
from ..models.train import model_factory
from ..settings import Settings, get_settings
from ..validation.report import scoreboard
from ..validation.walkforward import walk_forward_predict


@dataclass
class StrategyResult:
    label: str
    portfolio: BacktestResult
    per_symbol: dict[str, BacktestResult]
    scoreboard: dict
    n_oos: int
    oos_auc: float
    params: dict = field(default_factory=dict)


def _sorted_arrays(ds: Dataset):
    """Return X, y, t1, meta re-sorted by date (X/y/meta on a fresh RangeIndex).

    ``t1`` KEEPS its entry-date index (it is *not* reset) — `walk_forward_predict` reads that
    datetime index to purge train labels whose span overlaps the test block. Resetting it to a
    RangeIndex (the old bug) silently disabled the López-de-Prado purge, falling back to a bare
    positional embargo and leaking multi-bar (long/position-horizon) labels across the seam.
    """
    order = np.argsort(pd.to_datetime(ds.meta["date"]).to_numpy(), kind="stable")
    X = ds.X.iloc[order].reset_index(drop=True)
    y = ds.y.iloc[order].reset_index(drop=True)
    t1 = ds.t1.iloc[order]  # entry-date index preserved (sorted) for label-span purging
    meta = ds.meta.iloc[order].reset_index(drop=True)
    return X, y, t1, meta


def _auc(preds: pd.DataFrame) -> float:
    if preds.empty or pd.Series(preds["y_true"]).nunique() < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(preds["y_true"].astype(int), preds["y_prob"]))


def _adaptive(n: int, train_min: int | None, test_span: int | None) -> tuple[int, int]:
    train_min = train_min if train_min is not None else max(500, int(0.40 * n))
    test_span = test_span if test_span is not None else max(100, int(0.10 * n))
    return train_min, test_span


def _backtests_from_preds(
    preds: pd.DataFrame, meta: pd.DataFrame, tickers, *, client, cache, settings, universe,
    long_th, short_th, costs,
) -> dict[str, BacktestResult]:
    pmeta = meta.loc[preds.index]
    out: dict[str, BacktestResult] = {}
    for sym in tickers:
        mask = (pmeta["symbol"] == sym).to_numpy()
        if not mask.any():
            continue
        idx = preds.index[mask]
        sm = meta.loc[idx]
        dates = pd.to_datetime(sm["date"].to_numpy())
        predictions = pd.DataFrame({"prob_up": preds.loc[idx, "y_prob"].to_numpy()}, index=dates)
        labels = pd.DataFrame({
            "date": dates, "t1": pd.to_datetime(sm["t1"].to_numpy()),
            "ret": sm["ret"].to_numpy(), "label": sm["label"].to_numpy(),
        })
        prices = get_prices(client, cache, universe.spec(sym).fmp, settings.history_start)
        out[sym] = backtest_signals(
            predictions, labels, prices=prices[["date", "close"]],
            long_th=long_th, short_th=short_th, costs=costs,
        )
    return out


def _ctx(client, cache, universe, settings):
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()
    return cache, universe, settings


def run_pooled(
    tickers: list[str], start: str, end: str, horizon: str = "short", *, client,
    cache: ParquetCache | None = None, universe: Universe | None = None,
    settings: Settings | None = None, train_min: int | None = None,
    test_span: int | None = None, embargo_pct: float = 0.02, long_th: float = 0.6,
    short_th: float = 0.4, n_trials: int = 1, costs: CostModel | None = None,
    refresh: bool = False,
) -> StrategyResult:
    cache, universe, settings = _ctx(client, cache, universe, settings)
    costs = costs or CostModel()
    ds = build_dataset(tickers, start, end, horizon, client=client, cache=cache,
                       universe=universe, settings=settings, refresh=refresh, pooled=True)
    X, y, t1, meta = _sorted_arrays(ds)
    tm, ts = _adaptive(len(X), train_min, test_span)
    preds = walk_forward_predict(X, y, t1, model_factory(), train_min=tm, test_span=ts,
                                 embargo_pct=embargo_pct, proba=True)
    if preds.empty:
        raise RuntimeError("walk-forward produced no OOS predictions (need more data)")
    per_symbol = _backtests_from_preds(preds, meta, tickers, client=client, cache=cache,
                                       settings=settings, universe=universe, long_th=long_th,
                                       short_th=short_th, costs=costs)
    port = portfolio_backtest(per_symbol)
    rets = port.equity.pct_change().dropna()
    sb = scoreboard(rets, n_trials=n_trials, periods=252)
    return StrategyResult("pooled", port, per_symbol, sb, len(preds), _auc(preds),
                          params={"horizon": horizon, "train_min": tm, "test_span": ts,
                                  "embargo_pct": embargo_pct, "long_th": long_th,
                                  "short_th": short_th, "n_trials": n_trials})


def run_per_symbol(
    tickers: list[str], start: str, end: str, horizon: str = "short", *, client,
    cache: ParquetCache | None = None, universe: Universe | None = None,
    settings: Settings | None = None, train_min: int | None = None,
    test_span: int | None = None, embargo_pct: float = 0.02, long_th: float = 0.6,
    short_th: float = 0.4, n_trials: int = 1, costs: CostModel | None = None,
    refresh: bool = False,
) -> StrategyResult:
    cache, universe, settings = _ctx(client, cache, universe, settings)
    costs = costs or CostModel()
    datasets = per_symbol_datasets(tickers, start, end, horizon, client=client, cache=cache,
                                   universe=universe, settings=settings, refresh=refresh)
    per_symbol: dict[str, BacktestResult] = {}
    all_preds = []
    for sym, ds in datasets.items():
        X, y, t1, meta = _sorted_arrays(ds)
        tm, ts = _adaptive(len(X), train_min, test_span)
        if len(X) < tm + ts:
            continue
        preds = walk_forward_predict(X, y, t1, model_factory(), train_min=tm, test_span=ts,
                                     embargo_pct=embargo_pct, proba=True)
        if preds.empty:
            continue
        bts = _backtests_from_preds(preds, meta, [sym], client=client, cache=cache,
                                    settings=settings, universe=universe, long_th=long_th,
                                    short_th=short_th, costs=costs)
        per_symbol.update(bts)
        all_preds.append(preds)
    if not per_symbol:
        raise RuntimeError("no per-symbol model had enough data for walk-forward")
    port = portfolio_backtest(per_symbol)
    rets = port.equity.pct_change().dropna()
    sb = scoreboard(rets, n_trials=n_trials, periods=252)
    combined = pd.concat(all_preds) if all_preds else pd.DataFrame()
    return StrategyResult("per_symbol", port, per_symbol, sb, len(combined), _auc(combined),
                          params={"horizon": horizon, "embargo_pct": embargo_pct,
                                  "long_th": long_th, "short_th": short_th, "n_trials": n_trials})
