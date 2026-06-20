"""Build leakage-free training datasets for the directional swing classifier.

A :class:`Dataset` bundles everything a model and the validation layer need:

* ``X``    — numeric feature matrix (RangeIndex ``0..n-1``), no date / text columns;
* ``y``    — binary target (``1`` = profit-take touched first, ``0`` = stop first);
* ``t1``   — label end times (for purging / embargo), aligned to ``X``;
* ``w``    — López-de-Prado sample weights (mean ≈ 1), aligned to ``X``;
* ``meta`` — auxiliary columns ``[date, symbol, ret, label, t1]`` for analysis / attribution.

The build path is strictly point-in-time: features come from
:func:`startx.features.assemble.build_feature_matrix` (no ``fwd_*`` columns), labels from the
triple-barrier method. Features and labels are inner-joined on ``date`` per symbol, then
**timeout** labels (``label == 0``) are dropped — the directional classifier only learns the
sign of the *resolved* move. Sample weights are computed per symbol on the symbol's own bar
calendar (concurrency is only meaningful within a single instrument's timeline), so pooling
across tickers never inflates concurrency.

NaN policy
----------
Features are already trailing/PIT-warmed (rolling windows computed on full history, clipped at
the end), so residual NaNs are rare and live in a handful of low-coverage flow/regime columns.
We **drop** rows with any NaN feature when that costs < 2% of rows; otherwise we **median-fill**
per column (medians taken from the training-eligible rows only, i.e. the column as a whole — no
future-vs-past split is needed because medians are leakage-neutral location stats applied
uniformly). The chosen strategy is recorded in :attr:`Dataset.meta`'s ``attrs``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.cache import ParquetCache
from ..data.prices import get_prices
from ..data.universe import Universe, load_universe
from ..features.assemble import build_feature_matrix
from ..labeling.config import label_horizon
from ..labeling.weights import sample_weights
from ..settings import Settings, get_settings

#: Columns that are never features (identifiers / labels / text).
_NON_FEATURE = ("date", "symbol", "sector", "label", "ret", "t1", "touch", "upper", "lower")

#: Drop-NaN rows only while the cost stays under this fraction; else median-fill.
_MAX_DROP_FRAC = 0.02


@dataclass
class Dataset:
    """Model-ready bundle with a shared RangeIndex across ``X``/``y``/``t1``/``w``/``meta``."""

    X: pd.DataFrame
    y: pd.Series
    t1: pd.Series
    w: pd.Series
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        n = len(self.X)
        if not (len(self.y) == len(self.t1) == len(self.w) == len(self.meta) == n):
            raise ValueError("X, y, t1, w, meta must share one length")

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature column names."""
        return list(self.X.columns)

    def __len__(self) -> int:
        return len(self.X)


def _feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Numeric feature columns of a feature matrix (excludes identifiers/labels)."""
    cols = [c for c in matrix.columns if c not in _NON_FEATURE]
    numeric = matrix[cols].select_dtypes(include=[np.number]).columns
    return list(numeric)


def _build_one(
    ticker: str,
    start: str,
    end: str,
    horizon: str,
    *,
    client,
    cache: ParquetCache,
    universe: Universe,
    settings: Settings,
    refresh: bool,
) -> pd.DataFrame | None:
    """Per-ticker joined frame: features + labels + weights, timeouts dropped.

    Returns a frame carrying feature columns plus ``[date, symbol, sector, label, ret, t1, y,
    w]``, or ``None`` when the ticker yields no usable directional samples. Sample weights are
    computed here on the symbol's own bar calendar so cross-symbol pooling cannot distort
    label concurrency.
    """
    feats = build_feature_matrix(
        ticker, start, end, client=client, cache=cache, universe=universe,
        settings=settings, refresh=refresh,
    )
    if feats.empty:
        return None

    spec = universe.spec(ticker)
    prices = get_prices(client, cache, spec.fmp, settings.history_start, refresh)
    if prices.empty:
        return None
    labels = label_horizon(prices, horizon)
    if labels.empty:
        return None

    feats = feats.copy()
    labels = labels.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    joined = feats.merge(
        labels[["date", "t1", "label", "ret"]], on="date", how="inner"
    )
    if joined.empty:
        return None

    # Directional classifier: keep only resolved (non-timeout) labels.
    joined = joined[joined["label"] != 0].reset_index(drop=True)
    if joined.empty:
        return None

    joined["y"] = (joined["label"] == 1).astype(int)
    joined["t1"] = pd.to_datetime(joined["t1"])

    # Weights on the symbol's own calendar: bar_dates = entry dates; t1 indexed by entry date.
    entry_dates = pd.DatetimeIndex(joined["date"])
    t1_by_entry = pd.Series(joined["t1"].to_numpy(), index=entry_dates)
    ret_by_entry = pd.Series(joined["ret"].to_numpy(), index=entry_dates)
    if not entry_dates.is_unique:
        # Defensive: identical entry dates would collide in the weight index; keep last.
        keep = ~entry_dates.duplicated(keep="last")
        joined = joined[keep].reset_index(drop=True)
        entry_dates = pd.DatetimeIndex(joined["date"])
        t1_by_entry = pd.Series(joined["t1"].to_numpy(), index=entry_dates)
        ret_by_entry = pd.Series(joined["ret"].to_numpy(), index=entry_dates)

    w = sample_weights(entry_dates, t1_by_entry, ret=ret_by_entry)
    joined["w"] = w.to_numpy()
    return joined


def _finalize_features(matrix: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Apply the NaN policy to feature columns; return (clean X, strategy tag).

    Drops all-NaN feature columns first, then either drops NaN rows (cheap case) or
    median-fills per column. The strategy tag is recorded for transparency.
    """
    feats = matrix[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Remove feature columns that are entirely NaN (e.g. a flow block never populated).
    all_nan = feats.columns[feats.isna().all()].tolist()
    if all_nan:
        feats = feats.drop(columns=all_nan)

    n = len(feats)
    drop_mask = feats.isna().any(axis=1)
    drop_frac = float(drop_mask.mean()) if n else 0.0

    if n and drop_frac <= _MAX_DROP_FRAC:
        keep = ~drop_mask
        return feats[keep], f"drop_rows({drop_frac:.3%})"
    medians = feats.median(numeric_only=True)
    return feats.fillna(medians), f"median_fill(drop_would_be={drop_frac:.3%})"


def _assemble(
    frames: list[pd.DataFrame], *, pooled: bool
) -> Dataset:
    """Turn per-ticker joined frames into a single :class:`Dataset`."""
    matrix = pd.concat(frames, ignore_index=True)

    feature_cols = _feature_columns(matrix)

    if pooled:
        # Categorical identity codes become extra numeric features for the pooled model.
        matrix["symbol_code"] = matrix["symbol"].astype("category").cat.codes.astype(int)
        sector = matrix["sector"].astype("object").where(matrix["sector"].notna(), "UNKNOWN")
        matrix["sector_code"] = sector.astype("category").cat.codes.astype(int)
        feature_cols = feature_cols + ["symbol_code", "sector_code"]

    X, nan_strategy = _finalize_features(matrix, feature_cols)
    kept_idx = X.index  # rows surviving the NaN policy (subset of matrix rows)

    sub = matrix.loc[kept_idx].reset_index(drop=True)
    X = X.reset_index(drop=True)

    y = sub["y"].astype(int)
    t1 = pd.to_datetime(sub["t1"])
    w = sub["w"].astype(float)
    meta = sub[["date", "symbol", "ret", "label", "t1"]].copy()

    # Renormalize weights to mean 1 after row drops so downstream scaling is stable.
    if len(w) and w.sum() > 0:
        w = w * (len(w) / w.sum())

    # Share a clean RangeIndex across all members.
    rng = pd.RangeIndex(len(X))
    X.index = rng
    y.index = rng
    t1.index = rng
    w.index = rng
    meta.index = rng

    meta.attrs["nan_strategy"] = nan_strategy
    meta.attrs["pooled"] = pooled
    meta.attrs["feature_names"] = list(X.columns)
    return Dataset(X=X, y=y, t1=t1, w=w, meta=meta)


def build_dataset(
    tickers: list[str],
    start: str,
    end: str,
    horizon: str = "short",
    *,
    client,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    refresh: bool = False,
    pooled: bool = True,
) -> Dataset:
    """Build a model-ready :class:`Dataset` over ``tickers`` and ``[start, end]``.

    For each ticker: assemble PIT features, triple-barrier labels for ``horizon`` (``"short"``
    or ``"long"``), inner-join on ``date``, drop timeout labels, and compute sample weights on
    the symbol's own calendar. When ``pooled`` (default) all tickers are stacked and integer
    ``symbol_code`` / ``sector_code`` feature columns are added; otherwise the single-ticker
    frame is returned (still pooled-shaped if multiple tickers are passed). Use
    :func:`per_symbol_datasets` for separate per-ticker datasets.

    Raises ``ValueError`` if no ticker yields usable directional samples.
    """
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        frame = _build_one(
            ticker, start, end, horizon, client=client, cache=cache,
            universe=universe, settings=settings, refresh=refresh,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError(
            f"build_dataset produced no usable samples for {tickers!r} over [{start}, {end}]"
        )
    return _assemble(frames, pooled=pooled)


def per_symbol_datasets(
    tickers: list[str],
    start: str,
    end: str,
    horizon: str = "short",
    *,
    client,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    refresh: bool = False,
) -> dict[str, Dataset]:
    """One independent :class:`Dataset` per ticker (no pooling, no symbol/sector codes).

    Tickers that yield no usable samples are silently skipped.
    """
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    out: dict[str, Dataset] = {}
    for ticker in tickers:
        frame = _build_one(
            ticker, start, end, horizon, client=client, cache=cache,
            universe=universe, settings=settings, refresh=refresh,
        )
        if frame is not None and not frame.empty:
            out[ticker] = _assemble([frame], pooled=False)
    return out
