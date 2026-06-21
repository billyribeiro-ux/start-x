"""KEYSTONE pipeline — wire every committed qedge layer into one edge-discovery scan.

This module is the spine of the adversarial scanner. It threads a single symbol
through the whole stack — point-in-time data, point-in-time features, a
point-in-time regime conditioning column, triple-barrier labels with overlap
(uniqueness) weights, a Combinatorial Purged CV with the label-end times ``t1``
driving purging, a family-diverse meta-label classifier producing a *true*
out-of-sample take/skip return stream net of execution friction, and finally the
DSR/PBO survival gate with the number of configurations searched counted in a
shared :class:`~qedge.validation.TrialLedger`.

The governing principle is non-negotiable: **every strong backtest is guilty of
leakage until proven innocent.** Concretely that means:

* features and the regime label are computed causally (value at ``t`` uses only
  rows ``<= t`` — guaranteed by the feature registry and the HMM *filtered*
  inference), so no future bar perturbs a past conditioning value;
* events are labelled with the triple-barrier method and the resulting ``t1``
  feeds the purged + embargoed CPCV so a train label interval can never overlap a
  test label interval;
* the model is fit on train folds and interrogated only on held-out folds — the
  realized OOS return stream is assembled from held-out predictions exclusively;
* the verdict is the survival gate (Deflated Sharpe **and** Probability of
  Backtest Overfitting), deflated by the number of trials in the ledger so a wide
  search is penalised exactly as it should be;
* dimensions that have no data feed (options NBBO, market internals) are probed
  and honestly reported as *unpopulated* rather than silently skipped.

The default feed is the deterministic :class:`~qedge.data.synthetic.SyntheticMarket`
so a scan is byte-reproducible; a live :class:`~qedge.data.fmp_adapter.FMPPriceFeed`
(or any :class:`~qedge.data.protocols.PriceFeed`) can be injected instead.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

import qedge.features.technical  # noqa: F401  (import side effect: registers specs)
from qedge.config import QedgeConfig, get_config
from qedge.data.protocols import FeedNotAvailable, PriceFeed
from qedge.data.stub_adapters import (
    UnavailableInternalsFeed,
    UnavailableOptionsNBBOFeed,
)
from qedge.data.synthetic import SyntheticMarket
from qedge.data.universe import resolve_universe
from qedge.execution.capacity import break_even_aum
from qedge.execution.costs import EquityCostModel
from qedge.features.base import REGISTRY, compute_features
from qedge.labeling import label_long, label_short, uniqueness_weights
from qedge.modeling.ensemble import FamilyEnsembleClassifier
from qedge.modeling.importance import mda_importance_oos
from qedge.modeling.regime import HMMRegimeDetector
from qedge.repro import hash_frame, make_run_record, set_seeds
from qedge.validation import TrialLedger, make_cpcv, survival_gate

__all__ = [
    "EdgeResult",
    "ScanConfig",
    "scan_symbol",
    "run_scan",
]

#: The two horizons the desk runs as separate books (see the project memory).
Horizon = Literal["short", "long"]

#: Column name for the point-in-time regime state appended to the feature matrix.
_REGIME_COLUMN: Final[str] = "regime"

#: The meta-label positive class: the realized triple-barrier outcome was a win
#: (profit-taking barrier touched, ``label == +1``). Everything else (stop or
#: timeout) is the negative class. The classifier learns *when to take the bet*.
_WIN_LABEL: Final[int] = 1

#: The model's "take the bet" prediction. When the classifier predicts this the
#: event contributes its realized return to the OOS stream; otherwise it sits flat.
_TAKE_BET: Final[int] = 1

#: Basis points per unit return (1.0 == 10_000 bps); used to express the gross
#: OOS edge as the bps input the capacity model expects.
_BPS_PER_UNIT: Final[float] = 1.0e4

#: Honest-degradation labels for contract dimensions with no backing feed.
_DIM_OPTIONS_NBBO: Final[str] = "options_nbbo"
_DIM_INTERNALS: Final[str] = "internals"

#: A representative options contract probe (offset from ``asof``) used only to
#: confirm the NBBO feed is unavailable — never to fabricate a quote.
_OPTIONS_PROBE_EXPIRY_DAYS: Final[int] = 30
#: Top-feature cut for the report (D2 renders these); not a model threshold.
_TOP_FEATURES_K: Final[int] = 5

#: Capacity assumption (STATED, not fitted). The break-even AUM curve needs a
#: price and an average-daily-volume-in-shares anchor; with no live ADV feed we
#: state a conservative, liquid-ETF-scale assumption and document it on the
#: result. These drive only the reported capacity ceiling, never the gate.
_ASSUMED_PRICE_USD: Final[float] = 100.0
_ASSUMED_ADV_SHARES: Final[float] = 5.0e6


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """Small, explicit description of *what* to scan (the rest lives in QedgeConfig).

    Attributes
    ----------
    universe:
        Symbols to scan. Defaults to the configured seed ETF universe.
    horizon:
        Which book — ``"short"`` (1-10 day cap) or ``"long"`` (~63-day cap).
    history_start / history_end:
        Inclusive bounds of the data window pulled from the feed. Defaults to the
        locked analysis window (2019-01-01 -> 2026-06-30) so a default scan obeys
        the project memory without the caller restating it.
    """

    universe: tuple[str, ...]
    horizon: Horizon = "short"
    history_start: str = "2019-01-01"
    history_end: str = "2026-06-30"

    @classmethod
    def from_config(
        cls,
        config: QedgeConfig | None = None,
        *,
        universe: Sequence[str] | None = None,
        horizon: Horizon = "short",
    ) -> ScanConfig:
        """Build a :class:`ScanConfig`, defaulting the universe from ``cfg.data``."""
        cfg = config if config is not None else get_config()
        resolved = (
            tuple(universe)
            if universe is not None
            else tuple(resolve_universe(cfg).ordered)
        )
        return cls(universe=resolved, horizon=horizon)


@dataclass(frozen=True, slots=True)
class EdgeResult:
    """The verdict on one symbol/horizon — the unit D2's report consumes.

    Every field is a plain, serialisable scalar or list so the record round-trips
    cleanly through :class:`~qedge.repro.RunRecord` JSON. ``passed`` is the only
    field that should ever gate a downstream decision; the rest are evidence.
    """

    symbol: str
    horizon: str
    n_events: int
    n_trials: int
    oos_sharpe: float
    deflated_sharpe: float
    pbo: float
    passed: bool
    verdict: str
    break_even_aum_usd: float
    top_features: list[tuple[str, float]]
    unpopulated_dimensions: list[str]
    feature_snapshot_hash: str
    run_id: str


# ---------------------------------------------------------------------------
# Internal pipeline mechanics
# ---------------------------------------------------------------------------
def _resolve_feed(feed: PriceFeed | None, scan: ScanConfig, cfg: QedgeConfig) -> PriceFeed:
    """Return the supplied feed or the deterministic synthetic default.

    The synthetic market is the DEFAULT so a scan is reproducible with no live
    dependency; it is seeded from ``cfg.scanner.seed`` and generates exactly the
    requested universe over the requested window.
    """
    if feed is not None:
        return feed
    return SyntheticMarket(
        scan.history_start,
        scan.history_end,
        seed=cfg.scanner.seed,
        symbols=scan.universe,
        config=cfg,
    )


def _load_prices(feed: PriceFeed, symbol: str, scan: ScanConfig) -> pd.DataFrame:
    """Pull the full point-in-time OHLCV history for ``symbol`` over the window.

    ``asof`` is the window end: a daily bar is public at its close, so asking the
    feed for everything ``<= history_end`` yields the complete in-sample history
    while never returning a row stamped after the decision horizon.
    """
    asof = pd.Timestamp(scan.history_end)
    start = pd.Timestamp(scan.history_start)
    prices = feed.history(symbol, asof=asof, start=start)
    return prices.reset_index(drop=True)


def _feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the point-in-time feature matrix indexed by trading date.

    Every spec in the registry is a pure, trailing transform (value at ``t`` uses
    rows ``<= t`` only), so the matrix is causal row by row. It is reindexed by the
    bar ``date`` so events can be aligned by their entry date downstream.
    """
    features = compute_features(prices, REGISTRY.all_specs())
    features.index = pd.DatetimeIndex(prices["date"].to_numpy(), name="date")
    return features


def _add_regime_column(
    features: pd.DataFrame, cfg: QedgeConfig
) -> pd.DataFrame:
    """Fit an HMM on the (cleaned) features and append its FILTERED regime state.

    ``predict_pit`` runs the forward (alpha) recursion only, so the regime label
    at ``t`` consumes observations ``<= t`` — appending it cannot leak the future.
    The HMM is fit on the warmed-up rows (no NaNs) and inferred over the same rows;
    the regime column is then reindexed back onto the full matrix.
    """
    warmed = features.dropna()
    out = features.copy()
    if warmed.empty:
        out[_REGIME_COLUMN] = np.nan
        return out
    detector = HMMRegimeDetector(config=cfg).fit(warmed)
    regime = detector.predict_pit(warmed)
    out[_REGIME_COLUMN] = regime.reindex(out.index)
    return out


def _labels_for_horizon(prices: pd.DataFrame, horizon: Horizon) -> pd.DataFrame:
    """Triple-barrier labels for the requested book (short 10-day / long ~63-day)."""
    if horizon == "short":
        return label_short(prices)
    return label_long(prices)


#: Internal scratch column names (collision-proof prefixes) used only inside
#: :func:`_build_events` to carry label fields through the dropna alignment.
_RET_ALIGN_HELPER: Final[str] = "__qedge_ret__"
_T1_ALIGN_HELPER: Final[str] = "__qedge_t1__"
_LABEL_ALIGN_HELPER: Final[str] = "__qedge_label__"


def _build_events(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Align features at entry dates and form the meta-label task.

    Returns ``(X, y, t1, ret)`` all indexed by entry date, where:

    * ``X`` is the conditioning matrix at each event's entry bar (features +
      regime), with warm-up / unlabelled rows dropped so the model sees clean rows;
    * ``y`` is the meta-label — ``1`` iff the triple-barrier outcome was a win;
    * ``t1`` is the label-end time (drives purging in the CPCV);
    * ``ret`` is the realized event return (entry close -> first-touch close).
    """
    label_index = pd.DatetimeIndex(labels["date"].to_numpy(), name="date")
    aligned = features.reindex(label_index)
    aligned[_RET_ALIGN_HELPER] = labels["ret"].to_numpy()
    aligned[_T1_ALIGN_HELPER] = labels["t1"].to_numpy()
    aligned[_LABEL_ALIGN_HELPER] = labels["label"].to_numpy()
    clean = aligned.dropna()

    feature_cols = list(features.columns)
    x = clean[feature_cols].astype("float64")
    y = (clean[_LABEL_ALIGN_HELPER].to_numpy() == _WIN_LABEL).astype(np.int64)
    y_series = pd.Series(y, index=clean.index, name="y")
    t1 = pd.Series(
        pd.DatetimeIndex(clean[_T1_ALIGN_HELPER].to_numpy()),
        index=clean.index,
        name="t1",
    )
    ret = pd.Series(
        clean[_RET_ALIGN_HELPER].to_numpy(dtype=np.float64),
        index=clean.index,
        name="ret",
    )
    return x, y_series, t1, ret


def _oos_return_stream(
    x: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    ret: pd.Series,
    weights: pd.Series,
    cost_model: EquityCostModel,
    cfg: QedgeConfig,
) -> npt.NDArray[np.float64]:
    """Assemble the per-event OUT-OF-SAMPLE net return stream via purged CPCV.

    For every ``(train, test)`` split from the purged + embargoed CPCV a fresh
    :class:`FamilyEnsembleClassifier` is fit on the train events (sample-weighted
    by overlap uniqueness) and asked, on the *held-out* test events, whether to
    take each bet. Where it says take, that event contributes its realized return
    net of round-trip equity friction; where it says skip, the event sits flat
    (zero). Predictions come exclusively from held-out folds, so the stream is a
    genuine OOS measurement with no train data ever scored.

    A test event can appear in several folds (CPCV reuses each group across
    combinations); its OOS contributions are averaged so each event enters the
    stream exactly once.
    """
    cv = make_cpcv(t1)
    n = len(x)
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    realized = ret.to_numpy(dtype=np.float64)
    sample_w = weights.to_numpy(dtype=np.float64)

    for train_idx, test_idx in cv.split(x, y):
        y_train = y.iloc[train_idx]
        # A fold whose train slice carries a single class cannot calibrate a
        # take/skip decision; skip it rather than fabricate a degenerate model.
        if y_train.nunique() < 2 or test_idx.size == 0:
            continue
        model = FamilyEnsembleClassifier(cfg).fit(
            x.iloc[train_idx].to_numpy(dtype=np.float64),
            y_train.to_numpy(),
        )
        preds = model.predict(x.iloc[test_idx].to_numpy(dtype=np.float64))
        take = preds == _TAKE_BET
        # Realized return where we take the bet, net of round-trip cost; flat else.
        taken_ret = np.where(take, realized[test_idx], 0.0)
        turnover = np.where(take, 1.0, 0.0)
        net = cost_model.net_returns(taken_ret, turnover=turnover)
        sums[test_idx] += net
        counts[test_idx] += 1.0

    seen = counts > 0.0
    stream = np.zeros(int(seen.sum()), dtype=np.float64)
    # Average the (possibly multiple) OOS contributions per event.
    averaged = sums[seen] / counts[seen]
    stream[:] = averaged
    # Weight the stream by sample uniqueness so overlapping events do not
    # double-count their (correlated) returns in the Sharpe estimate.
    w = sample_w[seen]
    if w.sum() > 0.0:
        stream = stream * (w / w.mean())
    return np.asarray(stream, dtype=np.float64)


def _top_features(
    x: pd.DataFrame, y: pd.Series, t1: pd.Series, cfg: QedgeConfig
) -> list[tuple[str, float]]:
    """OOS clustered MDA importance, returned as the top features (name, score).

    Importance is measured purely out of sample (permute the held-out fold), per
    the importance module's contract. Each feature is its own singleton cluster so
    the returned index aligns 1:1 with the feature columns.
    """
    cv = make_cpcv(t1)

    def _factory() -> FamilyEnsembleClassifier:
        return FamilyEnsembleClassifier(cfg)

    try:
        importance = mda_importance_oos(
            _factory, x, y, cv=cv, scoring="accuracy", seed=cfg.scanner.seed
        )
    except ValueError:
        # No usable folds (e.g. too few events): no importance to report.
        return []
    columns = list(x.columns)
    ranked = importance.sort_values(ascending=False)
    out: list[tuple[str, float]] = []
    for cluster_id, score in ranked.head(_TOP_FEATURES_K).items():
        idx = int(cluster_id)
        name = columns[idx] if 0 <= idx < len(columns) else str(cluster_id)
        out.append((name, float(score)))
    return out


def _probe_unpopulated_dimensions(symbol: str, scan: ScanConfig) -> list[str]:
    """Probe the options/internals feeds; record the ones that are unavailable.

    The honest-degradation rule: never fabricate options NBBO or real-time
    internals. We *attempt* a point-in-time read with the stub adapters; if they
    raise :class:`FeedNotAvailable` the dimension is recorded as unpopulated so the
    report states plainly which contract dimensions this scan could not exercise.
    """
    asof = pd.Timestamp(scan.history_end)
    unpopulated: list[str] = []

    options: UnavailableOptionsNBBOFeed = UnavailableOptionsNBBOFeed()
    try:
        options.nbbo(
            symbol,
            asof=asof,
            expiry=asof + pd.Timedelta(days=_OPTIONS_PROBE_EXPIRY_DAYS),
            strike=_ASSUMED_PRICE_USD,
        )
    except FeedNotAvailable:
        unpopulated.append(_DIM_OPTIONS_NBBO)

    internals: UnavailableInternalsFeed = UnavailableInternalsFeed()
    try:
        internals.internals(asof=asof)
    except FeedNotAvailable:
        unpopulated.append(_DIM_INTERNALS)

    return unpopulated


def _per_obs_sharpe(stream: npt.NDArray[np.float64]) -> float:
    """Per-observation Sharpe of the OOS stream (NaN if undefined / degenerate)."""
    r = stream[~np.isnan(stream)]
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(r.mean() / sd)


def _empty_result(
    symbol: str,
    scan: ScanConfig,
    *,
    n_events: int,
    n_trials: int,
    feature_hash: str,
    unpopulated: list[str],
    cfg: QedgeConfig,
) -> EdgeResult:
    """Build a failing :class:`EdgeResult` when there is nothing to test.

    The verdict is the gate's own failure label so the report never confuses
    "no edge" with "not enough data" — both honestly read as not-robust.
    """
    record = make_run_record(
        config={"symbol": symbol, "horizon": scan.horizon, "n_trials": n_trials},
        data_hashes={"features": feature_hash},
        metrics={"n_events": n_events, "passed": False},
    )
    return EdgeResult(
        symbol=symbol,
        horizon=scan.horizon,
        n_events=n_events,
        n_trials=n_trials,
        oos_sharpe=float("nan"),
        deflated_sharpe=float("nan"),
        pbo=float("nan"),
        passed=False,
        verdict="LIKELY OVERFIT",
        break_even_aum_usd=0.0,
        top_features=[],
        unpopulated_dimensions=unpopulated,
        feature_snapshot_hash=feature_hash,
        run_id=record.run_id,
    )


def scan_symbol(
    symbol: str,
    *,
    feed: PriceFeed | None = None,
    horizon: Horizon = "short",
    trials: TrialLedger | None = None,
    config: QedgeConfig | None = None,
) -> EdgeResult:
    """Run the full adversarial pipeline for one symbol and return its verdict.

    The stages, in order: seed everything; load PIT prices; build the PIT feature
    matrix (hashed); append the PIT (filtered) regime state; triple-barrier label
    the events and form the meta-label task with overlap-uniqueness weights;
    assemble a *true* OOS take/skip return stream via purged CPCV net of execution
    friction; register the configuration in the trial ledger so multiple testing is
    counted; run the DSR/PBO survival gate; measure OOS feature importance; derive
    the capacity ceiling from the gross OOS edge under a stated ADV/price
    assumption; probe the options/internals feeds for unpopulated dimensions; and
    finally stamp a deterministic run id from the config + data snapshot.

    Parameters
    ----------
    symbol:
        The ticker to scan. Must be resolvable by ``feed``.
    feed:
        Any :class:`~qedge.data.protocols.PriceFeed`. Defaults to the deterministic
        :class:`~qedge.data.synthetic.SyntheticMarket` for reproducibility.
    horizon:
        ``"short"`` (1-10 day cap) or ``"long"`` (~63-day cap).
    trials:
        An optional shared :class:`~qedge.validation.TrialLedger`. When supplied,
        scanning several symbols accumulates the multiple-testing count across the
        whole search; when omitted a fresh single-trial ledger is used.
    config:
        Optional :class:`~qedge.config.QedgeConfig` override (process singleton by
        default).

    Returns
    -------
    EdgeResult
        The populated, serialisable verdict for this symbol/horizon.
    """
    cfg = config if config is not None else get_config()
    set_seeds(cfg.scanner.seed)

    scan = ScanConfig.from_config(cfg, universe=(symbol,), horizon=horizon)
    ledger = trials if trials is not None else TrialLedger()
    # Each (symbol, horizon) is one configuration tried — count it for DSR.
    ledger.register(f"{symbol}:{horizon}")

    resolved_feed = _resolve_feed(feed, scan, cfg)
    prices = _load_prices(resolved_feed, symbol, scan)

    features = _feature_matrix(prices)
    features = _add_regime_column(features, cfg)
    feature_hash = hash_frame(features)

    unpopulated = _probe_unpopulated_dimensions(symbol, scan)

    labels = _labels_for_horizon(prices, horizon)
    x, y, t1, ret = _build_events(features, labels)
    n_events = int(len(x))

    # Not enough labelled, clean events (or a single class) to test honestly.
    if n_events < cfg.validation.cpcv_n_groups or y.nunique() < 2:
        return _empty_result(
            symbol,
            scan,
            n_events=n_events,
            n_trials=ledger.count,
            feature_hash=feature_hash,
            unpopulated=unpopulated,
            cfg=cfg,
        )

    weights = uniqueness_weights(
        pd.DatetimeIndex(x.index), t1, ret=ret.abs()
    )
    cost_model = EquityCostModel.from_config(cfg.execution)
    stream = _oos_return_stream(x, y, t1, ret, weights, cost_model, cfg)

    gate = survival_gate(stream, n_trials=ledger.count)
    oos_sharpe = _per_obs_sharpe(stream)
    top_features = _top_features(x, y, t1, cfg)

    # Capacity ceiling from the GROSS OOS edge (mean event return, in bps) under a
    # STATED liquid-ETF price/ADV assumption (documented on the module constants).
    gross_edge_bps = float(np.nanmean(stream)) * _BPS_PER_UNIT
    capacity = break_even_aum(
        gross_edge_bps,
        price=_ASSUMED_PRICE_USD,
        adv_shares=_ASSUMED_ADV_SHARES,
    )

    record = make_run_record(
        config={
            "symbol": symbol,
            "horizon": horizon,
            "n_trials": ledger.count,
            "seed": cfg.scanner.seed,
            "history_start": scan.history_start,
            "history_end": scan.history_end,
        },
        data_hashes={"features": feature_hash},
        metrics={
            "n_events": n_events,
            "oos_sharpe": oos_sharpe,
            "deflated_sharpe": gate.deflated_sharpe,
            "pbo": gate.pbo,
            "passed": gate.passed,
            "break_even_aum_usd": capacity,
        },
    )

    return EdgeResult(
        symbol=symbol,
        horizon=horizon,
        n_events=n_events,
        n_trials=ledger.count,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=gate.deflated_sharpe,
        pbo=gate.pbo,
        passed=gate.passed,
        verdict=gate.verdict,
        break_even_aum_usd=capacity,
        top_features=top_features,
        unpopulated_dimensions=unpopulated,
        feature_snapshot_hash=feature_hash,
        run_id=record.run_id,
    )


def run_scan(
    universe: Sequence[str] | None = None,
    *,
    feed: PriceFeed | None = None,
    horizon: Horizon = "short",
    config: QedgeConfig | None = None,
) -> list[EdgeResult]:
    """Scan every symbol in ``universe``, sharing ONE trial ledger across the run.

    Sharing the ledger is the whole point: every symbol/horizon evaluated is an
    independent selection opportunity, so the Deflated Sharpe must be deflated by
    the *total* number of configurations searched across the scan — not reset per
    symbol. The ledger therefore accumulates as the loop proceeds and each
    symbol's :class:`EdgeResult` reports the running trial count at the time it was
    evaluated.

    Parameters
    ----------
    universe:
        Symbols to scan. Defaults to the configured seed ETF universe.
    feed:
        Shared :class:`~qedge.data.protocols.PriceFeed` (synthetic default).
    horizon:
        ``"short"`` or ``"long"`` — applied to every symbol in the scan.
    config:
        Optional config override (process singleton by default).

    Returns
    -------
    list[EdgeResult]
        One verdict per symbol, in universe order.
    """
    cfg = config if config is not None else get_config()
    scan = ScanConfig.from_config(cfg, universe=universe, horizon=horizon)
    ledger = TrialLedger()
    results: list[EdgeResult] = []
    for symbol in scan.universe:
        results.append(
            scan_symbol(
                symbol,
                feed=feed,
                horizon=horizon,
                trials=ledger,
                config=cfg,
            )
        )
    return results
