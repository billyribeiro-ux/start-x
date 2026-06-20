"""Forward / paper-testing engine: train on the past, paper-trade forward (strict no-lookahead).

The contract, bar by bar
------------------------
1. **Train once, on the past only.** :func:`forward_test` calls
   :func:`startx.models.dataset.build_dataset` over ``[train_start, train_end]`` and fits a
   fresh :func:`startx.models.train.make_model`. The fitted model is frozen; it never sees a
   single feature row or label from the forward window.
2. **Score forward, one bar at a time.** For each ticker we assemble the PIT feature matrix and
   the triple-barrier labels over ``[test_start, test_end]``. We walk the in-window dates in
   order. While flat for that symbol (``one_at_a_time``), we read ``prob_up`` from the model on
   the *current* bar's features and decide a side: ``prob_up >= long_th`` -> LONG,
   ``prob_up <= short_th`` -> SHORT, else nothing.
3. **Enter on the next bar's open.** A signal seen on bar ``t`` is filled at ``open[t+1]`` — the
   decision uses only data available at ``t``, and the fill price is the first price strictly
   *after* the decision. No bar's own outcome ever informs its entry.
4. **Exit at the label ``t1``.** The position is closed at ``close[t1]`` for the entry bar's
   triple-barrier label (``t1`` is, by construction, on or after the entry date). If ``t1`` runs
   past ``test_end`` the trade is left OPEN and marked-to-market at the last in-window close.
5. **Costs once per round trip.** Gross entry→exit return (signed by side) has the round-trip
   cost (:class:`startx.backtest.costs.CostModel`) subtracted exactly once.

Everything returned is plain pandas so the same service backs the Streamlit page and any later
API. No global state, no network beyond the injected FMP ``client``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..data.cache import ParquetCache
from ..data.universe import Universe, load_universe
from ..features.assemble import build_feature_matrix
from ..labeling.config import label_horizon
from ..models.dataset import build_dataset
from ..models.train import fit_model, make_model
from ..settings import Settings, get_settings
from ..validation.metrics import max_drawdown, profit_factor, sharpe

#: Columns that are never model features (identifiers / labels / targets / weights / text).
#: Mirrors ``startx.models.dataset._NON_FEATURE`` so forward features align with training.
_NON_FEATURE = (
    "date", "symbol", "sector", "label", "ret", "t1", "touch", "upper", "lower", "y", "w",
)

#: Stable blotter column order (every PaperTrade field), used for empty frames too.
BLOTTER_COLUMNS = [
    "symbol", "side", "trade_type", "entry_date", "entry_price", "exit_date", "exit_price",
    "pnl_pct", "pnl_cash", "status", "bars_held", "prob_up",
]


@dataclass
class PaperTrade:
    """One paper position — open or closed — with realised or mark-to-market P&L.

    ``side`` is ``'long'`` or ``'short'``; ``trade_type`` is the human-readable reason
    (``'long'`` / ``'short'``, optionally suffixed with the dominant catalyst label when
    ``tag_reason`` is on). For an OPEN trade ``exit_date``/``exit_price`` hold the
    mark-to-market date/price and ``pnl_*`` the unrealised P&L; ``status`` distinguishes the two.
    """

    symbol: str
    side: str
    trade_type: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None
    pnl_cash: float | None = None
    status: str = "open"
    bars_held: int = 0
    prob_up: float = float("nan")

    def as_row(self) -> dict:
        """Flat dict in :data:`BLOTTER_COLUMNS` order (for DataFrame assembly)."""
        row = asdict(self)
        return {k: row[k] for k in BLOTTER_COLUMNS}


def _feature_columns(matrix: pd.DataFrame, model_features: list[str]) -> list[str]:
    """Forward feature columns that match the trained model's expected columns.

    We intersect the assembled matrix with the model's training feature names (preserving the
    model's order) so the scored frame is column-for-column what the model was fit on. Pooled
    ``symbol_code`` / ``sector_code`` columns the forward matrix lacks are filled with 0.
    """
    cols = [c for c in matrix.columns if c not in _NON_FEATURE]
    numeric = set(matrix[cols].select_dtypes(include=[np.number]).columns)
    return [c for c in model_features if c in numeric]


def _score_frame(
    feats: pd.DataFrame, model, model_features: list[str]
) -> pd.DataFrame:
    """Return ``[date, prob_up]`` for every in-window bar, scored by the frozen model.

    The feature block is reindexed to the model's exact training columns (missing pooled
    identity columns -> 0), median-filled for residual NaNs, and run through ``predict_proba``.
    No label / outcome column is ever read here.
    """
    if feats.empty:
        return pd.DataFrame(columns=["date", "prob_up"])
    usable = _feature_columns(feats, model_features)
    X = feats[usable].apply(pd.to_numeric, errors="coerce")
    # Align to the model's full training column set; pooled identity codes default to 0.
    X = X.reindex(columns=model_features)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    proba = model.predict_proba(X)
    # Column 1 is P(class == 1) == P(up / profit-take touched first), matching the dataset target.
    prob_up = np.asarray(proba)[:, 1]
    return pd.DataFrame({"date": pd.to_datetime(feats["date"]).to_numpy(), "prob_up": prob_up})


def _reason_tag(
    symbol: str,
    entry_date: pd.Timestamp,
    side: str,
    catalysts: dict | None,
    lookback_days: int = 2,
) -> str:
    """Dominant catalyst label for an entry, via :func:`attribute_event` (best-effort).

    Returns the catalyst ``type`` of the top cause in ``[entry_date - lookback, entry_date]``,
    or ``''`` when nothing attributes. Never raises — attribution is decoration, not P&L.
    """
    if not catalysts:
        return ""
    try:
        from ..events.attribute import attribute_event

        direction = "up" if side == "long" else "down"
        ev = pd.Series({"date": entry_date, "direction": direction})
        _causes, top, _conf = attribute_event(ev, catalysts, lookback_days=lookback_days)
        return str(top["type"]) if top else ""
    except Exception:  # noqa: BLE001 — attribution is optional cosmetics; never break a trade.
        return ""


def _walk_symbol(
    symbol: str,
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    model_features: list[str],
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    long_th: float,
    short_th: float,
    capital_per_trade: float,
    costs: CostModel,
    one_at_a_time: bool,
    catalysts: dict | None,
) -> list[PaperTrade]:
    """Paper-trade one symbol forward; return its ordered list of :class:`PaperTrade`.

    Strict PIT: a bar-``t`` signal fills at ``open[t+1]`` and exits at ``close[t1]`` where
    ``t1 >= entry_date``. While ``one_at_a_time`` and a position is open, candidate signals are
    skipped until the open trade's exit date has passed.
    """
    scored = _score_frame(feats, model, model_features)
    if scored.empty:
        return []

    # Per-symbol price calendar (date-indexed) for next-bar-open entries and t1 exits.
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    date_arr = px["date"].to_numpy()
    open_arr = px["open"].astype(float).to_numpy()
    close_arr = px["close"].astype(float).to_numpy()
    pos_of_date = {d: i for i, d in enumerate(date_arr)}

    # Label lookup keyed by entry date -> (t1, label, ret).
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"])
    lab["t1"] = pd.to_datetime(lab["t1"])
    label_by_date = {row.date: (row.t1, int(row.label)) for row in lab.itertuples()}

    cost = costs.round_trip_cost()
    # Mark open positions at the last close ON OR BEFORE test_end — never peek past the window.
    in_window = np.flatnonzero(date_arr <= np.datetime64(test_end))
    last_close_pos = int(in_window[-1]) if in_window.size else len(close_arr) - 1
    last_close_date = pd.Timestamp(date_arr[last_close_pos])
    last_close_price = float(close_arr[last_close_pos])

    trades: list[PaperTrade] = []
    block_until: pd.Timestamp | None = None  # while set, stay flat through this exit date.

    scored = scored.sort_values("date").reset_index(drop=True)
    for srow in scored.itertuples():
        sig_date = pd.Timestamp(srow.date)
        if sig_date < test_start or sig_date > test_end:
            continue
        prob_up = float(srow.prob_up)

        if one_at_a_time and block_until is not None and sig_date <= block_until:
            continue  # a position is still open on this candidate bar.

        if prob_up >= long_th:
            side = "long"
        elif prob_up <= short_th:
            side = "short"
        else:
            continue

        # Entry on the NEXT bar's open (strictly after the decision bar). No fill if none exists.
        sig_pos = pos_of_date.get(sig_date)
        if sig_pos is None or sig_pos + 1 > last_close_pos:
            continue
        entry_pos = sig_pos + 1
        entry_date = pd.Timestamp(date_arr[entry_pos])
        entry_price = float(open_arr[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # Exit at the label's t1 (or mark-to-market open at the last in-window close).
        t1_label = label_by_date.get(sig_date)
        t1_date = t1_label[0] if t1_label is not None else None

        trade_type = side
        if catalysts is not None:
            tag = _reason_tag(symbol, entry_date, side, catalysts)
            if tag:
                trade_type = f"{side}:{tag}"

        sign = 1.0 if side == "long" else -1.0

        if t1_date is not None and t1_date <= test_end and t1_date in pos_of_date:
            exit_pos = pos_of_date[t1_date]
            # Defensive PIT guard: an exit can never precede its entry.
            if exit_pos < entry_pos:
                exit_pos = entry_pos
            exit_date = pd.Timestamp(date_arr[exit_pos])
            exit_price = float(close_arr[exit_pos])
            gross = sign * (exit_price / entry_price - 1.0)
            pnl_pct = gross - cost
            trades.append(PaperTrade(
                symbol=symbol, side=side, trade_type=trade_type,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=exit_date, exit_price=exit_price,
                pnl_pct=pnl_pct, pnl_cash=pnl_pct * capital_per_trade,
                status="closed", bars_held=exit_pos - entry_pos, prob_up=prob_up,
            ))
            block_until = exit_date
        else:
            # Still open at test_end: mark to market at the last in-window close (costs not yet
            # fully realised, but we still net the round-trip drag for a like-for-like MtM).
            mark_pos = last_close_pos
            mark_date = last_close_date
            mark_price = last_close_price
            gross = sign * (mark_price / entry_price - 1.0)
            pnl_pct = gross - cost
            trades.append(PaperTrade(
                symbol=symbol, side=side, trade_type=trade_type,
                entry_date=entry_date, entry_price=entry_price,
                exit_date=mark_date, exit_price=mark_price,
                pnl_pct=pnl_pct, pnl_cash=pnl_pct * capital_per_trade,
                status="open", bars_held=mark_pos - entry_pos, prob_up=prob_up,
            ))
            # One open position per symbol blocks the rest of the window.
            block_until = test_end
            break

    return trades


def _equity_curve(
    trades: list[PaperTrade], capital_per_trade: float, test_start, test_end
) -> pd.Series:
    """Daily cumulative cash P&L credited on each trade's exit/mark date.

    P&L is realised on the exit date (never at entry), so the curve only steps on exit days. The
    series spans ``[test_start, test_end]`` business days, forward-filled to a running total.
    """
    idx = pd.bdate_range(pd.Timestamp(test_start), pd.Timestamp(test_end))
    if len(idx) == 0:
        idx = pd.DatetimeIndex([pd.Timestamp(test_start)])
    daily = pd.Series(0.0, index=idx)
    for t in trades:
        if t.pnl_cash is None:
            continue
        exit_d = pd.Timestamp(t.exit_date)
        # Snap to the next available curve day (exit may land on a non-bday).
        loc = idx.searchsorted(exit_d)
        loc = min(loc, len(idx) - 1)
        daily.iloc[loc] += float(t.pnl_cash)
    return daily.cumsum()


def _summarize(trades: list[PaperTrade], capital_per_trade: float) -> dict:
    """Headline summary dict over the blotter (out-of-sample stats only)."""
    n_trades = len(trades)
    n_open = sum(1 for t in trades if t.status == "open")
    pnls_pct = np.array([t.pnl_pct for t in trades if t.pnl_pct is not None], dtype=float)
    pnls_cash = np.array([t.pnl_cash for t in trades if t.pnl_cash is not None], dtype=float)
    bars = np.array([t.bars_held for t in trades], dtype=float)

    total_pnl_cash = float(pnls_cash.sum()) if pnls_cash.size else 0.0
    deployed = capital_per_trade * n_trades
    total_return = float(total_pnl_cash / deployed) if deployed > 0 else float("nan")
    hit = float(np.mean(pnls_pct > 0)) if pnls_pct.size else float("nan")
    return {
        "n_trades": n_trades,
        "n_open": n_open,
        "hit_rate": hit,
        "total_pnl_cash": total_pnl_cash,
        "total_return": total_return,
        "sharpe": sharpe(pnls_pct) if pnls_pct.size > 1 else float("nan"),
        "max_drawdown": max_drawdown(pnls_pct, is_returns=True) if pnls_pct.size else float("nan"),
        "profit_factor": profit_factor(pnls_pct) if pnls_pct.size else float("nan"),
        "avg_bars_held": float(bars.mean()) if bars.size else float("nan"),
    }


def _blotter_frame(trades: list[PaperTrade]) -> pd.DataFrame:
    """Assemble the full blotter DataFrame (stable schema even when empty)."""
    if not trades:
        return pd.DataFrame(columns=BLOTTER_COLUMNS)
    df = pd.DataFrame([t.as_row() for t in trades], columns=BLOTTER_COLUMNS)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def forward_test(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    *,
    client,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    horizon: str = "short",
    long_th: float = 0.6,
    short_th: float = 0.4,
    capital_per_trade: float = 10_000.0,
    costs: CostModel = CostModel(),
    one_at_a_time: bool = True,
    tag_reason: bool = False,
) -> dict:
    """Train on ``[train_start, train_end]``, paper-trade ``[test_start, test_end]`` forward.

    Strict point-in-time: the model is fit once on the training window and never sees the
    forward data; signals on bar ``t`` fill at ``open[t+1]`` and exit at the triple-barrier
    ``t1`` (or mark open at ``test_end``); the round-trip :class:`CostModel` drag is netted once
    per trade. ``long_th`` / ``short_th`` gate long / short entries on ``prob_up`` (P(up)).

    Returns a dict with:

    * ``blotter`` — DataFrame of every :class:`PaperTrade` (closed + open), schema
      :data:`BLOTTER_COLUMNS`.
    * ``open_positions`` — the still-open subset, marked-to-market at ``test_end``.
    * ``equity`` — daily cumulative cash-P&L Series over the forward window.
    * ``summary`` — dict: ``n_trades, n_open, hit_rate, total_pnl_cash, total_return, sharpe,
      max_drawdown, profit_factor, avg_bars_held``.
    """
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)

    # (a) Train ONCE on the past. The model never sees the forward window.
    ds = build_dataset(
        tickers, train_start, train_end, horizon,
        client=client, cache=cache, universe=universe, settings=settings,
    )
    model = make_model()
    fit_model(model, ds.X, ds.y, ds.w)
    model_features = list(ds.X.columns)

    # (b) Walk each ticker forward.
    all_trades: list[PaperTrade] = []
    for ticker in tickers:
        spec = universe.spec(ticker)
        feats = build_feature_matrix(
            ticker, test_start, test_end, client=client, cache=cache,
            universe=universe, settings=settings,
        )
        if feats.empty:
            continue
        from ..data.prices import get_prices

        prices = get_prices(client, cache, spec.fmp, settings.history_start)
        if prices.empty:
            continue
        labels = label_horizon(prices, horizon)
        if labels.empty:
            continue

        catalysts = None
        if tag_reason:
            try:
                from ..data.catalysts import load_catalysts

                catalysts = load_catalysts(client, cache, spec, test_start, test_end)
            except Exception:  # noqa: BLE001 — tagging is optional decoration.
                catalysts = None

        trades = _walk_symbol(
            ticker, feats, labels, prices, model, model_features,
            test_start=ts_start, test_end=ts_end,
            long_th=long_th, short_th=short_th,
            capital_per_trade=capital_per_trade, costs=costs,
            one_at_a_time=one_at_a_time, catalysts=catalysts,
        )
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: (pd.Timestamp(t.entry_date), t.symbol))

    blotter = _blotter_frame(all_trades)
    open_positions = blotter[blotter["status"] == "open"].reset_index(drop=True)
    equity = _equity_curve(all_trades, capital_per_trade, ts_start, ts_end)
    summary = _summarize(all_trades, capital_per_trade)

    return {
        "blotter": blotter,
        "open_positions": open_positions,
        "equity": equity,
        "summary": summary,
    }


def live_signals(
    tickers: list[str],
    asof: str,
    lookback_model_years: int = 3,
    *,
    client,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    horizon: str = "short",
    long_th: float = 0.6,
    short_th: float = 0.4,
) -> dict:
    """The "market-open" hook: train on a trailing window, score *today* (``asof``).

    Trains on ``[asof - lookback_model_years, asof)`` (the training window ends strictly
    *before* ``asof`` — the model never sees ``asof`` data) and computes each ticker's most
    recent feature row with ``date <= asof``, returning the current up-probability and the
    would-be entry side (``long`` / ``short`` / ``flat``). This is intentionally a thin, correct
    snapshot — no fills, no exits — meant to be polled at the open.

    Returns ``{"asof", "train_start", "train_end", "signals"}`` where ``signals`` is a DataFrame
    with ``[symbol, date, prob_up, side]`` (one row per ticker that has data on/just before
    ``asof``) and ``side in {'long','short','flat'}``.
    """
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    asof_ts = pd.Timestamp(asof)
    train_start = (asof_ts - pd.DateOffset(years=lookback_model_years)).date().isoformat()
    # Train end is strictly before asof (no peeking at the asof bar).
    train_end = (asof_ts - pd.Timedelta(days=1)).date().isoformat()

    ds = build_dataset(
        tickers, train_start, train_end, horizon,
        client=client, cache=cache, universe=universe, settings=settings,
    )
    model = make_model()
    fit_model(model, ds.X, ds.y, ds.w)
    model_features = list(ds.X.columns)

    rows: list[dict] = []
    feat_start = train_start  # warm features over the same window through asof.
    for ticker in tickers:
        feats = build_feature_matrix(
            ticker, feat_start, asof, client=client, cache=cache,
            universe=universe, settings=settings,
        )
        if feats.empty:
            continue
        feats = feats.copy()
        feats["date"] = pd.to_datetime(feats["date"])
        feats = feats[feats["date"] <= asof_ts]
        if feats.empty:
            continue
        scored = _score_frame(feats, model, model_features)
        if scored.empty:
            continue
        last = scored.sort_values("date").iloc[-1]
        prob_up = float(last["prob_up"])
        side = "long" if prob_up >= long_th else "short" if prob_up <= short_th else "flat"
        rows.append({
            "symbol": ticker,
            "date": pd.Timestamp(last["date"]),
            "prob_up": prob_up,
            "side": side,
        })

    signals = pd.DataFrame(rows, columns=["symbol", "date", "prob_up", "side"])
    return {
        "asof": asof_ts,
        "train_start": train_start,
        "train_end": train_end,
        "signals": signals,
    }
