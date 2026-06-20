"""Sequential, no-lookahead backtest engine for long/short swing signals.

The engine consumes per-entry model probabilities and the triple-barrier label
outcomes and produces a realistic, one-position-at-a-time equity curve with
transaction costs, plus a headline metrics dict.

No-lookahead guarantees
-----------------------
* A trade's outcome is taken **entirely** from its own label row ``(ret, t1)``;
  nothing from a future row is used to decide the entry.
* ``one_at_a_time`` enforces non-overlap: after a position opens on ``entry``
  and exits on ``t1``, the next candidate entry must be *strictly after* ``t1``.
  The engine is flat (no exposure) between trades.
* Compounded net returns are stamped at each trade's **exit** date (``t1``) — the
  date the P&L is actually realised — never at entry. The daily curve is then
  forward-filled, so equity only steps up/down once a trade has closed.

Metrics convention
------------------
* ``sharpe``, ``sortino``, ``max_drawdown``, ``cagr`` are computed on the daily
  equity *return* series (the realistic timeline).
* ``profit_factor``, ``hit_rate`` are computed on the per-trade net returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from startx.backtest.costs import CostModel
from startx.validation.metrics import (
    cagr as _cagr,
    hit_rate as _hit_rate,
    max_drawdown as _max_drawdown,
    profit_factor as _profit_factor,
    sharpe as _sharpe,
    sortino as _sortino,
)

_TRADE_COLUMNS = [
    "entry_date",
    "exit_date",
    "side",
    "prob_up",
    "ret_gross",
    "ret_net",
    "holding_days",
]

_METRIC_KEYS = [
    "n_trades",
    "hit_rate",
    "avg_ret_net",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "profit_factor",
    "exposure",
    "avg_holding_days",
]


@dataclass
class BacktestResult:
    """Output of a backtest.

    Attributes
    ----------
    trades:
        One row per executed trade, columns :data:`_TRADE_COLUMNS`.
    equity:
        Daily equity curve (Series indexed by date), starting at ``capital`` and
        forward-filled flat between trades.
    metrics:
        Headline statistics keyed by :data:`_METRIC_KEYS`.
    """

    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict = field(default_factory=dict)


def _prob_series(predictions: pd.DataFrame) -> pd.Series:
    """Extract the up-probability series, preferring ``prob_up`` then ``y_prob``."""
    if "prob_up" in predictions.columns:
        col = "prob_up"
    elif "y_prob" in predictions.columns:
        col = "y_prob"
    else:
        raise KeyError("predictions must carry a 'prob_up' or 'y_prob' column")
    s = predictions[col].astype(float)
    s.name = "prob_up"
    return s


def _build_daily_equity(
    trades: pd.DataFrame,
    *,
    capital: float,
    prices: pd.DataFrame | None,
    span: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Compound net trade returns at their exit dates and reindex onto a daily grid.

    Equity = ``capital * cumprod(1 + ret_net)`` stamped at each ``exit_date``,
    then forward-filled across the daily timeline (flat between trades, ``capital``
    before the first exit). ``span`` is the candidate entry/t1 calendar used as a
    fallback timeline when ``prices`` is absent (so a flat curve still exists when
    no trade fires).
    """
    if trades.empty:
        timeline = _timeline(trades, prices, span)
        return pd.Series(float(capital), index=timeline, name="equity")

    exits = pd.to_datetime(trades["exit_date"]).to_numpy()
    equity_at_exit = float(capital) * np.cumprod(1.0 + trades["ret_net"].to_numpy())
    # Collapse trades that exit on the same date onto the last cumulative value.
    step = pd.Series(equity_at_exit, index=pd.DatetimeIndex(exits))
    step = step[~step.index.duplicated(keep="last")].sort_index()

    timeline = _timeline(trades, prices, span)
    equity = step.reindex(timeline.union(step.index)).ffill()
    # Before the first realised trade the account is flat at starting capital.
    equity = equity.fillna(float(capital))
    equity = equity.reindex(timeline)
    equity.name = "equity"
    return equity


def _timeline(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None,
    span: pd.DatetimeIndex | None = None,
) -> pd.DatetimeIndex:
    """Daily date index for the equity curve.

    Uses the ``prices`` date column when supplied (the realistic calendar);
    otherwise falls back to a business-day range spanning the trades, or — when
    no trade fired — the candidate ``span`` (entry/t1 dates of the join).
    """
    if prices is not None and not prices.empty and "date" in prices.columns:
        return pd.DatetimeIndex(pd.to_datetime(prices["date"]).unique()).sort_values()
    if not trades.empty:
        start = pd.to_datetime(trades["entry_date"]).min()
        end = pd.to_datetime(trades["exit_date"]).max()
        return pd.bdate_range(start, end)
    if span is not None and len(span):
        return pd.DatetimeIndex(span).sort_values()
    return pd.DatetimeIndex([], name="date")


def _exposure(
    trades: pd.DataFrame, timeline: pd.DatetimeIndex
) -> float:
    """Fraction of timeline days that fall inside an open position ``[entry, t1]``."""
    if len(timeline) == 0 or trades.empty:
        return 0.0
    days = timeline.to_numpy()
    in_pos = np.zeros(len(days), dtype=bool)
    entries = pd.to_datetime(trades["entry_date"]).to_numpy()
    exits = pd.to_datetime(trades["exit_date"]).to_numpy()
    for e, x in zip(entries, exits):
        in_pos |= (days >= e) & (days <= x)
    return float(in_pos.mean())


def _compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    capital: float,
    periods: int = 252,
) -> dict:
    """Assemble the headline metrics dict from trades and the daily equity curve."""
    n_trades = int(len(trades))
    daily_ret = equity.pct_change().dropna()
    timeline = pd.DatetimeIndex(equity.index)

    if n_trades:
        trade_net = trades["ret_net"].to_numpy()
        avg_ret_net = float(np.mean(trade_net))
        hit = float(_hit_rate(trade_net))
        pf = float(_profit_factor(trade_net))
        avg_hold = float(trades["holding_days"].mean())
    else:
        avg_ret_net = float("nan")
        hit = float("nan")
        pf = float("nan")
        avg_hold = float("nan")

    if len(equity):
        total_return = float(equity.iloc[-1] / float(capital) - 1.0)
    elif n_trades == 0:
        # Flat account with no trades realised no P&L: total return is exactly 0.
        total_return = 0.0
    else:
        total_return = float("nan")

    return {
        "n_trades": n_trades,
        "hit_rate": hit,
        "avg_ret_net": avg_ret_net,
        "total_return": total_return,
        "cagr": float(_cagr(daily_ret, periods=periods)) if len(daily_ret) else float("nan"),
        "sharpe": float(_sharpe(daily_ret, periods=periods)) if len(daily_ret) else float("nan"),
        "sortino": float(_sortino(daily_ret, periods=periods)) if len(daily_ret) else float("nan"),
        "max_drawdown": (
            float(_max_drawdown(equity.to_numpy(), is_returns=False))
            if len(equity)
            else float("nan")
        ),
        "profit_factor": pf,
        "exposure": _exposure(trades, timeline),
        "avg_holding_days": avg_hold,
    }


def backtest_signals(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    long_th: float = 0.6,
    short_th: float = 0.4,
    costs: CostModel = CostModel(),
    prices: pd.DataFrame | None = None,
    one_at_a_time: bool = True,
    capital: float = 1.0,
) -> BacktestResult:
    """Backtest probability signals against triple-barrier label outcomes.

    Parameters
    ----------
    predictions:
        DataFrame indexed by entry date with an up-probability column
        (``prob_up`` preferred, else ``y_prob``).
    labels:
        Triple-barrier output with columns ``date, t1, ret, label`` (extra
        columns ignored). ``ret`` is the realised entry→``t1`` close-to-close
        return; the trade outcome is fully determined by ``(ret, t1)``.
    long_th, short_th:
        Go LONG when ``prob_up >= long_th``; go SHORT when ``prob_up <= short_th``;
        otherwise stay flat. Require ``short_th <= long_th``.
    costs:
        Round-trip cost model; charged once per trade (entry + exit).
    prices:
        Optional ``date``/``close`` frame defining the daily equity calendar.
    one_at_a_time:
        When True (default) hold at most one position; the next entry must be
        strictly after the current trade's ``t1`` (no overlap).
    capital:
        Starting equity (curve begins here).

    Returns
    -------
    BacktestResult
        ``trades``, daily ``equity`` and ``metrics``.
    """
    if short_th > long_th:
        raise ValueError(f"short_th ({short_th}) must be <= long_th ({long_th})")

    prob = _prob_series(predictions)

    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"])
    lab["t1"] = pd.to_datetime(lab["t1"])
    lab = lab.sort_values("date").drop_duplicates(subset="date", keep="first")
    lab = lab.set_index("date")

    # Inner-join predictions to labels on entry date, ordered chronologically.
    prob.index = pd.to_datetime(prob.index)
    joined = lab.join(prob.to_frame("prob_up"), how="inner").dropna(subset=["prob_up"])
    joined = joined.sort_index()

    rt_cost = costs.round_trip_cost()
    records: list[dict] = []
    cursor: pd.Timestamp | None = None  # next entry must be strictly after this t1

    for entry_date, row in joined.iterrows():
        if one_at_a_time and cursor is not None and entry_date <= cursor:
            continue  # still inside a held position -> skip overlapping candidate

        p = float(row["prob_up"])
        if p >= long_th:
            side = 1
        elif p <= short_th:
            side = -1
        else:
            continue  # mid-band: stay flat, do not consume the cursor

        ret = float(row["ret"])
        t1 = pd.Timestamp(row["t1"])
        ret_gross = side * ret
        ret_net = ret_gross - rt_cost
        records.append(
            {
                "entry_date": pd.Timestamp(entry_date),
                "exit_date": t1,
                "side": int(side),
                "prob_up": p,
                "ret_gross": float(ret_gross),
                "ret_net": float(ret_net),
                "holding_days": int((t1 - pd.Timestamp(entry_date)).days),
            }
        )
        if one_at_a_time:
            cursor = t1

    trades = pd.DataFrame.from_records(records, columns=_TRADE_COLUMNS)
    # Fallback calendar (used only when prices is None) spanning all candidate
    # entry and exit dates, so a flat curve exists even if nothing trades.
    span = pd.DatetimeIndex(
        pd.concat([joined.index.to_series(), joined["t1"]]).unique()
    ).sort_values() if len(joined) else None
    equity = _build_daily_equity(trades, capital=capital, prices=prices, span=span)
    metrics = _compute_metrics(trades, equity, capital=capital)
    return BacktestResult(trades=trades, equity=equity, metrics=metrics)


def portfolio_backtest(results: dict[str, BacktestResult]) -> BacktestResult:
    """Equal-weight portfolio of per-symbol backtests.

    Aligns every symbol's daily equity curve on the union of their dates
    (forward-filled), equal-weights the *normalised* curves into one portfolio
    equity series, concatenates all trades, and recomputes portfolio metrics on
    the blended daily curve.
    """
    if not results:
        empty = pd.DataFrame(columns=_TRADE_COLUMNS)
        return BacktestResult(
            trades=empty,
            equity=pd.Series(dtype=float, name="equity"),
            metrics=_compute_metrics(empty, pd.Series(dtype=float), capital=1.0),
        )

    # Union daily index, ffill each curve, normalise to start at 1.0 so symbols
    # contribute on a comparable scale, then equal-weight average.
    index = None
    for res in results.values():
        idx = pd.DatetimeIndex(res.equity.index)
        index = idx if index is None else index.union(idx)
    index = index.sort_values()

    normed = []
    for res in results.values():
        eq = res.equity.reindex(index).ffill()
        first_valid = eq.first_valid_index()
        if first_valid is not None:
            base = eq.loc[first_valid]
            eq = eq / base if base != 0 else eq
        eq = eq.bfill()  # before this symbol traded it sits at its starting level (1.0)
        normed.append(eq)

    port_equity = pd.concat(normed, axis=1).mean(axis=1)
    port_equity.name = "equity"

    all_trades = [res.trades for res in results.values() if not res.trades.empty]
    trades = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame(columns=_TRADE_COLUMNS)
    )
    trades = trades.sort_values("entry_date").reset_index(drop=True) if not trades.empty else trades

    metrics = _compute_metrics(trades, port_equity, capital=1.0)
    return BacktestResult(trades=trades, equity=port_equity, metrics=metrics)
