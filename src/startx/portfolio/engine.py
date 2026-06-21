"""Production portfolio engine — the spine of the SPY swing book.

This module turns a set of *validated, independent sleeves* (momentum breakout, IBS
mean-reversion, VRP/VVIX volatility-premium, VIX capitulation) into ONE vol-sized,
risk-managed, concurrent book trading a single instrument (SPY).

Design principles (all locked by the desk):

* **One unified exit for every sleeve** — a chandelier trail (peak high since entry
  minus ``atr_mult``·ATR(14)) with a ``max_days`` time cap. The desks proved the 3.0-ATR
  chandelier is near-optimal and beats fixed/learned exits, so even the IBS sleeve (which
  historically used a fixed 1.5-ATR target) is run on the chandelier here. "Don't cut
  home-runners": a runner is ridden to exhaustion, never capped.

* **Volatility-target sizing** — each trade is sized so its risk (the fractional distance
  to its initial stop, ``atr_mult``·atr_pct) equals a fixed ``risk_per_trade`` fraction of
  equity. A tight-stop (calm) trade gets a bigger weight than a wide-stop (volatile) trade,
  so every position contributes the *same* risk budget. Concurrent risk and gross exposure
  are capped; a new trade is scaled down to fit the remaining room, or skipped if there is
  none.

* **Event-driven concurrency on one shared equity curve** — positions from different
  sleeves run simultaneously. Each calendar day we process *exits before entries* so freed
  risk budget is reusable same-day. At most one open position per sleeve (no stacking the
  same sleeve). Equity compounds multiplicatively at each exit: ``equity *= (1 + w·ret)``.

* **Risk overlays** — a flight-to-safety filter (skip new entries when gold is
  out-performing SPY, ``GLD/SPY`` above its 20-day average) and an equity drawdown breaker
  (halve new-trade size while underwater past ``dd_breaker`` until a new high water mark).

The engine is intentionally narrow: construction, sizing, risk. It does NOT decide entries
(those come from the sleeve callables) and it does NOT annotate trades with regime / thesis
(those columns are emitted empty for a separate Thesis agent to fill).

Sleeve interface contract
-------------------------
A sleeve is ``sig(spy_df, aux) -> pd.Series[bool]`` aligned to ``spy_df`` (sorted rows),
True on entry days. ``aux`` is ``dict[str, pd.DataFrame]`` of raw parquet frames keyed by
lower-case symbol (``'vix'``, ``'vvix'``, ``'gld'``, ``'hyg'``, ``'lqd'`` ...), each with a
datetime ``date`` column. ``run_portfolio`` takes ``sleeves: dict[name -> sig]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

# A sleeve signal: maps the SPY frame + auxiliary frames to an entry-day boolean mask.
SleeveSig = Callable[[pd.DataFrame, Mapping[str, pd.DataFrame]], pd.Series]

# Ledger schema (locked column order). ``regime``/``conviction``/``thesis`` are placeholders
# left empty here for a downstream Thesis agent to populate.
LEDGER_COLUMNS: list[str] = [
    "sleeve", "entry_date", "entry_price", "stop_price", "exit_date", "exit_price",
    "exit_reason", "bars_held", "weight", "ret", "pnl_contrib", "outcome",
    "regime", "conviction", "thesis",
]

TRADING_DAYS = 252  # annualisation factor for the daily-equity Sharpe


# --------------------------------------------------------------------------------------- #
# Indicators (kept local so the engine is self-contained and matches the sleeve ATR exactly)
# --------------------------------------------------------------------------------------- #
def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-style ATR via an EWM of True Range (matches the sleeve modules' ATR)."""
    h, l, c = prices["high"], prices["low"], prices["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


# --------------------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------------------- #
@dataclass
class PortfolioResult:
    """Output of :func:`run_portfolio`.

    Attributes
    ----------
    ledger:
        One row per closed trade, columns = :data:`LEDGER_COLUMNS`.
    equity:
        The realised book equity curve indexed by date (starts at 1.0 on the first SPY day
        in-window, steps multiplicatively as trades close).
    stats:
        Headline metrics plus a ``per_sleeve`` breakdown (see :func:`_book_stats`).
    """

    ledger: pd.DataFrame
    equity: pd.Series
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------- #
# Internal: an open position carried through the event loop
# --------------------------------------------------------------------------------------- #
@dataclass
class _OpenTrade:
    sleeve: str
    entry_idx: int            # positional index into the SPY frame
    entry_date: pd.Timestamp
    entry_price: float
    atr_at_entry: float       # absolute ATR at entry (price units)
    stop_price: float         # initial chandelier stop = entry - atr_mult*ATR
    weight: float             # fraction of equity allocated (vol-target sized)
    risk_frac: float          # fractional distance to the initial stop = atr_mult*atr_pct
    peak: float               # running peak high since entry (for the chandelier trail)
    last_day: int             # entry_idx + max_days (inclusive time cap)
    stop_mult: float          # hard-stop ATR multiple (1 ATR — cut the loss short)
    chand_mult: float         # trailing-chandelier ATR multiple (3 ATR — ride the winner)


# --------------------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------------------- #
def run_portfolio(
    spy: pd.DataFrame,
    aux: Mapping[str, pd.DataFrame],
    sleeves: Mapping[str, SleeveSig],
    *,
    risk_per_trade: float = 0.03,
    port_risk_cap: float = 0.12,
    gross_cap: float = 3.0,
    atr_mult: float = 3.0,
    max_days: int = 252,
    cost_bps: float = 2.0,
    gold_calm: bool = True,
    dd_breaker: float = 0.10,
    exits: Mapping[str, tuple[float, float, int]] | None = None,  # name -> (stop_mult, chand_mult, max_days)
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    entry_fill: str = "close",
) -> PortfolioResult:
    """Run the concurrent, vol-sized swing book.

    Parameters
    ----------
    spy:
        SPY OHLCV frame (``date, open, high, low, close, ...``). Sorted internally.
    aux:
        ``dict[symbol -> frame]`` of auxiliary series (vix, vvix, gld, ...) passed straight
        through to each sleeve and used by the gold-calm overlay (needs ``aux['gld']``).
    sleeves:
        ``dict[name -> sig]`` where each ``sig(spy_df, aux)`` returns an entry-day bool mask.
    risk_per_trade:
        Target risk per position as a fraction of equity (default 3%).
    port_risk_cap:
        Cap on concurrent summed risk = Σ wᵢ·risk_fracᵢ (default 12%).
    gross_cap:
        Cap on concurrent summed gross weight = Σ wᵢ (default 3.0×).
    atr_mult:
        Chandelier / initial-stop ATR multiple (default 3.0).
    max_days:
        Far-back SAFETY backstop in trading days (default 252 ≈ 1yr). The chandelier is the real
        exit — a short cap (e.g. 40) force-chops still-trending winners mid-move and, because the
        sleeve frees up the same bar, manufactures an immediate same-price re-entry (churn). Keep
        this large so the trail does the exiting; "don't cut home-runners".
    cost_bps:
        Round-trip transaction cost in basis points, charged once per trade on the return.
    gold_calm:
        If True, skip NEW entries on days where ``GLD/SPY`` is above its own SMA20
        (flight-to-safety: gold leading the index is a risk-off tell).
    dd_breaker:
        If equity is below ``peak·(1 - dd_breaker)``, halve the weight of NEW entries until
        equity reclaims the high-water mark.
    start, end:
        Optional inclusive window bounds on *entry* dates. The equity curve still spans the
        full in-window SPY date index.
    entry_fill:
        Where a new position is filled relative to the signal bar ``i``:

        * ``"close"`` (default, unchanged legacy behaviour) — fill at ``closes[i]``, the
          signal-day close. The entry bar is the signal bar, so the chandelier/exit window
          opens on the *next* bar (``i+1``).
        * ``"next_open"`` — fill at the NEXT bar's open ``opens[i+1]`` (one day later, a
          realistic non-anticipative execution: you see the signal at the close and trade the
          following open). ``entry_idx`` is set to ``i+1`` so the exit window starts after the
          fill bar, and the time cap ``last_day`` is measured from the fill bar. A signal on
          the last available bar (``i+1`` out of range) is skipped. Used to measure how much
          of the book's edge is a close-print artifact vs. survives realistic execution.

    Returns
    -------
    PortfolioResult
    """
    if not 0.0 < risk_per_trade <= 1.0:
        raise ValueError("risk_per_trade must be in (0, 1]")
    if port_risk_cap <= 0 or gross_cap <= 0:
        raise ValueError("caps must be positive")
    if entry_fill not in ("close", "next_open"):
        raise ValueError("entry_fill must be 'close' or 'next_open'")

    p = spy.sort_values("date").reset_index(drop=True)
    n = len(p)
    if n == 0:
        return PortfolioResult(_empty_ledger(), pd.Series(dtype=float), _book_stats(
            _empty_ledger(), pd.Series(dtype=float), [], cost_bps))

    dates = pd.to_datetime(p["date"])
    opens = p["open"].to_numpy(dtype=float)
    highs = p["high"].to_numpy(dtype=float)
    lows = p["low"].to_numpy(dtype=float)
    closes = p["close"].to_numpy(dtype=float)
    atr_abs = atr(p).to_numpy(dtype=float)

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    # --- entry masks per sleeve (run each sleeve once, align to the SPY index) ---------- #
    entry_idx_by_sleeve: dict[str, set[int]] = {}
    for name, sig in sleeves.items():
        mask = _aligned_mask(sig(p, aux), n)
        entry_idx_by_sleeve[name] = set(np.flatnonzero(mask).tolist())

    # --- gold-calm overlay: per-day "risk-off" flag (GLD/SPY ratio above its SMA20) ----- #
    gold_off = _gold_calm_flag(p, aux, dates) if gold_calm else np.zeros(n, dtype=bool)

    cost = cost_bps / 10000.0  # charged once per trade on the realised return

    # --- event-driven loop -------------------------------------------------------------- #
    # Two equity views, both tracked on the SAME book:
    #   * realised equity — compounds multiplicatively at each EXIT (the contract's
    #     ``equity *= (1 + w·ret)``); this drives sizing and the drawdown breaker.
    #   * mark-to-market (MTM) equity — realised equity grossed up by the open positions'
    #     *unrealised* weighted P&L each day; this is the curve we report Sharpe/maxDD on so
    #     intraday-of-trade risk is visible (a flat-at-exit-only curve understates both vol
    #     and drawdown). The two coincide whenever the book is flat, and the MTM curve passes
    #     exactly through the realised curve on every day all positions are closed.
    equity = 1.0          # realised equity
    hwm = 1.0             # high-water mark of the realised curve (drawdown breaker)
    equity_curve = np.ones(n, dtype=float)  # MTM equity, reported
    open_trades: list[_OpenTrade] = []
    open_sleeves: set[str] = set()
    rows: list[dict] = []

    for i in range(n):
        day = dates.iloc[i]
        exited_today: set[str] = set()  # a sleeve that exits today cannot re-enter today (no churn)
        opened_prices_today: set[float] = set()  # no two positions at the same entry price this bar

        # 1) EXITS FIRST — chandelier trail or time cap. Freed risk budget is reusable today.
        still_open: list[_OpenTrade] = []
        for t in open_trades:
            if i <= t.entry_idx:
                still_open.append(t)
                continue
            t.peak = max(t.peak, highs[i])
            # Two exits: a HARD STOP at entry - stop_mult*ATR (cut the loss short, e.g. 1 ATR) and a
            # trailing CHANDELIER at peak - chand_mult*ATR (ride the winner, e.g. 3 ATR). The binding
            # level is the higher of the two: the hard stop holds until the trade has run far enough
            # that the 3-ATR trail lifts above it (peak > entry + (chand-stop)*ATR), then the trail
            # takes over.
            hard = t.entry_price - t.stop_mult * t.atr_at_entry
            chand = t.peak - t.chand_mult * t.atr_at_entry
            stop_level = max(hard, chand)
            exit_price = exit_reason = None
            if lows[i] <= stop_level:
                # Realistic fill: if the bar GAPPED through the stop (opened beyond it), you fill at
                # the open, not the stop level. Filling at the stop on a gap-down overstates winners
                # and hides the true tail loss (e.g. the 2020-02-24 COVID gap was a real -2.9%).
                exit_price = min(stop_level, opens[i])
                exit_reason = "chandelier" if chand >= hard else "stop"
            elif i >= t.last_day:
                exit_price = closes[i]
                exit_reason = "time"
            if exit_price is None:
                still_open.append(t)
                continue
            # realise the trade: net of round-trip cost, compounded on the realised curve.
            gross_ret = exit_price / t.entry_price - 1.0
            net_ret = gross_ret - cost
            pnl_contrib = t.weight * net_ret
            equity *= (1.0 + pnl_contrib)
            hwm = max(hwm, equity)
            rows.append(_ledger_row(t, day, exit_price, exit_reason, i, net_ret, cost))
            open_sleeves.discard(t.sleeve)
            exited_today.add(t.sleeve)
        open_trades = still_open

        # 2) capacity used by the survivors (concurrent risk + gross), for sizing new trades
        used_risk = sum(t.risk_frac * t.weight for t in open_trades)
        used_gross = sum(t.weight for t in open_trades)

        # drawdown-breaker multiplier on NEW entries (halve while underwater past dd_breaker)
        dd_mult = 0.5 if equity < hwm * (1.0 - dd_breaker) else 1.0

        # 3) ENTRIES — only inside the entry window, never on a risk-off (gold) day.
        in_window = ((start_ts is None or day >= start_ts) and (end_ts is None or day <= end_ts))
        if in_window and not gold_off[i] and i + 1 < n:
            # Deterministic sleeve order so concurrent-day fills are reproducible.
            for name in sleeves:
                if name in open_sleeves:
                    continue  # one open position per sleeve
                if name in exited_today:
                    continue  # exited this bar -> no same-day same-price re-entry (churn guard)
                if i not in entry_idx_by_sleeve[name]:
                    continue
                av = atr_abs[i]
                # Fill bar & price depend on entry_fill. "close": fill this bar's close, entry
                # bar = signal bar i, exit window opens at i+1. "next_open": fill the NEXT
                # bar's open (a realistic non-anticipative execution one day later), so the
                # entry/fill bar IS i+1 and the exit window opens at i+2. ATR is the signal-bar
                # ATR in both cases (known at signal time; non-anticipative).
                if entry_fill == "next_open":
                    fill_idx = i + 1  # guaranteed in range by the i + 1 < n loop guard
                    entry_price = opens[fill_idx]
                else:
                    fill_idx = i
                    entry_price = closes[i]
                if not np.isfinite(av) or av <= 0 or entry_price <= 0:
                    continue
                if entry_price in opened_prices_today:
                    continue  # de-dup: another sleeve already opened this exact bar/price (no stacking)
                # per-sleeve exit: (stop_mult ATR hard stop, chand_mult ATR trailing chandelier,
                # max_days hold). Short-term sleeves (IBS dip) get a short cap; trend/capitulation
                # sleeves ride for months. Defaults keep the legacy single-trail behaviour.
                stop_mult, chand_mult, mdays = (exits or {}).get(name, (atr_mult, atr_mult, max_days))
                atr_pct = av / entry_price
                # Size on the chandelier distance so leverage stays ~unchanged; the tighter hard stop
                # only REDUCES realised loss per trade (max ~stop_mult ATR), it doesn't lever up.
                risk_frac = chand_mult * atr_pct
                if risk_frac <= 0:
                    continue
                # vol-target: weight so this trade's risk == risk_per_trade of equity
                w = (risk_per_trade * dd_mult) / risk_frac
                # scale DOWN to fit remaining risk and gross room; skip if no room
                w = _fit_to_caps(w, risk_frac, used_risk, used_gross,
                                 port_risk_cap, gross_cap)
                if w <= 0:
                    continue
                stop_price = entry_price - stop_mult * av  # the HARD stop (recorded for review)
                # entry_idx is the FILL bar: i for close fills, i+1 for next_open fills. The exit
                # loop only acts on bars strictly after entry_idx, and the time cap counts from it.
                open_trades.append(_OpenTrade(
                    sleeve=name, entry_idx=fill_idx, entry_date=dates.iloc[fill_idx],
                    entry_price=entry_price,
                    atr_at_entry=av, stop_price=stop_price, weight=w, risk_frac=risk_frac,
                    peak=entry_price, last_day=min(fill_idx + mdays, n - 1),
                    stop_mult=stop_mult, chand_mult=chand_mult))
                open_sleeves.add(name)
                opened_prices_today.add(entry_price)
                used_risk += risk_frac * w
                used_gross += w

        # 4) MARK-TO-MARKET: gross the realised equity up by the open book's unrealised
        #    weighted P&L (close-to-close, cost accrued once on the open leg). On a day with
        #    no open positions this equals the realised equity exactly.
        unreal = 0.0
        for t in open_trades:
            if i < t.entry_idx:
                continue
            mtm_ret = (closes[i] / t.entry_price - 1.0) - cost
            unreal += t.weight * mtm_ret
        equity_curve[i] = equity * (1.0 + unreal)

    # --- assemble outputs --------------------------------------------------------------- #
    ledger = _build_ledger(rows)
    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(dates), name="equity")

    # Headline stats are reported over the STUDY WINDOW [start, end] only. Entries are
    # window-restricted, so the book sits flat at 1.0 on every bar before `start` (and after
    # `end`); folding those flat zero-return days into the daily series dilutes the annualised
    # Sharpe by ~sqrt(total_days / in_window_days) — e.g. passing the full 1993→2026 SPY index
    # with start=2019 understated a true 1.07 Sharpe to 0.50. We slice (and rebase) to the
    # window so Sharpe / total_return / maxDD / exposure all describe the period actually traded.
    eq_win = equity_series
    if start_ts is not None or end_ts is not None:
        m = np.ones(len(equity_series), dtype=bool)
        if start_ts is not None:
            m &= (equity_series.index >= start_ts)
        if end_ts is not None:
            m &= (equity_series.index <= end_ts)
        if m.any():
            eq_win = equity_series[m]
    if len(eq_win) >= 2 and eq_win.iloc[0] != 0:
        eq_win = eq_win / eq_win.iloc[0]          # rebase so total_return is in-window
    daily_ret = eq_win.pct_change().fillna(0.0)
    stats = _book_stats(ledger, eq_win, list(sleeves.keys()), cost_bps, daily_ret=daily_ret)
    return PortfolioResult(ledger=ledger, equity=equity_series, stats=stats)


# --------------------------------------------------------------------------------------- #
# Sizing helpers
# --------------------------------------------------------------------------------------- #
def _fit_to_caps(w: float, risk_frac: float, used_risk: float, used_gross: float,
                 port_risk_cap: float, gross_cap: float) -> float:
    """Scale a candidate weight down so concurrent risk and gross stay within caps.

    Returns the largest non-negative weight ≤ the target ``w`` such that both
    ``used_risk + w·risk_frac ≤ port_risk_cap`` and ``used_gross + w ≤ gross_cap`` hold.
    Returns 0.0 when there is no remaining room (the trade is skipped).
    """
    risk_room = port_risk_cap - used_risk
    gross_room = gross_cap - used_gross
    if risk_room <= 0 or gross_room <= 0:
        return 0.0
    w = min(w, risk_room / risk_frac, gross_room)
    return max(w, 0.0)


# --------------------------------------------------------------------------------------- #
# Overlay + alignment helpers
# --------------------------------------------------------------------------------------- #
def _aligned_mask(sig: pd.Series, n: int) -> np.ndarray:
    """Coerce a sleeve signal into a length-``n`` boolean array aligned to the SPY rows.

    Sleeves return a Series indexed 0..n-1 (they reset_index internally). We defensively
    take values positionally and pad/truncate to ``n`` so a mis-sized signal can't crash
    the book or silently shift dates.
    """
    arr = np.asarray(pd.Series(sig).fillna(False).to_numpy(), dtype=bool)
    if arr.shape[0] == n:
        return arr
    out = np.zeros(n, dtype=bool)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m]
    return out


def _gold_calm_flag(p: pd.DataFrame, aux: Mapping[str, pd.DataFrame],
                    dates: pd.Series) -> np.ndarray:
    """Risk-off flag: True on days the GLD/SPY ratio is above its 20-day SMA.

    Gold leading the index (ratio rising above trend) is a flight-to-safety tell; we stand
    aside from NEW entries on those days. If GLD is unavailable, the overlay is a no-op.
    """
    gld = aux.get("gld")
    if gld is None or "close" not in getattr(gld, "columns", []):
        return np.zeros(len(p), dtype=bool)
    g = gld.sort_values("date").set_index("date")["close"]
    g = g.reindex(pd.DatetimeIndex(dates)).ffill().to_numpy(dtype=float)
    ratio = pd.Series(g / p["close"].to_numpy(dtype=float))
    sma20 = ratio.rolling(20).mean()
    flag = (ratio > sma20).to_numpy()
    return np.where(np.isnan(flag), False, flag).astype(bool)


# --------------------------------------------------------------------------------------- #
# Ledger assembly
# --------------------------------------------------------------------------------------- #
def _ledger_row(t: _OpenTrade, exit_date: pd.Timestamp, exit_price: float,
                exit_reason: str, exit_idx: int, net_ret: float, cost: float) -> dict:
    """Build one ledger row from a closed trade.

    Outcome convention (desk rule): a breakeven/scratch exit is a WIN/SCRATCH, never a LOSS.
    We classify on the NET return: > 0 → WIN, == 0 → SCRATCH, < 0 → LOSS.
    """
    if net_ret > 0:
        outcome = "WIN"
    elif net_ret == 0:
        outcome = "SCRATCH"
    else:
        outcome = "LOSS"
    return {
        "sleeve": t.sleeve,
        "entry_date": t.entry_date,
        "entry_price": round(t.entry_price, 4),
        "stop_price": round(t.stop_price, 4),
        "exit_date": exit_date,
        "exit_price": round(float(exit_price), 4),
        "exit_reason": exit_reason,
        "bars_held": int(exit_idx - t.entry_idx),
        "weight": float(t.weight),
        "ret": float(net_ret),
        "pnl_contrib": float(t.weight * net_ret),
        "outcome": outcome,
        # placeholders for the Thesis agent
        "regime": None,
        "conviction": np.nan,
        "thesis": None,
    }


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _build_ledger(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_ledger()
    df = pd.DataFrame(rows)
    df = df.sort_values("entry_date").reset_index(drop=True)
    return df[LEDGER_COLUMNS]


# --------------------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------------------- #
def _ann_sharpe(daily_ret: pd.Series) -> float:
    """Annualised Sharpe of the realised daily equity returns (rf = 0)."""
    r = daily_ret.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def _exposure(ledger: pd.DataFrame, equity: pd.Series) -> float:
    """Time-in-market: fraction of book days with at least one position open.

    Computed from trade [entry, exit] spans over the full equity date index.
    """
    if ledger.empty or equity.empty:
        return 0.0
    idx = equity.index
    occupied = np.zeros(len(idx), dtype=bool)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    for _, tr in ledger.iterrows():
        try:
            a = pos.get(pd.Timestamp(tr["entry_date"]))
            b = pos.get(pd.Timestamp(tr["exit_date"]))
        except Exception:
            continue
        if a is None or b is None:
            continue
        occupied[int(a):int(b) + 1] = True
    return float(occupied.mean())


def _book_stats(ledger: pd.DataFrame, equity: pd.Series, sleeve_names: list[str],
                cost_bps: float, daily_ret: pd.Series | None = None) -> dict:
    """Headline book metrics plus a per-sleeve breakdown.

    turnover := total gross traded weight / number of book years (a sizing-aware proxy for
    how hard the book churns; one round-trip of weight w contributes w to gross traded).
    """
    if daily_ret is None:
        daily_ret = equity.pct_change().fillna(0.0) if not equity.empty else pd.Series(dtype=float)

    n = int(len(ledger))
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

    if n:
        pnl = ledger["pnl_contrib"]
        wins = ledger["outcome"].eq("WIN")
        win_rate = float(wins.mean())
        profit_factor = _profit_factor(pnl)
        gross_traded = float(ledger["weight"].abs().sum())
    else:
        win_rate = float("nan")
        profit_factor = float("nan")
        gross_traded = 0.0

    if not equity.empty:
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    else:
        years = 1e-9
    turnover = float(gross_traded / years)

    stats = {
        "n": n,
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": _max_drawdown(equity),
        "profit_factor": profit_factor,
        "exposure": _exposure(ledger, equity),
        "turnover": turnover,
        "ann_sharpe": _ann_sharpe(daily_ret),
        "per_sleeve": _per_sleeve_stats(ledger, sleeve_names),
    }
    return stats


def _per_sleeve_stats(ledger: pd.DataFrame, sleeve_names: list[str]) -> dict:
    """Per-sleeve breakdown: count, win rate, summed P&L contribution, profit factor."""
    out: dict[str, dict] = {}
    for name in sleeve_names:
        sub = ledger[ledger["sleeve"] == name] if not ledger.empty else ledger
        if sub.empty:
            out[name] = {"n": 0, "win_rate": float("nan"),
                         "pnl_contrib": 0.0, "profit_factor": float("nan")}
            continue
        out[name] = {
            "n": int(len(sub)),
            "win_rate": float(sub["outcome"].eq("WIN").mean()),
            "pnl_contrib": float(sub["pnl_contrib"].sum()),
            "profit_factor": _profit_factor(sub["pnl_contrib"]),
        }
    return out
