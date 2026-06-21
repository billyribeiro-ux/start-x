"""Position / portfolio book — System #3: a long-hold 200-SMA trend core.

A long-only **regime allocation**: hold SPY while it trades above its 200-day SMA (with a hysteresis
band to cut whipsaw around the line), step to cash when it breaks below. This is the desk's THIRD
book — a months-to-years horizon — and it is deliberately NOT the swing engine: there is no 1-ATR
stop or 3-ATR chandelier here. The "stop" IS the trend breakdown (the 200-SMA band); a tight ATR
stop would knock you out within days and defeat the long hold.

Its job is **drawdown defense**, benchmarked head-to-head vs SPY buy-and-hold. The honest, evidence-
backed mandate (a 4-agent drill on deep history): it aims to **match the index's return at
materially lower drawdown** — cutting the full-cycle −50%+ bears (dot-com, GFC) roughly in half —
NOT to beat the index on raw return. Leverage and re-entry rails were tested and cannot turn it into
a B&H-beater out-of-sample; in a bull-only window (e.g. 2019-26) it LAGS the index, and that is
expected. Its value is insurance against the tail the locked window doesn't contain.

Everything is point-in-time: the regime state for bar ``i`` is decided on bar ``i-1``'s close (the
``state`` series is shifted), so a position is only ever held on information available the prior day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PositionResult:
    ledger: pd.DataFrame      # one row per long stint (entry-cross → exit-cross)
    equity: pd.Series         # in-window strategy equity (starts at 1.0)
    stats: dict = field(default_factory=dict)   # strategy stats (in-window)
    benchmark: dict = field(default_factory=dict)  # SPY buy-and-hold stats (same window)


def position_state(spy: pd.DataFrame, *, sma_len: int = 200, band: float = 0.03) -> pd.Series:
    """Long/flat regime state, **shifted to be actionable** (held on bar i per bar i-1's close).

    Hysteresis band: go long when close rises above ``SMA·(1+band)``, go flat when it falls below
    ``SMA·(1-band)``, otherwise hold the prior state. The band stops the ~daily flip-flopping that a
    bare ``close vs SMA`` cross produces when price hugs the line.
    """
    p = spy.sort_values("date").reset_index(drop=True)
    close = p["close"].astype(float)
    sma = close.rolling(sma_len).mean()
    upper = sma * (1.0 + band)
    lower = sma * (1.0 - band)

    raw = np.zeros(len(close), dtype=bool)
    long = False
    c = close.to_numpy()
    up = upper.to_numpy()
    lo = lower.to_numpy()
    for i in range(len(close)):
        if not np.isfinite(up[i]):      # SMA warm-up: stay flat
            long = False
        elif c[i] > up[i]:
            long = True
        elif c[i] < lo[i]:
            long = False
        # else: inside the band — hold the prior state (hysteresis)
        raw[i] = long
    state = pd.Series(raw, index=p.index).shift(1).fillna(False)  # decide on prior close → no lookahead
    return state.astype(bool)


def _curve_stats(daily_ret: pd.Series, *, exposure: float | None = None) -> dict:
    r = daily_ret.fillna(0.0)
    eq = (1.0 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and np.isfinite(sd) else float("nan")
    mdd = float((eq / eq.cummax() - 1.0).min())
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    out = {"total_return": float(eq.iloc[-1] - 1.0), "cagr": cagr, "ann_sharpe": sharpe,
           "vol_ann": float(sd * np.sqrt(TRADING_DAYS)) if np.isfinite(sd) else float("nan"),
           "max_drawdown": mdd, "calmar": calmar}
    if exposure is not None:
        out["exposure"] = float(exposure)
    return out


def position_book(spy: pd.DataFrame, *, sma_len: int = 200, band: float = 0.03,
                  cost_bps: float = 2.0, start: str | pd.Timestamp | None = None,
                  end: str | pd.Timestamp | None = None) -> PositionResult:
    """Run the 200-SMA ±band position core and benchmark it vs SPY buy-and-hold over ``[start,end]``.

    The ledger books one trade per long stint: entry at the close that triggered the long (the bar
    before the first held bar, so the trade return reconciles exactly with the equity curve), exit at
    the last held close (``exit_reason='sma_breakdown'``), or the window's last bar if still long
    (``exit_reason='open'``). ``ret`` is net of a one-off ``cost_bps`` round-trip charge.
    """
    p = spy.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(p["date"])
    close = p["close"].astype(float).to_numpy()
    n = len(p)
    cost = cost_bps / 10000.0

    state = position_state(spy, sma_len=sma_len, band=band).to_numpy()
    spy_ret = pd.Series(close, index=dates).pct_change().fillna(0.0).to_numpy()

    # entry transitions (flat→long) pay the one-off round-trip cost on the entry bar
    entry_tx = state & ~np.concatenate([[False], state[:-1]])
    strat_ret = state.astype(float) * spy_ret - entry_tx.astype(float) * cost

    start_ts = pd.Timestamp(start) if start is not None else dates.iloc[0]
    end_ts = pd.Timestamp(end) if end is not None else dates.iloc[-1]
    win = (dates >= start_ts) & (dates <= end_ts)
    widx = np.flatnonzero(win.to_numpy())
    a0, b0 = int(widx[0]), int(widx[-1])

    # --- strategy + benchmark stats over the window ----------------------------------------
    sret = pd.Series(strat_ret[a0:b0 + 1], index=dates.iloc[a0:b0 + 1])
    bret = pd.Series(spy_ret[a0:b0 + 1], index=dates.iloc[a0:b0 + 1]); bret.iloc[0] = 0.0
    exposure = float(state[a0:b0 + 1].mean())
    stats = _curve_stats(sret, exposure=exposure)
    benchmark = _curve_stats(bret, exposure=1.0)
    equity = (1.0 + sret).cumprod()

    # --- ledger: one row per long stint that overlaps the window ---------------------------
    rows: list[dict] = []
    i = 0
    while i < n:
        if not state[i]:
            i += 1
            continue
        a = i
        while i + 1 < n and state[i + 1]:
            i += 1
        b = i  # stint is [a..b]
        # entry at the signal close (bar a-1, where we bought); clamp to the window so a position
        # already open at the window start is booked from the window's first bar.
        e_idx = a - 1 if a > 0 else a
        if b >= a0 and e_idx <= b0:               # stint intersects the reporting window
            ce = max(e_idx, a0)                    # clamped in-window entry
            cx = min(b, b0)                        # clamped in-window exit
            e_px, x_px = float(close[ce]), float(close[cx])
            ret = x_px / e_px - 1.0 - cost
            rows.append({
                "sleeve": "position_trend",
                "entry_date": dates.iloc[ce],
                "entry_price": round(e_px, 4),
                "exit_date": dates.iloc[cx],
                "exit_price": round(x_px, 4),
                "exit_reason": "open" if b >= b0 else "sma_breakdown",
                "bars_held": int(cx - ce),
                "ret": float(ret),
                "outcome": "WIN" if ret > 0 else ("SCRATCH" if ret == 0 else "LOSS"),
            })
        i = b + 1

    ledger = pd.DataFrame(rows)
    return PositionResult(ledger=ledger, equity=equity, stats=stats, benchmark=benchmark)
