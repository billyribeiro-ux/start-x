"""Realistic execution / cost layer for the portfolio engine.

The sleeves backtest **close-to-close** by default — an idealized fill at the signal-bar close. That
flatters the book: in reality you cannot transact at a print you only learn at the bell, you cross a
spread, and a large order moves the tape. This module re-prices those fills under realistic entry
modes and a slippage + market-impact charge, and — crucially — exposes the **fill-sensitivity** of
each sleeve so we know which edges survive contact with the order book.

The cross-desk finding this reproduces
-------------------------------------
Drift-adjusted **alpha** = per-trade expectancy − the same-holding-period buy-and-hold drift (a
part-time long silently collects the bull; subtract it to see the real edge). Under realistic fills:

* **breakout** and **vrp** are *fill-insensitive on alpha* — their edge barely moves close ->
  next-open -> vwap, and the break-even cost that zeroes their drift-adjusted alpha is large. Trust.
* **ibs** and the **fear sleeves** (vrp's cousin vvix, vix-capitulation) *lose their alpha on
  next-open* — they front-run a gap that you do not actually capture if you fill at the next open or
  VWAP. Their alpha break-even is only **~0-6 bps/side** -> **beta-robust, alpha-fragile**: durable
  risk premia, not alpha. Size them small.

Convention
----------
* Trades are LONG the index unless a ``side`` column says otherwise (-1 = short).
* All prices are per-share; returns are fractional (0.01 = 1%).
* ``slippage_bps`` and impact are charged **per side** (entry and exit each pay), in basis points
  (1 bp = 0.0001). The net trade return is gross minus both legs' costs.
* Re-pricing is point-in-time: a ``next_open`` / ``next_vwap`` entry uses the bar **after** the
  signal bar (the first bar you could actually trade), never the signal bar itself.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

_BPS = 1.0e4  # basis points per unit (1.0 = 10_000 bps)

#: Entry/exit fill modes understood by the layer.
FILL_MODES = ("close", "next_open", "next_vwap")


# ==================================================================================================
# Market-impact models (optional, pluggable)
# ==================================================================================================
@dataclass(frozen=True)
class ATRImpact:
    """Volatility-scaled market impact: ``coef × ATR%`` basis points per side.

    A liquidity-demanding order pays more when the instrument is moving. The bar's ATR (as a % of
    price) is the volatility proxy; ``coef`` (dimensionless) scales it to basis points. A coef of
    0.0 disables it. This is the cheap, data-light impact term (no order-book depth needed).
    """

    coef: float = 0.0

    def bps(self, *, atr_pct: float, **_: float) -> float:
        if not np.isfinite(atr_pct):
            return 0.0
        return float(self.coef) * float(atr_pct) * 100.0  # atr_pct is a fraction -> to bps


@dataclass(frozen=True)
class SquareRootImpact:
    """Almgren-style square-root impact: ``coef × sqrt(participation) × ATR%`` bps per side.

    ``participation`` = order shares / bar volume (the fraction of the bar you are). Impact grows
    with the square root of participation and scales by volatility (ATR%). Falls back to the ATR
    term when volume is missing. ``shares`` defaults to 0 (no impact) so the model is inert unless
    the caller actually sizes orders.
    """

    coef: float = 0.1
    shares: float = 0.0

    def bps(self, *, atr_pct: float, volume: float = float("nan"), **_: float) -> float:
        if not np.isfinite(atr_pct):
            return 0.0
        vol_bps = float(atr_pct) * 100.0
        if not np.isfinite(volume) or volume <= 0 or self.shares <= 0:
            return 0.0  # no participation info -> no impact (slippage already covers the spread)
        participation = min(1.0, float(self.shares) / float(volume))
        return float(self.coef) * np.sqrt(participation) * vol_bps


ImpactModel = Callable[..., float]  # any object/callable exposing ``.bps(atr_pct=..., volume=...)``


def _impact_bps(model: ImpactModel | None, *, atr_pct: float, volume: float) -> float:
    """Evaluate an optional impact model defensively (callable or ``.bps``-bearing object)."""
    if model is None:
        return 0.0
    fn = getattr(model, "bps", model)
    try:
        return float(fn(atr_pct=atr_pct, volume=volume))
    except TypeError:
        return float(fn(atr_pct))


# ==================================================================================================
# Price-frame plumbing
# ==================================================================================================
def _indexed(prices: pd.DataFrame) -> pd.DataFrame:
    """Return ``prices`` sorted, de-duplicated and indexed by Timestamp date for O(1) date lookup."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return p


def _atr_pct_by_date(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR(window) as a fraction of close, indexed by date (the impact-model volatility input)."""
    p = _indexed(prices)
    h, l, c = p["high"], p["low"], p["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return (a / c).rename("atr_pct")


def _bar_price(bar: pd.Series, field: str) -> float:
    """Pull a price field from a bar, falling back gracefully (vwap -> typical -> close)."""
    if field == "vwap":
        if "vwap" in bar and np.isfinite(bar.get("vwap", np.nan)):
            return float(bar["vwap"])
        # Synthesize VWAP as the typical price (H+L+C)/3 when the feed lacks a vwap column.
        if {"high", "low", "close"} <= set(bar.index):
            return float((bar["high"] + bar["low"] + bar["close"]) / 3.0)
        return float(bar["close"])
    return float(bar[field])


# ==================================================================================================
# Core helper exposed to the portfolio engine
# ==================================================================================================
def realistic_fill_price(prices: pd.DataFrame, idx, side: int, mode: str,
                         slippage_bps: float = 1.0) -> float:
    """Realistic per-share fill price for a ``side`` order keyed on signal-bar date ``idx``.

    ``mode`` in :data:`FILL_MODES`:

    * ``close``     — fill at the signal bar's close (the idealized baseline).
    * ``next_open`` — fill at the **next** bar's open (the first executable price after a close
      signal); the honest fill for a strategy that decides on the close.
    * ``next_vwap`` — fill at the next bar's VWAP (synthesized as the typical price if no vwap
      column), a proxy for working the order through the session.

    Slippage is applied *adversely*: a buy (``side=+1``) pays up by ``slippage_bps``, a sell
    (``side=-1``) receives less. ``idx`` may be a Timestamp/date or an integer positional index.
    Raises ``KeyError`` if ``next_*`` is requested on the last bar (no next bar to fill on).
    """
    if mode not in FILL_MODES:
        raise ValueError(f"mode must be one of {FILL_MODES}, got {mode!r}")
    p = _indexed(prices)
    # Resolve the signal bar's positional location.
    if isinstance(idx, (int, np.integer)) and not isinstance(idx, bool):
        loc = int(idx)
    else:
        ts = pd.Timestamp(idx)
        if ts not in p.index:
            raise KeyError(f"date {ts!r} not in price frame")
        loc = int(p.index.get_loc(ts))

    if mode == "close":
        bar = p.iloc[loc]
        base = _bar_price(bar, "close")
    else:
        nxt = loc + 1
        if nxt >= len(p):
            raise KeyError("no next bar available for a next_open / next_vwap fill")
        bar = p.iloc[nxt]
        base = _bar_price(bar, "open" if mode == "next_open" else "vwap")

    slip = float(slippage_bps) / _BPS
    return float(base * (1.0 + side * slip))  # buy pays up, sell receives less


# ==================================================================================================
# Re-price a whole trade sheet under a chosen fill regime
# ==================================================================================================
def _side_series(trades: pd.DataFrame) -> pd.Series:
    """Trade direction as +1/-1; defaults to LONG when no ``side`` column is present."""
    if "side" in trades.columns:
        return np.sign(trades["side"]).replace(0, 1).astype(int)
    return pd.Series(1, index=trades.index, dtype=int)


def apply_fills(trades: pd.DataFrame, prices: pd.DataFrame, *, entry: str = "close",
                exit: str = "close", slippage_bps: float = 1.0,
                impact_model: ImpactModel | None = None) -> pd.DataFrame:
    """Re-price a trade sheet under realistic fills and return an adjusted copy.

    Parameters
    ----------
    trades:
        A sleeve's trade sheet. Must carry ``entry_date``/``exit_date`` and ``entry_price``/
        ``exit_price`` (the LOCKED schema shared across sleeves). An optional ``side`` column
        (+1 long / -1 short) is honored; absent, trades are LONG.
    prices:
        The index OHLC(V)+vwap frame the trades were generated on (``date`` column).
    entry, exit:
        Fill modes in :data:`FILL_MODES`. ``close`` reproduces the idealized backtest; ``next_open``
        / ``next_vwap`` are the executable alternatives (entry uses the bar **after** the signal).
    slippage_bps:
        Per-side spread/slippage in basis points, charged adversely on both legs.
    impact_model:
        Optional ATR/volume-scaled market-impact term (e.g. :class:`ATRImpact`,
        :class:`SquareRootImpact`); ``None`` disables it. Charged per side, added to slippage.

    Returns
    -------
    DataFrame
        A copy of ``trades`` with re-priced ``entry_price``/``exit_price``, the realized
        per-side cost in bps (``entry_cost_bps``/``exit_cost_bps``), gross and **net** returns
        (``ret_gross``/``ret_net``), and the realized exit dates resolved under the fill mode
        (``fill_entry_date``/``fill_exit_date``). Trades that cannot be filled (e.g. a ``next_*``
        signal on the last bar) are dropped, with a note logged to the returned frame's ``.attrs``.
    """
    if entry not in FILL_MODES or exit not in FILL_MODES:
        raise ValueError(f"entry/exit must be in {FILL_MODES}")
    out = trades.copy()
    if out.empty:
        for c in ("entry_price", "exit_price", "ret_gross", "ret_net",
                  "entry_cost_bps", "exit_cost_bps"):
            out[c] = pd.Series(dtype=float)
        return out

    p = _indexed(prices)
    atr_pct = _atr_pct_by_date(prices)
    sides = _side_series(out)
    slip = float(slippage_bps) / _BPS

    rows = []
    dropped = 0
    for (_, t), side in zip(out.iterrows(), sides):
        ed = pd.Timestamp(t["entry_date"])
        xd = pd.Timestamp(t["exit_date"])
        if ed not in p.index or xd not in p.index:
            dropped += 1
            continue
        e_loc = int(p.index.get_loc(ed))
        # --- entry leg --------------------------------------------------------------------------
        if entry == "close":
            e_fill_date = p.index[e_loc]
        else:
            if e_loc + 1 >= len(p):
                dropped += 1
                continue
            e_fill_date = p.index[e_loc + 1]
        e_atr = float(atr_pct.get(ed, np.nan))
        e_vol = float(p.iloc[e_loc].get("volume", np.nan))
        e_impact = _impact_bps(impact_model, atr_pct=e_atr, volume=e_vol)
        e_cost_bps = float(slippage_bps) + e_impact
        e_price = realistic_fill_price(prices, ed, int(side), entry, slippage_bps=0.0)
        e_price *= (1.0 + side * e_cost_bps / _BPS)  # slippage + impact, adverse to the order

        # --- exit leg ---------------------------------------------------------------------------
        # Exit closes the position, so its order side is the opposite of the entry.
        x_loc = int(p.index.get_loc(xd))
        if exit == "close":
            x_fill_date = p.index[x_loc]
        else:
            if x_loc + 1 >= len(p):
                dropped += 1
                continue
            x_fill_date = p.index[x_loc + 1]
        x_atr = float(atr_pct.get(xd, np.nan))
        x_vol = float(p.iloc[x_loc].get("volume", np.nan))
        x_impact = _impact_bps(impact_model, atr_pct=x_atr, volume=x_vol)
        x_cost_bps = float(slippage_bps) + x_impact
        x_price = realistic_fill_price(prices, xd, int(-side), exit, slippage_bps=0.0)
        x_price *= (1.0 - side * x_cost_bps / _BPS)  # exit is the opposite side -> adverse again

        gross = side * (x_price / e_price - 1.0)
        # Net of both legs' costs already baked into the fills; report the round-trip drag too.
        ret_net = gross
        ret_gross_ideal = side * (float(t["exit_price"]) / float(t["entry_price"]) - 1.0)

        rec = t.to_dict()
        rec.update(
            entry_price=round(float(e_price), 4),
            exit_price=round(float(x_price), 4),
            fill_entry_date=e_fill_date,
            fill_exit_date=x_fill_date,
            entry_cost_bps=round(e_cost_bps, 3),
            exit_cost_bps=round(x_cost_bps, 3),
            ret_gross=round(float(ret_gross_ideal), 6),
            ret_net=round(float(ret_net), 6),
        )
        rows.append(rec)

    result = pd.DataFrame(rows)
    result.attrs["dropped_unfillable"] = dropped
    result.attrs["fill_regime"] = {"entry": entry, "exit": exit,
                                   "slippage_bps": float(slippage_bps),
                                   "impact_model": type(impact_model).__name__ if impact_model
                                   else None}
    return result


# ==================================================================================================
# Drift-adjusted alpha + break-even cost
# ==================================================================================================
def holding_period_drift(trades: pd.DataFrame, prices: pd.DataFrame, *,
                         entry_col: str = "entry_date", exit_col: str = "exit_date") -> pd.Series:
    """Buy-and-hold index return over each trade's holding window (the drift to subtract).

    For every trade, the close-to-close index return from its entry bar to its exit bar — what a
    passive long would have collected over exactly that span. Aligned to ``trades.index``; trades
    whose dates fall outside the price frame get ``NaN``.
    """
    p = _indexed(prices)
    closes = p["close"].astype(float)
    out = []
    for _, t in trades.iterrows():
        ed, xd = pd.Timestamp(t[entry_col]), pd.Timestamp(t[exit_col])
        if ed in closes.index and xd in closes.index and closes.loc[ed] != 0:
            out.append(float(closes.loc[xd] / closes.loc[ed] - 1.0))
        else:
            out.append(np.nan)
    return pd.Series(out, index=trades.index, name="hp_drift")


def drift_adjusted_alpha(trade_returns, drift) -> float:
    """Mean of (per-trade net return − same-holding-period buy-and-hold drift).

    This is the project's honest edge metric: a part-time long silently collects the bull, so the
    only edge that counts is the *excess* over what passively holding the index for the same days
    would have earned. Positive -> real timing/selection edge; ~0 or negative -> beta dressed up.
    """
    r = pd.Series(np.asarray(trade_returns, dtype=float))
    d = pd.Series(np.asarray(drift, dtype=float))
    alpha = (r.to_numpy() - d.to_numpy())
    alpha = alpha[np.isfinite(alpha)]
    return float(alpha.mean()) if alpha.size else float("nan")


def break_even_cost(trade_returns, drift=None) -> float:
    """Per-side cost in **bps** that zeroes the (drift-adjusted) edge.

    The edge per trade is ``mean(ret - drift)`` (drift-adjusted alpha if ``drift`` is supplied,
    else raw expectancy). A round trip pays the per-side cost twice, so the break-even per side is

        break_even_bps = (edge / 2) * 1e4.

    A large value (tens of bps) means the edge is robust to realistic frictions; a small value
    (0-6 bps, the fear sleeves) means a single bp of slippage erases it — **alpha-fragile**.
    Returns ``0.0`` when the edge is already non-positive (nothing to give back).
    """
    r = np.asarray(trade_returns, dtype=float)
    if drift is not None:
        d = np.asarray(drift, dtype=float)
        edge_arr = r - d
    else:
        edge_arr = r
    edge_arr = edge_arr[np.isfinite(edge_arr)]
    if edge_arr.size == 0:
        return float("nan")
    edge = float(edge_arr.mean())
    if edge <= 0:
        return 0.0
    return float(edge / 2.0 * _BPS)


# ==================================================================================================
# Fill-sensitivity report (the headline deliverable)
# ==================================================================================================
def fill_sensitivity_report(sleeve_trades_by_name: Mapping[str, pd.DataFrame],
                            prices: pd.DataFrame, *, slippage_bps: float = 1.0,
                            impact_model: ImpactModel | None = None) -> pd.DataFrame:
    """Per-sleeve drift-adjusted alpha under close / next-open / vwap fills + break-even cost.

    For each named sleeve's trade sheet, re-price the fills under each entry mode (exit held at
    ``close`` — the exit is mechanical and not the front-running concern), compute the
    drift-adjusted alpha (per-trade net return minus same-holding-period buy-and-hold drift), and
    the per-side break-even cost in bps that zeroes the *close-fill* alpha.

    Reproduces the cross-desk finding:

    * **breakout** and **vrp** alpha barely moves across fill modes and has a large break-even ->
      *fill-insensitive*, the durable edges.
    * **ibs** and the **fear sleeves** (vvix, capitulation) bleed alpha on ``next_open`` and have a
      tiny break-even (~0-6 bps) -> *beta-robust, alpha-fragile* risk premia, sized small.

    Returns one row per sleeve, columns:
    ``n, alpha_close_pct, alpha_next_open_pct, alpha_vwap_pct, alpha_next_open_drop_pct,
    break_even_bps, expectancy_close_pct, fill_robust``.
    """
    rows = []
    for name, trades in sleeve_trades_by_name.items():
        if trades is None or trades.empty:
            rows.append(dict(sleeve=name, n=0, alpha_close_pct=np.nan,
                             alpha_next_open_pct=np.nan, alpha_vwap_pct=np.nan,
                             alpha_next_open_drop_pct=np.nan, break_even_bps=np.nan,
                             expectancy_close_pct=np.nan, fill_robust=False))
            continue

        alphas: dict[str, float] = {}
        expectancy_close = np.nan
        be = np.nan
        for mode in FILL_MODES:
            filled = apply_fills(trades, prices, entry=mode, exit="close",
                                 slippage_bps=slippage_bps, impact_model=impact_model)
            if filled.empty:
                alphas[mode] = np.nan
                continue
            drift = holding_period_drift(filled, prices,
                                         entry_col="entry_date", exit_col="exit_date")
            alphas[mode] = drift_adjusted_alpha(filled["ret_net"], drift)
            if mode == "close":
                expectancy_close = float(filled["ret_net"].mean())
                # Break-even on the *idealized* edge so it is a property of the signal, not the slip.
                be = break_even_cost(filled["ret_net"].to_numpy(), drift.to_numpy())

        a_close = alphas.get("close", np.nan)
        a_open = alphas.get("next_open", np.nan)
        drop = (a_close - a_open) if (np.isfinite(a_close) and np.isfinite(a_open)) else np.nan
        # "fill-robust" = alpha stays positive on next-open AND break-even is a meaningful buffer.
        robust = bool(np.isfinite(a_open) and a_open > 0 and np.isfinite(be) and be >= 6.0)
        rows.append(dict(
            sleeve=name,
            n=int(len(trades)),
            alpha_close_pct=_pct(a_close),
            alpha_next_open_pct=_pct(a_open),
            alpha_vwap_pct=_pct(alphas.get("next_vwap", np.nan)),
            alpha_next_open_drop_pct=_pct(drop),
            break_even_bps=round(float(be), 2) if np.isfinite(be) else np.nan,
            expectancy_close_pct=_pct(expectancy_close),
            fill_robust=robust,
        ))
    report = pd.DataFrame(rows).set_index("sleeve")
    return report


def _pct(x: float) -> float:
    """Fraction -> percent, NaN-safe, 4 dp."""
    return round(float(x) * 100.0, 4) if (x is not None and np.isfinite(x)) else np.nan
