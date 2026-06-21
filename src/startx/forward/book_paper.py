"""Genuine paper-forward harness driven by the THREE VALIDATED PRODUCTION BOOKS.

================================ WHAT THIS IS ================================
This is the *honest* forward harness the forensic audit asked for. The sibling
``startx.forward.paper`` runs the **ABANDONED ``prob_up`` directional model** (OOS
AUC ~0.50, a coin flip — see its banner and CLAUDE.md "Settled findings"); it is
research/illustrative only. THIS module instead drives the three LONG-ONLY books
that ``scripts/run_book.py`` runs and that the desk treats as the validated system:

  * short_swing — IBS<0.1 dip-buy, 1-ATR stop / 3-ATR chandelier, 10-day cap.
  * long_swing  — breakout + fear (VVIX) capitulation, same exits, 63-day cap.
  * position    — 200-SMA ±3% band trend core (its own engine; the trend
                  breakdown IS the stop, no ATR chandelier).

================================ HONESTY: WHAT THIS IS NOT ====================
This is a **backtest-driven forward SIMULATION on historical HOLDOUT data**, NOT a
live-broker track record. There is no real broker, no real fills, no real slippage
beyond the modelled ~2bp cost. The value is narrow and specific: by clipping every
ENTRY to bars at/after ``forward_start`` we get an OUT-OF-SAMPLE paper blotter on a
slice the book's parameters were NOT tuned on (the desk calibrated on the 2019-22
era; 2023-26 is genuinely held out). Read the numbers as "would the locked rules
have held up forward on data they never saw", NOT as "this is money the desk made".

Point-in-time guarantee
-----------------------
The underlying engines are causal and the window is applied at the ENTRY gate, so
no post-``forward_start`` information can decide a pre-existing trade:

  * ``run_portfolio(..., start=forward_start, end=end)`` restricts ENTRIES to the
    window (``in_window`` gate) and reports stats over the in-window equity slice.
    A signal/exit is computed bar-by-bar from causal indicators only.
  * ``position_book(..., start=forward_start, end=end)`` uses a state series that is
    decided on the *prior* close (shifted) and clamps any stint already open at the
    window start to the window's first bar.

Consequence (asserted in the test): **no trade in the OOS blotter has an
``entry_date`` strictly before ``forward_start``.** A position-book stint that was
already open when the holdout begins is booked from the window's first bar, never
from its true (pre-window) entry, so it cannot smuggle in pre-holdout P&L.

Everything returned is plain pandas — no global state, no network. Prices come from
the local parquet cache (the same source ``scripts/run_book.py`` reads).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..portfolio import run_portfolio
from ..strategy.mean_reversion import ibs_signals
from ..strategy.momentum_breakout import breakout_signals
from ..strategy.trend_position import position_book
from ..strategy.volatility_premium import fear_signals

#: Default holdout window. 2023-01-01 → 2026-06-19 is genuinely OOS vs the desk's
#: 2019-22 calibration era (see CLAUDE.md analysis window + run_book stress grid).
DEFAULT_FORWARD_START = "2023-01-01"
DEFAULT_END = "2026-06-19"

#: Where the price parquets live (same cache scripts/run_book.py reads).
_PRICE_DIR = Path("data/cache/prices")

#: Books that ride the concurrent chandelier engine (``run_portfolio``). Each value is
#: ``(sleeves, exits, gross_cap)`` exactly as the locked book definitions in run_book.py.
#: position is handled separately (its own engine), see :func:`_run_position_book`.
_PORTFOLIO_BOOKS = ("short_swing", "long_swing")

#: Stable blotter schema (one row per paper trade across ALL books, tagged by book/sleeve).
BLOTTER_COLUMNS: list[str] = [
    "book", "sleeve", "entry_date", "entry_price", "exit_date", "exit_price",
    "exit_reason", "bars_held", "ret", "pnl_per_share", "outcome", "status",
]


@dataclass
class BookForward:
    """One book's out-of-sample forward result on the holdout window.

    Attributes
    ----------
    book:
        ``'short_swing'`` / ``'long_swing'`` / ``'position'``.
    blotter:
        Per-trade paper blotter (schema :data:`BLOTTER_COLUMNS`), every ``entry_date``
        at/after ``forward_start``.
    equity:
        The book's in-window equity curve (starts at ~1.0 on the first in-window bar).
    stats:
        OOS headline stats — ``{n, win_rate, total_return, max_drawdown, ann_sharpe}``
        (plus the engine's native extras passed through).
    """

    book: str
    blotter: pd.DataFrame
    equity: pd.Series
    stats: dict = field(default_factory=dict)


@dataclass
class ForwardBooksResult:
    """The full forward run across all three books + the combined blotter.

    Attributes
    ----------
    forward_start, end:
        The holdout window bounds (entries clipped to ``[forward_start, end]``).
    books:
        ``dict[name -> BookForward]`` for each of the three books.
    combined_blotter:
        Every trade from every book in one frame, tagged by ``book``/``sleeve``,
        sorted by entry date (schema :data:`BLOTTER_COLUMNS`).
    summary:
        One honest one-line string describing the run (also see :func:`one_line_summary`).
    """

    forward_start: pd.Timestamp
    end: pd.Timestamp
    books: dict[str, BookForward]
    combined_blotter: pd.DataFrame
    summary: str = ""


# --------------------------------------------------------------------------------------- #
# Data loading (local parquet cache — same source as scripts/run_book.py)
# --------------------------------------------------------------------------------------- #
def _load_prices(symbol: str, price_dir: Path) -> pd.DataFrame:
    """Load one symbol's OHLCV parquet, sorted, with a datetime ``date`` column."""
    p = pd.read_parquet(price_dir / f"{symbol}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p.sort_values("date").reset_index(drop=True)


def _empty_blotter() -> pd.DataFrame:
    return pd.DataFrame(columns=BLOTTER_COLUMNS)


# --------------------------------------------------------------------------------------- #
# Per-trade -> unified blotter row
# --------------------------------------------------------------------------------------- #
def _blotter_from_ledger(ledger: pd.DataFrame, book: str, cost_bps: float = 2.0) -> pd.DataFrame:
    """Project an engine ledger onto the unified paper blotter schema.

    Adds ``book`` (the tag), a per-share P&L net of ``cost_bps`` (the locked CSV dollar
    column convention), and a ``status`` (``open`` when the trade was still on at the
    window end, else ``closed``). Outcome is taken straight from the engine (which already
    honours the desk rule that a +1-ATR / breakeven exit is a WIN/SCRATCH, never a LOSS).
    """
    if ledger is None or ledger.empty:
        return _empty_blotter()
    led = ledger.copy()
    led["book"] = book
    # per-share P&L net of ~cost_bps round-trip (mirrors run_book's locked dollar column)
    led["pnl_per_share"] = (
        (led["exit_price"] - led["entry_price"]) - led["entry_price"] * (cost_bps / 10000.0)
    ).round(4)
    # An "open" / "time"-capped-at-window-end stint is still on the book at the holdout end.
    led["status"] = np.where(led.get("exit_reason", "").astype(str).eq("open"), "open", "closed")
    for col in BLOTTER_COLUMNS:
        if col not in led.columns:
            led[col] = np.nan
    out = led[BLOTTER_COLUMNS].copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"] = pd.to_datetime(out["exit_date"])
    return out.sort_values(["entry_date", "sleeve"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------- #
# Stats normaliser — one OOS headline shape across the two different engines
# --------------------------------------------------------------------------------------- #
def _normalise_stats(raw: dict, ledger: pd.DataFrame) -> dict:
    """Coerce either engine's stats dict into one OOS headline shape.

    ``run_portfolio`` reports ``win_rate``; ``position_book`` doesn't (it reports a
    benchmark instead), so we derive win% from the ledger outcomes. Every value here is
    the engine's own in-window (holdout) figure — we do not recompute Sharpe/maxDD.
    """
    n = int(len(ledger)) if ledger is not None else 0
    if "win_rate" in raw and not pd.isna(raw.get("win_rate", float("nan"))):
        win_rate = float(raw["win_rate"])
    elif n:
        win_rate = float((ledger["outcome"] == "WIN").mean())
    else:
        win_rate = float("nan")
    out = {
        "n": raw.get("n", n),
        "win_rate": win_rate,
        "total_return": float(raw.get("total_return", float("nan"))),
        "max_drawdown": float(raw.get("max_drawdown", float("nan"))),
        "ann_sharpe": float(raw.get("ann_sharpe", float("nan"))),
    }
    # pass through the engine's native extras (profit_factor, exposure, cagr, calmar, ...)
    for k in ("profit_factor", "exposure", "cagr", "calmar", "vol_ann", "turnover"):
        if k in raw:
            out[k] = raw[k]
    return out


# --------------------------------------------------------------------------------------- #
# The three book runners (entries clipped to the holdout window = point-in-time OOS)
# --------------------------------------------------------------------------------------- #
def _run_portfolio_book(
    book: str,
    spy: pd.DataFrame,
    aux: dict,
    *,
    forward_start: str,
    end: str,
    gross_cap: float,
    cost_bps: float,
) -> BookForward:
    """Run a chandelier-engine book (short_swing / long_swing) over the holdout window."""
    if book == "short_swing":
        sleeves = {"ibs": lambda s, a: ibs_signals(s)}
        exits = {"ibs": (1.0, 3.0, 10)}
    elif book == "long_swing":
        sleeves = {
            "breakout": lambda s, a: breakout_signals(s, 20, 200),
            "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"]),
        }
        exits = {"breakout": (1.0, 3.0, 63), "fear": (1.0, 3.0, 63)}
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"{book!r} is not a chandelier-engine book")

    res = run_portfolio(
        spy, aux, sleeves, exits=exits, gross_cap=gross_cap,
        cost_bps=cost_bps, start=forward_start, end=end,
    )
    blotter = _blotter_from_ledger(res.ledger, book, cost_bps=cost_bps)
    stats = _normalise_stats(res.stats, res.ledger)
    return BookForward(book=book, blotter=blotter, equity=res.equity, stats=stats)


def _run_position_book(
    spy: pd.DataFrame,
    *,
    forward_start: str,
    end: str,
    cost_bps: float,
    sma_len: int = 200,
    band: float = 0.03,
) -> BookForward:
    """Run System #3 (200-SMA ±band trend core) over the holdout window."""
    res = position_book(spy, sma_len=sma_len, band=band, cost_bps=cost_bps,
                        start=forward_start, end=end)
    blotter = _blotter_from_ledger(res.ledger, "position", cost_bps=cost_bps)
    stats = _normalise_stats(res.stats, res.ledger)
    # keep the buy-and-hold benchmark visible — the position book's honest yardstick
    stats["benchmark"] = res.benchmark
    return BookForward(book="position", blotter=blotter, equity=res.equity, stats=stats)


# --------------------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------------------- #
def run_forward_books(
    forward_start: str | pd.Timestamp = DEFAULT_FORWARD_START,
    end: str | pd.Timestamp = DEFAULT_END,
    *,
    price_dir: str | Path = _PRICE_DIR,
    gross_cap: float = 1.5,
    cost_bps: float = 2.0,
    spy: pd.DataFrame | None = None,
    aux: dict | None = None,
) -> ForwardBooksResult:
    """Run all three validated books in a forward / holdout SIMULATION over ``[forward_start, end]``.

    Every book's ENTRIES are clipped to the window, so the resulting blotter is genuinely
    out-of-sample on a slice the parameters were not tuned on (see the module docstring for
    the point-in-time guarantee — and its honest limits: this is a historical-holdout
    backtest, NOT a live-broker track record).

    Parameters
    ----------
    forward_start, end:
        Inclusive holdout window. Entries before ``forward_start`` are never taken.
    price_dir:
        Directory of the OHLCV parquet cache (defaults to ``data/cache/prices``).
    gross_cap:
        Concurrent gross-leverage cap for the chandelier books (1.5× = run_book default).
    cost_bps:
        Round-trip cost in basis points, charged once per trade (~2bp SPY).
    spy, aux:
        Optional pre-loaded frames (mostly for tests). ``aux`` must carry ``'vix'`` /
        ``'vvix'`` for the long-swing fear sleeve and ``'gld'`` for the gold-calm overlay.
        If omitted, all are loaded from ``price_dir``.

    Returns
    -------
    ForwardBooksResult
        ``books`` (per-book :class:`BookForward`), the tagged ``combined_blotter`` across
        all books, and a one-line honest ``summary``.
    """
    fs = pd.Timestamp(forward_start)
    en = pd.Timestamp(end)
    price_dir = Path(price_dir)

    if spy is None:
        spy = _load_prices("SPY", price_dir)
        spy.attrs["symbol"] = "SPY"
    if aux is None:
        aux = {
            "vix": _load_prices("_VIX", price_dir),
            "vvix": _load_prices("_VVIX", price_dir),
            "gld": _load_prices("GLD", price_dir),
        }

    books: dict[str, BookForward] = {}
    for name in _PORTFOLIO_BOOKS:
        books[name] = _run_portfolio_book(
            name, spy, aux, forward_start=str(fs.date()), end=str(en.date()),
            gross_cap=gross_cap, cost_bps=cost_bps,
        )
    books["position"] = _run_position_book(
        spy, forward_start=str(fs.date()), end=str(en.date()), cost_bps=cost_bps,
    )

    # combined, tagged blotter (every trade from every book, sorted by entry)
    frames = [b.blotter for b in books.values() if not b.blotter.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["entry_date", "book", "sleeve"]).reset_index(drop=True)
    else:
        combined = _empty_blotter()

    summary = one_line_summary(fs, en, books, combined)
    return ForwardBooksResult(
        forward_start=fs, end=en, books=books, combined_blotter=combined, summary=summary,
    )


def one_line_summary(
    forward_start: pd.Timestamp, end: pd.Timestamp,
    books: dict[str, BookForward], combined: pd.DataFrame,
) -> str:
    """One honest sentence: what this is, the window, and each book's OOS headline."""
    bits = []
    for name, b in books.items():
        s = b.stats
        bits.append(
            f"{name}(n={s['n']}, tot={s['total_return']*100:+.1f}%, "
            f"Sharpe={s['ann_sharpe']:.2f}, win={s['win_rate']*100:.0f}%, "
            f"maxDD={s['max_drawdown']*100:.1f}%)"
        )
    return (
        f"OOS PAPER-FORWARD (historical-holdout SIMULATION, not a live broker) "
        f"{forward_start.date()}→{end.date()}, {len(combined)} trades across 3 books: "
        + " | ".join(bits)
    )
