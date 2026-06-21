"""Regression tests for the LOCKED trade-sheet totals block written by ``_write_ledger``.

``scripts/run_book.py`` is a script (not an installed module), so we import it by absolute path
via ``importlib.util.spec_from_file_location``. It imports ``startx.*`` (installed), so the import
resolves.

The desk's LOCKED rule: a +1-ATR or breakeven exit is a WIN/SCRATCH, NEVER a LOSS. Therefore a
SCRATCH is a NON-loser and its P&L must land on the WIN side, and NET TOTAL $ must reflect EVERY
trade (== TOTAL WIN $ + TOTAL LOSS $). This pins that math and the locked 3-row block shape.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

# --- import _write_ledger / _LEDGER_COLS from the script by absolute path -------------------
_RUN_BOOK = Path(__file__).resolve().parent.parent / "scripts" / "run_book.py"
_spec = importlib.util.spec_from_file_location("run_book", _RUN_BOOK)
run_book = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_book)

_write_ledger = run_book._write_ledger
_LEDGER_COLS = run_book._LEDGER_COLS


def _synthetic_ledger() -> pd.DataFrame:
    """A tiny ledger with one WIN, one LOSS, one SCRATCH. Every _LEDGER_COLS column is present so
    the body is written in full locked order. pnl_per_share / outcome are set explicitly."""
    rows = [
        # WIN: +2.00/share
        {"model": "base", "sleeve": "ibs", "entry_date": "2025-01-02", "entry_price": 100.0,
         "stop_price": 99.0, "stop_pct": -1.0, "exit_date": "2025-01-05", "exit_price": 102.0,
         "exit_reason": "chandelier", "bars_held": 3, "ret": 0.02, "pnl_per_share": 2.00,
         "pnl_contrib": 0.02, "risk_pct": 1.0, "R": 2.0, "entry_rule": "e", "exit_rule": "x",
         "regime": "bull", "conviction": "hi", "thesis": "t", "outcome": "WIN"},
        # LOSS: -1.50/share
        {"model": "base", "sleeve": "ibs", "entry_date": "2025-02-02", "entry_price": 100.0,
         "stop_price": 99.0, "stop_pct": -1.0, "exit_date": "2025-02-03", "exit_price": 98.5,
         "exit_reason": "hard_stop", "bars_held": 1, "ret": -0.015, "pnl_per_share": -1.50,
         "pnl_contrib": -0.015, "risk_pct": 1.0, "R": -1.5, "entry_rule": "e", "exit_rule": "x",
         "regime": "bull", "conviction": "hi", "thesis": "t", "outcome": "LOSS"},
        # SCRATCH: +0.30/share (a +1-ATR / breakeven NON-loser — must count on the WIN side)
        {"model": "base", "sleeve": "ibs", "entry_date": "2025-03-02", "entry_price": 100.0,
         "stop_price": 99.0, "stop_pct": -1.0, "exit_date": "2025-03-04", "exit_price": 100.30,
         "exit_reason": "max_hold", "bars_held": 2, "ret": 0.003, "pnl_per_share": 0.30,
         "pnl_contrib": 0.003, "risk_pct": 1.0, "R": 0.3, "entry_rule": "e", "exit_rule": "x",
         "regime": "bull", "conviction": "lo", "thesis": "t", "outcome": "SCRATCH"},
    ]
    return pd.DataFrame(rows)[_LEDGER_COLS]


def test_write_ledger_totals_include_scratch_on_win_side(tmp_path):
    led = _synthetic_ledger()
    path = tmp_path / "ledger.csv"
    nrows = _write_ledger(led, str(path))
    assert nrows == 3  # three body trades

    out = pd.read_csv(path)

    # (a) body columns are exactly _LEDGER_COLS (the ones present, in locked order)
    assert list(out.columns) == _LEDGER_COLS
    body = out.iloc[:3]
    assert list(body["outcome"]) == ["WIN", "LOSS", "SCRATCH"]

    # (b) the 3 totals rows exist with the right labels, in order, appended after the body
    totals = out.iloc[3:]
    assert list(totals["sleeve"]) == ["TOTAL WIN $", "TOTAL LOSS $", "NET TOTAL $"]

    tot = dict(zip(totals["sleeve"], totals["pnl_per_share"]))
    win_total = round(2.00 + 0.30, 2)   # WIN + SCRATCH
    loss_total = round(-1.50, 2)
    net_total = round(2.00 - 1.50 + 0.30, 2)  # every trade

    # (c) TOTAL WIN $ includes the scratch row (would be 2.00 if scratch were dropped)
    assert tot["TOTAL WIN $"] == win_total
    assert tot["TOTAL WIN $"] != round(2.00, 2)  # guard: scratch is NOT excluded
    assert tot["TOTAL LOSS $"] == loss_total

    # (d) NET == round(sum of ALL body pnl_per_share, 2) == WIN$ + LOSS$
    assert tot["NET TOTAL $"] == net_total
    assert tot["NET TOTAL $"] == round(float(body["pnl_per_share"].sum()), 2)
    assert tot["NET TOTAL $"] == round(tot["TOTAL WIN $"] + tot["TOTAL LOSS $"], 2)
