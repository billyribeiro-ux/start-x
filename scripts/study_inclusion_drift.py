#!/usr/bin/env python
"""S&P 500 INCLUSION/DELETION DRIFT — pre-registered, survivorship-free event study.

PRE-REGISTERED HYPOTHESES (fixed BEFORE looking at results; each with its own kill)
-----------------------------------------------------------------------------------
H1 (deletion rebound): stocks REMOVED from the index suffer forced selling into the
   removal, then rebound over the following 5-40 sessions. Other side: index funds
   who MUST sell at a known time regardless of price.
   Trade: LONG the removed stock, hedged short SPY. KILL: holdout (2022-2026) net
   rebound <= 0.
H2 (addition fade): stocks ADDED run up into inclusion and FADE over the following
   5-40 sessions (buy-the-rumor unwind). Other side: index funds who must buy.
   Trade: SHORT the added stock, hedged long SPY (a short — desk prior: expect kill).
   KILL: holdout net fade <= 0.

METHOD (locked)
---------------
* Real-data window: 2018-01-01 -> 2026-06-26 ONLY (pre-2016 survivorship-contaminated).
* Events from the survivorship-free FMP change feed (data/cache/sp500_changes.parquet,
  via startx.data.membership.SP500Membership). Feed date conventions are MIXED
  (sometimes the Friday rebalance-trade date, sometimes the Monday effective date),
  so entry = OPEN of the first SPY trading day STRICTLY AFTER the feed date. That is
  next-bar-open and lookahead-free under BOTH conventions (S&P announces ~5 days
  before effectiveness; we act at/after the recorded date, never before).
* Self-swap rows (added == removed, e.g. VFC/VFC): these encode demotions to the
  MidCap 400 or ticker-change/merger deaths — never genuine S&P 500 additions.
  Pre-registered rule: excluded from the ADD leg; included as REMOVE-leg candidates
  (ticker-change deaths auto-drop via the no-entry-bar tradability rule).
* Windows [t+1..t+k], k in {5, 10, 21, 40} sessions. Return = stock close(t+k) /
  stock open(t+1) - 1, minus same-timestamp SPY leg (abnormal vs SPY).
* Costs: 3 bp/side on the stock AND 3 bp/side on the SPY hedge = 12 bp per round
  trip, deducted from every event return ("net").
* Tradability (all counted, nothing silent):
    - no cached/fetchable price file            -> dropped (counted)
    - no stock bar at the entry day             -> untradeable (counted; this is the
      honest fate of merger/bankruptcy removals — the rebound trade never exists)
    - stock series ends mid-window & >10 days before the SPY cache tail
                                                -> genuine delisting: terminal exit
      at last bar (real economics of a cash-out/bankruptcy), included & counted
    - window extends past the study end / cache tail -> incomplete, excluded & counted
* DEV = feed date <= 2021-12-31.  HOLDOUT = 2022-01-01 .. 2026-06-26, touched once.
  Selection metric on DEV (win_rate BANNED): per-event net Sharpe (mean/std of net
  per-event returns), min n=10. ONE window per leg is pre-committed on DEV; the
  holdout verdict uses ONLY that window (full holdout table printed for transparency,
  not for selection).
* N_TRIALS = 8 (2 legs x 4 windows — every configuration this study ever touched).
  DSR = startx.validation.metrics.deflated_sharpe(per-event net returns, n_trials=8).
* Small-n honesty: n < 30 per leg-window => "UNDERPOWERED", never an edge claim.

Usage:  cd /home/user/start-x && source .venv/bin/activate && \
        python scripts/study_inclusion_drift.py
Writes per-event returns to data/cache/inclusion_drift_events.csv (audit trail).
Read-only w.r.t. every existing file; no git operations.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from startx.validation.metrics import deflated_sharpe  # noqa: E402

# ------------------------------------------------------------------ locked constants
STUDY_START = pd.Timestamp("2018-01-01")
STUDY_END = pd.Timestamp("2026-06-26")
DEV_END = pd.Timestamp("2021-12-31")          # DEV = 2018-2021, HOLDOUT = 2022-2026
WINDOWS = (5, 10, 21, 40)                     # sessions, [t+1 .. t+k]
COST_PER_SIDE_STOCK = 0.0003                  # 3 bp/side
COST_PER_SIDE_HEDGE = 0.0003                  # 3 bp/side on SPY hedge (conservative)
TOTAL_COST = 2 * COST_PER_SIDE_STOCK + 2 * COST_PER_SIDE_HEDGE   # 12 bp round trip
N_TRIALS = 8                                  # 2 legs x 4 windows, honest count
MIN_N_DEV = 10                                # a DEV cell needs >= this many events
UNDERPOWERED_N = 30
PRICES_DIR = "data/cache/prices"
CHANGES_PARQUET = "data/cache/sp500_changes.parquet"
EVENTS_OUT = "data/cache/inclusion_drift_events.csv"
DELIST_GAP_DAYS = 10   # series ending >10 calendar days before SPY tail = real delisting


# ------------------------------------------------------------------------- data I/O
def load_events() -> pd.DataFrame:
    """Event list from the survivorship-free change feed, with drop accounting."""
    chg = pd.read_parquet(CHANGES_PARQUET)
    chg["date"] = pd.to_datetime(chg["date"])
    w = chg[(chg["date"] >= STUDY_START) & (chg["date"] <= STUDY_END)].copy()

    rows = []
    for _, r in w.iterrows():
        added, removed = str(r["added"]), str(r["removed"])
        self_swap = bool(added) and added == removed
        if added and not self_swap:
            rows.append({"date": r["date"], "symbol": added, "leg": "add",
                         "self_swap": False})
        if removed:
            rows.append({"date": r["date"], "symbol": removed, "leg": "rem",
                         "self_swap": self_swap})
    ev = pd.DataFrame(rows).sort_values(["date", "leg", "symbol"]).reset_index(drop=True)
    return ev


def load_prices(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(PRICES_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------- event returns
def build_event_returns(ev: pd.DataFrame, spy: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-event, per-window abnormal returns with full drop accounting."""
    spy_dates = spy["date"].to_numpy()
    spy_open = spy["open"].to_numpy(dtype=float)
    spy_close = spy["close"].to_numpy(dtype=float)
    spy_last = pd.Timestamp(spy_dates[-1])

    drops = {"no_price_file": [], "untradeable_at_entry": []}
    incomplete = {k: 0 for k in WINDOWS}
    delisted_mid = {k: 0 for k in WINDOWS}
    out = []

    px_cache: dict[str, pd.DataFrame | None] = {}
    for _, e in ev.iterrows():
        sym, d, leg = e["symbol"], pd.Timestamp(e["date"]), e["leg"]
        if sym not in px_cache:
            px_cache[sym] = load_prices(sym)
        px = px_cache[sym]
        if px is None:
            drops["no_price_file"].append((str(d.date()), sym, leg))
            continue

        i1 = int(np.searchsorted(spy_dates, np.datetime64(d), side="right"))
        if i1 >= len(spy_dates):
            for k in WINDOWS:
                incomplete[k] += 1
            continue
        entry_day = pd.Timestamp(spy_dates[i1])

        stk = px.set_index("date")
        if entry_day not in stk.index or not np.isfinite(stk.at[entry_day, "open"]) \
                or stk.at[entry_day, "open"] <= 0:
            drops["untradeable_at_entry"].append((str(d.date()), sym, leg))
            continue
        stk_entry_open = float(stk.at[entry_day, "open"])
        spy_entry_open = float(spy_open[i1])
        stk_last = stk.index[-1]

        row = {"event_date": d, "symbol": sym, "leg": leg,
               "self_swap": bool(e["self_swap"]), "entry_day": entry_day,
               "split": "DEV" if d <= DEV_END else "HOLDOUT",
               "year": int(d.year)}
        any_window = False
        for k in WINDOWS:
            ix = i1 + k - 1
            if ix >= len(spy_dates) or pd.Timestamp(spy_dates[ix]) > STUDY_END:
                incomplete[k] += 1
                row[f"ar_{k}"] = np.nan
                continue
            exit_day = pd.Timestamp(spy_dates[ix])
            if stk_last < exit_day:
                # series ends mid-window: genuine delisting vs cache-tail truncation
                if stk_last <= spy_last - pd.Timedelta(days=DELIST_GAP_DAYS):
                    # real delisting (merger cash-out / bankruptcy): terminal exit at
                    # last bar, SPY hedge marked the same day. Real economics, included.
                    j = int(np.searchsorted(spy_dates, np.datetime64(stk_last),
                                            side="right")) - 1
                    stk_exit = float(stk["close"].iloc[-1])
                    spy_exit = float(spy_close[j])
                    delisted_mid[k] += 1
                    row[f"delisted_{k}"] = True
                else:
                    incomplete[k] += 1
                    row[f"ar_{k}"] = np.nan
                    continue
            else:
                # normal exit; if the stock is missing that exact bar (halt), use the
                # last stock bar at/before exit_day, SPY marked the same day
                sub = stk.loc[:exit_day]
                actual_exit = sub.index[-1]
                stk_exit = float(sub["close"].iloc[-1])
                j = int(np.searchsorted(spy_dates, np.datetime64(actual_exit),
                                        side="right")) - 1
                spy_exit = float(spy_close[j])
            ar = (stk_exit / stk_entry_open - 1.0) - (spy_exit / spy_entry_open - 1.0)
            row[f"ar_{k}"] = ar
            any_window = True
        if any_window or any(np.isfinite(row.get(f"ar_{k}", np.nan)) for k in WINDOWS):
            out.append(row)
        else:
            out.append(row)   # keep the shell row; all-window-incomplete is visible

    res = pd.DataFrame(out)
    acct = {"drops": drops, "incomplete": incomplete, "delisted_mid": delisted_mid}
    return res, acct


# ---------------------------------------------------------------------- aggregation
def leg_direction(leg: str) -> float:
    return 1.0 if leg == "rem" else -1.0     # H1 long deletions, H2 short additions


def net_returns(df: pd.DataFrame, leg: str, k: int) -> pd.Series:
    ar = df.loc[df["leg"] == leg, f"ar_{k}"].dropna()
    return leg_direction(leg) * ar - TOTAL_COST


def cell_stats(df: pd.DataFrame, leg: str, k: int) -> dict:
    ar = df.loc[df["leg"] == leg, f"ar_{k}"].dropna()
    net = leg_direction(leg) * ar - TOTAL_COST
    n = len(net)
    sd = net.std(ddof=1) if n > 1 else np.nan
    return {
        "window": f"t+1..t+{k}",
        "n": n,
        "gross_AR_mean_bp": ar.mean() * 1e4 if n else np.nan,
        "gross_AR_median_bp": ar.median() * 1e4 if n else np.nan,
        "net_mean_bp": net.mean() * 1e4 if n else np.nan,
        "net_median_bp": net.median() * 1e4 if n else np.nan,
        "net_std_bp": sd * 1e4 if n > 1 else np.nan,
        "net_sharpe_per_event": (net.mean() / sd) if (n > 1 and sd > 0) else np.nan,
        "underpowered": n < UNDERPOWERED_N,
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    assert STUDY_START >= pd.Timestamp("2018-01-01"), "locked real-data window"

    spy = load_prices("SPY")
    assert spy is not None and not spy.empty, "SPY cache missing"
    spy = spy[(spy["date"] >= STUDY_START - pd.Timedelta(days=10))
              & (spy["date"] <= STUDY_END)].reset_index(drop=True)

    ev = load_events()
    print("=" * 100)
    print("S&P 500 INCLUSION/DELETION DRIFT — pre-registered event study "
          f"({STUDY_START.date()} -> {STUDY_END.date()})")
    print("=" * 100)
    n_add = int((ev["leg"] == "add").sum())
    n_rem = int((ev["leg"] == "rem").sum())
    n_self = int(ev["self_swap"].sum())
    print(f"\nEVENT FEED: {len(ev)} leg-events  |  add={n_add}  rem={n_rem} "
          f"(of which {n_self} self-swap rows counted as deletions, 0 as additions)")

    res, acct = build_event_returns(ev, spy)
    res.to_csv(EVENTS_OUT, index=False)
    print(f"per-event audit trail -> {EVENTS_OUT}")

    # ------------------------------------------------------------- drop accounting
    print("\n--- DROP / TRADABILITY ACCOUNTING (nothing silent) " + "-" * 46)
    for reason, lst in acct["drops"].items():
        by_leg = pd.Series([leg for _, _, leg in lst]).value_counts().to_dict() if lst else {}
        print(f"{reason:>24}: {len(lst):3d}  {by_leg}")
        for d, s, leg in lst:
            print(f"{'':>26}{d}  {s:<6} [{leg}]")
    print(f"{'window incomplete':>24}: " + "  ".join(
        f"k={k}: {acct['incomplete'][k]}" for k in WINDOWS)
        + "   (cache tail / study end — excluded from that window)")
    print(f"{'delisted mid-window':>24}: " + "  ".join(
        f"k={k}: {acct['delisted_mid'][k]}" for k in WINDOWS)
        + "   (terminal exit at last bar — INCLUDED, real economics)")
    n_total = len(ev)
    n_dropped_entry = sum(len(v) for v in acct["drops"].values())
    print(f"{'dropped fraction':>24}: {n_dropped_entry}/{n_total} "
          f"= {n_dropped_entry / n_total:.1%} of leg-events never enter any window")

    # ---------------------------------------------------------------- results tables
    legs = {"rem": "H1 DELETION REBOUND (long removed stock vs SPY)",
            "add": "H2 ADDITION FADE (short added stock vs SPY)"}
    chosen: dict[str, int] = {}

    for split in ("DEV", "HOLDOUT"):
        sub = res[res["split"] == split]
        for leg, title in legs.items():
            tab = pd.DataFrame([cell_stats(sub, leg, k) for k in WINDOWS])
            print(f"\n--- {split} ({'2018-2021' if split == 'DEV' else '2022-2026'}) — "
                  f"{title}\n    costs: 12 bp round trip (3bp/side stock + 3bp/side SPY hedge)")
            print(tab.to_string(index=False))
            if split == "DEV":
                ok = tab[(tab["n"] >= MIN_N_DEV) & tab["net_sharpe_per_event"].notna()]
                if len(ok):
                    best = ok.loc[ok["net_sharpe_per_event"].idxmax()]
                    chosen[leg] = WINDOWS[list(tab["window"]).index(best["window"])]
                    print(f"    >> PRE-COMMITTED window for {leg}: {best['window']} "
                          f"(DEV net per-event Sharpe {best['net_sharpe_per_event']:.3f})")

    # ------------------------------------------------- holdout verdicts, chosen only
    print("\n" + "=" * 100)
    print(f"HOLDOUT VERDICTS — pre-committed window only, N_TRIALS={N_TRIALS} "
          "(2 legs x 4 windows, every config touched)")
    print("=" * 100)
    hold = res[res["split"] == "HOLDOUT"]
    dev = res[res["split"] == "DEV"]
    for leg, title in legs.items():
        if leg not in chosen:
            print(f"\n{title}: no DEV window met n>={MIN_N_DEV} — INCONCLUSIVE")
            continue
        k = chosen[leg]
        net_h = net_returns(hold, leg, k)
        net_d = net_returns(dev, leg, k)
        n = len(net_h)
        mean_bp = net_h.mean() * 1e4
        med_bp = net_h.median() * 1e4
        sd_bp = net_h.std(ddof=1) * 1e4
        dsr_h = deflated_sharpe(net_h.to_numpy(), n_trials=N_TRIALS)
        dsr_d = deflated_sharpe(net_d.to_numpy(), n_trials=N_TRIALS)
        print(f"\n{title}\n  window t+1..t+{k} | HOLDOUT n={n} | net mean {mean_bp:+.1f} bp"
              f" | median {med_bp:+.1f} bp | sd {sd_bp:,.0f} bp"
              f" | DSR(holdout, n_trials={N_TRIALS}) = {dsr_h:.3f}"
              f" | DSR(dev) = {dsr_d:.3f}")
        yr = (pd.DataFrame({"year": hold.loc[hold["leg"] == leg, "year"]
                            [hold.loc[hold["leg"] == leg, f"ar_{k}"].notna()],
                            "net": net_h})
              .groupby("year")["net"]
              .agg(n="size", mean_bp=lambda s: s.mean() * 1e4,
                   median_bp=lambda s: s.median() * 1e4))
        dyr = (pd.DataFrame({"year": dev.loc[dev["leg"] == leg, "year"]
                             [dev.loc[dev["leg"] == leg, f"ar_{k}"].notna()],
                             "net": net_d})
               .groupby("year")["net"]
               .agg(n="size", mean_bp=lambda s: s.mean() * 1e4,
                    median_bp=lambda s: s.median() * 1e4))
        print("  per-year consistency (DEV then HOLDOUT):")
        print(pd.concat([dyr, yr]).to_string())
        killed = mean_bp <= 0
        under = n < UNDERPOWERED_N
        if killed:
            verdict = "KILL (holdout net <= 0)"
        elif under:
            verdict = "INCONCLUSIVE / UNDERPOWERED (holdout net > 0 but n < 30)"
        elif dsr_h < 0.95:
            verdict = f"INCONCLUSIVE (net > 0 but DSR {dsr_h:.2f} < 0.95 at n_trials={N_TRIALS})"
        else:
            verdict = f"PROMOTE-CANDIDATE (net > 0, DSR {dsr_h:.2f} >= 0.95)"
        print(f"  VERDICT [{leg}]: {verdict}")

    # ------------------------------------------------------------- sanity: extremes
    print("\n--- extreme events (|gross AR| > 50% in any window; kept in, listed for audit)")
    ext = res[[c for c in res.columns if c.startswith("ar_")]].abs().max(axis=1) > 0.5
    cols = ["event_date", "symbol", "leg"] + [f"ar_{k}" for k in WINDOWS]
    print(res.loc[ext, cols].to_string(index=False) if ext.any() else "  none")


if __name__ == "__main__":
    main()
