"""The HONEST INVESTABLE stress-gated swing satellite + its ensemble with the core long model.

WHY (LEDGER R6 / DECODE headline box): the 'ensemble 1.80' from `decode_ensemble_v2.py` is
construction-optimistic — `improve.calendar_book` AVERAGES the returns of up to 10 concurrent
stress-day names as if they were independent bets, so breadth crushes daily vol (the same breadth
illusion that killed the short-scalp mirage). This script builds the defensible version: capital is
UNITISED (1/K per slot, idle capital earns 0), companies are DEDUPED, the satellite is vol-targeted
like the core, and the ensemble is sized the way an allocator would actually size a thin stress
sleeve (RP + 80/20 + 90/10). Every construction is pre-registered and counted as a trial for the DSR.

Inputs (all cached, no refit):
  * events: data/cache/reval_oos_sl1.0.parquet, filtered regime in {risk_off,crisis} & prob>=0.5,
    window 2018-01-01 -> 2026-06-26 (the desk's real-data window)
  * exit: the R6-banked asymmetric config — trades re-managed with the 2-ATR stop / 2-ATR target /
    10-day cap via `improve.resim_trades` (sl_mult = scan.STOP_ATR_MULT = 2.0), 3bp cost
  * core: the 'better long model' (vol-targeted RP across the two SPY swing books), built EXACTLY
    as in decode_ensemble_v2.py and CACHED to data/cache/ so it is constructed once

PRE-REGISTERED CONSTRUCTIONS (each = 1 trial; N_TRIALS = 12):
   T1  naive calendar_book, cap 10                (v2-style reference — the optimistic construction)
   T2  UNIT-CAPITAL book, K=10, linear P&L spread (1/K per slot, overflow SKIPPED, idle capital = 0)
   T3  UNIT-CAPITAL book, K=10, true daily MTM    (positions marked on actual closes — honest path vol)
   T4  T3 + COMPANY-DEDUP (GOOG/GOOGL = one company; re-entry while a position is open = skip)
   T5  VOL-TARGETED satellite: T4 scaled to 10% ann vol (63d trailing, lag 1d), leverage cap 2.0x,
       warm-up/unmeasurable trailing vol -> leverage 1.0 (unlevered)
   T6  T5 with the module-verbatim edge convention (warm-up lev = 0) — sensitivity
   T7  ENSEMBLE: risk-parity combine (long_model.vol_target_risk_parity, module defaults) of
       core + T5 satellite
   T8  ENSEMBLE: RP combine of core + T4 raw unit book (v2-comparable call shape) — sensitivity
   T9  ENSEMBLE: fixed 80/20 core/satellite(T5), daily-rebalanced constant mix
   T10 ENSEMBLE: fixed 90/10 core/satellite(T5)
   T11 T4 with K=5   (cap sensitivity)
   T12 T4 with K=20  (cap sensitivity)

Conventions (identical for every stream so comparisons are apples-to-apples):
  * all daily series live on the full SPY trading calendar 2018-01-01 -> 2026-06-26; days with no
    open position are 0.0 (uninvested capital earns nothing — that IS the honest satellite)
  * a position occupies its slot from entry day THROUGH exit day (frees the day after — same
    blocking convention as calendar_book); overflow entries are skipped permanently, not queued
  * stats: Sharpe, CAGR, vol, maxDD, Calmar, DSR (n_trials=12; headline also shown at the
    project-cumulative N~380), worst calendar month, 2020-excluded Sharpe, corr to core
  * win_rate is NOT computed anywhere (banned as a selection metric)

    python scripts/satellite_book.py [--rebuild-core]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.portfolio import run_portfolio
from startx.portfolio.long_model import vol_target_risk_parity
from startx.scanner.improve import calendar_book, resim_trades
from startx.scanner.scan import STOP_ATR_MULT
from startx.strategy.market_internals import load_internals
from startx.strategy.mean_reversion import ibs_signals
from startx.strategy.momentum_breakout import breakout_signals
from startx.strategy.volatility_premium import fear_signals
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
EVENTS = "data/cache/reval_oos_sl1.0.parquet"
CORE_CACHE = "data/cache/satellite_core_lm_2018_2026.parquet"
START, END = "2018-01-01", "2026-06-26"
K_SLOTS = 10
SAT_TARGET_VOL, SAT_LEV_CAP = 0.10, 2.0
COST = 3.0 / 1e4                     # resim_trades default cost_bps=3.0, applied once per trade
N_TRIALS = 12                        # this session's pre-registered constructions (see docstring)
N_CUMULATIVE = 380                   # ledger N~367 + these 12 (headline DSR robustness check)
TR = 252
COMPANY_MAP = {"GOOG": "GOOGL"}      # share classes of one company


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- core (built once, cached)
def build_core(rebuild: bool) -> pd.Series:
    if os.path.exists(CORE_CACHE) and not rebuild:
        core = pd.read_parquet(CORE_CACHE)["core"]
        core.index = pd.to_datetime(core.index)
        print(f"core long model loaded from cache ({CORE_CACHE}, {len(core)} days)")
        return core
    print("building the better long model (short IBS + long breakout/fear, vol-targeted RP)...")
    spy = _load("SPY")
    spy.attrs["symbol"] = "SPY"
    aux = {"vix": _load("_VIX"), "vvix": _load("_VVIX"), "gld": _load("GLD"),
           "internals": load_internals()}

    def _book_stream(sleeves, exits):
        eq = run_portfolio(spy, aux, sleeves, exits=exits, gross_cap=1.5, start=START, end=END).equity
        eq = eq[(eq.index >= pd.Timestamp(START)) & (eq.index <= pd.Timestamp(END))]
        return (eq / eq.iloc[0]).pct_change().fillna(0.0)

    short_r = _book_stream({"ibs": lambda s, a: ibs_signals(s)}, {"ibs": (1, 3, 10)})
    long_r = _book_stream({"breakout": lambda s, a: breakout_signals(s, 20, 200),
                           "fear": lambda s, a: fear_signals(s, a["vix"], a["vvix"])},
                          {"breakout": (1, 3, 63), "fear": (1, 3, 63)})
    core = vol_target_risk_parity({"short": short_r, "long": long_r}).combined.rename("core")
    core.to_frame().to_parquet(CORE_CACHE)
    print(f"core cached -> {CORE_CACHE} ({len(core)} days)")
    return core


# ---------------------------------------------------------------- satellite trades + daily paths
def load_trades():
    ev = pd.read_parquet(EVENTS)
    sub = ev[ev["regime"].isin(["risk_off", "crisis"]) & (ev["prob"] >= 0.5)
             & (ev["date"] >= pd.Timestamp(START)) & (ev["date"] <= pd.Timestamp(END))]
    sub = sub.sort_values(["entry_date", "date", "symbol"]).reset_index(drop=True)
    syms = sorted(sub["symbol"].unique())
    cached = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")}
    prices = {s: _load(s) for s in syms if s in cached}
    t2 = resim_trades(sub, prices, pt_mult=2.0, sl_mult=STOP_ATR_MULT, horizon=10)
    t2 = t2.sort_values(["entry_date", "date", "symbol"]).reset_index(drop=True)
    print(f"events: {len(sub)} stress-gated prob>=0.5 signals -> {len(t2)} resimmed trades "
          f"(2-ATR stop / 2-ATR target / 10d cap, 3bp), {t2['symbol'].nunique()} symbols, "
          f"{sub.duplicated(['symbol', 'date']).sum()} same-day duplicate (symbol,date) signals kept raw")
    return t2, prices


def build_paths(t2: pd.DataFrame, prices: dict) -> list[dict]:
    """Per-trade daily return paths. 'mtm' marks the position on actual closes (entry at next open,
    exit at the resimmed barrier/time price implied by net ret + cost); 'linear' spreads ret evenly
    over the trading days the position is open. Both compound to the same per-trade net return."""
    idx_cache = {}
    out = []
    for _, e in t2.iterrows():
        sym = e["symbol"]
        if sym not in idx_cache:
            p = prices[sym]
            idx_cache[sym] = (p, pd.DatetimeIndex(p["date"]))
        p, dates = idx_cache[sym]
        i = int(dates.get_indexer([pd.Timestamp(e["date"])])[0])
        jx = int(dates.get_indexer([pd.Timestamp(e["t1"])])[0])
        ie = i + 1
        if i < 0 or jx < ie:
            continue
        o = p["open"].to_numpy(float)
        c = p["close"].to_numpy(float)
        entry = o[ie]
        exit_px = entry * (1.0 + float(e["ret"]) + COST)        # ret is net of the 3bp cost
        days = dates[ie:jx + 1]
        if jx == ie:
            mtm = [exit_px / entry - 1.0 - COST]
        else:
            mtm = [c[ie] / entry - 1.0 - COST]
            mtm += [c[j] / c[j - 1] - 1.0 for j in range(ie + 1, jx)]
            mtm += [exit_px / c[jx - 1] - 1.0]
        lin = [float(e["ret"]) / len(days)] * len(days)
        out.append({"entry_date": days[0], "exit_date": days[-1],
                    "company": COMPANY_MAP.get(sym, sym), "symbol": sym,
                    "mtm": pd.Series(mtm, index=days), "lin": pd.Series(lin, index=days)})
    out.sort(key=lambda t: (t["entry_date"], t["symbol"]))
    return out


def unit_book(trades: list[dict], calendar: pd.DatetimeIndex, *, k: int, dedup: bool,
              path_key: str = "mtm"):
    """Unit-capital book: each accepted trade runs on exactly 1/k of capital; idle slots earn 0.
    Slots fill first-come (entry-date order); overflow and (optionally) same-company-while-open
    entries are SKIPPED permanently. Returns (daily_series_on_calendar, accounting dict)."""
    daily = pd.Series(0.0, index=calendar)
    open_pos: list[tuple] = []          # (exit_date, path)
    open_comp: dict = {}                # company -> exit_date
    n_take = n_over = n_dupe = 0
    ti = 0
    for d in calendar:
        open_pos = [x for x in open_pos if x[0] >= d]
        if dedup:
            open_comp = {cmp: x for cmp, x in open_comp.items() if x >= d}
        while ti < len(trades) and trades[ti]["entry_date"] <= d:
            tr = trades[ti]
            ti += 1
            if tr["entry_date"] < d:
                continue                                        # not on this calendar (should not happen)
            if dedup and tr["company"] in open_comp:
                n_dupe += 1
                continue
            if len(open_pos) >= k:
                n_over += 1
                continue
            open_pos.append((tr["exit_date"], tr[path_key]))
            n_take += 1
            if dedup:
                open_comp[tr["company"]] = tr["exit_date"]
        if open_pos:
            daily.loc[d] = sum(pth.get(d, 0.0) for _, pth in open_pos) / k
    n_elig = len(trades)
    acct = {"eligible": n_elig, "taken": n_take, "skip_overflow": n_over, "skip_dedup": n_dupe,
            "skip_rate": (n_over + n_dupe) / max(n_elig, 1)}
    return daily, acct


def vol_target_single(r: pd.Series, *, target=SAT_TARGET_VOL, cap=SAT_LEV_CAP, window=63,
                      warmup_lev=1.0):
    """Single-stream vol target, same formula/lag as long_model.vol_target_risk_parity:
    lev = clip(target / trailing-63d-ann-vol(lag 1), cap). Zero trailing vol -> inf -> cap (module
    behaviour); warm-up NaN -> ``warmup_lev`` (module uses 0; the primary here uses 1 = unlevered,
    so the first days of a stress cluster after a quiet stretch are not silently zeroed)."""
    cvol = r.rolling(window).std().shift(1) * np.sqrt(TR)
    lev = (target / cvol).clip(upper=cap)
    lev = lev.where(np.isfinite(lev), warmup_lev)
    return (r * lev).rename(r.name), lev


# ---------------------------------------------------------------- stats
def stats_row(label, r, core, *, n_trials=N_TRIALS):
    r = r.dropna()
    sd = r.std(ddof=1)
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(r.mean() / sd * np.sqrt(TR)) if sd > 0 else float("nan")
    monthly = (1 + r).resample("ME").prod() - 1
    ex20 = r[r.index.year != 2020]
    sh20 = float(ex20.mean() / ex20.std(ddof=1) * np.sqrt(TR)) if ex20.std(ddof=1) > 0 else float("nan")
    cc = pd.concat([r, core], axis=1).dropna()
    return {"label": label, "sharpe": sh, "cagr": cagr, "vol": float(sd * np.sqrt(TR)), "maxdd": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
            "dsr": float(deflated_sharpe(r.values, n_trials=n_trials)),
            "worst_mo": float(monthly.min()), "sharpe_ex2020": sh20,
            "corr_core": float(cc.iloc[:, 0].corr(cc.iloc[:, 1])) if len(cc) > 30 else float("nan")}


def print_table(rows, title):
    print(f"\n=== {title} ===")
    hdr = f"  {'stream':44}{'Sharpe':>7}{'CAGR':>8}{'vol':>7}{'maxDD':>8}{'Calmar':>8}" \
          f"{'DSR':>6}{'worstMo':>9}{'Shx20':>7}{'corr':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in rows:
        print(f"  {s['label']:44}{s['sharpe']:>7.2f}{s['cagr']*100:>7.1f}%{s['vol']*100:>6.1f}%"
              f"{s['maxdd']*100:>7.1f}%{s['calmar']:>8.2f}{s['dsr']:>6.2f}{s['worst_mo']*100:>8.1f}%"
              f"{s['sharpe_ex2020']:>7.2f}{s['corr_core']:>6.2f}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-core", action="store_true")
    args = ap.parse_args()

    core = build_core(args.rebuild_core)
    calendar = pd.DatetimeIndex(_load("SPY")["date"])
    calendar = calendar[(calendar >= pd.Timestamp(START)) & (calendar <= pd.Timestamp(END))]
    core = core.reindex(calendar).fillna(0.0)

    t2, prices = load_trades()
    trades = build_paths(t2, prices)
    print(f"paths built for {len(trades)} trades "
          f"(hold {np.mean([len(t['mtm']) for t in trades]):.1f} trading days avg)")

    Z = lambda s: s.reindex(calendar).fillna(0.0)

    # T1 naive v2-style calendar book (the optimistic construction, for reference)
    naive = Z(calendar_book(t2, t2["ret"].to_numpy(float), max_concurrent=K_SLOTS))

    # T2/T3 unit-capital; T4 + company dedup
    u_lin, a_lin = unit_book(trades, calendar, k=K_SLOTS, dedup=False, path_key="lin")
    u_mtm, a_mtm = unit_book(trades, calendar, k=K_SLOTS, dedup=False, path_key="mtm")
    u_ded, a_ded = unit_book(trades, calendar, k=K_SLOTS, dedup=True, path_key="mtm")
    for name, a in (("unit K=10 (T2/T3)", a_mtm), ("unit+dedup K=10 (T4)", a_ded)):
        print(f"  {name}: eligible {a['eligible']}, taken {a['taken']}, "
              f"skipped overflow {a['skip_overflow']} + dedup {a['skip_dedup']} "
              f"(skip rate {a['skip_rate']*100:.1f}%)")

    # T5/T6 vol-targeted satellite
    sat_vt, lev = vol_target_single(u_ded, warmup_lev=1.0)
    sat_vt_mod = Z(vol_target_risk_parity({"sat": u_ded}, target_vol=SAT_TARGET_VOL,
                                          lev_cap=SAT_LEV_CAP).combined)
    print(f"  vol-target: mean lev {lev.mean():.2f}, at-cap {(lev >= SAT_LEV_CAP).mean()*100:.0f}% of days")

    # T7-T10 ensembles
    rp_vt = Z(vol_target_risk_parity({"core": core, "sat": sat_vt}).combined)
    rp_raw = Z(vol_target_risk_parity({"core": core, "sat": u_ded}).combined)
    fix80 = 0.8 * core + 0.2 * sat_vt
    fix90 = 0.9 * core + 0.1 * sat_vt

    # T11/T12 cap sensitivity
    u_d5, a5 = unit_book(trades, calendar, k=5, dedup=True, path_key="mtm")
    u_d20, a20 = unit_book(trades, calendar, k=20, dedup=True, path_key="mtm")

    rows_sat = [
        stats_row("core: better long model (alone)", core, core),
        stats_row("T1 naive calendar_book cap10 (v2-style)", naive, core),
        stats_row("T2 unit-capital K=10, linear spread", u_lin, core),
        stats_row("T3 unit-capital K=10, true MTM", u_mtm, core),
        stats_row("T4 unit K=10 + company dedup (HONEST)", u_ded, core),
        stats_row("T5 satellite vol-target 10%/2x (HONEST)", sat_vt, core),
        stats_row("T6 satellite VT module-verbatim (sens.)", sat_vt_mod, core),
        stats_row("T11 unit+dedup K=5 (sens.)", u_d5, core),
        stats_row("T12 unit+dedup K=20 (sens.)", u_d20, core),
    ]
    rows_ens = [
        stats_row("core alone", core, core),
        stats_row("T7 RP(core, satellite T5)", rp_vt, core),
        stats_row("T8 RP(core, raw unit book T4) (sens.)", rp_raw, core),
        stats_row("T9 fixed 80/20 core/satellite", fix80, core),
        stats_row("T10 fixed 90/10 core/satellite", fix90, core),
    ]
    print_table(rows_sat, f"SATELLITE constructions {START}..{END} "
                          f"(full-calendar convention, idle capital = 0; DSR at N={N_TRIALS})")
    print_table(rows_ens, f"ENSEMBLES core + satellite (DSR at N={N_TRIALS})")

    # headline DSR at the project-cumulative trial count
    print(f"\n  headline DSR robustness at project-cumulative N~{N_CUMULATIVE}:")
    for lbl, r in (("core alone", core), ("T7 RP", rp_vt), ("T9 80/20", fix80), ("T10 90/10", fix90)):
        print(f"    {lbl:14} DSR@{N_CUMULATIVE} = "
              f"{deflated_sharpe(r.dropna().values, n_trials=N_CUMULATIVE):.2f}")

    core_sh = rows_ens[0]["sharpe"]
    hon = [r for r in rows_ens if r["label"].startswith(("T7", "T9", "T10"))]
    lo, hi = min(r["sharpe"] for r in hon), max(r["sharpe"] for r in hon)
    print(f"\n  THE DEFENSIBLE NUMBER: core alone Sharpe {core_sh:.2f} -> core+satellite (honest "
          f"constructions T7/T9/T10) {lo:.2f}..{hi:.2f}  (delta {lo-core_sh:+.2f}..{hi-core_sh:+.2f})")
    print(f"  vs the construction-optimistic v2-style satellite (T1 Sharpe "
          f"{rows_sat[1]['sharpe']:.2f}) that produced the 'ensemble 1.80'.")


if __name__ == "__main__":
    main()
