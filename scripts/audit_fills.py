"""Execution-quality AUDIT — gap-through-barrier fill realism for the banked swing config (read-only).

The banked R6 config (LEDGER R6): meta-model entries on a 1-ATR-stop label, trades managed with a
2-ATR stop / 2-ATR target / 10-day cap. Holdout being audited: 229 trades 2020-2026, +1.48%/trade,
PF 1.67, DSR 0.84 @ N=30 (events: data/cache/reval_oos_sl1.0.parquet, regime in {risk_off,crisis},
prob>=0.5, entry_date >= 2020-01-01).

The barrier walk in scanner/harness.py::build_dataset and scanner/improve.py::_resim/resim_blotter
fills at EXACTLY the barrier price whenever the day's high/low crosses it, and checks target BEFORE
stop within a day. Three fill-realism questions, each quantified here:

  1. GAP-THROUGH-STOP   — if day j OPENS below the stop, the real fill is the open (WORSE than the
                          stop). The banked walk books the stop price -> overstates returns.
  2. GAP-THROUGH-TARGET — if day j OPENS above the target, the real fill is the open (BETTER than
                          the target). The banked walk books the target price -> understates.
  3. SAME-DAY AMBIGUITY — when one daily bar touches BOTH barriers the walk assumes target-first
                          (optimistic). Bracketed both ways here and adjudicated with 1-min bars.

Plus the known artifact class: daily-feed extremes that were never traded in the regular session
(IBM 2025-04-08 entry: the 216.01 "stop" low is absent from RTH 1-min bars). The intraday stage
counts every daily-bar barrier touch that RTH 1-min data disagrees with.

Stages (all intermediates cached; every stage is resumable / re-runnable):
    python scripts/audit_fills.py --stage daily      # gap-aware re-walk on daily bars (no network)
    python scripts/audit_fills.py --stage intraday   # 1-min adjudication (FMP, chunked; --max-fetch)
    python scripts/audit_fills.py --stage report     # final tables + verdict inputs

AUDIT ONLY: writes nothing outside data/cache/audit_fills_*.parquet and the shared 1-min day cache
data/cache/intraday/ (same naming + NEVER-cache-empty convention as swing_blotter_intraday.py).
No production file is modified. win_rate is reported for context, never used for selection.
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

from startx.scanner.signals import _atr                    # same ATR the production resim uses
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
INTRADAY_DIR = "data/cache/intraday"
SOURCE = "data/cache/reval_oos_sl1.0.parquet"
DAILY_OUT = "data/cache/audit_fills_daily.parquet"
INTRA_OUT = "data/cache/audit_fills_intraday.parquet"
STRESS = ["risk_off", "crisis"]
SINCE = pd.Timestamp("2020-01-01")
COST_BPS = 3.0
HORIZON = 10
# The two exit configs of the R6 bank-it comparison (same 229 entries, only the exit differs):
CONFIGS = {"2atr_banked": dict(pt=2.0, sl=2.0), "1atr_old": dict(pt=2.0, sl=1.0)}
# Deflation bookkeeping for THIS audit: per config we look at 6 fill variants
# (base_tf, base_sf, gap_tf, gap_sf, intra_tf, intra_sf) x 2 configs = 12 trials.
N_TRIALS_AUDIT = 12


# --------------------------------------------------------------------------- shared plumbing
def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def load_trades():
    """The exact banked holdout book: stress + prob>=0.5, entry >= 2020-01-01 (dupes kept, as banked)."""
    oos = pd.read_parquet(SOURCE)
    t = oos[oos["regime"].isin(STRESS) & (oos["prob"] >= 0.5)].copy()
    t = t[pd.to_datetime(t["entry_date"]) >= SINCE]
    return t.reset_index(drop=True)


def price_arrays(symbols):
    """{sym: (o,h,l,c,dates,atr,posmap)} for every needed symbol."""
    out = {}
    for s in symbols:
        p = _load(s)
        if p is None:
            continue
        dates = pd.DatetimeIndex(p["date"])
        out[s] = (p["open"].to_numpy(float), p["high"].to_numpy(float), p["low"].to_numpy(float),
                  p["close"].to_numpy(float), dates, _atr(p).to_numpy(float),
                  {d: i for i, d in enumerate(dates)})
    return out


def locate(trade, px):
    """Reconstruct the exact production entry: signal bar i, entry = next open, ATR at signal bar."""
    arr = px.get(trade["symbol"])
    if arr is None:
        return None
    o, h, lo, c, dates, atr, pos = arr
    i = pos.get(pd.Timestamp(trade["date"]))
    if i is None or i + 1 >= len(c) or not np.isfinite(atr[i]) or atr[i] <= 0:
        return None
    ie = i + 1
    return dict(o=o, h=h, lo=lo, c=c, dates=dates, ie=ie, entry=o[ie], atrv=atr[i],
                last=min(ie + HORIZON, len(c) - 1))


# --------------------------------------------------------------------------- daily fill models
def walk_daily(L, pt_mult, sl_mult, *, gap_aware, stop_first):
    """One daily barrier walk. Returns dict(fill, j, reason, gap_slip).

    gap_aware=False, stop_first=False reproduces the production walk (harness.build_dataset /
    improve._resim / resim_blotter): barrier-price fills, target checked before stop each day.
    gap_aware=True: if day j (j > entry day; the entry day opens AT the entry) OPENS beyond a
    barrier, the fill is the OPEN — min(stop, open) on a gap-down, max(target, open) on a gap-up.
    stop_first flips the same-day both-touch tie-break (the entry-day/any-day ambiguity bracket).
    gap_slip = signed fill improvement vs the barrier-price fill on the exit day (fractions of entry).
    """
    o, h, lo, c = L["o"], L["h"], L["lo"], L["c"]
    ie, entry, last = L["ie"], L["entry"], L["last"]
    pt = entry + pt_mult * L["atrv"]
    sl = entry - sl_mult * L["atrv"]
    for j in range(ie, last + 1):
        if gap_aware and j > ie:
            if o[j] <= sl:                                  # overnight gap through the stop
                return dict(fill=o[j], j=j, reason="gap_stop", gap_slip=(o[j] - sl) / entry)
            if o[j] >= pt:                                  # overnight gap through the target
                return dict(fill=o[j], j=j, reason="gap_target", gap_slip=(o[j] - pt) / entry)
        hit_t, hit_s = h[j] >= pt, lo[j] <= sl
        if hit_t and hit_s:                                 # one bar straddles BOTH barriers
            return (dict(fill=sl, j=j, reason="stop_amb", gap_slip=0.0) if stop_first
                    else dict(fill=pt, j=j, reason="target_amb", gap_slip=0.0))
        if hit_t:
            return dict(fill=pt, j=j, reason="target", gap_slip=0.0)
        if hit_s:
            return dict(fill=sl, j=j, reason="stop", gap_slip=0.0)
    return dict(fill=c[last], j=last, reason="time", gap_slip=0.0)


def stage_daily():
    trades = load_trades()
    px = price_arrays(trades["symbol"].unique())
    cost = COST_BPS / 1e4
    variants = {"base_tf": dict(gap_aware=False, stop_first=False),
                "base_sf": dict(gap_aware=False, stop_first=True),
                "gap_tf": dict(gap_aware=True, stop_first=False),
                "gap_sf": dict(gap_aware=True, stop_first=True)}
    rows = []
    for _, tr in trades.iterrows():
        L = locate(tr, px)
        if L is None:
            continue
        rec = {"symbol": tr["symbol"], "date": pd.Timestamp(tr["date"]),
               "entry_date": L["dates"][L["ie"]], "trigger": tr["trigger"], "regime": tr["regime"],
               "prob": tr["prob"], "ret_cached": tr["ret"], "entry": L["entry"], "atr": L["atrv"]}
        for cfg, m in CONFIGS.items():
            for vn, kw in variants.items():
                w = walk_daily(L, m["pt"], m["sl"], **kw)
                rec[f"{cfg}_{vn}_ret"] = (w["fill"] / L["entry"] - 1.0) - cost
                rec[f"{cfg}_{vn}_reason"] = w["reason"]
                rec[f"{cfg}_{vn}_j"] = w["j"] - L["ie"]
                rec[f"{cfg}_{vn}_slip"] = w["gap_slip"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(DAILY_OUT)

    # reproduction anchors — the audit is void if the as-banked walk doesn't match the banked numbers
    print(f"daily stage: {len(df)} trades walked -> {DAILY_OUT}")
    print(f"  REPRO 1-ATR base_tf vs cached ret: max|diff| = "
          f"{(df['1atr_old_base_tf_ret'] - df['ret_cached']).abs().max():.2e}")
    print(f"  REPRO 2-ATR base_tf mean = {df['2atr_banked_base_tf_ret'].mean()*100:+.3f}%  "
          f"(banked headline +1.48%)")
    return df


# --------------------------------------------------------------------------- intraday adjudication
def _intraday_symbols(sym):
    out = [sym]
    if "-" in sym:
        out.append(sym.replace("-", "."))                   # BRK-B (EOD cache) -> BRK.B (FMP intraday)
    return out


class MinuteFeed:
    """Disk+memory cached 1-min day fetcher (same cache files & never-cache-empty rule as the
    production intraday blotter). Counts network fetches so runs can be chunked under a budget."""

    def __init__(self, client, max_fetch):
        self.client, self.max_fetch, self.fetches = client, max_fetch, 0
        self.mem = {}

    def day(self, sym, day):
        key = (sym, str(day))
        if key in self.mem:
            return self.mem[key]
        cache = f"{INTRADAY_DIR}/{sym}_{day}.parquet"
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            if not df.empty:
                self.mem[key] = df
                return df
        if self.fetches >= self.max_fetch:
            raise BudgetExceeded()
        from startx.fmp.endpoints import historical_chart
        for s in _intraday_symbols(sym):
            self.fetches += 1
            df = historical_chart(self.client, s, "1min", start=str(day), end=str(day))
            if not df.empty:
                os.makedirs(INTRADAY_DIR, exist_ok=True)
                df.to_parquet(cache)                        # only non-empty days are frozen to disk
                self.mem[key] = df
                return df
        self.mem[key] = pd.DataFrame()                      # empty: remember for THIS run only
        return self.mem[key]


class BudgetExceeded(Exception):
    pass


def walk_intraday(L, pt_mult, sl_mult, feed, sym, *, stop_first):
    """Day-by-day walk adjudicated with RTH 1-min bars. Fills: session-open when the open gaps
    through a barrier, else the barrier price at its first true minute touch; minute-level both-touch
    ties break per stop_first. Days with no 1-min data fall back to the daily gap-aware walk.
    Also counts daily-vs-intraday touch disagreements (the IBM-class artifact) along the way."""
    o, h, lo, c, dates = L["o"], L["h"], L["lo"], L["c"], L["dates"]
    ie, entry, last = L["ie"], L["entry"], L["last"]
    pt = entry + pt_mult * L["atrv"]
    sl = entry - sl_mult * L["atrv"]
    diag = dict(fallback_days=0, eod_only_days=0, intra_only_days=0, minute_tie=0, open_diff=np.nan)
    for j in range(ie, last + 1):
        d_hit_t, d_hit_s = h[j] >= pt, lo[j] <= sl
        bars = feed.day(sym, dates[j].date())
        if bars.empty:                                      # no RTH 1-min -> daily gap-aware fallback
            diag["fallback_days"] += 1
            if j > ie:
                if o[j] <= sl:
                    return dict(fill=o[j], j=j, reason="gap_stop", **diag)
                if o[j] >= pt:
                    return dict(fill=o[j], j=j, reason="gap_target", **diag)
            if d_hit_t and d_hit_s:
                return (dict(fill=sl, j=j, reason="stop_amb", **diag) if stop_first
                        else dict(fill=pt, j=j, reason="target_amb", **diag))
            if d_hit_t:
                return dict(fill=pt, j=j, reason="target", **diag)
            if d_hit_s:
                return dict(fill=sl, j=j, reason="stop", **diag)
            continue
        hi_m = bars["high"].to_numpy(float)
        lo_m = bars["low"].to_numpy(float)
        so = float(bars.iloc[0]["open"])                    # session open
        if j == ie:
            diag["open_diff"] = abs(so - entry) / entry     # daily-feed open vs 1-min 09:30 open
        i_hit_t, i_hit_s = bool((hi_m >= pt).any()), bool((lo_m <= sl).any())
        if (d_hit_t and not i_hit_t) or (d_hit_s and not i_hit_s):
            diag["eod_only_days"] += 1                      # daily feed touched, RTH 1-min never did
        if (i_hit_t and not d_hit_t) or (i_hit_s and not d_hit_s):
            diag["intra_only_days"] += 1
        if j > ie:
            if so <= sl:
                return dict(fill=so, j=j, reason="gap_stop", **diag)
            if so >= pt:
                return dict(fill=so, j=j, reason="gap_target", **diag)
        it = int(np.argmax(hi_m >= pt)) if i_hit_t else 10 ** 9
        isx = int(np.argmax(lo_m <= sl)) if i_hit_s else 10 ** 9
        if i_hit_t or i_hit_s:
            if it == isx:                                   # both inside ONE minute bar
                diag["minute_tie"] = 1
                return (dict(fill=sl, j=j, reason="stop_amb", **diag) if stop_first
                        else dict(fill=pt, j=j, reason="target_amb", **diag))
            if it < isx:
                return dict(fill=pt, j=j, reason="target", **diag)
            return dict(fill=sl, j=j, reason="stop", **diag)
    return dict(fill=c[last], j=last, reason="time", **diag)


def stage_intraday(max_fetch):
    from startx.fmp.client import FMPClient
    trades = load_trades()
    px = price_arrays(trades["symbol"].unique())
    cost = COST_BPS / 1e4
    done = pd.read_parquet(INTRA_OUT) if os.path.exists(INTRA_OUT) else pd.DataFrame()
    done_keys = set(zip(done["symbol"], done["date"].astype(str), done["trigger"])) if len(done) else set()
    rows, skipped = [], 0
    with FMPClient() as client:
        feed = MinuteFeed(client, max_fetch)
        try:
            for _, tr in trades.iterrows():
                key = (tr["symbol"], str(pd.Timestamp(tr["date"])), tr["trigger"])
                if key in done_keys:
                    skipped += 1
                    continue
                L = locate(tr, px)
                if L is None:
                    continue
                rec = {"symbol": tr["symbol"], "date": pd.Timestamp(tr["date"]),
                       "entry_date": L["dates"][L["ie"]], "trigger": tr["trigger"],
                       "regime": tr["regime"]}
                for cfg, m in CONFIGS.items():
                    for vn, sf in (("intra_tf", False), ("intra_sf", True)):
                        w = walk_intraday(L, m["pt"], m["sl"], feed, tr["symbol"], stop_first=sf)
                        rec[f"{cfg}_{vn}_ret"] = (w["fill"] / L["entry"] - 1.0) - cost
                        rec[f"{cfg}_{vn}_reason"] = w["reason"]
                        rec[f"{cfg}_{vn}_j"] = w["j"] - L["ie"]
                        if vn == "intra_sf":                # diagnostics identical across tie-breaks
                            for k in ("fallback_days", "eod_only_days", "intra_only_days",
                                      "minute_tie", "open_diff"):
                                rec[f"{cfg}_{k}"] = w[k]
                rows.append(rec)
        except BudgetExceeded:
            print(f"  fetch budget ({max_fetch}) reached — run the stage again to resume.")
    new = pd.DataFrame(rows)
    out = pd.concat([done, new], ignore_index=True) if len(done) else new
    if len(out):
        out.to_parquet(INTRA_OUT)
    print(f"intraday stage: +{len(new)} trades adjudicated this run ({skipped} already done, "
          f"{feed.fetches} fetches) -> {INTRA_OUT} [{len(out)}/{len(trades)} total]")
    return len(out) >= len(trades)


# --------------------------------------------------------------------------- reporting
def _stats(r, base=None):
    r = pd.Series(r).dropna()
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std(ddof=1)
    out = {"n": len(r), "exp%": r.mean() * 100,
           "PF": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
           "win%": (r > 0).mean() * 100,
           "Sharpe": r.mean() / sd * np.sqrt(25) if sd > 0 else np.nan,
           f"DSR@{N_TRIALS_AUDIT}": deflated_sharpe(r.values, n_trials=N_TRIALS_AUDIT),
           "DSR@30": deflated_sharpe(r.values, n_trials=30)}
    if base is not None:
        b = pd.Series(base).dropna()
        out["n_moved"] = int((~np.isclose(r.values, b.values, atol=1e-12)).sum())
        out["d_exp%"] = (r.mean() - b.mean()) * 100
    return out


def _print_table(title, rows):
    print(f"\n{title}")
    cols = ["variant", "n", "exp%", "PF", "win%", "Sharpe", f"DSR@{N_TRIALS_AUDIT}", "DSR@30",
            "n_moved", "d_exp%"]
    print("  " + "".join(f"{c:>10}" for c in cols))
    for name, st in rows:
        vals = [name] + [st.get(c, "") for c in cols[1:]]
        print("  " + "".join(f"{(f'{v:.2f}' if isinstance(v, float) else str(v)):>10}" for v in vals))


def stage_report():
    d = pd.read_parquet(DAILY_OUT)
    intra = pd.read_parquet(INTRA_OUT) if os.path.exists(INTRA_OUT) else pd.DataFrame()
    if len(intra):
        keys = ["symbol", "date", "trigger"]
        icols = [c for c in intra.columns if c not in keys and c not in ("entry_date", "regime")]
        d = d.merge(intra[keys + icols], on=keys, how="left")
    for cfg in CONFIGS:
        base = d[f"{cfg}_base_tf_ret"]
        rows = [("base_tf", _stats(base))]
        for vn in ("base_sf", "gap_tf", "gap_sf", "intra_tf", "intra_sf"):
            col = f"{cfg}_{vn}_ret"
            if col in d and d[col].notna().any():
                rows.append((vn, _stats(d[col], base)))
        _print_table(f"=== {cfg}  (base_tf = as-banked walk; deltas vs base_tf) ===", rows)
        # gap + ambiguity anatomy
        for vn in ("gap_sf",):
            rs = d[f"{cfg}_{vn}_reason"]
            gs = d.loc[rs == "gap_stop", f"{cfg}_{vn}_slip"]
            gt = d.loc[rs == "gap_target", f"{cfg}_{vn}_slip"]
            amb = d[f"{cfg}_base_tf_reason"].eq("target_amb") | d[f"{cfg}_base_sf_reason"].eq("stop_amb")
            print(f"  gap-through-stop: {len(gs)} trades, mean slip {gs.mean()*100 if len(gs) else 0:+.2f}% "
                  f"(worst {gs.min()*100 if len(gs) else 0:+.2f}%) | gap-through-target: {len(gt)} trades, "
                  f"mean slip {gt.mean()*100 if len(gt) else 0:+.2f}%")
            print(f"  same-day both-barrier (daily-ambiguous) trades: {int(amb.sum())}; "
                  f"tie-break moves exp by {(d[f'{cfg}_base_tf_ret'].mean()-d[f'{cfg}_base_sf_ret'].mean())*100:+.3f}% "
                  f"(tf vs sf, barrier-price fills)")
        if f"{cfg}_eod_only_days" in d and d[f"{cfg}_eod_only_days"].notna().any():
            eo = d[f"{cfg}_eod_only_days"].fillna(0)
            io = d[f"{cfg}_intra_only_days"].fillna(0)
            fb = d[f"{cfg}_fallback_days"].fillna(0)
            mt = d[f"{cfg}_minute_tie"].fillna(0)
            dis = ((d[f"{cfg}_intra_sf_ret"] - d[f"{cfg}_base_tf_ret"]).abs() > 1e-12)
            print(f"  intraday adjudication: {int((eo > 0).sum())} trades with EOD-only touches "
                  f"({int(eo.sum())} days), {int((io > 0).sum())} with intraday-only touches, "
                  f"{int((fb > 0).sum())} used daily fallback days, {int(mt.sum())} minute-ties; "
                  f"{int(dis.sum())} trades changed vs banked fills")
            od = d[f"{cfg}_open_diff"].dropna()
            if len(od):
                print(f"  daily-open vs 1-min-09:30-open: median {od.median()*1e4:.1f}bp, "
                      f"p95 {od.quantile(.95)*1e4:.1f}bp (entry-price feed check)")

    # the R6 comparison, re-checked like-for-like under realistic fills
    print("\n=== R6 RE-CHECK: 2-ATR minus 1-ATR per-trade expectancy, by fill model ===")
    for vn in ("base_tf", "base_sf", "gap_tf", "gap_sf", "intra_tf", "intra_sf"):
        a, b = f"2atr_banked_{vn}_ret", f"1atr_old_{vn}_ret"
        if a in d and d[a].notna().any():
            m = d[[a, b]].dropna()
            print(f"  {vn:>9}: 2-ATR {m[a].mean()*100:+.3f}%  1-ATR {m[b].mean()*100:+.3f}%  "
                  f"delta {(m[a].mean()-m[b].mean())*100:+.3f}%  (n={len(m)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["daily", "intraday", "report", "all"], default="all")
    ap.add_argument("--max-fetch", type=int, default=800, help="network budget per intraday run")
    args = ap.parse_args()
    if args.stage in ("daily", "all"):
        stage_daily()
    if args.stage in ("intraday", "all"):
        complete = stage_intraday(args.max_fetch)
        if args.stage == "all" and not complete:
            print("intraday incomplete — re-run '--stage intraday' until complete, then '--stage report'.")
            return
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
