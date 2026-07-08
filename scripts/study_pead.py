"""POST-EARNINGS-ANNOUNCEMENT DRIFT (PEAD) study — survivorship-free, firewall-honest.

PRE-REGISTERED HYPOTHESIS
    After an earnings report with a large surprise AND a large day-0 abnormal price
    reaction, price continues to DRIFT in the direction of the reaction for ~5-60
    trading days (institutional underreaction / anchoring). The other side of the
    trade is slow-rebalancing funds.

KILL CONDITION (pre-stated, before any data was touched)
    If the drift in the reaction's direction is <= 0 OOS net of costs on liquid
    names over 2018-2026, kill it.

OPERATING-STANDARD COMPLIANCE
    * Real-data window 2018-01-01 -> 2026-06-26 ONLY. Events from 2017 are used
      SOLELY as burn-in for the point-in-time quintile breakpoints (2016+ is within
      the membership module's REAL_DATA_START); no trade is reported pre-2018.
    * Universe: startx.data.membership.SP500Membership (point-in-time, incl.
      delisted). Per-event PIT gates: member as of the report date AND trailing
      63d median dollar volume >= $50M as of the reaction day's close.
    * Entries: next-bar OPEN after the reaction day (no signal-close fills).
      FMP earnings PIT convention reused from startx.fmp.endpoints.earnings —
      unknown session time == after-close (ts = 16:00:01), so the tradeable
      reaction day is t+1.
    * Costs: 3bp per side (6bp round trip); shorts additionally pay 50bp/yr
      borrow pro-rated over the hold.
    * win_rate is NOT a selection metric. Selection on DEV only; HOLDOUT touched
      once. DSR at the honest total trial count (N_TRIALS below).

TRIAL LEDGER (fixed BEFORE looking at any results; every cell below is a trial)
    rules   : L  = LONG  top-quintile positive abnormal reaction
              S  = SHORT bottom-quintile negative abnormal reaction
              Lc = L + surprise confirmation (EPS surprise > 0)
              Sc = S + surprise confirmation (EPS surprise < 0)
    horizons: 5, 10, 21, 42 trading days
    N_TRIALS = 4 rules x 4 horizons = 16

METHOD
    Event = earnings report with actual EPS. Reaction day r = first trading day
    on which the news is tradeable (per the PIT ts). Reaction = day-r close/close
    return minus SPY's. Quintile breakpoints are EXPANDING over all events with
    reaction day <= r (min 200 events; burn-in from 2017) — strictly point-in-time.
    Entry = OPEN of r+1. Exit = CLOSE of the H-th holding day. A name delisted
    mid-hold exits at its last available close (no survivorship pruning); events
    truncated by the end of the price sample are dropped for that horizon.

STAGES (resumable; intermediates cached under data/cache/earnings/)
    python scripts/study_pead.py --stage universe   # rebuild _universe.parquet
    python scripts/study_pead.py --stage fetch      # fetch earnings per symbol (skips cached)
    python scripts/study_pead.py --stage events     # build _events.parquet (reactions, fwd returns)
    python scripts/study_pead.py --stage dev        # DEV tables (2018-2021) — selection happens here
    python scripts/study_pead.py --stage holdout    # HOLDOUT tables (2022-2026) — run ONCE
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from startx.data.membership import SP500Membership  # noqa: E402
from startx.validation.metrics import deflated_sharpe  # noqa: E402

PRICE_DIR = "data/cache/prices"
EARN_DIR = "data/cache/earnings"
UNIVERSE_F = f"{EARN_DIR}/_universe.parquet"
EVENTS_F = f"{EARN_DIR}/_events.parquet"

BURN_START = pd.Timestamp("2017-01-01")   # breakpoint burn-in only, never reported
REPORT_START = pd.Timestamp("2018-01-01")  # locked real-data window
REPORT_END = pd.Timestamp("2026-06-26")
DEV_END = pd.Timestamp("2021-12-31")       # DEV = 2018-2021, HOLDOUT = 2022-2026

MIN_DOLLAR_VOL = 50e6      # PIT trailing 63d median dollar volume gate
COST_PER_SIDE = 0.0003     # 3bp per side
BORROW_ANNUAL = 0.005      # 50bp/yr borrow on shorts, pro-rated over hold
HORIZONS = (5, 10, 21, 42)
RULES = ("L", "S", "Lc", "Sc")
N_TRIALS = len(RULES) * len(HORIZONS)  # 16 — the honest deflation count
MIN_BURNIN_EVENTS = 200


# --------------------------------------------------------------------------- universe
def build_universe() -> pd.DataFrame:
    """Liquid, survivorship-free fetch superset: ever a PIT member 2017-2026, has prices,
    full-window median dollar volume >= $50M. (The per-event gate is PIT-trailing; this
    list only decides which symbols are worth an earnings fetch.)"""
    m = SP500Membership.load()
    avail = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRICE_DIR}/*.parquet")
             if not os.path.basename(f).startswith("_")}
    snap = pd.date_range(BURN_START, REPORT_END, freq="QS").tolist() + [REPORT_END]
    union: set[str] = set()
    for d in snap:
        union |= set(m.members_asof(d))
    rows = []
    for s in sorted(union & avail):
        d = pd.read_parquet(f"{PRICE_DIR}/{s}.parquet", columns=["date", "close", "volume"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= BURN_START]
        if len(d) < 60:
            continue
        dv = float((d["close"] * d["volume"]).median())
        rows.append({"symbol": s, "med_dv": dv})
    u = pd.DataFrame(rows)
    u = u[u["med_dv"] >= MIN_DOLLAR_VOL].sort_values("med_dv", ascending=False).reset_index(drop=True)
    os.makedirs(EARN_DIR, exist_ok=True)
    u.to_parquet(UNIVERSE_F, index=False)
    print(f"universe: {len(u)} liquid PIT names -> {UNIVERSE_F}")
    return u


# --------------------------------------------------------------------------- fetch
def fetch_earnings() -> None:
    """One earnings call per symbol (limit=80 ~ 20y), cached per symbol; resumable."""
    from startx.fmp.client import FMPClient
    from startx.fmp.endpoints import earnings
    from startx.settings import get_settings

    uni = pd.read_parquet(UNIVERSE_F)["symbol"].tolist()
    todo = [s for s in uni if not os.path.exists(f"{EARN_DIR}/{s}.parquet")]
    print(f"fetch: {len(uni)} in universe, {len(todo)} to fetch")
    if not todo:
        return
    s = get_settings()
    s.require_key()
    n_ok = n_empty = 0
    with FMPClient(s) as c:
        for i, sym in enumerate(todo, 1):
            df = earnings(c, sym, limit=80)
            if df.empty:
                # cache the emptiness too so reruns don't refetch; audit later
                df = pd.DataFrame({"symbol": pd.Series(dtype=str)})
                n_empty += 1
            else:
                n_ok += 1
            df.to_parquet(f"{EARN_DIR}/{sym}.parquet", index=False)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} fetched (ok={n_ok} empty={n_empty})")
    print(f"fetch done: ok={n_ok} empty={n_empty}")


# --------------------------------------------------------------------------- events
def _load_price(sym: str) -> pd.DataFrame | None:
    f = f"{PRICE_DIR}/{sym}.parquet"
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=["date", "open", "close", "volume"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def build_events() -> pd.DataFrame:
    """Per-event panel: PIT gates, abnormal reaction, EPS surprise, forward returns per horizon."""
    m = SP500Membership.load()
    uni = pd.read_parquet(UNIVERSE_F)["symbol"].tolist()

    spy = _load_price("SPY")
    spy_dates = spy["date"]
    spy_close = spy["close"].to_numpy(float)
    spy_open = spy["open"].to_numpy(float)
    spy_pos = {d: i for i, d in enumerate(spy_dates)}
    cal = spy_dates.to_numpy()                      # the trading calendar
    global_end = spy_dates.iloc[-1]

    n_no_feed = 0
    rows = []
    for k, sym in enumerate(uni, 1):
        ef = f"{EARN_DIR}/{sym}.parquet"
        if not os.path.exists(ef):
            n_no_feed += 1
            continue
        e = pd.read_parquet(ef)
        if e.empty or "date" not in e.columns:
            n_no_feed += 1
            continue
        e = e.dropna(subset=["epsActual"]).copy()   # realized reports only
        e["date"] = pd.to_datetime(e["date"], errors="coerce")
        e["ts"] = pd.to_datetime(e["ts"], errors="coerce")
        e = e.dropna(subset=["date", "ts"]).drop_duplicates(subset=["date"])
        e = e[(e["date"] >= BURN_START) & (e["date"] <= REPORT_END)]
        if e.empty:
            continue
        px = _load_price(sym)
        if px is None or len(px) < 80:
            continue
        pdates = px["date"].to_numpy()
        popen = px["open"].to_numpy(float)
        pclose = px["close"].to_numpy(float)
        pdv = px["close"].to_numpy(float) * px["volume"].to_numpy(float)
        sym_last = px["date"].iloc[-1]

        for _, ev in e.iterrows():
            ts = ev["ts"]
            # reaction day r = first trading day whose session STARTS after ts
            # (ts 16:00:01 on day t -> r is the next trading day; ts 00:00:00 -> r = t)
            if ts.time() > pd.Timestamp("1970-01-01 09:30").time():
                cut = ts.normalize() + pd.Timedelta(days=1)
            else:
                cut = ts.normalize()
            ci = np.searchsorted(cal, np.datetime64(cut))
            if ci >= len(cal):
                continue
            r_date = pd.Timestamp(cal[ci])
            # index of r in the SYMBOL's own calendar (must exist and have a prior bar)
            ri = int(np.searchsorted(pdates, np.datetime64(r_date)))
            if ri >= len(pdates) or pd.Timestamp(pdates[ri]) != r_date or ri < 1:
                continue
            si = spy_pos.get(r_date)
            if si is None or si < 1:
                continue
            # PIT gates: membership at the report date; trailing 63d median $vol through r
            if sym not in m.members_asof(ev["date"]):
                continue
            dv63 = np.nanmedian(pdv[max(0, ri - 62): ri + 1])
            if not np.isfinite(dv63) or dv63 < MIN_DOLLAR_VOL:
                continue
            # day-r abnormal (vs SPY) close-to-close reaction
            react = pclose[ri] / pclose[ri - 1] - 1.0
            react_spy = spy_close[si] / spy_close[si - 1] - 1.0
            abn = react - react_spy
            if not np.isfinite(abn):
                continue
            est = ev.get("epsEstimated")
            act = ev.get("epsActual")
            sue = float("nan")
            if pd.notna(est) and pd.notna(act):
                sue = (float(act) - float(est)) / max(abs(float(est)), 0.10)
            ei = ri + 1                              # entry bar = next OPEN after day r
            if ei >= len(pdates):
                continue
            entry_date = pd.Timestamp(pdates[ei])
            entry_px = popen[ei]
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
            sei = spy_pos.get(entry_date)
            row = {"symbol": sym, "report_date": ev["date"], "ts": ts, "r_date": r_date,
                   "entry_date": entry_date, "abn_react": abn, "raw_react": react, "sue": sue,
                   "dv63": dv63, "entry_open": entry_px}
            for H in HORIZONS:
                xi = ei + H - 1                       # exit bar = close of H-th holding day
                delisted = False
                if xi >= len(pdates):
                    # missing tail: end-of-sample truncation (drop) vs delisting (real exit)
                    if sym_last >= global_end - pd.Timedelta(days=7):
                        row[f"gross_{H}"] = np.nan
                        row[f"spy_{H}"] = np.nan
                        row[f"delist_{H}"] = False
                        continue
                    xi = len(pdates) - 1
                    delisted = True
                exit_date = pd.Timestamp(pdates[xi])
                sxi = spy_pos.get(exit_date)
                gross = pclose[xi] / entry_px - 1.0
                spy_win = np.nan
                if sei is not None and sxi is not None and sxi >= sei:
                    spy_win = spy_close[sxi] / spy_open[sei] - 1.0
                row[f"gross_{H}"] = gross
                row[f"spy_{H}"] = spy_win
                row[f"delist_{H}"] = delisted
            rows.append(row)
        if k % 100 == 0:
            print(f"  events: {k}/{len(uni)} symbols")

    ev = pd.DataFrame(rows).sort_values(["r_date", "symbol"]).reset_index(drop=True)

    # PIT expanding quintile breakpoints over all events with reaction day <= r
    # (everything in the breakpoint is known before the r+1 open entry)
    per_day = ev.groupby("r_date")["abn_react"].apply(list).sort_index()
    q80s, q20s, hist = {}, {}, []
    for d, vals in per_day.items():
        hist.extend(vals)
        if len(hist) >= MIN_BURNIN_EVENTS:
            arr = np.asarray(hist)
            q80s[d] = float(np.quantile(arr, 0.80))
            q20s[d] = float(np.quantile(arr, 0.20))
    ev["q80"] = ev["r_date"].map(q80s)
    ev["q20"] = ev["r_date"].map(q20s)
    ev["top_q"] = (ev["abn_react"] >= ev["q80"]) & ev["q80"].notna() & (ev["abn_react"] > 0)
    ev["bot_q"] = (ev["abn_react"] <= ev["q20"]) & ev["q20"].notna() & (ev["abn_react"] < 0)

    ev.to_parquet(EVENTS_F, index=False)
    n_rep = int((ev["entry_date"] >= REPORT_START).sum())
    print(f"events: {len(ev)} total ({n_rep} in report window) from "
          f"{ev['symbol'].nunique()} symbols; {n_no_feed} universe names had no earnings feed")
    print(f"  q80/q20 (last): {ev['q80'].iloc[-1]:+.4f} / {ev['q20'].iloc[-1]:+.4f}")
    return ev


# --------------------------------------------------------------------------- analysis
def _trade_returns(ev: pd.DataFrame, rule: str, H: int) -> pd.DataFrame:
    """Net per-trade returns for one (rule, horizon) trial. Positive = made money."""
    if rule == "L":
        sel = ev["top_q"]
    elif rule == "Lc":
        sel = ev["top_q"] & (ev["sue"] > 0)
    elif rule == "S":
        sel = ev["bot_q"]
    else:  # Sc
        sel = ev["bot_q"] & (ev["sue"] < 0)
    t = ev.loc[sel & ev[f"gross_{H}"].notna(),
               ["symbol", "entry_date", f"gross_{H}", f"spy_{H}", f"delist_{H}"]].copy()
    t.columns = ["symbol", "entry_date", "gross", "spy", "delisted"]
    rt = 2.0 * COST_PER_SIDE
    if rule.startswith("S"):
        t["net"] = -t["gross"] - rt - BORROW_ANNUAL * H / 252.0
    else:
        t["net"] = t["gross"] - rt
    return t


def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """OLS y = a + b*x; returns (alpha, beta, t_alpha)."""
    msk = np.isfinite(y) & np.isfinite(x)
    y, x = y[msk], x[msk]
    n = len(y)
    if n < 10:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[0]), float(coef[1]), float(coef[0] / np.sqrt(cov[0, 0]))


def _metrics(t: pd.DataFrame) -> dict:
    r = t["net"].to_numpy(float)
    n = len(r)
    if n < 5:
        return {"n": n}
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    pf = float(r[r > 0].sum() / max(-r[r < 0].sum(), 1e-12))
    sr = mean / sd if sd > 0 else float("nan")
    dsr = float(deflated_sharpe(r, n_trials=N_TRIALS))
    alpha, beta, t_a = _ols_alpha_beta(r, t["spy"].to_numpy(float))
    return {"n": n, "exp_bp": mean * 1e4, "med_bp": float(np.median(r)) * 1e4, "pf": pf,
            "sr_trade": sr, "t_stat": sr * np.sqrt(n), "dsr": dsr,
            "alpha_bp": alpha * 1e4, "t_alpha": t_a, "beta": beta,
            "n_delist": int(t["delisted"].sum())}


def _split(ev: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "dev":
        return ev[(ev["entry_date"] >= REPORT_START) & (ev["entry_date"] <= DEV_END)]
    return ev[(ev["entry_date"] > DEV_END) & (ev["entry_date"] <= REPORT_END)]


def _sleeve_monthly(ev: pd.DataFrame, rule: str, H: int, which: str) -> pd.DataFrame:
    """Daily equal-weight sleeve of all open positions -> monthly return stream."""
    spy = _load_price("SPY")
    spy = spy.set_index("date")
    t = _trade_returns(_split(ev, which), rule, H)
    px_cache: dict[str, pd.DataFrame] = {}
    daily: dict[pd.Timestamp, list[float]] = {}
    sign = -1.0 if rule.startswith("S") else 1.0
    for _, tr in t.iterrows():
        sym = tr["symbol"]
        if sym not in px_cache:
            px_cache[sym] = _load_price(sym).set_index("date")
        p = px_cache[sym]
        ei = p.index.searchsorted(tr["entry_date"])
        xi = min(ei + H - 1, len(p) - 1)
        entry_open = float(p["open"].iloc[ei])
        closes = p["close"].iloc[ei: xi + 1].astype(float)
        prev = entry_open
        for j, (d, c) in enumerate(closes.items()):
            rr = sign * (c / prev - 1.0)
            if j == 0:
                rr -= COST_PER_SIDE
            if j == len(closes) - 1:
                rr -= COST_PER_SIDE
            if rule.startswith("S"):
                rr -= BORROW_ANNUAL / 252.0
            daily.setdefault(pd.Timestamp(d), []).append(rr)
            prev = c
    if not daily:
        return pd.DataFrame()
    sleeve = pd.Series({d: float(np.mean(v)) for d, v in sorted(daily.items())}, name="sleeve")
    spy_ret = spy["close"].pct_change().rename("spy")
    both = pd.concat([sleeve, spy_ret], axis=1).dropna()
    monthly = (1.0 + both).groupby(pd.Grouper(freq="ME")).prod() - 1.0
    monthly = monthly[monthly.index.isin((1.0 + both["sleeve"]).groupby(pd.Grouper(freq="ME")).count().index)]
    return monthly


def analyze(which: str, sleeve_rule: str | None = None, sleeve_h: int | None = None) -> None:
    ev = pd.read_parquet(EVENTS_F)
    sub = _split(ev, which)
    label = {"dev": "DEV 2018-2021", "holdout": "HOLDOUT 2022-2026"}[which]
    print(f"\n================ {label} | events in split: {len(sub)} "
          f"(top_q={int(sub['top_q'].sum())}, bot_q={int(sub['bot_q'].sum())}) ================")
    print(f"n_trials for DSR deflation: {N_TRIALS}")
    rows = []
    for rule in RULES:
        for H in HORIZONS:
            t = _trade_returns(sub, rule, H)
            mt = _metrics(t)
            mt.update({"rule": rule, "H": H})
            rows.append(mt)
    tab = pd.DataFrame(rows).set_index(["rule", "H"])
    cols = ["n", "exp_bp", "med_bp", "pf", "sr_trade", "t_stat", "dsr", "alpha_bp", "t_alpha", "beta", "n_delist"]
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:8.3f}"):
        print(tab[[c for c in cols if c in tab.columns]].to_string())

    # drift by year (net exp/trade in bp) for the two primary reaction-direction rules
    print("\n-- drift by entry year (net bp/trade) --")
    for rule in ("L", "S"):
        for H in HORIZONS:
            t = _trade_returns(sub, rule, H)
            if t.empty:
                continue
            by = t.groupby(t["entry_date"].dt.year)["net"].agg(["mean", "count"])
            line = "  ".join(f"{y}:{m * 1e4:+7.1f}({int(c)})" for y, (m, c) in by.iterrows())
            print(f"  {rule} H={H:>2}: {line}")

    if sleeve_rule is not None and sleeve_h is not None:
        mon = _sleeve_monthly(ev, sleeve_rule, sleeve_h, which)
        if len(mon) >= 6:
            corr = float(mon["sleeve"].corr(mon["spy"]))
            srt = float(mon["sleeve"].mean() / mon["sleeve"].std(ddof=1) * np.sqrt(12))
            print(f"\n-- monthly sleeve ({sleeve_rule}, H={sleeve_h}) on {label} --")
            print(f"  months={len(mon)}  ann.mean={mon['sleeve'].mean() * 12:.2%}  "
                  f"ann.Sharpe={srt:.2f}  corr(SPY)={corr:+.2f}")
        else:
            print("\n-- monthly sleeve: too few months --")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["universe", "fetch", "events", "dev", "holdout"])
    ap.add_argument("--sleeve-rule", default=None, help="rule for the monthly sleeve stream (e.g. L)")
    ap.add_argument("--sleeve-h", type=int, default=None, help="horizon for the monthly sleeve stream")
    a = ap.parse_args()
    if a.stage == "universe":
        build_universe()
    elif a.stage == "fetch":
        fetch_earnings()
    elif a.stage == "events":
        build_events()
    else:
        analyze(a.stage, a.sleeve_rule, a.sleeve_h)


if __name__ == "__main__":
    main()
