"""RED-TEAM audit of the banked R4-R6 swing claim (adversarial, read-only).

Claim under attack: stress-gated (risk_off/crisis) LONG swing setups, walk-forward LightGBM meta-model
(prob>=0.5, 1-ATR-stop triple-barrier label), 2-ATR stop/target exit -> +1.48%/trade on the touch-once
2020-2026 holdout (229 trades, PF 1.67, DSR 0.84 at N=30, alpha +1.80%/trade beta 0.31).

Attack vectors (each quantified BEFORE vs AFTER):
  V1  duplicate-trade inflation (same symbol+day multi-trigger; dual share classes; overlapping holds)
  V2  event-day clustering (independent entry days; equal-weight day-cluster stream)
  V3  intrabar barrier ambiguity (PT checked before SL on the same daily bar = optimistic tie-break)
  V5  the 2020 anchor (per-year; ex-2020H1; ex-2020)
  V6  honest multiple-testing (DSR at project n_trials=367 on the deduped day-cluster stream)

  python scripts/redteam_swing.py --stage main      # V1/V2/V3/V5/V6
  python scripts/redteam_swing.py --stage regime    # V4 regime PIT truncation test
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

from startx.scanner.signals import _atr
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
CACHE = "data/cache/reval_oos_sl1.0.parquet"
SCRATCH = "/tmp/claude-0/-home-user-start-x/c5a649a2-ecce-553b-a8d8-cbfe0f58637e/scratchpad"
STRESS = ["risk_off", "crisis"]
SPLIT = pd.Timestamp("2020-01-01")
COMPANY_MAP = {"GOOG": "GOOGL"}          # dual share classes in the holdout universe
COST_BPS = 3.0
PT_MULT, SL_MULT, HORIZON = 2.0, 2.0, 10  # the BANKED exit (R6): 2-ATR stop / 2-ATR target / 10d cap


def _load_px(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def resim(taken: pd.DataFrame, prices: dict, *, pt_mult=PT_MULT, sl_mult=SL_MULT, horizon=HORIZON,
          cost_bps=COST_BPS, tiebreak="optimistic"):
    """Re-walk the barriers exactly like improve._resim, but with a controllable same-bar tie-break.

    tiebreak='optimistic'  -> PT checked first on each bar (mirrors harness.build_dataset / improve._resim)
    tiebreak='pessimistic' -> SL checked first on any bar where BOTH barriers are inside the bar range
    Returns DataFrame aligned to `taken`: ret, exit_date, reason, ambig (both barriers hit on the deciding bar).
    """
    cost = cost_bps / 1e4
    out = []
    cache = {}
    for _, e in taken.iterrows():
        sym = e["symbol"]
        if sym not in cache:
            p = prices.get(sym)
            cache[sym] = None if p is None else (p, pd.DatetimeIndex(p["date"]), _atr(p).to_numpy(float))
        if cache[sym] is None:
            out.append({"ret": np.nan, "exit_date": pd.NaT, "reason": "no_data", "ambig": False})
            continue
        p, dates, atr = cache[sym]
        i = int(dates.get_indexer([pd.Timestamp(e["date"])])[0])
        if i < 0 or i + 1 >= len(p) or not np.isfinite(atr[i]) or atr[i] <= 0:
            out.append({"ret": np.nan, "exit_date": pd.NaT, "reason": "no_data", "ambig": False})
            continue
        o = p["open"].to_numpy(float)
        h = p["high"].to_numpy(float)
        lo = p["low"].to_numpy(float)
        c = p["close"].to_numpy(float)
        ie = i + 1
        entry = o[ie]
        pt = entry + pt_mult * atr[i]
        sl = entry - sl_mult * atr[i]
        last = min(ie + horizon, len(c) - 1)
        exit_px, jx, reason, ambig = c[last], last, "time", False
        for j in range(ie, last + 1):
            hit_pt, hit_sl = h[j] >= pt, lo[j] <= sl
            if hit_pt and hit_sl:
                ambig = True
                exit_px, jx, reason = (pt, j, "target") if tiebreak == "optimistic" else (sl, j, "stop")
                break
            if hit_pt:
                exit_px, jx, reason = pt, j, "target"
                break
            if hit_sl:
                exit_px, jx, reason = sl, j, "stop"
                break
        out.append({"ret": (exit_px / entry - 1.0) - cost, "exit_date": dates[jx], "reason": reason,
                    "ambig": ambig})
    return pd.DataFrame(out, index=taken.index)


def stats(r, *, label="", n_trials_list=(1, 30, 367), ppy=25):
    r = pd.Series(np.asarray(r, dtype=float)).dropna()
    if len(r) < 5:
        return {"label": label, "n": len(r)}
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std(ddof=1)
    d = {"label": label, "n": int(len(r)), "exp%": 100 * r.mean(),
         "PF": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
         "win%": 100 * (r > 0).mean(),
         "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan}
    for nt in n_trials_list:
        d[f"DSR@{nt}"] = float(deflated_sharpe(r.values, n_trials=nt))
    return d


def print_stats(rows):
    cols = ["label", "n", "exp%", "PF", "win%", "sharpe", "DSR@1", "DSR@30", "DSR@367"]
    print(f"{'label':44}{'n':>5}{'exp%':>8}{'PF':>7}{'win%':>7}{'Shp':>7}{'DSR@1':>7}{'DSR@30':>8}{'DSR@367':>9}")
    for d in rows:
        print(f"{d['label']:44}{d['n']:>5}" + "".join(
            f"{d.get(k, float('nan')):>{w}.2f}" for k, w in
            [("exp%", 8), ("PF", 7), ("win%", 7), ("sharpe", 7), ("DSR@1", 7), ("DSR@30", 8), ("DSR@367", 9)]))


def alpha_beta(r: pd.Series, spy: pd.Series):
    """r indexed by entry_date (dupes allowed). SPY fwd-10d regression -> (alpha, beta)."""
    spyfwd = (spy.shift(-10) / spy - 1.0)
    x = spyfwd.reindex(r.index)
    df = pd.concat([r.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    beta = float(np.polyfit(df.x, df.y, 1)[0])
    return float(df.y.mean() - beta * df.x.mean()), beta


def dedup_overlap(t: pd.DataFrame) -> pd.DataFrame:
    """One trade per COMPANY per overlapping hold window: drop a re-entry while a prior trade is open."""
    keep = []
    for comp, g in t.sort_values(["company", "entry_date"]).groupby("company"):
        last_exit = pd.Timestamp.min
        for idx, row in g.iterrows():
            if row["entry_date"] > last_exit:
                keep.append(idx)
                last_exit = row["exit_date"]
    return t.loc[sorted(keep)]


def stage_main():
    df = pd.read_parquet(CACHE)
    hold = df[(df["regime"].isin(STRESS)) & (df["prob"] >= 0.5) &
              (df["entry_date"] >= SPLIT)].copy().reset_index(drop=True)
    print(f"holdout takes (banked filter): {len(hold)}  [claim: 229]")
    spy = _load_px("SPY").set_index("date")["close"]
    prices = {s: _load_px(s) for s in hold["symbol"].unique()}

    # -------- reproduce the banked numbers ------------------------------------------------------
    print("\n================ REPRODUCTION (before any attack) ================")
    rows = [stats(hold.set_index("entry_date")["ret"], label="1-ATR exit, as cached (LEDGER +0.99%)")]
    sim_o = resim(hold, prices, tiebreak="optimistic")
    sim_p = resim(hold, prices, tiebreak="pessimistic")
    hold["ret2"] = sim_o["ret"].values
    hold["ret2_pess"] = sim_p["ret"].values
    hold["exit_date"] = sim_o["exit_date"].values
    hold["reason"] = sim_o["reason"].values
    hold["ambig"] = sim_o["ambig"].values | sim_p["ambig"].values
    rows.append(stats(hold["ret2"], label="2-ATR exit resim (BANKED +1.48%)"))
    print_stats(rows)
    r2 = hold.set_index("entry_date")["ret2"].sort_index()
    a, b = alpha_beta(r2, spy)
    print(f"alpha/beta on banked stream: alpha {a*100:+.2f}%/trade  beta {b:+.2f}  [claim +1.80% / 0.31]")
    print("exit reasons:", hold["reason"].value_counts().to_dict())

    # -------- V3: intrabar tie-break ------------------------------------------------------------
    print("\n================ V3 INTRABAR AMBIGUITY (PT-before-SL tie-break) ================")
    n_amb = int(hold["ambig"].sum())
    flipped = hold[hold["ret2"] != hold["ret2_pess"]]
    print(f"trades whose deciding bar contains BOTH barriers: {n_amb} of {len(hold)} "
          f"({100*n_amb/len(hold):.1f}%)  -> outcome flips on {len(flipped)}")
    print_stats([stats(hold["ret2"], label="optimistic (banked convention)"),
                 stats(hold["ret2_pess"], label="pessimistic (stop-first on ambiguous bar)")])

    # -------- V1: dedup ladder ------------------------------------------------------------------
    print("\n================ V1 DUPLICATE-TRADE INFLATION ================")
    hold["company"] = hold["symbol"].map(lambda s: COMPANY_MAP.get(s, s))
    d1 = hold.drop_duplicates(subset=["symbol", "entry_date"])
    d2 = d1.drop_duplicates(subset=["company", "entry_date"])
    d3 = dedup_overlap(d2)
    dupe_trig = len(hold) - len(d1)
    dupe_class = len(d1) - len(d2)
    dupe_olap = len(d2) - len(d3)
    print(f"multi-trigger same symbol+day: {dupe_trig}   dual-class same company+day: {dupe_class}   "
          f"overlapping-hold re-entries: {dupe_olap}")
    same = hold.groupby(["symbol", "entry_date"])["ret2"].nunique()
    print(f"identical price paths among multi-trigger dupes: "
          f"{int((same[same.index.isin(hold.groupby(['symbol','entry_date']).size()[lambda s: s>1].index)] == 1).sum())}"
          f" of {int((hold.groupby(['symbol','entry_date']).size() > 1).sum())} dup groups have identical ret")
    ladder = [("L0 banked (229, dupes in)", hold), ("L1 one per SYMBOL+day", d1),
              ("L2 one per COMPANY+day", d2), ("L3 + no overlapping holds", d3)]
    print_stats([stats(t["ret2"], label=lab) for lab, t in ladder])
    for lab, t in ladder[1:]:
        a, b = alpha_beta(t.set_index("entry_date")["ret2"].sort_index(), spy)
        print(f"  {lab}: alpha {a*100:+.2f}% beta {b:+.2f}")

    # -------- V2: event-day clustering ----------------------------------------------------------
    print("\n================ V2 EVENT-DAY CLUSTERING ================")
    for lab, t in [("L0 banked", hold), ("L3 deduped", d3)]:
        days = t.groupby("entry_date").size()
        # max concurrent open positions (entry..exit inclusive)
        cal = pd.Series(0, index=pd.bdate_range(t["entry_date"].min(), t["exit_date"].max()))
        for _, e in t.iterrows():
            cal.loc[e["entry_date"]:e["exit_date"]] += 1
        print(f"{lab}: {len(t)} trades on {t['entry_date'].nunique()} unique entry days "
              f"(max {days.max()} entries/day, mean {days.mean():.1f}); max concurrent open = {cal.max()}")
    day_all = hold.groupby("entry_date")["ret2"].mean()
    day_d3 = d3.groupby("entry_date")["ret2"].mean()
    day_d3_pess = d3.groupby("entry_date")["ret2_pess"].mean()
    print_stats([stats(day_all, label="day-cluster stream (all 229 -> days)"),
                 stats(day_d3, label="day-cluster stream (deduped L3 -> days)"),
                 stats(day_d3_pess, label="day-cluster L3, pessimistic tie-break")])
    a, b = alpha_beta(day_d3, spy)
    print(f"day-cluster L3 alpha/beta: alpha {a*100:+.2f}%/day-cluster  beta {b:+.2f}")

    # -------- V5: the 2020 anchor ---------------------------------------------------------------
    print("\n================ V5 THE 2020 ANCHOR ================")
    for lab, t, rc in [("banked 229 / ret2", hold, "ret2"), ("deduped L3 / ret2", d3, "ret2")]:
        print(f"--- {lab} per calendar year ---")
        g = t.set_index("entry_date")
        rows = [stats(g.loc[str(y), rc], label=f"  {y}", n_trials_list=(1,))
                for y in range(2020, 2027) if str(y) in g.index.strftime("%Y").unique()]
        print_stats([r for r in rows if r.get("n", 0) >= 5])
        ex_h1 = g[~((g.index >= "2020-01-01") & (g.index < "2020-07-01"))][rc]
        ex_20 = g[g.index >= "2021-01-01"][rc]
        print_stats([stats(ex_h1, label="  ex-2020H1"), stats(ex_20, label="  ex-2020 entirely")])
        if lab.startswith("deduped"):
            dd = d3[~((d3["entry_date"] >= "2020-01-01") & (d3["entry_date"] < "2020-07-01"))]
            print_stats([stats(dd.groupby("entry_date")["ret2"].mean(), label="  ex-2020H1 day-cluster")])
    print("\nentry-day counts by year (deduped L3):")
    print(d3.groupby(d3["entry_date"].dt.year)["entry_date"].nunique().to_string())

    # -------- V6: honest deflation --------------------------------------------------------------
    print("\n================ V6 HONEST BAR (deduped day-cluster stream, n_trials=367) ================")
    print_stats([stats(day_d3, label="deduped day-cluster, DSR at N=1/30/367"),
                 stats(day_d3_pess, label="same, pessimistic tie-break")])
    out = os.path.join(SCRATCH, "redteam_holdout_enriched.parquet")
    hold.to_parquet(out)
    print(f"\nenriched holdout cached -> {out}")


def stage_regime():
    """V4: is classify() truly PIT? Recompute the regime on a TRUNCATED panel and compare labels."""
    from startx.data.membership import SP500Membership
    from startx.scanner.regime import classify, regime_panel
    panel_cache = os.path.join(SCRATCH, "regime_panel_full.parquet")
    if os.path.exists(panel_cache):
        panel = pd.read_parquet(panel_cache)
    else:
        mem = SP500Membership.load()
        panel = regime_panel(membership=mem)
        panel.to_parquet(panel_cache)
    full = classify(panel)
    print("full-sample regime counts:", full["regime"].value_counts(dropna=False).to_dict())
    for cut in ("2020-04-01", "2021-01-01", "2022-07-01", "2024-01-01"):
        trunc = classify(panel[panel.index < cut])
        j = full.loc[full.index < cut, "regime"]
        k = trunc["regime"]
        both = pd.concat([j.rename("full"), k.rename("trunc")], axis=1).dropna()
        mism = (both["full"] != both["trunc"]).sum()
        print(f"truncate@{cut}: {len(both)} labeled days, mismatches vs full-sample run = {mism}")
    # do the cached event regimes match a fresh classify()?  (catches stale/edited labels)
    df = pd.read_parquet(CACHE)
    hold = df[(df["regime"].isin(STRESS)) & (df["prob"] >= 0.5) & (df["entry_date"] >= SPLIT)]
    fresh = full["regime"].reindex(pd.to_datetime(hold["date"]))
    match = (fresh.to_numpy() == hold["regime"].to_numpy())
    print(f"cached event regime labels match fresh classify(): {match.sum()}/{len(match)}")
    stress_days = full[(full.index >= SPLIT) & full["regime"].isin(STRESS)]
    print(f"stress days in holdout window: {len(stress_days)} "
          f"({stress_days.index.year.value_counts().sort_index().to_dict()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "regime"], default="main")
    args = ap.parse_args()
    (stage_main if args.stage == "main" else stage_regime)()
