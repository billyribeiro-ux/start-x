"""JOB 1 — THE SHORT REGIME GATE.

Question: in which market states does shorting *single names* (cross-sectional) have POSITIVE
expectancy, vs being run over by the index's upward drift?

Method
------
Universal "short the weakest names" proxy book:
  * each trading day, rank the universe by trailing 21-day relative strength vs SPY
    (name 21d return minus SPY 21d return), all point-in-time (uses closes <= t);
  * SHORT the worst-decile names (lowest RS), equal-weight;
  * hold a fixed HORIZON_DAYS (10) and book the forward short return = -(P[t+h]/P[t] - 1);
  * subtract a round-trip cost of COST_BPS (2bp SPY-style) + BORROW_BPS (hard-to-borrow haircut).
The daily book return is the equal-weight mean short return of that day's worst decile.

Then measure that daily book return CONDITIONAL on each regime flag being ON vs OFF.
All flags are point-in-time and shift(1) (decision uses only data known at yesterday's close).

Universe
--------
The 503 CURRENT S&P 500 constituents (data/cache/sp500_members.parquet) that have a price
parquet. This is the SAME one-sided survivorship limitation as market_internals: current
survivors only, no delisted names. CRITICAL for honesty: missing delisted/failed names are
exactly the ones a short would have PROFITED from, so this universe BIASES short P&L UPWARD.
=> any negative short finding here is conservative (the real, survivorship-free short book would
be worse); any positive finding must clear that headwind to be believed.

Window: 2018-01-01 -> 2026-06-25 (headline), with TRAIN 2018-2022 / TEST 2023-2026 split.

Output: per-flag conditional table (mean fwd ret, hit rate, t-stat, %days-on) for TRAIN and TEST,
written to scratchpad CSV, plus the single most robust gate recommendation printed.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        python scripts/short_regime_gate.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- config ---------------------------------------------------------------------------------
PRICES = "data/cache/prices"
MEMBERS = "data/cache/sp500_members.parquet"
INTERNALS = "data/cache/internals.parquet"
OUT = ("/tmp/claude-0/-home-user-start-x/c5a649a2-ecce-553b-a8d8-cbfe0f58637e/"
       "scratchpad/short_regime_gate.csv")

RS_LOOKBACK = 21           # trailing relative-strength window
HORIZON_DAYS = 10          # hold (short-swing cap)
DECILE = 0.10              # short the worst 10% by RS
COST_BPS = 2.0             # round-trip SPY-style cost
BORROW_BPS = 3.0           # extra short-side friction (borrow / locate), per 10d hold
START = "2018-01-01"
END = "2026-06-25"
SPLIT = "2023-01-01"       # TRAIN < SPLIT <= TEST
MIN_NAMES_PER_DAY = 30     # need a real cross-section to form a decile

_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _load_close(sym: str) -> pd.Series | None:
    f = f"{PRICES}/{_SAFE.sub('_', sym)}.parquet"
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=["date", "close"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").set_index("date")["close"]
    return d


def build_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Return (close panel [date x symbol], SPY close) over the window + warm-up."""
    members = pd.read_parquet(MEMBERS)["symbol"].tolist()
    closes = {}
    for s in members:
        c = _load_close(s)
        if c is not None:
            closes[s] = c
    C = pd.DataFrame(closes).sort_index()
    spy = _load_close("SPY")
    # warm-up: need RS_LOOKBACK before START
    lo = pd.Timestamp(START) - pd.Timedelta(days=90)
    C = C.loc[C.index >= lo]
    spy = spy.loc[spy.index >= lo]
    C = C.reindex(spy.index)  # align to SPY trading calendar
    return C, spy


def proxy_short_book(C: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Daily equal-weight short-the-worst-decile book return (net), indexed by entry date.

    All signals are point-in-time: RS at day t uses closes[t] and closes[t-RS_LOOKBACK]; the
    book shorts at the close of t and covers at the close of t+HORIZON_DAYS.
    """
    # trailing RS_LOOKBACK simple return per name and for SPY
    name_ret = C / C.shift(RS_LOOKBACK) - 1.0
    spy_ret = spy / spy.shift(RS_LOOKBACK) - 1.0
    rs = name_ret.sub(spy_ret, axis=0)  # relative strength vs SPY

    # forward HORIZON_DAYS return per name (entry close t -> cover close t+h)
    fwd = C.shift(-HORIZON_DAYS) / C - 1.0
    short_fwd = -fwd  # short profits when price falls
    cost = (COST_BPS + BORROW_BPS) / 1e4

    rows = []
    dates = C.index
    for i, dt in enumerate(dates):
        # need the forward bar to exist
        if i + HORIZON_DAYS >= len(dates):
            break
        rs_row = rs.loc[dt].dropna()
        if len(rs_row) < MIN_NAMES_PER_DAY:
            continue
        k = max(1, int(np.floor(len(rs_row) * DECILE)))
        worst = rs_row.nsmallest(k).index
        sf = short_fwd.loc[dt, worst].dropna()
        if sf.empty:
            continue
        book = float(sf.mean()) - cost
        rows.append({"date": dt, "book_ret": book, "n_names": len(sf)})
    return pd.DataFrame(rows).set_index("date")


# ---- regime flags (all point-in-time; we shift(1) at the merge step) -----------------------
def regime_flags(spy: pd.Series) -> pd.DataFrame:
    """Build candidate daily regime flags from SPY, VIX, VVIX, GLD, and internals."""
    df = pd.DataFrame(index=spy.index)
    sma50 = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    df["spy_below_200"] = spy < sma200
    df["spy_below_50"] = spy < sma50
    df["spy_below_both"] = df["spy_below_200"] & df["spy_below_50"]
    df["death_cross"] = sma50 < sma200

    vix = _load_close("_VIX").reindex(spy.index).ffill()
    vix_sma20 = vix.rolling(20).mean()
    df["vix_above_rising_sma"] = (vix > vix_sma20) & (vix_sma20 > vix_sma20.shift(5))
    df["vix_gt_neutral"] = vix > 18.0  # project memory: VIX neutral ~18

    vvix = _load_close("_VVIX").reindex(spy.index).ffill()
    df["vvix_elevated"] = vvix > vvix.rolling(252, min_periods=60).quantile(0.80)

    gld = _load_close("GLD").reindex(spy.index).ffill()
    ratio = gld / spy
    df["gold_riskoff"] = ratio > ratio.rolling(20).mean()  # _gold_calm_flag logic

    # internals-based breadth deterioration
    I = pd.read_parquet(INTERNALS)
    I["date"] = pd.to_datetime(I["date"])
    I = I.sort_values("date").set_index("date")
    pa200 = I["pct_above_200"].reindex(spy.index).ffill()
    df["breadth_weak_200"] = pa200 < 40.0                       # <40% of names above 200d
    df["breadth_falling"] = pa200 < pa200.shift(10)             # breadth deteriorating
    nl = I["new_lo"].reindex(spy.index).ffill()
    df["newlows_expanding"] = nl > nl.shift(10)                 # new-lows expanding vs 10d ago
    df["newlows_high"] = nl > nl.rolling(252, min_periods=60).quantile(0.80)
    nhnl = I["nh_nl"].reindex(spy.index).ffill()
    df["nhnl_negative"] = nhnl < 0                              # more new lows than highs

    # a couple of small AND-combos
    df["bear_and_weakbreadth"] = df["spy_below_200"] & df["breadth_weak_200"]
    df["bear_and_volstress"] = df["spy_below_200"] & df["vix_above_rising_sma"]
    df["deathcross_and_weakbreadth"] = df["death_cross"] & df["breadth_weak_200"]

    # Deps-free stand-in for the regime.py HMM/VolCorrCluster "stress" state: top-quintile
    # trailing 21d realized vol (the vol-cluster detector is KMeans on exactly this feature).
    r = np.log(spy / spy.shift(1))
    rv = r.rolling(21).std()
    df["vol_stress"] = rv > rv.rolling(252, min_periods=60).quantile(0.80)
    return df


def hmm_flag(spy: pd.Series) -> pd.Series | None:
    """Point-in-time HMM stress-state flag from regime.py, if its ML deps are installed.

    Feature matrix: trailing 21d realized vol + 21d return of SPY (causal). The 'stress' state
    is the highest-vol state; we label it ON. predict_pit is filtered (no look-ahead).
    Falls back to None when hmmlearn/ruptures are not in the venv (qedge ML extras); the
    directly-computed `vol_stress` quintile flag below is the deps-free stand-in.
    """
    try:
        from qedge.modeling.regime import HMMRegimeDetector
    except Exception as e:  # pragma: no cover
        print(f"[hmm] unavailable ({e}); using deps-free vol_stress flag instead")
        return None
    r = np.log(spy / spy.shift(1))
    vol = r.rolling(21).std()
    mom = spy / spy.shift(21) - 1.0
    X = pd.concat([vol, mom], axis=1).dropna()
    X.columns = ["vol", "mom"]
    if len(X) < 500:
        return None
    det = HMMRegimeDetector().fit(X)
    states = det.predict_pit(X)
    meanvol = X.assign(s=states.to_numpy()).groupby("s")["vol"].mean()
    stress_state = int(meanvol.idxmax())
    flag = (states == stress_state)
    return flag.reindex(spy.index).fillna(False)


# ---- conditional analysis ------------------------------------------------------------------
def conditional_stats(book: pd.Series, flag_on: pd.Series) -> dict:
    """Mean fwd ret, hit, t-stat, %days-on for the book conditional on flag ON.

    NOTE: book returns are 10-day-overlapping; the naive t-stat overstates significance by
    ~sqrt(HORIZON_DAYS). We report a Newey-West-lite deflation: divide t by sqrt(HORIZON_DAYS)
    as an honest haircut for the overlap-induced autocorrelation.
    """
    on = book[flag_on.reindex(book.index).fillna(False)]
    if len(on) < 20:
        return {"mean": np.nan, "hit": np.nan, "t": np.nan, "t_adj": np.nan, "n": len(on)}
    mean = float(on.mean())
    hit = float((on > 0).mean())
    sd = float(on.std(ddof=1))
    t = mean / (sd / np.sqrt(len(on))) if sd > 0 else np.nan
    t_adj = t / np.sqrt(HORIZON_DAYS) if np.isfinite(t) else np.nan
    return {"mean": mean, "hit": hit, "t": t, "t_adj": t_adj, "n": len(on)}


def decile_book(C: pd.DataFrame, spy: pd.Series, *, side: str, worst: bool,
                cost_bps: float) -> pd.Series:
    """Generic decile book for the robustness/sign counterfactuals.

    side="short" books -(fwd ret); side="long" books +(fwd ret). worst=True picks the lowest-RS
    decile, worst=False the highest-RS decile. Returns net daily book return over [START, END].
    """
    name_ret = C / C.shift(RS_LOOKBACK) - 1.0
    spy_ret = spy / spy.shift(RS_LOOKBACK) - 1.0
    rs = name_ret.sub(spy_ret, axis=0)
    fwd = C.shift(-HORIZON_DAYS) / C - 1.0
    cost = cost_bps / 1e4
    out = {}
    idx = C.index
    for i, dt in enumerate(idx):
        if i + HORIZON_DAYS >= len(idx):
            break
        row = rs.loc[dt].dropna()
        if len(row) < MIN_NAMES_PER_DAY:
            continue
        k = max(1, int(len(row) * DECILE))
        names = row.nsmallest(k).index if worst else row.nlargest(k).index
        f = fwd.loc[dt, names].dropna()
        if f.empty:
            continue
        r = (-float(f.mean()) if side == "short" else float(f.mean())) - cost
        out[dt] = r
    s = pd.Series(out)
    return s.loc[(s.index >= START) & (s.index <= END)]


def robustness(C: pd.DataFrame, spy: pd.Series) -> None:
    """Prove the ungated loss is STRUCTURAL DRIFT, not a cost artifact, and report the
    cross-sectional sign (the 'weakest' names actually mean-revert UP)."""
    def stat(s: pd.Series, nm: str) -> None:
        m, h = s.mean(), (s > 0).mean()
        sd = s.std(ddof=1)
        t = m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan
        print(f"  {nm:38s} n={len(s):4d} mean={m:+.4%} hit={h:.1%} t_adj={t/np.sqrt(HORIZON_DAYS):+.2f}")
    print("\n=== ROBUSTNESS: is the ungated short loss cost, or structural drift? ===")
    stat(decile_book(C, spy, side="short", worst=True, cost_bps=COST_BPS + BORROW_BPS),
         "SHORT worst-decile (net 5bp)")
    stat(decile_book(C, spy, side="short", worst=True, cost_bps=0.0),
         "SHORT worst-decile (GROSS, 0 cost)")
    print("  --- cross-sectional sign counterfactuals (net 5bp) ---")
    stat(decile_book(C, spy, side="long", worst=True, cost_bps=COST_BPS + BORROW_BPS),
         "LONG worst-decile  (buy the losers)")
    stat(decile_book(C, spy, side="long", worst=False, cost_bps=COST_BPS + BORROW_BPS),
         "LONG best-decile   (buy the winners)")
    stat(decile_book(C, spy, side="short", worst=False, cost_bps=COST_BPS + BORROW_BPS),
         "SHORT best-decile  (fade the winners)")


def run() -> None:
    print("Building price panel (503 current S&P members)...")
    C, spy = build_panel()
    print(f"  panel: {C.shape[0]} dates x {C.shape[1]} names, {C.index.min().date()} -> {C.index.max().date()}")

    print("Building proxy short book (worst-decile RS, 10d hold, net cost)...")
    book = proxy_short_book(C, spy)
    book = book.loc[(book.index >= START) & (book.index <= END)]
    print(f"  book: {len(book)} entry days, mean names/day={book['n_names'].mean():.0f}")

    flags = regime_flags(spy)
    hf = hmm_flag(spy)
    if hf is not None:
        flags["hmm_stress"] = hf

    # shift(1): decision flag known at yesterday's close gates today's entry
    flags = flags.shift(1)

    b = book["book_ret"]
    train = b.loc[b.index < SPLIT]
    test = b.loc[b.index >= SPLIT]

    # ungated baselines
    print("\n=== UNGATED proxy short book (per-10d-trade book return, net) ===")
    for nm, seg in [("FULL", b), ("TRAIN", train), ("TEST", test)]:
        m, h = seg.mean(), (seg > 0).mean()
        sd = seg.std(ddof=1)
        t = m / (sd / np.sqrt(len(seg))) if sd > 0 else np.nan
        print(f"  {nm:6s} n={len(seg):4d} mean={m:+.4%} hit={h:.1%} t={t:+.2f} t_adj={t/np.sqrt(HORIZON_DAYS):+.2f}")

    rows = []
    for col in flags.columns:
        fon = flags[col].fillna(False).astype(bool)
        tr = conditional_stats(train, fon)
        te = conditional_stats(test, fon)
        # fraction of days on (over full book window)
        on_full = fon.reindex(b.index).fillna(False)
        pct_on_full = float(on_full.mean())
        pct_on_tr = float(fon.reindex(train.index).fillna(False).mean())
        pct_on_te = float(fon.reindex(test.index).fillna(False).mean())
        rows.append({
            "flag": col,
            "tr_mean": tr["mean"], "tr_hit": tr["hit"], "tr_t": tr["t"], "tr_t_adj": tr["t_adj"],
            "tr_n": tr["n"], "tr_pct_on": pct_on_tr,
            "te_mean": te["mean"], "te_hit": te["hit"], "te_t": te["t"], "te_t_adj": te["t_adj"],
            "te_n": te["n"], "te_pct_on": pct_on_te,
            "pct_on_full": pct_on_full,
        })
    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    R.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== CONDITIONAL short book return by regime flag (ON only) ===")
    print("(mean = per-10d-trade book return net of cost; t_adj = overlap-deflated t)\n")
    disp = R.copy()
    for c in ["tr_mean", "te_mean", "tr_hit", "te_hit", "tr_pct_on", "te_pct_on"]:
        disp[c] = (disp[c] * 100).round(2)
    for c in ["tr_t", "tr_t_adj", "te_t", "te_t_adj"]:
        disp[c] = disp[c].round(2)
    cols = ["flag", "tr_mean", "tr_hit", "tr_t_adj", "tr_pct_on",
            "te_mean", "te_hit", "te_t_adj", "te_pct_on", "te_n"]
    print(disp[cols].to_string(index=False))

    # recommendation: positive in BOTH train and test, with meaningful sample
    cand = R[(R.tr_mean > 0) & (R.te_mean > 0) & (R.te_n >= 30)].copy()
    cand["robust_score"] = cand[["tr_t_adj", "te_t_adj"]].min(axis=1)
    cand = cand.sort_values("robust_score", ascending=False)
    print("\n=== ROBUST GATES (positive net mean in BOTH train & test) ===")
    if cand.empty:
        print("  NONE. No regime flag flips the proxy single-name short to positive expectancy")
        print("  out-of-sample after costs. (See deliverable for interpretation.)")
    else:
        print(cand[["flag", "tr_mean", "te_mean", "tr_t_adj", "te_t_adj", "te_pct_on", "te_n"]].to_string(index=False))
    robustness(C, spy)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    run()
