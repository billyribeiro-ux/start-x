"""Cross-sectional SHORT edge research for Start-X.

Goal
----
Find a firewall-cleared SHORT edge in the cross-section of single names — the
short leg of a cross-sectional weakness ranker. The index (SPY) is un-shortable
(upward drift), so the only hope is RELATIVE weakness in single names.

Method
------
1. Build a wide daily panel of single-name closes + dollar volume aligned to the
   SPY trading calendar (done once, cached in the scratchpad).
2. Each rebalance (weekly = every 5 trading days, or monthly = 21) compute
   POINT-IN-TIME weakness features per name, ALL trailing / shift(1) — no
   lookahead:
     - mom21, mom63, mom126  : trailing total return (negative = weak)
     - rs63                  : name 63d return minus SPY 63d return (rel weakness)
     - d200, d50             : (close - SMA)/SMA distance below moving averages
     - vol21                 : 21d realised vol (high = junky)
     - semidev21             : 21d downside semi-deviation
     - rsi14                 : 14d RSI (high RSI rolling over = weak? we test low)
     - amihud21              : Amihud illiquidity (|ret|/dollar-vol)
   A composite z-score "weakness" rank is built (configurable feature set/sign).
3. Rank names each rebalance; SHORT bottom decile (weakest). Hold to next
   rebalance. Equal weight.
4. Build the daily return stream:
     - SHORT-ONLY  : r_short = -mean(name daily returns in basket)
     - MKT-NEUTRAL : r = mean(top decile) - mean(bottom decile)  (dollar-neutral)
   Net of costs: 2bp round-trip slippage on turnover + borrow (annual, daily-accrued)
   on the short notional.
5. Beta-hedge the short-only stream vs SPY (rolling/full-sample beta) and re-test.

The firewall (decides truth):
    from startx.validation.metrics import deflated_sharpe
    from qedge.validation import survival_gate  (DSR>=0.95 AND PBO<=0.5)

OOS protocol: TRAIN 2018-01-01..2022-12-31 (pick rule/decile/params here),
TEST 2023-01-01..2026-06-25 (never fit). Report full 2018->now too.

Run:
    source .venv/bin/activate && python scripts/short_xsec.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = "/home/user/start-x"
SCR = "/tmp/claude-0/-home-user-start-x/c5a649a2-ecce-553b-a8d8-cbfe0f58637e/scratchpad"

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2026-06-25")

SLIP_BP = 2e-4  # round-trip slippage+commission per leg turnover (2bp)


# ---------------------------------------------------------------------------
# Panel construction (cached)
# ---------------------------------------------------------------------------
def build_panel(start="2017-06-01", include_delisted=True, min_bars=1000):
    """Build wide close + volume panels aligned to the SPY calendar."""
    spy = pd.read_parquet(f"{ROOT}/data/cache/prices/SPY.parquet", columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy[spy["date"] >= pd.Timestamp(start)].set_index("date")["close"]
    cal = spy.index

    files = sorted(glob.glob(f"{ROOT}/data/cache/prices/*.parquet"))
    closes, vols = {}, {}
    last_dates = {}
    for f in files:
        sym = os.path.basename(f)[:-8]
        if sym == "SPY":
            continue
        try:
            df = pd.read_parquet(f, columns=["date", "close", "volume"])
        except Exception:
            continue
        if len(df) < min_bars:
            continue
        df["date"] = pd.to_datetime(df["date"])
        last_dates[sym] = df["date"].max()
        df = df[df["date"] >= pd.Timestamp(start)].set_index("date")
        if len(df) == 0:
            continue
        closes[sym] = df["close"]
        vols[sym] = df["volume"]

    C = pd.DataFrame(closes).reindex(cal)
    V = pd.DataFrame(vols).reindex(cal)
    last = pd.Series(last_dates)
    delisted = last[last < pd.Timestamp("2026-03-01")].index.tolist()
    if not include_delisted:
        keep = [c for c in C.columns if c not in set(delisted)]
        C, V = C[keep], V[keep]
    return C, V, spy, set(delisted)


def load_panel():
    C = pd.read_parquet(f"{SCR}/closes.parquet")
    V = pd.read_parquet(f"{SCR}/vols.parquet")
    spy = pd.read_parquet(f"{SCR}/spy.parquet")["SPY"]
    return C, V, spy


# ---------------------------------------------------------------------------
# Feature computation (all trailing, shift(1) applied at use site)
# ---------------------------------------------------------------------------
def compute_features(C, V, spy):
    """Return a dict of feature DataFrames (index=dates, cols=names).

    Every feature is computed from data up to and including date t. At signal
    time we will shift(1) so the rank at rebalance uses only info known at the
    PRIOR close, and positions are held over the *next* period.
    """
    R = C.pct_change()
    spy_r = spy.pct_change()
    logC = np.log(C)

    feats = {}
    # Trailing total returns (momentum). Weak = low / negative.
    feats["mom21"] = C / C.shift(21) - 1.0
    feats["mom63"] = C / C.shift(63) - 1.0
    feats["mom126"] = C / C.shift(126) - 1.0
    # Relative strength vs SPY over 63d (name ret - spy ret). Weak = low.
    spy_63 = (spy / spy.shift(63) - 1.0)
    feats["rs63"] = feats["mom63"].sub(spy_63, axis=0)
    spy_21 = (spy / spy.shift(21) - 1.0)
    feats["rs21"] = feats["mom21"].sub(spy_21, axis=0)
    # Distance below moving averages. Weak = far below (negative).
    sma200 = C.rolling(200).mean()
    sma50 = C.rolling(50).mean()
    feats["d200"] = C / sma200 - 1.0
    feats["d50"] = C / sma50 - 1.0
    # Realised vol / downside semidev (21d). Junky = high.
    feats["vol21"] = R.rolling(21).std()
    down = R.clip(upper=0.0)
    feats["semidev21"] = np.sqrt((down ** 2).rolling(21).mean())
    # RSI(14)
    delta = C.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    feats["rsi14"] = 100 - 100 / (1 + rs)
    # Amihud illiquidity: mean(|ret| / dollar_vol) * 1e9 over 21d. High = illiquid.
    dollar_vol = (C * V).replace(0, np.nan)
    amihud = (R.abs() / dollar_vol).rolling(21).mean() * 1e9
    feats["amihud21"] = amihud
    # Dollar volume (for liquidity screen, not a weakness feature)
    feats["_advol"] = dollar_vol.rolling(21).mean()
    return feats, R, spy_r


# Sign convention: weakness score is the SUM of signed z-scores so that a HIGHER
# score = WEAKER name (the short candidate). For each feature we give the sign
# that makes "weaker" -> "higher contribution".
#   negative momentum  -> weak  -> use  -mom
#   far below SMA      -> weak  -> use  -d200
#   high vol/semidev   -> junky -> use +vol
#   low RSI            -> weak  -> use -rsi
#   high amihud        -> illiq -> use +amihud
FEATURE_SIGNS = {
    "mom21": -1.0,
    "mom63": -1.0,
    "mom126": -1.0,
    "rs21": -1.0,
    "rs63": -1.0,
    "d200": -1.0,
    "d50": -1.0,
    "vol21": +1.0,
    "semidev21": +1.0,
    "rsi14": -1.0,
    "amihud21": +1.0,
}


def cross_sectional_z(df):
    """Row-wise (cross-sectional) z-score, robust to outliers via clip."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.clip(-3, 3)


@dataclass
class Config:
    features: tuple  # feature names used in composite
    rebal: int  # rebalance frequency in trading days
    decile: float  # fraction shorted (0.1 = bottom decile)
    min_advol: float  # min 21d avg dollar volume ($) liquidity screen
    name: str = ""


def weakness_score(feats, cfg: Config):
    """Composite weakness score (higher = weaker) from signed z-scores."""
    parts = []
    for fname in cfg.features:
        z = cross_sectional_z(feats[fname]) * FEATURE_SIGNS[fname]
        parts.append(z)
    score = sum(parts) / len(parts)
    return score


def backtest(feats, R, spy_r, cfg: Config):
    """Return daily return streams for short-only and market-neutral books.

    Positions are decided at rebalance dates using shift(1) features (info known
    at prior close), held for `cfg.rebal` days, equal-weighted, rebalanced.
    """
    score = weakness_score(feats, cfg).shift(1)  # no lookahead
    advol = feats["_advol"].shift(1)
    liquid = advol >= cfg.min_advol

    dates = R.index
    rebal_idx = list(range(0, len(dates), cfg.rebal))

    n = len(dates)
    short_ret = np.full(n, np.nan)
    mn_ret = np.full(n, np.nan)
    n_short = np.zeros(n)
    turnover = np.zeros(n)
    basket_log = {}

    prev_short = set()
    prev_long = set()
    for k, ri in enumerate(rebal_idx):
        d = dates[ri]
        sc = score.iloc[ri].copy()
        # liquidity + valid price screen
        valid = liquid.iloc[ri] & R.iloc[ri].notna() & sc.notna()
        sc = sc[valid]
        if len(sc) < 20:
            continue
        q_hi = sc.quantile(1 - cfg.decile)
        q_lo = sc.quantile(cfg.decile)
        short_names = sc[sc >= q_hi].index  # weakest -> short
        long_names = sc[sc <= q_lo].index  # strongest -> long
        cur_short, cur_long = set(short_names), set(long_names)
        basket_log[d] = sorted(cur_short)

        end = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else n
        seg = slice(ri, end)
        # daily returns over the holding segment
        seg_R = R.iloc[seg]
        if len(short_names) > 0:
            s = -seg_R[list(short_names)].mean(axis=1)  # short: profit when falls
            short_ret[seg] = s.values
            n_short[seg] = len(short_names)
        if len(long_names) > 0 and len(short_names) > 0:
            lo = seg_R[list(long_names)].mean(axis=1)
            sh = -seg_R[list(short_names)].mean(axis=1)
            mn_ret[seg] = (0.5 * lo + 0.5 * sh).values  # dollar-neutral halves

        # turnover at this rebalance (one-way name churn fraction)
        if prev_short:
            churn = len(cur_short ^ prev_short) / max(len(cur_short) + len(prev_short), 1)
        else:
            churn = 1.0
        turnover[ri] = churn
        prev_short, prev_long = cur_short, cur_long

    idx = dates
    short_s = pd.Series(short_ret, index=idx)
    mn_s = pd.Series(mn_ret, index=idx)
    n_short_s = pd.Series(n_short, index=idx)
    turn_s = pd.Series(turnover, index=idx)
    return short_s, mn_s, n_short_s, turn_s, basket_log


def apply_costs(gross, turn, n_short, borrow_annual, slip=SLIP_BP, is_neutral=False):
    """Net returns after slippage on turnover and borrow on short notional.

    Slippage: each rebalance churns `turn` fraction of the basket; round-trip
    cost = 2 * slip * turn applied on the rebalance day. (2bp per leg.)
    Borrow: short notional pays borrow_annual/252 every day held. Market-neutral
    pays borrow only on the short half (0.5 notional).
    """
    daily_borrow = borrow_annual / 252.0
    net = gross.copy()
    # slippage on rebalance days
    slip_cost = 2 * slip * turn  # both legs churn similarly; conservative
    net = net - slip_cost
    # borrow drag every day the short is on
    on = (n_short > 0).astype(float)
    notional = 0.5 if is_neutral else 1.0
    net = net - on * daily_borrow * notional
    return net


# ---------------------------------------------------------------------------
# Stats + firewall
# ---------------------------------------------------------------------------
def ann_stats(r, periods=252):
    r = r.dropna()
    if len(r) < 5:
        return dict(n=len(r), sharpe=np.nan, ann_ret=np.nan, ann_vol=np.nan, maxdd=np.nan, win=np.nan)
    mu = r.mean() * periods
    sd = r.std(ddof=1) * np.sqrt(periods)
    sharpe = mu / sd if sd > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    win = (r > 0).mean()
    return dict(n=len(r), sharpe=sharpe, ann_ret=mu, ann_vol=sd, maxdd=dd, win=win)


def beta_hedge(short_r, spy_r):
    """Remove SPY beta from short-only stream (full-sample OLS beta on overlap)."""
    df = pd.concat([short_r, spy_r], axis=1).dropna()
    df.columns = ["s", "m"]
    if len(df) < 30 or df["m"].var() == 0:
        return short_r, np.nan
    beta = df["s"].cov(df["m"]) / df["m"].var()
    hedged = short_r - beta * spy_r.reindex(short_r.index)
    return hedged, beta


def slice_window(r, lo, hi):
    return r[(r.index >= lo) & (r.index <= hi)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def make_grid():
    """Candidate configs explored on TRAIN. n_trials = len(this grid)."""
    feature_sets = {
        "mom_only": ("mom21", "mom63", "mom126"),
        "rs_only": ("rs21", "rs63"),
        "trend": ("mom63", "mom126", "d200", "d50"),
        "junk": ("vol21", "semidev21", "amihud21"),
        "full": ("mom21", "mom63", "mom126", "rs63", "d200", "d50",
                 "vol21", "semidev21", "rsi14", "amihud21"),
        "weak_trend_junk": ("mom63", "mom126", "d200", "vol21", "semidev21"),
        "rs_trend": ("rs21", "rs63", "d200", "d50"),
    }
    rebals = [5, 21]          # weekly, monthly
    deciles = [0.1, 0.2]      # bottom decile / quintile
    min_advols = [1e7]        # $10M ADV liquidity floor
    grid = []
    for fname, feats in feature_sets.items():
        for rb in rebals:
            for dc in deciles:
                for mv in min_advols:
                    grid.append(Config(features=feats, rebal=rb, decile=dc,
                                        min_advol=mv,
                                        name=f"{fname}|rb{rb}|dc{dc}"))
    return grid


def fmt(s):
    return (f"n={s['n']:5d} Sharpe={s['sharpe']:+.2f} annRet={s['ann_ret']:+.1%} "
            f"annVol={s['ann_vol']:.1%} maxDD={s['maxdd']:+.1%} win={s['win']:.1%}")


def main():
    from startx.validation.metrics import deflated_sharpe
    from qedge.validation import survival_gate

    C, V, spy = load_panel()
    feats, R, spy_r = compute_features(C, V, spy)
    grid = make_grid()
    n_trials = len(grid)
    print(f"=== n_trials (configs searched) = {n_trials} ===\n")

    BORROW_BASE = 0.01   # 1%/yr general collateral
    BORROW_HTB = 0.05    # 5%/yr hard-to-borrow stress

    # ---- TRAIN: backtest every config, score short-only Sharpe on TRAIN ----
    results = {}
    for cfg in grid:
        sh, mn, ns, tn, basket = backtest(feats, R, spy_r, cfg)
        sh_net = apply_costs(sh, tn, ns, BORROW_BASE)
        mn_net = apply_costs(mn, tn, ns, BORROW_BASE, is_neutral=True)
        results[cfg.name] = dict(cfg=cfg, sh=sh, mn=mn, ns=ns, tn=tn,
                                 sh_net=sh_net, mn_net=mn_net)

    # TRAIN performance of short-only net
    train_scores = []
    for name, d in results.items():
        tr = slice_window(d["sh_net"], TRAIN_START, TRAIN_END)
        s = ann_stats(tr)
        train_scores.append((name, s["sharpe"], s))
    train_scores.sort(key=lambda x: (x[1] if not np.isnan(x[1]) else -9), reverse=True)
    print("=== TRAIN (2018-2022) short-only NET Sharpe ranking (top 10) ===")
    for name, sh, s in train_scores[:10]:
        print(f"  {name:28s} {fmt(s)}")
    print()

    best_name = train_scores[0][0]
    best = results[best_name]
    bcfg = best["cfg"]
    print(f">>> SELECTED on TRAIN: {best_name}  features={bcfg.features}\n")

    # ---- OOS + full evaluation of the selected config ----
    def report(label, gross, net, ns, tn, is_neutral=False):
        print(f"--- {label} ---")
        for win_name, lo, hi in [("TRAIN 2018-2022", TRAIN_START, TRAIN_END),
                                  ("OOS   2023-now", TEST_START, TEST_END),
                                  ("FULL  2018-now", TRAIN_START, TEST_END)]:
            g = ann_stats(slice_window(gross, lo, hi))
            nt = ann_stats(slice_window(net, lo, hi))
            tw = slice_window(tn, lo, hi)
            tov = tw[tw > 0]
            avg_to = tov.mean() if len(tov) else np.nan
            print(f"  [{win_name}] GROSS {fmt(g)}")
            print(f"  [{win_name}] NET   {fmt(nt)}  turn/rebal={avg_to:.0%}")
        print()

    print("================ SELECTED CONFIG RESULTS ================\n")
    report("SHORT-ONLY (borrow 1%/yr)", best["sh"], best["sh_net"], best["ns"], best["tn"])
    report("MARKET-NEUTRAL (borrow 1%/yr short half)", best["mn"], best["mn_net"],
           best["ns"], best["tn"], is_neutral=True)

    # HTB stress on short-only
    sh_htb = apply_costs(best["sh"], best["tn"], best["ns"], BORROW_HTB)
    print("--- SHORT-ONLY HTB stress (borrow 5%/yr) ---")
    for win_name, lo, hi in [("OOS 2023-now", TEST_START, TEST_END),
                             ("FULL 2018-now", TRAIN_START, TEST_END)]:
        print(f"  [{win_name}] NET {fmt(ann_stats(slice_window(sh_htb, lo, hi)))}")
    print()

    # Borrow break-even: rate at which OOS net ann_ret -> 0
    oos_sh_gross = slice_window(best["sh"], TEST_START, TEST_END)
    oos_tn = slice_window(best["tn"], TEST_START, TEST_END)
    oos_ns = slice_window(best["ns"], TEST_START, TEST_END)
    g_after_slip = apply_costs(best["sh"], best["tn"], best["ns"], 0.0)
    oos_after_slip = slice_window(g_after_slip, TEST_START, TEST_END)
    ann_after_slip = oos_after_slip.dropna().mean() * 252
    on_frac = (oos_ns > 0).astype(float).reindex(oos_after_slip.index).mean()
    breakeven = ann_after_slip / on_frac if on_frac > 0 else np.nan
    print(f"Borrow break-even (OOS): edge dies at borrow ~= {breakeven:+.1%}/yr "
          f"(short on {on_frac:.0%} of days)\n")

    # ---- Beta hedge ----
    sh_net_full = best["sh_net"]
    hedged, beta = beta_hedge(sh_net_full, spy_r)
    print(f"=== BETA HEDGE (short-only net vs SPY, full-sample beta={beta:+.2f}) ===")
    for win_name, lo, hi in [("TRAIN", TRAIN_START, TRAIN_END),
                             ("OOS", TEST_START, TEST_END),
                             ("FULL", TRAIN_START, TEST_END)]:
        print(f"  [{win_name}] hedged NET {fmt(ann_stats(slice_window(hedged, lo, hi)))}")
    print()

    # ---- FIREWALL ----
    print("================ FIREWALL ================")
    # trial_returns = OOS short-only NET streams of EVERY config (for PBO)
    oos_streams = []
    for name, d in results.items():
        oos_streams.append(slice_window(d["sh_net"], TEST_START, TEST_END))
    # align lengths
    base = slice_window(best["sh_net"], TEST_START, TEST_END).dropna()

    def run_gate(stream, label):
        s = stream.dropna()
        dsr = deflated_sharpe(s.values, n_trials=n_trials)
        # build aligned trial matrix for PBO
        aligned = []
        ref_idx = s.index
        for st in oos_streams:
            a = st.reindex(ref_idx)
            if a.notna().sum() >= 0.8 * len(ref_idx):
                aligned.append(a.fillna(0.0).values)
        gate = survival_gate(s.values, n_trials=n_trials, trial_returns=aligned)
        print(f"  {label}: DSR={dsr:.3f} | gate.DSR={gate.deflated_sharpe:.3f} "
              f"PBO={gate.pbo:.3f} passed={gate.passed} ({gate.verdict})")
        return gate

    print("[OOS 2023-now]")
    run_gate(slice_window(best["sh_net"], TEST_START, TEST_END), "short-only NET")
    run_gate(slice_window(best["mn_net"], TEST_START, TEST_END), "mkt-neutral NET")
    run_gate(slice_window(hedged, TEST_START, TEST_END), "short-only beta-hedged")
    print("[FULL 2018-now]")
    run_gate(slice_window(best["sh_net"], TRAIN_START, TEST_END), "short-only NET")
    run_gate(slice_window(best["mn_net"], TRAIN_START, TEST_END), "mkt-neutral NET")
    print()
    return results, best, grid


if __name__ == "__main__":
    main()
