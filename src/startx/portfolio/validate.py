"""Validation & forward-test harness — the gate that says **REAL or OVERFIT**.

This module is the skeptic of the production book. It does not generate signals
and it does not size positions; it takes the realised *ledger* (one row per
trade) and the *equity* curve produced by the portfolio engine and answers, with
hard numbers, a single question: **is this edge real, or is it the best of N
lucky backtests?**

Three public entry points (the shared contract with the portfolio engine):

1. :func:`walk_forward_report` — anchored / rolling walk-forward. Tunes on a
   TRAIN window, evaluates the *untouched* TEST window, and reports the decay
   between the two. An edge that only exists in-sample is the headline failure
   mode and this is what exposes it.
2. :func:`confidence_scorecard` — the per-sleeve and whole-book scoreboard:
   Deflated Sharpe (penalised by ``n_trials`` to undo selection bias), the
   Probability of Backtest Overfitting (CSCV), stationary-block-bootstrap 95%
   confidence intervals on expectancy / profit-factor / annualised-Sharpe with
   the bootstrap probability that the true edge is ``<= 0``, trade-concentration
   diagnostics (top-k share, drop-top-k expectancy), per-year sign stability,
   and a final **REAL / FRAGILE / NOISE** verdict.
3. :func:`paper_forward` — replays the book on the most-recent unseen slice as
   if live, for ongoing monitoring (live-vs-expected, a drift-detection hook).

The project's success metric is :func:`drift_adjusted_alpha`: per-trade
expectancy *minus the same-holding-period SPY buy-and-hold drift*. A part-time
long silently collects the bull market; subtract it before claiming an edge.
This metric is threaded through every report here.

The module reuses the repo's validation tooling rather than reinventing it:
:func:`~startx.validation.metrics.deflated_sharpe_ratio`,
:func:`~startx.validation.metrics.probability_of_backtest_overfitting` (CSCV) and
:class:`~startx.validation.purged_cv.PurgedKFold`.

It is testable standalone: with no engine present it builds a multi-sleeve
ledger from the committed sleeves (``momentum_breakout.backtest`` and
``mean_reversion.backtest(signal='ibs')``) via :func:`build_self_ledger`, so the
whole harness can be exercised against a real, self-generated book.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from startx.validation.metrics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from startx.validation.purged_cv import PurgedKFold

# --------------------------------------------------------------------------- #
# Constants — the project's locked conventions.
# --------------------------------------------------------------------------- #

#: Locked study window (see CLAUDE.md). The headline is always the 2019->now slice.
TRAIN_START = "2019-01-01"
TRAIN_TEST_SPLIT = "2023-01-01"
STUDY_END = "2026-06-19"

#: Round-trip SPY transaction cost assumption (~2bp), applied where a sleeve has
#: not already netted it.
SPY_COST_BPS = 2.0

#: Trading periods per year for annualisation (daily bars).
PERIODS_PER_YEAR = 252

#: Verdict thresholds. The DSR is a probability the *true* Sharpe is positive;
#: 0.95 is the conventional "unlikely to be a fluke" bar. PBO above 0.5 means the
#: in-sample winner is, more often than not, a below-median performer OOS.
DSR_REAL = 0.95
DSR_FRAGILE = 0.70
PBO_MAX = 0.50

#: Minimum trade count below which any verdict is capped at NOISE regardless of
#: point estimates — small samples manufacture spurious significance.
MIN_TRADES_FOR_EDGE = 15


# --------------------------------------------------------------------------- #
# Ledger schema & normalisation.
# --------------------------------------------------------------------------- #

#: Canonical ledger columns. The portfolio engine emits these; sleeve backtests
#: emit a superset/subset that :func:`normalize_ledger` maps onto this schema.
LEDGER_COLUMNS = (
    "sleeve",
    "symbol",
    "entry_date",
    "exit_date",
    "ret",          # per-trade fractional return, net of cost
    "holding_days", # calendar days entry->exit (for drift matching)
    "weight",       # sleeve weight in the book (defaults to equal)
    "pnl_contrib",  # weighted return contribution to book equity
    "outcome",      # WIN / LOSS / SCRATCH  (+1ATR or breakeven = WIN/SCRATCH)
)


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("entry_date", "exit_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _label_outcome(ret: float) -> str:
    """Project labelling rule: a +1 ATR or breakeven exit is a WIN/SCRATCH, never
    a LOSS. We cannot see the ATR here, so the supportable rule is: strictly
    positive net return -> WIN, strictly negative -> LOSS, exactly flat -> SCRATCH.
    """
    if ret > 0:
        return "WIN"
    if ret < 0:
        return "LOSS"
    return "SCRATCH"


def normalize_ledger(ledger: pd.DataFrame, *, sleeve: str | None = None,
                     default_weight: float = 1.0) -> pd.DataFrame:
    """Map any sleeve/engine trade frame onto the canonical :data:`LEDGER_COLUMNS`.

    Handles the two schemas in the repo: the momentum / vix sleeves expose
    ``entry_price``/``exit_price``/``pnl_per_share``/``outcome``; the
    mean-reversion sleeve exposes a pre-netted ``ret`` and no ``outcome``.
    Missing ``weight`` defaults to ``default_weight``; ``pnl_contrib`` defaults
    to ``ret * weight``. ``holding_days`` is derived from the entry/exit dates.

    Parameters
    ----------
    ledger:
        A trade frame from a sleeve ``backtest`` or the engine's ``.ledger``.
    sleeve:
        Name to stamp on the ``sleeve`` column when the frame lacks one.
    default_weight:
        Weight to assign when the frame carries none.

    Returns
    -------
    pd.DataFrame
        One row per trade with exactly :data:`LEDGER_COLUMNS`, sorted by
        ``entry_date``.
    """
    if ledger is None or len(ledger) == 0:
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))

    df = _coerce_dates(ledger)

    # ret: prefer an explicit column, else derive from prices.
    if "ret" in df.columns:
        ret = df["ret"].astype(float)
    elif {"entry_price", "exit_price"}.issubset(df.columns):
        cost = SPY_COST_BPS / 10_000.0
        ret = (df["exit_price"].astype(float) / df["entry_price"].astype(float)
               - 1.0) - cost
    else:
        raise ValueError(
            "ledger needs either a 'ret' column or 'entry_price'+'exit_price'")

    sleeve_col = (df["sleeve"] if "sleeve" in df.columns
                  else pd.Series(sleeve or "book", index=df.index))
    symbol_col = (df["symbol"] if "symbol" in df.columns
                  else pd.Series("SPY", index=df.index))
    weight_col = (df["weight"].astype(float) if "weight" in df.columns
                  else pd.Series(float(default_weight), index=df.index))

    if "holding_days" in df.columns:
        holding = df["holding_days"].astype(float)
    elif {"entry_date", "exit_date"}.issubset(df.columns):
        holding = (df["exit_date"] - df["entry_date"]).dt.days.astype(float)
    else:
        holding = pd.Series(np.nan, index=df.index)

    if "pnl_contrib" in df.columns:
        pnl_contrib = df["pnl_contrib"].astype(float)
    else:
        pnl_contrib = ret * weight_col

    if "outcome" in df.columns:
        outcome = df["outcome"].astype(str)
    else:
        outcome = ret.map(_label_outcome)

    out = pd.DataFrame({
        "sleeve": sleeve_col.to_numpy(),
        "symbol": symbol_col.to_numpy(),
        "entry_date": df.get("entry_date", pd.NaT),
        "exit_date": df.get("exit_date", pd.NaT),
        "ret": ret.to_numpy(),
        "holding_days": holding.to_numpy(),
        "weight": weight_col.to_numpy(),
        "pnl_contrib": pnl_contrib.to_numpy(),
        "outcome": outcome.to_numpy(),
    })
    if "entry_date" in out.columns:
        out = out.sort_values("entry_date").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# The success metric: drift-adjusted alpha.
# --------------------------------------------------------------------------- #

def _spy_drift_for_holding(entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                           spy_close: pd.Series) -> float:
    """SPY buy-and-hold return over the *same* calendar window [entry, exit].

    Uses the SPY close at/after ``entry_date`` and at/before ``exit_date`` — the
    realised drift a passive long would have collected holding for the identical
    period. Returns ``nan`` when the window cannot be located.
    """
    if pd.isna(entry_date) or pd.isna(exit_date):
        return float("nan")
    idx = spy_close.index
    # close on or after entry; close on or before exit (as-of alignment)
    e_pos = idx.searchsorted(entry_date, side="left")
    x_pos = idx.searchsorted(exit_date, side="right") - 1
    if e_pos >= len(idx) or x_pos < 0 or x_pos <= e_pos:
        return float("nan")
    p0 = float(spy_close.iloc[e_pos])
    p1 = float(spy_close.iloc[x_pos])
    if p0 <= 0:
        return float("nan")
    return p1 / p0 - 1.0


def drift_adjusted_alpha(trade_returns: Sequence[float] | pd.Series,
                         holding_days: Sequence[float] | pd.Series | None,
                         spy: pd.DataFrame,
                         *,
                         entry_dates: Sequence[Any] | pd.Series | None = None,
                         exit_dates: Sequence[Any] | pd.Series | None = None) -> float:
    """Per-trade expectancy minus the same-holding-period SPY buy-and-hold drift.

    This is **the project's success metric**. A part-time long silently collects
    the bull-market drift; an honest edge is what remains after you subtract the
    return a passive SPY position would have earned over each trade's *own*
    holding window.

    Two matching modes, in order of fidelity:

    * If ``entry_dates`` and ``exit_dates`` are given, the drift is the realised
      SPY return over each trade's exact calendar window (preferred — exact).
    * Otherwise it falls back to ``holding_days``: the mean SPY daily log-drift
      times each trade's day count (an approximation when dates are unavailable).

    Parameters
    ----------
    trade_returns:
        Per-trade fractional returns (net of cost).
    holding_days:
        Calendar days held per trade (fallback drift matching).
    spy:
        SPY price frame with ``date`` and ``close`` columns.
    entry_dates, exit_dates:
        Optional per-trade entry/exit timestamps for exact window matching.

    Returns
    -------
    float
        Mean ``(trade_return - matched_SPY_drift)`` across trades. Positive means
        the book beat a same-duration passive SPY long; <= 0 means you would have
        done as well or better just holding the index.
    """
    r = np.asarray(pd.Series(trade_returns).astype(float))
    if r.size == 0:
        return float("nan")

    spy = spy.sort_values("date")
    spy_close = pd.Series(spy["close"].astype(float).to_numpy(),
                          index=pd.to_datetime(spy["date"]).to_numpy())

    if entry_dates is not None and exit_dates is not None:
        ed = pd.to_datetime(pd.Series(entry_dates).to_numpy())
        xd = pd.to_datetime(pd.Series(exit_dates).to_numpy())
        drift = np.array([
            _spy_drift_for_holding(e, x, spy_close)
            for e, x in zip(ed, xd)
        ])
    else:
        if holding_days is None:
            raise ValueError(
                "need holding_days when entry/exit dates are not provided")
        hd = np.asarray(pd.Series(holding_days).astype(float))
        # mean SPY daily log drift over the available history -> compounded per hold
        logret = np.log(spy_close / spy_close.shift(1)).dropna()
        mu = float(logret.mean()) if len(logret) else 0.0
        # calendar days -> trading days (~5/7) for fair comparison
        drift = np.expm1(mu * hd * (5.0 / 7.0))

    alpha = r - drift
    alpha = alpha[~np.isnan(alpha)]
    if alpha.size == 0:
        return float("nan")
    return float(alpha.mean())


# --------------------------------------------------------------------------- #
# Stationary block bootstrap.
# --------------------------------------------------------------------------- #

def _optimal_block_length(x: np.ndarray) -> float:
    """Politis & White (2004) rule-of-thumb mean block length for the stationary
    bootstrap, simplified. Falls back to ``n**(1/3)`` when autocorrelation is
    weak. Bounded to ``[1, n/2]``.
    """
    n = x.size
    if n < 8:
        return 1.0
    # lag-1 autocorrelation as a crude dependence proxy
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom == 0:
        return 1.0
    rho1 = float(np.sum(xc[1:] * xc[:-1]) / denom)
    base = n ** (1.0 / 3.0)
    # stretch the block when serial dependence is present
    factor = 1.0 + 2.0 * abs(rho1)
    return float(min(max(base * factor, 1.0), n / 2.0))


def stationary_bootstrap_indices(n: int, block_len: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap (Politis & Romano, 1994) resample of ``range(n)``.

    Blocks have geometrically-distributed lengths with mean ``block_len`` and
    wrap circularly, so the resample preserves serial dependence (autocorrelated
    streaks of wins/losses) instead of assuming IID trades.
    """
    if n == 0:
        return np.empty(0, dtype=int)
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(n, dtype=int)
    i = 0
    cur = int(rng.integers(0, n))
    while i < n:
        idx[i] = cur
        i += 1
        if rng.random() < p:
            cur = int(rng.integers(0, n))  # start a new block
        else:
            cur = (cur + 1) % n            # continue the block (circular)
    return idx


def block_bootstrap_ci(returns: Sequence[float] | pd.Series,
                       stat_fn,
                       *,
                       n_boot: int = 2000,
                       ci: float = 0.95,
                       block_len: float | None = None,
                       seed: int = 12345) -> dict:
    """Stationary-block-bootstrap confidence interval for ``stat_fn(returns)``.

    Resamples the *trade-return series* in dependence-preserving blocks and
    recomputes the statistic on each resample. Returns the point estimate, the
    two-sided ``ci`` interval, and ``p_le_zero`` — the bootstrap probability that
    the true statistic is ``<= 0`` (a one-sided "no edge" p-value).

    Parameters
    ----------
    returns:
        The per-trade return series.
    stat_fn:
        Callable mapping a 1-D return array to a scalar (e.g. expectancy,
        profit factor, annualised Sharpe).
    n_boot:
        Number of bootstrap resamples.
    ci:
        Two-sided coverage (0.95 -> 2.5%/97.5% percentile bounds).
    block_len:
        Mean block length; if ``None`` it is chosen by Politis-White.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{"point", "lo", "hi", "p_le_zero", "block_len", "n_boot"}``.
    """
    r = np.asarray(pd.Series(returns).astype(float).dropna())
    n = r.size
    if n < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_le_zero": float("nan"), "block_len": float("nan"),
                "n_boot": 0}

    bl = _optimal_block_length(r) if block_len is None else float(block_len)
    rng = np.random.default_rng(seed)

    point = float(stat_fn(r))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, bl, rng)
        stats[b] = stat_fn(r[idx])

    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        lo = hi = p_le = float("nan")
    else:
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(stats, alpha))
        hi = float(np.quantile(stats, 1.0 - alpha))
        p_le = float(np.mean(stats <= 0.0))
    return {"point": point, "lo": lo, "hi": hi, "p_le_zero": p_le,
            "block_len": float(bl), "n_boot": int(stats.size)}


# --------------------------------------------------------------------------- #
# Scalar performance statistics on a trade-return series.
# --------------------------------------------------------------------------- #

def _expectancy(r: np.ndarray) -> float:
    return float(np.mean(r)) if r.size else float("nan")


def _profit_factor(r: np.ndarray) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def _trade_sharpe_annual(r: np.ndarray,
                         trades_per_year: float = 50.0) -> float:
    """Annualised Sharpe of a *per-trade* return series.

    Per-trade returns are not daily, so we annualise by the empirical trade
    cadence ``trades_per_year`` rather than 252. The cadence is supplied by the
    caller (derived from the ledger's date span); the default is a placeholder
    for a bare array.
    """
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(trades_per_year))


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def _trades_per_year(ledger: pd.DataFrame) -> float:
    """Empirical trade cadence from the ledger's entry-date span."""
    if "entry_date" not in ledger.columns or ledger["entry_date"].isna().all():
        return 50.0
    span_days = (ledger["entry_date"].max() - ledger["entry_date"].min()).days
    if span_days <= 0:
        return float(len(ledger))
    return float(len(ledger)) / (span_days / 365.25)


# --------------------------------------------------------------------------- #
# Concentration & robustness diagnostics.
# --------------------------------------------------------------------------- #

def concentration_diagnostics(returns: np.ndarray, *, k: int = 5) -> dict:
    """How dependent is the book's profit on a handful of trades?

    Reports the share of total *gross profit* contributed by the top-``k`` winners
    and the expectancy after dropping the top-``k`` trades by return. A real edge
    survives the loss of its luckiest few; a NOISE edge (e.g. VIX-capitulation,
    whose profit lives in ~5 trades) collapses.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0:
        return {"top_k_profit_share": float("nan"),
                "expectancy_drop_top_k": float("nan"), "k": k}
    gains = r[r > 0]
    total_gain = gains.sum()
    kk = min(k, n)
    top_k_gain = np.sort(gains)[-kk:].sum() if gains.size else 0.0
    share = float(top_k_gain / total_gain) if total_gain > 0 else float("nan")
    # expectancy after removing the kk largest returns overall
    order = np.argsort(r)
    kept = r[order][:-kk] if kk < n else np.empty(0)
    exp_drop = float(kept.mean()) if kept.size else float("nan")
    return {"top_k_profit_share": share,
            "expectancy_drop_top_k": exp_drop, "k": kk}


def per_year_sign(ledger: pd.DataFrame) -> dict:
    """Per-calendar-year mean return and the fraction of years that were positive.

    Sign stability across years is the cheapest tell of a regime-specific fluke:
    an edge that is positive in one bull year and negative everywhere else is
    fragile no matter how good the pooled number looks.
    """
    if "entry_date" not in ledger.columns or ledger.empty:
        return {"years": {}, "frac_positive": float("nan"), "n_years": 0}
    g = ledger.dropna(subset=["entry_date"]).copy()
    g["year"] = g["entry_date"].dt.year
    by_year = g.groupby("year")["ret"].mean()
    frac_pos = float((by_year > 0).mean()) if len(by_year) else float("nan")
    return {"years": {int(y): float(v) for y, v in by_year.items()},
            "frac_positive": frac_pos, "n_years": int(len(by_year))}


# --------------------------------------------------------------------------- #
# Verdict.
# --------------------------------------------------------------------------- #

def _verdict(dsr: float, pbo: float | None, alpha: float, p_le_zero: float,
             n_trades: int, frac_year_pos: float,
             drop_top_k_exp: float) -> str:
    """Combine the evidence into REAL / FRAGILE / NOISE.

    REAL    — clears the DSR 0.95 bar, PBO acceptable, drift-adjusted alpha
              positive, the bootstrap "no edge" probability is small, enough
              trades, and the edge does not vanish once the luckiest trades are
              removed.
    NOISE   — fails the basic bars: too few trades, DSR below the fragile floor,
              negative drift-adjusted alpha, PBO indicating overfitting, or a
              bootstrap that cannot rule out zero edge.
    FRAGILE — everything in between: a signal that is suggestive but not yet
              trustworthy out-of-sample.
    """
    pbo_bad = (pbo is not None) and (pbo > PBO_MAX)

    # Hard NOISE gates.
    if n_trades < MIN_TRADES_FOR_EDGE:
        return "NOISE"
    if not np.isnan(alpha) and alpha <= 0:
        return "NOISE"
    if not np.isnan(dsr) and dsr < DSR_FRAGILE:
        return "NOISE"
    if pbo_bad:
        return "NOISE"
    # Edge that lives entirely in the luckiest trades -> NOISE.
    if not np.isnan(drop_top_k_exp) and drop_top_k_exp <= 0:
        return "NOISE"

    real = (
        (not np.isnan(dsr) and dsr >= DSR_REAL)
        and (not np.isnan(alpha) and alpha > 0)
        and (np.isnan(p_le_zero) or p_le_zero <= 0.10)
        and (np.isnan(frac_year_pos) or frac_year_pos >= 0.60)
    )
    return "REAL" if real else "FRAGILE"


# --------------------------------------------------------------------------- #
# 2. confidence_scorecard
# --------------------------------------------------------------------------- #

def _sleeve_row(name: str, sub: pd.DataFrame, spy: pd.DataFrame,
                n_trials: int, n_boot: int,
                perf_is: np.ndarray | None,
                perf_oos: np.ndarray | None) -> dict:
    """Compute one scorecard row for a sleeve (or the whole book)."""
    r = sub["ret"].to_numpy(dtype=float)
    n = r.size
    tpy = _trades_per_year(sub)

    sr_per_obs = (r.mean() / r.std(ddof=1)) if (n > 1 and r.std(ddof=1) > 0) \
        else float("nan")
    # DSR penalised by the number of trials that were searched.
    if n > 1 and np.isfinite(sr_per_obs):
        from scipy.stats import kurtosis as _k, skew as _s
        sk = float(_s(r, bias=False)) if n > 2 else 0.0
        kt = float(_k(r, fisher=False, bias=False)) if n > 3 else 3.0
        dsr = deflated_sharpe_ratio(sr_per_obs, n_trials=n_trials, n_obs=n,
                                    skew=sk, kurt=kt)
    else:
        dsr = float("nan")

    pbo = None
    if perf_is is not None and perf_oos is not None:
        try:
            pbo = probability_of_backtest_overfitting(perf_is, perf_oos)
        except ValueError:
            pbo = None

    exp_ci = block_bootstrap_ci(r, _expectancy, n_boot=n_boot)
    pf_ci = block_bootstrap_ci(r, _profit_factor, n_boot=n_boot)
    shp_ci = block_bootstrap_ci(
        r, lambda a: _trade_sharpe_annual(a, tpy), n_boot=n_boot)

    alpha = drift_adjusted_alpha(
        r, sub.get("holding_days"), spy,
        entry_dates=sub.get("entry_date"), exit_dates=sub.get("exit_date"))

    conc = concentration_diagnostics(r, k=5)
    yr = per_year_sign(sub)

    verdict = _verdict(dsr, pbo, alpha, exp_ci["p_le_zero"], n,
                       yr["frac_positive"], conc["expectancy_drop_top_k"])

    return {
        "sleeve": name,
        "n_trades": n,
        "win_rate": float(np.mean(r > 0)) if n else float("nan"),
        "expectancy": exp_ci["point"],
        "exp_ci_lo": exp_ci["lo"],
        "exp_ci_hi": exp_ci["hi"],
        "exp_p_le_0": exp_ci["p_le_zero"],
        "drift_adj_alpha": alpha,
        "profit_factor": pf_ci["point"],
        "pf_ci_lo": pf_ci["lo"],
        "pf_ci_hi": pf_ci["hi"],
        "sharpe_ann": shp_ci["point"],
        "sharpe_ci_lo": shp_ci["lo"],
        "sharpe_ci_hi": shp_ci["hi"],
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "max_drawdown": _max_drawdown_from_returns(r),
        "top5_profit_share": conc["top_k_profit_share"],
        "exp_drop_top5": conc["expectancy_drop_top_k"],
        "frac_years_positive": yr["frac_positive"],
        "n_years": yr["n_years"],
        "verdict": verdict,
    }


def _cscv_matrices_from_sleeves(ledger: pd.DataFrame, n_splits: int = 8
                                ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build CSCV in-sample / out-of-sample performance matrices for PBO.

    Treats each *sleeve* as a candidate "configuration" and each purged time
    fold as a split. ``perf[s, c]`` = the mean trade return of sleeve ``c`` on
    fold ``s``'s in-sample (resp. out-of-sample) trades. This is exactly the
    matrix shape :func:`probability_of_backtest_overfitting` consumes: it asks
    "does the sleeve that looked best in-sample stay good out-of-sample?".
    """
    sleeves = sorted(ledger["sleeve"].unique())
    if len(sleeves) < 2:
        return None, None
    led = ledger.dropna(subset=["entry_date"]).sort_values("entry_date")
    if led.empty:
        return None, None

    # Index folds over the union timeline of trade entry dates.
    dates = led["entry_date"].reset_index(drop=True)
    t1 = pd.Series(dates.to_numpy(), index=dates.to_numpy())
    try:
        pkf = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=0.01)
        folds = list(pkf.split(t1))
    except ValueError:
        return None, None

    is_rows, oos_rows = [], []
    for train_idx, test_idx in folds:
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        is_row, oos_row = [], []
        for s in sleeves:
            sub = led[led["sleeve"] == s]
            r_is = sub.loc[sub["entry_date"].isin(train_dates), "ret"]
            r_oos = sub.loc[sub["entry_date"].isin(test_dates), "ret"]
            is_row.append(float(r_is.mean()) if len(r_is) else 0.0)
            oos_row.append(float(r_oos.mean()) if len(r_oos) else 0.0)
        is_rows.append(is_row)
        oos_rows.append(oos_row)
    return np.asarray(is_rows), np.asarray(oos_rows)


def confidence_scorecard(ledger: pd.DataFrame, equity: pd.Series | None = None,
                         *, n_trials: int = 50, n_boot: int = 2000,
                         spy: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-sleeve and whole-book confidence scoreboard with a REAL/FRAGILE/NOISE
    verdict.

    For every sleeve and for the combined book this computes:

    * **Deflated Sharpe** — the probability the true Sharpe is positive after
      penalising for ``n_trials`` searched configurations (selection bias).
    * **PBO (CSCV)** — the probability the in-sample-best sleeve is a below-median
      OOS performer (only for the whole book, where >= 2 sleeves exist).
    * **Stationary-block-bootstrap 95% CIs** on expectancy, profit factor and
      annualised Sharpe, plus ``P(edge <= 0)``.
    * **Drift-adjusted alpha** — the project's success metric (expectancy minus
      same-holding-period SPY drift).
    * **Concentration** — top-5 profit share and drop-top-5 expectancy.
    * **Per-year sign** — fraction of calendar years with a positive mean.

    Parameters
    ----------
    ledger:
        Trade ledger (engine ``.ledger`` or a self-generated multi-sleeve frame).
        Normalised internally, so either schema is accepted.
    equity:
        Optional book equity curve (accepted for API symmetry / future
        equity-level diagnostics; the verdicts here are trade-level).
    n_trials:
        Number of configurations searched, for the DSR penalty. Be honest and
        large — the relearn searched dozens of entry/exit/threshold combos.
    n_boot:
        Bootstrap resamples per CI.
    spy:
        SPY price frame for drift matching. If ``None`` it is loaded from the
        cache.

    Returns
    -------
    pd.DataFrame
        One row per sleeve plus a final ``BOOK`` row, indexed by ``sleeve``.
    """
    led = normalize_ledger(ledger)
    if led.empty:
        return pd.DataFrame(columns=["sleeve", "n_trades", "verdict"])
    if spy is None:
        spy = _load_prices("SPY")

    rows: list[dict] = []
    for name in sorted(led["sleeve"].unique()):
        sub = led[led["sleeve"] == name]
        rows.append(_sleeve_row(name, sub, spy, n_trials, n_boot, None, None))

    # Whole book: combine sleeves chronologically; PBO across sleeves via CSCV.
    perf_is, perf_oos = _cscv_matrices_from_sleeves(led)
    book = led.copy()
    book["sleeve"] = "BOOK"
    rows.append(_sleeve_row("BOOK", book, spy, n_trials, n_boot,
                            perf_is, perf_oos))

    out = pd.DataFrame(rows).set_index("sleeve")
    return out


# --------------------------------------------------------------------------- #
# 1. walk_forward_report
# --------------------------------------------------------------------------- #

def _window_stats(sub: pd.DataFrame, spy: pd.DataFrame) -> dict:
    """Headline stats for a slice of the ledger (train or test window)."""
    if sub.empty:
        return {"n_trades": 0, "win_rate": float("nan"),
                "expectancy": float("nan"), "profit_factor": float("nan"),
                "drift_adj_alpha": float("nan"), "max_drawdown": float("nan"),
                "total_return": float("nan")}
    r = sub["ret"].to_numpy(dtype=float)
    eq = np.cumprod(1.0 + r)
    return {
        "n_trades": int(r.size),
        "win_rate": float(np.mean(r > 0)),
        "expectancy": float(r.mean()),
        "profit_factor": _profit_factor(r),
        "drift_adj_alpha": drift_adjusted_alpha(
            r, sub.get("holding_days"), spy,
            entry_dates=sub.get("entry_date"), exit_dates=sub.get("exit_date")),
        "max_drawdown": _max_drawdown_from_returns(r),
        "total_return": float(eq[-1] - 1.0),
    }


def walk_forward_report(spy: pd.DataFrame,
                        aux: Mapping[str, pd.DataFrame] | None,
                        sleeves_or_ledger: Any,
                        *,
                        train: str = TRAIN_START,
                        split: str = TRAIN_TEST_SPLIT,
                        end: str = STUDY_END,
                        n_rolling: int = 0) -> pd.DataFrame:
    """Anchored (and optionally rolling) walk-forward TRAIN-vs-TEST report.

    The headline failure mode in this project is an edge that exists only
    in-sample (the chandelier-tuned-to-2023 trap). This report tunes / measures
    on the closed TRAIN window ``[train, split)`` and evaluates the *untouched*
    TEST window ``[split, end]``, then reports the **decay** between the two —
    expectancy, profit factor, drift-adjusted alpha, win rate, drawdown.

    Parameters
    ----------
    spy:
        SPY price frame (for drift matching and to regenerate sleeves if needed).
    aux:
        Mapping of auxiliary frames (e.g. ``{"VIX": ...}``) for sleeves that need
        them. May be ``None``.
    sleeves_or_ledger:
        Either a ready trade ledger (DataFrame), or a sleeve specification that
        :func:`build_self_ledger` can turn into one (a list of sleeve names, or a
        mapping ``{sleeve: backtest_kwargs}``). When a ledger is given it is used
        as-is; when a spec is given the book is *rebuilt* over the full window.
    train, split, end:
        Window boundaries. TRAIN is ``[train, split)``; TEST is ``[split, end]``.
    n_rolling:
        If > 0, also emit ``n_rolling`` rolling-origin TEST windows of equal
        length after ``split`` (sign-stability across sub-periods). 0 disables.

    Returns
    -------
    pd.DataFrame
        One row per (sleeve, window) with window stats and a ``decay_*`` block on
        the per-sleeve TEST rows quantifying the TRAIN->TEST change.
    """
    if isinstance(sleeves_or_ledger, pd.DataFrame):
        led = normalize_ledger(sleeves_or_ledger)
    else:
        led = build_self_ledger(spy, aux, sleeves_or_ledger,
                                start=train, end=end)
    led = led.dropna(subset=["entry_date"])

    train_ts, split_ts, end_ts = (pd.Timestamp(train), pd.Timestamp(split),
                                  pd.Timestamp(end))

    rows: list[dict] = []
    sleeves = sorted(led["sleeve"].unique()) + ["BOOK"]
    for name in sleeves:
        sub = led if name == "BOOK" else led[led["sleeve"] == name]
        tr = sub[(sub["entry_date"] >= train_ts) & (sub["entry_date"] < split_ts)]
        te = sub[(sub["entry_date"] >= split_ts) & (sub["entry_date"] <= end_ts)]
        tr_s = _window_stats(tr, spy)
        te_s = _window_stats(te, spy)
        rows.append({"sleeve": name, "window": "TRAIN",
                     "span": f"{train}..{split}", **tr_s})
        decay = {
            "decay_expectancy": te_s["expectancy"] - tr_s["expectancy"],
            "decay_profit_factor": te_s["profit_factor"] - tr_s["profit_factor"],
            "decay_drift_alpha": (te_s["drift_adj_alpha"]
                                  - tr_s["drift_adj_alpha"]),
            "test_holds": bool(
                (not np.isnan(te_s["drift_adj_alpha"]))
                and te_s["drift_adj_alpha"] > 0),
        }
        rows.append({"sleeve": name, "window": "TEST",
                     "span": f"{split}..{end}", **te_s, **decay})

        if n_rolling > 0 and name == "BOOK":
            total_days = (end_ts - split_ts).days
            if total_days > 0:
                step = total_days // n_rolling
                for w in range(n_rolling):
                    lo = split_ts + pd.Timedelta(days=w * step)
                    hi = (split_ts + pd.Timedelta(days=(w + 1) * step)
                          if w < n_rolling - 1 else end_ts)
                    win = sub[(sub["entry_date"] >= lo)
                              & (sub["entry_date"] < hi)]
                    rows.append({"sleeve": "BOOK",
                                 "window": f"ROLL{w + 1}",
                                 "span": f"{lo.date()}..{hi.date()}",
                                 **_window_stats(win, spy)})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. paper_forward
# --------------------------------------------------------------------------- #

def paper_forward(spy: pd.DataFrame,
                  aux: Mapping[str, pd.DataFrame] | None,
                  sleeves: Any,
                  *,
                  since: str = "2025-06-20",
                  expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replay the book on the most-recent unseen slice as if live.

    This is the ongoing-monitoring harness: it rebuilds the book's trades from
    ``since`` to the end of available data, returns the live blotter, and — when
    an ``expected`` scorecard (e.g. the TRAIN/whole-history :func:`confidence_scorecard`
    output) is supplied — appends a **live-vs-expected drift** comparison per
    sleeve so a monitor can flag when realised expectancy falls outside the band
    the validation set led us to expect.

    Parameters
    ----------
    spy:
        SPY price frame.
    aux:
        Auxiliary frames for sleeves (e.g. ``{"VIX": ...}``); may be ``None``.
    sleeves:
        Sleeve specification (list of names or ``{name: kwargs}``) for
        :func:`build_self_ledger`. A ready ledger DataFrame is also accepted and
        simply filtered to ``since``.
    since:
        Start of the forward / paper slice (the most-recent *unseen* data).
    expected:
        Optional per-sleeve expectations (a :func:`confidence_scorecard` frame).
        When given, the output gains ``expected_expectancy``, ``exp_ci_lo``,
        ``exp_ci_hi`` and a boolean ``within_band`` per sleeve.

    Returns
    -------
    pd.DataFrame
        The live blotter (trades since ``since``) with per-trade
        drift-adjusted contribution, plus — when ``expected`` is given — a
        trailing block of per-sleeve live-vs-expected rows (``__summary__`` =
        True) for drift detection.
    """
    if isinstance(sleeves, pd.DataFrame):
        led = normalize_ledger(sleeves)
    else:
        led = build_self_ledger(spy, aux, sleeves, start=since, end=None)
    since_ts = pd.Timestamp(since)
    live = led[led["entry_date"] >= since_ts].copy()
    live["__summary__"] = False

    if expected is None or live.empty:
        return live.reset_index(drop=True)

    summary_rows: list[dict] = []
    for name in sorted(live["sleeve"].unique()):
        sub = live[live["sleeve"] == name]
        realised = float(sub["ret"].mean())
        exp_e = lo = hi = float("nan")
        if name in expected.index:
            exp_e = float(expected.loc[name].get("expectancy", float("nan")))
            lo = float(expected.loc[name].get("exp_ci_lo", float("nan")))
            hi = float(expected.loc[name].get("exp_ci_hi", float("nan")))
        within = (np.isnan(lo) or np.isnan(hi)) or (lo <= realised <= hi)
        summary_rows.append({
            "sleeve": name, "__summary__": True,
            "n_trades": int(len(sub)),
            "realised_expectancy": realised,
            "expected_expectancy": exp_e,
            "exp_ci_lo": lo, "exp_ci_hi": hi,
            "within_band": bool(within),
        })
    summary = pd.DataFrame(summary_rows)
    return pd.concat([live.reset_index(drop=True), summary],
                     ignore_index=True)


# --------------------------------------------------------------------------- #
# Self-generated ledger (standalone testability).
# --------------------------------------------------------------------------- #

def _load_prices(symbol: str) -> pd.DataFrame:
    """Load a cached price frame from ``data/cache/prices`` with ``date`` parsed.

    Kept dependency-light (reads the parquet directly) so the harness runs even
    when the FMP client / settings are unavailable.
    """
    from pathlib import Path
    # repo root = three parents up from this file (src/startx/portfolio/validate.py)
    root = Path(__file__).resolve().parents[3]
    path = root / "data" / "cache" / "prices" / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["symbol"] = symbol
    return df


#: Default self-test book: the two committed sleeves with a positive OOS thesis,
#: plus VIX-capitulation as a known NOISE canary.
DEFAULT_SLEEVES: dict[str, dict] = {
    "momentum_breakout": {},
    "ibs_meanrev": {"signal": "ibs"},
    "vix_capitulation": {},
}


def build_self_ledger(spy: pd.DataFrame | None = None,
                      aux: Mapping[str, pd.DataFrame] | None = None,
                      sleeves: Any = None,
                      *,
                      start: str | None = TRAIN_START,
                      end: str | None = STUDY_END) -> pd.DataFrame:
    """Build a multi-sleeve ledger directly from the committed sleeve backtests.

    This makes the whole harness testable with no portfolio engine present: it
    runs ``momentum_breakout.backtest`` and ``mean_reversion.backtest(signal='ibs')``
    (and optionally ``vix_capitulation.backtest``) on SPY over the locked window
    and normalises their trades into one :data:`LEDGER_COLUMNS` ledger.

    Parameters
    ----------
    spy:
        SPY price frame; loaded from cache if ``None``.
    aux:
        Auxiliary frames; ``{"VIX": ...}`` enables the vix-capitulation sleeve.
        Loaded from cache if needed and ``None``.
    sleeves:
        Sleeve spec. ``None`` -> :data:`DEFAULT_SLEEVES`. A list of names selects
        a subset with default kwargs; a mapping ``{name: kwargs}`` overrides
        backtest parameters. Sleeve names: ``momentum_breakout``, ``ibs_meanrev``
        (or ``mean_reversion``), ``vix_capitulation``.
    start, end:
        Backtest window passed to each sleeve.

    Returns
    -------
    pd.DataFrame
        Normalised, chronologically-sorted multi-sleeve ledger.
    """
    from startx.strategy import (
        mean_reversion as _mr,
        momentum_breakout as _mb,
        vix_capitulation as _vc,
    )

    if spy is None:
        spy = _load_prices("SPY")
    spy = spy.copy()
    spy.attrs.setdefault("symbol", "SPY")

    if sleeves is None:
        spec = dict(DEFAULT_SLEEVES)
    elif isinstance(sleeves, Mapping):
        spec = dict(sleeves)
    else:  # iterable of names
        spec = {name: DEFAULT_SLEEVES.get(name, {}) for name in sleeves}

    aux = dict(aux) if aux else {}

    frames: list[pd.DataFrame] = []
    for name, kwargs in spec.items():
        kw = dict(kwargs)
        if name in ("momentum_breakout", "momentum", "breakout"):
            trades, _ = _mb.backtest(spy, start=start, end=end, **kw)
            frames.append(normalize_ledger(trades, sleeve="momentum_breakout"))
        elif name in ("ibs_meanrev", "mean_reversion", "ibs", "meanrev"):
            kw.setdefault("signal", "ibs")
            trades, _ = _mr.backtest(spy, start=start, end=end, **kw)
            frames.append(normalize_ledger(trades, sleeve="ibs_meanrev"))
        elif name in ("vix_capitulation", "vix", "capitulation"):
            vix = aux.get("VIX")
            if vix is None:
                vix = _load_prices("_VIX")
            trades, _ = _vc.backtest(spy, vix, start=start, end=end, **kw)
            frames.append(normalize_ledger(trades, sleeve="vix_capitulation"))
        else:
            raise ValueError(f"unknown sleeve: {name!r}")

    if not frames:
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    led = pd.concat(frames, ignore_index=True)
    led = led.sort_values("entry_date").reset_index(drop=True)
    return led


def build_self_equity(ledger: pd.DataFrame) -> pd.Series:
    """A daily-ish book equity curve from a normalised ledger.

    Compounds each trade's weighted contribution at its exit date (the point the
    P&L is realised). This is a stand-in for the engine's ``.equity`` so the
    harness's equity-consuming paths are exercisable standalone.
    """
    led = normalize_ledger(ledger).dropna(subset=["exit_date"])
    if led.empty:
        return pd.Series(dtype=float)
    daily = led.groupby("exit_date")["pnl_contrib"].sum().sort_index()
    equity = (1.0 + daily).cumprod()
    return equity


__all__ = [
    "drift_adjusted_alpha",
    "walk_forward_report",
    "confidence_scorecard",
    "paper_forward",
    "normalize_ledger",
    "block_bootstrap_ci",
    "stationary_bootstrap_indices",
    "concentration_diagnostics",
    "per_year_sign",
    "build_self_ledger",
    "build_self_equity",
    "LEDGER_COLUMNS",
    "DEFAULT_SLEEVES",
]
