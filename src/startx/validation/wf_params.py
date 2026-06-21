"""Walk-forward PARAMETER selection — does a book's edge survive honest OOS tuning?

The three production books (``scripts/run_book.py``) had their parameters (IBS threshold,
max-hold caps, chandelier multiple, SMA length, band) picked while looking at the *full*
2019->2026 sample. That is in-sample selection: the reported Sharpe is the Sharpe of the
*winner of a search*, which is upward-biased even when no single config is special. This
module measures the honest number — the Sharpe you would actually have earned if you had
re-picked the parameters as you went, using only the past.

Method (López-de-Prado-style anchored / rolling walk-forward)
------------------------------------------------------------
For each book we sweep a small grid of parameter configs. Each config is run ONCE over the
full span to produce a deterministic daily-return series (the engine is point-in-time and
window-restricted, so a day's realised return does not depend on how we later slice it).
Then we roll a window forward:

1. TRAIN block = the trailing ``train`` trading days. Pick the config with the best
   in-window objective (default: annualised Sharpe of TRAIN daily returns) on TRAIN ONLY.
2. TEST block = the next ``test`` unseen trading days. Record the *picked* config's TEST
   daily returns into the stitched OOS series. Never peek at TEST when choosing.
3. Advance by one TEST block; repeat (anchored=expanding TRAIN, else rolling).

Concatenating all TEST blocks gives ONE out-of-sample return series produced entirely by
past-only parameter choices. We report its annualised Sharpe and a **deflated** Sharpe that
is penalised for the number of distinct configs the search tried (``deflated_sharpe`` with
``n_trials = len(grid)``) — the multiple-testing correction that stops a lucky best-of-N
from looking real. We compare that to the fixed-parameter (full-sample, in-sample) result
of the book's *production* config.

The harness is engine-agnostic: a *book spec* supplies (a) the parameter grid and (b) a
callable ``run(params) -> pd.Series`` returning a config's daily returns indexed by date.
``scripts/walkforward_params.py`` wires the three real books to it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from startx.validation.metrics import deflated_sharpe, max_drawdown, sharpe

TRADING_DAYS = 252


# --------------------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    """Output of :func:`walk_forward_select`.

    Attributes
    ----------
    oos_returns:
        The stitched out-of-sample daily-return series (one entry per TEST day, in time
        order) produced by past-only parameter choices.
    picks:
        One row per fold: ``fold, train_start, train_end, test_start, test_end, n_test``
        plus the chosen parameter values and the TRAIN/TEST Sharpe of the pick. Lets you
        see whether the selection is *stable* (same params chosen) or *thrashing*.
    oos_sharpe:
        Annualised Sharpe of ``oos_returns``.
    oos_dsr:
        Deflated Sharpe (prob. true Sharpe > 0) of ``oos_returns``, penalised for the
        number of configs tried (``n_trials = len(grid)``).
    oos_max_drawdown:
        Max drawdown of the stitched OOS equity curve (negative fraction).
    is_sharpe / is_dsr:
        In-sample (full-span) Sharpe / deflated-Sharpe of the *reference* (production)
        config — the number the walk-forward result is judged against.
    n_configs:
        Number of distinct parameter configs in the grid (the trial count).
    n_folds:
        Number of TEST blocks stitched into the OOS series.
    """

    oos_returns: pd.Series
    picks: pd.DataFrame
    oos_sharpe: float
    oos_dsr: float
    oos_max_drawdown: float
    is_sharpe: float
    is_dsr: float
    n_configs: int
    n_folds: int
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------- #
# Book spec
# --------------------------------------------------------------------------------------- #
@dataclass
class BookSpec:
    """A book's plug into the walk-forward harness.

    Attributes
    ----------
    name:
        Book label (e.g. ``"short_swing"``).
    grid:
        The list of parameter dicts to sweep. Each dict is one config.
    run:
        ``run(params) -> pd.Series`` returning that config's daily returns indexed by a
        DatetimeIndex over the full span. Called once per config (results are cached).
    reference:
        The production parameter dict, used for the fixed-param (in-sample) comparison.
        Must be one of ``grid`` (or at least runnable by ``run``).
    """

    name: str
    grid: Sequence[Mapping[str, Any]]
    run: Callable[[Mapping[str, Any]], pd.Series]
    reference: Mapping[str, Any]


# --------------------------------------------------------------------------------------- #
# Objectives (in-window scores used to PICK on TRAIN)
# --------------------------------------------------------------------------------------- #
def _sharpe_obj(returns: pd.Series) -> float:
    s = sharpe(returns.to_numpy(), periods=TRADING_DAYS)
    return s if np.isfinite(s) else -np.inf


OBJECTIVES: dict[str, Callable[[pd.Series], float]] = {"sharpe": _sharpe_obj}


# --------------------------------------------------------------------------------------- #
# Core walk-forward parameter selection
# --------------------------------------------------------------------------------------- #
def walk_forward_select(
    spec: BookSpec,
    *,
    train: int = 504,
    test: int = 126,
    anchored: bool = False,
    objective: str | Callable[[pd.Series], float] = "sharpe",
    min_train_obs: int | None = None,
) -> WalkForwardResult:
    """Run walk-forward parameter selection for one book.

    Parameters
    ----------
    spec:
        The :class:`BookSpec` (grid + per-config runner + reference config).
    train:
        TRAIN window length in trading days. With ``anchored=False`` this is a *rolling*
        window of fixed length; with ``anchored=True`` it is only the *minimum* — the
        TRAIN window expands from the start of the data to the test boundary.
    test:
        TEST block length in trading days (the OOS step size).
    anchored:
        Expanding TRAIN (``True``) vs rolling fixed-length TRAIN (``False``, default).
    objective:
        Name in :data:`OBJECTIVES` or a ``Series -> float`` callable; higher is better.
        Computed on TRAIN returns only.
    min_train_obs:
        Skip a fold whose TRAIN window has fewer than this many *non-NaN* return days
        (defaults to ``max(60, train // 4)``) so an early, near-empty window can't pick a
        config on noise.

    Returns
    -------
    WalkForwardResult
    """
    obj = OBJECTIVES[objective] if isinstance(objective, str) else objective
    if min_train_obs is None:
        min_train_obs = max(60, train // 4)

    grid = list(spec.grid)
    if len(grid) < 2:
        raise ValueError("need >= 2 configs to make walk-forward selection meaningful")

    # 1) run every config ONCE over the full span; align to a common date index.
    config_returns: list[pd.Series] = []
    for params in grid:
        r = spec.run(params)
        config_returns.append(pd.Series(r).astype(float))
    index = config_returns[0].index
    for r in config_returns:
        if not r.index.equals(index):
            # align defensively on the union, fill flat (no trade) days with 0.0
            index = index.union(r.index)
    ret_mat = pd.DataFrame(
        {i: r.reindex(index).fillna(0.0) for i, r in enumerate(config_returns)},
        index=index,
    ).sort_index()
    n = len(ret_mat)
    dates = ret_mat.index

    # 2) roll the window forward, picking on TRAIN, recording the pick's TEST returns.
    oos_chunks: list[pd.Series] = []
    pick_rows: list[dict] = []
    fold = 0
    test_start = train  # first TEST block begins after the first TRAIN window
    while test_start < n:
        test_end = min(test_start + test, n)
        train_lo = 0 if anchored else max(0, test_start - train)
        train_slice = slice(train_lo, test_start)
        te_slice = slice(test_start, test_end)

        train_mat = ret_mat.iloc[train_slice]
        if train_mat.shape[0] < min_train_obs:
            test_start = test_end
            continue

        # PICK: best in-window objective on TRAIN only.
        scores = np.array([obj(train_mat[c]) for c in train_mat.columns])
        best = int(np.nanargmax(scores))
        best_params = dict(grid[best])

        test_ret = ret_mat.iloc[te_slice, best]
        oos_chunks.append(test_ret)

        pick_rows.append({
            "fold": fold,
            "train_start": dates[train_lo],
            "train_end": dates[test_start - 1],
            "test_start": dates[test_start],
            "test_end": dates[test_end - 1],
            "n_test": int(test_end - test_start),
            **{f"param_{k}": v for k, v in best_params.items()},
            "train_sharpe": float(scores[best]),
            "test_sharpe": _sharpe_obj(test_ret),
        })
        fold += 1
        test_start = test_end

    if not oos_chunks:
        raise ValueError(
            "no walk-forward folds produced — span too short for the chosen train/test "
            f"(have {n} days, need > train={train})"
        )

    oos_returns = pd.concat(oos_chunks).sort_index()
    picks = pd.DataFrame(pick_rows)

    oos_sharpe = sharpe(oos_returns.to_numpy(), periods=TRADING_DAYS)
    oos_dsr = deflated_sharpe(oos_returns.to_numpy(), n_trials=len(grid))
    oos_mdd = max_drawdown(oos_returns.to_numpy(), is_returns=True)

    # In-sample reference: the production config over the same span the OOS series covers.
    ref_full = spec.run(spec.reference).astype(float)
    ref = ref_full.reindex(oos_returns.index).fillna(0.0)
    is_sharpe = sharpe(ref.to_numpy(), periods=TRADING_DAYS)
    is_dsr = deflated_sharpe(ref.to_numpy(), n_trials=len(grid))

    return WalkForwardResult(
        oos_returns=oos_returns,
        picks=picks,
        oos_sharpe=float(oos_sharpe),
        oos_dsr=float(oos_dsr),
        oos_max_drawdown=float(oos_mdd),
        is_sharpe=float(is_sharpe),
        is_dsr=float(is_dsr),
        n_configs=len(grid),
        n_folds=len(picks),
    )


def pick_stability(picks: pd.DataFrame) -> dict:
    """Summarise whether parameter picks are STABLE or THRASHING across folds.

    Returns, per swept parameter, the distinct values chosen, the modal (most-picked)
    value and the fraction of folds it won, plus the number of times the chosen config
    *changed* between consecutive folds (lower = more stable).
    """
    if picks.empty:
        return {"n_folds": 0, "params": {}, "n_switches": 0}
    pcols = [c for c in picks.columns if c.startswith("param_")]
    out: dict[str, Any] = {"n_folds": int(len(picks)), "params": {}}
    for c in pcols:
        vc = picks[c].value_counts()
        modal = vc.index[0]
        out["params"][c.removeprefix("param_")] = {
            "values": vc.to_dict(),
            "modal": modal,
            "modal_frac": float(vc.iloc[0] / len(picks)),
        }
    # count fold-to-fold changes in the full chosen tuple
    tuples = list(picks[pcols].itertuples(index=False, name=None))
    out["n_switches"] = int(sum(a != b for a, b in zip(tuples[:-1], tuples[1:])))
    return out
