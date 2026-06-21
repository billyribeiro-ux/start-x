"""First-class regime layer: HMM, change-point, and vol/correlation clustering.

Regime detection is a load-bearing layer of the scanner and is held to the same
point-in-time discipline as every other feature: the regime state assigned to
time ``t`` must depend ONLY on observations ``<= t``.

The subtle trap is that the usual hmmlearn entry points —
:meth:`~hmmlearn.hmm.GaussianHMM.predict` (Viterbi),
:meth:`~hmmlearn.hmm.GaussianHMM.predict_proba` and
:meth:`~hmmlearn.hmm.GaussianHMM.score_samples` (forward-backward posteriors) —
assign ``t``'s state using the WHOLE observation sequence, including the future.
That is look-ahead and is forbidden here.

Instead :class:`HMMRegimeDetector` runs the **forward (alpha) recursion** itself
from the fitted ``startprob_``, ``transmat_`` and the per-observation emission
log-likelihoods (``model._compute_log_likelihood``), normalising at each step to
obtain the *filtered* distribution ``P(state_t | obs_1..t)``. Because the alpha
recursion at step ``t`` consumes only ``obs_1..t``, appending future rows can
never perturb a past filtered state — this is the property the no-lookahead
canary in the test-suite verifies.

Fitting itself (Baum-Welch) does see the whole training sample, which is correct:
parameter estimation is an offline, train-set operation. Only *inference* of the
per-timestamp state must be causal, and that is what :meth:`predict_pit` provides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from qedge.config import QedgeConfig, get_config

# Trading days per year — used only to keep realized-vol annualisation explicit;
# clustering is rank/quantile based so the constant never enters a threshold.
_TRADING_DAYS_PER_YEAR: int = 252


def _as_matrix(X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
    """Coerce a feature input to a 2-D float64 ndarray (no copy when possible)."""
    values: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("feature matrix X must be 1-D or 2-D")
    return values


def _index_of(X: pd.DataFrame | NDArray[np.float64], n: int) -> pd.Index:
    """Return X's row index if it is a DataFrame, else a default RangeIndex."""
    if isinstance(X, pd.DataFrame):
        return X.index
    return pd.RangeIndex(n)


class HMMRegimeDetector:
    """Gaussian-HMM regime detector with point-in-time (filtered) inference.

    Args:
        n_states: Number of latent regimes. Defaults to ``cfg.regime.hmm_n_states``.
        covariance_type: hmmlearn covariance structure. Defaults to
            ``cfg.regime.hmm_covariance_type``.
        n_iter: Baum-Welch iterations. Defaults to ``cfg.regime.hmm_n_iter``.
        config: Optional config override (defaults to the process singleton).

    The estimator is deterministic: ``random_state`` is pinned to the scanner seed
    so a given training matrix always recovers the same parameters.
    """

    def __init__(
        self,
        n_states: int | None = None,
        covariance_type: str | None = None,
        n_iter: int | None = None,
        *,
        config: QedgeConfig | None = None,
    ) -> None:
        self._cfg: QedgeConfig = config if config is not None else get_config()
        self.n_states: int = (
            n_states if n_states is not None else self._cfg.regime.hmm_n_states
        )
        self.covariance_type: str = (
            covariance_type
            if covariance_type is not None
            else self._cfg.regime.hmm_covariance_type
        )
        self.n_iter: int = n_iter if n_iter is not None else self._cfg.regime.hmm_n_iter
        self._model: GaussianHMM | None = None

    @property
    def model(self) -> GaussianHMM:
        """The fitted underlying :class:`GaussianHMM` (raises if not yet fit)."""
        if self._model is None:
            raise RuntimeError("HMMRegimeDetector.fit must be called before inference")
        return self._model

    def fit(self, X: pd.DataFrame | NDArray[np.float64]) -> HMMRegimeDetector:
        """Fit the Gaussian HMM on the full training matrix (offline Baum-Welch).

        Returns ``self`` to allow chaining. Fitting sees the whole training sample
        by design; only per-timestamp *inference* is constrained to be causal.
        """
        matrix = _as_matrix(X)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self._cfg.scanner.seed,
        )
        model.fit(matrix)
        self._model = model
        return self

    def _filtered_log_alpha(
        self, matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Run the forward recursion, returning normalised filtered log-probs.

        Implements, in log-space for numerical stability::

            alpha_1(j)  ∝ startprob(j) * b_j(o_1)
            alpha_t(j)  ∝ [ Σ_i alpha_{t-1}(i) * A(i, j) ] * b_j(o_t)

        normalising alpha_t to sum to 1 at every step so the result is exactly the
        filtered posterior ``P(state_t = j | o_1..t)``. Crucially the value at row
        ``t`` is a function of rows ``0..t`` ONLY — no backward pass, no future.
        """
        model = self.model
        n_obs = matrix.shape[0]
        log_startprob = np.log(model.startprob_)
        log_transmat = np.log(model.transmat_)
        # Per-observation emission log-likelihoods b_j(o_t): shape (n_obs, n_states).
        framelogprob: NDArray[np.float64] = model._compute_log_likelihood(matrix)

        log_alpha = np.empty((n_obs, self.n_states), dtype=np.float64)
        # t = 0: prior * emission, then normalise.
        first = log_startprob + framelogprob[0]
        log_alpha[0] = first - logsumexp(first)
        # t >= 1: propagate through the transition matrix, then normalise.
        for t in range(1, n_obs):
            # work[i, j] = log_alpha[t-1, i] + log A(i, j)
            work = log_alpha[t - 1][:, np.newaxis] + log_transmat
            predicted = logsumexp(work, axis=0)  # over previous state i -> shape (n_states,)
            unnormalised = predicted + framelogprob[t]
            log_alpha[t] = unnormalised - logsumexp(unnormalised)
        return log_alpha

    def filtered_proba(
        self, X: pd.DataFrame | NDArray[np.float64]
    ) -> pd.DataFrame:
        """Return filtered state probabilities ``P(state_t | obs_1..t)`` per row.

        Columns are the integer state ids ``0..n_states-1``; the index mirrors X.
        """
        matrix = _as_matrix(X)
        log_alpha = self._filtered_log_alpha(matrix)
        proba = np.exp(log_alpha)
        return pd.DataFrame(
            proba,
            index=_index_of(X, matrix.shape[0]),
            columns=pd.RangeIndex(self.n_states),
        )

    def predict_pit(
        self, X: pd.DataFrame | NDArray[np.float64]
    ) -> pd.Series:
        """Return the FILTERED most-likely state at each ``t`` (data ``<= t`` only).

        This is the point-in-time regime label. It deliberately does NOT call the
        hmmlearn ``predict``/``predict_proba``/``score_samples`` smoothing routines,
        which would leak the future into ``t``'s label.
        """
        matrix = _as_matrix(X)
        log_alpha = self._filtered_log_alpha(matrix)
        states = np.argmax(log_alpha, axis=1).astype(np.int64)
        return pd.Series(
            states,
            index=_index_of(X, matrix.shape[0]),
            name="regime",
        )


class ChangePointDetector:
    """Offline change-point detection on a 1-D series via :mod:`ruptures` (PELT).

    Args:
        penalty: PELT penalty controlling segmentation granularity. Defaults to
            ``cfg.regime.changepoint_penalty``.
        config: Optional config override (defaults to the process singleton).
    """

    def __init__(
        self,
        penalty: float | None = None,
        *,
        config: QedgeConfig | None = None,
    ) -> None:
        self._cfg: QedgeConfig = config if config is not None else get_config()
        self.penalty: float = (
            penalty if penalty is not None else self._cfg.regime.changepoint_penalty
        )

    def detect(self, series: pd.Series | NDArray[np.float64]) -> list[int]:
        """Return breakpoint indices (exclusive segment ends) in ``series``.

        Uses PELT with an RBF cost, which detects shifts in mean and/or variance.
        The returned list is ruptures' convention: each entry is the index one past
        the end of a segment, with the final entry equal to ``len(series)``.
        """
        values = np.asarray(series, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("ChangePointDetector.detect expects a 1-D series")
        signal = values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(signal)
        breakpoints: list[int] = algo.predict(pen=self.penalty)
        return breakpoints


class VolCorrClusterDetector:
    """Cluster days into regimes by trailing realized vol (and optional correlation).

    The clustering is point-in-time: each day's regime feature is computed from a
    *trailing* window ending at that day, so no future bar can change a past label.
    The cluster assignment itself uses KMeans on those trailing features; KMeans is
    a transductive snapshot model fit on the trailing-feature panel (which is itself
    causal per row), and is seeded for determinism.

    Args:
        n_clusters: Number of vol regimes. Defaults to ``cfg.regime.hmm_n_states``.
        lookback_days: Trailing window for realized vol / correlation. Defaults to
            ``cfg.regime.regime_lookback_days``.
        vol_window: Short trailing window for the per-day realized-vol feature.
            Defaults to ``cfg.labeling.short_vol_span``.
        config: Optional config override (defaults to the process singleton).
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        *,
        lookback_days: int | None = None,
        vol_window: int | None = None,
        config: QedgeConfig | None = None,
    ) -> None:
        self._cfg: QedgeConfig = config if config is not None else get_config()
        self.n_clusters: int = (
            n_clusters if n_clusters is not None else self._cfg.regime.hmm_n_states
        )
        self.lookback_days: int = (
            lookback_days
            if lookback_days is not None
            else self._cfg.regime.regime_lookback_days
        )
        self.vol_window: int = (
            vol_window if vol_window is not None else self._cfg.labeling.short_vol_span
        )

    def _trailing_features(
        self,
        returns: pd.Series,
        cross_returns: pd.Series | None,
    ) -> pd.DataFrame:
        """Build the trailing (causal) feature panel: realized vol [+ correlation].

        Realized vol uses ``min_periods=2`` so an estimate exists as early as the
        window fills; rows without enough history are dropped before clustering.
        """
        rolling = returns.rolling(window=self.vol_window, min_periods=2)
        realized_vol = rolling.std(ddof=0) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        features = pd.DataFrame({"realized_vol": realized_vol})
        if cross_returns is not None:
            aligned = cross_returns.reindex(returns.index)
            corr = returns.rolling(
                window=self.lookback_days, min_periods=2
            ).corr(aligned)
            features["cross_corr"] = corr
        return features

    def fit_predict(
        self,
        returns: pd.Series,
        cross_returns: pd.Series | None = None,
    ) -> pd.Series:
        """Assign each day a vol regime label (higher label == higher vol cluster).

        Labels are remapped so that the cluster with the lowest mean realized vol is
        ``0`` and the highest is ``n_clusters - 1``, making the ordering meaningful
        and stable across runs. Days lacking enough trailing history are labelled
        ``-1`` (insufficient data).
        """
        features = self._trailing_features(returns, cross_returns)
        valid = features.dropna()
        labels = pd.Series(
            np.full(len(features), -1, dtype=np.int64),
            index=features.index,
            name="vol_regime",
        )
        if valid.empty:
            return labels

        effective_clusters = int(min(self.n_clusters, len(valid)))
        if effective_clusters <= 1:
            labels.loc[valid.index] = 0
            return labels

        kmeans = KMeans(
            n_clusters=effective_clusters,
            random_state=self._cfg.scanner.seed,
            n_init=10,
        )
        raw = kmeans.fit_predict(valid.to_numpy(dtype=np.float64))

        # Remap raw cluster ids -> rank by mean realized vol (ascending) so the
        # label ordering encodes "how stressed", not arbitrary KMeans ids.
        vol_by_cluster = (
            valid.assign(_raw=raw).groupby("_raw")["realized_vol"].mean()
        )
        order = vol_by_cluster.sort_values().index.to_numpy()
        remap = {int(old): new for new, old in enumerate(order)}
        ranked = np.array([remap[int(r)] for r in raw], dtype=np.int64)
        labels.loc[valid.index] = ranked
        return labels


@dataclass
class RegimeModel:
    """Convenience bundle of the three regime detectors sharing one config.

    Construct with :meth:`from_config` to wire every detector to the same
    :class:`QedgeConfig`; the individual detectors remain independently usable.
    """

    hmm: HMMRegimeDetector
    changepoint: ChangePointDetector
    vol_cluster: VolCorrClusterDetector

    @classmethod
    def from_config(cls, config: QedgeConfig | None = None) -> RegimeModel:
        """Build a :class:`RegimeModel` with all detectors bound to ``config``."""
        cfg = config if config is not None else get_config()
        return cls(
            hmm=HMMRegimeDetector(config=cfg),
            changepoint=ChangePointDetector(config=cfg),
            vol_cluster=VolCorrClusterDetector(config=cfg),
        )
