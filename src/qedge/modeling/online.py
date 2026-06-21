"""Online / incremental learning with explicit concept-drift detection.

The desk's offline edges are validated out-of-sample, but a live book faces a
**non-stationary** market: the data-generating process shifts (regime change,
vol breakout, liquidity dry-up). This module provides the streaming primitives
that adapt one sample at a time and that flag *when* the world changed so a
human can intervene.

Two complementary pieces:

* :class:`OnlineClassifier` wraps a `river` incremental pipeline
  (``StandardScaler | LogisticRegression`` by default). Every call sees exactly
  one sample; nothing here ever peeks at a future observation.
* :class:`DriftMonitor` runs `river`'s ADWIN **and** Page-Hinkley detectors in
  parallel over a scalar stream (e.g. the running prediction error) and reports
  drift when *either* fires — ADWIN for distributional change in a bounded
  signal, Page-Hinkley for a cumulative mean shift.

All tunables come from ``cfg.online`` (no magic numbers). Evaluation uses
:func:`prequential_accuracy` — strict test-then-train, the only honest way to
score an online learner without lookahead.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import cast

from river import compose, linear_model, preprocessing
from river.base import Classifier
from river.base.typing import ClfTarget
from river.drift import ADWIN, PageHinkley

from qedge.config import OnlineConfig

# A single streaming feature vector: hashable feature name -> numeric value.
Features = dict[Hashable, float]


def _default_pipeline() -> compose.Pipeline:
    """Build the default incremental pipeline.

    ``StandardScaler`` adapts its running mean/variance online, so logistic
    regression sees standardised inputs without any batch fit. Both stages
    update strictly from the samples seen so far — no lookahead.
    """
    scaler: preprocessing.StandardScaler = preprocessing.StandardScaler()
    model: linear_model.LogisticRegression = linear_model.LogisticRegression()
    pipeline: compose.Pipeline = scaler | model
    return pipeline


class OnlineClassifier:
    """A one-sample-at-a-time binary/multiclass classifier.

    Wraps a `river` classifier (a scaler-plus-model pipeline by default). The
    contract is purely incremental: :meth:`learn_one` ingests a single labelled
    sample and updates the model in place; the predict methods read the current
    state only. There is no batch ``fit`` and no buffering of future data.
    """

    def __init__(self, model: Classifier | None = None) -> None:
        """Create the classifier.

        Parameters
        ----------
        model:
            A pre-built `river` classifier (or pipeline). When ``None`` the
            default ``StandardScaler | LogisticRegression`` pipeline is used.
        """
        self.model: Classifier = model if model is not None else _default_pipeline()

    def learn_one(self, x: Features, y: ClfTarget) -> None:
        """Update the model in place from one labelled sample.

        Mutates ``self.model``; returns ``None`` so callers cannot accidentally
        rely on a fluent-but-mutating API.
        """
        self.model.learn_one(x, y)

    def predict_proba_one(self, x: Features) -> dict[ClfTarget, float]:
        """Return the predicted class-probability map for one sample.

        Before the model has seen any data this is an empty dict (river's
        cold-start behaviour); callers treating a missing class as probability
        zero stay correct.
        """
        return cast("dict[ClfTarget, float]", self.model.predict_proba_one(x))

    def predict_one(self, x: Features) -> ClfTarget | None:
        """Return the single most-likely label for one sample (``None`` cold)."""
        return self.model.predict_one(x)


class DriftMonitor:
    """Explicit concept-drift detection over a scalar stream.

    Runs ADWIN and Page-Hinkley side by side; :meth:`update` feeds one value to
    both and returns ``True`` when *either* flags drift on that step. ADWIN
    watches for a change in the distribution of a bounded signal (e.g. the 0/1
    correctness of each prediction); Page-Hinkley accumulates deviations from
    the running mean and trips on a sustained shift. Using both widens coverage
    of the abrupt-vs-gradual drift spectrum.

    Attributes
    ----------
    drift_detected:
        ``True`` iff the most recent :meth:`update` fired (either detector).
    n_drifts:
        Total number of update steps on which drift was flagged.
    last_detector:
        Name of the detector(s) that fired on the most recent flagged step
        (``"ADWIN"``, ``"PageHinkley"``, or ``"ADWIN+PageHinkley"``); ``None``
        until the first drift.
    """

    _ADWIN_NAME = "ADWIN"
    _PAGE_HINKLEY_NAME = "PageHinkley"

    def __init__(self, cfg: OnlineConfig) -> None:
        """Construct both detectors from ``cfg.online`` tunables.

        Parameters
        ----------
        cfg:
            The ``online`` config section supplying ``adwin_delta``,
            ``page_hinkley_threshold`` and ``page_hinkley_delta``.
        """
        self._adwin: ADWIN = ADWIN(delta=cfg.adwin_delta)
        self._page_hinkley: PageHinkley = PageHinkley(
            threshold=cfg.page_hinkley_threshold,
            delta=cfg.page_hinkley_delta,
        )
        self.drift_detected: bool = False
        self.n_drifts: int = 0
        self.last_detector: str | None = None

    def update(self, value: float) -> bool:
        """Feed one scalar to both detectors; return whether drift fired.

        Both detectors are always updated (no short-circuit) so neither falls
        behind the stream. Sees only the current value — no lookahead.
        """
        self._adwin.update(value)
        self._page_hinkley.update(value)

        adwin_fired: bool = bool(self._adwin.drift_detected)
        page_hinkley_fired: bool = bool(self._page_hinkley.drift_detected)
        fired: bool = adwin_fired or page_hinkley_fired

        self.drift_detected = fired
        if fired:
            self.n_drifts += 1
            fired_names: list[str] = []
            if adwin_fired:
                fired_names.append(self._ADWIN_NAME)
            if page_hinkley_fired:
                fired_names.append(self._PAGE_HINKLEY_NAME)
            self.last_detector = "+".join(fired_names)

        return fired


def prequential_accuracy(
    stream: Iterable[tuple[Features, ClfTarget]],
    model: OnlineClassifier | None = None,
) -> float:
    """Score an online classifier by interleaved test-then-train.

    For each ``(x, y)`` the model is **first** asked to predict (scored against
    the truth) and **then** trained on that same sample. Because every sample is
    predicted before it is learned, the running accuracy is an unbiased,
    no-lookahead estimate of live performance — the standard prequential (a.k.a.
    interleaved test-then-train) protocol.

    Parameters
    ----------
    stream:
        An iterable of ``(features, label)`` pairs, consumed once in order.
    model:
        The classifier to evaluate and train in place. A fresh default
        :class:`OnlineClassifier` is created when ``None``.

    Returns
    -------
    float
        Fraction of samples predicted correctly. ``0.0`` for an empty stream.
    """
    clf: OnlineClassifier = model if model is not None else OnlineClassifier()
    n_seen: int = 0
    n_correct: int = 0
    for x, y in stream:
        prediction: ClfTarget | None = clf.predict_one(x)
        if prediction is not None and prediction == y:
            n_correct += 1
        clf.learn_one(x, y)
        n_seen += 1
    if n_seen == 0:
        return 0.0
    return n_correct / n_seen
