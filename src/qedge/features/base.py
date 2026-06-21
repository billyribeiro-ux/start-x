"""Feature framework — the registry, the spec, and the leakage contract.

Every feature in qedge is a PURE function from an OHLCV frame to a
:class:`pandas.Series` aligned to that frame's index, where the value at row
``t`` is computed using ONLY data at rows ``<= t``. A :class:`FeatureSpec`
binds three things that must never drift apart:

* the feature's ``name`` (its column label in a computed feature matrix),
* the pure ``fn`` that produces it, and
* its :class:`~qedge.data.boundary.InformationBoundary` — the explicit,
  inspectable statement of exactly which data the feature is allowed to see.

The :class:`FeatureRegistry` collects specs (via :meth:`FeatureRegistry.register`
or the :meth:`FeatureRegistry.feature` decorator). A module-level singleton
:data:`REGISTRY` is the package-wide collection that downstream code and the
no-lookahead canary test iterate over. :func:`compute_features` applies a list
of specs to a frame, producing the aligned feature matrix.

Only stdlib, pandas and ``qedge.data.boundary`` are imported here.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from qedge.data.boundary import InformationBoundary

__all__ = [
    "REGISTRY",
    "FeatureFn",
    "FeatureRegistry",
    "FeatureSpec",
    "compute_features",
]

#: A feature function: a PURE map from an OHLCV frame to a Series aligned to the
#: frame's index, where the value at row ``t`` uses ONLY rows ``<= t``.
FeatureFn = Callable[[pd.DataFrame], "pd.Series[float]"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """An immutable binding of a feature name, its pure function, and its boundary.

    Attributes:
        name: Unique column label for the feature in a computed matrix.
        fn: The PURE feature function. It takes an OHLCV :class:`pandas.DataFrame`
            and returns a :class:`pandas.Series` aligned to the frame's index,
            using only data at rows ``<= t`` to produce the value at row ``t``.
        boundary: The explicit, non-null
            :class:`~qedge.data.boundary.InformationBoundary` the feature respects.
    """

    name: str
    fn: FeatureFn
    boundary: InformationBoundary

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec.name must be a non-empty string")
        # ``boundary`` is typed non-optional; the registry-wide "every spec has a
        # non-null InformationBoundary" guarantee is asserted in the test-suite.
        if not isinstance(self.boundary, InformationBoundary):
            raise ValueError(f"feature {self.name!r} must declare an InformationBoundary")


class FeatureRegistry:
    """A collection of :class:`FeatureSpec` objects keyed by unique name.

    Use :meth:`register` to add a pre-built spec, or :meth:`feature` as a
    decorator on a feature function. Names must be unique; re-registering a name
    raises :class:`ValueError` rather than silently shadowing an existing edge.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        """Register ``spec`` under its name and return it unchanged.

        Args:
            spec: The fully-formed feature spec to add.

        Returns:
            The same ``spec`` (so callers can keep a reference).

        Raises:
            ValueError: If a spec with the same name is already registered.
        """
        if spec.name in self._specs:
            raise ValueError(f"feature {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def feature(
        self, name: str, boundary: InformationBoundary
    ) -> Callable[[FeatureFn], FeatureFn]:
        """Decorator that registers the wrapped function as a feature.

        The decorated function is returned UNCHANGED (so it stays directly
        callable and unit-testable); a :class:`FeatureSpec` binding it to ``name``
        and ``boundary`` is added to the registry as a side effect.

        Args:
            name: Unique feature name.
            boundary: The feature's information boundary.

        Returns:
            A decorator that registers and returns the feature function as-is.
        """

        def decorator(fn: FeatureFn) -> FeatureFn:
            self.register(FeatureSpec(name=name, fn=fn, boundary=boundary))
            return fn

        return decorator

    def get(self, name: str) -> FeatureSpec:
        """Return the spec registered under ``name``.

        Raises:
            KeyError: If no feature with that name is registered.
        """
        return self._specs[name]

    def all_specs(self) -> list[FeatureSpec]:
        """Return all registered specs in registration order."""
        return list(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs


#: The package-wide feature registry. Importing :mod:`qedge.features.technical`
#: populates it; the no-lookahead canary test iterates over every spec here.
REGISTRY: FeatureRegistry = FeatureRegistry()


def compute_features(
    df: pd.DataFrame, specs: Iterable[FeatureSpec]
) -> pd.DataFrame:
    """Apply ``specs`` to ``df`` and return the aligned feature matrix.

    Each spec's pure function is evaluated on the full ``df`` and its output
    Series is placed in a column named after the spec. Because every feature is
    trailing/shifted (value at ``t`` uses only rows ``<= t``), the resulting
    matrix is point-in-time correct row by row. The output shares ``df``'s index.

    Args:
        df: An OHLCV frame (columns at least ``open/high/low/close/volume``).
        specs: The feature specs to compute.

    Returns:
        A :class:`pandas.DataFrame` indexed by ``df.index`` with one float column
        per spec, in iteration order. Empty (index only) if ``specs`` is empty.
    """
    columns: dict[str, pd.Series[float]] = {}
    for spec in specs:
        series = spec.fn(df)
        columns[spec.name] = series.reindex(df.index)
    return pd.DataFrame(columns, index=df.index)
