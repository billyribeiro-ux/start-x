"""Tests for the feature framework: registry behaviour and the leakage canary.

The headline test is the REGISTRY-ENFORCED NO-LOOKAHEAD CANARY: it parametrizes
over EVERY spec currently in :data:`qedge.features.REGISTRY`, computes each on a
synthetic frame truncated at ``T``, appends deliberately poisoned future rows
(huge and NaN values dated after ``T``), recomputes on the extended frame, and
asserts that every value at an index ``<= T`` is byte-identical. A leak in any
registered feature — present or added later — fails here automatically.

Importing :mod:`qedge.features.technical` is what populates the registry, so the
canary covers the full technical feature set without enumerating it by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qedge.features.technical  # noqa: F401  (import populates REGISTRY)

# A boundary value taken from the existing fracdiff feature's style: TRAILING,
# point statistic. Used only by the unit tests of the registry itself.
from qedge.data.boundary import BoundaryKind, InformationBoundary
from qedge.data.synthetic import SyntheticMarket
from qedge.features.base import (
    REGISTRY,
    FeatureRegistry,
    FeatureSpec,
    compute_features,
)

_DUMMY_BOUNDARY = InformationBoundary(
    kind=BoundaryKind.TRAILING,
    lookback_days=0,
    latency_days=0,
    description="test-only point statistic",
)

#: How many trading bars to generate before the truncation point T.
_PRE_BARS = 400
#: How many poisoned future rows to append after T.
_POISON_BARS = 60


def _synthetic_frame(n: int) -> pd.DataFrame:
    """An OHLCV frame on a DatetimeIndex from the deterministic synthetic market."""
    market = SyntheticMarket("2019-01-01", "2024-12-31", symbols=("SPY",))
    end = pd.Timestamp("2024-12-31")
    frame = market.history("SPY", asof=end).head(n).copy()
    frame = frame.set_index("date")
    return frame


# --------------------------------------------------------------------------- #
# FeatureSpec / FeatureRegistry behaviour
# --------------------------------------------------------------------------- #
def _const_fn(df: pd.DataFrame) -> pd.Series[float]:
    return pd.Series(np.ones(len(df), dtype="float64"), index=df.index)


def test_register_and_retrieve() -> None:
    reg = FeatureRegistry()
    spec = FeatureSpec(name="ones", fn=_const_fn, boundary=_DUMMY_BOUNDARY)
    returned = reg.register(spec)
    assert returned is spec
    assert "ones" in reg
    assert reg.get("ones") is spec
    assert reg.all_specs() == [spec]
    assert len(reg) == 1


def test_duplicate_name_raises() -> None:
    reg = FeatureRegistry()
    reg.register(FeatureSpec(name="x", fn=_const_fn, boundary=_DUMMY_BOUNDARY))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(FeatureSpec(name="x", fn=_const_fn, boundary=_DUMMY_BOUNDARY))


def test_decorator_registers_and_returns_unchanged() -> None:
    reg = FeatureRegistry()

    @reg.feature("dec", _DUMMY_BOUNDARY)
    def my_feature(df: pd.DataFrame) -> pd.Series[float]:
        return _const_fn(df)

    # Decorated function is returned unchanged and still directly callable.
    frame = _synthetic_frame(10)
    out = my_feature(frame)
    assert out.tolist() == [1.0] * 10
    assert "dec" in reg
    assert reg.get("dec").fn is my_feature


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FeatureSpec(name="", fn=_const_fn, boundary=_DUMMY_BOUNDARY)


def test_all_specs_preserves_registration_order() -> None:
    reg = FeatureRegistry()
    for name in ("a", "b", "c"):
        reg.register(FeatureSpec(name=name, fn=_const_fn, boundary=_DUMMY_BOUNDARY))
    assert [s.name for s in reg.all_specs()] == ["a", "b", "c"]


def test_compute_features_aligns_and_names_columns() -> None:
    reg = FeatureRegistry()
    reg.register(FeatureSpec(name="ones", fn=_const_fn, boundary=_DUMMY_BOUNDARY))
    frame = _synthetic_frame(20)
    out = compute_features(frame, reg.all_specs())
    assert list(out.columns) == ["ones"]
    assert out.index.equals(frame.index)
    assert (out["ones"] == 1.0).all()


def test_compute_features_empty_specs_returns_index_only() -> None:
    frame = _synthetic_frame(5)
    out = compute_features(frame, [])
    assert out.shape == (5, 0)
    assert out.index.equals(frame.index)


# --------------------------------------------------------------------------- #
# Every registered spec must carry a non-null InformationBoundary
# --------------------------------------------------------------------------- #
def test_registry_is_populated() -> None:
    # The technical module import must have registered features.
    assert len(REGISTRY) > 0


@pytest.mark.parametrize("spec", REGISTRY.all_specs(), ids=lambda s: s.name)
def test_every_spec_has_a_boundary(spec: FeatureSpec) -> None:
    assert spec.boundary is not None
    assert isinstance(spec.boundary, InformationBoundary)
    # The boundary self-validates its bounds in __post_init__; re-state intent.
    assert spec.boundary.lookback_days >= 0
    assert spec.boundary.latency_days >= 0


# --------------------------------------------------------------------------- #
# REGISTRY-ENFORCED NO-LOOKAHEAD CANARY (critical)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", REGISTRY.all_specs(), ids=lambda s: s.name)
def test_no_lookahead_canary_every_registered_spec(spec: FeatureSpec) -> None:
    """Poisoning rows after T must leave every value at index <= T byte-identical.

    Each registered feature is computed on a frame truncated at T, then on the
    same frame with ``_POISON_BARS`` extreme/NaN rows appended after T. Because
    every feature is trailing/shifted, the historical slice must match bit for
    bit (NaN-aware). Any look-ahead in a registered feature surfaces here.
    """
    base = _synthetic_frame(_PRE_BARS)
    out_base = spec.fn(base)
    assert out_base.index.equals(base.index)

    # Build poisoned future rows dated strictly after T with deliberately
    # destabilising values (huge magnitudes interleaved with NaNs) in every
    # OHLCV column — anything that reads them would corrupt the history.
    future_idx = pd.date_range(
        base.index[-1] + pd.offsets.BDay(1), periods=_POISON_BARS, freq="B"
    )
    poison_values = np.full(_POISON_BARS, 1e12, dtype="float64")
    poison_values[1::2] = np.nan
    poison = pd.DataFrame(
        {
            "open": poison_values,
            "high": poison_values,
            "low": poison_values,
            "close": poison_values,
            "volume": poison_values,
        },
        index=future_idx,
    )
    extended = pd.concat([base, poison])
    out_extended = spec.fn(extended)

    historical = out_extended.loc[base.index]
    pd.testing.assert_series_equal(historical, out_base, check_names=False)
    # Explicit NaN-aware byte equality on the underlying arrays.
    np.testing.assert_array_equal(historical.to_numpy(), out_base.to_numpy())


def test_compute_features_canary_on_full_registry() -> None:
    """The same canary, but through compute_features over the whole registry."""
    base = _synthetic_frame(_PRE_BARS)
    specs = REGISTRY.all_specs()
    feats_base = compute_features(base, specs)

    future_idx = pd.date_range(
        base.index[-1] + pd.offsets.BDay(1), periods=_POISON_BARS, freq="B"
    )
    poison_values = np.full(_POISON_BARS, 1e12, dtype="float64")
    poison_values[1::2] = np.nan
    poison = pd.DataFrame(
        {col: poison_values for col in ("open", "high", "low", "close", "volume")},
        index=future_idx,
    )
    extended = pd.concat([base, poison])
    feats_extended = compute_features(extended, specs)

    historical = feats_extended.loc[base.index]
    pd.testing.assert_frame_equal(historical, feats_base)
