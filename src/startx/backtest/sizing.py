"""Position-sizing helpers.

Each function returns a *weight* (fraction of capital) or a notional amount to
allocate to a single position. The v1 engine uses :func:`fixed_fraction` with
``frac=1.0`` (full capital per sequential trade); :func:`vol_target` is provided
for volatility-scaled sizing experiments.
"""
from __future__ import annotations


def fixed_fraction(capital: float, frac: float = 1.0) -> float:
    """Notional to allocate as a fixed fraction ``frac`` of ``capital``.

    ``frac`` is clamped to ``[0, 1]`` so a position never exceeds available
    capital (no implicit leverage).
    """
    frac = min(max(frac, 0.0), 1.0)
    return float(capital) * frac


def vol_target(target_vol: float, realized_vol: float, cap: float = 1.0) -> float:
    """Volatility-target position weight ``min(cap, target_vol / realized_vol)``.

    Scales exposure so that ``weight * realized_vol ≈ target_vol``. The weight is
    capped at ``cap`` to bound leverage. Returns ``cap`` when ``realized_vol`` is
    zero or non-finite (no volatility to scale against).
    """
    if realized_vol <= 0 or not _finite(realized_vol):
        return float(cap)
    weight = target_vol / realized_vol
    return float(min(cap, weight))


def _finite(x: float) -> bool:
    """True iff ``x`` is a finite real number."""
    return x == x and x not in (float("inf"), float("-inf"))
