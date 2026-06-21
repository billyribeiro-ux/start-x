"""Feature engineering for qedge — memory-preserving, leakage-hardened transforms.

The first member is fractional differentiation (Lopez de Prado FFD), which finds
the smallest differencing order that makes a price series stationary while
retaining the maximum amount of its memory. Its fixed-width backward window is an
explicit point-in-time guarantee, declared via the feature's
:class:`~qedge.data.boundary.InformationBoundary`.
"""
from __future__ import annotations

from qedge.features.fracdiff import (
    FRACDIFF_BOUNDARY,
    ffd_weights,
    frac_diff_ffd,
    fracdiff_boundary,
    min_d_adf,
)

__all__ = [
    "FRACDIFF_BOUNDARY",
    "ffd_weights",
    "frac_diff_ffd",
    "fracdiff_boundary",
    "min_d_adf",
]
