"""Feature engineering layer: point-in-time features for the swing-trading model.

Every feature here is computed from information available by day ``t``'s close. Forward-looking
columns (``fwd_ret_*``) produced upstream for descriptive event studies are deliberately never
propagated into any feature frame.
"""
from __future__ import annotations
