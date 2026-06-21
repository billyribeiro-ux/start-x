"""qedge modeling layer — importance, regime, online learning, ensembling.

Every component obeys the out-of-sample discipline: feature importance is
measured on OOS folds (MDA / clustered-SHAP, never in-sample MDI); regime state
is assigned with point-in-time information only (forward-filtering, never
smoothing); online learners adapt with explicit concept-drift detection.
"""
from __future__ import annotations
