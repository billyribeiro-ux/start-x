"""Tests for the Evidence Miner: ranking math on synthetic data + a network-gated smoke.

The synthetic case constructs an evidence frame where ONE feature cleanly separates up- from
down-moves and ANOTHER is pure noise, then asserts ``rank_conditions`` puts the separator on
top (AUC far from 0.5, high |corr|) and the noise feature near 0.5. We also check coverage /
mean computations and the catalyst tally. The smoke (skipped without an FMP key) verifies the
real PIT pipeline returns moves with feature snapshots and no all-NaN key columns.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from startx.events.evidence import (
    catalyst_match_ranking,
    collect_move_evidence,
    rank_conditions,
)


def _synthetic_evidence(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Half up / half down moves; ``sep`` separates them, ``noise`` does not."""
    rng = np.random.default_rng(seed)
    half = n // 2
    direction = np.array(["up"] * half + ["down"] * (n - half))
    # `sep`: up-moves drawn high, down-moves drawn low, non-overlapping -> AUC ~ 1.0.
    sep = np.concatenate([rng.uniform(10.0, 11.0, half), rng.uniform(0.0, 1.0, n - half)])
    # `noise`: identical distribution for both classes -> AUC ~ 0.5.
    noise = rng.normal(0.0, 1.0, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "direction": direction,
        "abs_ar_pct": rng.uniform(3.0, 8.0, n),
        "sep": sep,
        "noise": noise,
        # catalyst presence flags: earnings present on every up-move, news on every down-move.
        "cat_earnings": np.where(direction == "up", 1, 0),
        "cat_news": np.where(direction == "down", 1, 0),
        "top_catalyst": np.where(direction == "up", "earnings", "news"),
    })
    return df


def test_separating_feature_ranks_above_noise():
    ev = _synthetic_evidence()
    ranking = rank_conditions(ev)
    assert set(ranking) == {"up", "down"}

    up = ranking["up"].set_index("feature")
    # The clean separator should be the top-ranked condition for up-moves.
    assert ranking["up"].iloc[0]["feature"] == "sep"
    # Up-moves have HIGH sep -> AUC near 1.0; noise near 0.5.
    assert up.loc["sep", "single_feature_auc"] > 0.95
    assert abs(up.loc["noise", "single_feature_auc"] - 0.5) < 0.2
    # |point-biserial| strong for sep, weak for noise.
    assert abs(up.loc["sep", "point_biserial_corr"]) > 0.8
    assert abs(up.loc["noise", "point_biserial_corr"]) < 0.3
    # Strength ordering: separator strictly beats noise on AUC-distance-from-0.5.
    sep_strength = abs(up.loc["sep", "single_feature_auc"] - 0.5)
    noise_strength = abs(up.loc["noise", "single_feature_auc"] - 0.5)
    assert sep_strength > noise_strength


def test_down_direction_is_mirror_of_up():
    ev = _synthetic_evidence()
    ranking = rank_conditions(ev)
    down = ranking["down"].set_index("feature")
    # Down-moves have LOW sep -> AUC near 0.0 (still far from 0.5 => ranked top).
    assert ranking["down"].iloc[0]["feature"] == "sep"
    assert down.loc["sep", "single_feature_auc"] < 0.05
    # mean_at_move (down class) should be lower than mean_baseline (up class) for `sep`.
    assert down.loc["sep", "mean_at_move"] < down.loc["sep", "mean_baseline"]


def test_coverage_and_mean_computation():
    ev = _synthetic_evidence(n=40)
    # Knock out 25% of `sep` -> coverage must reflect the non-null fraction.
    ev.loc[ev.index[:10], "sep"] = np.nan
    ranking = rank_conditions(ev, min_coverage=0.05)
    up = ranking["up"].set_index("feature")
    assert up.loc["sep", "coverage"] == pytest.approx(0.75, abs=1e-9)

    # mean_at_move / mean_baseline computed only over valid rows of that feature.
    valid = ev.dropna(subset=["sep"])
    up_mean = valid.loc[valid["direction"] == "up", "sep"].mean()
    down_mean = valid.loc[valid["direction"] == "down", "sep"].mean()
    assert up.loc["sep", "mean_at_move"] == pytest.approx(up_mean, abs=1e-9)
    assert up.loc["sep", "mean_baseline"] == pytest.approx(down_mean, abs=1e-9)


def test_min_coverage_filters_sparse_features():
    ev = _synthetic_evidence(n=40)
    ev["sparse"] = np.nan
    ev.loc[ev.index[:1], "sparse"] = 1.0  # 2.5% coverage
    ranking = rank_conditions(ev, min_coverage=0.05)
    assert "sparse" not in set(ranking["up"]["feature"])
    assert "sep" in set(ranking["up"]["feature"])


def test_catalyst_match_ranking_tally():
    ev = _synthetic_evidence(n=60)
    cat = catalyst_match_ranking(ev).set_index("catalyst")
    n_up = int((ev["direction"] == "up").sum())
    n_down = int((ev["direction"] == "down").sum())

    # earnings present on every up-move only; news on every down-move only.
    assert cat.loc["earnings", "n_up"] == n_up
    assert cat.loc["earnings", "n_down"] == 0
    assert cat.loc["earnings", "freq_up"] == pytest.approx(1.0)
    assert cat.loc["earnings", "freq_down"] == pytest.approx(0.0)
    assert cat.loc["earnings", "dir_hit_rate"] == pytest.approx(1.0)  # all-present moves are up

    assert cat.loc["news", "n_down"] == n_down
    assert cat.loc["news", "n_up"] == 0
    assert cat.loc["news", "dir_hit_rate"] == pytest.approx(0.0)  # all-present moves are down


def test_empty_inputs_are_safe():
    empty = pd.DataFrame(columns=["date", "symbol", "direction", "abs_ar_pct"])
    ranking = rank_conditions(empty)
    assert ranking["up"].empty and ranking["down"].empty
    assert catalyst_match_ranking(empty).empty


# --------------------------------------------------------------------------
# Network-gated smoke: exercises the real PIT collection path against live FMP.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="needs live FMP_API_KEY")
def test_collect_move_evidence_smoke(capsys):
    from startx.data.cache import ParquetCache
    from startx.data.universe import load_universe
    from startx.fmp.client import FMPClient
    from startx.settings import get_settings

    settings = get_settings()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()
    with FMPClient(settings) as client:
        ev = collect_move_evidence(
            ["NVDA"], "2023-01-01", "2024-06-01",
            client=client, cache=cache, universe=universe, settings=settings,
        )
    assert not ev.empty, "expected >0 significant moves for NVDA over 18 months"
    for col in ("date", "symbol", "direction", "abs_ar_pct"):
        assert col in ev.columns
        assert ev[col].notna().any(), f"key column {col} is all-NaN"
    assert (ev["direction"].isin(["up", "down"])).all()

    # There must be PIT feature snapshot columns beyond the id/catalyst columns, not all-NaN.
    feat_cols = [c for c in ev.columns
                 if c not in ("date", "symbol", "direction", "abs_ar_pct", "ar_pct", "ar_z",
                              "vol_z", "top_catalyst") and not c.startswith("cat_")]
    assert feat_cols, "no feature snapshot columns present"
    non_null_feats = [c for c in feat_cols if ev[c].notna().any()]
    assert non_null_feats, "all feature snapshot columns are NaN-only"

    ranking = rank_conditions(ev)
    print(f"\n[smoke] NVDA moves analyzed: {len(ev)} "
          f"(up={int((ev['direction']=='up').sum())}, "
          f"down={int((ev['direction']=='down').sum())})")
    print("[smoke] TOP-5 UP conditions:")
    print(ranking["up"].head(5).to_string(index=False))
    print("[smoke] TOP-5 DOWN conditions:")
    print(ranking["down"].head(5).to_string(index=False))
    print("[smoke] catalyst match ranking:")
    print(catalyst_match_ranking(ev).to_string(index=False))
