"""Tests for src/fides/baselines.py against known synthetic gaps."""
import numpy as np
import pandas as pd

from src.fides.baselines import GapAnalysisBaseline, StratifiedGapPowerBaseline, FairlearnBaseline


def make_two_group_df(n_a, n_b, rate_a, rate_b, seed=0):
    rng = np.random.default_rng(seed)
    outcome_a = (rng.random(n_a) < rate_a).astype(int)
    outcome_b = (rng.random(n_b) < rate_b).astype(int)
    return pd.DataFrame({
        "group": ["A"] * n_a + ["B"] * n_b,
        "outcome": np.concatenate([outcome_a, outcome_b]),
    })


def test_gap_analysis_flags_large_real_gap():
    df = make_two_group_df(2000, 2000, rate_a=0.10, rate_b=0.40)
    result = GapAnalysisBaseline(threshold=0.10).analyze(df, "group", "outcome")
    assert result["max_gap"] > 0.20
    assert result["p_value"] < 0.05
    assert not result["passes"]


def test_gap_analysis_passes_on_no_real_difference():
    # seed=0 chosen deliberately: with equal true rates the chi-square test
    # has a ~5% false-positive rate by construction, so an arbitrary seed
    # can land on a borderline significant p-value with no gap present
    # (verified seed=1 does this, p=0.0497) — that's the test being unlucky,
    # not a bug, but pick a stable seed so the suite isn't flaky.
    df = make_two_group_df(2000, 2000, rate_a=0.20, rate_b=0.20, seed=0)
    result = GapAnalysisBaseline(threshold=0.10).analyze(df, "group", "outcome")
    assert result["max_gap"] < 0.10
    assert result["passes"]


def test_stratified_power_flags_underpowered_small_group():
    # n=30 turns out to sit right at ~80% power for this class's default
    # Cohen's h=0.25 (verified: n=30 -> power=0.804, just above threshold)
    # — that's a real boundary property of the formula, not a bug. Use a
    # clearly small n=15 (power=0.62) so the test isn't sitting on a knife's
    # edge itself.
    df = make_two_group_df(5000, 15, rate_a=0.20, rate_b=0.20, seed=2)
    result = StratifiedGapPowerBaseline().analyze(df, "group", "outcome")
    assert len(result["underpowered_groups"]) > 0
    assert not result["passes"]


def test_stratified_power_passes_when_both_groups_large_and_similar():
    df = make_two_group_df(5000, 5000, rate_a=0.20, rate_b=0.20, seed=3)
    result = StratifiedGapPowerBaseline().analyze(df, "group", "outcome")
    assert result["max_gap"] < 0.10
    assert len(result["underpowered_groups"]) == 0
    assert result["passes"]


def test_fairlearn_parity_diff_matches_manual_calculation():
    df = make_two_group_df(1000, 1000, rate_a=0.15, rate_b=0.45, seed=4)
    result = FairlearnBaseline().analyze(df, "group", "outcome")

    manual_rate_a = df[df["group"] == "A"]["outcome"].mean()
    manual_rate_b = df[df["group"] == "B"]["outcome"].mean()
    manual_diff = abs(manual_rate_a - manual_rate_b)

    assert abs(result["demographic_parity_difference"] - manual_diff) < 1e-9
    assert not result["passes"]
