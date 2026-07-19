"""
Tests for src/fides/causal.py using synthetic data with KNOWN ground truth.

These exist because this module previously had a real bug: factorize()
sign inversion made illegitimate_strength return exactly -200% regardless
of input data (confirmed on 11 real MIMIC-IV cohorts before the fix). These
tests would have caught that — they assert the correct DIRECTION and
MAGNITUDE of the computed effect against data constructed so the answer is
known in advance, not just that the function runs without raising.
"""
import numpy as np
import pandas as pd
import pytest

from src.fides.causal import CausalDAG, compute_psce


def make_dag():
    return CausalDAG(
        edges=[
            ("group", "mediator"),
            ("mediator", "outcome"),
            ("group", "outcome"),
        ],
        nodes=["group", "mediator", "outcome"],
    )


def test_requires_binary_protected_attr():
    df = pd.DataFrame({
        "group": ["a", "b", "c"] * 10,
        "mediator": np.random.randn(30),
        "outcome": np.random.randn(30),
    })
    dag = CausalDAG(edges=[("group", "outcome")], nodes=["group", "outcome"])
    with pytest.raises(ValueError, match="binary"):
        compute_psce(df, dag, "group", "outcome")


def test_mediator_fully_explains_gap():
    """
    Construct data where the ENTIRE group->outcome effect flows through the
    mediator: group predicts mediator, mediator predicts outcome, group has
    no other effect on outcome. Adjusting for the mediator should erase
    almost all of the raw effect: illegitimate_strength should be near 0,
    NOT near 1 and NOT a constant like -2.0 (the historical bug).
    """
    rng = np.random.default_rng(42)
    n = 5000
    group = rng.integers(0, 2, n).astype(float)
    mediator = group * 2.0 + rng.normal(0, 0.5, n)
    outcome = mediator * 1.0 + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"group": group, "mediator": mediator, "outcome": outcome})
    dag = make_dag()
    result = compute_psce(df, dag, "group", "outcome", mediators_by_path={0: ["mediator"]})

    assert result["total_effect_significant"] is True
    assert abs(result["total_effect"]) > 0.5  # real, substantial raw gap
    # direct_effect (adjusted for mediator) should be close to zero since
    # the mediator explains the whole effect
    assert abs(result["direct_effect"]) < 0.3
    assert result["illegitimate_strength"] is not None
    assert abs(result["illegitimate_strength"]) < 0.3


def test_mediator_explains_nothing():
    """
    Construct data where group affects outcome DIRECTLY and the mediator is
    pure noise unrelated to either. Adjusting for the mediator should barely
    change the effect: illegitimate_strength should be close to 1.0 (100%
    unexplained), not some unrelated constant.
    """
    rng = np.random.default_rng(7)
    n = 5000
    group = rng.integers(0, 2, n).astype(float)
    mediator = rng.normal(0, 1, n)  # unrelated to group or outcome
    outcome = group * 1.5 + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"group": group, "mediator": mediator, "outcome": outcome})
    dag = make_dag()
    result = compute_psce(df, dag, "group", "outcome", mediators_by_path={0: ["mediator"]})

    assert result["total_effect_significant"] is True
    assert result["illegitimate_strength"] is not None
    assert 0.8 < result["illegitimate_strength"] < 1.2


def test_no_significant_effect_returns_none_not_exploded_ratio():
    """
    The historical bug case: when the raw gap is ~0 (no real group effect),
    illegitimate_strength must be None (undefined), not an exploded ratio
    like 2367% (observed on real data pre-fix) or a suspicious constant
    like -200% (observed on every single disease pre-fix).
    """
    rng = np.random.default_rng(3)
    n = 5000
    group = rng.integers(0, 2, n).astype(float)
    mediator = rng.normal(0, 1, n)
    outcome = rng.normal(0, 1, n)  # no relationship to group at all

    df = pd.DataFrame({"group": group, "mediator": mediator, "outcome": outcome})
    dag = make_dag()
    result = compute_psce(df, dag, "group", "outcome", mediators_by_path={0: ["mediator"]})

    assert result["total_effect_significant"] is False
    assert result["illegitimate_strength"] is None
    assert result["note"] is not None


def test_direction_is_not_sign_flipped():
    """
    Regression test for the exact historical bug: factorize() assigning
    0/1 labels in an order that inverted the sign of the effect relative
    to a manual group-mean-difference calculation. total_effect from
    compute_psce must match manual mean(outcome|group=1) - mean(outcome|group=0)
    in both sign and magnitude.
    """
    rng = np.random.default_rng(11)
    n = 3000
    group = rng.integers(0, 2, n).astype(float)
    outcome = group * (-0.8) + rng.normal(0, 0.3, n)  # group=1 has LOWER outcome
    mediator = rng.normal(0, 1, n)

    df = pd.DataFrame({"group": group, "mediator": mediator, "outcome": outcome})
    dag = make_dag()
    result = compute_psce(df, dag, "group", "outcome", mediators_by_path={0: ["mediator"]})

    manual_diff = df[df["group"] == 1]["outcome"].mean() - df[df["group"] == 0]["outcome"].mean()
    assert manual_diff < 0  # sanity check on the constructed data itself
    assert result["total_effect"] < 0  # must match sign, not be flipped to positive
    assert abs(result["total_effect"] - manual_diff) < 0.1
