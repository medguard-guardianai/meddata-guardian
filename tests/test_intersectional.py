"""
Tests for src/fides/intersectional.py power calculation.

These exist because this module had a real bug: a missing abs() inverted
statistical power so that a 15,145-patient group (should have ~100% power)
reported 0% power. These tests assert power INCREASES with sample size and
effect size (the two things it should mathematically depend on), which the
buggy signed version violated.
"""
import numpy as np
import pandas as pd
import pytest

from src.fides.intersectional import compute_power_matrix, compute_required_sample_size


def make_df(n, event_rate, group_col_value="A"):
    rng = np.random.default_rng(0)
    outcome = (rng.random(n) < event_rate).astype(int)
    return pd.DataFrame({"group": [group_col_value] * n, "outcome": outcome})


def test_large_group_has_high_power_not_zero():
    """
    Regression test for the exact historical bug: a large group with a
    real, detectable event rate must show HIGH power, not 0%. Pre-fix, a
    15,145-patient group scored exactly 0.00% power due to a sign error.
    """
    df = make_df(n=15000, event_rate=0.20)
    result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
    power = result["('A',)"].power
    assert power > 0.90, f"expected high power for n=15000, got {power}"


def test_small_group_has_low_power():
    df = make_df(n=50, event_rate=0.20)
    result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
    power = result["('A',)"].power
    assert power < 0.80


def test_power_increases_with_sample_size():
    """Power must be monotonically non-decreasing as n grows, holding
    event rate and effect size fixed — this is the basic sanity property
    the sign-bug violated (bigger n produced power->0, not power->1)."""
    powers = []
    for n in [30, 100, 1000, 20000]:
        df = make_df(n=n, event_rate=0.25)
        result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
        powers.append(result["('A',)"].power)

    assert powers == sorted(powers), f"power should increase with n, got {powers}"
    assert powers[-1] > powers[0]


def test_degenerate_zero_event_rate_does_not_crash():
    """baseline_rate == 0 previously would divide by a zero std_error.
    Must return 0.0 power, not NaN/inf, and must not raise."""
    df = pd.DataFrame({"group": ["A"] * 100, "outcome": [0] * 100})
    result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
    power = result["('A',)"].power
    assert power == 0.0
    assert np.isfinite(power)


def test_degenerate_all_events_does_not_crash():
    """baseline_rate == 1 is the same degenerate case from the other side."""
    df = pd.DataFrame({"group": ["A"] * 100, "outcome": [1] * 100})
    result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
    power = result["('A',)"].power
    assert power == 0.0
    assert np.isfinite(power)


def test_tiny_group_below_ten_flagged_too_small():
    df = make_df(n=5, event_rate=0.2)
    result = compute_power_matrix(df, ["group"], "outcome", test_type="chi-squared")
    assert result["('A',)"].power == 0.0
    assert "too small" in result["('A',)"].interpretation.lower()


def test_required_sample_size_rejects_zero_effect():
    with pytest.raises(ValueError, match="nonzero"):
        compute_required_sample_size(baseline_rate=0.2, effect_size=0.0)


def test_required_sample_size_rejects_degenerate_baseline():
    with pytest.raises(ValueError, match="baseline_rate"):
        compute_required_sample_size(baseline_rate=0.0, effect_size=0.2)


def test_required_sample_size_reasonable_for_known_case():
    """A small effect on a moderate baseline rate should require a
    realistic four-to-five-figure sample size, not something absurd from a
    division bug (e.g. negative or single digits)."""
    n = compute_required_sample_size(baseline_rate=0.20, effect_size=0.20, power=0.80)
    assert 100 < n < 100000
