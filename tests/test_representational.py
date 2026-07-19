"""Tests for src/fides/representational.py against known distributions."""
import pandas as pd
import pytest

from src.fides.representational import compute_representation_gaps


def test_matches_expected_distribution_passes():
    """If observed matches expected exactly, gap should be ~0 and pass."""
    df = pd.DataFrame({"race": ["White"] * 500 + ["Black"] * 500})
    expected = {"White": 0.5, "Black": 0.5}
    gaps = compute_representation_gaps(df, "race", expected, threshold=0.10)
    assert gaps["White"].passes
    assert gaps["Black"].passes
    assert abs(gaps["White"].gap_pct) < 1.0


def test_underrepresented_group_fails_with_correct_sign():
    """900 White / 100 Black vs. 50/50 expected: Black should show a large
    NEGATIVE gap (underrepresented) and fail; White a large POSITIVE gap."""
    df = pd.DataFrame({"race": ["White"] * 900 + ["Black"] * 100})
    expected = {"White": 0.5, "Black": 0.5}
    gaps = compute_representation_gaps(df, "race", expected, threshold=0.10)

    assert gaps["Black"].gap_pct < 0
    assert not gaps["Black"].passes
    assert gaps["White"].gap_pct > 0
    assert not gaps["White"].passes
    # magnitude should be roughly 40pp each direction
    assert 35 < abs(gaps["Black"].gap_pct) < 45
    assert 35 < abs(gaps["White"].gap_pct) < 45


def test_uniform_default_distribution_used_when_none_given():
    """With 3 equally-sized groups and no expected_distribution, default
    should be uniform (1/3 each) and an exactly-uniform dataset should pass."""
    df = pd.DataFrame({"race": ["A"] * 100 + ["B"] * 100 + ["C"] * 100})
    gaps = compute_representation_gaps(df, "race", expected_distribution=None, threshold=0.10)
    for g in gaps.values():
        assert g.expected_pct == pytest.approx(33.33, abs=0.1)
        assert g.passes


def test_confidence_interval_contains_observed_rate():
    df = pd.DataFrame({"race": ["White"] * 700 + ["Black"] * 300})
    gaps = compute_representation_gaps(df, "race", {"White": 0.5, "Black": 0.5})
    ci = gaps["White"].confidence_interval
    assert ci[0] <= gaps["White"].observed_pct <= ci[1]
