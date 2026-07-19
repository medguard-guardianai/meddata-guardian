"""
Tests for src/fides/phenotypic.py.

Includes regression tests for a real crash risk found during review: qcut
with duplicates='drop' can silently produce fewer bins than requested when
the severity column has many tied values (common for integer counts like
comorbidity counts), which previously crashed because the code still
handed it the original n_bins worth of labels.
"""
import numpy as np
import pandas as pd
import pytest

from src.fides.phenotypic import compute_coverage


def test_zero_variance_column_raises_not_crashes_obscurely():
    df = pd.DataFrame({
        "race": ["White"] * 50 + ["Black"] * 50,
        "severity": [5] * 100,  # no variance at all
    })
    with pytest.raises(ValueError, match="no variance"):
        compute_coverage(df, "race", "severity")


def test_heavy_ties_reduces_bins_without_crashing():
    """Only 3 distinct severity values across 200 rows — requesting 4 bins
    should gracefully degrade to however many bins qcut can actually make,
    not raise a label-mismatch error."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "race": rng.choice(["White", "Black"], 200),
        "severity": rng.integers(0, 3, 200),  # only values 0, 1, 2
    })
    result = compute_coverage(df, "race", "severity", n_bins=4)
    assert len(result) == 2  # one entry per race group
    for cov in result.values():
        assert len(cov.severity_bins) < 4  # fewer bins than requested, not a crash


def test_evenly_distributed_group_has_full_coverage():
    """If a demographic group is spread proportionally across all severity
    bins (matching its overall population share), it should show high
    coverage and pass, since it appears everywhere in the same proportion."""
    rng = np.random.default_rng(2)
    n = 4000
    df = pd.DataFrame({
        "race": rng.choice(["White", "Black"], n, p=[0.7, 0.3]),
        "severity": rng.uniform(0, 100, n),  # independent of race
    })
    result = compute_coverage(df, "race", "severity", n_bins=4, threshold=0.15)
    # both groups appear in every bin roughly at their population proportion
    assert result["White"].passes
    assert result["Black"].passes


def test_group_missing_from_severe_bin_fails():
    """Construct data where Black patients NEVER appear in the top severity
    quartile — this should be flagged as a missing phenotype and fail."""
    rng = np.random.default_rng(3)
    n_white = 700
    n_black = 300
    white_severity = rng.uniform(0, 100, n_white)
    black_severity = rng.uniform(0, 60, n_black)  # never reaches the top quartile

    df = pd.DataFrame({
        "race": ["White"] * n_white + ["Black"] * n_black,
        "severity": np.concatenate([white_severity, black_severity]),
    })
    result = compute_coverage(df, "race", "severity", n_bins=4, threshold=0.10)
    assert not result["Black"].passes
    assert len(result["Black"].missing_phenotypes) > 0
