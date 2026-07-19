"""
Representational Sufficiency Condition

Checks whether all demographic groups are adequately represented in the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class RepresentationGap:
    """Result of representation analysis for a demographic group."""
    group_name: str
    observed_pct: float
    expected_pct: float
    gap_pct: float  # observed - expected (percentage points)
    gap_relative: float  # (observed - expected) / expected
    n_observed: int
    confidence_interval: Tuple[float, float]  # 95% CI
    passes: bool  # Gap within acceptable threshold
    threshold: float = 0.10  # 10 percentage point threshold


def compute_representation_gaps(
    df: pd.DataFrame,
    demographic_col: str,
    expected_distribution: Optional[Dict[str, float]] = None,
    threshold: float = 0.10
) -> Dict[str, RepresentationGap]:
    """
    Compute representation gaps for each demographic group.

    Args:
        df: Input dataframe
        demographic_col: Column name for demographic grouping (e.g., 'race')
        expected_distribution: Dict mapping group → expected proportion
                              If None, uses uniform distribution
        threshold: Max allowed gap in percentage points (default 0.10)

    Returns:
        Dict mapping group name → RepresentationGap results

    Example:
        >>> df = pd.DataFrame({'race': ['Black']*300 + ['White']*700})
        >>> expected = {'Black': 0.13, 'White': 0.87}
        >>> gaps = compute_representation_gaps(df, 'race', expected)
        >>> gaps['Black'].gap_pct  # -7.0 (observed 6%, expected 13%)
        >>> gaps['Black'].passes    # False (gap > threshold)
    """

    total = len(df)
    observed_counts = df[demographic_col].value_counts()
    observed_pcts = observed_counts / total

    # Default to uniform distribution if not provided
    if expected_distribution is None:
        n_groups = df[demographic_col].nunique()
        expected_distribution = {group: 1/n_groups for group in observed_counts.index}

    results = {}

    for group in observed_counts.index:
        observed_pct = observed_pcts[group]
        expected_pct = expected_distribution.get(group, 1/len(expected_distribution))

        gap_pct = (observed_pct - expected_pct) * 100  # Convert to percentage points
        gap_relative = (observed_pct - expected_pct) / expected_pct if expected_pct > 0 else 0

        # Compute 95% confidence interval using binomial proportion
        n = observed_counts[group]
        ci = stats.binom.interval(0.95, total, observed_pct)
        ci_pcts = (ci[0]/total * 100, ci[1]/total * 100)

        passes = abs(gap_pct) <= threshold

        results[group] = RepresentationGap(
            group_name=group,
            observed_pct=observed_pct * 100,
            expected_pct=expected_pct * 100,
            gap_pct=gap_pct,
            gap_relative=gap_relative,
            n_observed=n,
            confidence_interval=ci_pcts,
            passes=passes,
            threshold=threshold
        )

    return results


def representation_report(gaps: Dict[str, RepresentationGap]) -> str:
    """
    Generate human-readable representation report.

    Args:
        gaps: Dict of RepresentationGap results

    Returns:
        Formatted report string
    """

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("REPRESENTATIONAL SUFFICIENCY CONDITION")
    report_lines.append("=" * 60)

    all_pass = all(gap.passes for gap in gaps.values())
    report_lines.append(f"\nOverall Status: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    report_lines.append("\nPer-Group Analysis:")
    report_lines.append("-" * 60)

    for group_name, gap in gaps.items():
        status = "✓ PASS" if gap.passes else "✗ FAIL"
        report_lines.append(f"\n{group_name}: {status}")
        report_lines.append(f"  Observed:    {gap.observed_pct:.1f}%")
        report_lines.append(f"  Expected:    {gap.expected_pct:.1f}%")
        report_lines.append(f"  Gap:         {gap.gap_pct:+.1f} pp ({gap.gap_relative:+.1%})")
        report_lines.append(f"  N:           {gap.n_observed:,}")
        report_lines.append(f"  95% CI:      {gap.confidence_interval[0]:.1f}% - {gap.confidence_interval[1]:.1f}%")

    report_lines.append("\n" + "=" * 60)

    # Summary
    failed_groups = [g for g in gaps.values() if not g.passes]
    if failed_groups:
        report_lines.append(f"\n{len(failed_groups)} group(s) fail representational sufficiency:")
        for gap in failed_groups:
            if gap.gap_relative < 0:
                report_lines.append(f"  • {gap.group_name}: UNDERREPRESENTED ({gap.gap_relative:.1%})")
            else:
                report_lines.append(f"  • {gap.group_name}: OVERREPRESENTED ({gap.gap_relative:+.1%})")

    return "\n".join(report_lines)


def statistical_bounds(
    counts: Dict[str, int],
    total: int,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for group proportions.

    Args:
        counts: Dict mapping group → count
        total: Total number of observations
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dict mapping group → (lower_bound, upper_bound) proportions
    """

    results = {}
    for group, count in counts.items():
        proportion = count / total
        ci = stats.binom.interval(1 - alpha, total, proportion)
        results[group] = (ci[0]/total, ci[1]/total)

    return results
