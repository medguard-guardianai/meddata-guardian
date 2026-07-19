"""
Intersectional Sufficiency Condition & Insufficiency Masking Detection

Identifies demographic intersections with insufficient statistical power to
detect bias (insufficiency masking).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class IntersectionPower:
    """Statistical power analysis for a demographic intersection."""
    intersection: str
    subgroup_size: int
    outcome_events: int
    effect_size: float  # e.g., 0.2 for 20% relative difference
    power: float
    passes: bool  # power >= threshold
    threshold: float = 0.80
    interpretation: str = ""


def compute_power_matrix(
    df: pd.DataFrame,
    demographic_cols: List[str],
    outcome_col: str,
    alpha: float = 0.05,
    power_threshold: float = 0.80,
    effect_size: float = 0.20,
    test_type: str = "t-test"
) -> Dict[str, IntersectionPower]:
    """
    Compute statistical power for all demographic intersections.

    Args:
        df: Input dataframe
        demographic_cols: List of demographic columns to intersect
        outcome_col: Binary outcome column (0/1)
        alpha: Significance level (default 0.05)
        power_threshold: Minimum acceptable power (default 0.80)
        effect_size: Expected effect size (default 0.20 for 20% difference)
        test_type: Type of statistical test ('t-test', 'chi-squared')

    Returns:
        Dict mapping intersection → IntersectionPower results

    Example:
        >>> df = pd.DataFrame({
        ...     'race': ['Black']*50 + ['White']*950,
        ...     'sex': ['F']*500 + ['M']*500,
        ...     'outcome': [0,1]*500
        ... })
        >>> power = compute_power_matrix(df, ['race', 'sex'], 'outcome')
        >>> power['Black-Female'].power  # 0.72
        >>> power['Black-Female'].passes # False (< 0.80)
    """

    from statsmodels.stats.power import tt_ind_solve_power

    # Generate all intersections
    intersection_groups = df.groupby(demographic_cols)

    results = {}

    for group_tuple, group_df in intersection_groups:
        # Create intersection name
        if len(demographic_cols) == 1:
            intersection_name = str(group_tuple)
        else:
            intersection_name = "-".join(str(g) for g in group_tuple)

        n = len(group_df)
        n_events = (group_df[outcome_col] == 1).sum()
        baseline_rate = (df[outcome_col] == 1).mean()

        # Compute statistical power
        if n < 10:
            # Too small to meaningfully test
            power = 0.0
            interpretation = f"Group too small (n={n}). Cannot reliably detect bias."
        else:
            if test_type == "t-test":
                # For continuous outcome
                power = tt_ind_solve_power(
                    effect_size=effect_size,
                    nobs1=n,
                    alpha=alpha,
                    alternative='two-sided'
                )
            elif test_type == "chi-squared":
                # For binary outcome (two-proportion z-test)
                # Power to detect effect_size difference in proportions
                baseline = baseline_rate
                alternative = baseline * (1 + effect_size)
                alternative = min(alternative, 0.99)

                # One-sample proportion z-test power approximation.
                # Uses abs() because power depends on effect magnitude, not
                # its sign — a signed difference here would flip power to
                # ~0 for any well-detectable effect (confirmed bug: a
                # 15,000-patient group was scoring 0% power before this fix).
                std_error = np.sqrt(baseline * (1-baseline) / n)
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = abs(baseline - alternative) / std_error
                power = 1 - stats.norm.cdf(z_alpha - z_beta)
                power = np.clip(power, 0, 1)
            else:
                power = 0.5  # Unknown test type

            # Interpretation
            if power < 0.5:
                interpretation = f"SEVERE insufficiency. Power={power:.2f} (group too small)"
            elif power < power_threshold:
                interpretation = f"Insufficient power. Cannot reliably detect bias for this group."
            else:
                interpretation = "Adequate power for bias detection."

        passes = power >= power_threshold

        results[intersection_name] = IntersectionPower(
            intersection=intersection_name,
            subgroup_size=n,
            outcome_events=int(n_events),
            effect_size=effect_size,
            power=power,
            passes=passes,
            threshold=power_threshold,
            interpretation=interpretation
        )

    return results


def identify_insufficiency_masking(
    power_matrix: Dict[str, IntersectionPower],
    threshold: float = 0.80
) -> Dict[str, IntersectionPower]:
    """
    Identify all demographic intersections flagged as insufficient.

    Args:
        power_matrix: Output from compute_power_matrix
        threshold: Power threshold (default 0.80)

    Returns:
        Dict of insufficiently powered intersections
    """

    return {
        name: power for name, power in power_matrix.items()
        if power.power < threshold
    }


def insufficiency_report(
    power_matrix: Dict[str, IntersectionPower],
    threshold: float = 0.80
) -> str:
    """
    Generate human-readable report of insufficiency masking.

    Args:
        power_matrix: Output from compute_power_matrix
        threshold: Power threshold

    Returns:
        Formatted report string
    """

    insufficiency = identify_insufficiency_masking(power_matrix, threshold)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("INTERSECTIONAL SUFFICIENCY & INSUFFICIENCY MASKING")
    report_lines.append("=" * 70)

    all_pass = all(p.passes for p in power_matrix.values())
    report_lines.append(f"\nOverall Status: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    if insufficiency:
        report_lines.append(f"\n{len(insufficiency)} intersection(s) with INSUFFICIENT POWER:")
        report_lines.append("-" * 70)

        # Sort by power (worst first)
        sorted_insufficient = sorted(
            insufficiency.items(),
            key=lambda x: x[1].power
        )

        for intersection_name, power_info in sorted_insufficient:
            severity = "SEVERE" if power_info.power < 0.5 else "MODERATE"
            report_lines.append(f"\n{intersection_name}: {severity}")
            report_lines.append(f"  N:              {power_info.subgroup_size}")
            report_lines.append(f"  Outcome events: {power_info.outcome_events}")
            report_lines.append(f"  Power:          {power_info.power:.2%}")
            report_lines.append(f"  Threshold:      {power_info.threshold:.0%}")
            report_lines.append(f"  Status:         {'Cannot reliably detect bias' if power_info.power < 0.5 else 'Insufficient power'}")
            report_lines.append(f"  Note:           {power_info.interpretation}")
    else:
        report_lines.append("\nNo intersections with insufficient power. ✓")

    # Summary statistics
    report_lines.append("\n" + "-" * 70)
    report_lines.append("Summary:")
    n_sufficient = sum(1 for p in power_matrix.values() if p.passes)
    n_insufficient = len(power_matrix) - n_sufficient
    report_lines.append(f"  Total intersections: {len(power_matrix)}")
    report_lines.append(f"  Sufficient power:    {n_sufficient}")
    report_lines.append(f"  Insufficient power:  {n_insufficient}")
    report_lines.append(f"  Insufficiency rate:  {n_insufficient/len(power_matrix)*100:.1f}%")

    report_lines.append("\n" + "=" * 70)

    return "\n".join(report_lines)


def compute_required_sample_size(
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Compute required sample size for desired power.

    Args:
        baseline_rate: Baseline event rate (0-1)
        effect_size: Desired effect size to detect (e.g., 0.2 for 20%)
        alpha: Significance level
        power: Desired power (default 0.80)

    Returns:
        Required sample size per group
    """

    from statsmodels.stats.power import proportions_ztest

    # Simplified calculation for proportions
    p0 = baseline_rate
    p1 = baseline_rate * (1 + effect_size)
    p1 = min(p1, 0.99)

    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(1 - (1 - power)/2)

    p_avg = (p0 + p1) / 2
    denominator = (p0 - p1) ** 2
    numerator = (z_alpha + z_beta) ** 2 * p_avg * (1 - p_avg) * 2

    n = int(np.ceil(numerator / denominator))

    return max(n, 10)  # Minimum 10 per group
