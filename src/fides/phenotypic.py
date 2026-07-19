"""
Phenotypic Coverage Sufficiency Condition

Checks whether the full spectrum of clinical presentations/severity is
represented for each demographic group.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhenotypicCoverage:
    """Coverage analysis for a demographic group."""
    group_name: str
    severity_bins: Dict[str, float]  # bin → coverage percentage
    overall_coverage: float  # Mean coverage across bins
    missing_phenotypes: List[str]  # Bins with <50% coverage
    passes: bool  # No critical gaps
    threshold: float = 0.50


def compute_coverage(
    df: pd.DataFrame,
    demographic_col: str,
    severity_col: str,
    n_bins: int = 4,
    threshold: float = 0.50
) -> Dict[str, PhenotypicCoverage]:
    """
    Compute clinical severity coverage by demographic group.

    Args:
        df: Input dataframe
        demographic_col: Column for demographic grouping (e.g., 'race')
        severity_col: Column for clinical severity (e.g., 'troponin')
        n_bins: Number of severity bins to create (default 4)
        threshold: Minimum coverage required per bin (default 0.50)

    Returns:
        Dict mapping group → PhenotypicCoverage

    Example:
        >>> df = pd.DataFrame({
        ...     'race': ['Black']*30 + ['White']*70,
        ...     'troponin': list(range(100))
        ... })
        >>> coverage = compute_coverage(df, 'race', 'troponin')
        >>> coverage['Black'].missing_phenotypes  # Severe phenotypes missing
    """

    if df[severity_col].nunique() < 2:
        raise ValueError(
            f"'{severity_col}' has no variance (all values identical) — "
            f"cannot bin into severity quartiles for phenotypic coverage analysis"
        )

    # Create severity bins. With many tied values (common for integer counts
    # like comorbidity counts), qcut's duplicates='drop' can produce fewer
    # than n_bins actual bins, which crashes if we still hand it n_bins
    # labels — so bin first, then label by the number of bins actually
    # produced rather than assuming n_bins survived.
    df_copy = df.copy()
    binned, actual_edges = pd.qcut(
        df_copy[severity_col], q=n_bins, duplicates='drop', retbins=True
    )
    actual_n_bins = len(actual_edges) - 1
    bin_labels = [f'Bin{i}' for i in range(actual_n_bins)]
    df_copy['severity_bin'] = binned.cat.rename_categories(bin_labels)

    results = {}

    for group in df[demographic_col].unique():
        group_df = df_copy[df_copy[demographic_col] == group]
        all_severity_counts = df_copy['severity_bin'].value_counts()

        # Compute coverage for each bin
        bin_coverage = {}
        missing_phenotypes = []

        for bin_name in sorted(all_severity_counts.index):
            total_in_bin = all_severity_counts[bin_name]
            group_in_bin = len(group_df[group_df['severity_bin'] == bin_name])
            coverage_pct = group_in_bin / total_in_bin if total_in_bin > 0 else 0

            bin_coverage[bin_name] = coverage_pct * 100

            if coverage_pct < threshold:
                missing_phenotypes.append(bin_name)

        overall_coverage = np.mean(list(bin_coverage.values()))
        passes = len(missing_phenotypes) == 0

        results[group] = PhenotypicCoverage(
            group_name=str(group),
            severity_bins=bin_coverage,
            overall_coverage=overall_coverage,
            missing_phenotypes=missing_phenotypes,
            passes=passes,
            threshold=threshold
        )

    return results


def phenotypic_report(
    coverage: Dict[str, PhenotypicCoverage]
) -> str:
    """
    Generate human-readable phenotypic coverage report.

    Args:
        coverage: Dict of PhenotypicCoverage results

    Returns:
        Formatted report string
    """

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PHENOTYPIC COVERAGE SUFFICIENCY CONDITION")
    report_lines.append("=" * 70)

    all_pass = all(c.passes for c in coverage.values())
    report_lines.append(f"\nOverall Status: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    report_lines.append("\nPer-Group Analysis:")
    report_lines.append("-" * 70)

    for group_name, cov_info in coverage.items():
        status = "✓ PASS" if cov_info.passes else "✗ FAIL"
        report_lines.append(f"\n{group_name}: {status}")
        report_lines.append(f"  Overall coverage: {cov_info.overall_coverage:.1f}%")
        report_lines.append(f"  Severity bins:")

        for bin_name, pct in sorted(cov_info.severity_bins.items()):
            flag = "✗ MISSING" if bin_name in cov_info.missing_phenotypes else "✓"
            report_lines.append(f"    {bin_name}: {pct:.1f}% {flag}")

        if cov_info.missing_phenotypes:
            report_lines.append(f"  Missing phenotypes: {', '.join(cov_info.missing_phenotypes)}")

    report_lines.append("\n" + "=" * 70)

    # Summary
    groups_with_gaps = [g for g in coverage.values() if not g.passes]
    if groups_with_gaps:
        report_lines.append(f"\n{len(groups_with_gaps)} group(s) with missing clinical presentations:")
        for cov in groups_with_gaps:
            report_lines.append(f"  • {cov.group_name}: missing {len(cov.missing_phenotypes)} severity bin(s)")

    return "\n".join(report_lines)


def identify_missing_phenotypes(
    df: pd.DataFrame,
    demographic_col: str,
    severity_col: str,
    n_bins: int = 4,
    threshold: float = 0.50
) -> Dict[str, List[str]]:
    """
    Identify specific clinical presentations missing for each group.

    Args:
        df: Input dataframe
        demographic_col: Demographic column
        severity_col: Severity column
        n_bins: Number of bins
        threshold: Coverage threshold

    Returns:
        Dict mapping group → list of missing phenotypes
    """

    coverage = compute_coverage(df, demographic_col, severity_col, n_bins, threshold)
    return {group: cov.missing_phenotypes for group, cov in coverage.items()}
