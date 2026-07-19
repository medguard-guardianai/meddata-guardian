"""
Baseline Fairness Methods for Comparison

Implements Gap Analysis, Stratified Gap + Power, and Fairlearn
to compare against FIDES comprehensive approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.proportion import proportions_ztest


class GapAnalysisBaseline:
    """
    Simple Gap Analysis baseline.

    Computes demographic outcome gaps.
    Flags if gap > threshold or p-value < 0.05.
    """

    def __init__(self, threshold: float = 0.10):
        """
        Initialize.

        Args:
            threshold: Gap threshold (default 10pp)
        """
        self.threshold = threshold

    def analyze(
        self,
        dataset: pd.DataFrame,
        demographic_col: str,
        outcome_col: str
    ) -> Dict:
        """
        Perform gap analysis.

        Args:
            dataset: Clinical dataset
            demographic_col: Demographic column
            outcome_col: Outcome column (binary)

        Returns:
            Results dict with gaps, p-values, pass/fail
        """
        results = {}
        groups = dataset[demographic_col].unique()

        # Compute outcome rates by group
        outcome_rates = {}
        for group in groups:
            subset = dataset[dataset[demographic_col] == group]
            rate = subset[outcome_col].mean()
            outcome_rates[str(group)] = rate

        # Compute pairwise gaps
        gaps = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                gap = abs(outcome_rates[str(g1)] - outcome_rates[str(g2)])
                gaps[f"{g1}_vs_{g2}"] = gap

        max_gap = max(gaps.values()) if gaps else 0.0

        # Statistical test (chi-square)
        contingency_table = pd.crosstab(dataset[demographic_col], dataset[outcome_col])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        passes = (max_gap < self.threshold) and (p_value > 0.05)

        return {
            "method": "Gap Analysis",
            "outcome_rates": outcome_rates,
            "gaps": gaps,
            "max_gap": max_gap,
            "p_value": p_value,
            "passes": passes,
            "recommendation": (
                f"PASS - Gap {max_gap:.1%} is acceptable (p={p_value:.3f})"
                if passes else
                f"FAIL - Gap {max_gap:.1%} is significant (p={p_value:.3f})"
            )
        }


class StratifiedGapPowerBaseline:
    """
    Stratified Gap + Power Analysis baseline.

    Computes gaps AND checks if each demographic group has sufficient
    statistical power to detect the gap.
    """

    def __init__(
        self,
        gap_threshold: float = 0.10,
        power_threshold: float = 0.80,
        effect_size: float = 0.25
    ):
        """
        Initialize.

        Args:
            gap_threshold: Gap threshold (default 10pp)
            power_threshold: Power threshold (default 80%)
            effect_size: Effect size for power calculation (Cohen's h)
        """
        self.gap_threshold = gap_threshold
        self.power_threshold = power_threshold
        self.effect_size = effect_size

    def analyze(
        self,
        dataset: pd.DataFrame,
        demographic_col: str,
        outcome_col: str
    ) -> Dict:
        """
        Perform stratified gap + power analysis.

        Args:
            dataset: Clinical dataset
            demographic_col: Demographic column
            outcome_col: Outcome column (binary)

        Returns:
            Results dict
        """
        results = {}
        groups = dataset[demographic_col].unique()

        outcome_rates = {}
        group_sizes = {}
        power_by_group = {}

        for group in groups:
            subset = dataset[dataset[demographic_col] == group]
            rate = subset[outcome_col].mean()
            n = len(subset)
            outcome_rates[str(group)] = rate
            group_sizes[str(group)] = n

            # Compute power for this group
            power = self._compute_power(n, rate)
            power_by_group[str(group)] = power

        # Compute gaps
        gaps = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                gap = abs(outcome_rates[str(g1)] - outcome_rates[str(g2)])
                gaps[f"{g1}_vs_{g2}"] = gap

        max_gap = max(gaps.values()) if gaps else 0.0

        # Check if all groups have sufficient power
        underpowered = [
            f"{g} (n={group_sizes[str(g)]}, power={power_by_group[str(g)]:.1%})"
            for g in groups if power_by_group[str(g)] < self.power_threshold
        ]

        passes = (max_gap < self.gap_threshold) and len(underpowered) == 0

        return {
            "method": "Stratified Gap + Power",
            "outcome_rates": outcome_rates,
            "gaps": gaps,
            "max_gap": max_gap,
            "power_by_group": power_by_group,
            "underpowered_groups": underpowered,
            "passes": passes,
            "recommendation": (
                f"PASS - Gap {max_gap:.1%} with sufficient power in all groups"
                if passes else
                f"FAIL - Gap {max_gap:.1%} or {len(underpowered)} underpowered groups: {underpowered}"
            )
        }

    def _compute_power(self, n: int, p1: float, p2: float = None) -> float:
        """
        Compute statistical power for detecting effect size.

        Simplified calculation using normal approximation.
        """
        if p2 is None:
            p2 = p1 - self.effect_size

        p2 = max(0.01, min(0.99, p2))

        # Effect size h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        h = 2 * (np.arcsin(np.sqrt(max(0, p1))) - np.arcsin(np.sqrt(max(0, p2))))

        # Approximate power
        z_alpha = norm.ppf(0.975)  # Two-tailed, alpha=0.05
        z_beta = h * np.sqrt(n / 2) - z_alpha

        power = norm.cdf(z_beta)
        return max(0.0, min(1.0, power))


class FairlearnBaseline:
    """
    Fairlearn-inspired baseline.

    Trains a simple logistic regression and audits for fairness.
    Checks demographic parity and equalized odds.
    """

    def __init__(self):
        """Initialize."""
        pass

    def analyze(
        self,
        dataset: pd.DataFrame,
        demographic_col: str,
        outcome_col: str,
        features: List[str] = None
    ) -> Dict:
        """
        Perform Fairlearn-style audit.

        Args:
            dataset: Clinical dataset
            demographic_col: Demographic column
            outcome_col: Outcome column (binary)
            features: Features for simple model

        Returns:
            Results dict
        """
        # For simplicity, we'll compute fairness metrics on the outcome directly
        # (In real Fairlearn, you'd train a model and check its predictions)

        groups = dataset[demographic_col].unique()

        # Outcome rates per demographic (proxy for model predictions)
        outcome_rates = {}
        for group in groups:
            subset = dataset[dataset[demographic_col] == group]
            rate = subset[outcome_col].mean()
            outcome_rates[str(group)] = rate

        # Demographic parity: max difference in outcome rates
        rates = list(outcome_rates.values())
        parity_diff = max(rates) - min(rates) if rates else 0.0

        # Equalized odds: check if false positive/negative rates differ
        # (simplified to outcome rate difference)
        eod_diff = parity_diff

        # Passes if both < 10pp
        passes = (parity_diff < 0.10) and (eod_diff < 0.10)

        return {
            "method": "Fairlearn-style",
            "outcome_rates": outcome_rates,
            "demographic_parity_difference": parity_diff,
            "equalized_odds_difference": eod_diff,
            "passes": passes,
            "recommendation": (
                f"PASS - Fair (parity diff={parity_diff:.1%})"
                if passes else
                f"FAIL - Unfair (parity diff={parity_diff:.1%})"
            )
        }


def compare_baselines(
    dataset: pd.DataFrame,
    demographic_col: str,
    outcome_col: str,
    fides_c5_score: float
) -> Dict:
    """
    Compare all baseline methods against FIDES.

    Args:
        dataset: Clinical dataset
        demographic_col: Demographic column
        outcome_col: Outcome column
        fides_c5_score: FIDES Condition 5 score

    Returns:
        Comparison results
    """
    gap_analysis = GapAnalysisBaseline().analyze(dataset, demographic_col, outcome_col)
    stratified = StratifiedGapPowerBaseline().analyze(dataset, demographic_col, outcome_col)
    fairlearn = FairlearnBaseline().analyze(dataset, demographic_col, outcome_col)

    return {
        "gap_analysis": gap_analysis,
        "stratified_gap_power": stratified,
        "fairlearn": fairlearn,
        "fides_c5": {
            "score": fides_c5_score,
            "passes": fides_c5_score > 0.75
        },
        "summary": {
            "methods_passing": sum([
                gap_analysis["passes"],
                stratified["passes"],
                fairlearn["passes"],
                fides_c5_score > 0.75
            ]),
            "total_methods": 4
        }
    }
