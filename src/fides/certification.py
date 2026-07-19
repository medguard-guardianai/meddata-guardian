"""
FIDES Certification Framework - Main Entry Point

Coordinates all four sufficiency conditions and generates certification report.
"""

import pandas as pd
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from . import causal, representational, phenotypic, intersectional
from .condition_5_model_behavior import compute_condition_5


@dataclass
class CertificationResult:
    """Single certification condition result."""
    condition_name: str
    passes: bool
    findings: str
    confidence: float = 0.95


@dataclass
class CertificationReport:
    """Complete FIDES certification report."""
    dataset_name: str
    n_records: int
    timestamp: str
    certifications: Dict[str, CertificationResult]
    overall_passes: bool
    recommendation: str
    insufficiency_masking_detected: bool
    insufficiency_summary: str = ""

    def to_json(self) -> str:
        """Convert report to JSON."""
        return json.dumps(asdict(self), indent=2, default=str)

    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = []
        lines.append("# FIDES Certification Report\n")
        lines.append(f"**Dataset:** {self.dataset_name}")
        lines.append(f"**Records:** {self.n_records:,}")
        lines.append(f"**Timestamp:** {self.timestamp}\n")

        lines.append(f"## Overall Certification: {'✓ PASS' if self.overall_passes else '✗ FAIL'}\n")

        lines.append("## Condition Results\n")
        for cond_name, result in self.certifications.items():
            status = "✓ PASS" if result.passes else "✗ FAIL"
            lines.append(f"### {cond_name}: {status}")
            lines.append(f"{result.findings}\n")

        if self.insufficiency_masking_detected:
            lines.append("## ⚠️ Insufficiency Masking Detected\n")
            lines.append(self.insufficiency_summary + "\n")

        lines.append("## Recommendation\n")
        lines.append(self.recommendation)

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return self.to_markdown()


class FIDESCertifier:
    """
    Main FIDES Certifier class.

    Orchestrates all four sufficiency conditions and produces certification report.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        demographic_cols: List[str],
        outcome_col: str,
        causal_dag: Optional[causal.CausalDAG] = None,
        expected_distribution: Optional[Dict[str, float]] = None,
        severity_col: Optional[str] = None,
        dataset_name: str = "clinical_dataset",
        disease: str = "cardiac",
        enable_condition_5: bool = True,
        fm_model_name: str = "mistral"
    ):
        """
        Initialize FIDES Certifier.

        Args:
            dataset: Input DataFrame
            demographic_cols: List of demographic columns (e.g., ['race', 'sex'])
            outcome_col: Outcome column name
            causal_dag: Causal DAG for pathway analysis (optional)
            expected_distribution: Expected demographic distribution
            severity_col: Column representing clinical severity for phenotypic analysis
            dataset_name: Name for the dataset (appears in reports)
            disease: Disease type (cardiac, sepsis, pneumonia, etc.)
            enable_condition_5: Whether to include Condition 5 (model behavior)
            fm_model_name: Ollama model tag for Condition 5 testing
        """
        self.dataset = dataset
        self.demographic_cols = demographic_cols
        self.outcome_col = outcome_col
        self.causal_dag = causal_dag
        self.expected_distribution = expected_distribution
        self.severity_col = severity_col
        self.dataset_name = dataset_name
        self.disease = disease
        self.enable_condition_5 = enable_condition_5
        self.fm_model_name = fm_model_name

    def certify(self) -> CertificationReport:
        """
        Run all four FIDES sufficiency conditions.

        Returns:
            CertificationReport with all findings
        """

        certifications = {}
        insufficiency_list = []

        # Condition 1: Representational Sufficiency
        rep_gaps = representational.compute_representation_gaps(
            self.dataset,
            self.demographic_cols[0],  # Use first demographic for representation
            self.expected_distribution
        )
        rep_passes = all(gap.passes for gap in rep_gaps.values())
        rep_findings = representational.representation_report(rep_gaps)

        certifications['representational_sufficiency'] = CertificationResult(
            condition_name="Representational Sufficiency",
            passes=rep_passes,
            findings=rep_findings
        )

        # Condition 2: Care Pathway Sufficiency
        # NOTE: requires an explicit causal_dag. Unlike prior versions of this
        # code, an absent DAG is NOT treated as a silent pass — it is excluded
        # from the certification entirely so it can't be mistaken for a real
        # computed result.
        if self.causal_dag is not None:
            pathway_results = causal.compute_psce(
                self.dataset,
                self.causal_dag,
                self.demographic_cols[0],
                self.outcome_col
            )
            if not pathway_results['total_effect_significant']:
                # No statistically detectable raw disparity to begin with —
                # nothing for this condition to fail. The ratio is undefined
                # (see causal.compute_psce docstring), not zero.
                pathway_passes = True
                pathway_findings = (
                    f"Path-specific causal effect decomposition:\n"
                    f"  Total effect: {pathway_results['total_effect']:.4f} "
                    f"(p={pathway_results['total_effect_pvalue']:.3f}, not significant)\n"
                    f"  Status: PASS - No statistically detectable raw disparity to decompose"
                )
            else:
                pathway_passes = pathway_results['illegitimate_strength'] < 0.2  # <20% illegitimate
                pathway_findings = (
                    f"Path-specific causal effect decomposition:\n"
                    f"  Total effect: {pathway_results['total_effect']:.4f} "
                    f"(p={pathway_results['total_effect_pvalue']:.3f})\n"
                    f"  Illegitimate pathway strength: {pathway_results['illegitimate_strength']:.1%}\n"
                    f"  Status: {'PASS - Care pathways are legitimate' if pathway_passes else 'FAIL - Racial bias detected in care pathways'}"
                )
            certifications['care_pathway_sufficiency'] = CertificationResult(
                condition_name="Care Pathway Sufficiency",
                passes=pathway_passes,
                findings=pathway_findings
            )
        else:
            certifications['care_pathway_sufficiency'] = None  # not computed, excluded below

        # Condition 3: Phenotypic Coverage Sufficiency
        # Same rule: no severity_col means NOT COMPUTED, not an automatic pass.
        if self.severity_col is not None:
            pheno_coverage = phenotypic.compute_coverage(
                self.dataset,
                self.demographic_cols[0],
                self.severity_col
            )
            pheno_passes = all(cov.passes for cov in pheno_coverage.values())
            pheno_findings = phenotypic.phenotypic_report(pheno_coverage)
            certifications['phenotypic_coverage_sufficiency'] = CertificationResult(
                condition_name="Phenotypic Coverage Sufficiency",
                passes=pheno_passes,
                findings=pheno_findings
            )
        else:
            certifications['phenotypic_coverage_sufficiency'] = None  # not computed, excluded below

        # Condition 4: Intersectional Sufficiency & Insufficiency Masking
        inter_power = intersectional.compute_power_matrix(
            self.dataset,
            self.demographic_cols,
            self.outcome_col
        )
        inter_passes = all(p.passes for p in inter_power.values())
        inter_findings = intersectional.insufficiency_report(inter_power)

        insufficiency_detected = any(not p.passes for p in inter_power.values())
        if insufficiency_detected:
            insufficiency_list = [
                f"{name} (n={p.subgroup_size}, power={p.power:.2%})"
                for name, p in inter_power.items() if not p.passes
            ]

        certifications['intersectional_sufficiency'] = CertificationResult(
            condition_name="Intersectional Sufficiency",
            passes=inter_passes,
            findings=inter_findings
        )

        # Condition 5: Model Behavior Sufficiency (NEW)
        if self.enable_condition_5:
            c5_score_by_demo = {}
            for demographic_col in self.demographic_cols:
                c5_score, c5_result = compute_condition_5(
                    self.dataset,
                    self.disease,
                    demographic_col,
                    model_name=self.fm_model_name
                )
                c5_score_by_demo[demographic_col] = c5_score

            # Average C5 score across demographics
            c5_avg_score = sum(c5_score_by_demo.values()) / len(c5_score_by_demo) if c5_score_by_demo else 0.5
            c5_passes = c5_avg_score >= 0.75

            c5_findings = (
                f"Model Behavior Sufficiency (Condition 5):\n"
                f"  Local FM bias testing across demographics:\n"
                + "\n".join(
                    f"    {dim}: {score:.2f}"
                    for dim, score in c5_score_by_demo.items()
                ) +
                f"\n  Average C5 Score: {c5_avg_score:.3f}\n"
                f"  Status: {'PASS - Foundation model shows fair behavior' if c5_passes else 'FAIL - FM exhibits demographic bias in recommendations'}"
            )

            certifications['model_behavior_sufficiency'] = CertificationResult(
                condition_name="Model Behavior Sufficiency (Condition 5)",
                passes=c5_passes,
                findings=c5_findings
            )
        else:
            certifications['model_behavior_sufficiency'] = None  # not computed, excluded below

        # Drop conditions that weren't actually computed (no causal_dag,
        # no severity_col, or C5 disabled) instead of silently counting them
        # as passing.
        computed = {name: c for name, c in certifications.items() if c is not None}
        skipped = [name for name, c in certifications.items() if c is None]

        # Overall certification is only meaningful over conditions actually computed
        overall_passes = all(c.passes for c in computed.values()) if computed else False

        # Generate recommendation
        skip_note = f"\n(Not computed — missing required inputs: {', '.join(skipped)})" if skipped else ""
        if overall_passes:
            recommendation = (
                "✓ Dataset PASSES all computed sufficiency conditions. "
                f"Safe to proceed with training.{skip_note}"
            )
        else:
            failed_conditions = [name for name, c in computed.items() if not c.passes]
            recommendation = (
                f"✗ Dataset FAILS {len(failed_conditions)} condition(s):\n"
                f"  • {', '.join(failed_conditions)}\n"
                f"\nDO NOT TRAIN on this dataset. "
                f"Address the failing conditions before resubmission."
                f"{skip_note}"
            )

        certifications = computed

        insufficiency_summary = ""
        if insufficiency_detected:
            insufficiency_summary = (
                f"The following demographic intersections have insufficient statistical "
                f"power to detect bias, even if it exists:\n"
                + "\n".join(f"  • {item}" for item in insufficiency_list) +
                f"\n\nThese subgroups cannot be reliably audited for fairness."
            )

        report = CertificationReport(
            dataset_name=self.dataset_name,
            n_records=len(self.dataset),
            timestamp=datetime.now().isoformat(),
            certifications=certifications,
            overall_passes=overall_passes,
            recommendation=recommendation,
            insufficiency_masking_detected=insufficiency_detected,
            insufficiency_summary=insufficiency_summary
        )

        return report
