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
        use_mock_fm: bool = True
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
            use_mock_fm: Use mock FM for testing (no GPU needed)
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
        self.use_mock_fm = use_mock_fm

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
        if self.causal_dag:
            pathway_results = causal.compute_psce(
                self.dataset,
                self.causal_dag,
                self.demographic_cols[0],
                self.outcome_col
            )
            pathway_passes = pathway_results['illegitimate_strength'] < 0.2  # <20% illegitimate
            pathway_findings = (
                f"Path-specific causal effect decomposition:\n"
                f"  Total effect: {pathway_results['total_effect']:.3f}\n"
                f"  Illegitimate pathway strength: {pathway_results['illegitimate_strength']:.1%}\n"
                f"  Status: {'PASS - Care pathways are legitimate' if pathway_passes else 'FAIL - Racial bias detected in care pathways'}"
            )
        else:
            pathway_passes = True  # Cannot test without DAG
            pathway_findings = "Causal DAG not provided. Care pathway analysis skipped."

        certifications['care_pathway_sufficiency'] = CertificationResult(
            condition_name="Care Pathway Sufficiency",
            passes=pathway_passes,
            findings=pathway_findings
        )

        # Condition 3: Phenotypic Coverage Sufficiency
        if self.severity_col:
            pheno_coverage = phenotypic.compute_coverage(
                self.dataset,
                self.demographic_cols[0],
                self.severity_col
            )
            pheno_passes = all(cov.passes for cov in pheno_coverage.values())
            pheno_findings = phenotypic.phenotypic_report(pheno_coverage)
        else:
            pheno_passes = True
            pheno_findings = "Severity column not provided. Phenotypic coverage analysis skipped."

        certifications['phenotypic_coverage_sufficiency'] = CertificationResult(
            condition_name="Phenotypic Coverage Sufficiency",
            passes=pheno_passes,
            findings=pheno_findings
        )

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
                    use_mock=self.use_mock_fm
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
            c5_avg_score = 1.0
            c5_passes = True

        # Overall certification
        overall_passes = all(c.passes for c in certifications.values())

        # Generate recommendation
        if overall_passes:
            recommendation = (
                "✓ Dataset PASSES all sufficiency conditions. "
                "Safe to proceed with training."
            )
        else:
            failed_conditions = [name for name, c in certifications.items() if not c.passes]
            recommendation = (
                f"✗ Dataset FAILS {len(failed_conditions)} condition(s):\n"
                f"  • {', '.join(failed_conditions)}\n"
                f"\nDO NOT TRAIN on this dataset. "
                f"Address the failing conditions before resubmission."
            )

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
