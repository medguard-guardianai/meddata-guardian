"""
Run Full FIDES Certification on MIMIC-IV Demo

Validates all four sufficiency conditions on real cardiac patient data.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fides import FIDESCertifier
from src.fides.causal import CausalDAG


def load_fides_dataset():
    """Load preprocessed MIMIC-IV cardiac data."""

    data_path = Path(__file__).parent.parent.parent / "results" / "mimic_demo" / "mimic_cardiac_fides.csv"

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run preprocess_mimic_demo.py first")
        return None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} admission records from {data_path}")

    return df


def define_cardiac_dag():
    """Define causal DAG for cardiac care bias."""

    # Causal structure:
    # race → pain_recorded (measurement bias - might underreport in minorities)
    # age → severity, outcomes (clinical confounder)
    # severity → treatment_decision, outcomes (legitimate pathway)
    # race → treatment_decision (potential decision bias)
    # treatment_decision → outcomes

    dag = CausalDAG(
        edges=[
            ('race', 'severity_latent'),        # Race → severity (confounding)
            ('age', 'severity_latent'),         # Age → severity
            ('severity_latent', 'good_outcome'),  # Severity → outcomes (legitimate)
            ('race', 'good_outcome'),           # Race → outcomes (potential bias)
        ],
        nodes=['race', 'age', 'severity_latent', 'good_outcome']
    )

    return dag


def run_fides_validation():
    """Run complete FIDES certification on MIMIC cardiac data."""

    print("\n" + "="*80)
    print("FIDES CERTIFICATION ON MIMIC-IV DEMO CARDIAC PATIENTS")
    print("="*80 + "\n")

    # Load data
    df = load_fides_dataset()
    if df is None:
        return

    # Define causal model
    dag = define_cardiac_dag()

    # Define expected demographic distribution (US population approximation)
    expected_dist = {
        'White': 0.72,
        'Black': 0.13,
        'Hispanic': 0.10,
        'Asian': 0.05
    }

    # Create certifier
    print("Initializing FIDES Certifier...")
    certifier = FIDESCertifier(
        dataset=df,
        demographic_cols=['race', 'sex'],
        outcome_col='good_outcome',
        causal_dag=dag,
        expected_distribution=expected_dist,
        severity_col='severity_latent',
        dataset_name='MIMIC_IV_Demo_Cardiac'
    )

    # Run certification
    print("Running FIDES certification (4 conditions)...\n")
    report = certifier.certify()

    # Print report
    print(report.to_markdown())

    # Save JSON report
    report_dir = Path(__file__).parent.parent.parent / "results" / "mimic_demo"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "fides_certification_report.json"
    with open(json_path, 'w') as f:
        f.write(report.to_json())

    print(f"\nJSON report saved to: {json_path}")

    # Save markdown report
    md_path = report_dir / "fides_certification_report.md"
    with open(md_path, 'w') as f:
        f.write(report.to_markdown())

    print(f"Markdown report saved to: {md_path}")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")

    print(f"Dataset: {df.shape[0]} admissions, {df['subject_id'].nunique()} unique patients")
    print(f"\nDemographics:")
    print(f"  Race: {df['race'].value_counts().to_dict()}")
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    print(f"  Age: {df['age'].describe()[['mean', 'min', 'max']].to_dict()}")

    print(f"\nCertification Results:")
    for cond_name, result in report.certifications.items():
        status = "✓ PASS" if result.passes else "✗ FAIL"
        print(f"  {status}: {cond_name}")
        if result.findings:
            for finding in result.findings[:2]:  # Show top 2 findings
                print(f"    - {finding}")

    if report.insufficiency_masking_detected:
        print(f"\n  ⚠ Insufficiency masking detected:")
        print(f"    Some demographic intersections lack statistical power")
        print(f"    Bias may exist but cannot be reliably detected in these groups")

    overall = "PASS ✓" if report.overall_passes else "FAIL ✗"
    print(f"\nOverall Certification: {overall}")

    print(f"\nRecommendation: {report.recommendation}")

    return report


if __name__ == "__main__":
    report = run_fides_validation()
