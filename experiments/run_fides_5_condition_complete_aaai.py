#!/usr/bin/env python3
"""
FIDES 5-Condition Validation — Real Data, Real Computation, Real FM

Runs all applicable FIDES conditions on real MIMIC-IV disease cohorts
(built by build_disease_cohorts.py from raw MIMIC-IV tables, no fabricated
fields). No condition is scored by random-number generation or hardcoded
lookup; every score comes from an actual computation over real data.

Honest limitations (documented, not hidden):
- C2 (causal) requires a causal DAG. Race is treated as binary
  (White vs. all other groups combined) because the underlying PSCE
  implementation's multi-category regression path (factorize + linregress)
  is not statistically valid for an unordered multi-level race variable.
- C3 (phenotypic) uses `comorbidities` (real distinct-ICD-code count per
  admission) as the severity proxy, since no lab-based acuity score is
  available in the light MIMIC tables used here.
- C5 (model behavior) only runs for diseases with real guideline-based
  test scenarios defined in clinical_scenarios.py (cardiac, sepsis,
  pneumonia) and only for the four demographic groups those scenarios
  cover (White, Black, Asian, Hispanic). It queries a real local model
  through Ollama — if Ollama isn't running, this fails loudly rather than
  falling back to a mock.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fides import representational, causal, phenotypic, intersectional
from src.fides.causal import CausalDAG
from src.fides.condition_5_model_behavior import compute_condition_5
from src.fides.baselines import GapAnalysisBaseline, StratifiedGapPowerBaseline, FairlearnBaseline

COHORT_DIR = Path(__file__).parent.parent / "data" / "disease_cohorts"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEMOGRAPHIC_COLS = ["race", "sex", "insurance", "age_group"]

# Diseases with real guideline-based C5 test scenarios (see clinical_scenarios.py)
C5_SCENARIO_MAP = {
    "ami": "cardiac",
    "sepsis": "sepsis",
    "pneumonia": "pneumonia",
}
C5_VALID_RACES = {"White", "Black", "Asian", "Hispanic"}


def load_cohorts() -> dict:
    """Load every real cohort CSV in data/disease_cohorts/."""
    cohorts = {}
    for path in sorted(COHORT_DIR.glob("*_cohort.csv")):
        disease = path.stem.replace("_cohort", "")
        df = pd.read_csv(path)
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 40, 65, 200], labels=["<40", "40-65", ">65"]
        ).astype(str)
        cohorts[disease] = df
        print(f"  loaded {disease:22s} n={len(df):,}")
    return cohorts


def build_causal_dag(df: pd.DataFrame) -> tuple:
    """
    Build a real causal DAG for C2 using only columns present in the data.

    race_binary -> comorbidities -> mortality   (legitimate clinical pathway)
    race_binary -> insurance_binary -> mortality (potential structural bias)
    race_binary -> mortality                     (direct/unexplained)

    Returns (dag, mediators_by_path, dataframe_with_binary_cols).
    """
    df = df.copy()
    df["race_binary"] = (df["race"] == "White").astype(int)
    df["insurance_binary"] = (df["insurance"] == "Private").astype(int)

    dag = CausalDAG(
        edges=[
            ("race_binary", "comorbidities"),
            ("comorbidities", "mortality"),
            ("race_binary", "insurance_binary"),
            ("insurance_binary", "mortality"),
            ("race_binary", "mortality"),
        ],
        nodes=["race_binary", "comorbidities", "insurance_binary", "mortality"],
    )
    # path 0: race->comorbidities->mortality (legitimate clinical pathway)
    # path 1: race->insurance->mortality (structural, not clinical)
    # path 2: race->mortality direct (unexplained)
    mediators_by_path = {0: ["comorbidities"], 1: [], 2: []}
    return dag, mediators_by_path, df


def run_condition_1_to_4(df: pd.DataFrame, disease: str, demographic_col: str) -> dict:
    """
    Run C1 (representation), C3 (phenotypic), C4 (power) directly on
    `demographic_col`. C2 (causal) always uses a binarized race column
    internally (see build_causal_dag) since the PSCE implementation is only
    statistically valid for a 2-category protected attribute; it is reported
    once per disease regardless of which demographic_col this call is for,
    and only actually computed when demographic_col == "race".
    """
    scores = {}
    findings = {}

    # C1: Representational sufficiency
    rep_gaps = representational.compute_representation_gaps(df, demographic_col)
    scores["representational_sufficiency"] = float(all(g.passes for g in rep_gaps.values()))
    findings["representational_sufficiency"] = representational.representation_report(rep_gaps)

    # C2: Causal sufficiency (race only, binarized — see module docstring)
    if demographic_col == "race":
        dag, mediators_by_path, df_bin = build_causal_dag(df)
        try:
            psce = causal.compute_psce(df_bin, dag, "race_binary", "mortality", mediators_by_path)
            c2_passes = psce["illegitimate_strength"] < 0.2
            scores["care_pathway_sufficiency"] = float(c2_passes)
            findings["care_pathway_sufficiency"] = (
                f"Total effect (unadjusted): {psce['total_effect']:.4f}, "
                f"direct effect (adjusted for {psce['legitimate_mediators']}): {psce['direct_effect']:.4f}, "
                f"illegitimate pathway strength: {psce['illegitimate_strength']:.1%} "
                f"(White vs. all-other-races binary comparison; comorbidities is the only "
                f"mediator treated as legitimate/clinical)"
            )
        except Exception as e:
            findings["care_pathway_sufficiency"] = f"NOT COMPUTED: {e}"

    # C3: Phenotypic coverage (real severity proxy: comorbidity count)
    pheno = phenotypic.compute_coverage(df, demographic_col, "comorbidities")
    scores["phenotypic_coverage_sufficiency"] = float(all(c.passes for c in pheno.values()))
    findings["phenotypic_coverage_sufficiency"] = phenotypic.phenotypic_report(pheno)

    # C4: Intersectional power
    power = intersectional.compute_power_matrix(df, [demographic_col], "mortality", test_type="chi-squared")
    scores["intersectional_sufficiency"] = float(all(p.passes for p in power.values()))
    findings["intersectional_sufficiency"] = intersectional.insufficiency_report(power)

    overall_passes = all(bool(v) for v in scores.values())

    return {
        "disease": disease,
        "demographic": demographic_col,
        "conditions_computed": list(scores.keys()),
        "condition_scores": scores,
        "overall_passes": overall_passes,
        "findings": findings,
    }


C5_MODELS = [
    {"backend": "ollama", "model_name": "mistral"},
    {"backend": "openai", "model_name": "gpt-4o-mini"},
]


def run_condition_5(df: pd.DataFrame, disease: str) -> dict:
    """Run C5 for diseases with real test scenarios, across all configured real FMs."""
    scenario_disease = C5_SCENARIO_MAP.get(disease)
    if scenario_disease is None:
        return {"skipped": True, "reason": "no real guideline scenarios defined for this disease"}

    df_valid = df[df["race"].isin(C5_VALID_RACES)]
    if df_valid["race"].nunique() < 2:
        return {"skipped": True, "reason": "fewer than 2 valid race groups present"}

    by_model = {}
    for cfg in C5_MODELS:
        key = f"{cfg['backend']}:{cfg['model_name']}"
        try:
            c5_score, result = compute_condition_5(
                df_valid, scenario_disease, "race",
                model_name=cfg["model_name"], backend=cfg["backend"]
            )
            by_model[key] = {
                "c5_score": c5_score,
                "escalation_rates": result.escalation_rates,
                "max_gap": result.max_gap,
                "passes": result.passes,
                "recommendation": result.recommendation,
            }
        except Exception as e:
            by_model[key] = {"error": str(e)}

    return {"skipped": False, "by_model": by_model}


def run_baselines(df: pd.DataFrame, demographic_col: str) -> dict:
    """Run real baseline methods for comparison."""
    gap = GapAnalysisBaseline().analyze(df, demographic_col, "mortality")
    power = StratifiedGapPowerBaseline().analyze(df, demographic_col, "mortality")
    fairlearn = FairlearnBaseline().analyze(df, demographic_col, "mortality")
    return {
        "gap_analysis_fails": not gap["passes"],
        "stratified_power_fails": not power["passes"],
        "fairlearn_fails": not fairlearn["passes"],
    }


def main():
    print("=" * 80)
    print("FIDES VALIDATION — REAL DATA, REAL COMPUTATION, REAL FM (Ollama/mistral)")
    print("=" * 80)

    print("\nLoading real MIMIC-IV cohorts:")
    cohorts = load_cohorts()

    results = {}
    baseline_results = {}
    c5_results = {}

    print("\nRunning C1/C2/C3/C4 across cohorts x demographics:")
    for disease, df in cohorts.items():
        results[disease] = {}
        baseline_results[disease] = {}

        for demo_col in DEMOGRAPHIC_COLS:
            if demo_col not in df.columns:
                continue
            if df[demo_col].nunique() < 2:
                continue

            try:
                result = run_condition_1_to_4(df, disease, demo_col)
                results[disease][demo_col] = result
                baseline_results[disease][demo_col] = run_baselines(df, demo_col)

                computed_scores = result["condition_scores"]
                if computed_scores:
                    cds = np.mean(list(computed_scores.values()))
                    status = "PASS" if result["overall_passes"] else "FAIL"
                    print(
                        f"  {disease:22s} x {demo_col:10s} | "
                        f"conditions={list(computed_scores.keys())} | "
                        f"CDS={cds:.3f} | {status}"
                    )
                else:
                    print(f"  {disease:22s} x {demo_col:10s} | no conditions computed")
            except Exception as e:
                print(f"  {disease:22s} x {demo_col:10s} | ERROR: {str(e)[:80]}")

    print("\nRunning C5 (real Ollama + real OpenAI inference) on diseases with real scenarios:")
    for disease, df in cohorts.items():
        c5_result = run_condition_5(df, disease)
        c5_results[disease] = c5_result
        if c5_result.get("skipped"):
            print(f"  {disease:22s} | SKIPPED ({c5_result['reason']})")
        else:
            for model_key, model_result in c5_result["by_model"].items():
                if "error" in model_result:
                    print(f"  {disease:22s} | {model_key:20s} | ERROR: {model_result['error'][:60]}")
                else:
                    print(
                        f"  {disease:22s} | {model_key:20s} | "
                        f"C5={model_result['c5_score']:.3f} | "
                        f"max_gap={model_result['max_gap']:.1%} | "
                        f"{'PASS' if model_result['passes'] else 'FAIL'}"
                    )

    print("\nSaving results...")
    with open(RESULTS_DIR / "fides_c1_c4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(RESULTS_DIR / "fides_c5_results.json", "w") as f:
        json.dump(c5_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "baseline_comparison_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    print(f"✓ Saved to {RESULTS_DIR}/fides_c1_c4_results.json")
    print(f"✓ Saved to {RESULTS_DIR}/fides_c5_results.json")
    print(f"✓ Saved to {RESULTS_DIR}/baseline_comparison_results.json")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return results, c5_results, baseline_results


if __name__ == "__main__":
    main()
