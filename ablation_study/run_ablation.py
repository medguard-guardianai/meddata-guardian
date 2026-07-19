#!/usr/bin/env python3
"""Comprehensive Ablation Study for FIDES"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

COHORT_DIR = Path(__file__).parent.parent / "results" / "disease_cohorts"
RESULTS_DIR = Path(__file__).parent / "results"

def compute_c1_score(df, demographic_col):
    """Condition 1: Representational"""
    if demographic_col not in df.columns:
        return 0.75
    demos = df[demographic_col].value_counts()
    return float(max(0.0, min(1.0, 1.0 - np.std(demos / len(df)))))

def compute_c2_score(df, demographic_col, outcome_col):
    """Condition 2: Care Pathway"""
    if demographic_col not in df.columns or outcome_col not in df.columns:
        return 0.75
    df_copy = df.copy()
    df_copy[demographic_col] = (df_copy[demographic_col].astype(str).str.len() > 0).astype(int)
    df_copy[outcome_col] = (df_copy[outcome_col].astype(float) > df_copy[outcome_col].astype(float).median()).astype(int)
    exposed = df_copy[df_copy[demographic_col] == 1][outcome_col].mean()
    unexposed = df_copy[df_copy[demographic_col] == 0][outcome_col].mean()
    effect = abs(exposed - unexposed)
    return float(max(0.0, min(1.0, 1.0 - effect)))

def compute_c3_score(df):
    """Condition 3: Phenotypic"""
    try:
        if 'mortality' in df.columns:
            outcome_variance = df['mortality'].std()
            return float(max(0.0, min(1.0, outcome_variance)))
        return 0.75
    except:
        return 0.75

def compute_c4_score(df, demographic_col, outcome_col):
    """Condition 4: Intersectional Power"""
    if demographic_col not in df.columns or outcome_col not in df.columns:
        return 0.75
    demographics = df[demographic_col].unique()
    power_scores = []
    for demo in demographics:
        subset = df[df[demographic_col] == demo]
        n = len(subset)
        if n < 30:
            power = 0.0
        elif n < 100:
            power = 0.5
        elif n < 500:
            power = 0.8
        else:
            power = 0.95
        power_scores.append(power)
    return float(np.mean(power_scores) if power_scores else 0.75)

def compute_c5_score(df, demographic_col, outcome_col):
    """Condition 5: Model Behavior"""
    if demographic_col not in df.columns or outcome_col not in df.columns:
        return 0.75
    outcome_by_demo = df.groupby(demographic_col)[outcome_col].mean()
    if len(outcome_by_demo) > 1:
        max_gap = outcome_by_demo.max() - outcome_by_demo.min()
        return float(max(0.0, min(1.0, 1.0 - (max_gap / 0.3))))
    return 0.75

def compute_cds_with_conditions(df, demographic_col, outcome_col, conditions):
    """Compute CDS with specified conditions"""
    scores = []
    if 1 in conditions:
        scores.append(compute_c1_score(df, demographic_col))
    if 2 in conditions:
        scores.append(compute_c2_score(df, demographic_col, outcome_col))
    if 3 in conditions:
        scores.append(compute_c3_score(df))
    if 4 in conditions:
        scores.append(compute_c4_score(df, demographic_col, outcome_col))
    if 5 in conditions:
        scores.append(compute_c5_score(df, demographic_col, outcome_col))
    return float(np.mean(scores)) if scores else 0.5

def run_ablation():
    """Run ablation study"""
    cohorts = {"cardiac": "readmission_cohort.csv", "sepsis": "sepsis_cohort.csv",
               "pneumonia": "pneumonia_cohort.csv", "aki": "aki_cohort.csv", "stroke": "stroke_cohort.csv"}
    demographics = ["race", "insurance", "sex", "age"]
    ablation_results = {"c1_only": {}, "c1_c2": {}, "c1_c3": {}, "c1_c4": {}, "c1_c5_full": {}}

    print("\n" + "="*80)
    print("ABLATION STUDY: CONDITION CONTRIBUTION ANALYSIS")
    print("="*80)

    for disease, filename in cohorts.items():
        path = COHORT_DIR / filename
        if not path.exists():
            continue
        print(f"\n📊 {disease.upper()}")
        df = pd.read_csv(path)

        for demo in demographics:
            if demo not in df.columns:
                continue
            outcome_col = "mortality" if "mortality" in df.columns else "readmitted" if "readmitted" in df.columns else df.columns[-1]
            print(f"  {demo:12s}", end="", flush=True)

            c1_only = compute_cds_with_conditions(df, demo, outcome_col, [1])
            c1_c2 = compute_cds_with_conditions(df, demo, outcome_col, [1, 2])
            c1_c3 = compute_cds_with_conditions(df, demo, outcome_col, [1, 2, 3])
            c1_c4 = compute_cds_with_conditions(df, demo, outcome_col, [1, 2, 3, 4])
            c1_c5 = compute_cds_with_conditions(df, demo, outcome_col, [1, 2, 3, 4, 5])

            key = f"{disease}_{demo}"
            ablation_results["c1_only"][key] = c1_only
            ablation_results["c1_c2"][key] = c1_c2
            ablation_results["c1_c3"][key] = c1_c3
            ablation_results["c1_c4"][key] = c1_c4
            ablation_results["c1_c5_full"][key] = c1_c5

            print(f" | C1:{c1_only:.3f} | +C2:{c1_c2:.3f} | +C3:{c1_c3:.3f} | +C4:{c1_c4:.3f} | +C5:{c1_c5:.3f}")

    return ablation_results

def compute_detection_rates(ablation_results):
    """Compute detection rates"""
    detection = {}
    for condition_set, scores in ablation_results.items():
        failures = sum(1 for score in scores.values() if score < 0.75)
        total = len(scores)
        detection[condition_set] = {
            "failing_datasets": failures,
            "detection_rate": failures / total if total > 0 else 0
        }
    return detection

def main():
    """Main"""
    ablation_results = run_ablation()
    detection_rates = compute_detection_rates(ablation_results)

    # Save
    results_file = RESULTS_DIR / "ablation_study_results.json"
    with open(results_file, "w") as f:
        json.dump({"ablation": ablation_results, "detection": detection_rates}, f, indent=2)
    print(f"\n✓ Saved: {results_file}")

    # Summary
    print("\n" + "="*80)
    print("DETECTION RATES BY CONDITION SET")
    print("="*80)
    for cond_set in ["c1_only", "c1_c2", "c1_c3", "c1_c4", "c1_c5_full"]:
        info = detection_rates[cond_set]
        pct = 100 * info['detection_rate']
        print(f"  {cond_set:15s}: {info['failing_datasets']:2d} datasets | {pct:5.1f}%")

if __name__ == "__main__":
    main()
