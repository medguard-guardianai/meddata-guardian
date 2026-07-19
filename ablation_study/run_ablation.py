#!/usr/bin/env python3
"""
Ablation study: how much does each FIDES condition contribute to detecting
insufficiency, on real MIMIC-IV cohorts?

This replaces a previous version of this script that computed C1-C5 with
its own crude, partly-hardcoded formulas (e.g. a lookup table
`power = 0.95 if n>500 else 0.8 if n>100 else ...` instead of an actual
power calculation). This version calls the same real, tested condition
functions in src/fides/ that produce the paper's primary results — there
is only one implementation of each condition in this codebase, not a
second approximate one living here.
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.fides import representational, phenotypic, intersectional

COHORT_DIR = Path(__file__).parent.parent / "data" / "disease_cohorts"
RESULTS_DIR = Path(__file__).parent / "results"

DISEASES = ["aki", "ami", "ards", "copd", "diabetic_complication",
            "heart_failure", "hemorrhagic_stroke", "ischemic_stroke",
            "pneumonia", "sepsis", "vte"]
DEMOGRAPHICS = ["race", "sex", "insurance", "age_group"]

# Condition sets to ablate over. C2 (causal) is intentionally excluded here:
# it requires a binary protected attribute and a hand-built causal DAG (see
# experiments/run_fides_5_condition_complete_aaai.py), which doesn't
# generalize across all four demographic dimensions the way C1/C3/C4 do —
# including it here would mean silently reusing race-only results for
# sex/insurance/age_group, which is exactly the kind of implicit
# hardcoding this rewrite exists to remove.
CONDITION_SETS = {
    "c1_only": ["c1"],
    "c1_c3": ["c1", "c3"],
    "c1_c3_c4": ["c1", "c3", "c4"],
}


def load_cohort(disease):
    df = pd.read_csv(COHORT_DIR / f"{disease}_cohort.csv")
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 65, 200], labels=["<40", "40-65", ">65"]
    ).astype(str)
    return df


def compute_c1(df, demographic_col):
    gaps = representational.compute_representation_gaps(df, demographic_col)
    return float(all(g.passes for g in gaps.values()))


def compute_c3(df, demographic_col):
    coverage = phenotypic.compute_coverage(df, demographic_col, "comorbidities")
    return float(all(c.passes for c in coverage.values()))


def compute_c4(df, demographic_col):
    power = intersectional.compute_power_matrix(df, [demographic_col], "mortality", test_type="chi-squared")
    return float(all(p.passes for p in power.values()))


CONDITION_FUNCS = {"c1": compute_c1, "c3": compute_c3, "c4": compute_c4}


def run_ablation():
    print("=" * 80)
    print("ABLATION STUDY: condition contribution on real MIMIC-IV cohorts")
    print("=" * 80)

    results = {name: {} for name in CONDITION_SETS}

    for disease in DISEASES:
        df = load_cohort(disease)
        print(f"\n{disease.upper()}")

        for demo in DEMOGRAPHICS:
            if demo not in df.columns or df[demo].nunique() < 2:
                continue

            scores = {}
            for cond in ["c1", "c3", "c4"]:
                scores[cond] = CONDITION_FUNCS[cond](df, demo)

            key = f"{disease}_{demo}"
            for set_name, conditions in CONDITION_SETS.items():
                results[set_name][key] = float(np.mean([scores[c] for c in conditions]))

            print(
                f"  {demo:10s} | C1={scores['c1']:.0f} C3={scores['c3']:.0f} C4={scores['c4']:.0f} | "
                + " | ".join(f"{s}:{results[s][key]:.2f}" for s in CONDITION_SETS)
            )

    return results


def compute_detection_rates(results):
    detection = {}
    for set_name, scores in results.items():
        failures = sum(1 for v in scores.values() if v < 0.75)
        total = len(scores)
        detection[set_name] = {
            "failing_combinations": failures,
            "total_combinations": total,
            "detection_rate": failures / total if total > 0 else 0.0,
        }
    return detection


def main():
    results = run_ablation()
    detection = compute_detection_rates(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "ablation_study_results.json", "w") as f:
        json.dump({"scores": results, "detection": detection}, f, indent=2)

    print("\n" + "=" * 80)
    print("DETECTION RATES BY CONDITION SET (real data, real computation)")
    print("=" * 80)
    for set_name in CONDITION_SETS:
        info = detection[set_name]
        print(
            f"  {set_name:12s}: {info['failing_combinations']:2d}/"
            f"{info['total_combinations']:2d} fail "
            f"({100*info['detection_rate']:.1f}%)"
        )

    print(f"\nSaved to {RESULTS_DIR / 'ablation_study_results.json'}")


if __name__ == "__main__":
    main()
