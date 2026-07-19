#!/usr/bin/env python3
"""Supplementary C5 run: real GPT-4o-mini via OpenAI, for the 3 diseases
that have real guideline-based test scenarios. Run separately from the
main sweep so the expensive Ollama pass doesn't have to be redone.
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.fides.condition_5_model_behavior import compute_condition_5

COHORT_DIR = Path(__file__).parent.parent / "data" / "disease_cohorts"
RESULTS_DIR = Path(__file__).parent.parent / "results"

C5_SCENARIO_MAP = {"ami": "cardiac", "sepsis": "sepsis", "pneumonia": "pneumonia"}
C5_VALID_RACES = {"White", "Black", "Asian", "Hispanic"}


def main():
    results = {}
    for disease, scenario_disease in C5_SCENARIO_MAP.items():
        df = pd.read_csv(COHORT_DIR / f"{disease}_cohort.csv")
        df_valid = df[df["race"].isin(C5_VALID_RACES)]

        print(f"Running gpt-4o-mini on {disease} ({scenario_disease} scenarios)...")
        c5_score, result = compute_condition_5(
            df_valid, scenario_disease, "race", model_name="gpt-4o-mini", backend="openai"
        )
        results[disease] = {
            "c5_score": c5_score,
            "escalation_rates": result.escalation_rates,
            "max_gap": result.max_gap,
            "passes": result.passes,
            "recommendation": result.recommendation,
        }
        print(f"  C5={c5_score:.3f} max_gap={result.max_gap:.1%} {'PASS' if result.passes else 'FAIL'}")
        print(f"  escalation_rates={result.escalation_rates}")

    with open(RESULTS_DIR / "fides_c5_openai_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/fides_c5_openai_results.json")


if __name__ == "__main__":
    main()
