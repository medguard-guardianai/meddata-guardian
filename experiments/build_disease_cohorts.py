#!/usr/bin/env python3
"""
Build MIMIC-IV disease cohorts for FIDES validation.

Every field is derived directly from raw MIMIC-IV tables. No column is
randomly generated or hardcoded as a placeholder — if a real value isn't
available, the cohort/column is skipped rather than faked.

Outcome: in-hospital mortality (admissions.hospital_expire_flag) — the
only universally available, real, comparable outcome across cohorts.

Severity proxy: comorbidity count (real count of distinct ICD codes per
admission from diagnoses_icd) — used for phenotypic coverage analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

MIMIC_PATH = Path(__file__).parent.parent / "data" / "mimic-iv-3.1" / "hosp"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "disease_cohorts"

# Disease definitions: matched by case-insensitive substring search against
# the real ICD long_title text, which is stable across ICD-9/ICD-10.
DISEASE_DEFINITIONS = {
    "sepsis": ["sepsis", "septicemia", "septic shock"],
    "aki": ["acute kidney failure", "acute renal failure"],
    "pneumonia": ["pneumonia"],
    "copd": ["chronic obstructive pulmonary", "copd"],
    "ards": ["acute respiratory distress", "ards", "acute respiratory failure"],
    "heart_failure": ["heart failure"],
    "ami": ["acute myocardial infarction", "st elevation", "non-st elevation"],
    "ischemic_stroke": ["cerebral infarction", "occlusion and stenosis of cerebral"],
    "hemorrhagic_stroke": ["intracerebral hemorrhage", "subarachnoid hemorrhage"],
    "vte": ["pulmonary embolism", "deep vein thrombosis", "venous thromboembolism"],
    "diabetic_complication": ["diabetes with", "diabetic ketoacidosis", "diabetic nephropathy"],
}


def load_mimic_table(table_name: str) -> pd.DataFrame:
    """Load a MIMIC-IV table from the local hosp directory."""
    path = MIMIC_PATH / f"{table_name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{table_name} not found at {path}")
    print(f"Loading {table_name}...", end=" ", flush=True)
    df = pd.read_csv(path, low_memory=False)
    print(f"OK ({len(df):,} rows)")
    return df


def find_icd_codes(d_icd: pd.DataFrame, keywords: list) -> np.ndarray:
    """Find ICD codes whose long_title matches any keyword (case-insensitive)."""
    pattern = "|".join(keywords)
    mask = d_icd["long_title"].str.contains(pattern, case=False, na=False, regex=True)
    return d_icd.loc[mask, "icd_code"].unique()


def build_cohort(
    disease: str,
    keywords: list,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    d_icd: pd.DataFrame,
    comorbidity_counts: pd.Series,
) -> pd.DataFrame:
    """Build a single disease cohort using only real MIMIC-IV fields."""
    codes = find_icd_codes(d_icd, keywords)
    if len(codes) == 0:
        print(f"  ✗ {disease}: no matching ICD codes found")
        return None

    matching_hadm_ids = diagnoses.loc[diagnoses["icd_code"].isin(codes), "hadm_id"].unique()
    if len(matching_hadm_ids) == 0:
        print(f"  ✗ {disease}: no admissions found")
        return None

    cohort = admissions[admissions["hadm_id"].isin(matching_hadm_ids)].copy()
    cohort = cohort.merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )

    # Real outcome: in-hospital mortality
    cohort["mortality"] = cohort["hospital_expire_flag"].astype(int)

    # Real severity proxy: number of distinct diagnosis codes for this admission
    cohort["comorbidities"] = cohort["hadm_id"].map(comorbidity_counts).fillna(0).astype(int)

    # Real length of stay, computed from real timestamps
    admit = pd.to_datetime(cohort["admittime"], errors="coerce")
    disch = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort["los_days"] = (disch - admit).dt.total_seconds() / 86400.0
    cohort = cohort[cohort["los_days"].notna() & (cohort["los_days"] >= 0)]

    cohort["race"] = cohort["race"].fillna("Unknown")
    cohort["insurance"] = cohort["insurance"].fillna("Unknown")
    cohort["gender"] = cohort["gender"].fillna("Unknown")
    cohort["admission_type"] = cohort["admission_type"].fillna("Unknown")

    cohort = cohort.rename(columns={"gender": "sex", "anchor_age": "age"})
    # Deliberately exclude subject_id/hadm_id from the saved output: they are
    # real MIMIC-IV patient/admission identifiers under a PhysioNet Data Use
    # Agreement, and no FIDES condition needs them — every computation here
    # only ever uses race/sex/age/insurance/mortality/comorbidities/los_days.
    cohort = cohort[[
        "race", "sex", "age", "insurance",
        "admission_type", "mortality", "comorbidities", "los_days",
    ]].reset_index(drop=True)

    print(
        f"  ✓ {disease:22s} | n={len(cohort):,} | "
        f"mortality={cohort['mortality'].mean()*100:.1f}% | "
        f"races={dict(cohort['race'].value_counts().head(4))}"
    )
    return cohort


def main():
    print("=" * 80)
    print("MIMIC-IV DISEASE COHORT BUILDER (real data, no fabricated fields)")
    print("=" * 80)

    patients = load_mimic_table("patients")
    admissions = load_mimic_table("admissions")
    diagnoses = load_mimic_table("diagnoses_icd")
    d_icd = load_mimic_table("d_icd_diagnoses")

    print("\nComputing real comorbidity counts per admission...")
    comorbidity_counts = diagnoses.groupby("hadm_id")["icd_code"].nunique()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding cohorts:")
    summary = {}
    for disease, keywords in DISEASE_DEFINITIONS.items():
        cohort = build_cohort(
            disease, keywords, patients, admissions, diagnoses, d_icd, comorbidity_counts
        )
        if cohort is not None and len(cohort) > 0:
            out_path = OUTPUT_DIR / f"{disease}_cohort.csv"
            cohort.to_csv(out_path, index=False)
            summary[disease] = len(cohort)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for disease, n in summary.items():
        print(f"  {disease:22s}: {n:,} admissions")
    print(f"\nTotal cohorts built: {len(summary)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
