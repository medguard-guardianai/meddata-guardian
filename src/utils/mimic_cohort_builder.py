"""
FIDES — MIMIC-IV Cohort Builder
Reshapes raw MIMIC-IV tables (patients, admissions, diagnoses_icd, d_icd_diagnoses)
into a single flat, one-row-per-admission cohort CSV shaped for FIDES's existing
pipeline (test_fides.py-style flat table + scalar binary target).

Does NOT touch the synthetic-data pipeline or any other FIDES stage.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Race bucketing ────────────────────────────────────────────────────────────
# MIMIC's `race` field has ~30 fine-grained strings. Bucket into broad OMB-style
# categories so subgroup sample sizes are large enough for CDS statistical/
# intersectional sufficiency checks (Stage 3) to be meaningful.
_RACE_BUCKET_RULES = [
    ("hispanic", "Hispanic"),
    ("latino", "Hispanic"),
    ("black", "Black"),
    ("african", "Black"),
    ("caribbean", "Black"),
    ("cape verdean", "Black"),
    ("asian", "Asian"),
    ("native hawaiian", "Pacific Islander"),
    ("pacific islander", "Pacific Islander"),
    ("american indian", "American Indian/Alaska Native"),
    ("alaska native", "American Indian/Alaska Native"),
    ("white", "White"),
    ("portuguese", "White"),
    ("unknown", "Unknown"),
    ("unable to obtain", "Unknown"),
    ("declined to answer", "Unknown"),
]


def _bucket_race(raw: str) -> str:
    norm = str(raw).lower()
    for keyword, bucket in _RACE_BUCKET_RULES:
        if keyword in norm:
            return bucket
    return "Other"


def _bucket_language(raw: str) -> str:
    """Collapse MIMIC's 20+ language values to English / Non-English.
    MIMIC's raw `language` field has dozens of values (Hindi, Polish, Amharic,
    Thai, ...) most with only a handful of patients — this alone can drive
    CDS statistical/intersectional sufficiency to near zero regardless of
    sample size. Coarser, defensible buckets avoid that artifact."""
    return "English" if str(raw).strip().upper() == "ENGLISH" else "Non-English"


def build_cohort(
    icd_prefixes: list[str],
    target_name: str,
    mimic_dir: str = "data/mimic",
    icd_version: int | None = None,
    sample_size: int | None = 5000,
    random_state: int = 42,
    bucket_language: bool = False,
) -> pd.DataFrame:
    """
    Build a flat, one-row-per-admission MIMIC-IV cohort with a derived binary
    target based on whether any diagnoses_icd.icd_code for that admission
    starts with one of `icd_prefixes`.

    `sample_size` (default 5000) draws a class-stratified random sample of the
    final cohort — FIDES's CDS Assessor (Stage 3) runs an O(n^2) Gaussian KDE
    for phenotypic coverage, which is infeasible at MIMIC's native scale
    (500K+ admissions); the synthetic datasets FIDES was built against are
    ~1,200 rows. Pass `sample_size=None` to keep the full cohort.

    `bucket_language` (default False) collapses MIMIC's 20+ language values
    down to English/Non-English. Leave False to reproduce the "insufficient
    data" cohort (many tiny language subgroups tank CDS statistical/
    intersectional scores); set True for a "sufficient data" cohort where
    every subgroup can plausibly reach adequate sample size.

    Example (diabetes, ICD-9 250.x and ICD-10 E08-E13):
        build_cohort(
            icd_prefixes=["250", "E08", "E09", "E10", "E11", "E12", "E13"],
            target_name="diabetes",
        )
    """
    mimic_dir = Path(mimic_dir)

    patients = pd.read_csv(
        mimic_dir / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = pd.read_csv(
        mimic_dir / "admissions.csv.gz",
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "insurance", "language", "marital_status",
            "race", "hospital_expire_flag",
        ],
    )
    diagnoses = pd.read_csv(
        mimic_dir / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )

    if icd_version is not None:
        diagnoses = diagnoses[diagnoses["icd_version"] == icd_version]

    # ── Derive binary target from long-format diagnoses ──────────────────────
    prefix_tuple = tuple(icd_prefixes)
    diagnoses["_is_target"] = diagnoses["icd_code"].str.startswith(prefix_tuple)
    target_flags = (
        diagnoses.groupby("hadm_id")["_is_target"]
        .any()
        .rename(target_name)
        .reset_index()
    )

    # ── Merge patients + admissions (one row per admission) ──────────────────
    cohort = admissions.merge(patients, on="subject_id", how="inner")
    cohort = cohort.merge(target_flags, on="hadm_id", how="left")
    cohort[target_name] = cohort[target_name].infer_objects(copy=False).fillna(False).astype(int)

    # ── Derive length_of_stay, drop raw timestamps (HIPAA-safe by construction) ─
    admit = pd.to_datetime(cohort["admittime"], errors="coerce")
    disch = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort["length_of_stay"] = (disch - admit).dt.total_seconds() / 86400.0
    cohort["length_of_stay"] = cohort["length_of_stay"].clip(lower=0, upper=365)

    # ── Rename / bucket ────────────────────────────────────────────────────────
    cohort = cohort.rename(columns={"anchor_age": "age", "admission_type": "admit_type"})
    cohort["race"] = cohort["race"].apply(_bucket_race)
    if bucket_language:
        cohort["language"] = cohort["language"].apply(_bucket_language)

    keep_cols = [
        "age", "gender", "race", "insurance", "marital_status", "language",
        "admit_type", "length_of_stay", "hospital_expire_flag", target_name,
    ]
    cohort = cohort[keep_cols].dropna(subset=["age", "length_of_stay"])

    # A small fraction of MIMIC admissions genuinely lack recorded insurance/
    # marital_status/language. Label as "Unknown" rather than leave NaN (which
    # breaks causal-learn's PC algorithm) or drop those real patients.
    for col in ["insurance", "marital_status", "language"]:
        cohort[col] = cohort[col].fillna("Unknown")

    if sample_size is not None and len(cohort) > sample_size:
        frac_positive = cohort[target_name].mean()
        n_positive = max(1, round(sample_size * frac_positive))
        n_negative = sample_size - n_positive
        pos = cohort[cohort[target_name] == 1].sample(
            n=min(n_positive, (cohort[target_name] == 1).sum()), random_state=random_state
        )
        neg = cohort[cohort[target_name] == 0].sample(
            n=min(n_negative, (cohort[target_name] == 0).sum()), random_state=random_state
        )
        cohort = pd.concat([pos, neg]).sample(frac=1, random_state=random_state)

    return cohort.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a flat MIMIC-IV cohort CSV for FIDES.")
    parser.add_argument("--target-name", default="diabetes")
    parser.add_argument(
        "--icd-prefixes", nargs="+",
        default=["250", "E08", "E09", "E10", "E11", "E12", "E13"],
        help="ICD code prefixes defining a positive case (default: diabetes).",
    )
    parser.add_argument("--mimic-dir", default="data/mimic")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--bucket-language", action="store_true",
        help="Collapse language to English/Non-English (use for a 'sufficient data' cohort).",
    )
    args = parser.parse_args()

    df = build_cohort(
        args.icd_prefixes, args.target_name, args.mimic_dir,
        sample_size=args.sample_size, bucket_language=args.bucket_language,
    )

    out_path = args.out or f"data/mimic/cohort_{args.target_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"Cohort written to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df[args.target_name].value_counts().to_string()}")
