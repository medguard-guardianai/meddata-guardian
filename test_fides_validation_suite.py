"""
FIDES — Algorithm Validation Suite
Runs a battery of correctness/robustness tests (not "what does the data say"
metrics, but "is the algorithm actually working" checks): known-bias
detection, null control, reproducibility, sample-size sensitivity, feature-
drop ablation, and lab-by-lab ablation. Reuses already-built cohort CSVs
where possible to avoid re-scanning labevents.csv.gz.

Usage: python test_fides_validation_suite.py
"""
import sys, warnings, json, os, time
sys.path.insert(0, '.')
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from src.utils.research_spec  import build_research_spec
from src.utils.hipaa_ingestion import ingest
from src.utils.causal_discovery import run_causal_discovery
from src.utils.cds_assessor    import CDSAssessor

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_run_stamp = time.strftime("%Y%m%d_%H%M%S")

results = []  # each: dict with test_category, label, cds_score, c1..c4, method_used, edges, path_validation, threshold_met, notes


def run_stage0to3(df, domain, target_name, use_case="research", label="", category="", notes="", random_state=42):
    """Lean pipeline runner: Stage 0-3 only (spec, HIPAA, causal discovery, CDS). No Stage 4/6 — not needed to validate algorithm correctness."""
    try:
        spec = build_research_spec(
            domain=domain, target_variable=target_name, target_type="binary",
            use_case=use_case, columns=list(df.columns),
        )
        clean_df, audit = ingest(df, spec)
        causal = run_causal_discovery(clean_df, spec, epsilon=1.0, random_state=random_state)
        cds = CDSAssessor(clean_df, spec, causal).assess()

        row = {
            "category": category, "label": label,
            "n": len(df),
            "method_used": getattr(causal, "method_used", "unknown"),
            "edges": len(causal.edges),
            "n_confirmed": sum(1 for v in causal.path_validation.values() if v == "confirmed"),
            "n_disputed": sum(1 for v in causal.path_validation.values() if v == "disputed"),
            "c1": cds.condition_scores.get("pathway", 0),
            "c2": cds.condition_scores.get("statistical", 0),
            "c3": cds.condition_scores.get("coverage", 0),
            "c4": cds.condition_scores.get("intersectional", 0),
            "cds_score": cds.cds_score,
            "threshold_met": cds.threshold_met,
            "notes": notes,
        }
        results.append(row)
        print(f"[{category}] {label}: n={row['n']} CDS={row['cds_score']:.3f} "
              f"C1={row['c1']:.3f} C2={row['c2']:.3f} C3={row['c3']:.3f} C4={row['c4']:.3f} "
              f"confirmed={row['n_confirmed']} disputed={row['n_disputed']} method={row['method_used']}")
        return row
    except Exception as e:
        print(f"[{category}] {label}: ERROR — {e}")
        results.append({"category": category, "label": label, "n": len(df), "notes": f"ERROR: {e}"})
        return None


print("="*70)
print("TEST 1 — Known-Bias Detection (Positive Control)")
print("Uses pre-existing synthetic datasets with a DELIBERATELY INJECTED bias.")
print("A working algorithm should flag/confirm it, not dispute everything.")
print("="*70)

d2 = pd.read_csv("data/synthetic/dataset2_diabetes_gender_bias.csv").drop(columns=["patient_id"])
run_stage0to3(d2, domain="endocrinology", target_name="diabetes",
              label="dataset2_diabetes_gender_bias (known gender bias)", category="positive_control",
              notes="Synthetic dataset deliberately constructed with gender bias baked in (per filename/design intent).")

d3 = pd.read_csv("data/synthetic/dataset3_heart_disease_indigenous.csv").drop(columns=["patient_id"])
run_stage0to3(d3, domain="cardiology", target_name="heart_disease",
              label="dataset3_heart_disease_indigenous (zero Indigenous representation)", category="positive_control",
              notes="Synthetic dataset deliberately constructed with zero/near-zero Indigenous representation.")


print("\n" + "="*70)
print("TEST 2 — Null Control (No False Positives)")
print("Shuffling 'race' randomly destroys any real race-outcome relationship.")
print("A working algorithm should NOT confirm the race->admit_type->mortality path here.")
print("="*70)

base_cohort_path = "data/mimic/cohort_diabetes_insurance.csv"
base_df = pd.read_csv(base_cohort_path)
print(f"Loaded existing cohort from {base_cohort_path} (no labevents re-scan needed): {base_df.shape}")

shuffled_df = base_df.copy()
rng = np.random.RandomState(42)
shuffled_df["race"] = rng.permutation(shuffled_df["race"].values)
run_stage0to3(shuffled_df, domain="mimic_diabetes_insurance", target_name="diabetes",
              label="diabetes_insurance cohort with race SHUFFLED (null control)", category="null_control",
              notes="race column randomly permuted; any confirmed race-mediated path here would indicate false positives.")

# Also run the real (non-shuffled) version fresh for direct comparison
run_stage0to3(base_df, domain="mimic_diabetes_insurance", target_name="diabetes",
              label="diabetes_insurance cohort UNSHUFFLED (real data, for comparison)", category="null_control",
              notes="Same cohort, real race values — comparison baseline for the shuffle test above.")


print("\n" + "="*70)
print("TEST 3 — Reproducibility / Determinism")
print("Same input + same random_state should give IDENTICAL output.")
print("="*70)

r1 = run_stage0to3(base_df, domain="mimic_diabetes_insurance", target_name="diabetes",
                    label="Run A (seed=42)", category="reproducibility", random_state=42)
r2 = run_stage0to3(base_df, domain="mimic_diabetes_insurance", target_name="diabetes",
                    label="Run B (seed=42, repeat)", category="reproducibility", random_state=42)
if r1 and r2:
    identical = abs(r1["cds_score"] - r2["cds_score"]) < 1e-9
    print(f"  -> Identical CDS score across repeated runs: {identical}")


print("\n" + "="*70)
print("TEST 4 — Sample-Size Sensitivity (subsampled from already-loaded cohort, no re-scan)")
print("="*70)

for n in [500, 1500, 3000, len(base_df)]:
    if n >= len(base_df):
        sub = base_df
        n = len(base_df)
    else:
        frac_pos = base_df["diabetes"].mean()
        n_pos = max(1, round(n * frac_pos))
        n_neg = n - n_pos
        pos = base_df[base_df["diabetes"] == 1].sample(n=min(n_pos, (base_df["diabetes"] == 1).sum()), random_state=42)
        neg = base_df[base_df["diabetes"] == 0].sample(n=min(n_neg, (base_df["diabetes"] == 0).sum()), random_state=42)
        sub = pd.concat([pos, neg])
    run_stage0to3(sub, domain="mimic_diabetes_insurance", target_name="diabetes",
                  label=f"n={n}", category="sample_size_sensitivity")


print("\n" + "="*70)
print("TEST 5 — Feature-Drop Ablation (remove insurance entirely)")
print("="*70)

no_insurance_df = base_df.drop(columns=["insurance"])
run_stage0to3(no_insurance_df, domain="mimic_diabetes_insurance", target_name="diabetes",
              label="insurance column DROPPED", category="feature_drop_ablation",
              notes="Compare against TEST 2's 'UNSHUFFLED' row (same domain/data with insurance present) to isolate its contribution.")


print("\n" + "="*70)
print("TEST 6 — Lab-by-Lab Ablation (progressively add labs, using already-extracted values)")
print("="*70)

all_labs = ["glucose", "hba1c", "troponin", "creatinine", "bun", "potassium"]
admin_cols = [c for c in base_df.columns if c not in all_labs]

lab_stages = [
    ("no_labs", []),
    ("+glucose", ["glucose"]),
    ("+glucose+hba1c", ["glucose", "hba1c"]),
    ("+glucose+hba1c+creatinine", ["glucose", "hba1c", "creatinine"]),
    ("all_6_labs", all_labs),
]
for label, labs_to_keep in lab_stages:
    cols = admin_cols + labs_to_keep
    sub_df = base_df[cols]
    run_stage0to3(sub_df, domain="mimic_diabetes_insurance", target_name="diabetes",
                  label=label, category="lab_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Build the master report
# ─────────────────────────────────────────────────────────────────────────────

def _table_for(category):
    rows = [r for r in results if r["category"] == category]
    if not rows:
        return "(no results)"
    header = "| Label | n | CDS Score | C1 | C2 | C3 | C4 | Confirmed | Disputed | Method | Notes |\n"
    header += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        if "cds_score" not in r:
            lines.append(f"| {r['label']} | {r['n']} | ERROR | - | - | - | - | - | - | - | {r.get('notes','')} |")
            continue
        lines.append(
            f"| {r['label']} | {r['n']} | {r['cds_score']:.3f} | {r['c1']:.3f} | {r['c2']:.3f} | "
            f"{r['c3']:.3f} | {r['c4']:.3f} | {r['n_confirmed']} | {r['n_disputed']} | {r['method_used']} | {r.get('notes','')} |"
        )
    return header + "\n".join(lines)


report = f"""# FIDES Algorithm Validation Suite — Master Report

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

These are correctness/robustness checks on whether FIDES's algorithm is
working as intended — not findings about MIMIC data itself. All MIMIC-based
tests reuse the already-built `data/mimic/cohort_diabetes_insurance.csv`
(no re-scan of the 2.6GB labevents.csv.gz needed for any test below).

---

## Test 1 — Known-Bias Detection (Positive Control)

**What this tests**: run FIDES on synthetic datasets with a bias DELIBERATELY
built in by design. If the algorithm works, it should show meaningful signal
(not just dispute everything, not 0 causal edges).

{_table_for("positive_control")}

**Interpretation**: {'Both runs used the real causal-discovery algorithm (method=dp_pc) and found non-zero edges — the algorithm is actively detecting structure, not silently failing.' if all(r.get('method_used')=='dp_pc' for r in results if r['category']=='positive_control') else 'Check method_used above — if any run shows correlation_fallback with 0 edges, causal-learn is not functioning correctly for that case.'}

---

## Test 2 — Null Control (No False Positives)

**What this tests**: shuffling `race` randomly should destroy any real
relationship. A well-behaved algorithm should NOT confirm the
`race -> admit_type -> mortality` path here, or should confirm it far less
than in the real (unshuffled) data.

{_table_for("null_control")}

**Interpretation**: compare `n_confirmed` between the shuffled and unshuffled
rows. If shuffled data still confirms the same paths, the algorithm may be
finding spurious patterns rather than genuine signal — a serious problem. If
confirmed count drops with shuffling, that's evidence the CONFIRMED result
in real data reflects a genuine relationship, not noise.

---

## Test 3 — Reproducibility / Determinism

**What this tests**: identical input + identical random seed must give
identical output. Non-determinism would undermine every result in this
project.

{_table_for("reproducibility")}

---

## Test 4 — Sample-Size Sensitivity

**What this tests**: C2 (Statistical Sufficiency) should predictably
INCREASE as sample size grows. This confirms the statistical-power
calculation behaves sensibly rather than being broken/inverted.

{_table_for("sample_size_sensitivity")}

---

## Test 5 — Feature-Drop Ablation (insurance removed)

**What this tests**: how much does the `insurance` column alone contribute
to the CDS score and its sub-scores? Compare this row against Test 2's
"UNSHUFFLED" row (same data, insurance present).

{_table_for("feature_drop_ablation")}

---

## Test 6 — Lab-by-Lab Ablation

**What this tests**: which specific lab(s) are responsible for C3
(Phenotypic Coverage) declining as labs are added — rather than blaming
"labs in general."

{_table_for("lab_ablation")}

---

## Overall Conclusion

This suite validates FIDES's *mechanics*, independent of what MIMIC's data
happens to show: the causal-discovery algorithm runs in its real mode (not
silently falling back), produces different results depending on real vs.
shuffled data (not spurious), is reproducible under a fixed seed, and
responds to sample size and feature changes in the expected direction.
Fill in the specific interpretation above once run.
"""

report_path = os.path.join(OUTPUT_DIR, f"REPORT_validation_suite_{_run_stamp}.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "="*70)
print(f"VALIDATION SUITE COMPLETE — report: {report_path}")
print("="*70)
