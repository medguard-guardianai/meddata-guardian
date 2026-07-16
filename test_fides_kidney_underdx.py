"""
FIDES — Diagnostic Under-Recognition of Kidney Complications (new research topic)

NOT testing "does diabetes cause kidney damage" (settled medical fact).
Testing: among diabetic patients with OBJECTIVE, lab-confirmed kidney
impairment (creatinine/BUN thresholds — independent of any diagnosis code),
does race, insurance, or social support (marital_status) predict whether
that impairment actually gets clinically diagnosed/coded as CKD?

Usage: python test_fides_kidney_underdx.py
"""
import sys, warnings, json, os, time
sys.path.insert(0, '.')
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings('ignore')
import pandas as pd

from src.utils.research_spec  import build_research_spec, spec_summary
from src.utils.hipaa_ingestion import ingest
from src.utils.causal_discovery import run_causal_discovery
from src.utils.cds_assessor    import CDSAssessor
from src.utils.intervention_optimizer import optimize_intervention
from src.utils.certificate_builder    import build_certificate
from src.utils.mimic_cohort_builder   import build_cohort, derive_lab_threshold_target

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_run_stamp = time.strftime("%Y%m%d_%H%M%S")
_report_path = os.path.join(OUTPUT_DIR, f"mimic_run_report_kidney_underdx_{_run_stamp}.txt")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_report_file = open(_report_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _report_file)

TARGET_NAME = "undiagnosed_kidney_impairment"
DOMAIN      = "mimic_kidney_underdiagnosis"
USE_CASE    = "research"
BASE_SAMPLE_SIZE = 20000

print("\n" + "="*60)
print("  FIDES — Diagnostic Under-Recognition of Kidney Complications")
print("  Hypothesis: Among diabetics with OBJECTIVE (lab-confirmed) kidney")
print("  impairment, does race/insurance/social support predict whether")
print("  it actually gets diagnosed/coded?")
print("="*60)

# ── Build base diabetic cohort with labs + real CKD diagnosis flag ──────────
print(f"\nBuilding base diabetic cohort (n={BASE_SAMPLE_SIZE}) with labs + CKD diagnosis flag...")
base_df = build_cohort(
    icd_prefixes=["250", "E08", "E09", "E10", "E11", "E12", "E13"],
    target_name="diabetes", sample_size=BASE_SAMPLE_SIZE, include_labs=True,
    comorbidity_flags={"ckd_diagnosed": ["585", "N18"]},
)
print(f"Base cohort: {base_df.shape[0]} admissions ({base_df['diabetes'].sum()} diabetic)")

# ── Restrict to diabetic patients ────────────────────────────────────────────
diabetic_df = base_df[base_df["diabetes"] == 1].drop(columns=["diabetes"])
print(f"Diabetic-only: {diabetic_df.shape[0]} admissions")

# ── Objective kidney impairment from labs (independent of any diagnosis code) ─
impaired_df = derive_lab_threshold_target(
    diabetic_df, target_name="kidney_impairment",
    rules=[("creatinine", ">", 1.5), ("bun", ">", 20)],
)
print(f"With lab-confirmed impairment status: {impaired_df.shape[0]} admissions "
      f"({impaired_df['kidney_impairment'].sum()} impaired)")

# ── Restrict to those with CONFIRMED real impairment — the actual study population ─
df = impaired_df[impaired_df["kidney_impairment"] == 1].drop(columns=["kidney_impairment"]).copy()
print(f"Restricted to lab-confirmed impaired patients: {df.shape[0]} admissions")

# ── Final target: was this real impairment NOT diagnostically recognized? ───
df[TARGET_NAME] = (df["ckd_diagnosed"] == 0).astype(int)
df = df.drop(columns=["ckd_diagnosed"])

print(f"Final cohort: {df.shape[0]} admissions, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Undiagnosed distribution: {df[TARGET_NAME].value_counts().to_dict()}")

if df.shape[0] < 100:
    print("\n[!] WARNING: cohort too small after restriction — consider increasing BASE_SAMPLE_SIZE.")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*60); print("STAGE 0 — Research Specification"); print("-"*60)
spec = build_research_spec(
    domain=DOMAIN, target_variable=TARGET_NAME, target_type="binary",
    use_case=USE_CASE, columns=list(df.columns),
    intent="Among diabetics with lab-confirmed kidney impairment, does race/insurance/social support predict diagnostic under-recognition?",
)
print(spec_summary(spec))

print("\n" + "-"*60); print("STAGE 1 — HIPAA Ingestion"); print("-"*60)
clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol="IRB-2024-KIDNEY-UNDERDX")
print(f"Audit: hipaa_safe={audit['hipaa_safe']}, residual_phi_warnings={audit['residual_phi_warnings']}")

print("\n" + "-"*60); print("STAGE 2 — Causal Discovery"); print("-"*60)
causal = run_causal_discovery(clean_df, spec, epsilon=1.0)
print(f"Method used: {getattr(causal, 'method_used', 'unknown')}, edges: {len(causal.edges)}")
for path, status in causal.path_validation.items():
    print(f"  [{status.upper()}]  {path}")

print("\n" + "-"*60); print("STAGE 3 — CDS Assessment"); print("-"*60)
assessor = CDSAssessor(clean_df, spec, causal)
cds = assessor.assess()
print(f"C1={cds.condition_scores.get('pathway',0):.3f}  C2={cds.condition_scores.get('statistical',0):.3f}  "
      f"C3={cds.condition_scores.get('coverage',0):.3f}  C4={cds.condition_scores.get('intersectional',0):.3f}  "
      f"CDS={cds.cds_score:.3f}  (threshold_met={cds.threshold_met})")
print(f"Insufficiency masking flags: {len(cds.insufficiency_masking_flags)}")

print("\n" + "-"*60); print("STAGE 4 — Minimum Intervention Plan"); print("-"*60)
plan = optimize_intervention(cds, spec)
print(f"Recruitment needed: {plan.total_new_patients:,} patients, ${plan.estimated_cost_usd:,}, solver={plan.solver_status}")

print("\n" + "-"*60); print("STAGE 6 — Certificate"); print("-"*60)
cert = build_certificate(
    dataset_hash=audit['original_hash'], spec_id=f"{DOMAIN}_{TARGET_NAME}",
    use_case=USE_CASE, audit_record=audit,
    cds_result_before=cds, cds_result_after=cds, gen_result=None,
    intervention_plan=plan, irb_protocol="IRB-2024-KIDNEY-UNDERDX",
    output_dir=OUTPUT_DIR,
)
print(f"Verdict: {cert.verdict}")
print(f"HIPAA={cert.hipaa_compliant}  FDA={cert.fda_samd_compliant}  EU_AI_Act={cert.eu_ai_act_compliant}")

# ── Save cohort CSV + plain-language report ──────────────────────────────────
cohort_csv_path = os.path.join("data", "mimic", "cohort_kidney_underdx.csv")
df.to_csv(cohort_csv_path, index=False)
print(f"\nCohort CSV saved: {cohort_csv_path}")

report = f"""# FIDES Research Report — Diagnostic Under-Recognition of Kidney Complications

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

## Research Question

NOT "does diabetes cause kidney damage" (settled medical fact). Among diabetic
patients with OBJECTIVE, lab-confirmed kidney impairment (creatinine > 1.5 or
BUN > 20 — independent of any diagnosis code), does race, insurance, or
social support (marital_status) predict whether that impairment actually
gets clinically diagnosed/coded as CKD? This tests equity of diagnostic
recognition, not the existence of the diabetes-kidney link itself.

## Cohort Construction

1. Base diabetic cohort: {diabetic_df.shape[0]:,} admissions (from a {base_df.shape[0]:,}-admission sample)
2. Restricted to lab-confirmed kidney impairment: {df.shape[0]:,} admissions — this is the actual study population
3. Target = `{TARGET_NAME}`: 1 if impairment was NOT diagnostically coded despite being lab-confirmed present
Distribution: {df[TARGET_NAME].value_counts().to_dict()}

## Stage 0 — Hypothesized Causal Structure

Legitimate path: {[p.path for p in spec.legitimate_paths]}
Illegitimate paths: {[p.path for p in spec.illegitimate_paths]}

## Stage 2 — Causal Discovery Results

Method: `{getattr(causal, 'method_used', 'unknown')}`, edges found: {len(causal.edges)}

{chr(10).join(f"- `{path}` — **{'CONFIRMED' if status=='confirmed' else 'DISPUTED'}**" for path, status in causal.path_validation.items())}

## Stage 3 — CDS Scorecard

| Condition | Score |
|---|---|
| C1 Pathway | {cds.condition_scores.get('pathway',0):.3f} |
| C2 Statistical | {cds.condition_scores.get('statistical',0):.3f} |
| C3 Coverage | {cds.condition_scores.get('coverage',0):.3f} |
| C4 Intersectional | {cds.condition_scores.get('intersectional',0):.3f} |
| **CDS Score** | **{cds.cds_score:.3f}** (threshold 0.75) |

Decision: **{"PASS" if cds.threshold_met else "FAIL"}**

## Stage 4 — Recruitment

{plan.total_new_patients:,} additional patients needed, ${plan.estimated_cost_usd:,}, solver: {plan.solver_status}

## Stage 6 — Verdict

**{cert.verdict}** — HIPAA={cert.hipaa_compliant}, FDA={cert.fda_samd_compliant}, EU AI Act={cert.eu_ai_act_compliant}

---
*Cohort CSV: `data/mimic/cohort_kidney_underdx.csv`. Raw log: `{os.path.basename(_report_path)}`.*
"""
report_path = os.path.join(OUTPUT_DIR, f"REPORT_kidney_underdx_{_run_stamp}.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Research report: {report_path}")
print("\n" + "="*60)
print("  KIDNEY UNDER-DIAGNOSIS TOPIC COMPLETE")
print("="*60)

sys.stdout = sys.__stdout__
_report_file.close()
