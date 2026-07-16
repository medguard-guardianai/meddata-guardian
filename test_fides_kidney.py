"""
FIDES — Diabetic Kidney Impairment Severity (new research topic)
Restricts to diabetic patients, derives a kidney-impairment target from real
creatinine/BUN lab thresholds (not an ICD code), and tests whether race
predicts impairment severity via admission urgency — independent of glucose
(legitimate clinical severity). Insurance is deliberately not used here;
marital_status (social support proxy) is the secondary mediator instead.

Usage: python test_fides_kidney.py
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
_report_path = os.path.join(OUTPUT_DIR, f"mimic_run_report_kidney_{_run_stamp}.txt")

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

TARGET_NAME = "kidney_impairment"
DOMAIN      = "mimic_kidney_impairment"
USE_CASE    = "research"

print("\n" + "="*60)
print("  FIDES — Diabetic Kidney Impairment Severity")
print("  Hypothesis: Does race predict kidney complication severity in")
print("  diabetic patients via admission urgency, independent of glucose?")
print("="*60)

# ── Build base diabetic cohort with labs, then derive kidney target ─────────
print("\nBuilding base diabetic cohort (larger sample, since we'll filter to diabetes=1 only)...")
base_df = build_cohort(
    icd_prefixes=["250", "E08", "E09", "E10", "E11", "E12", "E13"],
    target_name="diabetes", sample_size=20000, include_labs=True,
)
print(f"Base cohort: {base_df.shape[0]} admissions ({base_df['diabetes'].sum()} diabetic)")

df = derive_lab_threshold_target(
    base_df, target_name=TARGET_NAME,
    rules=[("creatinine", ">", 1.5), ("bun", ">", 20)],
    restrict_positive_col="diabetes",
)
print(f"Diabetic-only cohort with kidney target: {df.shape[0]} admissions, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Kidney impairment distribution: {df[TARGET_NAME].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*60); print("STAGE 0 — Research Specification"); print("-"*60)
spec = build_research_spec(
    domain=DOMAIN, target_variable=TARGET_NAME, target_type="binary",
    use_case=USE_CASE, columns=list(df.columns),
    intent="Does race predict diabetic kidney complication severity via admission urgency, independent of glucose?",
)
print(spec_summary(spec))

print("\n" + "-"*60); print("STAGE 1 — HIPAA Ingestion"); print("-"*60)
clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol="IRB-2024-KIDNEY")
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
    intervention_plan=plan, irb_protocol="IRB-2024-KIDNEY",
    output_dir=OUTPUT_DIR,
)
print(f"Verdict: {cert.verdict}")
print(f"HIPAA={cert.hipaa_compliant}  FDA={cert.fda_samd_compliant}  EU_AI_Act={cert.eu_ai_act_compliant}")

# ── Save cohort CSV + plain-language report ──────────────────────────────────
cohort_csv_path = os.path.join("data", "mimic", "cohort_kidney_impairment.csv")
df.to_csv(cohort_csv_path, index=False)
print(f"\nCohort CSV saved: {cohort_csv_path}")

report = f"""# FIDES Research Report — Diabetic Kidney Impairment Severity

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

## Research Question

Does race predict diabetic kidney complication severity via admission urgency,
independent of legitimate glycemic severity (glucose)? Insurance was
deliberately excluded from this topic; marital_status (social support proxy)
is used as the secondary mediator instead.

## Cohort

{df.shape[0]:,} diabetic admissions (restricted from a {base_df.shape[0]:,}-admission base cohort),
target = `{TARGET_NAME}` (creatinine > 1.5 mg/dL OR BUN > 20 mg/dL).
Target distribution: {df[TARGET_NAME].value_counts().to_dict()}

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
*Cohort CSV: `data/mimic/cohort_kidney_impairment.csv`. Raw log: `{os.path.basename(_report_path)}`.*
"""
report_path = os.path.join(OUTPUT_DIR, f"REPORT_kidney_impairment_{_run_stamp}.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Research report: {report_path}")
print("\n" + "="*60)
print("  KIDNEY IMPAIRMENT TOPIC COMPLETE")
print("="*60)

sys.stdout = sys.__stdout__
_report_file.close()
