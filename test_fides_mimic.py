"""
FIDES Demo on MIMIC-IV — Run this to see the full pipeline against a real
MIMIC-IV derived cohort instead of the synthetic datasets.

Usage:
  python test_fides_mimic.py                 # "insufficient data" cohort (default)
  python test_fides_mimic.py --mode sufficient   # "sufficient data" cohort (larger, bucketed language)
"""
import sys, warnings, json, os, time, argparse
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
from src.utils.mimic_cohort_builder   import build_cohort

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument(
    "--mode", choices=["insufficient", "sufficient"], default="insufficient",
    help="'insufficient': small sample, fine-grained language (fails CDS threshold). "
         "'sufficient': larger sample, English/Non-English bucketed language (designed to pass).",
)
_args = _arg_parser.parse_args()

# ── Pick a target condition ──────────────────────────────────────────────────
TARGET_NAME  = "diabetes"
ICD_PREFIXES = ["250", "E08", "E09", "E10", "E11", "E12", "E13"]
DOMAIN       = "mimic_admin"
USE_CASE     = "research"        # research | fda_submission | irb_audit | clinical_deployment
OUTPUT_DIR   = "outputs"         # avoids the /tmp default (Windows-safe)

if _args.mode == "sufficient":
    COHORT_SAMPLE_SIZE = 30000
    COHORT_BUCKET_LANGUAGE = True
else:
    COHORT_SAMPLE_SIZE = 5000
    COHORT_BUCKET_LANGUAGE = False

# ── Tee all console output to a timestamped report file ─────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
_run_stamp = time.strftime("%Y%m%d_%H%M%S")
_report_path = os.path.join(OUTPUT_DIR, f"mimic_run_report_{_args.mode}_{_run_stamp}.txt")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_report_file = open(_report_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _report_file)
print(f"[Full run report will be saved to: {_report_path}]")

print("\n" + "="*60)
print(f"  FIDES — MIMIC-IV Pipeline Demo  [mode = {_args.mode}]")
print("="*60)

# ── Build cohort from raw MIMIC tables ───────────────────────────────────────
print(f"\nBuilding '{_args.mode}' cohort from data/mimic/*.csv.gz ...")
df = build_cohort(
    icd_prefixes=ICD_PREFIXES, target_name=TARGET_NAME,
    sample_size=COHORT_SAMPLE_SIZE, bucket_language=COHORT_BUCKET_LANGUAGE,
)
print(f"Cohort built: {df.shape[0]} admissions, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Target distribution:\n{df[TARGET_NAME].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: Research Specification (NO DATA — just column names)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 0 — Research Specification (Zero PHI)")
print("─"*60)

spec = build_research_spec(
    domain         = DOMAIN,
    target_variable= TARGET_NAME,
    target_type    = "binary",
    use_case       = USE_CASE,
    columns        = list(df.columns),
    intent         = f"Predict {TARGET_NAME} diagnosis in MIMIC-IV admission cohort"
)
print(spec_summary(spec))

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: HIPAA Ingestion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 1 — HIPAA Ingestion")
print("─"*60)

clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol="IRB-2024-MIMIC-DEMO")
print(f"\nAudit record: {json.dumps({k:v for k,v in audit.items() if k != 'original_hash'}, indent=2)}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Causal Discovery
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 2 — Causal Discovery (DP-PC Algorithm)")
print("─"*60)

causal = run_causal_discovery(clean_df, spec, epsilon=1.0)
print(f"\nCausal edges found: {len(causal.edges)}")
for src, tgt in causal.edges[:10]:
    print(f"  {src}  →  {tgt}")
if len(causal.edges) > 10:
    print(f"  ... and {len(causal.edges)-10} more")
print(f"\nPath validation:")
for path, status in causal.path_validation.items():
    print(f"  [{status.upper()}]  {path}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: CDS Assessment
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 3 — Four-Condition CDS Assessment")
print("─"*60)

assessor = CDSAssessor(clean_df, spec, causal)
cds = assessor.assess()

print(f"""
  ┌─────────────────────────────────────────────┐
  │          CDS SCORECARD                      │
  ├─────────────────────────────────────────────┤
  │  Pathway Sufficiency:    {cds.condition_scores.get('pathway',0):.3f}               │
  │  Statistical Sufficiency:{cds.condition_scores.get('statistical',0):.3f}               │
  │  Phenotypic Coverage:    {cds.condition_scores.get('coverage',0):.3f}               │
  │  Intersectional:         {cds.condition_scores.get('intersectional',0):.3f}               │
  ├─────────────────────────────────────────────┤
  │  OVERALL CDS SCORE:      {cds.cds_score:.3f}               │
  │  95% CI: [{cds.confidence_interval[0]:.2f}, {cds.confidence_interval[1]:.2f}]                  │
  │  Threshold: {0.75 if USE_CASE=='research' else 0.85:.2f}  → {'PASS ✓' if cds.threshold_met else 'FAIL ✗'}                  │
  └─────────────────────────────────────────────┘""")

if cds.insufficiency_masking_flags:
    print("\n  ⚠  INSUFFICIENCY MASKING WARNINGS:")
    for flag in cds.insufficiency_masking_flags[:3]:
        print(f"    {flag[:100]}...")

print("\n  Recommendations:")
for rec in cds.recommendations:
    print(f"    • {rec}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Minimum Intervention Optimizer
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 4 — Minimum Intervention Plan (MILP)")
print("─"*60)

plan = optimize_intervention(cds, spec)

print(f"\n  Total new patients needed: {plan.total_new_patients}")
print(f"  Estimated recruitment cost: ${plan.estimated_cost_usd:,}")
print(f"  Solver status: {plan.solver_status}")
print(f"\n  Recruitment targets:")
for g in plan.groups:
    print(f"\n    Group: {g.group_name}")
    print(f"    Need:  {g.n_required} patients")
    print(f"    Why:   {g.condition_driving_need}")
    print(f"    Sites: {', '.join(g.recommended_sites[:2])}")
    if g.phenotype_profile:
        for feat, vals in list(g.phenotype_profile.items())[:2]:
            print(f"    {feat}: {vals}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: Certification
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("STAGE 6 — Deployment Fitness Certificate")
print("─"*60)

cert = build_certificate(
    dataset_hash    = audit['original_hash'],
    spec_id         = f"{DOMAIN}_{TARGET_NAME}",
    use_case        = USE_CASE,
    audit_record    = audit,
    cds_result_before = cds,
    cds_result_after  = cds,
    gen_result        = None,
    intervention_plan = plan,
    irb_protocol      = "IRB-2024-MIMIC-DEMO",
    output_dir        = OUTPUT_DIR,
)

print(f"""
  ╔══════════════════════════════════════════════╗
  ║  FIDES DEPLOYMENT FITNESS CERTIFICATE        ║
  ╠══════════════════════════════════════════════╣
  ║  Cert ID:   {cert.cert_id[:20]}...   ║
  ║  Verdict:   {cert.verdict:<35}║
  ║  CDS Score: {cert.cds_score_after:.3f} / 1.000                        ║
  ╠══════════════════════════════════════════════╣
  ║  HIPAA Compliant:   {'YES ✓' if cert.hipaa_compliant else 'NO ✗':<29}║
  ║  FDA SaMD:          {'YES ✓' if cert.fda_samd_compliant else 'NO ✗':<29}║
  ║  EU AI Act Art.10:  {'YES ✓' if cert.eu_ai_act_compliant else 'NO ✗':<29}║
  ╠══════════════════════════════════════════════╣
  ║  DP Guarantee: ε={cert.dp_epsilon:.3f}, δ=1e-6              ║
  ║  Re-id Risk:   {cert.reidentification_risk:.4f} (<0.09 threshold)       ║
  ╚══════════════════════════════════════════════╝""")

if cert.conditions:
    print("\n  Conditions for full approval:")
    for c in cert.conditions:
        print(f"    • {c}")

print(f"\n  Saved to: {OUTPUT_DIR}/fides_cert_{cert.cert_id[:12]}.pdf")
print(f"  Signature: {cert.signature[:50]}...")
print("\n" + "="*60)
print("  FIDES MIMIC PIPELINE COMPLETE")
print("="*60)
print(f"\n  Full report saved to: {_report_path}")
print(f"  Certificate JSON: {OUTPUT_DIR}/fides_cert_{cert.cert_id[:12]}.json\n")

# ─────────────────────────────────────────────────────────────────────────────
# Plain-language research report — regenerated fresh every run, from this
# run's real numbers (no hardcoded/copied values from a prior run).
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(x):
    return f"{x*100:.1f}%"

def _top_masking_flags(flags, n=10):
    return "\n".join(f"- {f}" for f in flags[:n]) if flags else "- (none)"

def _top_recruitment_groups(groups, n=10):
    lines = []
    for g in sorted(groups, key=lambda g: -g.n_required)[:n]:
        lines.append(f"- **{g.group_name}** — needs {g.n_required} more patients (driver: {g.condition_driving_need})")
    return "\n".join(lines) if lines else "- (none — no additional recruitment needed)"

causal_method = getattr(causal, "method_used", "unknown")
causal_caveat = (
    "**⚠️ Caveat**: causal discovery ran in a simplified correlation-only fallback mode "
    "(`causal-learn` was not installed), not the full DP-PC algorithm. The C1 Pathway "
    "Sufficiency score below should not be treated as fully reliable until this is "
    "re-run with `pip install causal-learn` and `causal-learn` is actually used."
    if "fallback" in str(causal_method).lower() else
    "Causal discovery ran using the full DP-PC algorithm (not a fallback)."
)
solver_caveat = (
    "**⚠️ Caveat**: the recruitment-plan optimizer used a greedy fallback heuristic "
    "(`pulp` was not installed), not the true MILP solver — recruitment numbers below "
    "may not be cost-optimal. Re-run with `pip install pulp` for the optimal plan."
    if "greedy" in str(plan.solver_status).lower() else
    "The recruitment plan was solved using the true MILP optimizer (not a fallback)."
)

_research_report = f"""# FIDES on MIMIC-IV — Research Report ({_args.mode} cohort)

*Auto-generated from this run's actual results — {time.strftime("%Y-%m-%d %H:%M:%S")}*

---

## 1. What this run tested

Mode: **{_args.mode}** ({"small sample, fine-grained language — designed to reveal insufficiency" if _args.mode == "insufficient" else "larger sample, English/Non-English bucketed language — designed to resolve insufficiency"})

Cohort: {df.shape[0]:,} MIMIC-IV admissions, target = `{TARGET_NAME}` (ICD prefixes: {", ".join(ICD_PREFIXES)}), domain = `{DOMAIN}`, use case = `{USE_CASE}`.

Target distribution: {df[TARGET_NAME].value_counts().to_dict()} ({_fmt_pct(df[TARGET_NAME].mean())} positive)

---

## 2. Stage-by-stage results

### Stage 0 — Research Specification
- Protected attributes detected: {", ".join(s.attribute for s in spec.protected_attributes) or "(none)"}
- Proxy variables detected: {", ".join(p.attribute for p in spec.proxy_variables) or "(none)"}
- Legitimate causal paths defined: {len(spec.legitimate_paths)}
- Illegitimate causal paths defined: {len(spec.illegitimate_paths)}

### Stage 1 — HIPAA De-identification
- Dataset shape after cleaning: {audit['original_shape']}
- Residual PHI warnings: {audit['residual_phi_warnings']}
- HIPAA-safe: {"YES" if audit['hipaa_safe'] else "NO"}

### Stage 2 — Causal Discovery
- Method used: `{causal_method}`
- Causal edges found: {len(causal.edges)}
- Path validation: {dict(causal.path_validation)}
- {causal_caveat}

### Stage 3 — Four-Condition CDS Assessment

| Condition | Score | Weight |
|---|---|---|
| C1 Pathway Sufficiency | {cds.condition_scores.get('pathway', 0):.3f} | {spec.fairness_weights.get('pathway', 0):.2f} |
| C2 Statistical Sufficiency | {cds.condition_scores.get('statistical', 0):.3f} | {spec.fairness_weights.get('statistical', 0):.2f} |
| C3 Phenotypic Coverage | {cds.condition_scores.get('coverage', 0):.3f} | {spec.fairness_weights.get('coverage', 0):.2f} |
| C4 Intersectional Sufficiency | {cds.condition_scores.get('intersectional', 0):.3f} | {spec.fairness_weights.get('intersectional', 0):.2f} |
| **Overall CDS Score** | **{cds.cds_score:.3f}** | (threshold: {0.75 if USE_CASE=='research' else 0.85:.2f}) |

**Decision: {"PASS" if cds.threshold_met else "FAIL"}**

Insufficiency masking flags ({len(cds.insufficiency_masking_flags)} total, showing up to 10):
{_top_masking_flags(cds.insufficiency_masking_flags)}

Recommendations from the tool:
{chr(10).join(f"- {r}" for r in cds.recommendations) if cds.recommendations else "- (none)"}

### Stage 4 — Minimum Intervention Plan
- Total new patients recommended: {plan.total_new_patients:,}
- Estimated recruitment cost: ${plan.estimated_cost_usd:,}
- Solver status: {plan.solver_status}
- {solver_caveat}

Top recruitment priorities:
{_top_recruitment_groups(plan.groups)}

### Stage 6 — Deployment Fitness Certificate
- **Verdict: {cert.verdict}**
- CDS Score: {cert.cds_score_after:.3f} / 1.000
- HIPAA compliant: {"YES" if cert.hipaa_compliant else "NO"}
- FDA SaMD compliant: {"YES" if cert.fda_samd_compliant else "NO"}
- EU AI Act Art.10 compliant: {"YES" if cert.eu_ai_act_compliant else "NO"}
- Differential privacy: ε={cert.dp_epsilon:.3f}, δ=1e-6
- Re-identification risk: {cert.reidentification_risk:.4f} (safe threshold: <0.09)
- k-anonymity achieved: k={getattr(cert, 'k_anonymity', 'N/A')}
- Conditions for full approval: {", ".join(cert.conditions) if cert.conditions else "(none — fully approved)"}

---

## 3. Plain-language summary

This run assessed whether the **{_args.mode}** MIMIC-IV cohort ({df.shape[0]:,} patients) contains enough data,
across enough demographic subgroups and their combinations, to trust a fairness claim about a
`{TARGET_NAME}` prediction model trained on it. The result was **{cert.verdict}**, with an overall
sufficiency score of **{cds.cds_score:.3f}** against a required threshold of **{0.75 if USE_CASE=='research' else 0.85:.2f}**.

{"The dataset did NOT contain enough patients in enough demographic sub-combinations (see the insufficiency masking flags above) to reliably confirm fairness — this is a legitimate finding about data sufficiency, not a bug." if not cds.threshold_met else "The dataset DID contain enough patients across demographic sub-combinations to meet the sufficiency bar for this use case."}

---

*Generated automatically by `test_fides_mimic.py --mode {_args.mode}`. Full raw console log: `{os.path.basename(_report_path)}`.*
"""

_research_report_path = os.path.join(OUTPUT_DIR, f"RESEARCH_REPORT_{_args.mode}_{_run_stamp}.md")
with open(_research_report_path, "w", encoding="utf-8") as f:
    f.write(_research_report)

print(f"  Research report (plain-language): {_research_report_path}\n")

sys.stdout = sys.__stdout__
_report_file.close()
