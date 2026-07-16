"""
FIDES — Two Research Topics on MIMIC-IV
Runs the full FIDES pipeline once per research idea (diabetes / heart disease),
each with its own hypothesized causal structure, and generates a plain-language
report per topic plus a combined comparison report.

Usage: python test_fides_topics.py
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
from src.utils.mimic_cohort_builder   import build_cohort

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_run_stamp = time.strftime("%Y%m%d_%H%M%S")

TOPICS = [
    {
        "key": "diabetes_insurance",
        "title": "Diabetes — Insurance-Mediated Care Access",
        "target_name": "diabetes",
        "icd_prefixes": ["250", "E08", "E09", "E10", "E11", "E12", "E13"],
        "domain": "mimic_diabetes_insurance",
        "use_case": "research",
        "hypothesis": (
            "Does insurance status illegitimately mediate diabetes admission urgency "
            "and mortality risk, independent of legitimate clinical need (age/comorbidity)?"
        ),
    },
    {
        "key": "heart_admission_urgency",
        "title": "Heart Disease — Admission-Urgency-Mediated Racial Disparity",
        "target_name": "heart_disease",
        "icd_prefixes": ["410", "411", "412", "413", "414", "I20", "I21", "I22", "I23", "I24", "I25"],
        "domain": "mimic_heart_admission_urgency",
        "use_case": "research",
        "hypothesis": (
            "Does admission urgency (emergency vs. elective) mediate racial disparity "
            "in in-hospital mortality for heart disease patients?"
        ),
    },
]

SAMPLE_SIZE = 5000


def run_topic(topic):
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("\n" + "=" * 60)
    log(f"  RESEARCH TOPIC: {topic['title']}")
    log(f"  Hypothesis: {topic['hypothesis']}")
    log("=" * 60)

    df = build_cohort(
        icd_prefixes=topic["icd_prefixes"], target_name=topic["target_name"],
        sample_size=SAMPLE_SIZE, bucket_language=False, include_labs=True,
    )
    log(f"\nCohort built: {df.shape[0]} admissions, {df.shape[1]} columns")
    log(f"Target distribution: {df[topic['target_name']].value_counts().to_dict()}")

    cohort_csv_path = os.path.join("data", "mimic", f"cohort_{topic['key']}.csv")
    df.to_csv(cohort_csv_path, index=False)
    log(f"Cohort CSV saved: {cohort_csv_path}")

    log("\n" + "-" * 60)
    log("STAGE 0 — Research Specification")
    log("-" * 60)
    spec = build_research_spec(
        domain=topic["domain"], target_variable=topic["target_name"], target_type="binary",
        use_case=topic["use_case"], columns=list(df.columns), intent=topic["hypothesis"],
    )
    log(spec_summary(spec))

    log("\n" + "-" * 60)
    log("STAGE 1 — HIPAA Ingestion")
    log("-" * 60)
    clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol=f"IRB-2024-{topic['key'].upper()}")
    log(f"Audit: hipaa_safe={audit['hipaa_safe']}, residual_phi_warnings={audit['residual_phi_warnings']}")

    log("\n" + "-" * 60)
    log("STAGE 2 — Causal Discovery")
    log("-" * 60)
    causal = run_causal_discovery(clean_df, spec, epsilon=1.0)
    log(f"Method used: {getattr(causal, 'method_used', 'unknown')}")
    log(f"Causal edges found: {len(causal.edges)}")
    for src, tgt in causal.edges:
        log(f"  {src} -> {tgt}")
    log("Path validation:")
    for path, status in causal.path_validation.items():
        log(f"  [{status.upper()}]  {path}")

    log("\n" + "-" * 60)
    log("STAGE 3 — CDS Assessment")
    log("-" * 60)
    assessor = CDSAssessor(clean_df, spec, causal)
    cds = assessor.assess()
    log(f"C1 Pathway        : {cds.condition_scores.get('pathway', 0):.3f}")
    log(f"C2 Statistical    : {cds.condition_scores.get('statistical', 0):.3f}")
    log(f"C3 Coverage       : {cds.condition_scores.get('coverage', 0):.3f}")
    log(f"C4 Intersectional : {cds.condition_scores.get('intersectional', 0):.3f}")
    log(f"CDS Score         : {cds.cds_score:.3f}  (threshold met: {cds.threshold_met})")
    log(f"Insufficiency masking flags: {len(cds.insufficiency_masking_flags)}")

    log("\n" + "-" * 60)
    log("STAGE 4 — Minimum Intervention Plan")
    log("-" * 60)
    plan = optimize_intervention(cds, spec)
    log(f"Total new patients needed: {plan.total_new_patients:,}")
    log(f"Estimated cost: ${plan.estimated_cost_usd:,}")
    log(f"Solver status: {plan.solver_status}")

    log("\n" + "-" * 60)
    log("STAGE 6 — Certificate")
    log("-" * 60)
    cert = build_certificate(
        dataset_hash=audit['original_hash'], spec_id=f"{topic['domain']}_{topic['target_name']}",
        use_case=topic["use_case"], audit_record=audit,
        cds_result_before=cds, cds_result_after=cds, gen_result=None,
        intervention_plan=plan, irb_protocol=f"IRB-2024-{topic['key'].upper()}",
        output_dir=OUTPUT_DIR,
    )
    log(f"Verdict: {cert.verdict}")
    log(f"HIPAA={cert.hipaa_compliant}  FDA={cert.fda_samd_compliant}  EU_AI_Act={cert.eu_ai_act_compliant}")

    raw_log_path = os.path.join(OUTPUT_DIR, f"topic_{topic['key']}_{_run_stamp}.txt")
    with open(raw_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return {
        "topic": topic, "df": df, "spec": spec, "audit": audit,
        "causal": causal, "cds": cds, "plan": plan, "cert": cert,
        "raw_log_path": raw_log_path,
    }


def render_topic_report(result):
    topic, df, spec, audit = result["topic"], result["df"], result["spec"], result["audit"]
    causal, cds, plan, cert = result["causal"], result["cds"], result["plan"], result["cert"]

    causal_method = getattr(causal, "method_used", "unknown")
    used_fallback_causal = "fallback" in str(causal_method).lower()
    used_fallback_solver = "greedy" in str(plan.solver_status).lower()

    bias_findings = []
    for path, status in causal.path_validation.items():
        verdict = "CONFIRMED in real data" if status == "confirmed" else "NOT confirmed (disputed) in real data"
        bias_findings.append(f"- `{path}` — **{verdict}**")

    masking_examples = "\n".join(f"- {f}" for f in cds.insufficiency_masking_flags[:8]) or "- (none)"

    report = f"""# FIDES Research Report — {topic['title']}

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

## Research Question

{topic['hypothesis']}

## Cohort

{df.shape[0]:,} MIMIC-IV admissions, target = `{topic['target_name']}` (ICD prefixes: {", ".join(topic['icd_prefixes'])}).
Target distribution: {df[topic['target_name']].value_counts().to_dict()}

## Stage 0 — Hypothesized Causal Structure

Legitimate path (expected to be biologically justified):
{chr(10).join(f"- `{p.path}` — {p.reason}" for p in spec.legitimate_paths) or "- (none)"}

Illegitimate paths (hypothesized bias, should NOT be confirmed by real data in a fair dataset):
{chr(10).join(f"- `{p.path}` — {p.reason}" for p in spec.illegitimate_paths) or "- (none)"}

## Stage 1 — HIPAA De-identification

HIPAA-safe: {"YES" if audit['hipaa_safe'] else "NO"}, residual PHI warnings: {audit['residual_phi_warnings']}

## Stage 2 — Where Bias Was (or Wasn't) Found in the Real Data

Causal discovery method: `{causal_method}`{"  ⚠️ fallback mode — see caveat below" if used_fallback_causal else " (full DP-PC algorithm)"}

{chr(10).join(bias_findings)}

**What this means**: for each hypothesized path above, FIDES checked whether that exact
relationship actually shows up in the real MIMIC data's causal structure. A path marked
CONFIRMED means the data supports that hypothesis actually happening; DISPUTED means the
real data does NOT show that specific relationship (which, for an *illegitimate* path, is
the reassuring outcome — it means the suspected bias mechanism isn't showing up this way).

{"**⚠️ Caveat**: causal discovery fell back to a simplified correlation-only method — these confirmed/disputed results should be treated as unreliable." if used_fallback_causal else ""}

## Stage 3 — Data Sufficiency Scorecard

| Condition | Score |
|---|---|
| C1 Pathway Sufficiency | {cds.condition_scores.get('pathway', 0):.3f} |
| C2 Statistical Sufficiency | {cds.condition_scores.get('statistical', 0):.3f} |
| C3 Phenotypic Coverage | {cds.condition_scores.get('coverage', 0):.3f} |
| C4 Intersectional Sufficiency | {cds.condition_scores.get('intersectional', 0):.3f} |
| **Overall CDS Score** | **{cds.cds_score:.3f}** (threshold: {0.75 if topic['use_case']=='research' else 0.85}) |

Decision: **{"PASS" if cds.threshold_met else "FAIL"}**

Example under-powered subgroups (can't reliably detect disparities in these groups):
{masking_examples}

## Stage 4 — Recruitment Needed to Fix Gaps

{plan.total_new_patients:,} additional patients recommended, estimated cost ${plan.estimated_cost_usd:,}.
Solver: {plan.solver_status}{"  ⚠️ fallback heuristic, not optimal" if used_fallback_solver else " (true MILP optimum)"}

## Stage 6 — Final Verdict

**{cert.verdict}** — CDS {cert.cds_score_after:.3f}, HIPAA={"YES" if cert.hipaa_compliant else "NO"}, FDA SaMD={"YES" if cert.fda_samd_compliant else "NO"}, EU AI Act={"YES" if cert.eu_ai_act_compliant else "NO"}

## Conclusion for this topic

{"This dataset does not currently contain enough patients across the demographic combinations relevant to this research question to reliably confirm or refute the hypothesized bias pathways. The FAIL verdict is a statement about data sufficiency, not a claim that bias does or doesn't exist." if not cds.threshold_met else "This dataset meets the sufficiency bar for this research question — the confirmed/disputed causal findings above can be reasonably trusted."}

---
*Raw stage-by-stage log: `{os.path.basename(result['raw_log_path'])}`*
"""
    path = os.path.join(OUTPUT_DIR, f"REPORT_{topic['key']}_{_run_stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def render_combined_report(results):
    rows = []
    for r in results:
        t, cds, cert = r["topic"], r["cds"], r["cert"]
        rows.append(
            f"| {t['title']} | {r['df'].shape[0]:,} | {cds.cds_score:.3f} | "
            f"{'PASS' if cds.threshold_met else 'FAIL'} | {cert.verdict} |"
        )

    report = f"""# FIDES — Combined Research Topics Comparison

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

Both research topics were run against the same real, unmodified MIMIC-IV administrative
cohort construction method (only the ICD-code target and hypothesized causal structure differ).

| Research Topic | N | CDS Score | Decision | Certificate Verdict |
|---|---|---|---|---|
{chr(10).join(rows)}

## Why this comparison matters

Running two distinct research hypotheses against the same real patient population tests
whether FIDES's sufficiency assessment is specific to one disease's cohort quirks, or reflects
a structural property of this MIMIC population's demographic granularity (rare languages,
fine-grained race categories) that would affect any research question asked of it.

"""
    for r in results:
        report += f"\n### {r['topic']['title']}\n{r['topic']['hypothesis']}\n\nSee: `REPORT_{r['topic']['key']}_{_run_stamp}.md`\n"

    path = os.path.join(OUTPUT_DIR, f"REPORT_combined_{_run_stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


if __name__ == "__main__":
    results = [run_topic(t) for t in TOPICS]
    topic_report_paths = [render_topic_report(r) for r in results]
    combined_path = render_combined_report(results)

    print("\n\n" + "=" * 60)
    print("  ALL TOPICS COMPLETE")
    print("=" * 60)
    for p in topic_report_paths:
        print(f"  Report: {p}")
    print(f"  Combined comparison: {combined_path}")
