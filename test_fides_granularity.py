"""
FIDES — Language Granularity Sensitivity Test
Runs both research topics (diabetes/insurance, heart disease/admission urgency)
twice each: once with MIMIC's raw language values (20+ categories), once with
language bucketed to English/Non-English — isolating whether categorization
granularity, not the underlying data or the causal findings, drives the
CDS sufficiency verdict.

Usage: python test_fides_granularity.py
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
LANGUAGE_MODES = [
    {"key": "raw", "label": "Raw (20+ MIMIC categories)", "bucket_language": False},
    {"key": "bucketed", "label": "Bucketed (English / Non-English)", "bucket_language": True},
]


def run_combo(topic, lang_mode):
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    combo_key = f"{topic['key']}__{lang_mode['key']}"
    log("\n" + "=" * 60)
    log(f"  TOPIC: {topic['title']}  |  LANGUAGE: {lang_mode['label']}")
    log("=" * 60)

    df = build_cohort(
        icd_prefixes=topic["icd_prefixes"], target_name=topic["target_name"],
        sample_size=SAMPLE_SIZE, bucket_language=lang_mode["bucket_language"],
    )
    log(f"Cohort built: {df.shape[0]} admissions, {df.shape[1]} columns")
    log(f"Target distribution: {df[topic['target_name']].value_counts().to_dict()}")

    spec = build_research_spec(
        domain=topic["domain"], target_variable=topic["target_name"], target_type="binary",
        use_case=topic["use_case"], columns=list(df.columns), intent=topic["hypothesis"],
    )
    clean_df, audit = ingest(df, spec, user_id="demo_user", irb_protocol=f"IRB-2024-{combo_key.upper()}")
    log(f"HIPAA: safe={audit['hipaa_safe']}, residual_phi_warnings={audit['residual_phi_warnings']}")

    causal = run_causal_discovery(clean_df, spec, epsilon=1.0)
    log(f"Causal method: {getattr(causal, 'method_used', 'unknown')}, edges: {len(causal.edges)}")
    for path, status in causal.path_validation.items():
        log(f"  [{status.upper()}]  {path}")

    assessor = CDSAssessor(clean_df, spec, causal)
    cds = assessor.assess()
    log(f"C1={cds.condition_scores.get('pathway',0):.3f}  C2={cds.condition_scores.get('statistical',0):.3f}  "
        f"C3={cds.condition_scores.get('coverage',0):.3f}  C4={cds.condition_scores.get('intersectional',0):.3f}  "
        f"CDS={cds.cds_score:.3f}  (threshold_met={cds.threshold_met})")

    plan = optimize_intervention(cds, spec)
    log(f"Recruitment needed: {plan.total_new_patients:,} patients, ${plan.estimated_cost_usd:,}, solver={plan.solver_status}")

    cert = build_certificate(
        dataset_hash=audit['original_hash'], spec_id=f"{topic['domain']}_{topic['target_name']}_{lang_mode['key']}",
        use_case=topic["use_case"], audit_record=audit,
        cds_result_before=cds, cds_result_after=cds, gen_result=None,
        intervention_plan=plan, irb_protocol=f"IRB-2024-{combo_key.upper()}",
        output_dir=OUTPUT_DIR,
    )
    log(f"Verdict: {cert.verdict}")

    raw_log_path = os.path.join(OUTPUT_DIR, f"granularity_{combo_key}_{_run_stamp}.txt")
    with open(raw_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return {
        "topic": topic, "lang_mode": lang_mode, "df": df, "spec": spec,
        "causal": causal, "cds": cds, "plan": plan, "cert": cert,
        "raw_log_path": raw_log_path,
    }


def render_comparison_report(results):
    rows = []
    for r in results:
        t, lm, cds, cert = r["topic"], r["lang_mode"], r["cds"], r["cert"]
        rows.append(
            f"| {t['title']} | {lm['label']} | {cds.condition_scores.get('statistical',0):.3f} | "
            f"{cds.condition_scores.get('intersectional',0):.3f} | {cds.cds_score:.3f} | "
            f"{'PASS' if cds.threshold_met else 'FAIL'} | {cert.verdict} |"
        )

    # Pair up raw vs bucketed per topic for delta commentary
    by_topic = {}
    for r in results:
        by_topic.setdefault(r["topic"]["key"], {})[r["lang_mode"]["key"]] = r

    deltas = []
    for topic_key, pair in by_topic.items():
        if "raw" in pair and "bucketed" in pair:
            raw_r, buck_r = pair["raw"], pair["bucketed"]
            title = raw_r["topic"]["title"]
            d_cds = buck_r["cds"].cds_score - raw_r["cds"].cds_score
            d_c2 = buck_r["cds"].condition_scores.get("statistical", 0) - raw_r["cds"].condition_scores.get("statistical", 0)
            d_c4 = buck_r["cds"].condition_scores.get("intersectional", 0) - raw_r["cds"].condition_scores.get("intersectional", 0)
            same_causal = raw_r["causal"].path_validation == buck_r["causal"].path_validation
            deltas.append(
                f"### {title}\n"
                f"- CDS score change from bucketing: {d_cds:+.3f} ({raw_r['cds'].cds_score:.3f} → {buck_r['cds'].cds_score:.3f})\n"
                f"- C2 Statistical change: {d_c2:+.3f}\n"
                f"- C4 Intersectional change: {d_c4:+.3f}\n"
                f"- Verdict change: {raw_r['cert'].verdict} → {buck_r['cert'].verdict}\n"
                f"- Causal findings (confirmed/disputed paths) identical between raw and bucketed: "
                f"{'YES — bias detection unaffected by this design choice' if same_causal else 'NO — see raw logs for details'}\n"
            )

    report = f"""# FIDES — Attribute Granularity Sensitivity Test

*Auto-generated {time.strftime("%Y-%m-%d %H:%M:%S")}*

## What this tests

Does the CDS sufficiency verdict depend on how finely the `language` attribute is
categorized, holding the underlying patients, the disease target, and the
hypothesized causal structure identical? Two treatments compared:
- **Raw**: MIMIC's actual 20+ language values, unmodified.
- **Bucketed**: collapsed to English / Non-English (same simplification already
  applied to `race` in every run).

## Results

| Research Topic | Language Treatment | C2 Statistical | C4 Intersectional | CDS Score | Decision | Certificate Verdict |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Per-topic effect of bucketing

{chr(10).join(deltas)}

## Interpretation

If bucketing raises C2/C4 substantially while leaving the Stage 2 causal findings
(which paths were confirmed/disputed) unchanged, that supports the claim that the
original FAIL verdicts were driven by categorical granularity of one variable, not
by the underlying bias signal or overall data quality. If the verdict still fails
even after bucketing, the bottleneck lies elsewhere (sample size, race/insurance
sparsity), not language.

---
*Raw per-run logs: see `granularity_<topic>__<mode>_{_run_stamp}.txt` files in outputs/.*
"""
    path = os.path.join(OUTPUT_DIR, f"REPORT_granularity_sensitivity_{_run_stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


if __name__ == "__main__":
    results = []
    for topic in TOPICS:
        for lang_mode in LANGUAGE_MODES:
            results.append(run_combo(topic, lang_mode))

    report_path = render_comparison_report(results)

    print("\n\n" + "=" * 60)
    print("  GRANULARITY SENSITIVITY TEST COMPLETE")
    print("=" * 60)
    print(f"  Comparison report: {report_path}")
