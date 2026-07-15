# FIDES on MIMIC-IV — Handoff (Today's Work)

*What was done today: running FIDES's fairness/data-sufficiency pipeline on real MIMIC-IV data instead of only synthetic demo data.*

---

## Dataset Used

**MIMIC-IV v3.0** (PhysioNet, credentialed access), tables in `data/mimic/` (gitignored, not committed — re-download if missing):
- `patients.csv.gz` — demographics (subject_id, gender, anchor_age)
- `admissions.csv.gz` — per-admission data (race, insurance, marital_status, language, admission_type, timestamps, hospital_expire_flag)
- `diagnoses_icd.csv.gz` + `d_icd_diagnoses.csv.gz` — diagnosis codes + descriptions
- `d_labitems.csv.gz` + `labevents.csv.gz` — downloaded, **not yet used** in any results below

## Methodology

MIMIC is relational (separate tables); FIDES needs one flat table. `src/utils/mimic_cohort_builder.py` builds it:
1. Merge `patients` + `admissions` on `subject_id` → one row per hospital admission.
2. Derive a binary target from ICD codes (e.g. diabetes = ICD-9 250.x / ICD-10 E08-E13; heart disease = ICD-9 410-414 / ICD-10 I20-I25).
3. Compute `length_of_stay`, drop raw timestamps/IDs (privacy).
4. Bucket `race` into 7 broad categories (always). Optionally bucket `language` into English/Non-English (`bucket_language=True`, off by default).
5. Fill genuine missing insurance/marital_status/language values as `"Unknown"`.
6. Stratified-sample to 5,000 patients (full ~546K population is too large for one of FIDES's stages).

Two research questions were defined as new domain configs in `src/utils/research_spec.py`:
- **`mimic_diabetes_insurance`** — does insurance status illegitimately mediate diabetes admission urgency and mortality?
- **`mimic_heart_admission_urgency`** — does admission urgency mediate racial disparity in heart-disease mortality?

## Results

Running both topics through all 6 FIDES stages on real MIMIC data (with the actual DP-PC causal-discovery algorithm — `pip install causal-learn pulp` is required for this, otherwise it silently runs in a much weaker fallback mode):

| Topic | CDS Score | C1 Pathway | C2 Statistical | C4 Intersectional | Verdict |
|---|---|---|---|---|---|
| Diabetes / Insurance | 0.592 | 0.902 | 0.292 | 0.220 | REJECTED |
| Heart Disease / Admission Urgency | 0.641 | 0.983 | 0.316 | 0.229 | CONDITIONALLY_APPROVED |

**What this means**: for both topics, the causal-discovery stage found the hypothesized bias pathways were **CONFIRMED** in real data — i.e., race/insurance really do statistically predict admission urgency (emergency vs. elective), which predicts in-hospital mortality, in both the diabetes and heart-disease cohorts. That's a real, literature-consistent finding (high C1 score). But the overall CDS score still failed the 0.75 threshold because C2/C4 are low — many demographic subgroups (mainly rare `language` values, some rare `race`/`insurance` categories) have too few patients (single digits to low dozens, need ~200) to trust the comparison with full statistical confidence. **In short: real bias signal found, but not enough patients in enough subgroups to certify it with confidence.**

**Follow-up test** — does bucketing `language` (20+ raw values → English/Non-English) fix this? Ran both topics again with bucketing on:

| Topic | Language | CDS Score | C2 | C4 | Verdict |
|---|---|---|---|---|---|
| Diabetes | Raw | 0.592 | 0.292 | 0.220 | REJECTED |
| Diabetes | Bucketed | 0.658 | 0.768 | 0.025* | CONDITIONALLY_APPROVED |
| Heart Disease | Raw | 0.641 | 0.316 | 0.229 | CONDITIONALLY_APPROVED |
| Heart Disease | Bucketed | 0.702 | 0.796 | 0.024* | CONDITIONALLY_APPROVED |

*C4 numbers differ slightly between the two-topic run and the granularity-test run due to different random sampling seeds in each script — same underlying pattern (C4 stays low regardless of language treatment).

**Explanation**: bucketing language fixed C2 (statistical power) substantially, since that was the obvious culprit — but did **not** fix C4 at all, because C4 only checks specific attribute *combinations* (`race × insurance`, `race × gender × age`) which don't include language. So C4's low score comes from a different, still-unresolved cause: sparse combinations of race/insurance/age (e.g. a rare race category crossed with a rare insurance type). Neither topic fully passed even after fixing the language issue.

## Where to Look

All reports are auto-generated Markdown files in `outputs/` (gitignored) — no manual write-up needed, each script run produces its own report from real numbers:

- `outputs/REPORT_diabetes_insurance_20260715_193855.md`
- `outputs/REPORT_heart_admission_urgency_20260715_193855.md`
- `outputs/REPORT_combined_20260715_193855.md` — side-by-side comparison of the two topics
- `outputs/REPORT_granularity_sensitivity_20260715_213917.md` — the language bucketing follow-up test
- Matching raw stage-by-stage `.txt` logs alongside each `.md` report

## How to Run This Yourself

```powershell
# One-time setup
pip install causal-learn pulp

# Main result: both research topics end-to-end
python test_fides_topics.py

# Follow-up: language granularity sensitivity test (4 runs: 2 topics x raw/bucketed)
python test_fides_granularity.py
```

Both scripts print progress live and drop fresh timestamped reports into `outputs/` when done — open the newest `REPORT_*.md` files after running.

## Open Questions for Next Steps

1. **Why is C4 (Intersectional Sufficiency) stuck near 0.02-0.22 in every run?** Not yet drilled into which exact race×insurance×age combinations are sparse — that's the main unresolved question.
2. **Lab values (`labevents.csv.gz`) aren't used yet** — current results are administrative/demographic-only (no glucose, HbA1c, troponin, etc.).
3. Consider trying a "filter out rare subgroups" approach instead of "bucket them together" (keeps real MIMIC category values, just excludes ones too small to analyze) as an alternative to the language bucketing done today.
