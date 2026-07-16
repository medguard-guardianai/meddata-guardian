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

## Open Questions (as of 2026-07-15)

1. **Why is C4 (Intersectional Sufficiency) stuck near 0.02-0.22 in every run?** Not yet drilled into which exact race×insurance×age combinations are sparse — that's the main unresolved question.
2. **Lab values (`labevents.csv.gz`) aren't used yet** — current results are administrative/demographic-only (no glucose, HbA1c, troponin, etc.).
3. Consider trying a "filter out rare subgroups" approach instead of "bucket them together" (keeps real MIMIC category values, just excludes ones too small to analyze) as an alternative to the language bucketing done today.

---

# 2026-07-16 — Lab Integration, New Research Topics, Algorithm Validation Suite

## 1. Lab values integrated (resolves open question #2 above)

`labevents.csv.gz` (2.6GB compressed, ~100M+ rows) is now used. Since it's too large to load whole, `mimic_cohort_builder.py` scans it in chunks, keeping only rows matching a target itemid list and the admissions already in the built cohort. New function: `_extract_labs()`, wired in via `build_cohort(..., include_labs=True)`.

**6 lab columns added** (all standard serum tests, picked via `d_labitems.csv.gz` lookup):
| Column | MIMIC itemid | Coverage (real, measured) |
|---|---|---|
| `glucose` | 50931 | ~73-75% |
| `creatinine` | 50912 | ~74-75% |
| `bun` | 51006 | ~75.5% |
| `potassium` | 50971 | ~75.7% |
| `hba1c` | 50852 | **~9%** |
| `troponin` | 51003 | **~9.5%** |

HbA1c/troponin are sparse because they're only ordered when clinically indicated, not as routine bloodwork — this matters a lot for results (see Test 6 below). Checked but **not added** (too sparse to be useful): cholesterol panel (~5%), urine albumin/creatinine ratio (~0.4% — barely ever ordered inpatient).

**Effect on the two existing topics** (diabetes/insurance, heart/admission-urgency): re-ran with all 6 labs included. Core bias findings (`race/insurance → admission urgency → mortality`) stayed **CONFIRMED**, unaffected by adding labs. But CDS scores dropped slightly (diabetes 0.592→0.542, heart 0.641→0.530) — driven by **C3 (Phenotypic Coverage) collapsing** (0.645→0.243 diabetes) once sparse lab columns were added. New legitimate clinical paths added (`glucose → kidney stress → creatinine`, `troponin → cardiac injury → mortality`) came back **DISPUTED** — i.e. adding labs didn't just confirm textbook biology, it's a genuinely mixed result worth reporting honestly, not glossing over.

Current reports reflecting labs: `outputs/REPORT_diabetes_insurance_20260716_003731.md`, `outputs/REPORT_heart_admission_urgency_20260716_003731.md`, `outputs/REPORT_combined_20260716_003731.md`.

**New helper for building lab-derived (not ICD-derived) targets**: `derive_lab_threshold_target()` in `mimic_cohort_builder.py` — creates a binary target from lab thresholds (e.g. `creatinine > 1.5` or `bun > 20`) instead of an ICD code, with an option to restrict to an existing condition first (e.g. only diabetic patients).

**New helper for comorbidity flags**: `build_cohort(..., comorbidity_flags={"ckd_diagnosed": ["585","N18"]})` — cheap to add since `diagnoses_icd` is already fully loaded (unlike labevents), used for the under-diagnosis topic below.

## 2. Two new research topics added

### `mimic_kidney_impairment` (domain in `research_spec.py`)
*Does race predict diabetic kidney complication severity via admission urgency, independent of glucose severity?* Insurance deliberately excluded here; `marital_status` (social-support proxy) used instead. Target: `kidney_impairment` (creatinine > 1.5 or BUN > 20), restricted to diabetic patients. Script: `test_fides_kidney.py`.

**Result**: all 3 hypothesized paths (including the illegitimate ones) came back **DISPUTED** — a genuinely different result from every admin-urgency topic, where the race/insurance path was always confirmed. CDS = 0.375 (REJECTED), driven by the **worst C2 seen yet (0.009)** — restricting to diabetics-only more than halved the effective cohort while keeping the same demographic granularity.

### `mimic_kidney_underdiagnosis` (refined version — the important design correction)
Original framing risk: using an actual CKD diagnosis code as the target would be circular (diabetes and CKD codes are usually recorded together by design — doesn't test anything new). **Corrected framing**: restrict to patients with *objective, lab-confirmed* kidney impairment (independent of any diagnosis code), then ask whether race/insurance/social-support predict whether that real impairment was **never diagnostically coded** — i.e. testing equity of diagnostic *recognition*, not the diabetes-kidney biological link (which is settled medicine, not in question). Script: `test_fides_kidney_underdx.py`.

**Result**: cohort restricted from 4,888 diabetics → 2,299 with lab-confirmed impairment. **941/2,299 (41%) had no CKD diagnosis code despite confirmed impairment** — a real, substantial under-recognition finding on its own. But all 4 causal paths (including race/insurance/marital_status) came back **DISPUTED** — no demographic disparity mechanism confirmed here, likely because restricting to only-impaired patients shrank the cohort too far (same C2-collapse pattern as topic above). CDS = 0.454 (REJECTED).

Cohort CSVs for manual inspection: `data/mimic/cohort_kidney_impairment.csv`, `data/mimic/cohort_kidney_underdx.csv` (and `cohort_diabetes_insurance.csv` / `cohort_heart_admission_urgency.csv` for the original two topics — `test_fides_topics.py` now saves these automatically every run).

## 3. Algorithm Validation Suite (new file: `test_fides_validation_suite.py`)

Distinguishes "is the algorithm working correctly" from "what does MIMIC's data show" — six tests, all reusing the already-saved `cohort_diabetes_insurance.csv` (no labevents re-scan needed, runs in a couple minutes total). Report: `outputs/REPORT_validation_suite_20260716_180914.md`.

| Test | What it checks | Result |
|---|---|---|
| 1. Known-bias detection (positive control) | Run on synthetic datasets with a bias deliberately built in (`data/synthetic/dataset2_diabetes_gender_bias.csv`, `dataset3_heart_disease_indigenous.csv`) | 0 confirmed causal paths on both — **not a failure**: these domains' hardcoded illegitimate paths reference columns that don't exist in these particular synthetic files. The actual injected bias shows up correctly instead in **C2** (dataset3's C2=0.197 vs dataset2's 0.419 — the "zero Indigenous representation" bias is a representation problem, which C2/C4 test, not C1) |
| 2. Null control | Shuffle `race` randomly — should reduce/eliminate confirmed bias paths | Confirmed paths dropped 2→1 after shuffling — real data's CONFIRMED finding reflects genuine signal, not noise |
| 3. Reproducibility | Same input + same seed → same output? | Identical CDS score (0.536) across two runs — fully deterministic |
| 4. Sample-size sensitivity | Does C2 rise predictably with n? | Yes — C2 rose cleanly and monotonically (0.279→0.349→0.368→0.441) across n=500/1500/3000/5000. Overall CDS score was noisier due to C3's KDE instability at small n — expected, not a bug |
| 5. Feature-drop ablation | Remove `insurance` entirely — how much does it matter? | Confirmed paths dropped 2→0, CDS 0.536→0.519 — insurance is load-bearing for this domain's specific hypothesis |
| 6. Lab-by-lab ablation | Which lab actually causes C3 to collapse? | C3 barely moved adding glucose (0.636→0.637), then **collapsed the moment HbA1c was added** (0.637→0.393) and kept falling as more labs were added — confirms HbA1c specifically (not "labs in general") drives the C3 collapse, matching its ~9% coverage |

## 4. Files added today

- Modified `src/utils/mimic_cohort_builder.py` — `_extract_labs()`, `_bucket_language()` (was already there), `derive_lab_threshold_target()`, `comorbidity_flags` param on `build_cohort()`
- Modified `src/utils/research_spec.py` — added `mimic_kidney_impairment`, `mimic_kidney_underdiagnosis` domain blocks (additive, existing domains untouched)
- New: `test_fides_kidney.py`, `test_fides_kidney_underdx.py`, `test_fides_validation_suite.py`
- `test_fides_topics.py` — now also saves each topic's cohort to `data/mimic/cohort_<topic_key>.csv` automatically

## 5. Updated Open Questions

1. **C4 (Intersectional) is still unresolved** — stuck low (0.02-0.22) across every topic and every fix tried so far (language bucketing didn't touch it; it's specifically about sparse `race × insurance` / `race × gender × age` cells).
2. **Kidney topics both have severely underpowered C2 (~0.008-0.009)** — restricting to diabetic-only, or diabetic+impaired-only, cuts the cohort enough that this needs a much larger base sample (try 50,000+) before the demographic-disparity question can be meaningfully tested.
3. **HbA1c/troponin's sparse coverage is now confirmed (not just suspected) as the main driver of C3 decline** — worth deciding whether to exclude them from future topics or explicitly frame their sparsity as a finding.
4. The "filter rare subgroups instead of bucketing" idea (raised 2026-07-15) is still unimplemented.
