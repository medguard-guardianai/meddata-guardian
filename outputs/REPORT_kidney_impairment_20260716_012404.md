# FIDES Research Report — Diabetic Kidney Impairment Severity

*Auto-generated 2026-07-16 01:27:41*

## Research Question

Does race predict diabetic kidney complication severity via admission urgency,
independent of legitimate glycemic severity (glucose)? Insurance was
deliberately excluded from this topic; marital_status (social support proxy)
is used as the secondary mediator instead.

## Cohort

4,303 diabetic admissions (restricted from a 20,000-admission base cohort),
target = `kidney_impairment` (creatinine > 1.5 mg/dL OR BUN > 20 mg/dL).
Target distribution: {1: 2299, 0: 2004}

## Stage 0 — Hypothesized Causal Structure

Legitimate path: ['glucose → diabetic_nephropathy_mechanism → kidney_impairment']
Illegitimate paths: ['race → admit_type → kidney_impairment', 'marital_status → length_of_stay → kidney_impairment']

## Stage 2 — Causal Discovery Results

Method: `dp_pc`, edges found: 27

- `glucose → diabetic_nephropathy_mechanism → kidney_impairment` — **DISPUTED**
- `race → admit_type → kidney_impairment` — **DISPUTED**
- `marital_status → length_of_stay → kidney_impairment` — **DISPUTED**

## Stage 3 — CDS Scorecard

| Condition | Score |
|---|---|
| C1 Pathway | 0.725 |
| C2 Statistical | 0.009 |
| C3 Coverage | 0.232 |
| C4 Intersectional | 0.187 |
| **CDS Score** | **0.375** (threshold 0.75) |

Decision: **FAIL**

## Stage 4 — Recruitment

8,055 additional patients needed, $12,082,500, solver: Optimal

## Stage 6 — Verdict

**REJECTED** — HIPAA=True, FDA=False, EU AI Act=False

---
*Cohort CSV: `data/mimic/cohort_kidney_impairment.csv`. Raw log: `mimic_run_report_kidney_20260716_012404.txt`.*
