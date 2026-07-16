# FIDES Research Report — Diagnostic Under-Recognition of Kidney Complications

*Auto-generated 2026-07-16 12:24:18*

## Research Question

NOT "does diabetes cause kidney damage" (settled medical fact). Among diabetic
patients with OBJECTIVE, lab-confirmed kidney impairment (creatinine > 1.5 or
BUN > 20 — independent of any diagnosis code), does race, insurance, or
social support (marital_status) predict whether that impairment actually
gets clinically diagnosed/coded as CKD? This tests equity of diagnostic
recognition, not the existence of the diabetes-kidney link itself.

## Cohort Construction

1. Base diabetic cohort: 4,888 admissions (from a 20,000-admission sample)
2. Restricted to lab-confirmed kidney impairment: 2,299 admissions — this is the actual study population
3. Target = `undiagnosed_kidney_impairment`: 1 if impairment was NOT diagnostically coded despite being lab-confirmed present
Distribution: {0: 1358, 1: 941}

## Stage 0 — Hypothesized Causal Structure

Legitimate path: ['creatinine → impairment_severity → undiagnosed_kidney_impairment']
Illegitimate paths: ['race → admit_type → undiagnosed_kidney_impairment', 'insurance → admit_type → undiagnosed_kidney_impairment', 'marital_status → length_of_stay → undiagnosed_kidney_impairment']

## Stage 2 — Causal Discovery Results

Method: `dp_pc`, edges found: 16

- `creatinine → impairment_severity → undiagnosed_kidney_impairment` — **DISPUTED**
- `race → admit_type → undiagnosed_kidney_impairment` — **DISPUTED**
- `insurance → admit_type → undiagnosed_kidney_impairment` — **DISPUTED**
- `marital_status → length_of_stay → undiagnosed_kidney_impairment` — **DISPUTED**

## Stage 3 — CDS Scorecard

| Condition | Score |
|---|---|
| C1 Pathway | 0.959 |
| C2 Statistical | 0.008 |
| C3 Coverage | 0.202 |
| C4 Intersectional | 0.144 |
| **CDS Score** | **0.454** (threshold 0.75) |

Decision: **FAIL**

## Stage 4 — Recruitment

9,665 additional patients needed, $14,497,500, solver: Optimal

## Stage 6 — Verdict

**REJECTED** — HIPAA=True, FDA=False, EU AI Act=False

---
*Cohort CSV: `data/mimic/cohort_kidney_underdx.csv`. Raw log: `mimic_run_report_kidney_underdx_20260716_122036.txt`.*
