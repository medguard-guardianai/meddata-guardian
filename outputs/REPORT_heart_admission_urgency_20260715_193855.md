# FIDES Research Report — Heart Disease — Admission-Urgency-Mediated Racial Disparity

*Auto-generated 2026-07-15 19:39:14*

## Research Question

Does admission urgency (emergency vs. elective) mediate racial disparity in in-hospital mortality for heart disease patients?

## Cohort

5,000 MIMIC-IV admissions, target = `heart_disease` (ICD prefixes: 410, 411, 412, 413, 414, I20, I21, I22, I23, I24, I25).
Target distribution: {0: 4003, 1: 997}

## Stage 0 — Hypothesized Causal Structure

Legitimate path (expected to be biologically justified):
- `age → arterial_stiffness → length_of_stay` — Clinical: older cardiac patients have more complex, legitimately longer admissions

Illegitimate paths (hypothesized bias, should NOT be confirmed by real data in a fair dataset):
- `race → admit_type → hospital_expire_flag` — Disparity: differential emergency-vs-elective admission routing by race, proxy for delayed/missed preventive cardiology care
- `insurance → length_of_stay → hospital_expire_flag` — Proxy: under-insured patients discharged earlier than clinically ideal, elevating mortality risk

## Stage 1 — HIPAA De-identification

HIPAA-safe: YES, residual PHI warnings: 0

## Stage 2 — Where Bias Was (or Wasn't) Found in the Real Data

Causal discovery method: `dp_pc` (full DP-PC algorithm)

- `age → arterial_stiffness → length_of_stay` — **NOT confirmed (disputed) in real data**
- `race → admit_type → hospital_expire_flag` — **CONFIRMED in real data**
- `insurance → length_of_stay → hospital_expire_flag` — **CONFIRMED in real data**

**What this means**: for each hypothesized path above, FIDES checked whether that exact
relationship actually shows up in the real MIMIC data's causal structure. A path marked
CONFIRMED means the data supports that hypothesis actually happening; DISPUTED means the
real data does NOT show that specific relationship (which, for an *illegitimate* path, is
the reassuring outcome — it means the suspected bias mechanism isn't showing up this way).



## Stage 3 — Data Sufficiency Scorecard

| Condition | Score |
|---|---|
| C1 Pathway Sufficiency | 0.983 |
| C2 Statistical Sufficiency | 0.316 |
| C3 Phenotypic Coverage | 0.692 |
| C4 Intersectional Sufficiency | 0.229 |
| **Overall CDS Score** | **0.641** (threshold: 0.75) |

Decision: **FAIL**

Example under-powered subgroups (can't reliably detect disparities in these groups):
- INSUFFICIENCY MASKING: subgroup 'age=Other' has Power=0.362 < 0.80 (n=15, n*=200, d=0.166). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Portuguese' has Power=0.151 < 0.80 (n=17, n*=200, d=0.090). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=French' has Power=0.025 < 0.80 (n=1, n*=200, d=0.000). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Arabic' has Power=0.063 < 0.80 (n=9, n*=200, d=0.057). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Somali' has Power=0.581 < 0.80 (n=3, n*=200, d=0.499). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Polish' has Power=0.581 < 0.80 (n=3, n*=200, d=0.499). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Khmer' has Power=0.423 < 0.80 (n=2, n*=200, d=0.499). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Korean' has Power=0.705 < 0.80 (n=4, n*=200, d=0.499). Disparities may be undetectable — subgroup is under-powered.

## Stage 4 — Recruitment Needed to Fix Gaps

8,236 additional patients recommended, estimated cost $12,354,000.
Solver: Optimal (true MILP optimum)

## Stage 6 — Final Verdict

**CONDITIONALLY_APPROVED** — CDS 0.641, HIPAA=YES, FDA SaMD=NO, EU AI Act=NO

## Conclusion for this topic

This dataset does not currently contain enough patients across the demographic combinations relevant to this research question to reliably confirm or refute the hypothesized bias pathways. The FAIL verdict is a statement about data sufficiency, not a claim that bias does or doesn't exist.

---
*Raw stage-by-stage log: `topic_heart_admission_urgency_20260715_193855.txt`*
