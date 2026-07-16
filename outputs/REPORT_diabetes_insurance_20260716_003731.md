# FIDES Research Report — Diabetes — Insurance-Mediated Care Access

*Auto-generated 2026-07-16 00:51:17*

## Research Question

Does insurance status illegitimately mediate diabetes admission urgency and mortality risk, independent of legitimate clinical need (age/comorbidity)?

## Cohort

5,000 MIMIC-IV admissions, target = `diabetes` (ICD prefixes: 250, E08, E09, E10, E11, E12, E13).
Target distribution: {0: 3778, 1: 1222}

## Stage 0 — Hypothesized Causal Structure

Legitimate path (expected to be biologically justified):
- `age → comorbidity_burden → length_of_stay` — Clinical: older diabetic patients have more complications, legitimately longer stays
- `glucose → glycemic_severity → length_of_stay` — Clinical: poor glycemic control (high glucose/HbA1c) legitimately predicts longer, more complex admissions
- `glucose → diabetic_kidney_stress → creatinine` — Clinical: poor glycemic control legitimately damages kidney function over time (diabetic nephropathy)

Illegitimate paths (hypothesized bias, should NOT be confirmed by real data in a fair dataset):
- `race → insurance → admit_type` — Disparity: uninsured/Medicaid diabetic patients routed to emergency rather than elective admission, proxy for delayed outpatient diagnosis
- `insurance → admit_type → hospital_expire_flag` — Proxy: insurance status affects care urgency and thus mortality risk, independent of disease severity

## Stage 1 — HIPAA De-identification

HIPAA-safe: YES, residual PHI warnings: 0

## Stage 2 — Where Bias Was (or Wasn't) Found in the Real Data

Causal discovery method: `dp_pc` (full DP-PC algorithm)

- `age → comorbidity_burden → length_of_stay` — **NOT confirmed (disputed) in real data**
- `glucose → glycemic_severity → length_of_stay` — **NOT confirmed (disputed) in real data**
- `glucose → diabetic_kidney_stress → creatinine` — **NOT confirmed (disputed) in real data**
- `race → insurance → admit_type` — **CONFIRMED in real data**
- `insurance → admit_type → hospital_expire_flag` — **CONFIRMED in real data**

**What this means**: for each hypothesized path above, FIDES checked whether that exact
relationship actually shows up in the real MIMIC data's causal structure. A path marked
CONFIRMED means the data supports that hypothesis actually happening; DISPUTED means the
real data does NOT show that specific relationship (which, for an *illegitimate* path, is
the reassuring outcome — it means the suspected bias mechanism isn't showing up this way).



## Stage 3 — Data Sufficiency Scorecard

| Condition | Score |
|---|---|
| C1 Pathway Sufficiency | 0.904 |
| C2 Statistical Sufficiency | 0.292 |
| C3 Phenotypic Coverage | 0.181 |
| C4 Intersectional Sufficiency | 0.220 |
| **Overall CDS Score** | **0.500** (threshold: 0.75) |

Decision: **FAIL**

Example under-powered subgroups (can't reliably detect disparities in these groups):
- INSUFFICIENCY MASKING: subgroup 'age=Other' has Power=0.426 < 0.80 (n=13, n*=200, d=0.211). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Chinese' has Power=0.109 < 0.80 (n=70, n*=200, d=0.037). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Persian' has Power=0.030 < 0.80 (n=8, n*=200, d=0.013). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Arabic' has Power=0.309 < 0.80 (n=12, n*=200, d=0.181). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Kabuverdianu' has Power=0.575 < 0.80 (n=50, n*=200, d=0.131). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Italian' has Power=0.218 < 0.80 (n=6, n*=200, d=0.207). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Hindi' has Power=0.465 < 0.80 (n=2, n*=200, d=0.569). Disparities may be undetectable — subgroup is under-powered.
- INSUFFICIENCY MASKING: subgroup 'age=Thai' has Power=0.754 < 0.80 (n=4, n*=200, d=0.569). Disparities may be undetectable — subgroup is under-powered.

## Stage 4 — Recruitment Needed to Fix Gaps

8,550 additional patients recommended, estimated cost $12,825,000.
Solver: Optimal (true MILP optimum)

## Stage 6 — Final Verdict

**REJECTED** — CDS 0.500, HIPAA=YES, FDA SaMD=NO, EU AI Act=NO

## Conclusion for this topic

This dataset does not currently contain enough patients across the demographic combinations relevant to this research question to reliably confirm or refute the hypothesized bias pathways. The FAIL verdict is a statement about data sufficiency, not a claim that bias does or doesn't exist.

---
*Raw stage-by-stage log: `topic_diabetes_insurance_20260716_003731.txt`*
