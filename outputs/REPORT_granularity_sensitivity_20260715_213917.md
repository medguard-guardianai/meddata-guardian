# FIDES — Attribute Granularity Sensitivity Test

*Auto-generated 2026-07-15 21:39:58*

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
| Diabetes — Insurance-Mediated Care Access | Raw (20+ MIMIC categories) | 0.441 | 0.025 | 0.584 | FAIL | REJECTED |
| Diabetes — Insurance-Mediated Care Access | Bucketed (English / Non-English) | 0.768 | 0.025 | 0.658 | FAIL | CONDITIONALLY_APPROVED |
| Heart Disease — Admission-Urgency-Mediated Racial Disparity | Raw (20+ MIMIC categories) | 0.465 | 0.024 | 0.629 | FAIL | REJECTED |
| Heart Disease — Admission-Urgency-Mediated Racial Disparity | Bucketed (English / Non-English) | 0.796 | 0.024 | 0.702 | FAIL | CONDITIONALLY_APPROVED |

## Per-topic effect of bucketing

### Diabetes — Insurance-Mediated Care Access
- CDS score change from bucketing: +0.074 (0.584 → 0.658)
- C2 Statistical change: +0.327
- C4 Intersectional change: +0.000
- Verdict change: REJECTED → CONDITIONALLY_APPROVED
- Causal findings (confirmed/disputed paths) identical between raw and bucketed: YES — bias detection unaffected by this design choice

### Heart Disease — Admission-Urgency-Mediated Racial Disparity
- CDS score change from bucketing: +0.073 (0.629 → 0.702)
- C2 Statistical change: +0.331
- C4 Intersectional change: +0.000
- Verdict change: REJECTED → CONDITIONALLY_APPROVED
- Causal findings (confirmed/disputed paths) identical between raw and bucketed: YES — bias detection unaffected by this design choice


## Interpretation

If bucketing raises C2/C4 substantially while leaving the Stage 2 causal findings
(which paths were confirmed/disputed) unchanged, that supports the claim that the
original FAIL verdicts were driven by categorical granularity of one variable, not
by the underlying bias signal or overall data quality. If the verdict still fails
even after bucketing, the bottleneck lies elsewhere (sample size, race/insurance
sparsity), not language.

---
*Raw per-run logs: see `granularity_<topic>__<mode>_20260715_213917.txt` files in outputs/.*
