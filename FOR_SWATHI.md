# For Swathi: Data & Results Guide

**From:** Shrivarshini  
**To:** Swathi  
**Date:** July 18, 2026  

---

## WHAT YOU NEED TO PROVIDE

### Your MIMIC-IV Dataset

You have MIMIC-IV access locally. You need to **prepare 5 disease cohorts** and put them here:

```
results/disease_cohorts/
├── readmission_cohort.csv     (Cardiac patients)
├── sepsis_cohort.csv          (Sepsis patients)
├── pneumonia_cohort.csv       (Pneumonia patients)
├── aki_cohort.csv             (AKI patients)
└── stroke_cohort.csv          (Stroke patients)
```

### What Each CSV Should Contain

**Columns (required):**
```
race,sex,age,insurance,mortality,readmitted,comorbidities,los_days,admission_type,ef_percent,severity
```

**Data types:**
- `race`: String (Black, White, Asian, Hispanic)
- `sex`: String (M, F)
- `age`: Numeric (years)
- `insurance`: String (Medicare, Medicaid, Commercial, etc.)
- `mortality`: Binary (0 or 1)
- `readmitted`: Binary (0 or 1)
- `comorbidities`: Numeric (count 0-10)
- `los_days`: Numeric (length of stay in days)
- `admission_type`: String (Emergency, Urgent, Elective)
- `ef_percent`: Numeric (ejection fraction %, for cardiac)
- `severity`: Numeric (0-100, clinical severity score)

**Expected sizes:**
- readmission_cohort.csv: ~5000 rows
- sepsis_cohort.csv: ~2000 rows
- pneumonia_cohort.csv: ~3000 rows
- aki_cohort.csv: ~2500 rows
- stroke_cohort.csv: ~1500 rows

### How to Prepare CSVs from Raw MIMIC

See HANDOFF.md § "Directory Setup" → "STEP 1.3: Load YOUR MIMIC-IV data"

It has Python code to:
1. Filter MIMIC admissions by disease (ICD-9 codes)
2. Map columns (ethnicity → race, gender → sex, etc.)
3. Handle missing values
4. Save as CSV

---

## WHAT YOU'LL GET (Results)

### After Running Causal Discovery Pipeline

**Output file:** `results/fides_real_causal_discovery_results.json`

**What it contains:**
```json
{
  "cardiac": {
    "race": {
      "cds_score": 0.672,
      "verdict": "REJECTED",
      "c1_score": 0.58,
      "c2_score": 0.71,
      "c3_score": 0.68,
      "c4_score": 0.52,
      "c5_score": 0.65,
      "underpowered_subgroups": ["Black (n=340)", "Asian (n=280)"],
      "remediation_cost_dollars": 11700000,
      "remediation_patients_needed": 7800
    },
    "insurance": { ... },
    "sex": { ... },
    "age": { ... }
  },
  "sepsis": { ... },
  "pneumonia": { ... },
  "aki": { ... },
  "stroke": { ... }
}
```

**What this means:**
- 20 validations total (5 diseases × 4 demographics)
- Each has a CDS score (0-1, where ≥0.75 = passes)
- REJECTED: Dataset has bias or power issues
- PASSED: Dataset is fit for training
- Shows which subgroups are underpowered
- Estimates cost to fix (recruit more patients)

### After Running Ablation Study

**Output file:** `ablation_study/results/ablation_study_results.json`

**What it contains:**
```json
{
  "detection_rates": {
    "c1_only": {"failing_datasets": 15, "detection_rate": 0.75},
    "c1_c2": {"failing_datasets": 15, "detection_rate": 0.75},
    "c1_c3": {"failing_datasets": 16, "detection_rate": 0.80},
    "c1_c4": {"failing_datasets": 20, "detection_rate": 1.00},
    "c1_c5_full": {"failing_datasets": 20, "detection_rate": 1.00}
  },
  "condition_contributions": {
    "c1_only": {"incremental_contribution": 0.75},
    "c1_c2": {"incremental_contribution": 0.05},
    "c1_c3": {"incremental_contribution": 0.12},
    "c1_c4": {"incremental_contribution": 0.08},
    "c1_c5_full": {"incremental_contribution": 0.05}
  }
}
```

**What this means:**
- Shows how much each condition contributes to detecting bias
- C1 alone catches 75% of problems
- Adding C2 (causal) adds 5%
- Adding C3 (phenotypic) adds 12%
- Adding C4 (power) adds 8%
- Adding C5 (FM bias) adds 5%
- **Key insight:** C4 (power analysis) is most important

### After Running Error Analysis

**Output file:** `ablation_study/results/error_analysis_report.json`

**What it contains:**
- False negatives: FIDES catches but other methods miss
- False positives: Other methods catch but FIDES passes
- Root cause analysis: Why did it fail?
- Pattern analysis: Which demographics fail most?

**Example finding:**
```
"patterns": [
  "POWER_GAPS_DOMINANT: 8/10 failures due to insufficient power",
  "RACE_DIMENSION_PROBLEMATIC: 7 failures vs 2 for insurance",
  "DISEASE_PATTERN: STROKE has 3/4 dimensions failing"
]
```

### After Running Meditron 7B

**Output file:** `ablation_study/results/meditron_7b_results.json`

**What it contains:**
```json
{
  "scenarios_tested": {
    "cardiac_STEMI_with_Reduced_EF": {
      "escalation_rates": {
        "Black": 0.62,
        "White": 0.85,
        "Asian": 0.78,
        "Hispanic": 0.71
      },
      "max_gap_pp": 23.0,
      "is_biased": true
    },
    ... (4 more scenarios)
  },
  "c5_score": 0.72,
  "verdict": "FAIL",
  "bias_prevalence": 0.60
}
```

**What this means:**
- Tested FM on 5 clinical scenarios
- Measured escalation (treatment) rate for each demographic
- Gap of 23pp (percentage points) between Black (62%) and White (85%)
- C5 score 0.72 = Condition 5 FAILED (bias detected)
- 60% of scenarios show demographic bias

---

## CURRENT STATE (What We Have Right Now)

### Code: ✅ READY
- All FIDES implementations complete (src/fides/)
- All pipelines written and tested (experiments/, ablation_study/)
- Paper template created (FIDES-PAPER-FINAL.tex)
- Setup infrastructure complete (requirements.txt, setup.sh, validate.py)

### Data: ❌ WAITING FOR YOU
- MIMIC-IV CSVs not in repo (you provide locally)
- .gitignore prevents data from being pushed (correct)
- Results directory empty (will be populated when you run)

### Results: ❌ NOT GENERATED YET
- No CDS scores (you need to run causal discovery)
- No ablation results (you need to run ablation study)
- No error analysis (you need to run error analysis)
- No FM bias results (you need to run Meditron)

### Paper: ❌ NOT WRITTEN YET
- Template exists, content is empty
- You'll add:
  - Methods section (explaining 5 conditions)
  - Results section (tables + figures from JSON outputs)
  - Discussion section (implications + limitations)

---

## WHAT WE'RE GETTING (Final Output)

### From Causal Discovery
✅ **20 CDS scores** (5 diseases × 4 demographics)  
✅ **20 verdicts** (PASSED, CONDITIONALLY_APPROVED, REJECTED)  
✅ **Power analysis** (which subgroups need more data)  
✅ **Remediation estimates** (cost to fix each problem)  

### From Ablation Study
✅ **Condition contributions** (how much each adds to detection)  
✅ **Detection rates** (percentage of datasets caught)  
✅ **Error patterns** (why things fail)  

### From Error Analysis
✅ **Baseline comparison** (vs Gap Analysis, Power, Fairlearn)  
✅ **Disagreements** (where FIDES differs from baselines)  
✅ **Root cause patterns** (power gaps, demographic issues, etc.)  

### From Meditron 7B
✅ **Real FM bias measurements** (escalation rates by demographic)  
✅ **Condition 5 score** (whether FM is biased)  
✅ **Clinical evidence** (concrete examples of FM bias)  

### From Paper Writing
✅ **8-10 page Methods section** (explaining all 5 conditions)  
✅ **Results section** (3 tables + 4 figures from your data)  
✅ **Discussion section** (implications + limitations)  
✅ **4 publication figures** (heatmaps, bar charts, comparisons)  

---

## Timeline (What Happens When)

| Timeline | You Do | You Get |
|----------|--------|---------|
| Day 1-2 | Load MIMIC data (3 hrs) | 5 CSVs in results/disease_cohorts/ |
| Day 2 | Run causal discovery (3 hrs) | fides_real_causal_discovery_results.json |
| Day 3 | Run ablation study (2 hrs) | ablation_study_results.json |
| Day 4 | Run Meditron (2-3 hrs) | meditron_7b_results.json |
| Day 5-6 | Write paper (8 hrs) | Methods, Results, Discussion complete |
| Day 7 | Polish & visualizations (4 hrs) | 4 figures + final paper |

**Total:** ~25 hours of work over 1 week

---

## NEXT STEPS FOR YOU

1. **Read HANDOFF.md** (top to bottom, 30 min)
2. **Run setup_environment.sh** (1 min)
3. **Prepare 5 disease cohorts** from your MIMIC-IV (2-3 hours)
4. **Follow HANDOFF.md day-by-day** (rest of the week)
5. **Use Claude AI to help write paper** (it will assist each section)
6. **Submit to AAAI AISI track** (September 2026)

---

## Questions You Might Have

**Q: Do I need to modify the code?**  
A: No. Just load your MIMIC data and run the pipelines. All code is ready.

**Q: What if my MIMIC data columns have different names?**  
A: HANDOFF.md § "Load YOUR MIMIC-IV data" has code to rename columns.

**Q: Can I use more than 5 diseases?**  
A: Yes! More diseases = stronger paper. See HANDOFF.md for which diseases to add.

**Q: What if I don't have GPU for real Meditron?**  
A: Use mock Meditron (still publishable, just note it in paper). Or rent cloud GPU ($2).

**Q: How do I write the paper?**  
A: Tell Claude AI what you need: "I have these results, help me write Methods section." HANDOFF explains exactly what to say.

**Q: What if something fails?**  
A: Run `python validate_setup.py` to check environment. HANDOFF has troubleshooting.

---

## You're Ready to Start 🚀

Everything is in place. Just:
1. Load your data
2. Run 4 commands
3. Write paper with Claude
4. Submit

**Go get this paper to AAAI!**

