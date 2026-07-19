# Development Context — Clinical AI Bias Auditing Research

**Last updated:** June 8, 2026  
**Status:** Framework complete, ready for full implementation

---

## Current Situation

You are building three research papers for publication (AMIA/NeurIPS target):

1. **FIDES** — Pre-training dataset certification with four sufficiency conditions
2. **PrivateFM** — HIPAA-compliant local FM bias auditing 
3. **BiasIRL** — Bayesian IRL decision-level bias detection

All three are **novel, don't exist in published form**, and are complementary (each catches what others miss).

---

## What's Done

✅ Research problem statements written and validated against literature  
✅ Novelty verified (searched 2024-2026 papers, no direct competitors found)  
✅ Demo code exists: `experiments/foundation_model_bias/` — fully working foundation model bias detector in mock mode  
✅ Synthetic dataset with embedded bias: `synthetic_dataset.csv` — 2000 cardiac patients, racial pain underrecording, care pathway bias  
✅ RESEARCH_ROADMAP.md created with full implementation plan  

---

## What's Next (Priority Order)

### Phase 1: Data & Utilities (Start here)
- [ ] Create `src/utils/data_loading.py` — MIMIC-IV loader (or use synthetic for initial development)
- [ ] Create `src/utils/demographics.py` — handle race/sex/age/intersections properly
- [ ] Create `src/utils/reporting.py` — structured output formats (JSON, markdown, plots)
- [ ] Add type hints and docstrings to existing `experiments/foundation_model_bias/` code

### Phase 2: FIDES Implementation
- [ ] `src/fides/causal.py` — path-specific causal effect decomposition
- [ ] `src/fides/representational.py` — demographic gap detection
- [ ] `src/fides/phenotypic.py` — clinical presentation coverage analysis
- [ ] `src/fides/intersectional.py` — **insufficiency masking** (core novelty)
- [ ] `src/fides/certification.py` — structured certification report
- [ ] Validation: Run on synthetic_dataset.csv, print all four condition scores

### Phase 3: PrivateFM Implementation
- [ ] Extend `experiments/foundation_model_bias/fm_bias_detector.py` into `src/privatefm/`
- [ ] `src/privatefm/local_inference.py` — proper FM wrapper, batching, optimization
- [ ] `src/privatefm/guidelines.py` — embed ACC/AHA guidelines in prompts
- [ ] `src/privatefm/evaluation.py` — structured evaluation JSON with hallucination mitigation
- [ ] `src/privatefm/aggregation.py` — demographic comparison engine
- [ ] Validation: Compare FM flags vs FIDES on synthetic_dataset.csv

### Phase 4: BiasIRL Implementation (Coordinate with Swathi)
- [ ] `src/biasirl/matching.py` — matched cohort construction
- [ ] `src/biasirl/mcmc.py` — Bayesian IRL sampler (or use pymc3)
- [ ] `src/biasirl/divergence.py` — reward function comparison + posteriors
- [ ] `src/biasirl/outcome_audit.py` — baseline outcome tests for comparison
- [ ] Validation: Prove p=0.79 (outcomes pass) vs p<1e-12 (rewards diverge)

### Phase 5: Integration & Experiments
- [ ] Full pipeline: FIDES → PrivateFM → BiasIRL
- [ ] Experiments on synthetic data with ground truth
- [ ] Experiments on MIMIC-IV subset (if available)
- [ ] Generate all figures, tables, results

### Phase 6: Paper Writing
- [ ] Three paper drafts (Markdown or LaTeX)
- [ ] Results tables and figures
- [ ] Reproduce all results end-to-end

---

## Key Technical Decisions

### 1. FIDES — Causal Decomposition Method
- **Decision needed:** Use causal DAG + do-calculus (rigorous) or simpler path analysis (practical)?
- **Recommendation:** DAG + do-calculus for research credibility. Specify domain knowledge upfront.
- **Implementation:** Use `networkx` for DAG, `dowhy` or custom implementation for do-calculus

### 2. PrivateFM — Which Local Model?
- **Options:** Meditron 7B, BioMistral 7B, fine-tuned LLaMA-2 7B
- **Current:** Demo uses Claude API. For local: test Meditron first (best healthcare performance)
- **Framework:** Use `vLLM` or `llama.cpp` for inference optimization

### 3. BiasIRL — MCMC Sampler
- **Decision:** Implement custom Metropolis-Hastings or use PyMC3?
- **Recommendation:** Use PyMC3 for reliability, let Swathi lead this
- **Key requirement:** Posterior samples, not point estimates. Credible intervals are critical.

### 4. Data: Synthetic vs MIMIC-IV
- **Start:** Synthetic_dataset.csv (2000 records, ground truth known)
- **Validate concepts on synthetic first**, then scale to MIMIC-IV
- **MIMIC-IV access:** Requires credentialed access (PhysioNet). Check before depending on it.

---

## Team Split

- **You (Shrivarshini):** FIDES + PrivateFM implementation and validation
- **Swathi:** BiasIRL implementation (Bayesian IRL methodology expertise)
- **Coordination:** Share synthetic dataset validation results, agree on trajectory formatting

---

## Testing Strategy

### Synthetic Data Validation (High Priority)
```python
# Generate data with KNOWN biases
df = generate_synthetic_data(n=2000, seed=42)

# FIDES should detect:
# - Black patients underrepresented (6% vs 13%)
# - Care pathway bias (1.5 point pain gap)
# - Insufficient intersectional power (e.g., Black women > 65)

# PrivateFM should detect:
# - Guideline violations concentrated in Black patients
# - Gap vs FIDES: care quality issues not statistical issues

# BiasIRL should detect:
# - Reward function divergence: E[escalate|Black] != E[escalate|White]
# - Outcome audit p-value > 0.05, reward audit p-value < 1e-6
```

### Unit Tests
- Each module (fides, privatefm, biasirl) needs pytest coverage
- Test edge cases: empty subgroups, single-patient subgroups, perfect balance
- Test error handling: missing data, invalid demographics

---

## Common Pitfalls to Avoid

1. **FIDES insufficiency masking:** Don't just compute power and ignore it. Flag it as a FAILURE.
2. **PrivateFM hallucination:** FM can say "this care is appropriate" even for clearly bad trajectories. Add validation layer.
3. **BiasIRL convergence:** MCMC can fail silently. Always check Rhat, effective sample size.
4. **Demographic handling:** Race as a variable is sensitive. Document assumptions clearly. Use proper aggregation (don't average binary outcomes naively).
5. **Publication:** These papers will be reviewed by statisticians and clinicians. Methodology must be rigorous.

---

## Code Standards for This Project

All code should have:
- **Docstrings:** Explain WHAT and WHY, not just syntax
- **Type hints:** `def compute_gap(df: pd.DataFrame, race_col: str) -> float:`
- **Error handling:** Catch missing data, invalid demographics, empty subgroups
- **Logging:** Print progress, warn on suspicious results
- **Tests:** Unit tests for all functions. Validation on known datasets.

**Example pattern:**
```python
def compute_insufficiency_masking(
    df: pd.DataFrame,
    demographic_col: str,
    outcome_col: str,
    power_threshold: float = 0.8,
    alpha: float = 0.05
) -> Dict[str, bool]:
    """
    Identify demographic subgroups with insufficient statistical power to detect bias.
    
    Args:
        df: Clinical dataset with outcomes
        demographic_col: Column name for demographic grouping
        outcome_col: Column name for outcome (binary)
        power_threshold: Minimum power required (default 0.8)
        alpha: Significance level (default 0.05)
    
    Returns:
        dict mapping subgroup → True (insufficient) / False (sufficient)
    
    Raises:
        ValueError: if demographic_col or outcome_col not in df
    """
    ...
```

---

## Collaboration Notes (If working with Swathi)

**BiasIRL coordination:**
- Share synthetic dataset format and matched cohort construction code
- Agree on trajectory representation (state space, action space)
- Provide reference implementation of outcome audit baseline
- Validate MCMC convergence diagnostics together

**PrivateFM + BiasIRL coordination:**
- PrivateFM outputs per-patient care quality flags
- BiasIRL uses those flags as additional reward signal (optional)
- Both use same demographic groupings and stratifications

---

## References for Implementation

### FIDES
- Path-specific causal effects: Vanderweele & Vansteelandt (2013) "Mediation Analysis with Multiple Mediators Using the Product of Coefficients Approach"
- Causal DAGs: Pearl "Causality" (2009)
- Statistical power: Pocock "Clinical Trials" (1983)

### PrivateFM
- Local LLM inference: vLLM paper, llama.cpp
- Prompt engineering for structured output: OpenAI structured outputs guidance
- HIPAA compliance: OCR guidance on AI/LLM in healthcare

### BiasIRL
- Bayesian IRL: Ng & Russell (2000) "Algorithms for Inverse Reinforcement Learning"
- MCMC diagnostics: Gelman et al. "Bayesian Data Analysis" (2013)
- Clinical IRL baseline: Harvard "Pruning the Path to Optimal Care" (2411.05237)

---

## Timeline Estimate

- **Phase 1 (data/utils):** 1 week
- **Phase 2 (FIDES):** 2-3 weeks
- **Phase 3 (PrivateFM):** 2-3 weeks
- **Phase 4 (BiasIRL, with Swathi):** 2-3 weeks
- **Phase 5 (integration & experiments):** 2-3 weeks
- **Phase 6 (paper writing):** 2-3 weeks

**Total: 6-7 months** → Target submission Dec 2026 / Jan 2027

---

## Files to Preserve (Do NOT delete)

```
experiments/foundation_model_bias/        ← Demo code, working baseline
synthetic_dataset.csv                     ← Ground truth data
RESEARCH_ROADMAP.md                       ← This implementation plan
CLAUDE.md                                 ← (This file)
```

Everything else in the repo is context. Keep it.

---

## Next Steps (When Resuming)

1. Read `RESEARCH_ROADMAP.md` to remind yourself of the three papers
2. Create the directory structure under `src/`
3. Start with Phase 1: `src/utils/data_loading.py`
4. Run validation on `synthetic_dataset.csv` to build intuition
5. Move to FIDES Phase 2

---

**You are ready to implement. This is real research with genuine novelty.**
