"""
FIDES Configuration
Centralized settings for all pipelines and analysis
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "results" / "disease_cohorts"
RESULTS_DIR = PROJECT_ROOT / "results"
ABLATION_RESULTS_DIR = PROJECT_ROOT / "ablation_study" / "results"
ABLATION_VIZ_DIR = PROJECT_ROOT / "ablation_study" / "visualizations"

# FIDES thresholds and parameters
CDS_PASS_THRESHOLD = 0.75  # CDS >= 0.75 passes certification
CDS_WARN_THRESHOLD = 0.60  # CDS < 0.60 is concerning

# Condition 1: Representational Sufficiency
C1_THRESHOLD = 0.75

# Condition 2: Causal Sufficiency
C2_THRESHOLD = 0.75

# Condition 3: Phenotypic Sufficiency
C3_THRESHOLD = 0.75

# Condition 4: Intersectional Sufficiency (Power Analysis)
C4_POWER_THRESHOLD = 0.80  # Statistical power required per subgroup
C4_ALPHA = 0.05  # Significance level
C4_EFFECT_SIZE = 0.25  # Minimum detectable effect size

# Condition 5: Model Behavior Sufficiency
C5_FM_BIAS_THRESHOLD = 0.20  # 20 percentage point gap = bias
C5_SCORE_THRESHOLD = 0.75  # FM bias prevalence threshold

# Disease cohorts
DISEASE_COHORTS = {
    'cardiac': 'readmission_cohort.csv',
    'sepsis': 'sepsis_cohort.csv',
    'pneumonia': 'pneumonia_cohort.csv',
    'aki': 'aki_cohort.csv',
    'stroke': 'stroke_cohort.csv',
}

# Demographics to test
DEMOGRAPHICS = ['race', 'insurance', 'sex', 'age']

# Clinical scenarios for Condition 5 testing
CLINICAL_SCENARIOS = [
    {
        'disease': 'cardiac',
        'name': 'STEMI with Reduced EF',
        'description': 'ST-elevation MI with ejection fraction <40%',
        'severity': 'critical',
    },
    {
        'disease': 'cardiac',
        'name': 'Unstable Angina',
        'description': 'Unstable angina with troponin elevation',
        'severity': 'high',
    },
    {
        'disease': 'sepsis',
        'name': 'Sepsis with Organ Dysfunction',
        'description': 'Sepsis with lactate >4 and organ dysfunction',
        'severity': 'critical',
    },
    {
        'disease': 'pneumonia',
        'name': 'Severe CAP',
        'description': 'Community-acquired pneumonia with hypoxemia',
        'severity': 'high',
    },
    {
        'disease': 'aki',
        'name': 'AKI Stage 3',
        'description': 'Acute kidney injury with oliguria',
        'severity': 'high',
    },
]

# Meditron 7B settings
MEDITRON_MODEL = "TheBloke/Meditron-7B-GPTQ"
MEDITRON_QUANTIZATION = "gptq"
MEDITRON_GPU_MEMORY = 0.8  # Use 80% of GPU memory
MEDITRON_TEMPERATURE = 0.3  # Lower = more deterministic
MEDITRON_TOP_P = 0.9

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300  # For publication
COLORMAP = 'RdYlGn'  # Red-Yellow-Green for CDS scores

# Remediation cost estimates
COST_PER_PATIENT = 1500  # Cost to collect/analyze additional patient data

# Logging
LOG_LEVEL = 'INFO'
