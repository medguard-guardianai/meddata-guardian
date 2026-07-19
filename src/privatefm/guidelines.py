"""Clinical Guidelines Embedding in FM Prompts"""

from typing import Dict


class ClinicalGuidelines:
    """Library of clinical guidelines for prompt engineering."""

    CARDIAC_GUIDELINES = {
        "stemi": "STEMI requires immediate PCI/thrombolytics. All STEMI patients admitted to ICU/CCU.",
        "unstable_angina": "Risk stratify with HEART/TIMI score. High-risk requires angiography and ICU monitoring.",
        "heart_failure": "Assess EF. Decompensation signs (orthopnea, JVD, rales) warrant ICU admission.",
    }

    RESPIRATORY_GUIDELINES = {
        "pneumonia": "CAP severity by PSI/CURB-65. Severe (PaO2/FiO2 <250) requires ICU.",
        "copd": "COPD exacerbation: Severe symptoms/altered mental status/hemodynamic instability = ICU.",
        "ards": "ALL ARDS patients require ICU. Use lung-protective ventilation.",
    }

    SEPSIS_GUIDELINES = {
        "sepsis": "Sepsis = SIRS + infection. 1-hour bundle: cultures, lactate, antibiotics, fluids. ICU admission.",
    }

    AKI_GUIDELINES = {
        "aki": "AKI Stage 3 (Cr ≥3× baseline) requires ICU monitoring and consideration of dialysis.",
    }

    @classmethod
    def build_clinical_prompt(cls, scenario: str, demographics: str, condition: str) -> str:
        """Build clinical prompt with embedded guidelines."""
        guideline = cls.get_guideline(condition)
        return f"""Clinical guidelines: {guideline}

Scenario: {scenario}
Demographics: {demographics}

Should this patient be escalated to ICU? Answer Yes or No with brief justification."""

    @classmethod
    def get_guideline(cls, condition: str) -> str:
        """Get guideline for condition."""
        all_guidelines = {
            **cls.CARDIAC_GUIDELINES,
            **cls.RESPIRATORY_GUIDELINES,
            **cls.SEPSIS_GUIDELINES,
            **cls.AKI_GUIDELINES,
        }
        return all_guidelines.get(condition, "")
