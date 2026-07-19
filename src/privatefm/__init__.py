"""PrivateFM: HIPAA-Compliant Local Foundation Model Bias Auditing

This module provides tools for testing foundation model bias on clinical scenarios
without sending patient data to external APIs (HIPAA-compliant local inference).
"""

from .local_inference import LocalFMInference
from .guidelines import ClinicalGuidelines
from .evaluation import FMBiasEvaluator
from .aggregation import BiasAggregator

__all__ = [
    "LocalFMInference",
    "ClinicalGuidelines",
    "FMBiasEvaluator",
    "BiasAggregator",
]
