"""
Medical Advisor Agent - Provides clinical context using RAG + Instructor
"""

import ollama
from pydantic import BaseModel, Field
from typing import Literal, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.medical_rag import MedicalKnowledgeBase


class MedicalAdviceSchema(BaseModel):
    """
    Instructor schema - prevents hallucination
    Forces AI to cite sources and stay grounded
    """
    recommendation: Literal["keep", "remove", "review", "modify"]
    
    clinical_reasoning: str = Field(
        max_length=250,
        description="Explain based ONLY on provided medical documents"
    )
    
    source_documents: str = Field(
        description="Which documents were referenced"
    )
    
    medical_threshold: Optional[str] = Field(
        default=None,
        description="Clinical threshold if applicable (e.g., '>240 mg/dL is high')"
    )


class MedicalAdvisor:
    """
    AI agent that provides medical context using RAG
    """
    
    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.kb.load_documents()
        print("✅ Medical Advisor Agent ready")
    
    def analyze_outliers(self, column: str, outlier_info: dict) -> dict:
        """
        Determine if outliers should be kept based on medical knowledge
        """
        # Query knowledge base
        query = f"Should {column} outliers above {outlier_info['bounds']['upper']} be kept in disease prediction?"
        
        medical_context = self.kb.get_context_for_query(query, n_results=2)
        
        # Prepare prompt for Llama
        prompt = f"""
You are a medical data expert. Use ONLY the medical knowledge provided below.

{medical_context}

QUESTION:
Dataset has {outlier_info['count']} {column} values above {outlier_info['bounds']['upper']}.
Sample values: {outlier_info['values_sample']}

Should these be kept, removed, or reviewed?

Provide medical reasoning based ONLY on the documents above.
Cite which source you used.
"""
        
        # Call Llama (we'll add Instructor in next step)
        response = ollama.generate(
            model='llama3.2:3b',
            prompt=prompt,
            options={'temperature': 0.3}
        )
        
        advice_text = response['response']
        
        # Parse response (simplified for now)
        return {
            'recommendation': 'keep' if 'keep' in advice_text.lower() else 'review',
            'clinical_reasoning': advice_text[:250],
            'source_documents': medical_context[:100]
        }
    
    def explain_bias_impact(self, bias_type: str, bias_info: dict) -> str:
        """
        Explain why demographic bias matters using medical knowledge
        """
        # Query knowledge base
        if 'gender' in bias_type.lower() or 'sex' in bias_type.lower():
            query = "gender differences in heart disease symptoms and diagnosis"
        elif 'indigenous' in bias_type.lower():
            query = "Indigenous health disparities cardiovascular disease"
        elif 'race' in bias_type.lower():
            query = "racial disparities in healthcare outcomes"
        else:
            query = f"{bias_type} bias in medical AI"
        
        medical_context = self.kb.get_context_for_query(query, n_results=2)
        
        prompt = f"""
Based on this medical knowledge:

{medical_context}

Explain in 2-3 sentences why having {bias_info} matters for patient care and model fairness.
Focus on clinical consequences.
"""
        
        response = ollama.generate(
            model='llama3.2:3b',
            prompt=prompt,
            options={'temperature': 0.3}
        )
        
        return response['response']


# Test if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Medical Advisor Agent")
    print("=" * 60)
    
    # Initialize agent
    advisor = MedicalAdvisor()
    
    # Test 1: Outlier advice
    print("\n" + "=" * 60)
    print("TEST 1: Should we keep high cholesterol outliers?")
    print("=" * 60)
    
    outlier_info = {
        'count': 14,
        'values_sample': [487, 512, 564, 603],
        'bounds': {'upper': 350}
    }
    
    advice = advisor.analyze_outliers('cholesterol', outlier_info)
    print(f"\nRecommendation: {advice['recommendation']}")
    print(f"\nReasoning:\n{advice['clinical_reasoning']}")
    
    # Test 2: Gender bias impact
    print("\n" + "=" * 60)
    print("TEST 2: Why does 73% male bias matter?")
    print("=" * 60)
    
    bias_info = "73% male, 27% female in heart disease dataset"
    impact = advisor.explain_bias_impact('gender', bias_info)
    
    print(f"\nImpact Explanation:\n{impact}")
    
    print("\n✅ Medical Advisor test complete!")
