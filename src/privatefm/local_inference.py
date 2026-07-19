"""Local Foundation Model Inference (HIPAA-Compliant)

Wrapper for running inference on local FM models without sending patient data to external APIs.
"""

import requests
import json
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for local FM model."""
    name: str
    base_url: str
    endpoint: str
    temperature: float = 0.3
    timeout: int = 120


class LocalFMInference:
    """Inference wrapper for local foundation models."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """Initialize local FM inference."""
        self.config = ModelConfig(
            name=model,
            base_url=base_url,
            endpoint=f"{base_url}/api/generate"
        )

    def generate(self, prompt: str) -> str:
        """Generate text using local FM."""
        try:
            response = requests.post(
                self.config.endpoint,
                json={
                    "model": self.config.name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.config.temperature
                },
                timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"FM error {response.status_code}")

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to FM at {self.config.base_url}") from e

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt)
                results.append(result)
            except Exception as e:
                results.append("")
        return results
