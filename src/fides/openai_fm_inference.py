"""
OpenAI Foundation Model Inference for FIDES Condition 5

Real calls to the OpenAI API (e.g. gpt-4o-mini) for demographic bias testing.
This sends the synthetic clinical vignettes defined in clinical_scenarios.py
(no real patient data) to OpenAI's API — this is NOT the HIPAA-compliant
local path (see PrivateFM / LocalFMInference for that); use this only when
comparing a cloud model against a local one is explicitly wanted.

The API key is read from the OPENAI_API_KEY environment variable only.
It is never hardcoded, logged, or written to any file by this module.
"""

import os
import time
from typing import List

from openai import OpenAI, APIError, APIConnectionError, RateLimitError


class OpenAIFMInference:
    """Duck-type compatible with LocalFMInference: exposes .generate(prompt)."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it in your shell — never hardcode it in source."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Generate a real completion from the OpenAI API, with bounded
        retry on transient errors (rate limits, connection drops). Raises
        after retries are exhausted — never falls back to fake output."""
        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                # Non-transient (bad request, auth, etc.) — fail immediately
                raise RuntimeError(f"OpenAI generation failed: {e}") from e
        raise RuntimeError(f"OpenAI generation failed after {max_attempts} attempts: {last_error}") from last_error

    def batch_generate(self, prompts: List[str], max_tokens: int = 150, temperature: float = 0.3) -> List[str]:
        return [self.generate(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]
