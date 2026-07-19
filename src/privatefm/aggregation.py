"""Bias Aggregation and Demographic Comparison"""

import json
from typing import Dict, List


class BiasAggregator:
    """Aggregate bias results across demographics and scenarios."""

    def __init__(self):
        self.results = {}

    def add_scenario_results(self, scenario_name: str, demographic_escalations: Dict[str, bool]):
        """Add results for a scenario."""
        self.results[scenario_name] = demographic_escalations

    def compute_demographic_parity(self) -> Dict[str, float]:
        """Compute demographic parity (escalation rates by demographic)."""
        if not self.results:
            return {}

        demographic_rates = {}
        for scenario_name, escalations in self.results.items():
            for demographic, decision in escalations.items():
                if demographic not in demographic_rates:
                    demographic_rates[demographic] = []
                demographic_rates[demographic].append(decision)

        return {
            demographic: sum(decisions) / len(decisions)
            for demographic, decisions in demographic_rates.items()
        }

    def compute_max_demographic_gap(self) -> float:
        """Compute maximum gap in escalation rates across demographics."""
        parity = self.compute_demographic_parity()
        if not parity:
            return 0.0
        return max(parity.values()) - min(parity.values())

    def generate_summary(self) -> str:
        """Generate summary of bias findings."""
        parity = self.compute_demographic_parity()
        max_gap = self.compute_max_demographic_gap()

        summary = f"""
BIAS AGGREGATION SUMMARY
========================

Escalation Rates by Demographic:
"""
        for demographic, rate in parity.items():
            summary += f"\n  {demographic}: {rate*100:.0f}%"

        summary += f"\n\nMaximum Demographic Gap: {max_gap*100:.0f}pp"
        summary += f"\nVerdict: {'BIASED' if max_gap > 0.20 else 'UNBIASED'}"

        return summary

    def export_json(self) -> str:
        """Export results as JSON."""
        parity = self.compute_demographic_parity()
        export_data = {
            "max_gap": self.compute_max_demographic_gap(),
            "escalation_by_demographic": parity,
            "total_scenarios": len(self.results),
        }
        return json.dumps(export_data, indent=2)
