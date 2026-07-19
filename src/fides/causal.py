"""
Path-Specific Causal Effect (PSCE) Decomposition

Decomposes outcome gaps by causal pathway to separate legitimate (clinical)
from illegitimate (race-mediated) pathways.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class CausalDAG:
    """Representation of a causal directed acyclic graph."""

    edges: List[Tuple[str, str]]  # List of (from, to) edges
    nodes: List[str]               # All node names

    def __post_init__(self):
        """Validate DAG structure."""
        if not self.edges:
            raise ValueError("DAG must have at least one edge")

        # Check that all edges reference valid nodes
        edge_nodes = set()
        for src, dst in self.edges:
            edge_nodes.add(src)
            edge_nodes.add(dst)

        missing = edge_nodes - set(self.nodes)
        if missing:
            raise ValueError(f"Edge references undefined nodes: {missing}")

    def paths_from_to(self, source: str, target: str) -> List[List[str]]:
        """Find all paths from source to target in the DAG."""
        if source not in self.nodes:
            raise ValueError(f"Source {source} not in DAG")
        if target not in self.nodes:
            raise ValueError(f"Target {target} not in DAG")

        # Build adjacency list
        adj = {node: [] for node in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)

        # DFS to find all paths
        all_paths = []

        def dfs(current, path):
            if current == target:
                all_paths.append(path)
                return

            for neighbor in adj[current]:
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, path + [neighbor])

        dfs(source, [source])
        return all_paths


def compute_psce(
    df: pd.DataFrame,
    dag: CausalDAG,
    protected_attr: str,
    outcome_col: str,
    mediators_by_path: Optional[Dict[int, List[str]]] = None
) -> Dict:
    """
    Decompose the total effect of a protected attribute on an outcome into
    a portion explained by legitimate clinical mediators and a leftover
    ("illegitimate") portion, using the classic difference-in-coefficients
    mediation method (Vanderweele & Vansteelandt 2013):

        total_effect    = coefficient of protected_attr in: outcome ~ protected_attr
        direct_effect   = coefficient of protected_attr in: outcome ~ protected_attr + legitimate_mediators
        illegitimate_strength = direct_effect / total_effect

    `protected_attr` must be a binary (0/1) column — encode multi-category
    attributes (e.g. race) into a binary contrast (e.g. majority vs. rest)
    before calling this, since a multi-category regressor has no single
    coefficient with a well-defined direction.

    Legitimate mediators are the union of `mediators_by_path` entries in
    the DAG whose path is marked legitimate; all other paths' mediators are
    ignored for the adjustment (their effect stays folded into the leftover).

    Args:
        df: Input dataframe with all variables
        dag: Causal DAG specifying relationships (used only to enumerate
            paths and label them legitimate/illegitimate via mediators_by_path)
        protected_attr: Binary (0/1) protected attribute column
        outcome_col: Outcome column name
        mediators_by_path: Dict mapping path index -> list of mediator names
            that make that path "legitimate" if present on it. A path with
            an empty list here is treated as illegitimate/unadjusted.

    Returns:
        Dict with total_effect, direct_effect, illegitimate_strength (a
        ratio, or None when undefined — see below), and the path
        enumeration for transparency.

    Note on `illegitimate_strength` stability: this is a ratio
    (direct_effect / total_effect), which is only meaningful when
    total_effect is itself distinguishable from zero. If the raw
    protected_attr -> outcome effect isn't statistically significant
    (p >= 0.05), there is no detected disparity to decompose in the first
    place, and dividing by a near-zero denominator produces an arbitrarily
    large, meaningless ratio (observed: 2367% on one real cohort where the
    raw gap was ~0). In that case `illegitimate_strength` is set to None
    and `total_effect_significant` is False — callers should treat "no
    significant raw effect" as passing this condition, not as an undefined
    failure.
    """
    if df[protected_attr].nunique() != 2:
        raise ValueError(
            f"compute_psce requires a binary protected_attr; "
            f"'{protected_attr}' has {df[protected_attr].nunique()} categories"
        )

    paths = dag.paths_from_to(protected_attr, outcome_col)
    if not paths:
        raise ValueError(f"No paths found from {protected_attr} to {outcome_col} in DAG")

    import statsmodels.api as sm

    y = df[outcome_col].astype(float).values

    # Total effect: regression of outcome on protected_attr alone, with
    # its p-value so we know whether the raw gap is even real.
    X_total = sm.add_constant(df[[protected_attr]].astype(float))
    model_total = sm.OLS(y, X_total).fit()
    total_effect = float(model_total.params[protected_attr])
    total_effect_pvalue = float(model_total.pvalues[protected_attr])
    total_effect_significant = total_effect_pvalue < 0.05

    # Legitimate mediators = union of mediators on paths marked legitimate
    legitimate_mediators = []
    if mediators_by_path:
        for path_idx, path in enumerate(paths):
            path_mediators = path[1:-1]
            marked = mediators_by_path.get(path_idx, [])
            if any(m in path_mediators for m in marked):
                legitimate_mediators.extend(m for m in path_mediators if m in marked)
    legitimate_mediators = sorted(set(legitimate_mediators))

    # Direct effect: regression of outcome on protected_attr + legitimate mediators
    if legitimate_mediators:
        design_cols = [protected_attr] + legitimate_mediators
        X_direct = sm.add_constant(df[design_cols].astype(float))
        model_direct = sm.OLS(y, X_direct).fit()
        direct_effect = float(model_direct.params[protected_attr])
    else:
        direct_effect = total_effect  # nothing legitimate to adjust for

    if total_effect_significant:
        illegitimate_strength = direct_effect / total_effect
        note = None
    else:
        illegitimate_strength = None
        note = (
            f"Raw effect not statistically distinguishable from zero "
            f"(p={total_effect_pvalue:.3f}); illegitimate_strength ratio is "
            f"undefined and not reported. No detected disparity to decompose."
        )

    return {
        'total_effect': total_effect,
        'total_effect_pvalue': total_effect_pvalue,
        'total_effect_significant': total_effect_significant,
        'direct_effect': direct_effect,
        'legitimate_mediators': legitimate_mediators,
        'illegitimate_strength': illegitimate_strength,
        'note': note,
        'paths': [' → '.join(p) for p in paths],
    }


def identify_illegitimate_paths(
    dag: CausalDAG,
    protected_attr: str,
    clinical_mediators: List[str]
) -> List[List[str]]:
    """
    Identify paths from protected attribute to outcome that do NOT flow through
    clinical mediators (i.e., illegitimate paths).

    Args:
        dag: Causal DAG
        protected_attr: Protected attribute (e.g., 'race')
        clinical_mediators: List of mediators that represent legitimate clinical pathways

    Returns:
        List of illegitimate paths

    Example:
        >>> dag = CausalDAG(
        ...     edges=[('race', 'pain_recorded'), ('pain_recorded', 'action'),
        ...            ('severity', 'action'), ('severity', 'outcome'),
        ...            ('action', 'outcome')],
        ...     nodes=['race', 'pain_recorded', 'severity', 'action', 'outcome']
        ... )
        >>> illegitimate = identify_illegitimate_paths(
        ...     dag, 'race', clinical_mediators=['severity']
        ... )
        >>> # Returns paths through race→pain_recorded (bias) not through severity (clinical)
    """

    # Find all paths from protected attr to outcome
    all_paths = dag.paths_from_to(protected_attr, 'outcome')  # Assuming outcome is always target

    illegitimate_paths = []
    for path in all_paths:
        # Check if any clinical mediator is on this path
        mediators_on_path = set(path[1:-1])
        has_clinical = any(m in clinical_mediators for m in mediators_on_path)

        if not has_clinical:
            illegitimate_paths.append(path)

    return illegitimate_paths
