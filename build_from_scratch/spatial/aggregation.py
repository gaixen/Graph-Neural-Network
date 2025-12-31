"""Sum / mean / max aggregation, injectivity analysis, and permutation-invariance proofs.

This module analyzes aggregation functions, which are the critical component
determining GNN expressivity.

Key theorem (Xu et al., 2019):
    GNN expressivity is bounded by the injectivity of the aggregation function.
    
    - Injective aggregation → GNN can distinguish any structurally different neighborhoods
    - Non-injective → information loss → reduced expressivity

Mathematical framework:
    Aggregation: AGG: 𝒫(ℝ^d) → ℝ^d
    
    Must satisfy:
    1. Permutation invariance: AGG({x₁, ..., xₙ}) = AGG(π({x₁, ..., xₙ}))
    2. (Ideally) Injectivity: AGG(S₁) = AGG(S₂) ⇒ S₁ = S₂

Expressivity hierarchy:
    SUM > MEAN ~ MAX
    
    SUM is injective over countable multisets (infinite feature dimension)
    MEAN and MAX are provably not injective

Deep Sets theorem (Zaheer et al., 2017):
    Universal permutation-invariant function:
    f({x₁, ..., xₙ}) = ρ(Σᵢ φ(xᵢ))
    
    where φ, ρ are arbitrary functions.
"""

from typing import Iterable, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class InjectivityAnalysis:
    """Result of injectivity analysis for aggregation function.
    
    Attributes:
        is_injective: Whether aggregation is injective
        counterexample: If not injective, example of collision
        explanation: Human-readable explanation
    """
    is_injective: bool
    counterexample: Optional[Tuple[List, List]] = None
    explanation: str = ""


def aggregate_sum(xs: Iterable[float]) -> float:
    """Sum aggregation: Σᵢ xᵢ.
    
    Mathematical properties:
        - Permutation invariant: ✓
        - Injective: ✓ (for integer/rational multisets with unbounded values)
        - Size-dependent: ✓ (sum grows with neighborhood size)
    
    Proof of injectivity:
        For multisets over ℕ (natural numbers):
        Each multiset has unique prime factorization representation.
        
        For general ℝ^d with d → ∞:
        Can encode multiset as sum with unique "one-hot" dimensions per element.
    
    GNN implication:
        GIN (Graph Isomorphism Network) uses sum aggregation to achieve
        maximal expressivity among MPNNs.
    
    Args:
        xs: Multiset of values
    
    Returns:
        Sum of all values
    """
    return sum(xs)


def aggregate_mean(xs: Iterable[float]) -> float:
    """Mean aggregation: (1/|S|) Σᵢ xᵢ.
    
    Mathematical properties:
        - Permutation invariant: ✓
        - Injective: ✗ (loses cardinality)
        - Size-normalized: ✓ (independent of neighborhood size)
    
    Counterexample to injectivity:
        MEAN({1, 1, 1}) = 1 = MEAN({1})
        But {1, 1, 1} ≠ {1}
    
    Information lost:
        - Cardinality |S| (number of neighbors)
        - Cannot distinguish n copies of x from 1 copy of x
    
    GNN implication:
        GCN uses (implicit) mean aggregation.
        This limits expressivity below 1-WL test.
    
    Args:
        xs: Multiset of values
    
    Returns:
        Mean of all values (or 0 if empty)
    """
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_max(xs: Iterable[float]) -> float:
    """Max aggregation: maxᵢ xᵢ.
    
    Mathematical properties:
        - Permutation invariant: ✓
        - Injective: ✗ (loses all but maximum)
        - Sparse: only one element contributes
    
    Counterexample to injectivity:
        MAX({1, 5}) = 5 = MAX({5, 5})
        But {1, 5} ≠ {5, 5}
    
    Information lost:
        - All elements except maximum
        - Multiplicities (how many times max appears)
        - All smaller values
    
    Gradient issue:
        ∇(max) is undefined at ties!
        When multiple elements equal maximum, gradient is ambiguous.
    
    GNN implication:
        GraphSAGE uses max aggregation (also mean).
        Less expressive than GIN but computationally simpler.
    
    Args:
        xs: Multiset of values
    
    Returns:
        Maximum value (or -inf if empty)
    """
    xs = list(xs)
    return max(xs) if xs else float('-inf')


def analyze_injectivity(
    aggregate_fn: Callable[[List[float]], float],
    test_domain: List[List[float]]
) -> InjectivityAnalysis:
    """Test if aggregation function is injective on given domain.
    
    Method:
        Enumerate all pairs in domain, check for collisions:
        AGG(S₁) = AGG(S₂) but S₁ ≠ S₂
    
    Args:
        aggregate_fn: Aggregation function to test
        test_domain: List of multisets to test
    
    Returns:
        InjectivityAnalysis with result
    """
    # Compute aggregated values
    agg_values = [(aggregate_fn(S), S) for S in test_domain]
    
    # Check for collisions
    for i in range(len(agg_values)):
        for j in range(i + 1, len(agg_values)):
            val_i, S_i = agg_values[i]
            val_j, S_j = agg_values[j]
            
            if abs(val_i - val_j) < 1e-10:  # Equal within tolerance
                # Found collision!
                return InjectivityAnalysis(
                    is_injective=False,
                    counterexample=(S_i, S_j),
                    explanation=f"AGG({S_i}) = {val_i:.4f} = AGG({S_j}), but sets differ"
                )
    
    # No collisions found (doesn't prove injectivity, but suggests it)
    return InjectivityAnalysis(
        is_injective=True,
        explanation="No collisions found in test domain (not a proof of injectivity)"
    )


def prove_sum_injective_for_naturals() -> str:
    """Explain why sum is injective for multisets of natural numbers.
    
    Returns:
        Proof sketch as string
    """
    return """
    THEOREM: Sum aggregation is injective over multisets of natural numbers.
    
    PROOF SKETCH:
        Consider multisets represented as functions m: ℕ → ℕ
        where m(x) = multiplicity of x in the multiset.
        
        Key observation:
        Two multisets S₁, S₂ have the same sum iff:
        Σₓ x · m₁(x) = Σₓ x · m₂(x)
        
        This is equivalent to:
        Σₓ x · (m₁(x) - m₂(x)) = 0
        
        For finite support (only finitely many distinct elements),
        this defines a unique linear equation.
        
        By fundamental theorem of arithmetic (unique prime factorization),
        we can encode multisets such that sum is injective.
        
    PRACTICAL LIMITATION:
        This requires unbounded integer precision!
        With floating point, collisions occur due to rounding.
    
    GNN IMPLICATION:
        In practice, use high-dimensional embeddings + sum aggregation
        to approximate injectivity.
    """


def deep_sets_universal_aggregation(
    elements: List[np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
    rho: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Universal permutation-invariant function (Deep Sets theorem).
    
    Theorem (Zaheer et al., 2017):
        Any permutation-invariant function f: 𝒫(ℝ^d) → ℝ^k can be written as:
        
        f({x₁, ..., xₙ}) = ρ(Σᵢ φ(xᵢ))
        
        for some functions φ: ℝ^d → ℝ^m and ρ: ℝ^m → ℝ^k.
    
    Proof intuition:
        - φ embeds each element independently
        - Σ aggregates (permutation-invariant!)
        - ρ post-processes the aggregated representation
    
    GNN connection:
        This justifies the architecture:
        1. Per-element transformation: hⱼ' = MLP(hⱼ)  [φ]
        2. Sum aggregation: m = Σⱼ hⱼ'  [Σ]
        3. Readout: output = MLP(m)  [ρ]
    
    Args:
        elements: List of vectors {x₁, ..., xₙ}
        phi: Element-wise transformation
        rho: Aggregated transformation
    
    Returns:
        f({elements}) = ρ(Σᵢ φ(xᵢ))
    """
    # Apply φ to each element
    transformed = [phi(x) for x in elements]
    
    # Sum aggregation
    aggregated = np.sum(transformed, axis=0) if transformed else np.zeros_like(phi(elements[0]))
    
    # Apply ρ
    output = rho(aggregated)
    
    return output


def compare_aggregations(
    test_multisets: List[List[float]]
) -> dict:
    """Compare expressivity of different aggregations.
    
    Returns:
        Dictionary mapping aggregation name to injectivity analysis
    """
    aggregations = {
        'sum': aggregate_sum,
        'mean': aggregate_mean,
        'max': aggregate_max,
    }
    
    results = {}
    for name, agg_fn in aggregations.items():
        results[name] = analyze_injectivity(agg_fn, test_multisets)
    
    return results


def multi_aggregation(
    elements: List[np.ndarray]
) -> np.ndarray:
    """Combine multiple aggregations to increase expressivity.
    
    Idea:
        h_agg = [SUM(elements), MEAN(elements), MAX(elements), STD(elements)]
    
    Rationale:
        While no single aggregation is fully injective,
        combining multiple aggregations preserves more information.
    
    This is used in:
        - GraphSAGE (concatenates mean/max)
        - Principal Neighbourhood Aggregation (PNA)
    
    Args:
        elements: List of vectors
    
    Returns:
        Concatenated aggregations
    """
    if not elements:
        # Empty neighborhood: return zeros
        dim = len(elements[0]) if elements else 1
        return np.zeros(4 * dim)
    
    elements_array = np.array(elements)
    
    sum_agg = np.sum(elements_array, axis=0)
    mean_agg = np.mean(elements_array, axis=0)
    max_agg = np.max(elements_array, axis=0)
    std_agg = np.std(elements_array, axis=0)
    
    # Concatenate all aggregations
    return np.concatenate([sum_agg, mean_agg, max_agg, std_agg])


def prove_mean_not_injective() -> InjectivityAnalysis:
    """Provide explicit counterexample showing mean is not injective.
    
    Returns:
        InjectivityAnalysis with counterexample
    """
    S1 = [1.0, 1.0, 1.0]  # Three 1's
    S2 = [1.0]             # One 1
    
    mean1 = aggregate_mean(S1)
    mean2 = aggregate_mean(S2)
    
    return InjectivityAnalysis(
        is_injective=False,
        counterexample=(S1, S2),
        explanation=(
            f"MEAN({S1}) = {mean1} = MEAN({S2}), but |S1| = {len(S1)} ≠ {len(S2)} = |S2|. "
            "Mean aggregation loses cardinality information."
        )
    )


def prove_max_not_injective() -> InjectivityAnalysis:
    """Provide explicit counterexample showing max is not injective.
    
    Returns:
        InjectivityAnalysis with counterexample
    """
    S1 = [1.0, 5.0]  # Min=1, Max=5
    S2 = [5.0]       # Just the max
    
    max1 = aggregate_max(S1)
    max2 = aggregate_max(S2)
    
    return InjectivityAnalysis(
        is_injective=False,
        counterexample=(S1, S2),
        explanation=(
            f"MAX({S1}) = {max1} = MAX({S2}), but S1 contains additional element 1.0. "
            "Max aggregation discards all non-maximal elements."
        )
    )
