"""Functional analysis for aggregation operators and permutation-invariant maps.

This module provides the theoretical foundation for understanding why aggregation
in message-passing neural networks is inherently lossy. Key concepts:

1. Permutation-invariant functions: φ: Multiset → ℝ where order doesn't matter
2. Injectivity vs universal approximation: the fundamental tradeoff
3. Characterization of aggregation operators (sum, mean, max, etc.)

Mathematical core:
- Aggregation collapses potentially infinite multisets into fixed-dimensional outputs
- By pigeonhole principle, information loss is unavoidable
- Only sum-based aggregation can be injective on countable multisets (Zaheer et al.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Any, TypeVar, Generic
from itertools import permutations
from collections import Counter
import numpy as np

T = TypeVar("T")


class PermutationInvariantFunction(ABC, Generic[T]):
    """Abstract base for permutation-invariant functions φ: Multiset[T] → ℝ^d.

    A function φ is permutation-invariant if for any permutation π:
        φ({x_1, ..., x_n}) = φ({x_π(1), ..., x_π(n)})

    This is the key constraint for aggregation in GNNs.
    """

    @abstractmethod
    def __call__(self, multiset: Iterable[T]) -> Any:
        """Apply the permutation-invariant operator."""
        pass

    def verify_invariance(self, multiset: Iterable[T], exhaustive: bool = True) -> bool:
        """Verify permutation invariance on a given multiset.

        Parameters:
            multiset: Input multiset to test
            exhaustive: If True, check all permutations (only feasible for small inputs)

        Returns:
            True if function is invariant on this input
        """
        lst = list(multiset)
        if len(lst) > 8 and exhaustive:
            raise ValueError("Exhaustive check infeasible for large inputs")

        reference_output = self(lst)

        if exhaustive:
            for perm in permutations(lst):
                if not np.array_equal(self(list(perm)), reference_output):
                    return False
        else:
            # Sample random permutations
            for _ in range(min(100, len(lst) * 10)):
                perm_lst = lst.copy()
                np.random.shuffle(perm_lst)
                if not np.array_equal(self(perm_lst), reference_output):
                    return False

        return True


class SumAggregation(PermutationInvariantFunction[float]):
    """Sum aggregation: φ(X) = Σ_{x ∈ X} x

    Properties:
        - Permutation-invariant: ✓
        - Injective on finite multisets: ✗ (e.g., {1,2} and {3} both sum to 3)
        - Injective on countable multisets with bounded elements: ✓ (with appropriate encoding)
        - Universal approximation (with MLP): ✓ (Deep Sets theorem)

    Mathematical form:
        AGG({h_1, ..., h_k}) = Σ_i h_i

    This is the aggregation used in GIN (Graph Isomorphism Network).
    """

    def __call__(self, multiset: Iterable[float]) -> float:
        return sum(multiset)


class MeanAggregation(PermutationInvariantFunction[float]):
    """Mean aggregation: φ(X) = (1/|X|) Σ_{x ∈ X} x

    Properties:
        - Permutation-invariant: ✓
        - Injective: ✗ (loses cardinality information)
        - Universal approximation: ✗ (cannot distinguish {1,2,3} from {2,2,2})

    Mathematical form:
        AGG({h_1, ..., h_k}) = (1/k) Σ_i h_i

    Used in GCN and GraphSAGE-mean. Normalizes by degree, but loses information
    about neighborhood size.
    """

    def __call__(self, multiset: Iterable[float]) -> float:
        lst = list(multiset)
        return sum(lst) / len(lst) if lst else 0.0


class MaxAggregation(PermutationInvariantFunction[float]):
    """Max aggregation: φ(X) = max_{x ∈ X} x

    Properties:
        - Permutation-invariant: ✓
        - Injective: ✗ (loses all but maximum element)
        - Universal approximation: ✗ (very lossy)

    Mathematical form:
        AGG({h_1, ..., h_k}) = max_i h_i

    Used in GraphSAGE-pool. Extremely lossy but captures range.
    """

    def __call__(self, multiset: Iterable[float]) -> float:
        return max(multiset) if multiset else float("-inf")


class MultisetHistogram(PermutationInvariantFunction):
    """Histogram-based multiset representation.

    Maps multiset to vector of counts in discretized bins.
    This is permutation-invariant and more expressive than simple aggregation,
    but requires discretization (introducing approximation error).

    Mathematical form:
        φ(X) = [count(x ∈ bin_1), count(x ∈ bin_2), ...]
    """

    def __init__(self, bins: np.ndarray):
        """Initialize with bin edges.

        Parameters:
            bins: Bin edges for histogram (length = num_bins + 1)
        """
        self.bins = bins

    def __call__(self, multiset: Iterable[float]) -> np.ndarray:
        counts, _ = np.histogram(list(multiset), bins=self.bins)
        return counts


@dataclass
class InjectivityAnalysis:
    """Results of injectivity analysis for an aggregation function."""

    is_injective: bool
    counterexample: tuple[list, list] | None = None
    explanation: str = ""


def check_injectivity_on_domain(
    agg_fn: Callable, domain_samples: list[list], tol: float = 1e-10
) -> InjectivityAnalysis:
    """Check if aggregation function is injective on given domain.

    Tests whether distinct multisets map to distinct outputs:
        X ≠ Y  ⟹  agg_fn(X) ≠ agg_fn(Y)

    Parameters:
        agg_fn: Aggregation function to test
        domain_samples: List of multisets to test
        tol: Numerical tolerance for equality

    Returns:
        InjectivityAnalysis with results and potential counterexample

    Note:
        This can only prove non-injectivity (by finding collision),
        not injectivity (would require checking infinite domain).
    """
    outputs = {}

    for multiset in domain_samples:
        output = agg_fn(multiset)

        # Check if we've seen this output before
        for prev_multiset, prev_output in outputs.items():
            if np.allclose(output, prev_output, atol=tol):
                # Found collision
                if Counter(multiset) != Counter(prev_multiset):
                    return InjectivityAnalysis(
                        is_injective=False,
                        counterexample=(list(prev_multiset), multiset),
                        explanation=f"Distinct multisets map to same output: "
                        f"{prev_multiset} and {multiset} both → {output}",
                    )

        outputs[tuple(sorted(multiset))] = output

    return InjectivityAnalysis(
        is_injective=True,
        explanation="No collisions found in tested domain (not a proof of injectivity)",
    )


def deep_sets_decomposition(phi: Callable, rho: Callable) -> Callable:
    """Construct Deep Sets architecture: f(X) = ρ(Σ_{x ∈ X} φ(x)).

    Theorem (Zaheer et al., 2017):
        Any permutation-invariant function f: Multiset → ℝ can be written as:
            f({x_1, ..., x_n}) = ρ(Σ_i φ(x_i))
        for suitable φ and ρ.

    This is the theoretical foundation for sum-based GNN aggregation.

    Parameters:
        phi: Element-wise transformation φ: ℝ^d → ℝ^D
        rho: Readout function ρ: ℝ^D → ℝ^k

    Returns:
        Permutation-invariant function with Deep Sets structure

    Notes:
        - φ is typically an MLP (the "message" function)
        - ρ is typically an MLP (the "readout" function)
        - Aggregation must be sum for universal approximation
    """

    def deep_sets_fn(multiset: Iterable) -> Any:
        aggregated = sum(phi(x) for x in multiset)
        return rho(aggregated)

    return deep_sets_fn


def multiset_cardinality_encoding(
    multiset: Iterable[float], max_count: int = 10
) -> np.ndarray:
    """Encode multiset with element values and multiplicities.

    This representation preserves more information than simple aggregation
    but requires discretization or truncation.

    Parameters:
        multiset: Input multiset
        max_count: Maximum multiplicity to encode

    Returns:
        Encoding vector with (value, count) information

    Notes:
        - More expressive than sum/mean/max
        - Not permutation-invariant in raw form (needs sorting)
        - Used in some higher-order GNNs
    """
    counts = Counter(multiset)
    # Sort by value for canonical representation
    sorted_items = sorted(counts.items())

    encoding = []
    for value, count in sorted_items[:max_count]:
        encoding.extend([value, min(count, max_count)])

    # Pad to fixed size
    while len(encoding) < 2 * max_count:
        encoding.extend([0.0, 0])

    return np.array(encoding[: 2 * max_count])
