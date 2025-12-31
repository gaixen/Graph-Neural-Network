"""Formalize equivariance and invariance for permutations and group actions.

This module provides the group-theoretic foundation for understanding GNN symmetries:
- Why GNNs must be permutation-equivariant
- What invariance means for graph-level predictions
- How to verify equivariance properties

Mathematical framework:
    For group G acting on sets X and Y, a function f: X → Y is:
    - G-equivariant if f(g·x) = g·f(x) for all g ∈ G
    - G-invariant if f(g·x) = f(x) for all g ∈ G

    For GNNs, G = S_n (symmetric group) acting by permuting node indices.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Protocol
import numpy as np
from itertools import permutations


X = TypeVar("X")
Y = TypeVar("Y")


class GroupAction(Protocol[X]):
    """Protocol for group action g: G × X → X."""

    def __call__(self, g: any, x: X) -> X:
        """Apply group element g to point x."""
        ...


@dataclass
class PermutationAction:
    """Permutation group S_n acting on sequences/arrays by index reordering.

    For permutation π ∈ S_n and vector v ∈ ℝ^n:
        π · v = [v_π(0), v_π(1), ..., v_π(n-1)]

    This is the fundamental symmetry for GNNs: node ordering is arbitrary.
    """

    @staticmethod
    def apply_to_array(perm: np.ndarray, array: np.ndarray) -> np.ndarray:
        """Apply permutation to array by reindexing.

        Parameters:
            perm: Permutation as array of indices (perm[i] = new position of element i)
            array: Array to permute

        Returns:
            Permuted array
        """
        return array[perm]

    @staticmethod
    def apply_to_matrix(perm: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply permutation to square matrix (adjacency, features, etc.).

        For adjacency matrix A, permutation π acts as:
            (π · A)_ij = A_π(i),π(j)

        Equivalently: π · A = P A P^T where P is the permutation matrix.

        Parameters:
            perm: Permutation as array of indices
            matrix: Square matrix to permute

        Returns:
            Permuted matrix
        """
        return matrix[perm, :][:, perm]

    @staticmethod
    def to_permutation_matrix(perm: np.ndarray) -> np.ndarray:
        """Convert permutation to permutation matrix P.

        P_ij = 1 if perm[i] = j, else 0

        Properties:
            - P^T P = I (orthogonal)
            - det(P) = ±1
            - Used to transform: π · A = P A P^T
        """
        n = len(perm)
        P = np.zeros((n, n))
        P[np.arange(n), perm] = 1
        return P


class EquivariantFunction(ABC, Generic[X, Y]):
    """Abstract base for G-equivariant functions.

    A function f: X → Y is G-equivariant if:
        f(g · x) = g · f(x)  for all g ∈ G, x ∈ X

    For GNNs:
        - Node features transform equivariantly under permutation
        - Edge features transform equivariantly
        - Adjacency matrix transforms equivariantly
    """

    @abstractmethod
    def __call__(self, x: X) -> Y:
        """Apply the equivariant function."""
        pass

    @abstractmethod
    def group_action_input(self, g: any, x: X) -> X:
        """How group acts on input space."""
        pass

    @abstractmethod
    def group_action_output(self, g: any, y: Y) -> Y:
        """How group acts on output space."""
        pass

    def verify_equivariance(
        self, x: X, group_elements: list, tol: float = 1e-6
    ) -> bool:
        """Verify f(g·x) = g·f(x) for given group elements.

        Parameters:
            x: Input to test
            group_elements: List of group elements to check
            tol: Numerical tolerance

        Returns:
            True if equivariant for all tested elements
        """
        fx = self(x)

        for g in group_elements:
            # Compute f(g·x)
            gx = self.group_action_input(g, x)
            f_gx = self(gx)

            # Compute g·f(x)
            g_fx = self.group_action_output(g, fx)

            # Check equality
            if not np.allclose(f_gx, g_fx, atol=tol):
                return False

        return True


class InvariantFunction(ABC, Generic[X]):
    """Abstract base for G-invariant functions.

    A function f: X → ℝ is G-invariant if:
        f(g · x) = f(x)  for all g ∈ G, x ∈ X

    For GNNs:
        - Graph-level predictions must be permutation-invariant
        - Aggregation (sum/mean/max) produces invariant features
    """

    @abstractmethod
    def __call__(self, x: X) -> float | np.ndarray:
        """Apply the invariant function."""
        pass

    @abstractmethod
    def group_action(self, g: any, x: X) -> X:
        """How group acts on input space."""
        pass

    def verify_invariance(self, x: X, group_elements: list, tol: float = 1e-6) -> bool:
        """Verify f(g·x) = f(x) for given group elements.

        Parameters:
            x: Input to test
            group_elements: List of group elements to check
            tol: Numerical tolerance

        Returns:
            True if invariant for all tested elements
        """
        fx = self(x)

        for g in group_elements:
            gx = self.group_action(g, x)
            f_gx = self(gx)

            if not np.allclose(f_gx, fx, atol=tol):
                return False

        return True


class NodeFeatureTransformation(EquivariantFunction[np.ndarray, np.ndarray]):
    """Example: Linear layer on node features is equivariant.

    For h ∈ ℝ^{n×d} (n nodes, d features), linear transformation W:
        f(h) = h W

    is permutation-equivariant because:
        f(π · h) = (π · h) W = π · (h W) = π · f(h)
    """

    def __init__(self, weight_matrix: np.ndarray):
        """Initialize with weight matrix W ∈ ℝ^{d×d'}."""
        self.W = weight_matrix

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Apply linear transformation."""
        return h @ self.W

    def group_action_input(self, perm: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Permutation acts by reordering rows."""
        return h[perm, :]

    def group_action_output(self, perm: np.ndarray, h_out: np.ndarray) -> np.ndarray:
        """Permutation acts by reordering rows."""
        return h_out[perm, :]


class GraphLevelReadout(InvariantFunction[np.ndarray]):
    """Example: Sum pooling is permutation-invariant.

    For node features h ∈ ℝ^{n×d}:
        readout(h) = Σ_i h_i

    is permutation-invariant because sum is commutative.
    """

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Sum over nodes (axis 0)."""
        return np.sum(h, axis=0)

    def group_action(self, perm: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Permutation acts by reordering rows."""
        return h[perm, :]


def generate_random_permutations(n: int, num_samples: int = 100) -> list[np.ndarray]:
    """Generate random permutations for equivariance testing.

    Parameters:
        n: Size of permutation group S_n
        num_samples: Number of random permutations to generate

    Returns:
        List of permutation arrays
    """
    perms = []
    for _ in range(num_samples):
        perm = np.random.permutation(n)
        perms.append(perm)
    return perms


def all_permutations(n: int) -> list[np.ndarray]:
    """Generate all permutations of {0, ..., n-1}.

    Warning: Only feasible for small n (n! grows quickly).

    Parameters:
        n: Size of set to permute

    Returns:
        List of all n! permutations
    """
    if n > 8:
        raise ValueError(
            f"Generating all {np.math.factorial(n)} permutations is infeasible"
        )

    return [np.array(p) for p in permutations(range(n))]
