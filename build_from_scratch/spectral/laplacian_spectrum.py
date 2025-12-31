"""Eigen-decomposition of Laplacians and spectral interpretations.

This module analyzes the spectral properties of graph Laplacians:
1. Eigenvalues encode connectivity and mixing time
2. Eigenvectors form a basis (graph Fourier basis)
3. Spectral gap relates to expansion and convergence
4. Fiedler vector (λ₂) detects communities

Mathematical framework:
    Graph Laplacian: L = D - A
    Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    Random walk Laplacian: L_rw = I - D^{-1} A

Spectral theorem:
    L is symmetric positive semi-definite:
    L = Σᵢ λᵢ vᵢvᵢᵀ where 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ

Key insight:
    Eigenvalues control GNN oversmoothing:
    - Small spectral gap (λ₂ ≈ 0) → slow mixing → delayed oversmoothing
    - Large spectral gap (λ₂ large) → fast mixing → rapid oversmoothing
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class LaplacianType(Enum):
    """Type of graph Laplacian."""

    UNNORMALIZED = "unnormalized"  # L = D - A
    SYMMETRIC = "symmetric"  # L_sym = I - D^{-1/2} A D^{-1/2}
    RANDOM_WALK = "random_walk"  # L_rw = I - D^{-1} A


@dataclass
class SpectralDecomposition:
    """Result of spectral decomposition of graph Laplacian.

    Attributes:
        eigenvalues: Sorted eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ
        eigenvectors: Corresponding eigenvectors as columns
        laplacian_type: Which Laplacian was decomposed
        num_connected_components: Multiplicity of λ = 0
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    laplacian_type: LaplacianType
    num_connected_components: int

    @property
    def spectral_gap(self) -> float:
        """Gap between first and second eigenvalues.

        Interpretation:
            - λ₂ = 0: graph is disconnected
            - λ₂ small: weak connectivity, slow mixing
            - λ₂ large: strong connectivity, fast mixing
        """
        if len(self.eigenvalues) < 2:
            return 0.0
        return float(self.eigenvalues[self.num_connected_components])

    @property
    def fiedler_vector(self) -> np.ndarray:
        """Second eigenvector (Fiedler vector).

        Applications:
            - Graph partitioning: sign(v₂) splits into two communities
            - Centrality measure
            - Embedding coordinate
        """
        if len(self.eigenvalues) < 2:
            raise ValueError("Graph has < 2 eigenvalues")
        return self.eigenvectors[:, self.num_connected_components]

    @property
    def spectral_radius(self) -> float:
        """Maximum eigenvalue."""
        return float(np.max(self.eigenvalues))


def laplacian(
    A: np.ndarray, laplacian_type: LaplacianType = LaplacianType.UNNORMALIZED
) -> np.ndarray:
    """Compute graph Laplacian matrix.

    Mathematical definitions:
        - Unnormalized: L = D - A
        - Symmetric: L_sym = I - D^{-1/2} A D^{-1/2}
        - Random walk: L_rw = I - D^{-1} A = D^{-1}L

    Properties of unnormalized Laplacian:
        - Positive semi-definite: xᵀLx = Σ_{(i,j)∈E} (x_i - x_j)²
        - Kernel: constant vector 1 (L1 = 0)
        - Eigenvalues: 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ ≤ 2·max_degree

    Properties of normalized Laplacian:
        - Eigenvalues in [0, 2]
        - Better for graphs with heterogeneous degrees
        - Used in spectral clustering

    GNN connection:
        - GCN uses I - D^{-1/2} A D^{-1/2} = -L_sym (negative symmetric Laplacian)
        - Spectral filtering: g(L) applies frequency filter

    Args:
        A: Adjacency matrix (n × n)
        laplacian_type: Which Laplacian to compute

    Returns:
        Laplacian matrix (n × n)
    """
    n = A.shape[0]
    d = np.sum(A, axis=1)
    D = np.diag(d)

    if laplacian_type == LaplacianType.UNNORMALIZED:
        return D - A

    elif laplacian_type == LaplacianType.SYMMETRIC:
        # Handle isolated nodes (degree = 0)
        d_inv_sqrt = np.zeros(n)
        d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
        D_inv_sqrt = np.diag(d_inv_sqrt)

        return np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    elif laplacian_type == LaplacianType.RANDOM_WALK:
        # Handle isolated nodes
        d_inv = np.zeros(n)
        d_inv[d > 0] = 1.0 / d[d > 0]
        D_inv = np.diag(d_inv)

        return np.eye(n) - D_inv @ A

    else:
        raise ValueError(f"Unknown Laplacian type: {laplacian_type}")


def spectrum(
    L: np.ndarray,
    laplacian_type: LaplacianType = LaplacianType.UNNORMALIZED,
    tol: float = 1e-10,
) -> SpectralDecomposition:
    """Compute eigendecomposition of graph Laplacian.

    Mathematical result:
        L = VΛVᵀ where Λ = diag(λ₁, ..., λₙ) and V = [v₁ | ... | vₙ]

    Properties:
        - L is symmetric → real eigenvalues, orthogonal eigenvectors
        - L is PSD → all eigenvalues ≥ 0
        - Multiplicity of λ = 0 = number of connected components

    Eigenvalue interpretation:
        - λᵢ measures "frequency" of eigenvector vᵢ
        - Low frequency (λ ≈ 0): smooth across graph
        - High frequency (λ large): rapid variation across edges

    GNN connection:
        - Oversmoothing = convergence to lowest eigenvector (constant)
        - Rate controlled by spectral gap λ₂

    Args:
        L: Laplacian matrix (must be symmetric!)
        laplacian_type: Type of Laplacian (for interpretation)
        tol: Tolerance for determining zero eigenvalues

    Returns:
        SpectralDecomposition with sorted eigenvalues/vectors
    """
    # Compute eigendecomposition
    vals, vecs = np.linalg.eigh(L)  # eigh for symmetric matrices

    # Sort in ascending order (eigh should already do this, but be explicit)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Count connected components (multiplicity of λ = 0)
    num_components = np.sum(np.abs(vals) < tol)

    return SpectralDecomposition(
        eigenvalues=vals,
        eigenvectors=vecs,
        laplacian_type=laplacian_type,
        num_connected_components=int(num_components),
    )


def analyze_connectivity(decomp: SpectralDecomposition) -> dict:
    """Analyze graph connectivity from spectrum.

    Mathematical theorems:
        1. λ₁ = 0 always (constant eigenvector)
        2. λ₂ = 0 ⟺ graph is disconnected
        3. λ₂ (algebraic connectivity) lower bounds edge connectivity
        4. Cheeger inequality: h/2 ≤ λ₂ ≤ 2h where h = edge expansion

    Returns:
        Dictionary with connectivity metrics
    """
    gap = decomp.spectral_gap

    analysis = {
        "num_components": decomp.num_connected_components,
        "is_connected": (decomp.num_connected_components == 1),
        "spectral_gap": gap,
        "algebraic_connectivity": gap,
    }

    # Estimate mixing time (for random walk)
    if gap > 1e-10:
        # Mixing time ~ 1/λ₂ (for random walk Laplacian)
        analysis["mixing_time_estimate"] = 1.0 / gap
    else:
        analysis["mixing_time_estimate"] = float("inf")

    # Oversmoothing rate (for GNN)
    # After k layers, difference from equilibrium ~ (1 - λ₂)^k
    if decomp.laplacian_type == LaplacianType.SYMMETRIC:
        convergence_rate = 1.0 - gap if gap < 1.0 else 0.0
        analysis["oversmoothing_rate"] = convergence_rate

    return analysis


def cheeger_inequality(
    decomp: SpectralDecomposition, max_degree: Optional[float] = None
) -> Tuple[float, float]:
    """Cheeger inequality: relates λ₂ to edge expansion.

    Theorem (Cheeger inequality):
        For normalized Laplacian:
        h²/2 ≤ λ₂ ≤ 2h

        where h = edge expansion (Cheeger constant)

    This gives two-sided bound:
        - Lower bound on expansion from λ₂
        - Upper bound on λ₂ from expansion

    Interpretation:
        - Large λ₂ → graph is well-connected (expander)
        - Small λ₂ → graph has bottleneck (sparse cut)

    Args:
        decomp: Spectral decomposition (should be normalized Laplacian)
        max_degree: Maximum degree (for unnormalized case)

    Returns:
        (lower_bound, upper_bound) on edge expansion
    """
    lambda_2 = decomp.spectral_gap

    if decomp.laplacian_type == LaplacianType.SYMMETRIC:
        # h²/2 ≤ λ₂ ≤ 2h
        # → λ₂/2 ≤ h ≤ √(2λ₂)
        lower = lambda_2 / 2.0
        upper = np.sqrt(2.0 * lambda_2)
    elif decomp.laplacian_type == LaplacianType.UNNORMALIZED and max_degree is not None:
        # For unnormalized: need to normalize by max degree
        lambda_2_norm = lambda_2 / max_degree
        lower = lambda_2_norm / 2.0
        upper = np.sqrt(2.0 * lambda_2_norm)
    else:
        # Cannot apply Cheeger inequality
        lower, upper = None, None

    return lower, upper


def fiedler_partitioning(
    decomp: SpectralDecomposition, nodes: Optional[List] = None
) -> Tuple[List, List]:
    """Spectral graph partitioning using Fiedler vector.

    Algorithm:
        1. Compute Fiedler vector v₂ (eigenvector for λ₂)
        2. Partition: S₁ = {i : v₂[i] ≥ 0}, S₂ = {i : v₂[i] < 0}

    Theorem (Fiedler):
        This minimizes the Cheeger cut (approximately).

    Mathematical intuition:
        - Fiedler vector solves:
          min vᵀLv subject to vᵀ1 = 0, ‖v‖ = 1
        - Relaxation of discrete cut problem

    GNN connection:
        - Sign of Fiedler vector gives natural "label" for nodes
        - Can use as positional encoding
        - Captures community structure

    Args:
        decomp: Spectral decomposition
        nodes: Optional node list (default: indices)

    Returns:
        (partition1, partition2) as lists of nodes
    """
    fiedler = decomp.fiedler_vector
    n = len(fiedler)

    if nodes is None:
        nodes = list(range(n))

    partition1 = [nodes[i] for i in range(n) if fiedler[i] >= 0]
    partition2 = [nodes[i] for i in range(n) if fiedler[i] < 0]

    return partition1, partition2


def spectral_embedding(decomp: SpectralDecomposition, k: int = 2) -> np.ndarray:
    """Embed graph nodes using k smallest eigenvectors.

    Mathematical construction:
        Embed node i to: [v₂[i], v₃[i], ..., v_{k+1}[i]] ∈ ℝᵏ

    Properties:
        - Preserves graph structure (nearby nodes → nearby embeddings)
        - Smooth embeddings (minimize Σ ‖embedding[i] - embedding[j]‖²)

    Theorem:
        This embedding minimizes:
        Σ_{(i,j)∈E} ‖xᵢ - xⱼ‖²
        subject to orthogonality constraints.

    GNN connection:
        - Can use as positional encoding
        - Provides global graph structure information
        - Complements local message passing

    Args:
        decomp: Spectral decomposition
        k: Embedding dimension

    Returns:
        Embedding matrix (n × k)
    """
    # Skip first k_skip zero eigenvalues (one per component)
    k_skip = decomp.num_connected_components

    if k_skip + k > len(decomp.eigenvalues):
        raise ValueError(
            f"Not enough eigenvectors: need {k_skip + k}, have {len(decomp.eigenvalues)}"
        )

    # Use eigenvectors 2, 3, ..., k+1 (skipping the constant ones)
    embedding = decomp.eigenvectors[:, k_skip : k_skip + k]

    return embedding
