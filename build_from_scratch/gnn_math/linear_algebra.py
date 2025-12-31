"""Linear algebra foundations for graph operators.

This module provides core linear algebra primitives essential for understanding
GNN behavior, particularly:
- Spectral decomposition and analysis
- Matrix power iteration and convergence
- Projection operators and subspace analysis
- Rank collapse detection

These tools reveal why oversmoothing is mathematically inevitable in deep GNNs:
repeated application of normalized adjacency matrices causes node representations
to converge to the dominant eigenspace, collapsing information.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import warnings


@dataclass
class EigenDecomposition:
    """Container for eigenvalue decomposition results.

    Attributes:
        eigenvalues: Array of eigenvalues λ sorted in descending order by magnitude
        eigenvectors: Matrix where column i is the eigenvector for eigenvalues[i]
        is_symmetric: Whether the original matrix was symmetric (guarantees real eigenvalues)
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_symmetric: bool = False


def eigen_decomposition(
    A: np.ndarray, symmetric: Optional[bool] = None, sort_by_magnitude: bool = True
) -> EigenDecomposition:
    """Compute eigenvalue decomposition of matrix A.

    For a matrix A ∈ ℝ^{n×n}, computes:
        A = V Λ V^{-1}
    where Λ is diagonal with eigenvalues λ_i and V contains eigenvectors.

    Parameters:
        A: Square matrix to decompose
        symmetric: If True, use optimized symmetric solver. If None, auto-detect.
        sort_by_magnitude: Sort eigenvalues by |λ_i| in descending order

    Returns:
        EigenDecomposition with eigenvalues and eigenvectors

    Notes:
        - For symmetric matrices, eigenvalues are real and eigenvectors orthogonal
        - For graph Laplacians L, eigenvalues satisfy 0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1}
        - Spectral gap (λ_1 - λ_0) relates to graph connectivity and mixing time
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")

    # Auto-detect symmetry if not specified
    if symmetric is None:
        symmetric = np.allclose(A, A.T)

    if symmetric:
        # Use optimized solver for real symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(A)

    if sort_by_magnitude:
        # Sort by magnitude in descending order
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    return EigenDecomposition(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors, is_symmetric=symmetric
    )


def spectral_radius(A: np.ndarray) -> float:
    """Compute the spectral radius ρ(A) of matrix A.

    The spectral radius is defined as:
        ρ(A) = max{|λ_i| : λ_i ∈ spec(A)}

    Parameters:
        A: Square matrix

    Returns:
        Maximum absolute eigenvalue

    Notes:
        - ρ(A) bounds the convergence rate of matrix powers: A^k → 0 iff ρ(A) < 1
        - For normalized adjacency Ã = D^{-1/2}AD^{-1/2}, we have ρ(Ã) ≤ 1
        - If ρ(A) = 1 with a unique dominant eigenvector, power iteration converges
    """
    eigenvalues = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigenvalues)))


def matrix_power_sequence(
    A: np.ndarray, x0: np.ndarray, num_iterations: int, return_all: bool = False
) -> np.ndarray | list[np.ndarray]:
    """Compute sequence x_0, x_1 = Ax_0, x_2 = A^2x_0, ..., x_k = A^k x_0.

    This reveals the convergence behavior of repeated graph convolutions:
        h^{(k+1)} = Ã h^{(k)}

    Parameters:
        A: Transition matrix (e.g., normalized adjacency)
        x0: Initial vector
        num_iterations: Number of power iterations k
        return_all: If True, return list of all iterates; else return only final

    Returns:
        Final iterate A^k x_0, or list of all iterates if return_all=True

    Notes:
        - For normalized adjacency with ρ(Ã) = 1, sequence converges to projection
          onto dominant eigenspace
        - Convergence rate is geometric in spectral gap: ‖x_k - x_∞‖ ≈ O(λ_2^k)
    """
    x = x0.astype(float).copy()
    if return_all:
        sequence = [x.copy()]
        for _ in range(num_iterations):
            x = A @ x
            sequence.append(x.copy())
        return sequence
    else:
        for _ in range(num_iterations):
            x = A @ x
        return x


def power_iteration_convergence(
    P: np.ndarray, x0: np.ndarray, tol: float = 1e-8, max_iter: int = 1000
) -> Tuple[bool, int, float]:
    """Analyze convergence of power iteration x_{t+1} = P x_t.

    For normalized adjacency matrices:
        h^{(k+1)} = D^{-1/2}AD^{-1/2} h^{(k)}

    this reveals when and why oversmoothing occurs.

    Parameters:
        P: Transition/diffusion matrix
        x0: Initial vector
        tol: Convergence tolerance ‖x_{t+1} - x_t‖ < tol
        max_iter: Maximum iterations

    Returns:
        Tuple of (converged, num_iterations, final_residual)

    Mathematical insight:
        If P has spectral radius ρ(P) = 1 with unique dominant eigenvector v,
        then x_t → (v^T x_0) v, collapsing all information into a 1D subspace.
        This is the root cause of oversmoothing in deep GNNs.
    """
    x = x0.astype(float).copy()

    for iteration in range(max_iter):
        x_next = P @ x
        residual = np.linalg.norm(x_next - x)

        if residual < tol:
            return True, iteration + 1, float(residual)

        x = x_next

    final_residual = np.linalg.norm(P @ x - x)
    return False, max_iter, float(final_residual)


def orthogonal_projection(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project vector v onto column space of U.

    Computes:
        proj_U(v) = U(U^T U)^{-1} U^T v = QQ^T v

    where Q is the orthonormal basis obtained via QR decomposition.

    Parameters:
        U: Matrix whose columns span the subspace (n × k)
        v: Vector to project (n,)

    Returns:
        Projected vector in col(U)

    Notes:
        - If U has orthonormal columns (U^T U = I), simplifies to UU^T v
        - Used to analyze which components of signals survive aggregation
        - In GNNs, repeated aggregation projects onto low-frequency eigenspaces
    """
    Q, _ = np.linalg.qr(U)
    return Q @ (Q.T @ v)


def rank_and_nullspace(A: np.ndarray, tol: float = 1e-10) -> Tuple[int, np.ndarray]:
    """Compute numerical rank and nullspace basis of matrix A.

    Parameters:
        A: Matrix to analyze (m × n)
        tol: Singular values < tol are considered zero

    Returns:
        Tuple of (numerical_rank, nullspace_basis)

    Notes:
        - Rank collapse occurs when rank(A^k) < rank(A^{k-1})
        - For adjacency powers, this indicates information bottleneck
        - Nullspace dimension = n - rank (by rank-nullity theorem)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    rank = np.sum(s > tol)

    # Nullspace is spanned by right singular vectors with zero singular values
    if rank < A.shape[1]:
        nullspace_basis = Vt[rank:, :].T
    else:
        nullspace_basis = np.zeros((A.shape[1], 0))

    return int(rank), nullspace_basis


def effective_rank(A: np.ndarray) -> float:
    """Compute effective rank using entropy of normalized singular values.

    Defined as:
        rank_eff(A) = exp(- Σ_i p_i log p_i)
    where p_i = σ_i / Σ_j σ_j are normalized singular values.

    Parameters:
        A: Matrix to analyze

    Returns:
        Effective rank (between 1 and min(m,n))

    Notes:
        - Effective rank smoothly captures "how close" a matrix is to low rank
        - For GNN analysis: tracks how quickly representations collapse
        - rank_eff ≈ min(m,n) means uniform spectrum (no collapse)
        - rank_eff ≈ 1 indicates severe information loss
    """
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 1e-12]  # Filter numerical zeros

    if len(s) == 0:
        return 0.0

    # Normalize to probability distribution
    p = s / np.sum(s)

    # Compute entropy and exponentiate
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def spectral_gap(L: np.ndarray, k: int = 1) -> float:
    """Compute the k-th spectral gap of Laplacian L.

    For graph Laplacian with eigenvalues 0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1},
    the spectral gap is λ_k - λ_{k-1}.

    Parameters:
        L: Graph Laplacian matrix
        k: Which gap to compute (default: 1, the algebraic connectivity)

    Returns:
        Spectral gap λ_k - λ_{k-1}

    Notes:
        - First gap λ_1 is the algebraic connectivity (Fiedler value)
        - λ_1 = 0 iff graph is disconnected
        - Larger λ_1 → faster mixing → faster oversmoothing
        - Gap controls convergence rate of random walks and diffusions
    """
    if not np.allclose(L, L.T):
        warnings.warn("Matrix is not symmetric; results may be incorrect")

    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)

    if k >= len(eigenvalues):
        raise ValueError(f"k={k} exceeds matrix dimension {len(eigenvalues)}")

    return float(eigenvalues[k] - eigenvalues[k - 1])
