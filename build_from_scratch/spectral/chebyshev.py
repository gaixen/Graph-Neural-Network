"""Polynomial approximation of spectral filters using Chebyshev polynomials.

This module explains the key insight behind GCN and ChebNet:
    Spectral filters can be approximated by polynomials of the Laplacian,
    and polynomial filters are inherently K-hop local.

Mathematical foundation:
    g(L) = Σ_{k=0}^K θ_k T_k(L̃)

    where T_k are Chebyshev polynomials and L̃ = (2/λ_max)L - I

This transforms global spectral convolution into local spatial convolution!

Key theorem (Hammond et al., 2011):
    A filter g(λ) that decays rapidly can be well-approximated by
    a K-th order polynomial, yielding K-hop localization.

GCN simplification (Kipf & Welling, 2017):
    Restrict to K=1 and λ_max≈2: g(L) = θ_0 I + θ_1 L
    Further simplify to: g(L) = θ(I + D^{-1/2}AD^{-1/2})
"""

from typing import Callable, List
import numpy as np
import warnings


def monomial_polynomial_filter(
    L: np.ndarray, coefficients: List[float], signal: np.ndarray
) -> np.ndarray:
    """Apply polynomial filter using monomial basis: g(L) = Σ θ_k L^k.

    Computes:
        output = Σ_{k=0}^K θ_k L^k x

    where x is the input signal.

    Parameters:
        L: Graph Laplacian (or any matrix operator)
        coefficients: Polynomial coefficients [θ_0, θ_1, ..., θ_K]
        signal: Input signal x ∈ ℝ^n

    Returns:
        Filtered signal g(L)x

    Locality property:
        The k-th term L^k x depends only on k-hop neighbors.
        Thus, deg(polynomial) = K ⟹ K-hop receptive field.

    Computational complexity:
        O(K · |E|) since each L^k multiplication is O(|E|) for sparse L.

    Note:
        Monomial basis has numerical issues for large K.
        Chebyshev basis is preferred for K > 3.
    """
    n = len(signal)
    output = np.zeros(n)

    # L^0 = I
    L_power = np.eye(n)

    for k, theta_k in enumerate(coefficients):
        output += theta_k * (L_power @ signal)

        if k < len(coefficients) - 1:
            L_power = L_power @ L

    return output


def chebyshev_polynomial_filter(
    L: np.ndarray,
    coefficients: List[float],
    signal: np.ndarray,
    lambda_max: float | None = None,
) -> np.ndarray:
    """Apply polynomial filter using Chebyshev basis for numerical stability.

    Chebyshev polynomials T_k satisfy:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)

    For the Laplacian, we rescale to [-1, 1]:
        L̃ = (2/λ_max) L - I

    Then apply:
        output = Σ_{k=0}^K θ_k T_k(L̃) x

    Parameters:
        L: Graph Laplacian
        coefficients: Chebyshev coefficients [θ_0, θ_1, ..., θ_K]
        signal: Input signal x
        lambda_max: Largest eigenvalue of L (estimated if None)

    Returns:
        Filtered signal

    Advantages over monomial basis:
        - Numerically stable for large K
        - Better approximation quality
        - Recursion avoids computing L^k explicitly

    Applications:
        - ChebNet (Defferrard et al., 2016)
        - Spectral GCN precursor
        - Graph wavelets
    """
    if lambda_max is None:
        # Estimate λ_max using power iteration or exact computation
        eigenvalues = np.linalg.eigvalsh(L)
        lambda_max = float(np.max(eigenvalues))

        if lambda_max < 1e-10:
            warnings.warn("λ_max is very small; graph may be disconnected")
            lambda_max = 1.0  # Avoid division by zero

    # Rescale Laplacian to [-1, 1]
    L_rescaled = (2.0 / lambda_max) * L - np.eye(len(L))

    K = len(coefficients) - 1
    n = len(signal)

    # Initialize recursion
    T_k_minus_1 = signal  # T_0(L̃) x = x
    T_k = L_rescaled @ signal  # T_1(L̃) x = L̃ x

    output = coefficients[0] * T_k_minus_1
    if K >= 1:
        output += coefficients[1] * T_k

    # Recursive evaluation
    for k in range(2, K + 1):
        T_k_plus_1 = 2 * (L_rescaled @ T_k) - T_k_minus_1
        output += coefficients[k] * T_k_plus_1

        # Update for next iteration
        T_k_minus_1 = T_k
        T_k = T_k_plus_1

    return output


def gcn_filter(
    A: np.ndarray, signal: np.ndarray, add_self_loops: bool = True
) -> np.ndarray:
    """Simplified GCN filter: 1st-order Chebyshev approximation.

    GCN derivation:
        Start with K=1 Chebyshev: g(L) ≈ θ_0 T_0 + θ_1 T_1
        Approximate λ_max ≈ 2 for normalized Laplacian
        Simplify: g(L) ≈ θ_0 I + θ_1 (L - I) = θ_0 I - θ_1 L

        Since L = I - D^{-1/2}AD^{-1/2}:
            g(L) = θ_0 I - θ_1 (I - D^{-1/2}AD^{-1/2})
                 = (θ_0 - θ_1) I + θ_1 D^{-1/2}AD^{-1/2}

        Constraint θ_0 = -θ_1 for symmetry:
            g(L) = θ (I + D^{-1/2}AD^{-1/2}) x

    In GCN, the "I +" part adds self-connections.

    Parameters:
        A: Adjacency matrix
        signal: Input signal x ∈ ℝ^n
        add_self_loops: Add identity (I + A) before normalization

    Returns:
        Filtered signal (unnormalized by θ; that's absorbed into layer weights)

    Interpretation:
        Averages each node's features with its neighbors' features.
        This is exactly 1-hop neighborhood aggregation!
    """
    if add_self_loops:
        A_tilde = A + np.eye(len(A))
    else:
        A_tilde = A

    # Symmetric normalization
    degrees = np.sum(A_tilde, axis=1)
    D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d > 0 else 0.0 for d in degrees])

    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return A_norm @ signal


def polynomial_degree_vs_locality(K: int, graph_diameter: int) -> str:
    """Explain relationship between polynomial degree and receptive field.

    Parameters:
        K: Polynomial degree
        graph_diameter: Graph diameter (longest shortest path)

    Returns:
        Explanation string

    Key insight:
        - K=1: Each node sees 1-hop neighbors
        - K=2: Each node sees 2-hop neighbors (friends of friends)
        - K=diameter: Each node sees entire graph (global)

    Theorem:
        For a filter g(L) = Σ_{k=0}^K θ_k L^k,
        node i's output depends only on nodes within distance K.

    Proof sketch:
        (L^k)_{ij} = 0 if shortest path d(i,j) > k
        (by induction on powers of Laplacian)

    GNN implication:
        Depth K of polynomial ≈ depth K of message-passing layers.
        Both have K-hop receptive field.
    """
    if K >= graph_diameter:
        return (
            f"Polynomial degree K={K} ≥ graph diameter {graph_diameter}. "
            "Filter is effectively global; all nodes interact."
        )
    else:
        return (
            f"Polynomial degree K={K} < graph diameter {graph_diameter}. "
            f"Each node sees only {K}-hop neighborhood. "
            "Filter is localized."
        )


def approximate_filter_with_polynomial(
    filter_function: Callable[[float], float],
    degree: int,
    lambda_min: float = 0.0,
    lambda_max: float = 2.0,
    method: str = "least_squares",
) -> List[float]:
    """Approximate arbitrary spectral filter with polynomial.

    Given desired filter response g(λ), find polynomial p(λ) = Σ θ_k λ^k
    such that p(λ) ≈ g(λ) for λ ∈ [λ_min, λ_max].

    Parameters:
        filter_function: Desired filter g(λ) (e.g., low-pass, high-pass)
        degree: Polynomial degree K
        lambda_min: Minimum eigenvalue
        lambda_max: Maximum eigenvalue
        method: Approximation method ("least_squares", "chebyshev_points")

    Returns:
        Polynomial coefficients [θ_0, ..., θ_K]

    Applications:
        - Design custom graph filters (wavelets, smoothing, sharpening)
        - Generalize GCN to arbitrary spectral responses

    Example:
        Low-pass (smooth):  g(λ) = exp(-λ)
        High-pass (sharp):  g(λ) = λ
        Band-pass:          g(λ) = exp(-(λ - μ)²)
    """
    if method == "least_squares":
        # Sample eigenvalue range and solve least-squares problem
        num_samples = max(100, 10 * degree)
        lambdas = np.linspace(lambda_min, lambda_max, num_samples)

        # Build Vandermonde matrix
        V = np.vander(lambdas, N=degree + 1, increasing=True)

        # Target filter responses
        targets = np.array([filter_function(lam) for lam in lambdas])

        # Solve least-squares: V θ ≈ targets
        coefficients, _, _, _ = np.linalg.lstsq(V, targets, rcond=None)

        return list(coefficients)

    else:
        raise ValueError(f"Unknown method: {method}")
