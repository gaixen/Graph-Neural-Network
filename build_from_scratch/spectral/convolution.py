"""Spectral convolution as a global operator and why it does not localize by default.

This module demonstrates the fundamental insight behind modern GNNs:
    Spectral convolution is GLOBAL (requires full eigendecomposition),
    but can be LOCALIZED using polynomial approximations.

Key theorem:
    Convolution theorem for graphs:

    Spatial domain:  (f ★ g)[i] = Σⱼ f[j] · g(d(i,j))
    Spectral domain: f̂ ★ g = f̂ ⊙ ĝ (pointwise multiplication)

    where ⊙ = element-wise product in Fourier domain.

Crucial insight (Hammond et al., 2011):
    If filter ĝ(λ) is a polynomial of degree K:
    ĝ(λ) = Σₖ θₖ λᵏ

    Then spectral convolution becomes K-localized in spatial domain!

    This transforms GLOBAL spectral filtering into LOCAL message passing.

GNN evolution:
    1. Spectral GNN (Bruna et al., 2013): Full eigendecomposition - O(n²)
    2. ChebNet (Defferrard et al., 2016): Polynomial filters - O(Km)
    3. GCN (Kipf & Welling, 2017): K=1 approximation - O(m)
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpectralFilter:
    """Spectral filter: function g(λ) applied to Laplacian eigenvalues.

    Attributes:
        filter_func: Function g: ℝ → ℝ
        is_polynomial: Whether filter is a polynomial
        polynomial_degree: Degree if polynomial (determines locality)
    """

    filter_func: Callable[[float], float]
    is_polynomial: bool = False
    polynomial_degree: Optional[int] = None

    @property
    def is_local(self) -> bool:
        """Local iff filter is polynomial."""
        return self.is_polynomial

    @property
    def receptive_field_radius(self) -> Optional[int]:
        """Spatial receptive field = polynomial degree."""
        return self.polynomial_degree if self.is_polynomial else None


def spectral_filter(L: np.ndarray, theta: Callable[[float], float]) -> np.ndarray:
    """Apply spectral filter: global operation requiring eigendecomposition.

    Mathematical operation:
        F = V · g(Λ) · Vᵀ

        where L = VΛVᵀ and g(Λ) = diag(g(λ₁), ..., g(λₙ))

    Then for signal s:
        s_filtered = F · s = V · g(Λ) · Vᵀ · s

    Complexity:
        - Eigendecomposition: O(n³)
        - Per application: O(n²)
        - INTRACTABLE for large graphs!

    Information encoded:
        - Complete frequency response g(λ)
        - Can implement arbitrary filters

    Information lost:
        - Spatial locality (global operation)
        - Computational efficiency

    Why not practical:
        - Scales as O(n³) for preprocessing + O(n²) per forward pass
        - Cannot batch graphs of different sizes
        - Eigenvectors not stable to perturbations

    Args:
        L: Graph Laplacian (n × n)
        theta: Filter function g(λ)

    Returns:
        Filter matrix F (n × n)

    Use case:
        Only for small graphs or theoretical analysis.
        For practical GNNs, use polynomial approximations!
    """
    vals, vecs = np.linalg.eigh(L)
    filt = np.diag([theta(lam) for lam in vals])
    return vecs @ filt @ vecs.T


def spectral_convolution(
    L: np.ndarray, signal: np.ndarray, filter_func: Callable[[float], float]
) -> np.ndarray:
    """Spectral convolution: filter signal in frequency domain.

    Mathematical operation:
        s_out = V · g(Λ) · Vᵀ · s

    Interpretation:
        1. Transform to Fourier domain: ŝ = Vᵀs
        2. Multiply by filter: ŝ_out = g(Λ) ŝ
        3. Transform back: s_out = V ŝ_out

    This is the ORIGINAL spectral GNN approach (Bruna et al., 2013).

    Limitations:
        - Requires eigendecomposition (O(n³))
        - Not spatially local
        - Learned filters are graph-dependent (cannot transfer)

    Args:
        L: Laplacian
        signal: Input signal
        filter_func: Spectral filter g(λ)

    Returns:
        Filtered signal
    """
    F = spectral_filter(L, filter_func)
    return F @ signal


def polynomial_spectral_filter(
    L: np.ndarray, coefficients: np.ndarray, signal: np.ndarray
) -> np.ndarray:
    """Polynomial spectral filter: g(L) = Σₖ θₖ Lᵏ.

    Key insight:
        g(L)s = (Σₖ θₖ Lᵏ)s = Σₖ θₖ (Lᵏs)

    Complexity:
        - Lᵏs computed via K matrix-vector products: O(Km)
        - No eigendecomposition needed!
        - K-localized: (Lᵏ)ᵢⱼ ≠ 0 only if d(i,j) ≤ k

    Mathematical proof of locality:
        Lᵏ encodes k-hop paths:
        (Lᵏ)ᵢⱼ involves only nodes within k hops of i and j.

    This is the foundation of ChebNet and GCN!

    Args:
        L: Laplacian
        coefficients: Polynomial coefficients [θ₀, θ₁, ..., θ_K]
        signal: Input signal

    Returns:
        Filtered signal: Σₖ θₖ Lᵏs

    Complexity:
        Time: O(K · m · d) where m = edges, d = signal dimension
        Space: O(n · d)
    """
    K = len(coefficients) - 1
    n = signal.shape[0]

    # Initialize with θ₀ · I · s
    result = coefficients[0] * signal

    # Iteratively compute Lᵏs
    L_k_signal = signal.copy()

    for k in range(1, K + 1):
        L_k_signal = L @ L_k_signal  # Lᵏs = L · (Lᵏ⁻¹s)
        result += coefficients[k] * L_k_signal

    return result


def verify_localization(L: np.ndarray, K: int, adjacency: np.ndarray) -> bool:
    """Verify that Lᵏ is K-localized.

    Theorem:
        (Lᵏ)ᵢⱼ = 0 if shortest path distance d(i, j) > k

    Proof:
        L = D - A, so Lᵏ involves paths of length ≤ k.

    This function checks this empirically.

    Args:
        L: Laplacian
        K: Polynomial degree
        adjacency: Adjacency matrix

    Returns:
        True if Lᵏ respects K-hop locality
    """
    n = L.shape[0]

    # Compute Lᵏ
    L_k = np.linalg.matrix_power(L, K)

    # Compute k-hop adjacency: (I + A)ᵏ
    k_hop_adj = np.linalg.matrix_power(np.eye(n) + adjacency, K)

    # Check: Lᵏ should be zero where k-hop adjacency is zero
    # (Allowing small numerical errors)
    tol = 1e-10

    for i in range(n):
        for j in range(n):
            if k_hop_adj[i, j] == 0 and abs(L_k[i, j]) > tol:
                # Found non-zero entry beyond k hops!
                return False

    return True


def analyze_filter_locality(
    filter_func: Callable[[float], float], eigenvalues: np.ndarray, K: int
) -> dict:
    """Analyze how well a filter can be approximated by degree-K polynomial.

    This measures the "polynomial approximability" of a spectral filter.

    Method:
        Fit polynomial p_K(λ) to minimize Σᵢ |g(λᵢ) - p_K(λᵢ)|²

    Interpretation:
        - Small error → filter is approximately K-local
        - Large error → requires higher degree or not localizable

    Args:
        filter_func: Spectral filter g(λ)
        eigenvalues: Laplacian eigenvalues
        K: Polynomial degree

    Returns:
        Dictionary with approximation analysis
    """
    # Evaluate filter at eigenvalues
    filter_values = np.array([filter_func(lam) for lam in eigenvalues])

    # Fit polynomial using least squares
    # Build Vandermonde matrix
    n = len(eigenvalues)
    V = np.vander(eigenvalues, N=K + 1, increasing=True)

    # Solve: min_θ ‖Vθ - filter_values‖²
    coeffs, residuals, rank, s = np.linalg.lstsq(V, filter_values, rcond=None)

    # Compute approximation error
    approx_values = V @ coeffs
    max_error = np.max(np.abs(filter_values - approx_values))
    mean_error = np.mean(np.abs(filter_values - approx_values))
    rel_error = mean_error / (np.mean(np.abs(filter_values)) + 1e-10)

    return {
        "polynomial_degree": K,
        "coefficients": coeffs,
        "max_approximation_error": float(max_error),
        "mean_approximation_error": float(mean_error),
        "relative_error": float(rel_error),
        "is_well_approximated": (rel_error < 0.1),  # Heuristic threshold
    }


def demonstrate_global_vs_local(
    L: np.ndarray, signal: np.ndarray, filter_func: Callable[[float], float], K: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compare global spectral filter vs. polynomial approximation.

    This demonstrates the key tradeoff:
        - Global filter: exact but expensive (O(n³))
        - Polynomial approximation: approximate but efficient (O(Km))

    Args:
        L: Laplacian
        signal: Input signal
        filter_func: Target filter
        K: Polynomial degree for approximation

    Returns:
        (global_filtered, polynomial_filtered, error)
    """
    # Global spectral filtering
    global_filtered = spectral_convolution(L, signal, filter_func)

    # Polynomial approximation
    vals = np.linalg.eigvalsh(L)
    analysis = analyze_filter_locality(filter_func, vals, K)
    coeffs = analysis["coefficients"]

    polynomial_filtered = polynomial_spectral_filter(L, coeffs, signal)

    # Compute error
    error = np.linalg.norm(global_filtered - polynomial_filtered) / np.linalg.norm(
        global_filtered
    )

    return global_filtered, polynomial_filtered, float(error)


def gcn_as_spectral_filter() -> SpectralFilter:
    """GCN as a spectral filter: g(λ) ≈ 1 - λ.

    Derivation (Kipf & Welling, 2017):
        Start with: g(Λ) = θ₀I + θ₁Λ (first-order Chebyshev)
        Assume λ_max ≈ 2 and θ₀ = -θ₁
        Get: g(Λ) = θ(I - Λ) where Λ normalized

    This is a LOW-PASS filter:
        - g(0) = θ (passes DC component)
        - g(λ) decreases with λ (attenuates high frequencies)

    Consequence:
        GCN smooths signals, leading to oversmoothing!

    Returns:
        SpectralFilter object for GCN
    """

    def gcn_filter(lam: float, theta: float = 1.0) -> float:
        """GCN filter: g(λ) = θ(1 - λ/2)"""
        return theta * (1.0 - lam / 2.0)

    return SpectralFilter(
        filter_func=lambda lam: gcn_filter(lam), is_polynomial=True, polynomial_degree=1
    )


def explain_why_polynomials_localize() -> str:
    """Explain the mathematical reason polynomials localize.

    Returns:
        Explanation string
    """
    return """
    WHY POLYNOMIAL FILTERS LOCALIZE:
    
    Theorem:
        If g(λ) is a polynomial of degree K, then:
        (g(L))ᵢⱼ = 0 whenever d(i, j) > K
    
    Proof:
        1. L = D - A encodes 1-hop structure:
           Lᵢⱼ ≠ 0 only if i = j or (i,j) ∈ E
        
        2. L² encodes 2-hop structure:
           (L²)ᵢⱼ = Σₖ Lᵢₖ Lₖⱼ
           This is non-zero only if there's a 2-hop path i → k → j
        
        3. By induction, Lᵏ encodes k-hop structure:
           (Lᵏ)ᵢⱼ ≠ 0 only if d(i, j) ≤ k
        
        4. Polynomial g(L) = Σₖ θₖLᵏ is a linear combination:
           (g(L))ᵢⱼ = Σₖ θₖ(Lᵏ)ᵢⱼ
           
           This is zero if all (Lᵏ)ᵢⱼ = 0, which happens when d(i, j) > K.
    
    CONSEQUENCE:
        Spectral convolution with polynomial filter of degree K
        = K-hop local message passing!
        
    This transforms global spectral filtering (O(n²)) into
    local spatial convolution (O(Km)).
    
    KEY PAPERS:
        - Hammond et al., 2011: Wavelets on graphs via spectral graph theory
        - Defferrard et al., 2016: ChebNet (polynomial filters)
        - Kipf & Welling, 2017: GCN (K=1 simplification)
    """
