"""Spectral view of oversmoothing: repeated diffusion leads to representation collapse.

OVERSMOOTHING is the phenomenon where deep GNNs make all node representations identical.

Mathematical characterization:
    After k layers:
    h^(k) = (D^{-1/2} A D^{-1/2})^k h^(0)

    As k → ∞:
    h^(k) → constant vector (all nodes have same features)

Why this happens:
    1. Graph convolution is LOW-PASS FILTER
    2. Repeated application SMOOTHS signal
    3. Signal converges to EIGENSPACE of λ=1
    4. All high-frequency information LOST

THEOREM (Li et al., 2018):
    For connected graph G:
    lim_{k→∞} (D^{-1/2} A D^{-1/2})^k = (1/n) 11^T

    All nodes converge to average signal!

Consequences:
    - Deep GNNs lose discriminative power
    - Node representations become indistinguishable
    - Gradients vanish (all derivatives same)
    - Performance degrades with depth

This module provides:
    - Spectral analysis of oversmoothing
    - Fixed-point characterization
    - Metrics to measure smoothing
    - Conditions for collapse
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class OversmoothingAnalysis:
    """Results of oversmoothing analysis.

    Attributes:
        layer_variances: Variance of node features at each layer
        layer_similarities: Average cosine similarity at each layer
        spectral_energy: Energy in each eigenspace at each layer
        convergence_rate: Rate of convergence to fixed point
        mad_gap: Mean Average Distance between nodes
    """

    layer_variances: List[float]
    layer_similarities: List[float]
    spectral_energy: List[np.ndarray]
    convergence_rate: float
    mad_gap: List[float]


def analyze_oversmoothing(
    adjacency: np.ndarray, initial_features: np.ndarray, num_layers: int = 10
) -> OversmoothingAnalysis:
    """Analyze oversmoothing through spectral lens.

    Mathematical analysis:
        1. Compute graph Laplacian spectrum
        2. Simulate k-layer propagation
        3. Track signal energy in each eigenspace
        4. Measure convergence to fixed point

    Args:
        adjacency: Adjacency matrix A
        initial_features: Initial node features h^(0)
        num_layers: Number of layers to simulate

    Returns:
        Oversmoothing analysis results
    """
    n = adjacency.shape[0]

    # Compute normalized adjacency (GCN-style)
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt

    # Add self-loops
    A_norm = A_norm + np.eye(n)

    # Renormalize
    D2 = np.diag(np.sum(A_norm, axis=1))
    D2_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D2) + 1e-10))
    A_norm = D2_inv_sqrt @ A_norm @ D2_inv_sqrt

    # Eigendecomposition
    eigvals, eigvecs = la.eigh(A_norm)

    # Initialize tracking
    layer_variances = []
    layer_similarities = []
    spectral_energy = []
    mad_gap = []

    # Current features
    h = initial_features.copy()

    for layer in range(num_layers):
        # Measure variance
        var = np.var(h, axis=0).mean()
        layer_variances.append(var)

        # Measure pairwise similarity
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-10)
        sim_matrix = h_norm @ h_norm.T
        avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1) + 1e-10)
        layer_similarities.append(avg_sim)

        # Measure spectral energy
        energy = np.zeros(n)
        for i in range(n):
            projection = eigvecs[:, i].T @ h
            energy[i] = np.linalg.norm(projection) ** 2
        spectral_energy.append(energy / (np.sum(energy) + 1e-10))

        # Measure MAD (Mean Average Distance)
        mad = 0
        for i in range(n):
            for j in range(i + 1, n):
                mad += np.linalg.norm(h[i] - h[j])
        mad /= n * (n - 1) / 2
        mad_gap.append(mad)

        # Apply one layer of propagation
        h = A_norm @ h

    # Estimate convergence rate (largest eigenvalue < 1)
    # For convergence: λ_max < 1
    convergence_rate = np.max(np.abs(eigvals[:-1]))  # Exclude λ=1

    return OversmoothingAnalysis(
        layer_variances=layer_variances,
        layer_similarities=layer_similarities,
        spectral_energy=spectral_energy,
        convergence_rate=convergence_rate,
        mad_gap=mad_gap,
    )


def prove_oversmoothing_theorem() -> str:
    """Formal statement of oversmoothing convergence.

    Returns:
        Theorem statement
    """
    return """
    THEOREM (Oversmoothing, Li et al. 2018):
        For connected graph G with adjacency A:
        
        Let P = D^{-1/2} A D^{-1/2} (normalized adjacency)
        
        Then:
        lim_{k→∞} P^k = (1/n) d d^T
        
        where d = D^{1/2} 1 (degree-weighted vector)
    
    PROOF SKETCH:
        1. P is symmetric, eigenvalues in [-1, 1]
        2. For connected graph: λ_1 = 1, |λ_i| < 1 for i > 1
        3. Eigenvector for λ=1 is v_1 ∝ D^{1/2} 1
        4. Power iteration: P^k = ∑ λ_i^k v_i v_i^T
        5. As k → ∞: only λ=1 term survives
        6. Result: P^k → v_1 v_1^T = (1/n) d d^T
    
    IMPLICATION:
        h^(k) = P^k h^(0) → (1/n) d d^T h^(0)
        
        All nodes converge to SAME representation!
        (Weighted average of initial features)
    
    WHAT IS LOST:
        - All high-frequency components (small eigenvalues)
        - Local structure (neighborhoods)
        - Node-level discrimination
    """


def fixed_point_analysis(adjacency: np.ndarray) -> Tuple[np.ndarray, str]:
    """Analyze fixed point of repeated convolution.

    Args:
        adjacency: Graph adjacency matrix

    Returns:
        (fixed_point_operator, explanation)
    """
    n = adjacency.shape[0]

    # Normalized adjacency
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    P = D_inv_sqrt @ adjacency @ D_inv_sqrt

    # Add self-loops and renormalize
    P = P + np.eye(n)
    D2 = np.diag(np.sum(P, axis=1))
    D2_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D2) + 1e-10))
    P = D2_inv_sqrt @ P @ D2_inv_sqrt

    # Compute P^∞ (fixed point)
    eigvals, eigvecs = la.eigh(P)

    # Find eigenvalue closest to 1
    idx = np.argmax(eigvals)
    v1 = eigvecs[:, idx]

    # Fixed point operator
    P_inf = np.outer(v1, v1)

    explanation = f"""
    Fixed point P^∞:
        - All signals converge to eigenspace of λ=1
        - Projection operator: (v_1)(v_1^T)
        - Rank: 1 (extreme oversmoothing!)
        - Information capacity: log(1) = 0 bits
        
    Convergence rate:
        Determined by second-largest eigenvalue λ_2 = {eigvals[-2]:.4f}
        Faster convergence if λ_2 << 1
    """

    return P_inf, explanation


def dirichlet_energy_decay(
    adjacency: np.ndarray, signal: np.ndarray, num_layers: int = 10
) -> List[float]:
    """Track Dirichlet energy decay (smoothness metric).

    Dirichlet energy:
        E(f) = (1/2) ∑_{(i,j) ∈ E} (f_i - f_j)^2

    Low energy = smooth signal (neighbors have similar values)
    High energy = rough signal (neighbors differ)

    Oversmoothing ≡ E(f) → 0

    Args:
        adjacency: Graph adjacency
        signal: Node signal
        num_layers: Number of layers

    Returns:
        Energy at each layer
    """
    n = adjacency.shape[0]

    # Normalized adjacency
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    P = D_inv_sqrt @ adjacency @ D_inv_sqrt
    P = P + np.eye(n)
    D2 = np.diag(np.sum(P, axis=1))
    D2_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D2) + 1e-10))
    P = D2_inv_sqrt @ P @ D2_inv_sqrt

    energies = []
    f = signal.copy()

    for layer in range(num_layers):
        # Compute Dirichlet energy
        energy = 0
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] > 0:
                    energy += np.sum((f[i] - f[j]) ** 2)
        energy /= 2.0
        energies.append(energy)

        # Apply convolution
        f = P @ f

    return energies


def mitigation_strategies() -> Dict[str, str]:
    """Strategies to prevent oversmoothing.

    Returns:
        Strategy descriptions
    """
    return {
        "residual_connections": """
            h^(k+1) = σ(P h^(k)) + h^(k)
            
            Preserves high-frequency information
            Prevents complete collapse
            
            Tradeoff: May not capture long-range dependencies
        """,
        "adaptive_depth": """
            Stop adding layers when smoothing detected
            
            Monitor: variance, MAD, Dirichlet energy
            Stop when metric drops below threshold
            
            Tradeoff: Limits receptive field
        """,
        "initial_residual": """
            h^(k+1) = σ(P h^(k)) + α h^(0)
            
            Always connect to initial features
            Prevents information loss
            
            Tradeoff: May ignore intermediate layers
        """,
        "pairnorm": """
            After each layer:
            h_i = h_i - mean(h)
            h_i = h_i / sqrt(mean(||h_i||^2))
            
            Maintains total energy and centering
            Prevents convergence to constant
            
            Tradeoff: Changes operator properties
        """,
        "graph_rewiring": """
            Add/remove edges to improve spectral gap
            
            Goal: Increase λ_2 - λ_n (spectral gap)
            
            Tradeoff: Changes graph structure
        """,
    }
