"""Graph Convolutional Network (GCN) Layer Implementation.

This module implements the GCN layer from Kipf & Welling (2017) and analyzes:
1. Exact operator applied
2. Permutation equivariance property
3. Why and when oversmoothing occurs
4. Baked-in assumptions

GCN formula:
    H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})

where:
    Ã = A + I  (adjacency + self-loops)
    D̃ = degree matrix of Ã
    W^{(l)} = learnable weights
    σ = activation function

Mathematical insight:
    This is the 1st-order approximation of spectral graph convolutions,
    simplified to local neighborhood averaging with learned weights.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import warnings


@dataclass
class GCNConfig:
    """Configuration for GCN layer.
    
    Attributes:
        add_self_loops: Add self-connections (I) to adjacency
        normalize: Apply symmetric normalization D^{-1/2}AD^{-1/2}
        activation: Activation function (None for linear)
        dropout_rate: Dropout probability (0.0 = no dropout)
    """
    add_self_loops: bool = True
    normalize: bool = True
    activation: Optional[Callable] = None
    dropout_rate: float = 0.0


def preprocess_adjacency(
    A: np.ndarray,
    add_self_loops: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """Preprocess adjacency matrix for GCN propagation.
    
    Steps:
        1. Optionally add self-loops: Ã = A + I
        2. Optionally normalize: D̃^{-1/2} Ã D̃^{-1/2}
    
    Parameters:
        A: Original adjacency matrix
        add_self_loops: Add identity matrix
        normalize: Apply symmetric normalization
    
    Returns:
        Preprocessed adjacency Â
    
    Properties of output:
        - If normalized: eigenvalues in [-1, 1]
        - If self-loops added: no zero-degree nodes
        - Symmetric (if input is undirected)
    
    Why self-loops:
        Without self-loops, node features are replaced by neighbor average,
        losing the node's own information.
        With self-loops, the node contributes to its own aggregation.
    
    Why normalization:
        Prevents high-degree nodes from dominating the aggregation.
        Ensures stable gradient flow and prevents exploding values.
    """
    if add_self_loops:
        A_tilde = A + np.eye(len(A))
    else:
        A_tilde = A
    
    if not normalize:
        return A_tilde
    
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degrees = np.sum(A_tilde, axis=1)
    
    # Handle isolated nodes
    if np.any(degrees == 0):
        warnings.warn("Found isolated nodes (degree 0) in graph. "
                     "Setting normalization to 0 for these nodes.")
    
    D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d > 0 else 0.0 for d in degrees])
    A_normalized = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    return A_normalized


def gcn_propagate(
    A_norm: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    activation: Optional[Callable] = None
) -> np.ndarray:
    """Apply one GCN propagation step.
    
    Formula:
        H' = σ(Â H W)
    
    where Â = D̃^{-1/2}(A+I)D̃^{-1/2} is preprocessed adjacency.
    
    Parameters:
        A_norm: Preprocessed adjacency matrix Â
        H: Node feature matrix (n × d_in)
        W: Weight matrix (d_in × d_out)
        activation: Activation function σ (e.g., ReLU, tanh)
    
    Returns:
        Updated features H' (n × d_out)
    
    Decomposition:
        1. Transform: H W (apply weights)
        2. Aggregate: Â (H W) (neighborhood averaging)
        3. Activate: σ(...)
    
    Permutation equivariance:
        For permutation π:
            gcn_propagate(Π A Π^T, Π H, W) = Π · gcn_propagate(A, H, W)
        
        Proof:
            LHS = σ(Π Â Π^T · Π H · W)
                = σ(Π (Â H W))  [since Π^T Π = I]
                = Π σ(Â H W)  [activation is element-wise]
                = RHS
    """
    # Linear transformation
    H_transformed = H @ W
    
    # Neighborhood aggregation
    H_aggregated = A_norm @ H_transformed
    
    # Activation
    if activation is not None:
        return activation(H_aggregated)
    
    return H_aggregated


def gcn_forward(
    A: np.ndarray,
    X: np.ndarray,
    weights: list[np.ndarray],
    config: GCNConfig = GCNConfig()
) -> np.ndarray:
    """Multi-layer GCN forward pass.
    
    Applies L layers:
        H^{(0)} = X
        H^{(l+1)} = σ(Â H^{(l)} W^{(l)})  for l = 0, ..., L-1
    
    Parameters:
        A: Adjacency matrix
        X: Input node features (n × d_in)
        weights: List of weight matrices [W^{(0)}, ..., W^{(L-1)}]
        config: GCN configuration
    
    Returns:
        Final layer features H^{(L)}
    
    Oversmoothing analysis:
        As L → ∞, H^{(L)} converges to rank-1 matrix (all rows identical).
        
        Intuition:
            Each layer averages with neighbors.
            Repeated averaging diffuses information globally.
            Eventually, all nodes have the same (graph-average) features.
        
        Mathematical:
            Â has largest eigenvalue ≈ 1 with eigenvector v ∝ D^{1/2} 1.
            Power iteration: Â^L converges to projection onto span{v}.
            Thus: H^{(L)} → (v v^T / ||v||²) H^{(0)} (all rows parallel to v)
    
    Empirical observation (Li et al., 2018):
        Performance degrades for L > 3-4 layers on many benchmarks.
        This is structural (oversmoothing), not an optimization issue.
    """
    # Preprocess adjacency once
    A_norm = preprocess_adjacency(
        A,
        add_self_loops=config.add_self_loops,
        normalize=config.normalize
    )
    
    H = X
    
    for layer_idx, W in enumerate(weights):
        # Propagate
        H = gcn_propagate(
            A_norm,
            H,
            W,
            activation=config.activation if layer_idx < len(weights) - 1 else None
        )
        
        # Optional dropout (not implemented here, would need training mode)
        # In practice: H = dropout(H, config.dropout_rate)
    
    return H


def analyze_oversmoothing(
    A: np.ndarray,
    num_layers: int = 10,
    add_self_loops: bool = True
) -> dict:
    """Analyze how features collapse with increasing GCN layers.
    
    Computes powers of normalized adjacency Â^k for k=1..num_layers
    and analyzes their spectral properties.
    
    Parameters:
        A: Adjacency matrix
        num_layers: Number of layers to simulate
        add_self_loops: Add self-loops to A
    
    Returns:
        Dictionary with analysis results:
        - eigenvalues: Spectrum of Â
        - power_norms: ||Â^k|| for k=1..num_layers
        - effective_ranks: Rank(Â^k) for k=1..num_layers
        - oversmoothing_rate: Exponential decay rate
    
    Interpretation:
        - If power_norms decay → oversmoothing occurs
        - If effective_ranks → 1 → severe collapse
        - Decay rate ≈ second-largest eigenvalue |λ_2|
    
    Recommendations:
        - Use residual connections: H^{(l+1)} = H^{(l)} + gcn(H^{(l)})
        - Limit depth: L ≤ 3 for most graphs
        - Add skip connections or attention
        - Use normalization layers (BatchNorm, LayerNorm)
    """
    A_norm = preprocess_adjacency(A, add_self_loops, normalize=True)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(A_norm)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    # Analyze powers
    power_norms = []
    effective_ranks = []
    
    A_power = np.eye(len(A))
    
    for k in range(1, num_layers + 1):
        A_power = A_power @ A_norm
        
        # Matrix norm
        norm = np.linalg.norm(A_power, ord=2)
        power_norms.append(norm)
        
        # Effective rank (entropy of singular values)
        singular_values = np.linalg.svd(A_power, compute_uv=False)
        singular_values = singular_values[singular_values > 1e-10]
        if len(singular_values) > 0:
            probs = singular_values / singular_values.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            eff_rank = np.exp(entropy)
        else:
            eff_rank = 0.0
        
        effective_ranks.append(eff_rank)
    
    # Estimate decay rate from spectral gap
    spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    
    return {
        "eigenvalues": eigenvalues,
        "spectral_gap": float(spectral_gap),
        "power_norms": power_norms,
        "effective_ranks": effective_ranks,
        "largest_eigenvalue": float(eigenvalues[0]),
        "second_largest_eigenvalue": float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        "theoretical_decay_rate": float(abs(eigenvalues[1])) if len(eigenvalues) > 1 else 1.0
    }


def verify_equivariance(
    A: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    permutation: np.ndarray,
    tol: float = 1e-6
) -> bool:
    """Verify that GCN layer is permutation-equivariant.
    
    Tests:
        gcn(Π A Π^T, Π H, W) ?= Π gcn(A, H, W)
    
    for a given permutation Π.
    
    Parameters:
        A: Adjacency matrix
        H: Features
        W: Weights
        permutation: Permutation array (permutation[i] = new index of node i)
        tol: Numerical tolerance
    
    Returns:
        True if equivariant (within tolerance)
    
    This is a fundamental property: GCNs respect graph symmetries.
    """
    # Permutation matrix
    n = len(A)
    P = np.zeros((n, n))
    P[np.arange(n), permutation] = 1
    
    # Left side: apply GCN to permuted inputs
    A_perm = P @ A @ P.T
    H_perm = P @ H
    
    A_perm_norm = preprocess_adjacency(A_perm, add_self_loops=True, normalize=True)
    output_left = gcn_propagate(A_perm_norm, H_perm, W)
    
    # Right side: permute GCN output
    A_norm = preprocess_adjacency(A, add_self_loops=True, normalize=True)
    output_right_unperm = gcn_propagate(A_norm, H, W)
    output_right = P @ output_right_unperm
    
    # Check equality
    return np.allclose(output_left, output_right, atol=tol)
