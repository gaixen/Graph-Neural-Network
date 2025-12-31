"""Tradeoffs and limits when increasing depth: stability, gradients, expressivity.

DEPTH PARADOX in GNNs:
    - Deeper should = more expressive (larger receptive field)
    - But: deeper often = worse performance!

Why depth fails:
    1. **Oversmoothing**: Representations collapse
    2. **Oversquashing**: Information bottlenecks
    3. **Vanishing gradients**: Can't train deep networks
    4. **Numerical instability**: Repeated matrix multiplications
    5. **Overparametrization**: More layers = more overfitting

THEOREM (Empirical, Li et al. 2018):
    For most GNN architectures:
    Optimal depth k* = 2-4 layers

    Beyond k*, performance degrades!

This is OPPOSITE of CNNs:
    - CNNs: Deeper = better (ResNet-152, etc.)
    - GNNs: Deeper = worse (beyond 4 layers)

Why the difference?
    - CNNs: Local receptive fields (no global collapse)
    - GNNs: Global coupling (graph connectivity)
    - CNNs: Hierarchical features (composition)
    - GNNs: Flat structure (no clear hierarchy)

This module explores:
    - Gradient flow analysis
    - Numerical stability
    - Effective depth vs nominal depth
    - When depth helps vs hurts
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class DepthAnalysis:
    """Analysis of depth effects.

    Attributes:
        gradient_norms: Gradient norms at each layer
        effective_depth: Number of layers actually used
        receptive_fields: Receptive field size at each layer
        stability_metrics: Condition numbers, spectral norms
        layer_contributions: How much each layer contributes
    """

    gradient_norms: List[float]
    effective_depth: int
    receptive_fields: List[int]
    stability_metrics: Dict[str, List[float]]
    layer_contributions: List[float]


def analyze_depth_limits(
    adjacency: np.ndarray,
    initial_features: np.ndarray,
    max_depth: int = 20,
    target_gradient: Optional[np.ndarray] = None,
) -> DepthAnalysis:
    """Analyze how depth affects trainability and expressivity.

    Args:
        adjacency: Graph adjacency matrix
        initial_features: Initial node features
        max_depth: Maximum depth to analyze
        target_gradient: Target gradient for backprop simulation

    Returns:
        Depth analysis results
    """
    n = adjacency.shape[0]

    # Normalized adjacency
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt
    A_norm = A_norm + np.eye(n)
    D2 = np.diag(np.sum(A_norm, axis=1))
    D2_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D2) + 1e-10))
    A_norm = D2_inv_sqrt @ A_norm @ D2_inv_sqrt

    # Track metrics
    gradient_norms = []
    receptive_fields = []
    condition_numbers = []
    spectral_norms = []
    layer_contributions = []

    # Forward pass
    activations = [initial_features]
    h = initial_features.copy()

    for layer in range(max_depth):
        # Apply convolution
        h_prev = h
        h = A_norm @ h
        activations.append(h)

        # Measure contribution (how much changed)
        contribution = np.linalg.norm(h - h_prev) / (np.linalg.norm(h_prev) + 1e-10)
        layer_contributions.append(contribution)

        # Compute receptive field (diameter of k-hop neighborhood)
        # Approximate by matrix power
        A_k = np.linalg.matrix_power(adjacency > 0, layer + 1)
        avg_receptive = np.mean(np.sum(A_k > 0, axis=1))
        receptive_fields.append(int(avg_receptive))

        # Stability metrics
        condition_numbers.append(np.linalg.cond(A_norm))
        spectral_norms.append(np.linalg.norm(A_norm, 2))

    # Backward pass (simulate gradient flow)
    if target_gradient is not None:
        grad = target_gradient
        for layer in range(max_depth - 1, -1, -1):
            grad_norm = np.linalg.norm(grad)
            gradient_norms.append(grad_norm)

            # Backprop through convolution
            grad = A_norm.T @ grad
    else:
        gradient_norms = [1.0] * max_depth  # Placeholder

    gradient_norms = list(reversed(gradient_norms))

    # Compute effective depth (when gradients become negligible)
    effective_depth = max_depth
    for i, gn in enumerate(gradient_norms):
        if gn < 1e-5:  # Vanished
            effective_depth = i
            break

    return DepthAnalysis(
        gradient_norms=gradient_norms,
        effective_depth=effective_depth,
        receptive_fields=receptive_fields,
        stability_metrics={
            "condition_numbers": condition_numbers,
            "spectral_norms": spectral_norms,
        },
        layer_contributions=layer_contributions,
    )


def vanishing_gradient_analysis() -> str:
    """Explain vanishing gradients in deep GNNs.

    Returns:
        Explanation
    """
    return """
    VANISHING GRADIENTS IN GNNs:
    
    Forward pass:
        h^(k) = σ(A h^(k-1) W^(k))
    
    Backward pass:
        ∂L/∂h^(k-1) = (A^T ∘ σ') ∂L/∂h^(k)
    
    Chain rule over k layers:
        ∂L/∂h^(0) = ∏_{i=1}^k (A^T ∘ σ') ∂L/∂h^(k)
    
    PROBLEM:
        If ||A|| < 1:
        ||∂L/∂h^(0)|| ≤ ||A||^k ||∂L/∂h^(k)||
        
        Exponential decay!
    
    For GCN: A = D^{-1/2} Ã D^{-1/2}
        Eigenvalues in [-1, 1]
        Typically ||A|| ≈ 1 or slightly less
        
    CONSEQUENCE:
        Gradients vanish for k > 10-20 layers
        Early layers don't train!
    
    WHY DIFFERENT FROM CNNs?
        CNNs: Local convolution (bounded receptive field)
        GNNs: Global coupling (all nodes interact)
        
        CNNs: Hierarchical (features at each level)
        GNNs: Flat (no clear hierarchy)
    
    SOLUTIONS:
        - Residual connections
        - Layer normalization
        - Careful initialization
        - Skip connections
    """


def numerical_instability_analysis() -> str:
    """Explain numerical issues in deep GNNs.

    Returns:
        Explanation
    """
    return """
    NUMERICAL INSTABILITY:
    
    Repeated matrix multiplication:
        h^(k) = A^k h^(0)
    
    ISSUE 1: Accumulated rounding errors
        Each multiplication introduces error ε
        After k multiplications: error ~ kε
        
    ISSUE 2: Condition number growth
        cond(A^k) ≤ cond(A)^k
        
        High condition number = small perturbations amplified!
        
    ISSUE 3: Range collapse
        If λ_max < 1: signal decays
        If λ_max > 1: signal explodes
        
        Both cause numerical issues!
    
    PRACTICAL MANIFESTATION:
        - Features become NaN
        - Gradients explode/vanish
        - Training unstable
        - Sensitive to learning rate
    
    SOLUTIONS:
        - Normalize at each layer
        - Gradient clipping
        - Careful normalization of A
        - Mixed precision training
    """


def effective_vs_nominal_depth() -> str:
    """Explain why nominal depth ≠ effective depth.

    Returns:
        Explanation
    """
    return """
    EFFECTIVE DEPTH vs NOMINAL DEPTH:
    
    Nominal depth: Number of layers in network
    Effective depth: Number of layers actually used
    
    PHENOMENON:
        With residual connections:
        h^(k) = σ(A h^(k-1) W) + h^(k-1)
        
        Network may learn to SKIP layers!
        
    EXAMPLE:
        If W ≈ 0:
        h^(k) ≈ h^(k-1)
        
        Layer does nothing!
        
    MEASUREMENT:
        Effective depth = # layers with significant contribution
        
        Contribution = ||h^(k) - h^(k-1)|| / ||h^(k-1)||
        
        If contribution < threshold: layer skipped
    
    EMPIRICAL FINDING:
        Effective depth often much less than nominal
        
        Example:
        - Nominal: 32 layers
        - Effective: 4-5 layers
        
        Most layers just pass through!
    
    WHY THIS HAPPENS:
        - Oversmoothing forces early layers to be identity
        - Gradients too small to train deep layers
        - Network learns shallow function
    
    IMPLICATION:
        Adding more layers doesn't help!
        Better to use fewer, wider layers
    """


def when_does_depth_help() -> Dict[str, str]:
    """Scenarios where depth is beneficial.

    Returns:
        Scenarios and explanations
    """
    return {
        "sparse_graphs": """
            Large diameter graphs need depth
            
            Example: Long chains, trees
            Depth k = diameter needed
            
            But: Still limited by oversmoothing!
        """,
        "hierarchical_structure": """
            Graphs with clear hierarchy
            
            Example: Molecules, parse trees
            Each layer = one hierarchy level
            
            Works well up to hierarchy depth
        """,
        "with_residuals": """
            Residual connections prevent collapse
            
            h^(k) = σ(A h^(k-1) W) + h^(k-1)
            
            Can train deeper (8-16 layers)
            But diminishing returns
        """,
        "adaptive_depth": """
            Different nodes need different depth
            
            Central nodes: Shallow depth
            Peripheral nodes: Deep depth
            
            Solution: Per-node early stopping
        """,
    }


def depth_vs_width_tradeoff() -> str:
    """Compare depth vs width for GNNs.

    Returns:
        Comparison
    """
    return """
    DEPTH vs WIDTH TRADEOFF:
    
    DEPTH (more layers):
        Pros:
        - Larger receptive field
        - More non-linearity
        - Hierarchical features
        
        Cons:
        - Oversmoothing
        - Vanishing gradients
        - Numerical instability
        
    WIDTH (more hidden dimensions):
        Pros:
        - More capacity per layer
        - Better gradient flow
        - No oversmoothing
        
        Cons:
        - Limited receptive field
        - More parameters (overfitting)
        - Computational cost
    
    EMPIRICAL FINDING:
        For GNNs: Width usually better than depth!
        
        Example:
        - 4 layers × 256 dim > 16 layers × 64 dim
        
    OPTIMAL STRATEGY:
        Moderate depth (3-5 layers)
        Wide layers (128-512 dim)
        Residual connections
        Layer normalization
    
    EXCEPTION:
        Very large diameter graphs may need depth
        But use skip connections!
    """
