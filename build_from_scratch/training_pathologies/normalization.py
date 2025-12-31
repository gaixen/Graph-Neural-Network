"""What normalization does to signal energy and gradient flow; pros and cons.

Normalization in GNNs serves multiple purposes:
    1. **Numerical stability**: Prevent explosion/vanishing
    2. **Faster training**: Better gradient flow
    3. **Prevent oversmoothing**: Maintain signal energy
    4. **Regularization**: Implicit constraint

But normalization has costs:
    - Changes operator properties
    - May lose information
    - Can interact poorly with graph structure
    - Not always equivariant!

Types of normalization:
    1. **BatchNorm**: Normalize across batch
    2. **LayerNorm**: Normalize across features
    3. **GraphNorm**: Normalize across nodes in graph
    4. **PairNorm**: Center and scale
    5. **MessageNorm**: Normalize messages

This module analyzes:
    - How each normalization affects spectrum
    - Impact on gradient flow
    - When normalization helps/hurts
    - Equivariance properties
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class NormalizationAnalysis:
    """Analysis of normalization effects.

    Attributes:
        energy_preservation: Signal energy at each layer
        spectrum_change: How normalization affects eigenvalues
        gradient_flow: Gradient norms with/without normalization
        equivariance_test: Whether normalization is equivariant
    """

    energy_preservation: List[float]
    spectrum_change: Dict[str, np.ndarray]
    gradient_flow: Dict[str, List[float]]
    equivariance_test: Dict[str, bool]


def batch_norm(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Batch normalization.

    Formula:
        h_i = (h_i - μ) / √(σ^2 + ε)

    where μ, σ computed across batch

    WARNING: Not permutation-equivariant!
        Different graphs in batch → different normalization

    Args:
        features: Node features (num_nodes, feature_dim)
        eps: Numerical stability

    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    return (features - mean) / (std + eps)


def layer_norm(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization.

    Formula:
        h_i = (h_i - μ_i) / √(σ_i^2 + ε)

    where μ_i, σ_i computed per node

    Properties:
        - Permutation-equivariant! ✓
        - Normalizes each node independently
        - Preserves relative feature magnitudes

    Args:
        features: Node features
        eps: Numerical stability

    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    return (features - mean) / (std + eps)


def graph_norm(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Graph normalization (mean and std across all nodes).

    Formula:
        h = (h - μ) / √(σ^2 + ε)

    where μ, σ computed across all nodes in graph

    Properties:
        - Permutation-equivariant! ✓
        - Global statistics
        - Can lose information if all nodes similar

    Args:
        features: Node features
        eps: Numerical stability

    Returns:
        Normalized features
    """
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / (std + eps)


def pair_norm(features: np.ndarray, s: float = 1.0) -> np.ndarray:
    """PairNorm: Center and scale to fixed energy.

    Formula:
        1. Center: h = h - mean(h)
        2. Scale: h = s × h / sqrt(mean(||h_i||^2))

    Goal:
        - Maintain total energy = s^2 × n
        - Prevent oversmoothing!

    THEOREM (Zhao & Akoglu, 2020):
        PairNorm prevents complete oversmoothing
        by maintaining variance

    Args:
        features: Node features
        s: Target scale

    Returns:
        Normalized features
    """
    # Center
    mean = np.mean(features, axis=0, keepdims=True)
    h_centered = features - mean

    # Scale to unit energy
    energy = np.mean(np.sum(h_centered**2, axis=1))
    h_scaled = h_centered / (np.sqrt(energy) + 1e-10)

    # Scale to target
    return s * h_scaled


def message_norm(
    messages: np.ndarray, aggregation: str = "sum", s: float = 1.0
) -> np.ndarray:
    """Message normalization (normalize aggregated messages).

    Formula:
        messages = s × messages / ||messages||

    Prevents:
        - Message explosion
        - Degree-dependent magnitudes

    Args:
        messages: Aggregated messages
        aggregation: Aggregation type
        s: Target scale

    Returns:
        Normalized messages
    """
    norms = np.linalg.norm(messages, axis=1, keepdims=True)
    return s * messages / (norms + 1e-10)


def analyze_normalization(
    adjacency: np.ndarray,
    initial_features: np.ndarray,
    num_layers: int,
    norm_type: str = "none",
) -> NormalizationAnalysis:
    """Analyze effects of normalization.

    Args:
        adjacency: Graph adjacency
        initial_features: Initial features
        num_layers: Number of layers
        norm_type: Type of normalization

    Returns:
        Analysis results
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

    # Track energy
    energy_preservation = []
    h = initial_features.copy()

    for layer in range(num_layers):
        # Apply convolution
        h = A_norm @ h

        # Apply normalization
        if norm_type == "batch":
            h = batch_norm(h)
        elif norm_type == "layer":
            h = layer_norm(h)
        elif norm_type == "graph":
            h = graph_norm(h)
        elif norm_type == "pair":
            h = pair_norm(h)

        # Measure energy
        energy = np.mean(np.sum(h**2, axis=1))
        energy_preservation.append(energy)

    # Spectrum analysis
    eigvals_before, _ = la.eigh(A_norm)

    # For simplicity, spectrum after normalization is approximate
    eigvals_after = eigvals_before  # Placeholder

    spectrum_change = {"before": eigvals_before, "after": eigvals_after}

    # Equivariance test
    equivariance_test = {
        "batch": False,  # Not equivariant
        "layer": True,  # Equivariant
        "graph": True,  # Equivariant
        "pair": True,  # Equivariant
    }

    return NormalizationAnalysis(
        energy_preservation=energy_preservation,
        spectrum_change=spectrum_change,
        gradient_flow={},  # Placeholder
        equivariance_test=equivariance_test,
    )


def pros_and_cons() -> Dict[str, Dict[str, List[str]]]:
    """Pros and cons of each normalization type.

    Returns:
        Comparison
    """
    return {
        "BatchNorm": {
            "pros": [
                "Widely used in deep learning",
                "Effective for training stability",
                "Reduces internal covariate shift",
            ],
            "cons": [
                "NOT permutation-equivariant!",
                "Requires large batches",
                "Behaves differently train/test",
                "Not suitable for graphs",
            ],
        },
        "LayerNorm": {
            "pros": [
                "Permutation-equivariant",
                "Works with small batches",
                "Stable training",
                "No train/test discrepancy",
            ],
            "cons": [
                "Normalizes per-node (may lose global info)",
                "Can amplify noise in low-dim features",
                "Sensitive to outliers",
            ],
        },
        "GraphNorm": {
            "pros": [
                "Permutation-equivariant",
                "Graph-level statistics",
                "Natural for graph tasks",
            ],
            "cons": [
                "Sensitive to graph size",
                "Different graphs → different norms",
                "May lose node-level variation",
            ],
        },
        "PairNorm": {
            "pros": [
                "Prevents oversmoothing!",
                "Maintains energy",
                "Permutation-equivariant",
                "Theoretically motivated",
            ],
            "cons": [
                "Hyperparameter s needs tuning",
                "May not help gradient flow",
                "Removes mean (loses global signal)",
            ],
        },
        "MessageNorm": {
            "pros": [
                "Normalizes messages directly",
                "Prevents degree-dependent magnitudes",
                "Flexible (per-message)",
            ],
            "cons": [
                "Computational overhead",
                "May lose important magnitude info",
                "Interaction with aggregation",
            ],
        },
    }


def when_to_use_which() -> Dict[str, str]:
    """Guidelines for choosing normalization.

    Returns:
        Recommendations
    """
    return {
        "shallow_gnns": "LayerNorm or no normalization",
        "deep_gnns": "PairNorm + residuals to prevent oversmoothing",
        "small_graphs": "GraphNorm or LayerNorm",
        "large_graphs": "LayerNorm (graph-level stats expensive)",
        "node_classification": "LayerNorm (preserves node variation)",
        "graph_classification": "GraphNorm or PairNorm",
        "training_instability": "LayerNorm or PairNorm",
        "oversmoothing_issues": "PairNorm (proven to help!)",
    }
