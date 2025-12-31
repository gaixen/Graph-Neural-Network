"""Graph Attention Network (GAT) operator with attention weights.

GAT (Veličković et al., 2018) introduced attention mechanisms to GNNs,
allowing the model to learn which neighbors are more important.

Mathematical formulation:
    hᵢ' = σ(Σⱼ∈𝐩(ᵢ) αᵢⱼ W hⱼ)

    where attention coefficients:
    αᵢⱼ = softmaxⱼ(eᵢⱼ)
    eᵢⱼ = LeakyReLU(aᵀ [Whᵢ || Whⱼ])

Key components:
    1. **Attention mechanism**: Learn importance weights αᵢⱼ
    2. **Multi-head attention**: Stabilize learning with K independent heads
    3. **Self-attention**: Attention is content-based (not fixed by structure)

Advantages over GCN:
    - Adaptive: Different neighbors get different weights
    - Interpretable: Attention weights show importance
    - No need for graph normalization: Attention is automatically normalized

Limitations:
    - Still limited by 1-WL expressivity (permutation-invariant aggregation)
    - Attention sparsity: softmax can over-concentrate
    - Computational cost: O(|E| × d²) vs GCN's O(|E| × d)

Invariance properties:
    - Attention: permutation-equivariant (depends on neighbor identity)
    - Aggregation: permutation-invariant (weighted sum)
    - Full layer: permutation-equivariant
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AttentionWeights:
    """Attention weights computed by GAT.

    Attributes:
        edge_weights: Attention coefficient for each edge (|E|,)
        edge_index: Corresponding edge indices (2, |E|)
        attention_scores: Raw scores before softmax (for analysis)
    """

    edge_weights: np.ndarray  # After softmax: Σⱼ αᵢⱼ = 1
    edge_index: np.ndarray
    attention_scores: np.ndarray  # Before softmax


def attention_coefficient(
    h_i: np.ndarray, h_j: np.ndarray, a: np.ndarray, negative_slope: float = 0.2
) -> float:
    """Compute attention score e_ij.

    Mathematical operation:
        eᵢⱼ = LeakyReLU(aᵀ [hᵢ || hⱼ])

    Where:
        - [hᵢ || hⱼ] = concatenation of features
        - a = learned attention vector
        - LeakyReLU(x) = max(αx, x) where α = negative_slope

    Design choices:
        - LeakyReLU (not ReLU): Allows negative attention
        - Concatenation (not add): Can distinguish source vs target
        - Linear projection: Simplicity (could use MLP)

    Args:
        h_i: Target node features
        h_j: Source node features
        a: Attention parameter vector (2d,)
        negative_slope: LeakyReLU slope for negative values

    Returns:
        Attention score (scalar)
    """
    # Concatenate features
    concat = np.concatenate([h_i, h_j])

    # Linear attention
    score = a @ concat

    # LeakyReLU
    if score < 0:
        score = negative_slope * score

    return float(score)


def attention(
    scores: np.ndarray, values: np.ndarray, return_weights: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute attention-weighted aggregation.

    Mathematical operation:
        output = Σᵢ αᵢ · vᵢ

        where α = softmax(scores)

    Softmax ensures:
        - Σᵢ αᵢ = 1 (probabilities)
        - αᵢ > 0 (non-negative weights)
        - Differentiable (gradient flows to scores)

    Numerical stability:
        softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
        Subtracting max prevents overflow.

    Args:
        scores: Attention scores (n,)
        values: Value vectors (n, d)
        return_weights: Whether to return attention weights

    Returns:
        (weighted_sum, attention_weights) if return_weights else weighted_sum
    """
    # Numerical stability: subtract max
    scores_stable = scores - np.max(scores)

    # Softmax
    exp_scores = np.exp(scores_stable)
    attention_weights = exp_scores / np.sum(exp_scores)

    # Weighted sum
    if values.ndim == 1:
        output = np.sum(attention_weights * values)
    else:
        output = np.sum(attention_weights[:, None] * values, axis=0)

    if return_weights:
        return output, attention_weights
    return output


def gat_layer(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    return_attention: bool = False,
    add_self_loops: bool = True,
) -> Tuple[np.ndarray, Optional[AttentionWeights]]:
    """Graph Attention Layer (single head).

    Mathematical operation:
        For each node i:
        1. Compute eᵢⱼ = attention_coefficient(Whᵢ, Whⱼ) for j ∈ N(i)
        2. Normalize: αᵢⱼ = softmaxⱼ(eᵢⱼ)
        3. Aggregate: hᵢ' = σ(Σⱼ αᵢⱼ Whⱼ)

    Differences from GCN:
        - GCN: Fixed weights (degree normalization)
        - GAT: Learned weights (attention)
        - GCN: Symmetric (undirected)
        - GAT: Asymmetric (αᵢⱼ ≠ αⱼᵢ in general)

    Args:
        node_features: Input features (num_nodes, in_dim)
        edge_index: Edges (2, num_edges)
        W: Feature transformation (out_dim, in_dim)
        a: Attention parameters (2 * out_dim,)
        return_attention: Whether to return attention weights
        add_self_loops: Whether to include self-connections

    Returns:
        (new_features, attention_weights) if return_attention else new_features
    """
    num_nodes = node_features.shape[0]
    out_dim = W.shape[0]

    # Transform features
    transformed = (W @ node_features.T).T  # (num_nodes, out_dim)

    # Add self-loops to edge_index if requested
    if add_self_loops:
        self_loops = np.array([[i, i] for i in range(num_nodes)]).T
        edge_index = np.concatenate([edge_index, self_loops], axis=1)

    # Build adjacency list
    adj_list = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        adj_list[tgt].append(src)

    # Compute attention for all edges
    all_scores = []
    all_edges = []

    for tgt in range(num_nodes):
        for src in adj_list[tgt]:
            score = attention_coefficient(transformed[tgt], transformed[src], a)
            all_scores.append(score)
            all_edges.append((src, tgt))

    # Apply attention for each node
    new_features = []
    attention_weights_list = []

    for i in range(num_nodes):
        neighbors = adj_list[i]

        if not neighbors:
            # Isolated node: just use transformed self features
            new_features.append(transformed[i])
            continue

        # Get scores for this node's edges
        neighbor_scores = []
        neighbor_features = []
        for j in neighbors:
            # Find edge (j, i)
            edge_idx = all_edges.index((j, i))
            neighbor_scores.append(all_scores[edge_idx])
            neighbor_features.append(transformed[j])

        # Attention-weighted aggregation
        neighbor_scores = np.array(neighbor_scores)
        neighbor_features = np.array(neighbor_features)

        h_new, attn_weights = attention(
            neighbor_scores, neighbor_features, return_weights=True
        )

        # Apply activation (ELU in original paper)
        h_new = np.maximum(0, h_new)  # ReLU for simplicity

        new_features.append(h_new)
        attention_weights_list.extend(attn_weights)

    new_features = np.array(new_features)

    if return_attention:
        attention_result = AttentionWeights(
            edge_weights=np.array(attention_weights_list),
            edge_index=edge_index,
            attention_scores=np.array(all_scores),
        )
        return new_features, attention_result

    return new_features


def multi_head_attention(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    W_heads: List[np.ndarray],
    a_heads: List[np.ndarray],
    concat: bool = True,
) -> np.ndarray:
    """Multi-head attention (GAT with K heads).

    Mathematical operation:
        If concat=True (intermediate layers):
        hᵢ' = ||ₖ₌₁ᴷ σ(Σⱼ αᵢⱼᵏ Wᵏhⱼ)

        If concat=False (final layer):
        hᵢ' = σ((1/K) Σₖ Σⱼ αᵢⱼᵏ Wᵏhⱼ)

    Rationale:
        - Multiple heads stabilize attention learning
        - Different heads can attend to different patterns
        - Analogous to multi-head attention in Transformers

    Args:
        node_features: Input features
        edge_index: Edges
        W_heads: Weight matrices for each head
        a_heads: Attention parameters for each head
        concat: Concatenate heads (True) or average (False)

    Returns:
        Multi-head output features
    """
    num_heads = len(W_heads)
    head_outputs = []

    for k in range(num_heads):
        h_k = gat_layer(node_features, edge_index, W_heads[k], a_heads[k])
        head_outputs.append(h_k)

    if concat:
        # Concatenate: [h¹ || h² || ... || hᴷ]
        return np.concatenate(head_outputs, axis=1)
    else:
        # Average: (h¹ + h² + ... + hᴷ) / K
        return np.mean(head_outputs, axis=0)


def analyze_attention_distribution(attention_weights: AttentionWeights) -> dict:
    """Analyze learned attention distribution.

    Interesting questions:
        - Is attention uniform or concentrated?
        - Which edges get high attention?
        - Does attention correlate with graph structure?

    Args:
        attention_weights: Computed attention weights

    Returns:
        Dictionary with attention statistics
    """
    weights = attention_weights.edge_weights

    # Entropy: measures uniformity
    # H = -Σ p log p
    # High entropy → uniform attention
    # Low entropy → concentrated attention
    entropy = -np.sum(weights * np.log(weights + 1e-10))
    max_entropy = np.log(len(weights))  # Uniform distribution
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "num_edges": len(weights),
    }


def visualize_attention(
    attention_weights: AttentionWeights, threshold: float = 0.1
) -> List[Tuple[int, int, float]]:
    """Extract high-attention edges for visualization.

    Args:
        attention_weights: Attention weights
        threshold: Minimum weight to include

    Returns:
        List of (source, target, weight) for edges above threshold
    """
    important_edges = []

    for e in range(len(attention_weights.edge_weights)):
        weight = attention_weights.edge_weights[e]
        if weight >= threshold:
            src = int(attention_weights.edge_index[0, e])
            tgt = int(attention_weights.edge_index[1, e])
            important_edges.append((src, tgt, float(weight)))

    # Sort by weight (descending)
    important_edges.sort(key=lambda x: x[2], reverse=True)

    return important_edges
