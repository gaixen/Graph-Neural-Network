"""GraphSAGE operator: neighborhood sampling and aggregator forms.

GraphSAGE (Hamilton et al., 2017) introduced two key innovations:
1. **Neighborhood sampling**: Sample fixed-size neighborhoods (scalability)
2. **Multiple aggregators**: Mean, LSTM, pooling (expressivity)

Mathematical formulation:
    hᵢ⁽ᵗ⁾ = σ(W · CONCAT(hᵢ⁽ᵗ⁻¹⁾, AGG({hⱼ⁽ᵗ⁻¹⁾ : j ∈ 𝐩(i)})))

    where 𝐩(i) = sampled subset of N(i)

Key differences from GCN:
    - GCN: Normalized weighted average (implicit mean)
    - GraphSAGE: Explicit self + neighbor separation
    - GCN: Uses full neighborhood
    - GraphSAGE: Samples fixed-size neighborhood

Invariance/Equivariance:
    - Aggregation: permutation-invariant (over neighbors)
    - Full layer: permutation-equivariant (over nodes)
    - Sampling breaks determinism but preserves expected equivariance

Scalability insight:
    Sampling K neighbors per node → O(K^L) complexity for L layers
    vs. full neighborhood → O(d_max^L) where d_max can be huge
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import random


class AggregatorType(Enum):
    """GraphSAGE aggregator types."""

    MEAN = "mean"
    POOL = "pool"  # Max pooling after MLP
    LSTM = "lstm"  # Order-dependent (requires random permutation)
    GCN = "gcn"  # GCN-style normalized


@dataclass
class GraphSAGEConfig:
    """Configuration for GraphSAGE layer.

    Attributes:
        aggregator: Type of aggregation function
        sample_size: Number of neighbors to sample (None = use all)
        normalize: Whether to L2-normalize outputs
        concat_self: Whether to concatenate self features (True) or add (False)
    """

    aggregator: AggregatorType = AggregatorType.MEAN
    sample_size: Optional[int] = None
    normalize: bool = True
    concat_self: bool = True


def sample_neighbors(
    node: int,
    neighbors: List[int],
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Sample fixed-size neighborhood.

    Sampling strategies:
        - If |N(i)| > K: Uniform random sample of K neighbors
        - If |N(i)| ≤ K: Use all neighbors (possibly with replacement)

    Information preserved:
        - Approximate neighborhood structure

    Information lost:
        - Exact neighborhood composition
        - Determinism (sampling is stochastic)

    Variance reduction:
        Using fixed sample size reduces variance across batches.

    Args:
        node: Central node
        neighbors: All neighbors of node
        sample_size: Number to sample (None = use all)
        seed: Random seed for reproducibility

    Returns:
        Sampled neighbor list
    """
    if seed is not None:
        random.seed(seed)

    if sample_size is None or sample_size >= len(neighbors):
        # Use all neighbors
        return neighbors

    # Uniform sampling without replacement
    return random.sample(neighbors, sample_size)


def mean_aggregator(neighbor_features: List[np.ndarray]) -> np.ndarray:
    """Mean aggregator: average neighbor features.

    Mathematical operation:
        AGG({hⱼ}) = (1/|N|) Σⱼ hⱼ

    Properties:
        - Permutation invariant: ✓
        - Injective: ✗ (loses cardinality)
        - Size-normalized: ✓

    This is the simplest and most commonly used GraphSAGE aggregator.

    Args:
        neighbor_features: List of neighbor feature vectors

    Returns:
        Mean of neighbor features
    """
    if not neighbor_features:
        # No neighbors: return zero vector
        return np.zeros_like(neighbor_features[0]) if neighbor_features else np.zeros(1)

    return np.mean(neighbor_features, axis=0)


def pool_aggregator(
    neighbor_features: List[np.ndarray],
    mlp: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Pooling aggregator: max pooling after MLP transformation.

    Mathematical operation:
        AGG({hⱼ}) = maxⱼ σ(W_pool · hⱼ + b)

    Rationale:
        - MLP transforms each neighbor independently
        - Max pooling extracts most "salient" features

    Properties:
        - More expressive than raw max (MLP adds capacity)
        - Still not injective (max is lossy)

    Args:
        neighbor_features: List of neighbor vectors
        mlp: Optional transformation (default: identity)

    Returns:
        Max-pooled features after MLP
    """
    if not neighbor_features:
        return np.zeros(1)

    # Apply MLP to each neighbor
    if mlp is not None:
        transformed = [mlp(h) for h in neighbor_features]
    else:
        transformed = neighbor_features

    # Max pooling (element-wise)
    return np.max(transformed, axis=0)


def lstm_aggregator(
    neighbor_features: List[np.ndarray], lstm_cell: Optional[Callable] = None
) -> np.ndarray:
    """LSTM aggregator: process neighbors sequentially.

    Mathematical operation:
        - Randomly permute neighbors: π({hⱼ})
        - Run LSTM: h_out = LSTM(π({hⱼ}))

    WARNING:
        LSTM is NOT permutation-invariant!
        GraphSAGE paper randomly permutes neighbors to approximate invariance.

    Theoretical issue:
        Different permutations → different outputs
        Expected value over permutations should be invariant,
        but single forward pass is not.

    Why it still works:
        - Empirically performs well
        - Random permutation acts as data augmentation
        - Learning compensates for order sensitivity

    Args:
        neighbor_features: List of neighbor vectors
        lstm_cell: LSTM implementation (simplified here)

    Returns:
        LSTM final hidden state
    """
    if not neighbor_features:
        return np.zeros(1)

    # Randomly permute neighbors
    permuted = random.sample(neighbor_features, len(neighbor_features))

    # Simplified LSTM: just use mean (real version would use LSTM cell)
    # In practice, this would be a learned LSTM
    return np.mean(permuted, axis=0)


def graphsage_aggregate(
    neigh_feats: List[np.ndarray],
    self_feat: np.ndarray,
    aggregator: AggregatorType = AggregatorType.MEAN,
) -> np.ndarray:
    """GraphSAGE aggregation step.

    Args:
        neigh_feats: Neighbor features
        self_feat: Central node features
        aggregator: Aggregation method

    Returns:
        Aggregated neighbor features (without self)
    """
    if aggregator == AggregatorType.MEAN:
        return mean_aggregator(neigh_feats)
    elif aggregator == AggregatorType.POOL:
        return pool_aggregator(neigh_feats)
    elif aggregator == AggregatorType.LSTM:
        return lstm_aggregator(neigh_feats)
    elif aggregator == AggregatorType.GCN:
        # GCN-style: include self in mean
        all_feats = neigh_feats + [self_feat]
        return mean_aggregator(all_feats)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")


def graphsage_layer(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    W_self: np.ndarray,
    W_neigh: np.ndarray,
    config: GraphSAGEConfig = GraphSAGEConfig(),
) -> np.ndarray:
    """Full GraphSAGE layer.

    Mathematical operation:
        hᵢ' = σ(W_self · hᵢ + W_neigh · AGG({hⱼ : j ∈ 𝐩(i)}))

    Or with concatenation:
        hᵢ' = σ(W · [hᵢ || AGG({hⱼ : j ∈ 𝐩(i)})])

    Then optionally L2-normalize:
        hᵢ' ← hᵢ' / ||hᵢ'||

    Args:
        node_features: Node feature matrix (num_nodes, in_dim)
        edge_index: Edge index (2, num_edges)
        W_self: Weight matrix for self features
        W_neigh: Weight matrix for neighbor aggregation
        config: GraphSAGE configuration

    Returns:
        Updated node features (num_nodes, out_dim)
    """
    num_nodes = node_features.shape[0]

    # Build adjacency list
    adj_list = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        adj_list[tgt].append(src)

    # Aggregate for each node
    new_features = []
    for i in range(num_nodes):
        # Sample neighbors
        neighbors = adj_list[i]
        sampled_neighbors = sample_neighbors(i, neighbors, config.sample_size)

        # Get neighbor features
        neigh_feats = [node_features[j] for j in sampled_neighbors]

        # Aggregate
        agg_neigh = graphsage_aggregate(
            neigh_feats, node_features[i], config.aggregator
        )

        # Combine self and neighbor
        if config.concat_self:
            # Concatenate: [h_i || agg_neigh]
            combined = np.concatenate([node_features[i], agg_neigh])
            # Would need W with doubled input dimension
            h_new = W_self @ combined  # Simplified
        else:
            # Separate transformations then add
            h_new = W_self @ node_features[i] + W_neigh @ agg_neigh

        # Activation (ReLU)
        h_new = np.maximum(0, h_new)

        # Normalize
        if config.normalize:
            norm = np.linalg.norm(h_new)
            if norm > 0:
                h_new = h_new / norm

        new_features.append(h_new)

    return np.array(new_features)


def analyze_sampling_variance(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_idx: int,
    sample_size: int,
    num_trials: int = 100,
) -> dict:
    """Analyze variance introduced by neighborhood sampling.

    Sampling introduces stochasticity:
        Different samples → different outputs

    This function measures the variance.

    Args:
        node_features: Node features
        edge_index: Edges
        node_idx: Node to analyze
        sample_size: Sample size
        num_trials: Number of sampling trials

    Returns:
        Dictionary with variance statistics
    """
    # Build adjacency
    neighbors = []
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        if tgt == node_idx:
            neighbors.append(src)

    # Sample multiple times
    aggregated_outputs = []
    for trial in range(num_trials):
        sampled = sample_neighbors(node_idx, neighbors, sample_size, seed=trial)
        neigh_feats = [node_features[j] for j in sampled]
        agg = mean_aggregator(neigh_feats)
        aggregated_outputs.append(agg)

    # Compute statistics
    outputs_array = np.array(aggregated_outputs)
    mean_output = np.mean(outputs_array, axis=0)
    std_output = np.std(outputs_array, axis=0)

    return {
        "mean": mean_output,
        "std": std_output,
        "coefficient_of_variation": np.mean(std_output / (np.abs(mean_output) + 1e-10)),
        "num_neighbors": len(neighbors),
        "sample_size": sample_size,
    }
