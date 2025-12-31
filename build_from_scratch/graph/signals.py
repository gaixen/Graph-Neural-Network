"""Node/edge/graph-level signals and transforms between them.

This module analyzes:
1. Representation of signals on graph domains (nodes, edges, graph-level)
2. What information is lost when converting to vector representation
3. Transformations between different signal domains
4. Pooling/readout functions

Mathematical framework:
    - Node signal: f: V → R^d  (function on vertices)
    - Edge signal: g: E → R^d  (function on edges)
    - Graph signal: h: G → R^d (global graph representation)

Information loss analysis:
    - Node ordering: Signals must be permutation-equivariant
    - Pooling: Irreversible aggregation from local to global
    - Edge-to-node: Which node gets the edge signal?

Key insight:
    All representations impose structure. The choice of representation
    determines what operations are natural and what information can be preserved.
"""

from typing import Iterable, Any, Callable, Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from enum import Enum


class SignalDomain(Enum):
    """Signal domain type for graph signals."""

    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


@dataclass
class NodeSignal:
    """Signal defined on graph nodes.

    Mathematical representation:
        f: V → R^d

    Information encoded:
        - Per-node features
        - Node attributes

    Information lost:
        - Node ordering (must use permutation-invariant representation)
        - Spatial position (unless explicitly encoded)

    Storage:
        Typically as matrix X ∈ R^{|V| × d} where row i = f(v_i).
        CRITICAL: Row ordering is arbitrary! Permuting rows yields
        equivalent representation.
    """

    node_ids: np.ndarray  # Shape: (n,)
    features: np.ndarray  # Shape: (n, d)

    def __post_init__(self):
        """Validate signal consistency."""
        if self.node_ids.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"Node count mismatch: {self.node_ids.shape[0]} IDs vs "
                f"{self.features.shape[0]} feature rows"
            )

    @property
    def num_nodes(self) -> int:
        """Number of nodes in signal domain."""
        return len(self.node_ids)

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality."""
        return self.features.shape[1] if self.features.ndim > 1 else 1

    def reorder(self, permutation: np.ndarray) -> "NodeSignal":
        """Apply permutation to node ordering.

        Mathematical property:
            This should not change the represented function f: V → R^d,
            only its storage representation.

        Args:
            permutation: Permutation array where permutation[i] = j means
                        new position i gets old position j

        Returns:
            Reordered signal (same function, different storage)
        """
        return NodeSignal(
            node_ids=self.node_ids[permutation], features=self.features[permutation]
        )


@dataclass
class EdgeSignal:
    """Signal defined on graph edges.

    Mathematical representation:
        g: E → R^d

    Information encoded:
        - Per-edge features
        - Edge attributes
        - Relational information

    Information lost:
        - Edge ordering (must be permutation-invariant)
        - Direction (if graph is undirected, g(u,v) = g(v,u) loses orientation)
    """

    edge_list: np.ndarray  # Shape: (m, 2) - pairs of node indices
    features: np.ndarray  # Shape: (m, d)
    directed: bool = False

    def __post_init__(self):
        """Validate signal consistency."""
        if self.edge_list.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"Edge count mismatch: {self.edge_list.shape[0]} edges vs "
                f"{self.features.shape[0]} feature rows"
            )

    @property
    def num_edges(self) -> int:
        """Number of edges in signal domain."""
        return len(self.edge_list)

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality."""
        return self.features.shape[1] if self.features.ndim > 1 else 1


@dataclass
class GraphSignal:
    """Signal for entire graph (graph-level representation).

    Mathematical representation:
        h: G → R^d

    Information encoded:
        - Global graph properties
        - Graph-level attributes

    Information lost:
        - All structural details (unless preserved by construction)
        - Cannot reconstruct nodes/edges from this alone

    This is the output of pooling/readout functions.
    """

    features: np.ndarray  # Shape: (d,) or (1, d)

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality."""
        return self.features.shape[-1] if self.features.ndim > 1 else len(self.features)


def lift_node_signal(
    nodes: Iterable[Any], signal: Iterable[float], feature_dim: int = 1
) -> NodeSignal:
    """Lift node identifiers and values to NodeSignal representation.

    Mathematical operation:
        Given nodes V and signal values {f(v)}, construct f: V → R^d

    Information encoded:
        - Node-to-feature mapping

    Information lost:
        - Original node ordering (we impose arbitrary ordering)
        - Any structure in the node identifiers themselves

    Args:
        nodes: Node identifiers (any hashable type)
        signal: Signal values (parallel to nodes)
        feature_dim: Feature dimensionality (reshape if needed)

    Returns:
        NodeSignal object
    """
    node_array = np.array(list(nodes))
    signal_array = np.array(list(signal))

    # Reshape to (n, d) if needed
    if signal_array.ndim == 1:
        signal_array = signal_array.reshape(-1, feature_dim)

    return NodeSignal(node_ids=node_array, features=signal_array)


def node_to_edge_signal(
    node_signal: NodeSignal, edge_list: np.ndarray, aggregation: str = "mean"
) -> EdgeSignal:
    """Transform node signal to edge signal.

    Mathematical operation:
        g(u, v) = AGG(f(u), f(v))

    Information encoded:
        - Pairwise node feature combinations

    Information lost:
        - Cannot distinguish f(u) from f(v) if aggregation is symmetric
        - Aggregation is lossy (e.g., mean of (1,9) and (5,5) both = 5)

    Common aggregations:
        - 'mean': g(u,v) = (f(u) + f(v)) / 2
        - 'sum': g(u,v) = f(u) + f(v)
        - 'concat': g(u,v) = [f(u); f(v)]  (preserves order)
        - 'diff': g(u,v) = f(u) - f(v)  (asymmetric, preserves direction)

    Args:
        node_signal: Signal on nodes
        edge_list: Edge list as (num_edges, 2) array
        aggregation: How to combine node features

    Returns:
        EdgeSignal object
    """
    num_edges = edge_list.shape[0]
    feature_dim = node_signal.feature_dim

    # Extract features for source and target nodes
    src_features = node_signal.features[edge_list[:, 0]]
    tgt_features = node_signal.features[edge_list[:, 1]]

    if aggregation == "mean":
        edge_features = (src_features + tgt_features) / 2
    elif aggregation == "sum":
        edge_features = src_features + tgt_features
    elif aggregation == "concat":
        edge_features = np.concatenate([src_features, tgt_features], axis=1)
    elif aggregation == "diff":
        edge_features = src_features - tgt_features
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return EdgeSignal(
        edge_list=edge_list,
        features=edge_features,
        directed=(aggregation in ["diff", "concat"]),
    )


def pool_sum(node_values: np.ndarray) -> float:
    """Sum pooling: aggregate node features to graph-level.

    Mathematical operation:
        h = Σ_{v ∈ V} f(v)

    Properties:
        - Permutation invariant: sum doesn't depend on node ordering
        - Size-dependent: larger graphs → larger sums

    Information encoded:
        - Total "amount" of feature

    Information lost:
        - Individual node values
        - Graph structure
        - Number of nodes (cannot distinguish n nodes with value x/n
          from 1 node with value x)

    Use case:
        When total quantity matters (e.g., total mass, total charge)

    Args:
        node_values: Node feature matrix (n, d)

    Returns:
        Graph-level feature vector (d,)
    """
    return np.sum(node_values, axis=0)


def pool_mean(node_values: np.ndarray) -> float:
    """Mean pooling: average node features to graph-level.

    Mathematical operation:
        h = (1/|V|) Σ_{v ∈ V} f(v)

    Properties:
        - Permutation invariant
        - Size-normalized: graphs of different sizes are comparable

    Information encoded:
        - Average feature value

    Information lost:
        - Individual node values
        - Graph structure
        - Number of nodes (unlike sum pooling)
        - Variance (100 nodes with value 5 same as 10 nodes with value 5)

    Use case:
        When average/typical value matters, not total quantity

    Args:
        node_values: Node feature matrix (n, d)

    Returns:
        Graph-level feature vector (d,)
    """
    return np.mean(node_values, axis=0)


def pool_max(node_values: np.ndarray) -> float:
    """Max pooling: take maximum node feature values.

    Mathematical operation:
        h = max_{v ∈ V} f(v)  (element-wise max)

    Properties:
        - Permutation invariant
        - Size-independent
        - Sparse: only one node contributes per dimension

    Information encoded:
        - Maximum feature values
        - Upper bound on node features

    Information lost:
        - All nodes except argmax
        - Graph structure
        - Number of nodes
        - Distribution of values (max of {1, 1, 10} same as max of {10, 10, 10})

    Use case:
        When peak/extreme values matter

    Limitation:
        Not differentiable at ties! Gradient undefined when multiple nodes
        achieve maximum.

    Args:
        node_values: Node feature matrix (n, d)

    Returns:
        Graph-level feature vector (d,)
    """
    return np.max(node_values, axis=0)


def readout_set2set(
    node_values: np.ndarray, num_iterations: int = 3, hidden_dim: Optional[int] = None
) -> np.ndarray:
    """Set2Set readout with attention mechanism.

    Mathematical formulation (Vinyals et al., 2016):
        This is an LSTM-based attention readout:

        q_t = LSTM(q_{t-1})
        a_t = softmax(node_values · q_t)
        r_t = Σ_i a_t^i · node_values_i

        Repeat for T steps, return final [q_T; r_T]

    Information encoded:
        - Attended combination of node features
        - More expressive than simple pooling

    Information lost:
        - Still cannot reconstruct individual nodes
        - Order-invariant (permutation of nodes gives same result)

    Advantage over mean/max/sum:
        - Learnable attention allows different importances
        - Multiple "reads" can extract richer information

    Note:
        This is a simplified version. Full implementation requires
        LSTM and learned parameters. Here we show the structure.

    Args:
        node_values: Node feature matrix (n, d)
        num_iterations: Number of attention iterations
        hidden_dim: Hidden dimensionality (defaults to feature dim)

    Returns:
        Graph-level feature vector
    """
    if hidden_dim is None:
        hidden_dim = node_values.shape[1]

    # Simplified version: just use learned weighted sum
    # (Real version would use LSTM state)

    # Initialize query (would be LSTM hidden state)
    query = np.mean(node_values, axis=0)

    for t in range(num_iterations):
        # Compute attention weights
        scores = node_values @ query  # (n,)
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention = exp_scores / np.sum(exp_scores)  # (n,)

        # Weighted sum
        readout = attention @ node_values  # (d,)

        # Update query (simplified: just use readout)
        query = readout

    return query


def pool_with_statistics(node_values: np.ndarray) -> np.ndarray:
    """Pooling that preserves multiple statistics.

    Mathematical operation:
        h = [mean(f), std(f), min(f), max(f), ...]

    Intuition:
        More statistics → more information preserved
        but still cannot reconstruct original values

    Information encoded:
        - Mean, variance, extrema
        - Basic distribution shape

    Information lost:
        - Individual values
        - Exact distribution (many distributions share same statistics)
        - Graph structure

    Advantage:
        - Richer than single pooling
        - Still permutation-invariant

    Args:
        node_values: Node feature matrix (n, d)

    Returns:
        Graph-level feature vector with multiple statistics
    """
    return np.concatenate(
        [
            np.mean(node_values, axis=0),
            np.std(node_values, axis=0),
            np.min(node_values, axis=0),
            np.max(node_values, axis=0),
        ]
    )
