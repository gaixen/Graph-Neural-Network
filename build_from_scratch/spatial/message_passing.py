"""Generic Message Passing Neural Network (MPNN) framework.

This module implements the unified MPNN framework (Gilmer et al., 2017),
which encompasses most spatial GNN architectures.

Mathematical formulation:
    hᵢ⁽ᵗ⁾ = UPDATEᵗ(hᵢ⁽ᵗ⁻¹⁾, AGGᵗ({MESSAGEᵗ(hᵢ⁽ᵗ⁻¹⁾, hⱼ⁽ᵗ⁻¹⁾) : j ∈ N(i)}))

Three-stage computation:
    1. MESSAGE: Compute messages from neighbors
       mᵢⱼ⁽ᵗ⁾ = MESSAGEᵗ(hᵢ⁽ᵗ⁻¹⁾, hⱼ⁽ᵗ⁻¹⁾, eᵢⱼ)

    2. AGGREGATE: Combine messages (permutation-invariant!)
       mᵢ⁽ᵗ⁾ = AGGᵗ({mᵢⱼ⁽ᵗ⁾ : j ∈ N(i)})

    3. UPDATE: Compute new node state
       hᵢ⁽ᵗ⁾ = UPDATEᵗ(hᵢ⁽ᵗ⁻¹⁾, mᵢ⁽ᵗ⁾)

Key insight:
    Different GNN architectures = different choices of MESSAGE/AGGREGATE/UPDATE

    - GCN: MESSAGE(hᵢ, hⱼ) = hⱼ/√(dᵢdⱼ), AGG = SUM
    - GraphSAGE: MESSAGE(hᵢ, hⱼ) = hⱼ, AGG = MEAN/MAX
    - GAT: MESSAGE(hᵢ, hⱼ) = αᵢⱼ hⱼ, AGG = SUM (attention-weighted)
    - GIN: MESSAGE(hᵢ, hⱼ) = hⱼ, AGG = SUM, UPDATE = MLP((1+ε)hᵢ + mᵢ)

Expressivity bound (Xu et al., 2019):
    MPNNs with sum aggregation are at most as powerful as 1-WL test.
"""

from typing import Callable, Dict, Any, Iterable, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


class MessageFunction(ABC):
    """Abstract base class for message functions.

    Mathematical signature:
        MESSAGE: (hᵢ, hⱼ, eᵢⱼ) → mᵢⱼ ∈ ℝᵈ
    """

    @abstractmethod
    def __call__(
        self, h_i: np.ndarray, h_j: np.ndarray, edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute message from node j to node i.

        Args:
            h_i: Node i features (receiver)
            h_j: Node j features (sender)
            edge_attr: Optional edge attributes

        Returns:
            Message vector m_ij
        """
        pass


class AggregateFunction(ABC):
    """Abstract base class for aggregation functions.

    Mathematical signature:
        AGGREGATE: ℘(ℝᵈ) → ℝᵈ

    CRITICAL property:
        Must be permutation-invariant!
        AGG({m₁, m₂, ..., mₙ}) = AGG(π({m₁, m₂, ..., mₙ})) for any permutation π
    """

    @abstractmethod
    def __call__(self, messages: List[np.ndarray]) -> np.ndarray:
        """Aggregate list of messages.

        Args:
            messages: List of message vectors (possibly empty!)

        Returns:
            Aggregated message
        """
        pass

    @property
    @abstractmethod
    def is_injective(self) -> bool:
        """Whether aggregation is injective (preserves all information).

        Theorem (Xu et al., 2019):
            MPNN is maximally expressive ⟺ AGG is injective
            SUM is injective over countable multisets
            MEAN, MAX are NOT injective
        """
        pass


class UpdateFunction(ABC):
    """Abstract base class for update functions.

    Mathematical signature:
        UPDATE: (ℝᵈ, ℝᵈ) → ℝᵈ'
    """

    @abstractmethod
    def __call__(self, h_old: np.ndarray, m_agg: np.ndarray) -> np.ndarray:
        """Compute updated node features.

        Args:
            h_old: Previous node features
            m_agg: Aggregated messages from neighbors

        Returns:
            Updated node features
        """
        pass


@dataclass
class MPNNResult:
    """Result of one MPNN step.

    Attributes:
        new_features: Updated node features
        messages: Messages computed (for analysis)
        aggregated: Aggregated messages per node
    """

    new_features: Dict[Any, np.ndarray]
    messages: Dict[Tuple[Any, Any], np.ndarray]
    aggregated: Dict[Any, np.ndarray]


def mp_step(
    nodes: Iterable[Any],
    edges: Iterable[Tuple[Any, Any]],
    node_features: Dict[Any, np.ndarray],
    message_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    aggregate_fn: Callable[[List[np.ndarray]], np.ndarray],
    update_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    edge_features: Optional[Dict[Tuple[Any, Any], np.ndarray]] = None,
) -> MPNNResult:
    """Execute one step of message passing.

    Mathematical operation:
        For each node i:
        1. Collect messages: Mᵢ = {MESSAGE(hᵢ, hⱼ, eᵢⱼ) : j ∈ N(i)}
        2. Aggregate: mᵢ = AGG(Mᵢ)
        3. Update: hᵢ' = UPDATE(hᵢ, mᵢ)

    Information flow:
        Each node i receives information from its 1-hop neighborhood N(i).
        K layers → K-hop receptive field.

    Equivariance property:
        If we permute node labels, output permutes accordingly:
        MP(π(G), π(H)) = π(MP(G, H))

    Args:
        nodes: Node identifiers
        edges: Edge list (directed or undirected)
        node_features: Current node features
        message_fn: MESSAGE function
        aggregate_fn: AGGREGATE function (must be permutation-invariant!)
        update_fn: UPDATE function
        edge_features: Optional edge attributes

    Returns:
        MPNNResult with updated features and intermediate computations
    """
    # Step 1: Compute messages
    messages: Dict[Any, List[np.ndarray]] = {v: [] for v in nodes}
    all_messages: Dict[Tuple[Any, Any], np.ndarray] = {}

    for u, v in edges:
        # Message from u to v
        edge_attr = edge_features.get((u, v)) if edge_features else None

        if edge_attr is not None:
            m = message_fn(node_features[v], node_features[u], edge_attr)
        else:
            m = message_fn(node_features[v], node_features[u])

        messages[v].append(m)
        all_messages[(u, v)] = m

    # Step 2: Aggregate messages
    aggregated = {}
    for v in nodes:
        aggregated[v] = aggregate_fn(messages[v])

    # Step 3: Update node features
    new_features = {}
    for v in nodes:
        new_features[v] = update_fn(node_features[v], aggregated[v])

    return MPNNResult(
        new_features=new_features, messages=all_messages, aggregated=aggregated
    )


def mp_layer(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    message_fn: MessageFunction,
    aggregate_fn: AggregateFunction,
    update_fn: UpdateFunction,
    edge_attr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized MPNN layer (more efficient than mp_step).

    Args:
        node_features: Node feature matrix (num_nodes, feature_dim)
        edge_index: Edge index (2, num_edges) where edge_index[0] = sources, edge_index[1] = targets
        message_fn: Message function
        aggregate_fn: Aggregation function
        update_fn: Update function
        edge_attr: Optional edge attributes (num_edges, edge_dim)

    Returns:
        Updated node features (num_nodes, new_feature_dim)
    """
    num_nodes = node_features.shape[0]
    num_edges = edge_index.shape[1]

    # Step 1: Compute all messages
    messages = []
    for e in range(num_edges):
        src, tgt = edge_index[0, e], edge_index[1, e]
        h_i = node_features[tgt]
        h_j = node_features[src]

        if edge_attr is not None:
            m = message_fn(h_i, h_j, edge_attr[e])
        else:
            m = message_fn(h_i, h_j)

        messages.append(m)

    # Step 2: Aggregate messages per node
    aggregated = []
    for i in range(num_nodes):
        # Find messages for node i
        incoming_mask = edge_index[1] == i
        incoming_messages = [messages[e] for e in range(num_edges) if incoming_mask[e]]

        agg = aggregate_fn(incoming_messages)
        aggregated.append(agg)

    aggregated = np.array(aggregated)

    # Step 3: Update
    new_features = np.array(
        [update_fn(node_features[i], aggregated[i]) for i in range(num_nodes)]
    )

    return new_features


def verify_permutation_equivariance(
    nodes: List[Any],
    edges: List[Tuple[Any, Any]],
    node_features: Dict[Any, np.ndarray],
    message_fn: Callable,
    aggregate_fn: Callable,
    update_fn: Callable,
    num_trials: int = 10,
) -> bool:
    """Verify that MPNN is permutation-equivariant.

    Theorem:
        If AGG is permutation-invariant, then MPNN is permutation-equivariant:

        MP(π(G), π(features)) = π(MP(G, features))

    This function tests this empirically.

    Args:
        nodes, edges, node_features: Graph and features
        message_fn, aggregate_fn, update_fn: MPNN functions
        num_trials: Number of random permutations to test

    Returns:
        True if equivariance holds (within numerical tolerance)
    """
    import random

    # Run MPNN on original graph
    result_orig = mp_step(
        nodes, edges, node_features, message_fn, aggregate_fn, update_fn
    )

    for _ in range(num_trials):
        # Generate random permutation
        node_list = list(nodes)
        perm = list(node_list)
        random.shuffle(perm)

        perm_map = {node_list[i]: perm[i] for i in range(len(node_list))}
        inv_perm_map = {perm[i]: node_list[i] for i in range(len(node_list))}

        # Permute graph
        perm_edges = [(perm_map[u], perm_map[v]) for u, v in edges]
        perm_features = {perm_map[v]: node_features[v] for v in nodes}

        # Run MPNN on permuted graph
        result_perm = mp_step(
            perm, perm_edges, perm_features, message_fn, aggregate_fn, update_fn
        )

        # Check: result_perm[π(v)] should equal result_orig[v]
        for v in nodes:
            perm_v = perm_map[v]

            orig_feat = result_orig.new_features[v]
            perm_feat = result_perm.new_features[perm_v]

            if not np.allclose(orig_feat, perm_feat, atol=1e-6):
                return False

    return True


class SimpleMessage(MessageFunction):
    """Simple message: just pass neighbor features."""

    def __call__(
        self, h_i: np.ndarray, h_j: np.ndarray, edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return h_j


class SumAggregate(AggregateFunction):
    """Sum aggregation: Σ mᵢ.

    Properties:
        - Permutation-invariant: ✓
        - Injective: ✓ (for countable multisets)
    """

    def __call__(self, messages: List[np.ndarray]) -> np.ndarray:
        if not messages:
            # Empty neighborhood: return zero vector
            # (Assumes all messages have same dimension)
            return np.zeros_like(messages[0]) if messages else np.zeros(1)
        return np.sum(messages, axis=0)

    @property
    def is_injective(self) -> bool:
        return True


class MeanAggregate(AggregateFunction):
    """Mean aggregation: (1/|N|) Σ mᵢ.

    Properties:
        - Permutation-invariant: ✓
        - Injective: ✗ (loses cardinality information)
    """

    def __call__(self, messages: List[np.ndarray]) -> np.ndarray:
        if not messages:
            return np.zeros(1)  # Fallback
        return np.mean(messages, axis=0)

    @property
    def is_injective(self) -> bool:
        return False  # Mean({1,1,1}) = Mean({3}), but different multisets


class MaxAggregate(AggregateFunction):
    """Max aggregation: maxᵢ mᵢ (element-wise).

    Properties:
        - Permutation-invariant: ✓
        - Injective: ✗ (loses all but maximum)
    """

    def __call__(self, messages: List[np.ndarray]) -> np.ndarray:
        if not messages:
            return np.zeros(1)
        return np.max(messages, axis=0)

    @property
    def is_injective(self) -> bool:
        return False  # Max({1, 5}) = Max({5, 5})


class MLPUpdate(UpdateFunction):
    """MLP update: σ(W₁h + W₂m + b).

    This is a simple parameterized update function.
    In practice, use learned parameters.
    """

    def __init__(self, W1: np.ndarray, W2: np.ndarray, b: np.ndarray):
        self.W1 = W1
        self.W2 = W2
        self.b = b

    def __call__(self, h_old: np.ndarray, m_agg: np.ndarray) -> np.ndarray:
        # Linear combination + bias
        combined = self.W1 @ h_old + self.W2 @ m_agg + self.b
        # ReLU activation
        return np.maximum(0, combined)
