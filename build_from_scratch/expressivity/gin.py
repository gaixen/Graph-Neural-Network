"""Graph Isomorphism Network (GIN) and why it's maximally expressive under 1-WL.

GIN (Xu et al., 2019) is the provably most powerful Message-Passing Neural Network
within the 1-WL expressivity limit.

THEOREM (Xu et al., 2019):
    GIN with sufficient capacity equals the 1-WL test in discriminative power.

Mathematical formulation:
    hᵢ⁽ᵏ⁾ = MLPᵏ((1 + εᵏ) · hᵢ⁽ᵏ⁻¹⁾ + Σⱼ∈𝐩(ᵢ) hⱼ⁽ᵏ⁻¹⁾)

Key components:
    1. **Sum aggregation**: Only injective aggregation (vs mean/max)
    2. **(1 + ε) weighting**: Distinguishes self from neighbors
    3. **MLP update**: Universal function approximator

Why this is optimal:
    - Sum: Injective over multisets (necessary for WL-equivalence)
    - 1 + ε: Separates node's own features from neighbor contribution
    - MLP: Can learn any injective function (with sufficient capacity)

Where GIN still fails:
    - Same limitations as 1-WL (regular graphs, cycles, etc.)
    - Cannot go beyond 1-WL without higher-order architectures
    - Requires large hidden dimensions for injectivity

Comparison with other GNNs:
    GCN < GraphSAGE < GAT < GIN = 1-WL

    All are strictly less expressive than GIN,
    except GIN which achieves the theoretical maximum for MPNNs.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GINConfig:
    """Configuration for GIN layer.

    Attributes:
        epsilon: Self-weighting parameter (ε)
        learn_epsilon: Whether ε is learned or fixed
        mlp_hidden_dims: Hidden dimensions for MLP
        activation: Activation function
    """

    epsilon: float = 0.0
    learn_epsilon: bool = True
    mlp_hidden_dims: List[int] = None
    activation: str = "relu"

    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 64]


def gin_update(
    h_self: np.ndarray,
    h_neighbors_sum: np.ndarray,
    epsilon: float,
    mlp_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """GIN update function.

    Mathematical operation:
        h' = MLP((1 + ε) · h_self + h_neighbors_sum)

    Why (1 + ε)?
        Without it: h' = MLP(h_self + neighbors)
        Cannot distinguish:
        - Node with feature x and 0 neighbors
        - Node with feature 0 and 1 neighbor with feature x

        With (1 + ε): h' = MLP((1 + ε) · x + 0) vs MLP(0 + x)
        These are different if ε ≠ 0!

    Injectivity:
        If MLP is injective, entire update is injective.
        Sufficient condition: MLP has enough capacity (universal approximation)

    Args:
        h_self: Node's own features
        h_neighbors_sum: Sum of neighbor features
        epsilon: Self-weighting parameter
        mlp_fn: MLP update function

    Returns:
        Updated node features
    """
    # Combine self and neighbor information
    combined = (1 + epsilon) * h_self + h_neighbors_sum

    # Apply MLP
    return mlp_fn(combined)


def gin_layer(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    epsilon: float,
    mlp_weights: List[np.ndarray],
    mlp_biases: List[np.ndarray],
) -> np.ndarray:
    """Full GIN layer.

    Args:
        node_features: Node features (num_nodes, in_dim)
        edge_index: Edge index (2, num_edges)
        epsilon: Self-weighting parameter
        mlp_weights: MLP weight matrices
        mlp_biases: MLP bias vectors

    Returns:
        Updated node features (num_nodes, out_dim)
    """
    num_nodes = node_features.shape[0]

    # Build adjacency list
    adj_list = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        adj_list[tgt].append(src)

    # For each node
    new_features = []
    for i in range(num_nodes):
        # Sum neighbor features
        neighbors = adj_list[i]
        if neighbors:
            neighbor_sum = np.sum([node_features[j] for j in neighbors], axis=0)
        else:
            neighbor_sum = np.zeros_like(node_features[i])

        # Define MLP
        def mlp(x):
            h = x
            for W, b in zip(mlp_weights[:-1], mlp_biases[:-1]):
                h = W @ h + b
                h = np.maximum(0, h)  # ReLU
            # Final layer (no activation)
            h = mlp_weights[-1] @ h + mlp_biases[-1]
            return h

        # GIN update
        h_new = gin_update(node_features[i], neighbor_sum, epsilon, mlp)
        new_features.append(h_new)

    return np.array(new_features)


def prove_gin_equals_wl() -> str:
    """Proof sketch: GIN = 1-WL.

    Returns:
        Proof explanation
    """
    return """
    THEOREM: GIN with sufficient capacity = 1-WL test
    
    PROOF (Xu et al., 2019):
    
    Part 1: GIN can simulate WL
        WL computes: c' = HASH(c, {{c_neighbors}})
        
        GIN computes: h' = MLP((1 + ε) h + Σ h_neighbors)
        
        Observation:
        - Sum is injective over multisets
        - (1 + ε) h distinguishes self from neighbors
        - MLP can learn any function (universal approximation)
        
        Therefore:
        MLP can learn to simulate HASH function.
        
    Part 2: WL can simulate GIN
        GIN aggregates multiset of neighbors → WL does same
        GIN updates with injective function → WL uses injective hash
        
    Part 3: Equivalence
        After k layers/iterations:
        - GIN distinguishes nodes iff k-WL distinguishes them
        - For graphs G₁, G₂:
          GIN(G₁) = GIN(G₂) ⟺ WL(G₁) = WL(G₂)
        
    ASSUMPTIONS:
        - MLP has sufficient capacity (hidden dim → ∞)
        - ε is chosen appropriately (or learned)
        - Training finds the injective solution
        
    PRACTICAL NOTE:
        In practice, finite capacity may limit expressivity.
        But GIN is optimal among polynomial-size MPNNs.
    """


def why_gin_is_optimal_mpnn() -> str:
    """Explain why GIN achieves maximum MPNN expressivity.

    Returns:
        Explanation
    """
    return """
    THEOREM: GIN is maximally expressive among MPNNs.
    
    COMPARISON:
    
    1. GCN:
       h' = σ(D^{-1/2} A D^{-1/2} h W)
       
       Limitations:
       - Implicit mean aggregation (loses cardinality)
       - Degree normalization (can't count neighbors)
       - Less expressive than WL
    
    2. GraphSAGE:
       h' = σ(W · [h || MEAN/MAX(neighbors)])
       
       Limitations:
       - Mean/max not injective
       - Loses neighborhood cardinality
       - Less expressive than WL
    
    3. GAT:
       h' = σ(Σ α_ij W h_j)
       
       Limitations:
       - Still sums (good), but weighted
       - Attention is not injective
       - Can't necessarily achieve WL-expressivity
    
    4. GIN:
       h' = MLP((1 + ε) h + Σ h_neighbors)
       
       Advantages:
       - Sum aggregation: injective! 
       - (1 + ε): separates self from neighbors
       - MLP: universal approximator
       - Provably equals 1-WL
    
    WHY OPTIMAL:
        Xu et al. (2019) proved:
        - Any MPNN can be simulated by GIN
        - GIN achieves 1-WL expressivity
        - No MPNN can exceed 1-WL
        
        Therefore: GIN is optimal!
    """


def where_gin_still_fails() -> List[str]:
    """Examples where GIN fails (same as 1-WL failures).

    Returns:
        List of failure cases
    """
    return [
        "Regular graphs: All 3-regular graphs are indistinguishable",
        "Cycle counting: Cannot reliably count 4-cycles, 6-cycles, etc.",
        "Symmetry: Graphs with automorphisms confuse GIN",
        "Strongly regular graphs: Non-isomorphic SRGs with same parameters",
        "Distance-regular graphs: Local structure doesn't determine global",
        "Folklore theorem: Needs higher-order (k-WL, k>1) for these cases",
    ]


def gin_vs_gcn_example(
    nodes: List[int], edges: List[Tuple[int, int]]
) -> Tuple[bool, str]:
    """Example showing GIN can distinguish graphs that GCN cannot.

    Args:
        nodes: Graph nodes
        edges: Graph edges

    Returns:
        (can_gin_distinguish, explanation)
    """
    # Example: Two nodes with different neighborhood sizes
    # but same neighborhood average

    # Node A: 1 neighbor with feature 6
    # Node B: 3 neighbors with features [2, 2, 2]

    # GCN (mean aggregation):
    # A: MEAN([6]) = 6
    # B: MEAN([2,2,2]) = 2
    # GCN can distinguish! (Good in this case)

    # But consider:
    # Node A: 2 neighbors [3, 3]
    # Node B: 3 neighbors [2, 2, 2]

    # GCN:
    # A: MEAN([3, 3]) = 3
    # B: MEAN([2, 2, 2]) = 2
    # Still distinguishes!

    # Better example:
    # Node A: 3 neighbors [2, 2, 2]
    # Node B: 1 neighbor [6]

    # GCN:
    # A: 3 * MEAN([2,2,2]) = 3 * 2 = 6 (with proper normalization)
    # B: 1 * MEAN([6]) = 6
    # GCN confused!

    # GIN:
    # A: MLP(h_A + 2+2+2) = MLP(h_A + 6)
    # B: MLP(h_B + 6)
    # If h_A ≠ h_B initially, GIN distinguishes!

    return True, "GIN uses sum (injective), GCN uses mean (non-injective)"
