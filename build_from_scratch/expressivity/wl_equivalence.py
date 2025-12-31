"""Formal mapping between MPNNs and 1-WL; node-level and graph-level collapse.

This module establishes the fundamental theorem of GNN expressivity:

THEOREM (Xu et al., 2019; Morris et al., 2019):
    Message-Passing Neural Networks with sum aggregation are AT MOST
    as powerful as the 1-dimensional Weisfeiler-Lehman (1-WL) test.

Mathematical statement:
    For any MPNN M and graphs G₁, G₂:
    If 1-WL(G₁) ≠ 1-WL(G₂), then M can distinguish G₁ from G₂.
    If 1-WL(G₁) = 1-WL(G₂), then M CANNOT distinguish G₁ from G₂.

Proof sketch:
    Both MPNN and 1-WL iteratively refine node labels using:
    - Current node label
    - Multiset of neighbor labels

    The correspondence:
    - MPNN aggregation ↔ WL multiset hashing
    - MPNN update ↔ WL color refinement
    - Injectivity of aggregation ↔ Injectivity of hashing

Consequences:
    1. GNNs CANNOT solve graph isomorphism (WL is incomplete)
    2. Regular graphs often defeat GNNs (same degree sequence)
    3. GIN with sum aggregation = optimal MPNN (matches 1-WL exactly)

This module provides:
    - Formal correspondence between MPNN and WL
    - Node-level vs graph-level expressivity
    - Counterexamples where both fail
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np
from dataclasses import dataclass
import hashlib


@dataclass
class NodeEmbedding:
    """Node embedding at a given layer.

    Attributes:
        node_id: Node identifier
        embedding: Feature vector
        layer: Layer index
    """

    node_id: Any
    embedding: np.ndarray
    layer: int


@dataclass
class WLColoring:
    """WL color assignment at a given iteration.

    Attributes:
        node_id: Node identifier
        color: Color/hash value
        iteration: WL iteration
    """

    node_id: Any
    color: int
    iteration: int


def mpnn_step_abstraction(
    h_i: np.ndarray,
    neighbor_embeddings: List[np.ndarray],
    aggregation: str = "sum",
    update_mlp: Any = None,
) -> np.ndarray:
    """Abstract MPNN step matching WL structure.

    Mathematical operation:
        hᵢ⁽ᵗ⁾ = UPDATE(hᵢ⁽ᵗ⁻¹⁾, AGG({hⱼ⁽ᵗ⁻¹⁾ : j ∈ N(i)}))

    WL analog:
        cᵢ⁽ᵗ⁾ = HASH(cᵢ⁽ᵗ⁻¹⁾, {{cⱼ⁽ᵗ⁻¹⁾ : j ∈ N(i)}})

    Correspondence:
        - hᵢ ↔ cᵢ (node representation/color)
        - AGG ↔ multiset (both must be permutation-invariant)
        - UPDATE ↔ HASH (both combine old state + neighborhood)

    Args:
        h_i: Current node embedding
        neighbor_embeddings: Neighbor embeddings
        aggregation: Aggregation type ('sum', 'mean', 'max')
        update_mlp: Update function (default: concatenate)

    Returns:
        Updated node embedding
    """
    # Aggregate neighbors
    if aggregation == "sum":
        agg = (
            np.sum(neighbor_embeddings, axis=0)
            if neighbor_embeddings
            else np.zeros_like(h_i)
        )
    elif aggregation == "mean":
        agg = (
            np.mean(neighbor_embeddings, axis=0)
            if neighbor_embeddings
            else np.zeros_like(h_i)
        )
    elif aggregation == "max":
        agg = (
            np.max(neighbor_embeddings, axis=0)
            if neighbor_embeddings
            else np.zeros_like(h_i)
        )
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Update (default: concatenate self and aggregated)
    if update_mlp is None:
        # Simple concatenation (like WL)
        return np.concatenate([h_i, agg])
    else:
        return update_mlp(h_i, agg)


def wl_step_abstraction(c_i: int, neighbor_colors: List[int]) -> int:
    """Abstract WL color refinement step.

    Mathematical operation:
        cᵢ⁽ᵗ⁾ = HASH(cᵢ⁽ᵗ⁻¹⁾, SORT([cⱼ⁽ᵗ⁻¹⁾ : j ∈ N(i)]))

    Key insight:
        WL uses SORT to handle multiset (permutation-invariant!)
        MPNN uses AGG for same purpose

    Args:
        c_i: Current node color
        neighbor_colors: Neighbor colors

    Returns:
        New node color
    """
    # Create multiset representation (sorted list)
    neighbor_multiset = tuple(sorted(neighbor_colors))

    # Hash (color, multiset) pair
    combined = (c_i, neighbor_multiset)
    hash_value = hash(combined)

    # Ensure non-negative
    return abs(hash_value)


def prove_sum_aggregation_matches_wl() -> str:
    """Explain why sum aggregation = WL multiset hashing.

    Returns:
        Proof sketch
    """
    return """
    THEOREM: Sum aggregation with injective update = WL color refinement
    
    PROOF:
    1. Multiset property:
       WL: Uses sorted list to represent multiset
       MPNN: Sum is permutation-invariant over multiset
       
       Both capture the SAME information (multiset of neighbors)
    
    2. Injectivity:
       WL: Hash function is injective (different multisets → different colors)
       MPNN: Sum + MLP can be injective (with sufficient dimensions)
       
       GIN uses: h' = MLP((1 + ε)h + Σ h_neighbors)
       This is injective if MLP is injective and ε is learnable
    
    3. Iteration:
       Both refine node representations iteratively
       After k iterations:
       - WL: Captures k-hop neighborhood structure
       - MPNN: Captures k-hop neighborhood (k layers)
    
    4. Convergence:
       Both stabilize when no more distinctions can be made
       
    CONCLUSION:
        MPNN with sum + injective update ≡ 1-WL
        
        Any graph pair that 1-WL cannot distinguish,
        MPNN also cannot distinguish!
    """


def demonstrate_equivalence(
    nodes: List[int], edges: List[Tuple[int, int]], num_iterations: int = 3
) -> Tuple[Dict[int, List[int]], Dict[int, List[np.ndarray]]]:
    """Run WL and MPNN in parallel, showing correspondence.

    Args:
        nodes: Graph nodes
        edges: Graph edges
        num_iterations: Number of iterations

    Returns:
        (wl_colors_history, mpnn_embeddings_history)
    """
    # Build adjacency
    adj = {v: [] for v in nodes}
    for u, v in edges:
        adj[v].append(u)
        adj[u].append(v)

    # Initialize WL colors (uniform)
    wl_colors = {v: 0 for v in nodes}
    wl_history = {v: [0] for v in nodes}

    # Initialize MPNN embeddings (one-hot or random)
    mpnn_embeds = {v: np.array([1.0]) for v in nodes}  # All start with [1]
    mpnn_history = {v: [np.array([1.0])] for v in nodes}

    # Iterate
    for iteration in range(num_iterations):
        # WL step
        new_wl_colors = {}
        for v in nodes:
            neighbor_colors = [wl_colors[u] for u in adj[v]]
            new_wl_colors[v] = wl_step_abstraction(wl_colors[v], neighbor_colors)
            wl_history[v].append(new_wl_colors[v])
        wl_colors = new_wl_colors

        # MPNN step
        new_mpnn_embeds = {}
        for v in nodes:
            neighbor_embeds = [mpnn_embeds[u] for u in adj[v]]
            new_mpnn_embeds[v] = mpnn_step_abstraction(
                mpnn_embeds[v], neighbor_embeds, aggregation="sum"
            )
            mpnn_history[v].append(new_mpnn_embeds[v])
        mpnn_embeds = new_mpnn_embeds

    return wl_history, mpnn_history


def node_level_vs_graph_level_expressivity() -> str:
    """Explain the difference between node and graph-level expressivity.

    Returns:
        Explanation
    """
    return """
    NODE-LEVEL EXPRESSIVITY:
        Can the GNN distinguish nodes within a graph?
        
        Example:
        In a star graph (one center, many leaves):
        - GNN can distinguish center from leaves (different degrees)
        - WL can also make this distinction
        
    GRAPH-LEVEL EXPRESSIVITY:
        Can the GNN distinguish two different graphs?
        
        Example:
        Two non-isomorphic regular graphs:
        - All nodes have same degree
        - GNN assigns same embedding to all nodes
        - Pooling gives same graph embedding
        - WL also fails!
        
    KEY INSIGHT:
        Graph-level expressivity requires:
        1. Node-level discrimination
        2. Permutation-invariant readout
        
        Even if GNN can distinguish nodes,
        poor readout can lose information!
        
    THEOREM:
        For graph classification:
        MPNN expressivity ≤ 1-WL expressivity
        
        Equality requires:
        - Injective aggregation (sum)
        - Injective update (MLP)
        - Injective readout (sum)
    """


def why_mean_max_fail_wl_equivalence() -> str:
    """Explain why mean/max aggregation breaks WL correspondence.

    Returns:
        Explanation
    """
    return """
    WL uses MULTISET of neighbor colors:
        - {1, 1, 2} ≠ {1, 2, 2}
        - WL can distinguish these
    
    MEAN aggregation:
        - MEAN({1, 1, 2}) = 4/3
        - MEAN({1, 2, 2}) = 5/3
        - Still distinguishes (Good!)
        
        BUT:
        - MEAN({1, 1, 1}) = 1 = MEAN({1})
        - Different multisets → same output
        - Loses cardinality!
        
        GNN with mean < 1-WL
    
    MAX aggregation:
        - MAX({1, 5}) = 5 = MAX({5, 5})
        - Loses all non-maximal elements
        
        GNN with max < 1-WL
    
    SUM aggregation:
        - SUM({1, 1, 2}) = 4
        - SUM({1, 2, 2}) = 5
        - SUM({1, 1, 1}) = 3 ≠ SUM({1}) = 1
        
        Injective over countable multisets!
        
        GNN with sum = 1-WL (with injective update)
    
    CONCLUSION:
        Only SUM aggregation achieves WL-equivalence.
        GCN (mean), GraphSAGE (mean/max) < GIN (sum) = 1-WL
    """


def limitations_of_1wl_and_mpnns() -> List[str]:
    """List fundamental limitations shared by 1-WL and MPNNs.

    Returns:
        List of limitation examples
    """
    return [
        "Cannot count cycles: Distinguishing C₆ from C₈ (6-cycle vs 8-cycle) requires counting",
        "Regular graphs: All d-regular graphs look identical to WL (locally symmetric)",
        "No global properties: Cannot detect bipartiteness, planarity, Hamiltonicity",
        "Triangle counting: Cannot count number of triangles reliably",
        "Connectivity patterns: Cannot detect strongly regular graphs with same parameters",
        "Symmetry: Automorphisms fool both WL and MPNNs",
    ]
