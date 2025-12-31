"""Subgraph GNNs and their theoretical targets: capture local substructure at higher expressivity.

SUBGRAPH GNNs = Operate on subgraphs instead of individual nodes

Key idea:
    Instead of node-level message passing,
    extract subgraphs and apply GNN to each subgraph
    
    Node embedding = function of its subgraph context

Motivation:
    - Regular MPNNs limited to 1-WL
    - Subgraph methods can exceed 1-WL
    - Can count patterns (triangles, motifs)
    - Better expressivity-complexity tradeoff than k-WL

Types of subgraph GNNs:
    1. **Ego-networks**: k-hop neighborhood around each node
    2. **Subgraph sampling**: Random subgraphs containing node
    3. **Induced subgraphs**: All nodes + edges in radius
    4. **GNN-AK**: All subgraphs up to size k
    5. **ESAN**: Permutation-invariant subgraph aggregation

THEOREM (Frasca et al., 2022; Bevilacqua et al., 2021):
    Subgraph GNNs with appropriate aggregation
    can distinguish any two non-isomorphic graphs
    
    (Exceeds k-WL for any k!)

TRADEOFFS:
    Pros:
        - Higher expressivity than MPNNs
        - Can count substructures
        - Modular (use any GNN architecture)
    
    Cons:
        - Expensive preprocessing (extract subgraphs)
        - Memory intensive (many subgraphs)
        - Complex implementation
        - May not generalize across graph sizes

This module provides:
    - Subgraph extraction methods
    - Theoretical expressivity analysis
    - When to use subgraph methods
    - Complexity analysis
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Subgraph:
    """A subgraph of the original graph.
    
    Attributes:
        nodes: Nodes in subgraph
        edges: Edges in subgraph (as (u,v) pairs)
        anchor_node: Central node (if ego-network)
        node_features: Features of nodes in subgraph
        subgraph_id: Unique identifier
    """
    nodes: List[int]
    edges: List[Tuple[int, int]]
    anchor_node: Optional[int]
    node_features: Optional[np.ndarray]
    subgraph_id: int


def extract_ego_network(
    node: int,
    adjacency: np.ndarray,
    hop: int = 2
) -> Subgraph:
    """Extract k-hop ego-network around a node.
    
    Ego-network = induced subgraph of all nodes within k hops
    
    Mathematical:
        E_k(v) = {u : d(u, v) ≤ k}
        Subgraph = G[E_k(v)]
    
    Properties:
        - Contains all local structure
        - Size grows exponentially with k
        - Overlapping for different nodes
    
    Args:
        node: Anchor node
        adjacency: Full graph adjacency
        hop: Number of hops
    
    Returns:
        Ego-network subgraph
    """
    n = adjacency.shape[0]
    
    # BFS to find k-hop neighbors
    visited = {node}
    current_level = {node}
    
    for _ in range(hop):
        next_level = set()
        for v in current_level:
            for u in range(n):
                if adjacency[v, u] > 0 and u not in visited:
                    visited.add(u)
                    next_level.add(u)
        current_level = next_level
    
    nodes = list(visited)
    
    # Extract edges
    edges = []
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and adjacency[u, v] > 0:
                edges.append((i, j))  # Local indices
    
    return Subgraph(
        nodes=nodes,
        edges=edges,
        anchor_node=node,
        node_features=None,
        subgraph_id=node
    )


def extract_all_ego_networks(
    adjacency: np.ndarray,
    hop: int = 2
) -> List[Subgraph]:
    """Extract ego-networks for all nodes.
    
    Returns:
        List of ego-networks (one per node)
    """
    n = adjacency.shape[0]
    return [extract_ego_network(i, adjacency, hop) for i in range(n)]


def subgraph_gnn_forward(
    subgraphs: List[Subgraph],
    base_gnn: Any,
    readout: str = 'mean'
) -> np.ndarray:
    """Forward pass for subgraph GNN.
    
    Procedure:
        1. For each subgraph:
           a. Apply base GNN
           b. Readout to get subgraph embedding
        2. Aggregate subgraph embeddings back to nodes
    
    Mathematical:
        h_v = AGG({GNN(G_s) : v ∈ G_s})
        
        where G_s are subgraphs containing v
    
    Args:
        subgraphs: List of subgraphs
        base_gnn: Base GNN function
        readout: Readout function
    
    Returns:
        Node embeddings
    """
    # Placeholder: In practice, apply GNN to each subgraph
    # and aggregate results
    
    # For demonstration:
    num_nodes = max(max(s.nodes) for s in subgraphs) + 1
    node_embeddings = np.zeros((num_nodes, 64))  # Placeholder dimension
    
    for subgraph in subgraphs:
        # Apply GNN to subgraph (simplified)
        # subgraph_emb = base_gnn(subgraph)
        subgraph_emb = np.random.randn(64)  # Placeholder
        
        # Assign to anchor node
        if subgraph.anchor_node is not None:
            node_embeddings[subgraph.anchor_node] = subgraph_emb
    
    return node_embeddings


def why_subgraph_gnns_more_expressive() -> str:
    """Explain expressivity gains.
    
    Returns:
        Explanation
    """
    return """
    WHY SUBGRAPH GNNs > MPNNs:
    
    1. CAPTURE NON-LOCAL PATTERNS:
       MPNN: Only aggregates neighbors iteratively
       Subgraph GNN: Sees entire k-hop neighborhood at once
       
       Can detect:
       - Triangles
       - Cliques
       - Motifs
       
    2. EXCEED 1-WL:
       MPNN: Limited to 1-WL
       Subgraph GNN: Can achieve 3-WL expressivity!
       
       THEOREM (Frasca et al., 2022):
       Ego-network GNN with k=2 hops → 3-WL
       
    3. COUNT SUBSTRUCTURES:
       MPNN: Cannot reliably count triangles, 4-cycles
       Subgraph GNN: Can count by examining subgraphs
       
    4. BREAK SYMMETRY:
       Regular graphs defeat MPNNs
       Subgraph GNN: Different ego-networks → distinguishable
    
    EXAMPLE:
        Graph: 6-cycle C₆
        MPNN: All nodes identical
        Subgraph GNN: Each ego-network different (unless highly symmetric)
    
    THEORETICAL LIMIT:
        THEOREM (Bevilacqua et al., 2021):
        With all subgraphs up to size k,
        can achieve k-WL expressivity!
        
        With all subgraphs → solve graph isomorphism!
    """


def complexity_analysis() -> Dict[str, Any]:
    """Analyze complexity of subgraph methods.
    
    Returns:
        Complexity breakdown
    """
    return {
        "preprocessing": {
            "ego_networks": "O(n × d^k) where d=avg degree, k=hops",
            "all_subgraphs_size_k": "O(n choose k) = O(n^k) → exponential!",
            "sampling": "O(n × s) where s=samples per node",
        },
        "memory": {
            "ego_networks": "O(n × |ego|) ≈ O(n × d^k)",
            "all_subgraphs": "O(n^k) → intractable for large k",
        },
        "forward_pass": {
            "per_subgraph": "O(GNN_cost) × num_subgraphs",
            "ego_networks": "O(n × GNN_cost_on_ego)",
        },
        "practical_limits": {
            "small_graphs": "<1000 nodes, all methods work",
            "medium_graphs": "1k-10k nodes, use ego-networks or sampling",
            "large_graphs": ">10k nodes, sampling only or avoid",
        },
    }


def gnn_ak_method() -> str:
    """Explain GNN-AK (all subgraphs up to size k).
    
    Returns:
        Explanation
    """
    return """
    GNN-AK (Zhao et al., 2021):
    
    IDEA:
        Enumerate all connected subgraphs up to size k
        Apply GNN to each
        Aggregate results
    
    ALGORITHM:
        1. For each node v:
           - Find all connected subgraphs of size ≤ k containing v
        2. For each subgraph S:
           - Apply GNN: emb_S = GNN(S)
        3. Aggregate:
           h_v = AGG({emb_S : v ∈ S})
    
    EXPRESSIVITY:
        THEOREM: GNN-AK achieves k-WL expressivity
        
        Proof: Can distinguish all graphs that k-WL distinguishes
    
    COMPLEXITY:
        Exponential in k!
        
        Number of subgraphs: O(n^k)
        Practical: k ≤ 5
    
    WHEN TO USE:
        - Need high expressivity
        - Small graphs (n < 1000)
        - Small k (k ≤ 3)
    """


def esan_method() -> str:
    """Explain ESAN (Equivariant Subgraph Aggregation Networks).
    
    Returns:
        Explanation
    """
    return """
    ESAN (Bevilacqua et al., 2021):
    
    PROBLEM:
        Subgraph methods lose permutation equivariance
        (Different subgraph extraction → different result)
    
    SOLUTION:
        Design equivariant subgraph aggregation
    
    METHOD:
        1. Extract policy: Define which subgraphs to use
           (e.g., all k-hop ego-networks)
        2. For each subgraph policy:
           - Extract subgraphs
           - Apply GNN
           - Pool to graph-level
        3. Aggregate across policies (equivariantly)
    
    KEY INSIGHT:
        If policy is equivariant (e.g., "all ego-networks"),
        then aggregation preserves equivariance
    
    THEOREM:
        ESAN with appropriate policies can:
        - Exceed k-WL for any k
        - Solve graph isomorphism (in limit)
        - Maintain equivariance
    
    PRACTICAL:
        Use few policies (2-3)
        Each policy = different subgraph type
        
        Example policies:
        - 1-hop ego-networks
        - 2-hop ego-networks
        - Node-deleted subgraphs
    """


def when_to_use_subgraph_gnns() -> Dict[str, str]:
    """Guidelines for using subgraph GNNs.
    
    Returns:
        Recommendations
    """
    return {
        "need_high_expressivity": "Use subgraph GNNs (exceed 1-WL)",
        "count_patterns": "Use subgraph GNNs (can count triangles, motifs)",
        "small_graphs": "Feasible (n < 1000)",
        "large_graphs": "Use sampling or avoid (too expensive)",
        "regular_graphs": "Subgraph GNNs help (MPNNs fail)",
        "local_structure": "Ego-networks sufficient",
        "global_properties": "Need larger subgraphs or different method",
        "limited_compute": "Use MPNNs instead (subgraphs expensive)",
    }


def limitations() -> List[str]:
    """Limitations of subgraph GNNs.
    
    Returns:
        List of limitations
    """
    return [
        "Computational cost: Exponential in subgraph size",
        "Memory: Must store many subgraphs",
        "Preprocessing: Expensive subgraph extraction",
        "Scalability: Doesn't scale to large graphs",
        "Generalization: May not generalize across graph sizes",
        "Implementation complexity: Harder to implement than MPNNs",
        "Overlapping subgraphs: Information redundancy",
    ]
