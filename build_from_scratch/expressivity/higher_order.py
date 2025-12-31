"""Higher-order methods that go beyond 1-WL: k-WL, tuples, and the tradeoffs.

The 1-WL test and MPNNs share fundamental limitations.
To exceed their expressivity, we need higher-order methods.

Key approaches:
    1. **k-WL test**: Refine k-tuples of nodes instead of single nodes
    2. **Tensor GNNs**: Operate on edge/triangle/k-clique tensors
    3. **Subgraph GNNs**: Count/embed subgraph patterns
    4. **Higher-order message passing**: Messages on k-tuples

Expressivity hierarchy:
    1-WL < 2-WL < 3-WL < ... < k-WL < ... < Graph Isomorphism

THEOREM (Cai, Fürer, Immerman, 1992):
    For any fixed k, there exist non-isomorphic graphs that k-WL cannot distinguish.
    But k-WL is complete for graphs of size n with k ≥ n.

The tradeoff:
    - Expressivity: k-WL > (k-1)-WL
    - Complexity: k-WL requires O(n^k) time and space
    - Practicality: k=2 already expensive; k=3 usually intractable

This module explores:
    - k-WL algorithms for k > 1
    - Where k-WL succeeds and fails
    - Tensor-based GNN architectures
    - Practical complexity vs expressivity tradeoffs
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Any
from itertools import product, combinations
from dataclasses import dataclass
import hashlib


@dataclass
class KWLColoring:
    """k-WL color assignment.

    Attributes:
        tuple_nodes: k-tuple of node IDs
        color: Color/hash value
        iteration: Iteration number
    """

    tuple_nodes: Tuple[int, ...]
    color: int
    iteration: int


def k_wl_refine(
    node_tuples: List[Tuple[int, ...]],
    colors: Dict[Tuple[int, ...], int],
    edges: Set[Tuple[int, int]],
    k: int,
) -> Dict[Tuple[int, ...], int]:
    """One iteration of k-WL color refinement.

    k-WL algorithm:
        For each k-tuple t = (v₁, v₂, ..., vₖ):
        1. Collect colors of all k-tuples differing in exactly one position
        2. Hash (current_color, multiset of neighbor colors)
        3. Assign new color

    Mathematical formulation:
        c'(v₁,...,vₖ) = HASH(c(v₁,...,vₖ),
                             {{c(v₁,...,vᵢ₋₁, u, vᵢ₊₁,...,vₖ) : u ∈ V}})

    Complexity:
        - Number of tuples: O(n^k)
        - Neighbors per tuple: O(k · n)
        - Total per iteration: O(k · n^(k+1))

    Args:
        node_tuples: All k-tuples of nodes
        colors: Current coloring
        edges: Graph edges
        k: Tuple size

    Returns:
        New coloring
    """
    new_colors = {}

    for t in node_tuples:
        # Current color
        current_color = colors[t]

        # For each position i in tuple
        neighbor_color_multisets = []
        for i in range(k):
            # Collect all colors where position i is replaced
            neighbor_colors = []
            for u in range(len(set(sum([list(tup) for tup in node_tuples], [])))):
                # Create new tuple with u at position i
                new_tuple = t[:i] + (u,) + t[i + 1 :]
                if new_tuple in colors:
                    neighbor_colors.append(colors[new_tuple])
            neighbor_color_multisets.append(tuple(sorted(neighbor_colors)))

        # Hash (current_color, all neighbor multisets)
        combined = (current_color, tuple(neighbor_color_multisets))
        new_colors[t] = abs(hash(combined))

    return new_colors


def two_wl_test(
    nodes: List[int], edges: List[Tuple[int, int]], max_iterations: int = 10
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """2-WL test (operates on node pairs).

    2-WL is stronger than 1-WL:
        - Can distinguish some regular graphs
        - Can count triangles reliably
        - Still fails on strongly regular graphs

    Complexity:
        - Space: O(n^2) colors
        - Time per iteration: O(n^3)
        - More practical than higher k

    Args:
        nodes: Graph nodes
        edges: Graph edges
        max_iterations: Maximum iterations

    Returns:
        (final_coloring, num_iterations)
    """
    n = len(nodes)

    # Generate all node pairs
    pairs = [(i, j) for i in nodes for j in nodes]

    # Initialize coloring based on edge presence
    edge_set = set(edges)
    colors = {}
    for i, j in pairs:
        if i == j:
            colors[(i, j)] = 0  # Diagonal
        elif (i, j) in edge_set or (j, i) in edge_set:
            colors[(i, j)] = 1  # Edge
        else:
            colors[(i, j)] = 2  # Non-edge

    # Iterate
    for iteration in range(max_iterations):
        new_colors = {}

        for i, j in pairs:
            # Collect neighbor colors
            # For 2-WL: neighbors are (i, k) and (k, j) for all k
            neighbor_colors_i = [colors[(i, k)] for k in nodes]
            neighbor_colors_j = [colors[(k, j)] for k in nodes]

            # Hash
            current = colors[(i, j)]
            multiset = (
                tuple(sorted(neighbor_colors_i)),
                tuple(sorted(neighbor_colors_j)),
            )
            new_colors[(i, j)] = abs(hash((current, multiset)))

        # Check convergence
        if new_colors == colors:
            return colors, iteration + 1

        colors = new_colors

    return colors, max_iterations


def when_does_2wl_beat_1wl() -> str:
    """Examples where 2-WL succeeds but 1-WL fails.

    Returns:
        Explanation
    """
    return """
    THEOREM: 2-WL > 1-WL (strictly more expressive)
    
    EXAMPLES WHERE 2-WL SUCCEEDS:
    
    1. **Triangle counting**:
       1-WL: Cannot reliably count triangles
       2-WL: Can count triangles exactly!
       
       Reason: 2-WL operates on pairs (i,j)
       Can detect when both (i,k) and (k,j) are edges
       
    2. **4-cycles**:
       1-WL: Cannot distinguish graphs with different 4-cycle counts
       2-WL: Can count 4-cycles
       
    3. **Some regular graphs**:
       Example: Shrikhande graph vs 4×4 Rook graph
       - Both 6-regular
       - 1-WL: Cannot distinguish
       - 2-WL: CAN distinguish!
       
    WHERE 2-WL STILL FAILS:
    
    1. **Strongly regular graphs**:
       Some SRG pairs defeat 2-WL
       (Need 3-WL or higher)
       
    2. **CFI graphs**:
       Cai-Fürer-Immerman construction
       Defeats k-WL for any fixed k
       
    COMPLEXITY:
       1-WL: O(n^2) time, O(n) space
       2-WL: O(n^3) time, O(n^2) space
       
       2-WL is practical for medium graphs (<10k nodes)
    """


def three_wl_and_beyond() -> str:
    """What 3-WL and higher can do.

    Returns:
        Explanation
    """
    return """
    k-WL HIERARCHY:
    
    THEOREM (CFI, 1992):
        For all k, there exist graphs that k-WL cannot distinguish.
        
    But k-WL gets progressively more powerful:
    
    3-WL:
        - Operates on node triples (i, j, k)
        - Complexity: O(n^4) time, O(n^3) space
        - Can distinguish most graphs in practice
        - Fails on certain strongly regular graphs
    
    n-WL (k = n):
        - Operates on all n-tuples
        - Complexity: O(n^(n+1)) time, O(n^n) space
        - THEOREM: Equivalent to graph isomorphism for n nodes!
        - Completely impractical
    
    FOLKLORE THEOREM:
        For graphs with n nodes:
        If k ≥ n + 1, then k-WL solves graph isomorphism exactly.
        
        But this is exponential time!
    
    PRACTICAL TRADEOFF:
        - 1-WL: Very fast, limited expressivity (MPNNs)
        - 2-WL: Practical, moderate expressivity
        - 3-WL: Expensive, high expressivity
        - k>3-WL: Usually impractical
    
    OPEN QUESTION:
        Is there a polynomial-time algorithm that beats k-WL
        for all k, without solving GI?
        
        (Related to P vs NP!)
    """


def tensor_gnn_approach() -> str:
    """Tensor-based GNNs as alternative to k-WL.

    Returns:
        Explanation
    """
    return """
    TENSOR GNNs:
    
    Instead of message passing on nodes,
    operate on higher-order tensors:
    
    1. **Edge-level GNN**:
       - Features on edges, not nodes
       - Messages between adjacent edges
       - Can represent 2-WL
       
    2. **Triangle-level GNN**:
       - Features on triangles
       - Messages between triangles sharing edges
       - More expressive than 2-WL
       
    3. **k-clique GNN**:
       - Features on k-cliques
       - Messages between overlapping cliques
       - Expressive but expensive
    
    ADVANTAGES:
        - More structured than raw k-WL
        - Can use standard GNN architectures
        - Interpretable (geometric objects)
    
    DISADVANTAGES:
        - Still O(n^k) complexity
        - Sparse graphs have few high-order cliques
        - Implementation complexity
    
    EXAMPLES:
        - Simplicial Neural Networks (SNNs)
        - Cell Complex Neural Networks
        - Hodge Laplacians on simplicial complexes
    """


def practical_expressivity_tradeoff() -> Dict[str, Any]:
    """Summarize expressivity vs complexity tradeoffs.

    Returns:
        Comparison table
    """
    return {
        "1-WL / MPNNs": {
            "expressivity": "Low (fails on regular graphs)",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(n)",
            "practical_limit": "Millions of nodes",
            "examples": ["GCN", "GraphSAGE", "GAT", "GIN"],
        },
        "2-WL": {
            "expressivity": "Medium (counts triangles, some regular graphs)",
            "time_complexity": "O(n^3)",
            "space_complexity": "O(n^2)",
            "practical_limit": "10k-100k nodes",
            "examples": ["Edge GNNs", "2-FWL GNNs"],
        },
        "3-WL": {
            "expressivity": "High (most graphs)",
            "time_complexity": "O(n^4)",
            "space_complexity": "O(n^3)",
            "practical_limit": "1k-10k nodes",
            "examples": ["3-WL", "Triangle GNNs"],
        },
        "k-WL (k>3)": {
            "expressivity": "Very high (diminishing returns)",
            "time_complexity": "O(n^(k+1))",
            "space_complexity": "O(n^k)",
            "practical_limit": "<1k nodes",
            "examples": ["Theoretical only"],
        },
        "Graph Isomorphism": {
            "expressivity": "Perfect (by definition)",
            "time_complexity": "Quasi-polynomial (best known)",
            "space_complexity": "Polynomial",
            "practical_limit": "Unknown",
            "examples": ["Nauty", "Bliss", "Babai's algorithm"],
        },
    }
