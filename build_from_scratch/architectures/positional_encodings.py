"""Positional encodings for graphs: eigenvectors, random walks, structural encodings; tradeoffs.

POSITIONAL ENCODINGS = Adding position information to node features

Why needed?
    - GNNs are permutation-invariant (nodes have no inherent order)
    - But position can be important (center vs peripheral)
    - Transformers especially need PE (no positional info otherwise)

Key challenge for graphs:
    - No natural ordering like sequences (word position)
    - Must define "position" meaningfully
    - Should respect graph structure

Types of positional encodings:
    1. **Spectral**: Laplacian eigenvectors
    2. **Structural**: Degree, centrality, clustering coefficient
    3. **Random walk**: Stationary distribution, landing probabilities
    4. **Distance**: Shortest path distances, anchor distances
    5. **Learned**: Optimized during training

THEOREM (Sato et al., 2021):
    With appropriate positional encodings,
    GNNs can distinguish ANY two non-isomorphic graphs!

    (Exceeds 1-WL limitation)

TRADEOFF:
    - Better expressivity
    - But: May leak graph identity (overfitting)
    - May break theoretical invariances

This module provides:
    - Various PE implementations
    - Analysis of expressivity gains
    - When PEs help/hurt
    - Comparison of methods
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class PositionalEncoding:
    """Container for positional encoding.

    Attributes:
        encoding: Position encoding matrix (num_nodes, encoding_dim)
        encoding_type: Type of encoding
        properties: Additional properties
    """

    encoding: np.ndarray
    encoding_type: str
    properties: Dict[str, Any]


def laplacian_positional_encoding(
    adjacency: np.ndarray, k: int = 8
) -> PositionalEncoding:
    """Laplacian eigenvector positional encoding.

    Method:
        Compute k smallest non-trivial eigenvectors of graph Laplacian
        Use these as positional features

    Mathematical:
        L = D - A (combinatorial Laplacian)
        L ϕ_i = λ_i ϕ_i

        PE_i = [ϕ_1(i), ϕ_2(i), ..., ϕ_k(i)]

    Intuition:
        - ϕ_1: Constant (discard)
        - ϕ_2: Fiedler vector (bisection)
        - Higher ϕ: Finer structural features

    Properties:
        - Unique up to sign flip
        - Captures global structure
        - Generalizes Fourier basis

    Args:
        adjacency: Adjacency matrix
        k: Number of eigenvectors

    Returns:
        Positional encoding
    """
    n = adjacency.shape[0]

    # Compute Laplacian
    D = np.diag(np.sum(adjacency, axis=1))
    L = D - adjacency

    # Eigendecomposition
    try:
        eigvals, eigvecs = la.eigh(L)
    except:
        # Fallback: random encoding
        return PositionalEncoding(
            encoding=np.random.randn(n, k),
            encoding_type="random_fallback",
            properties={"error": "eigendecomposition_failed"},
        )

    # Take k smallest (excluding first trivial one)
    # eigvals are sorted ascending
    pe = eigvecs[:, 1 : k + 1]  # Skip λ_0 = 0

    # Handle sign ambiguity (fix sign to make deterministic)
    for i in range(pe.shape[1]):
        if pe[0, i] < 0:
            pe[:, i] *= -1

    return PositionalEncoding(
        encoding=pe,
        encoding_type="laplacian_eigenvectors",
        properties={
            "eigenvalues": eigvals[1 : k + 1],
            "spectral_gap": eigvals[1] if len(eigvals) > 1 else 0,
        },
    )


def random_walk_positional_encoding(
    adjacency: np.ndarray, k: int = 16
) -> PositionalEncoding:
    """Random walk positional encoding.

    Method:
        Compute k-step random walk landing probabilities
        from each node

    Mathematical:
        P = D^{-1} A (transition matrix)
        PE_i = [P(i,.), P^2(i,.), ..., P^k(i,.)]

        (Landing probabilities after 1, 2, ..., k steps)

    Properties:
        - Captures local neighborhood structure
        - Related to PageRank
        - Expensive to compute for large graphs

    Args:
        adjacency: Adjacency matrix
        k: Number of steps

    Returns:
        Positional encoding
    """
    n = adjacency.shape[0]

    # Transition matrix
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv = np.linalg.inv(D + 1e-10 * np.eye(n))
    P = D_inv @ adjacency

    # Compute powers of P
    encodings = []
    P_k = np.eye(n)

    for step in range(1, k + 1):
        P_k = P_k @ P
        # Use diagonal (self-return probability)
        encodings.append(np.diag(P_k))

    pe = np.array(encodings).T  # (n, k)

    return PositionalEncoding(
        encoding=pe, encoding_type="random_walk", properties={"num_steps": k}
    )


def structural_positional_encoding(adjacency: np.ndarray) -> PositionalEncoding:
    """Structural features as positional encoding.

    Features:
        - Degree centrality
        - Clustering coefficient
        - Betweenness (approximated)
        - Eccentricity (max distance to other nodes)

    Properties:
        - Interpretable
        - Fast to compute
        - Captures local and global structure

    Args:
        adjacency: Adjacency matrix

    Returns:
        Positional encoding
    """
    n = adjacency.shape[0]

    features = []

    # 1. Degree
    degree = np.sum(adjacency, axis=1)
    features.append(degree)

    # 2. Clustering coefficient
    clustering = []
    for i in range(n):
        neighbors = np.where(adjacency[i] > 0)[0]
        if len(neighbors) < 2:
            clustering.append(0)
        else:
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[k]] > 0:
                        triangles += 1
            max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            clustering.append(triangles / max_triangles if max_triangles > 0 else 0)
    features.append(np.array(clustering))

    # 3. Betweenness (simplified: degree^2 as proxy)
    betweenness = degree**2
    features.append(betweenness)

    # 4. Eccentricity (max shortest path distance)
    # Approximate via BFS from each node
    eccentricity = []
    for i in range(n):
        visited = {i}
        queue = [(i, 0)]
        max_dist = 0

        while queue:
            v, dist = queue.pop(0)
            max_dist = max(max_dist, dist)

            for u in range(n):
                if adjacency[v, u] > 0 and u not in visited:
                    visited.add(u)
                    queue.append((u, dist + 1))

        eccentricity.append(max_dist)
    features.append(np.array(eccentricity))

    pe = np.column_stack(features)

    return PositionalEncoding(
        encoding=pe,
        encoding_type="structural",
        properties={
            "features": ["degree", "clustering", "betweenness", "eccentricity"]
        },
    )


def distance_encoding(
    adjacency: np.ndarray,
    anchor_nodes: Optional[List[int]] = None,
    num_anchors: int = 10,
) -> PositionalEncoding:
    """Distance-based positional encoding.

    Method:
        Select k anchor nodes
        Compute shortest path distance from each node to each anchor
        Use distance vector as PE

    Mathematical:
        PE_i = [d(i, a_1), d(i, a_2), ..., d(i, a_k)]

        where a_j are anchor nodes

    Properties:
        - Captures global position relative to anchors
        - Can reconstruct graph (approximately)
        - Sensitive to anchor selection

    Args:
        adjacency: Adjacency matrix
        anchor_nodes: Pre-selected anchors (or None for random)
        num_anchors: Number of anchors if not provided

    Returns:
        Positional encoding
    """
    n = adjacency.shape[0]

    # Select anchors
    if anchor_nodes is None:
        anchor_nodes = np.random.choice(n, min(num_anchors, n), replace=False).tolist()

    # Compute distances via BFS
    distances = []

    for anchor in anchor_nodes:
        dist = np.full(n, np.inf)
        dist[anchor] = 0
        queue = [anchor]
        visited = {anchor}

        while queue:
            v = queue.pop(0)
            for u in range(n):
                if adjacency[v, u] > 0 and u not in visited:
                    visited.add(u)
                    dist[u] = dist[v] + 1
                    queue.append(u)

        distances.append(dist)

    pe = np.array(distances).T  # (n, num_anchors)

    # Replace inf with large value
    pe[pe == np.inf] = n

    return PositionalEncoding(
        encoding=pe, encoding_type="distance", properties={"anchors": anchor_nodes}
    )


def compare_positional_encodings() -> Dict[str, Dict[str, Any]]:
    """Compare different PE methods.

    Returns:
        Comparison table
    """
    return {
        "Laplacian": {
            "pros": [
                "Theoretically grounded (spectral graph theory)",
                "Captures global structure",
                "Generalizes Fourier basis",
            ],
            "cons": [
                "Expensive (eigendecomposition O(n^3))",
                "Sign ambiguity (need fixing)",
                "May leak graph identity",
            ],
            "when_to_use": "Small-medium graphs, need global structure",
        },
        "Random Walk": {
            "pros": [
                "Captures local neighborhoods",
                "Related to PageRank",
                "Interpretable",
            ],
            "cons": [
                "Expensive (matrix powers)",
                "Local only (doesn't capture global)",
                "Sensitive to graph density",
            ],
            "when_to_use": "Need local neighborhood info",
        },
        "Structural": {
            "pros": [
                "Fast to compute",
                "Interpretable features",
                "Robust",
            ],
            "cons": [
                "Limited expressivity",
                "May not capture complex patterns",
                "Fixed features (not learned)",
            ],
            "when_to_use": "Large graphs, need speed",
        },
        "Distance": {
            "pros": [
                "Captures global position",
                "Can reconstruct graph",
                "Flexible (choose anchors)",
            ],
            "cons": [
                "Sensitive to anchor selection",
                "Expensive for many anchors",
                "High dimensional",
            ],
            "when_to_use": "Need global position info",
        },
    }


def expressivity_with_positional_encodings() -> str:
    """How PEs increase expressivity beyond 1-WL.

    Returns:
        Explanation
    """
    return """
    THEOREM (Sato et al., 2021; Kreuzer et al., 2021):
        GNN + Positional Encodings can exceed 1-WL!
    
    WHY:
        1-WL limitation:
        - Cannot distinguish nodes with identical local structure
        
        With PE:
        - Each node has unique position information
        - Even if neighborhoods identical, PE differs
        
    EXAMPLE:
        Regular graph: All nodes identical to 1-WL
        With Laplacian PE: Eigenvector values differ
        → GNN can now distinguish nodes!
    
    THEORETICAL BOUND:
        With appropriate PE (e.g., all pairwise distances):
        GNN can solve GRAPH ISOMORPHISM!
        
        (In practice: Computationally expensive)
    
    CAUTION:
        Too powerful PE can OVERFIT:
        - PE encodes entire graph
        - GNN just memorizes
        - Doesn't generalize!
        
    BEST PRACTICE:
        Use low-dimensional PE (k=8-16)
        Let GNN learn to use PE
        Regularize to prevent overfitting
    """


def when_pes_hurt() -> List[str]:
    """Cases where PEs can harm performance.

    Returns:
        Failure cases
    """
    return [
        "Overfitting: PE too expressive, memorizes training graphs",
        "Graph identity leakage: PE uniquely identifies graph (test != train)",
        "Breaks invariances: PE may not be permutation-invariant",
        "Computational cost: Expensive to compute for large graphs",
        "Hyperparameter sensitivity: k, anchor selection, etc.",
        "Domain mismatch: PE assumptions don't match problem structure",
    ]
