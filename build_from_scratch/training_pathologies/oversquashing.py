"""Combinatorial bottlenecks and oversquashing: why long-range information gets compressed.

OVERSQUASHING is the phenomenon where long-range information is compressed
through narrow bottlenecks in the graph, causing information loss.

Difference from oversmoothing:
    - OVERSMOOTHING: Nodes become too similar (spectral issue)
    - OVERSQUASHING: Distant information compressed (combinatorial issue)

Mathematical characterization:
    For node v at distance k:
    - Number of k-hop paths: potentially exponential
    - Bottleneck edges: cut capacity << path count
    - Information: O(exp(k)) bits compressed into O(d) dimensions

THEOREM (Alon-Yahav, 2020; Topping et al., 2021):
    For expander graphs:
    Information from k-hop neighborhood compressed by factor:

    Compression ≥ exp(k) / d

    where d = feature dimension

Why this happens:
    1. Exponential path growth
    2. Fixed-size node representations
    3. Bottleneck edges (cuts)
    4. Poor graph expansion

Consequences:
    - Long-range dependencies lost
    - Path information discarded
    - Gradients vanish through bottlenecks
    - Deep GNNs don't help!

This module provides:
    - Expansion analysis
    - Bottleneck detection
    - Path counting
    - Curvature measures
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Any
from dataclasses import dataclass
from collections import deque
import scipy.linalg as la


@dataclass
class OversquashingAnalysis:
    """Results of oversquashing analysis.

    Attributes:
        expansion_profile: Expansion at each distance
        bottleneck_edges: Edges that are bottlenecks
        path_counts: Number of paths at each distance
        curvature: Ollivier-Ricci curvature
        effective_resistance: Electrical resistance between nodes
    """

    expansion_profile: List[float]
    bottleneck_edges: List[Tuple[int, int, float]]
    path_counts: Dict[int, List[int]]
    curvature: np.ndarray
    effective_resistance: np.ndarray


def analyze_oversquashing(
    nodes: List[int], edges: List[Tuple[int, int]], max_distance: int = 5
) -> OversquashingAnalysis:
    """Analyze oversquashing through combinatorial lens.

    Args:
        nodes: Graph nodes
        edges: Graph edges
        max_distance: Maximum distance to analyze

    Returns:
        Oversquashing analysis
    """
    n = len(nodes)

    # Build adjacency
    adjacency = np.zeros((n, n))
    adj_list = {v: [] for v in nodes}
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Compute expansion profile
    expansion_profile = compute_expansion_profile(nodes, edges, max_distance)

    # Detect bottleneck edges
    bottleneck_edges = detect_bottlenecks(adjacency, edges)

    # Count paths at each distance
    path_counts = count_paths_by_distance(adj_list, nodes, max_distance)

    # Compute curvature
    curvature = compute_ollivier_ricci_curvature(adjacency, edges)

    # Compute effective resistance
    effective_resistance = compute_effective_resistance(adjacency)

    return OversquashingAnalysis(
        expansion_profile=expansion_profile,
        bottleneck_edges=bottleneck_edges,
        path_counts=path_counts,
        curvature=curvature,
        effective_resistance=effective_resistance,
    )


def compute_expansion_profile(
    nodes: List[int], edges: List[Tuple[int, int]], max_distance: int
) -> List[float]:
    """Compute expansion at each distance.

    Expansion of set S:
        h(S) = |∂S| / min(|S|, |V\S|)

    where ∂S = edges leaving S

    Good expansion: h(S) ≥ α for all small sets S
    Poor expansion: ∃ S with h(S) << 1 (bottleneck!)

    Args:
        nodes: Graph nodes
        edges: Graph edges
        max_distance: Max distance

    Returns:
        Expansion at each distance
    """
    n = len(nodes)

    # Build adjacency list
    adj_list = {v: [] for v in nodes}
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    expansion_profile = []

    # For each node, compute expansion of k-hop neighborhood
    for k in range(1, max_distance + 1):
        expansions = []

        for start in nodes:
            # BFS to k hops
            visited = {start}
            current_level = {start}

            for _ in range(k):
                next_level = set()
                for v in current_level:
                    for u in adj_list[v]:
                        if u not in visited:
                            visited.add(u)
                            next_level.add(u)
                current_level = next_level

            # Count boundary edges
            S = visited
            boundary = 0
            for v in S:
                for u in adj_list[v]:
                    if u not in S:
                        boundary += 1

            # Expansion ratio
            size_S = len(S)
            size_complement = n - size_S
            denominator = min(size_S, size_complement)
            if denominator > 0:
                expansion = boundary / denominator
                expansions.append(expansion)

        expansion_profile.append(np.mean(expansions) if expansions else 0)

    return expansion_profile


def detect_bottlenecks(
    adjacency: np.ndarray, edges: List[Tuple[int, int]], threshold: float = 0.1
) -> List[Tuple[int, int, float]]:
    """Detect bottleneck edges (high betweenness, low local expansion).

    Bottleneck edge = many paths go through it

    Args:
        adjacency: Adjacency matrix
        edges: Graph edges
        threshold: Betweenness threshold

    Returns:
        List of (u, v, betweenness_centrality)
    """
    n = adjacency.shape[0]

    # Compute edge betweenness
    edge_betweenness = {}

    for edge in edges:
        # Remove edge and check connectivity impact
        u, v = edge

        # Simple heuristic: count shortest paths through edge
        # (Full betweenness is expensive; this is approximation)

        # For now, use degree product as proxy
        deg_u = np.sum(adjacency[u])
        deg_v = np.sum(adjacency[v])

        # High betweenness ≈ connects high-degree nodes
        betweenness = deg_u * deg_v

        edge_betweenness[edge] = betweenness

    # Normalize
    max_between = max(edge_betweenness.values()) if edge_betweenness else 1
    edge_betweenness = {k: v / max_between for k, v in edge_betweenness.items()}

    # Find bottlenecks
    bottlenecks = []
    for (u, v), between in edge_betweenness.items():
        if between > threshold:
            bottlenecks.append((u, v, between))

    return sorted(bottlenecks, key=lambda x: -x[2])


def count_paths_by_distance(
    adj_list: Dict[int, List[int]], nodes: List[int], max_distance: int
) -> Dict[int, List[int]]:
    """Count number of paths of each length from each node.

    EXPONENTIAL GROWTH demonstrates oversquashing:
        - Paths at distance k: O(d^k) where d = avg degree
        - Node features: O(d_hidden) dimensions
        - Compression: d^k → d_hidden (exponential!)

    Args:
        adj_list: Adjacency list
        nodes: Graph nodes
        max_distance: Maximum distance

    Returns:
        {node_id: [path_counts_at_distance]}
    """
    path_counts = {}

    for start in nodes:
        counts = []

        # Dynamic programming: count paths
        # dp[k][v] = number of k-length paths from start to v
        dp = [{start: 1}]

        for k in range(1, max_distance + 1):
            new_dp = {}
            for v, count in dp[-1].items():
                for u in adj_list[v]:
                    new_dp[u] = new_dp.get(u, 0) + count
            dp.append(new_dp)

            # Total paths at distance k
            total = sum(new_dp.values())
            counts.append(total)

        path_counts[start] = counts

    return path_counts


def compute_ollivier_ricci_curvature(
    adjacency: np.ndarray, edges: List[Tuple[int, int]]
) -> np.ndarray:
    """Compute Ollivier-Ricci curvature (discrete curvature).

    Negative curvature → tree-like, poor expansion, oversquashing!
    Positive curvature → mesh-like, good expansion

    Simplified approximation:
        κ(e) ≈ (common_neighbors) / (total_neighbors)

    Args:
        adjacency: Adjacency matrix
        edges: Graph edges

    Returns:
        Curvature matrix
    """
    n = adjacency.shape[0]
    curvature = np.zeros((n, n))

    for u, v in edges:
        # Find common neighbors
        neighbors_u = set(np.where(adjacency[u] > 0)[0])
        neighbors_v = set(np.where(adjacency[v] > 0)[0])

        common = len(neighbors_u & neighbors_v)
        total = len(neighbors_u | neighbors_v)

        # Curvature approximation
        if total > 0:
            curv = common / total
        else:
            curv = 0

        curvature[u, v] = curv
        curvature[v, u] = curv

    return curvature


def compute_effective_resistance(adjacency: np.ndarray) -> np.ndarray:
    """Compute effective resistance (electrical distance).

    High resistance = information must travel through bottleneck

    Mathematical definition:
        R_ij = (e_i - e_j)^T L^+ (e_i - e_j)

    where L^+ = Moore-Penrose pseudoinverse of Laplacian

    Args:
        adjacency: Adjacency matrix

    Returns:
        Resistance matrix
    """
    n = adjacency.shape[0]

    # Compute Laplacian
    D = np.diag(np.sum(adjacency, axis=1))
    L = D - adjacency

    # Pseudoinverse
    try:
        L_pinv = la.pinv(L)
    except:
        return np.zeros((n, n))

    # Effective resistance
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            e_diff = np.zeros(n)
            e_diff[i] = 1
            e_diff[j] = -1

            R[i, j] = e_diff.T @ L_pinv @ e_diff
            R[j, i] = R[i, j]

    return R


def mitigation_strategies() -> Dict[str, str]:
    """Strategies to mitigate oversquashing.

    Returns:
        Strategy descriptions
    """
    return {
        "graph_rewiring": """
            Add edges to improve expansion
            
            Methods:
            - Add edges based on curvature
            - Connect distant high-resistance nodes
            - Balance edge additions
            
            Goal: Reduce effective resistance
        """,
        "virtual_nodes": """
            Add virtual super-nodes
            Connect to clusters
            
            Effect: Bypass bottlenecks
            Tradeoff: Changes graph semantics
        """,
        "attention_mechanisms": """
            Learn to bypass bottlenecks
            
            Attention can weight long-range connections
            Adaptively find paths
            
            Tradeoff: Computational cost
        """,
        "graph_transformers": """
            Full attention (no graph structure)
            
            Eliminates structural bottlenecks
            
            Tradeoff: Loses graph inductive bias
        """,
    }
