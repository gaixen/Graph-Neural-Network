"""Shortest path algorithms: unweighted, Dijkstra, Bellman-Ford.

This module implements shortest path algorithms and connects them to GNN concepts:

1. **Unweighted shortest paths (BFS)**: K-hop neighborhoods
2. **Dijkstra's algorithm**: Non-negative weighted shortest paths
3. **Bellman-Ford**: Handles negative weights, detects negative cycles
4. **Floyd-Warshall**: All-pairs shortest paths

GNN connection:
    - Shortest path distance = minimum number of message-passing steps
    - Graph diameter = minimum GNN depth for global information flow
    - Bottlenecks in shortest paths → over-squashing in GNNs

Mathematical framework:
    - Distance metric: d(u, v) = length of shortest path
    - Triangle inequality: d(u, v) ≤ d(u, w) + d(w, v)
    - Diameter: max{d(u, v) : u, v ∈ V}
"""

import heapq
import numpy as np
from typing import Dict, Iterable, Tuple, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ShortestPathResult:
    """Result of shortest path computation.

    Attributes:
        distances: Distance from source to each node
        parents: Parent in shortest path tree
        has_negative_cycle: True if negative cycle detected (Bellman-Ford only)
    """

    distances: Dict[Any, float]
    parents: Dict[Any, Optional[Any]]
    has_negative_cycle: bool = False

    def reconstruct_path(self, target: Any) -> Optional[List[Any]]:
        """Reconstruct shortest path from source to target.

        Returns:
            Path as list of nodes, or None if no path exists
        """
        if target not in self.parents:
            return None

        if self.distances[target] == float("inf"):
            return None

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = self.parents.get(current)

        return list(reversed(path))


def dijkstra(
    nodes: Iterable[Any],
    edges: Iterable[Tuple[Any, Any, float]],
    source: Any,
    directed: bool = True,
) -> ShortestPathResult:
    """Dijkstra's algorithm for non-negative weighted shortest paths.

    Mathematical algorithm:
        Greedy selection: always expand closest unvisited node.
        Correctness requires: all edge weights ≥ 0.

    Optimality:
        When edge weights are non-negative, Dijkstra finds optimal paths.

    Information encoded:
        - Exact shortest path distances
        - Shortest path tree

    Information lost:
        - Alternative paths with same length
        - Paths that are not shortest

    GNN connection:
        - Distance d(s, v) = minimum hops in unweighted case
        - Weighted distance = importance of different paths
        - Can use as positional encoding: PE(v) = [d(v₁, v), ..., d(v_n, v)]

    Preconditions:
        - All edge weights ≥ 0 (otherwise Dijkstra is incorrect!)

    Args:
        nodes: Graph nodes
        edges: Weighted edges as (u, v, weight) tuples
        source: Starting node
        directed: If False, add reverse edges

    Returns:
        ShortestPathResult with distances and parent pointers

    Complexity:
        Time: O((|V| + |E|) log |V|) with binary heap
        Space: O(|V|)
    """
    # Build adjacency list
    adj = defaultdict(list)
    for u, v, w in edges:
        if w < 0:
            raise ValueError(
                f"Dijkstra requires non-negative weights, got edge ({u}, {v}, {w})"
            )
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))

    # Initialize distances
    dist = {v: float("inf") for v in nodes}
    dist[source] = 0.0
    parents = {source: None}

    # Priority queue: (distance, node)
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        # Skip if we've already found a better path
        if d > dist[u]:
            continue

        # Relax edges
        for v, w in adj[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                parents[v] = u
                heapq.heappush(pq, (new_dist, v))

    return ShortestPathResult(distances=dist, parents=parents)


def bellman_ford(
    nodes: Iterable[Any],
    edges: Iterable[Tuple[Any, Any, float]],
    source: Any,
    directed: bool = True,
) -> ShortestPathResult:
    """Bellman-Ford algorithm: handles negative weights, detects negative cycles.

    Mathematical algorithm:
        Dynamic programming: relax all edges |V| - 1 times.
        One more iteration detects negative cycles.

    Correctness:
        - Works with negative edge weights
        - Detects negative cycles (no shortest path exists!)
        - Slower than Dijkstra but more general

    Information encoded:
        - Shortest path distances (if no negative cycles)
        - Negative cycle detection

    Mathematical insight:
        Any simple path has at most |V| - 1 edges.
        After |V| - 1 iterations, all shortest paths found.
        If relaxation still helps, there's a negative cycle.

    GNN connection:
        - Negative cycles analogous to unstable message passing
        - Number of iterations = GNN depth
        - Relaxation = message update

    Args:
        nodes: Graph nodes
        edges: Weighted edges (can have negative weights)
        source: Starting node
        directed: If False, add reverse edges

    Returns:
        ShortestPathResult, with has_negative_cycle flag

    Complexity:
        Time: O(|V| · |E|)
        Space: O(|V|)
    """
    node_list = list(nodes)
    n = len(node_list)

    # Build edge list
    edge_list = list(edges)
    if not directed:
        edge_list += [(v, u, w) for u, v, w in edges]

    # Initialize distances
    dist = {v: float("inf") for v in node_list}
    dist[source] = 0.0
    parents = {source: None}

    # Relax edges |V| - 1 times
    for _ in range(n - 1):
        for u, v, w in edge_list:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parents[v] = u

    # Check for negative cycles
    has_negative_cycle = False
    for u, v, w in edge_list:
        if dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return ShortestPathResult(
        distances=dist, parents=parents, has_negative_cycle=has_negative_cycle
    )


def floyd_warshall(
    nodes: List[Any], edges: Iterable[Tuple[Any, Any, float]], directed: bool = True
) -> np.ndarray:
    """Floyd-Warshall: all-pairs shortest paths.

    Mathematical algorithm:
        Dynamic programming on intermediate vertices:
        d_k[i, j] = min path from i to j using only {1, ..., k} as intermediate

    Recurrence:
        d_k[i, j] = min(d_{k-1}[i, j], d_{k-1}[i, k] + d_{k-1}[k, j])

    Information encoded:
        - All pairwise distances in single matrix
        - Complete distance metric

    GNN connection:
        - Distance matrix can be used as positional encoding
        - Captures global graph structure
        - Expensive: O(n³) computation

    Output interpretation:
        Matrix D where D[i, j] = shortest distance from nodes[i] to nodes[j]

    Args:
        nodes: List of nodes (order matters for matrix indexing!)
        edges: Weighted edges
        directed: If False, add reverse edges

    Returns:
        Distance matrix as numpy array (n × n)

    Complexity:
        Time: O(|V|³)
        Space: O(|V|²)
    """
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize distance matrix
    D = np.full((n, n), float("inf"))
    np.fill_diagonal(D, 0.0)

    # Add edges
    for u, v, w in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        D[i, j] = min(D[i, j], w)  # Handle multiple edges
        if not directed:
            D[j, i] = min(D[j, i], w)

    # Floyd-Warshall iterations
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] = min(D[i, j], D[i, k] + D[k, j])

    return D


def compute_graph_diameter(
    nodes: List[Any], edges: Iterable[Tuple[Any, Any, float]], directed: bool = False
) -> float:
    """Compute graph diameter: maximum shortest path distance.

    Mathematical definition:
        diam(G) = max{d(u, v) : u, v ∈ V, path exists}

    GNN implication:
        - Need at least diam(G) layers for global information flow
        - Diameter = 1 (complete graph) → immediate mixing
        - Diameter = n (path graph) → slow propagation

    Information encoded:
        - Worst-case communication distance

    Args:
        nodes: Graph nodes
        edges: Edges (assume weight 1 if unweighted)
        directed: If False, use undirected graph

    Returns:
        Diameter (maximum finite distance), or inf if graph is disconnected
    """
    D = floyd_warshall(nodes, edges, directed=directed)

    # Find maximum finite distance
    finite_distances = D[D < float("inf")]

    if len(finite_distances) == 0:
        return float("inf")  # No paths exist

    return float(np.max(finite_distances))


def shortest_path_bottleneck_analysis(
    nodes: List[Any], edges: Iterable[Tuple[Any, Any, float]]
) -> Dict[Any, int]:
    """Analyze which nodes are bottlenecks in shortest paths.

    A node is a bottleneck if many shortest paths pass through it.

    Mathematical concept: Betweenness centrality
        BC(v) = Σ_{s≠v≠t} (# shortest paths s→t through v) / (# shortest paths s→t)

    GNN connection:
        - Bottlenecks → over-squashing
        - Information from many nodes compressed through bottleneck
        - Limited by bottleneck node's feature dimension

    This is a simplified version counting paths (not normalized).

    Args:
        nodes: Graph nodes
        edges: Graph edges

    Returns:
        Dictionary mapping each node to count of shortest paths through it
    """
    bottleneck_count = {node: 0 for node in nodes}

    # For each pair of nodes, count paths through intermediate nodes
    for source in nodes:
        result = dijkstra(nodes, edges, source, directed=False)

        for target in nodes:
            if source == target:
                continue

            # Reconstruct path
            path = result.reconstruct_path(target)
            if path and len(path) > 2:
                # Count intermediate nodes
                for intermediate in path[1:-1]:
                    bottleneck_count[intermediate] += 1

    return bottleneck_count
