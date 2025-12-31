"""Pure graph generators for experiments: ER, SBM, rings, lattices.

This module provides graph generators with careful documentation of:
1. What structural properties are preserved
2. What randomness is introduced
3. Expected graph statistics

All generators return (nodes, edges) tuples for consistency.

Mathematical properties documented:
- Degree distribution
- Clustering coefficient
- Diameter
- Spectral properties (where known)
"""

import random
from typing import List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class GraphStatistics:
    """Expected statistics for generated graphs.

    Used to document theoretical properties of graph models.
    """

    expected_edges: float
    expected_avg_degree: float
    expected_diameter: Optional[float] = None
    expected_clustering: Optional[float] = None
    spectral_gap_bound: Optional[float] = None


def erdos_renyi(
    n: int, p: float, seed: Optional[int] = None
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate Erdős-Rényi random graph G(n, p).

    Mathematical model:
        - n vertices
        - Each edge (i,j) exists independently with probability p

    Expected properties:
        - Number of edges: E[m] = p · n(n-1)/2
        - Average degree: E[deg] = p(n-1)
        - Degree distribution: Binomial(n-1, p) → Poisson(pn) for large n
        - Clustering coefficient: C = p (edges are independent)
        - Diameter: O(log n / log(pn)) with high probability

    Information encoded:
        - Random connectivity with uniform edge probability

    Limitations:
        - No community structure
        - No degree heterogeneity (beyond binomial variance)
        - Poor model for real-world networks

    Use cases:
        - Baseline/null model
        - Testing GNN behavior on random graphs
        - Understanding role of clustering vs. randomness

    Args:
        n: Number of nodes
        p: Edge probability (must be in [0, 1])
        seed: Random seed for reproducibility

    Returns:
        (nodes, edges) where nodes = [0, 1, ..., n-1]
        and edges is list of (i, j) pairs with i < j

    Complexity:
        Time: O(n²) - must check all pairs
        Space: O(m) where m ~ p·n²/2
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Edge probability p must be in [0, 1], got {p}")

    if seed is not None:
        random.seed(seed)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append((i, j))

    return list(range(n)), edges


def stochastic_block_model(
    block_sizes: List[int],
    intra_prob: float,
    inter_prob: float,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate Stochastic Block Model (SBM) graph.

    Mathematical model:
        - Nodes partitioned into K blocks of sizes n₁, ..., n_K
        - Edge (i,j) exists with probability:
            * p_intra if i,j in same block
            * p_inter if i,j in different blocks

    Expected properties:
        - Intra-block edges: Σ_k p_intra · n_k(n_k-1)/2
        - Inter-block edges: p_inter · Σ_{k≠l} n_k·n_l
        - Modularity: Q ~ (p_intra - p_inter) / p_avg (for balanced blocks)

    Information encoded:
        - Community structure
        - Block assignments (implicit in connectivity)

    Limitations:
        - All nodes in same block are statistically equivalent
        - Sharp block boundaries (real communities are fuzzy)
        - No hierarchical structure

    Use cases:
        - Testing community detection algorithms
        - Studying GNN expressivity on clustered graphs
        - Understanding pooling behavior on hierarchical structure

    Args:
        block_sizes: Number of nodes in each block
        intra_prob: Probability of edge within same block
        inter_prob: Probability of edge between different blocks
        seed: Random seed

    Returns:
        (nodes, edges) with implicit block structure

    Note:
        Node i belongs to block k if:
            Σ_{j=0}^{k-1} block_sizes[j] <= i < Σ_{j=0}^k block_sizes[j]
    """
    if seed is not None:
        random.seed(seed)

    n = sum(block_sizes)
    nodes = list(range(n))

    # Build block assignment
    block_assignment = []
    for block_id, size in enumerate(block_sizes):
        block_assignment.extend([block_id] * size)

    # Generate edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            same_block = block_assignment[i] == block_assignment[j]
            p = intra_prob if same_block else inter_prob

            if random.random() < p:
                edges.append((i, j))

    return nodes, edges


def ring_lattice(n: int, k: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate ring lattice (circular graph with k nearest neighbors).

    Mathematical model:
        - n nodes arranged in a circle
        - Each node i connected to nodes (i±1), (i±2), ..., (i±k/2) mod n

    Properties:
        - Regular graph: all nodes have degree k
        - Number of edges: m = nk/2
        - Diameter: ⌊n/(k+1)⌋
        - Clustering coefficient: C = 3(k-2)/(4(k-1)) → 3/4 as k → ∞
        - High clustering, high diameter (opposite of random graphs)

    Spectral properties:
        - Eigenvalues known exactly:
          λ_j = 2Σ_{r=1}^{k/2} cos(2πjr/n) for j=0,...,n-1
        - Spectral gap: λ₁ - λ₂ = O(1/n²) (slow mixing!)

    Information encoded:
        - Local regular structure
        - Geometric embedding (implicit circle)

    Use cases:
        - Baseline for small-world networks (before rewiring)
        - Testing oversmoothing (slow spectral gap → slow convergence)
        - Understanding locality in GNNs

    Args:
        n: Number of nodes
        k: Number of nearest neighbors (must be even)

    Returns:
        (nodes, edges) forming ring lattice
    """
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")
    if k >= n:
        raise ValueError(f"k={k} must be less than n={n}")

    nodes = list(range(n))
    edges = []

    for i in range(n):
        for r in range(1, k // 2 + 1):
            j = (i + r) % n
            if i < j:  # Avoid duplicates
                edges.append((i, j))

    return nodes, edges


def watts_strogatz(
    n: int, k: int, p_rewire: float, seed: Optional[int] = None
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate Watts-Strogatz small-world graph.

    Mathematical model:
        1. Start with ring lattice(n, k)
        2. For each edge (i, j):
           - With probability p_rewire, rewire to random node
           - Keep with probability 1 - p_rewire

    Properties as function of p_rewire:
        - p=0: Regular lattice (high clustering, high diameter)
        - p=1: Random graph (low clustering, low diameter)
        - p∈(0,1): Small-world (high clustering, LOW diameter!)

    Expected diameter:
        - O(log n) for p > 0 (even small p!)
        - "Six degrees of separation" phenomenon

    Clustering coefficient:
        - Decreases smoothly from lattice value to random graph value
        - Sweet spot: p ~ 0.01 gives clustering ≈ lattice, diameter ≈ random

    Information encoded:
        - Local structure (high clustering)
        - Long-range connections (low diameter)

    Use cases:
        - Modeling real-world networks (social, neural, etc.)
        - Testing GNN receptive field vs. diameter
        - Understanding expressivity limitations (WL cannot detect "small-world-ness")

    Args:
        n: Number of nodes
        k: Nearest neighbors in initial lattice
        p_rewire: Rewiring probability
        seed: Random seed

    Returns:
        (nodes, edges) forming small-world graph
    """
    if seed is not None:
        random.seed(seed)

    # Start with ring lattice
    nodes, edges = ring_lattice(n, k)

    # Rewire edges
    rewired_edges = []
    edge_set = set()

    for i, j in edges:
        if random.random() < p_rewire:
            # Rewire: choose new target for i
            attempts = 0
            while attempts < 100:  # Avoid infinite loop
                new_j = random.randint(0, n - 1)
                new_edge = tuple(sorted([i, new_j]))

                # Valid if: not self-loop, not duplicate
                if new_j != i and new_edge not in edge_set:
                    rewired_edges.append(new_edge)
                    edge_set.add(new_edge)
                    break
                attempts += 1
            else:
                # Couldn't rewire, keep original
                edge = tuple(sorted([i, j]))
                rewired_edges.append(edge)
                edge_set.add(edge)
        else:
            # Keep original edge
            edge = tuple(sorted([i, j]))
            rewired_edges.append(edge)
            edge_set.add(edge)

    return nodes, rewired_edges


def grid_2d(
    height: int, width: int, periodic: bool = False
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate 2D grid graph.

    Mathematical model:
        - Nodes: (i, j) for i ∈ {0, ..., height-1}, j ∈ {0, ..., width-1}
        - Edges: 4-neighbor connectivity (up, down, left, right)
        - If periodic: torus topology (wrap around)

    Properties:
        - Regular graph: degree 4 (2 on boundary if not periodic)
        - Diameter: height + width (Manhattan distance)
        - Clustering: 0 (no triangles in grid!)
        - Spectral gap: λ₁ - λ₂ = O(1/n) (slow mixing)

    Spectral properties:
        - Eigenvalues known exactly for periodic case:
          λ_{i,j} = 2[cos(2πi/h) + cos(2πj/w)] - 4
        - Graph Laplacian is discrete Laplacian operator

    Information encoded:
        - Spatial embedding in 2D
        - Regular local structure

    Limitations:
        - No triangles → clustering = 0
        - High diameter (diffusion is slow)

    Use cases:
        - Image/pixel graphs (GNNs on images)
        - Testing spatial convolution (CNN vs. GNN)
        - Understanding locality (K-hop = K-pixel radius)

    Args:
        height: Number of rows
        width: Number of columns
        periodic: If True, wrap around to form torus

    Returns:
        (nodes, edges) where node i represents (i // width, i % width)
    """
    n = height * width
    nodes = list(range(n))
    edges = []

    def to_idx(i: int, j: int) -> int:
        """Convert (row, col) to node index."""
        return i * width + j

    for i in range(height):
        for j in range(width):
            current = to_idx(i, j)

            # Right neighbor
            if j < width - 1:
                edges.append((current, to_idx(i, j + 1)))
            elif periodic:
                edges.append((current, to_idx(i, 0)))

            # Down neighbor
            if i < height - 1:
                edges.append((current, to_idx(i + 1, j)))
            elif periodic:
                edges.append((current, to_idx(0, j)))

    return nodes, edges


def complete_graph(n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate complete graph K_n.

    Mathematical model:
        - All (n choose 2) edges present

    Properties:
        - Degree: n-1 (every node connected to all others)
        - Edges: n(n-1)/2
        - Diameter: 1
        - Clustering: 1 (all triangles exist)
        - Adjacency matrix: A = J - I (all ones except diagonal)

    Spectral properties:
        - Eigenvalues of A: {n-1 (multiplicity 1), -1 (multiplicity n-1)}
        - Laplacian eigenvalues: {0, n, n, ..., n}
        - Spectral gap: maximal!

    Information encoded:
        - Complete connectivity (no structure)

    Information lost:
        - All graphs isomorphic to K_n are identical
        - WL test assigns same color to all nodes immediately

    Use cases:
        - Baseline: maximum connectivity
        - Testing GNN behavior when all nodes can communicate directly
        - Understanding over-smoothing (immediate convergence!)

    Args:
        n: Number of nodes

    Returns:
        (nodes, edges) forming complete graph
    """
    nodes = list(range(n))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return nodes, edges


def tree(depth: int, branching_factor: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate regular tree.

    Mathematical model:
        - Root node with `branching_factor` children
        - Each non-leaf node has `branching_factor` children
        - Tree has `depth` levels

    Properties:
        - Nodes: n = (b^{d+1} - 1) / (b - 1) for branching b, depth d
        - Edges: n - 1 (connected acyclic)
        - Diameter: 2·depth
        - No cycles: Tree is bipartite

    Spectral properties:
        - Laplacian has many small eigenvalues (slow mixing)
        - Spectral gap: O(1/depth²)
        - Poor expansion (bottleneck at root)

    Information encoded:
        - Hierarchical structure
        - Clear "levels"

    Limitations:
        - Severe bottleneck at root
        - Over-squashing: messages from leaves to root must pass through root
        - GNNs struggle with long-range dependencies in trees

    Use cases:
        - Testing over-squashing
        - Understanding bottlenecks in message passing
        - Hierarchical pooling experiments

    Args:
        depth: Depth of tree (root is level 0)
        branching_factor: Children per internal node

    Returns:
        (nodes, edges) forming regular tree
    """
    if branching_factor == 1:
        # Special case: path graph
        n = depth + 1
        nodes = list(range(n))
        edges = [(i, i + 1) for i in range(n - 1)]
        return nodes, edges

    # Compute number of nodes
    n = (branching_factor ** (depth + 1) - 1) // (branching_factor - 1)
    nodes = list(range(n))
    edges = []

    # Build tree level by level
    current_idx = 0
    for d in range(depth):
        level_size = branching_factor**d
        for _ in range(level_size):
            parent = current_idx
            current_idx += 1
            for _ in range(branching_factor):
                child = current_idx
                current_idx += 1
                if child < n:
                    edges.append((parent, child))

    return nodes, edges


def barabasi_albert(
    n: int, m: int, seed: Optional[int] = None
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate Barabási-Albert scale-free graph using preferential attachment.

    Mathematical model:
        - Start with m nodes fully connected
        - Add nodes one at a time
        - Each new node connects to m existing nodes
        - Probability of connecting to node i: P(i) ∝ degree(i)

    Properties:
        - Degree distribution: P(k) ~ k^{-γ} with γ ≈ 3 (power law!)
        - Average degree: 2m
        - Contains hubs (nodes with very high degree)
        - "Rich get richer" dynamics

    Information encoded:
        - Scale-free structure
        - Hub-and-spoke connectivity

    Limitations:
        - No community structure
        - Degree distribution is only asymptotically scale-free

    Use cases:
        - Modeling social networks, citation networks, web graph
        - Testing GNN behavior on heterogeneous degrees
        - Understanding importance of node degree normalization

    Args:
        n: Final number of nodes
        m: Number of edges for each new node (also size of initial clique)
        seed: Random seed

    Returns:
        (nodes, edges) forming scale-free graph
    """
    if seed is not None:
        random.seed(seed)

    if m < 1 or m >= n:
        raise ValueError(f"m must be in [1, n), got m={m}, n={n}")

    # Start with complete graph of m nodes
    nodes = list(range(n))
    edges = [(i, j) for i in range(m) for j in range(i + 1, m)]

    # Degree tracking for preferential attachment
    degree = [m - 1] * m + [0] * (n - m)

    # Add nodes one by one
    for new_node in range(m, n):
        # Choose m distinct targets using preferential attachment
        targets = set()
        total_degree = sum(degree[:new_node])

        while len(targets) < m:
            # Sample proportional to degree
            r = random.random() * total_degree
            cumsum = 0
            for target in range(new_node):
                cumsum += degree[target]
                if cumsum >= r:
                    targets.add(target)
                    break

        # Add edges to chosen targets
        for target in targets:
            edges.append((target, new_node))
            degree[target] += 1
            degree[new_node] += 1

    return nodes, edges
