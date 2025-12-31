"""Naïve and canonicalization routines for graph isomorphism.

This module addresses the graph isomorphism problem:
    Given graphs G₁ = (V₁, E₁) and G₂ = (V₂, E₂),
    does there exist a bijection φ: V₁ → V₂ such that
    (u, v) ∈ E₁ ⟺ (φ(u), φ(v)) ∈ E₂?

Complexity:
    - Unknown if in P or NP-complete!
    - Best known: quasi-polynomial time (Babai, 2016)
    - For most graphs, practical heuristics work well

GNN connection:
    - GNNs cannot solve graph isomorphism (Xu et al., 2019)
    - At best, GNN expressivity = Weisfeiler-Lehman test
    - WL test is incomplete for isomorphism

This module provides:
1. Naive brute-force (for small graphs)
2. Invariant-based filtering
3. Canonical form computation
"""

from itertools import permutations
from typing import Iterable, Tuple, Any, List, Set, Dict, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter


@dataclass
class GraphInvariants:
    """Graph invariants: properties preserved under isomorphism.

    Two isomorphic graphs MUST have identical invariants.
    Non-identical invariants → definitely not isomorphic.
    Identical invariants → maybe isomorphic (not sufficient!).

    Attributes:
        num_nodes: |V|
        num_edges: |E|
        degree_sequence: Sorted list of node degrees
        num_triangles: Number of 3-cycles
        num_connected_components: Connected components
        spectrum: Sorted eigenvalues of adjacency matrix
    """

    num_nodes: int
    num_edges: int
    degree_sequence: List[int]
    num_triangles: int
    num_connected_components: int = 1
    spectrum: Optional[List[float]] = None

    def __eq__(self, other: "GraphInvariants") -> bool:
        """Check if invariants match (necessary for isomorphism)."""
        if self.num_nodes != other.num_nodes:
            return False
        if self.num_edges != other.num_edges:
            return False
        if self.degree_sequence != other.degree_sequence:
            return False
        if self.num_triangles != other.num_triangles:
            return False
        return True


def compute_invariants(
    nodes: Iterable[Any], edges: Iterable[Tuple[Any, Any]]
) -> GraphInvariants:
    """Compute graph invariants for isomorphism filtering.

    Mathematical properties:
        All these values are isomorphism invariants:
        if G₁ ≅ G₂, then Inv(G₁) = Inv(G₂).

    Use case:
        Quick rejection: if invariants differ, graphs are not isomorphic.

    Information encoded:
        - Basic structural properties

    Limitations:
        - NOT sufficient for isomorphism
        - Many non-isomorphic graphs share invariants

    Args:
        nodes: Graph nodes
        edges: Graph edges

    Returns:
        GraphInvariants object
    """
    node_list = list(nodes)
    edge_list = list(edges)
    n = len(node_list)
    m = len(edge_list)

    # Compute degree sequence
    degree = Counter()
    for u, v in edge_list:
        degree[u] += 1
        degree[v] += 1
    degree_sequence = sorted(degree.values(), reverse=True)

    # Count triangles
    edge_set = set()
    for u, v in edge_list:
        edge_set.add((min(u, v), max(u, v)))

    num_triangles = 0
    for u, v in edge_list:
        # For each edge (u, v), count common neighbors
        for w in node_list:
            if w != u and w != v:
                e1 = (min(u, w), max(u, w))
                e2 = (min(v, w), max(v, w))
                if e1 in edge_set and e2 in edge_set:
                    num_triangles += 1
    num_triangles //= 3  # Each triangle counted 3 times

    return GraphInvariants(
        num_nodes=n,
        num_edges=m,
        degree_sequence=degree_sequence,
        num_triangles=num_triangles,
    )


def are_isomorphic(nodes1, edges1, nodes2, edges2) -> bool:
    """Naive brute-force graph isomorphism test.

    Algorithm:
        Try all n! permutations of node labels.
        For each permutation, check if edge sets match.

    Correctness:
        Complete: always finds isomorphism if it exists.

    Complexity:
        Time: O(n! · m) - INTRACTABLE for n > 10!
        Space: O(n + m)

    Use case:
        Only for tiny graphs (n ≤ 8).
        For larger graphs, use WL test or invariants.

    Information encoded:
        - Binary answer: isomorphic or not

    Information lost:
        - The actual isomorphism mapping (not returned)
        - Could be modified to return the mapping

    Args:
        nodes1, edges1: First graph
        nodes2, edges2: Second graph

    Returns:
        True if graphs are isomorphic, False otherwise

    Warning:
        DO NOT use for n > 10! Will hang.
    """
    # Quick reject: check basic invariants
    if len(nodes1) != len(nodes2) or len(edges1) != len(edges2):
        return False

    n = len(nodes1)
    if n > 10:
        raise ValueError(f"Naive isomorphism test too slow for n={n} > 10")

    nodes1 = list(nodes1)
    nodes2 = list(nodes2)
    e1 = set((min(u, v), max(u, v)) for u, v in edges1)

    # Try all permutations
    for perm in permutations(range(n)):
        mapping = {nodes2[i]: nodes1[perm[i]] for i in range(n)}
        mapped = set(
            (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
            for u, v in edges2
        )
        if mapped == e1:
            return True

    return False


def find_isomorphism(
    nodes1: List[Any],
    edges1: List[Tuple[Any, Any]],
    nodes2: List[Any],
    edges2: List[Tuple[Any, Any]],
) -> Optional[Dict[Any, Any]]:
    """Find isomorphism mapping if it exists.

    Returns the mapping φ: V₁ → V₂ such that:
        (u, v) ∈ E₁ ⟺ (φ(u), φ(v)) ∈ E₂

    Args:
        nodes1, edges1: First graph
        nodes2, edges2: Second graph

    Returns:
        Mapping dictionary, or None if no isomorphism exists
    """
    if len(nodes1) != len(nodes2) or len(edges1) != len(edges2):
        return None

    n = len(nodes1)
    if n > 10:
        raise ValueError(f"Naive isomorphism too slow for n={n} > 10")

    e1 = set((min(u, v), max(u, v)) for u, v in edges1)

    for perm in permutations(range(n)):
        mapping = {nodes2[i]: nodes1[perm[i]] for i in range(n)}
        mapped = set(
            (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
            for u, v in edges2
        )

        if mapped == e1:
            return mapping

    return None


def canonical_labeling(
    nodes: List[Any], edges: List[Tuple[Any, Any]], method: str = "degree"
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Compute canonical labeling of graph.

    Goal:
        Assign labels 0, 1, ..., n-1 to nodes such that
        isomorphic graphs get identical labelings.

    Mathematical ideal:
        A canonical form CF(G) such that:
        G₁ ≅ G₂ ⟺ CF(G₁) = CF(G₂)

    Reality:
        Perfect canonical form is as hard as graph isomorphism!
        We use heuristics that work for many graphs.

    Methods:
        - 'degree': Sort by degree (fails for regular graphs)
        - 'degree_then_neighbors': Refine by neighbor degrees

    Limitations:
        - Not a true canonical form (can fail)
        - Good enough for many practical cases
        - Use WL test for better expressivity

    Args:
        nodes: Graph nodes
        edges: Graph edges
        method: Labeling heuristic

    Returns:
        (canonical_nodes, canonical_edges) with integer labels
    """
    # Compute degree sequence
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    if method == "degree":
        # Sort nodes by degree (descending), break ties arbitrarily
        sorted_nodes = sorted(nodes, key=lambda v: (-degree[v], str(v)))
    elif method == "degree_then_neighbors":
        # More sophisticated: sort by (degree, sum of neighbor degrees)
        neighbor_degree_sum = {}
        adj = {v: [] for v in nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for v in nodes:
            neighbor_degree_sum[v] = sum(degree[u] for u in adj[v])

        sorted_nodes = sorted(
            nodes, key=lambda v: (-degree[v], -neighbor_degree_sum[v], str(v))
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create mapping
    mapping = {node: i for i, node in enumerate(sorted_nodes)}

    # Relabel edges
    canonical_edges = sorted(
        [(min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) for u, v in edges]
    )

    return list(range(len(nodes))), canonical_edges


def automorphism_group_size_lower_bound(
    nodes: List[Any], edges: List[Tuple[Any, Any]]
) -> int:
    """Compute lower bound on automorphism group size.

    Automorphism:
        Isomorphism from graph to itself.
        Forms a group under composition.

    Information:
        |Aut(G)| measures graph symmetry.
        - |Aut(K_n)| = n! (complete graph: all permutations)
        - |Aut(path)| = 2 (flip)
        - |Aut(cycle)| = 2n (rotations + reflection)

    GNN connection:
        Nodes in same orbit of Aut(G) receive identical embeddings
        from permutation-equivariant GNNs.

    This function returns a LOWER BOUND (easy to compute).
    Exact automorphism group requires graph isomorphism machinery.

    Method:
        Count nodes with identical neighborhoods.

    Args:
        nodes: Graph nodes
        edges: Graph edges

    Returns:
        Lower bound on |Aut(G)|
    """
    # Build adjacency lists
    adj = {v: set() for v in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Compute degree and neighbor signature
    signatures = {}
    for v in nodes:
        degree = len(adj[v])
        neighbor_degrees = tuple(sorted(len(adj[u]) for u in adj[v]))
        signatures[v] = (degree, neighbor_degrees)

    # Count duplicates
    signature_counts = Counter(signatures.values())

    # Lower bound: factorial of each count
    import math

    lower_bound = 1
    for count in signature_counts.values():
        if count > 1:
            lower_bound *= math.factorial(count)

    return lower_bound
