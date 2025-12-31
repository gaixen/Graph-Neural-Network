"""Rewiring strategies with theoretical guarantees and failure modes.

GRAPH REWIRING = modifying graph structure to improve GNN performance

Motivation:
    - Original graph may have poor properties for GNNs
    - Bottlenecks cause oversquashing
    - Poor expansion causes oversmoothing
    - Long-range dependencies missing

Rewiring strategies:
    1. **Add edges**: Connect distant nodes
    2. **Remove edges**: Eliminate redundant connections
    3. **Rewire**: Change edge endpoints
    4. **Add virtual nodes**: Create shortcut hubs

Goals:
    - Improve spectral gap (better expansion)
    - Reduce diameter (shorter paths)
    - Balance degree distribution
    - Preserve semantic information

Danger:
    Rewiring can BREAK the graph!
    - Lose domain-specific structure
    - Change semantic meaning
    - Introduce spurious correlations
    - Violate problem constraints

This module explores:
    - Curvature-based rewiring
    - Spectral rewiring
    - Diffusion-based rewiring
    - Theoretical guarantees
    - Failure modes
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Any
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class RewiringResult:
    """Result of graph rewiring.

    Attributes:
        new_edges: Added edges
        removed_edges: Removed edges
        spectral_gap_before: Spectral gap before rewiring
        spectral_gap_after: Spectral gap after rewiring
        diameter_before: Diameter before
        diameter_after: Diameter after
        metrics: Various graph metrics
    """

    new_edges: List[Tuple[int, int]]
    removed_edges: List[Tuple[int, int]]
    spectral_gap_before: float
    spectral_gap_after: float
    diameter_before: int
    diameter_after: int
    metrics: Dict[str, Any]


def spectral_gap_rewiring(
    adjacency: np.ndarray,
    edges: List[Tuple[int, int]],
    num_edges_to_add: int,
    method: str = "fiedler",
) -> RewiringResult:
    """Rewire to maximize spectral gap.

    Spectral gap = λ_2 - λ_n (Fiedler value - smallest eigenvalue)

    Larger gap → better expansion → less oversmoothing

    Strategy:
        Add edges that increase λ_2 (connectivity)
        Remove edges that don't contribute

    Args:
        adjacency: Current adjacency matrix
        edges: Current edges
        num_edges_to_add: Budget of edges to add
        method: 'fiedler' or 'resistance'

    Returns:
        Rewiring result
    """
    n = adjacency.shape[0]

    # Compute current Laplacian
    D = np.diag(np.sum(adjacency, axis=1))
    L = D - adjacency

    eigvals_before = np.sort(la.eigvalsh(L))
    spectral_gap_before = (
        eigvals_before[1] - eigvals_before[0] if len(eigvals_before) > 1 else 0
    )

    # Diameter (approximate via BFS)
    diameter_before = compute_diameter_approximate(adjacency)

    # Add edges to maximize spectral gap
    new_edges = []

    if method == "fiedler":
        # Compute Fiedler vector (eigenvector of λ_2)
        eigvals, eigvecs = la.eigh(L)
        fiedler_vec = eigvecs[:, 1]  # Second smallest eigenvalue

        # Connect nodes with large |v_i - v_j| (opposite sides of cut)
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] == 0:  # Not already connected
                    gap = abs(fiedler_vec[i] - fiedler_vec[j])
                    candidates.append((i, j, gap))

        # Sort by gap (largest first)
        candidates.sort(key=lambda x: -x[2])

        # Add top edges
        for i, j, _ in candidates[:num_edges_to_add]:
            new_edges.append((i, j))
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    # Recompute spectral gap
    D_new = np.diag(np.sum(adjacency, axis=1))
    L_new = D_new - adjacency
    eigvals_after = np.sort(la.eigvalsh(L_new))
    spectral_gap_after = (
        eigvals_after[1] - eigvals_after[0] if len(eigvals_after) > 1 else 0
    )

    diameter_after = compute_diameter_approximate(adjacency)

    return RewiringResult(
        new_edges=new_edges,
        removed_edges=[],
        spectral_gap_before=spectral_gap_before,
        spectral_gap_after=spectral_gap_after,
        diameter_before=diameter_before,
        diameter_after=diameter_after,
        metrics={},
    )


def curvature_based_rewiring(
    adjacency: np.ndarray, edges: List[Tuple[int, int]], num_edges_to_add: int
) -> RewiringResult:
    """Rewire based on Ricci curvature.

    Negative curvature = tree-like, poor expansion
    Add edges to negative-curvature regions

    Ollivier-Ricci curvature:
        κ(e) ≈ #(common neighbors) / #(total neighbors)

    Strategy:
        Add edges where curvature is most negative

    Args:
        adjacency: Adjacency matrix
        edges: Current edges
        num_edges_to_add: Budget

    Returns:
        Rewiring result
    """
    n = adjacency.shape[0]

    # Compute curvature for each edge
    curvatures = {}
    for u, v in edges:
        neighbors_u = set(np.where(adjacency[u] > 0)[0])
        neighbors_v = set(np.where(adjacency[v] > 0)[0])
        common = len(neighbors_u & neighbors_v)
        total = len(neighbors_u | neighbors_v)
        curv = common / total if total > 0 else 0
        curvatures[(u, v)] = curv

    # Find regions with negative curvature
    # (Here: low curvature = need more connections)

    # Add edges to low-curvature regions
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0:
                # Estimate curvature if this edge existed
                neighbors_i = set(np.where(adjacency[i] > 0)[0])
                neighbors_j = set(np.where(adjacency[j] > 0)[0])
                common = len(neighbors_i & neighbors_j)
                total = len(neighbors_i | neighbors_j)
                potential_curv = common / total if total > 0 else 0

                # Prefer connecting nodes with shared context
                candidates.append((i, j, potential_curv))

    # Sort by curvature (add where curvature would be positive)
    candidates.sort(key=lambda x: -x[2])

    new_edges = []
    for i, j, _ in candidates[:num_edges_to_add]:
        new_edges.append((i, j))
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return RewiringResult(
        new_edges=new_edges,
        removed_edges=[],
        spectral_gap_before=0,
        spectral_gap_after=0,
        diameter_before=0,
        diameter_after=0,
        metrics={"curvatures": curvatures},
    )


def diffusion_based_rewiring(
    adjacency: np.ndarray, num_steps: int = 3, threshold: float = 0.01
) -> RewiringResult:
    """Rewire based on diffusion distances.

    Diffusion: Nodes that are close in diffusion metric should be connected

    Diffusion distance:
        After k steps of random walk,
        probability distributions similar → nodes functionally close

    Strategy:
        1. Compute k-step random walk probabilities
        2. Connect nodes with high probability of co-visitation
        3. Remove edges with low probability

    Args:
        adjacency: Adjacency matrix
        num_steps: Diffusion steps
        threshold: Connection threshold

    Returns:
        Rewiring result
    """
    n = adjacency.shape[0]

    # Compute transition matrix (random walk)
    D = np.diag(np.sum(adjacency, axis=1))
    D_inv = np.linalg.inv(D + 1e-10 * np.eye(n))
    P = D_inv @ adjacency

    # k-step transition probabilities
    P_k = np.linalg.matrix_power(P, num_steps)

    # Add edges where P_k is high but edge doesn't exist
    new_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0 and P_k[i, j] > threshold:
                new_edges.append((i, j))
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    # Remove edges where P_k is low
    removed_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0 and P_k[i, j] < threshold / 10:
                removed_edges.append((i, j))
                adjacency[i, j] = 0
                adjacency[j, i] = 0

    return RewiringResult(
        new_edges=new_edges,
        removed_edges=removed_edges,
        spectral_gap_before=0,
        spectral_gap_after=0,
        diameter_before=0,
        diameter_after=0,
        metrics={"diffusion_matrix": P_k},
    )


def compute_diameter_approximate(adjacency: np.ndarray, sample_size: int = 10) -> int:
    """Approximate graph diameter via BFS sampling.

    Args:
        adjacency: Adjacency matrix
        sample_size: Number of nodes to sample

    Returns:
        Approximate diameter
    """
    n = adjacency.shape[0]
    max_dist = 0

    # Sample nodes
    sampled = np.random.choice(n, min(sample_size, n), replace=False)

    for start in sampled:
        # BFS
        visited = {start}
        queue = [(start, 0)]
        local_max = 0

        while queue:
            v, dist = queue.pop(0)
            local_max = max(local_max, dist)

            for u in range(n):
                if adjacency[v, u] > 0 and u not in visited:
                    visited.add(u)
                    queue.append((u, dist + 1))

        max_dist = max(max_dist, local_max)

    return max_dist


def theoretical_guarantees() -> Dict[str, str]:
    """Theoretical guarantees of rewiring methods.

    Returns:
        Guarantees
    """
    return {
        "spectral_gap": """
            THEOREM: Adding k edges can increase λ_2 by at most O(k)
            
            But: Optimal edge placement is NP-hard!
            
            Heuristic: Fiedler vector gives good approximation
        """,
        "curvature": """
            THEOREM: Positive curvature → better expansion
            
            Ricci flow increases curvature over time
            Converges to more uniform geometry
            
            But: May destroy semantic structure!
        """,
        "diffusion": """
            THEOREM: Diffusion distance = effective graph metric
            
            Captures functional proximity
            Robust to noise
            
            But: Doesn't preserve local structure
        """,
    }


def failure_modes() -> List[str]:
    """Common failure modes of rewiring.

    Returns:
        Failure cases
    """
    return [
        "Over-rewiring: Too many edges → complete graph → loses structure",
        "Breaking semantics: Connecting incompatible nodes (e.g., wrong molecule bonds)",
        "Heterophily: Adding edges between dissimilar nodes can hurt performance",
        "Computational cost: Rewiring can be expensive for large graphs",
        "Overfitting: Rewiring on training data may not generalize",
        "Loss of interpretability: Rewired graph no longer matches domain",
    ]
