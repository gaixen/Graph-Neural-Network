"""Matrix representations of graphs and documentation on information loss.

This module constructs all standard graph matrices and explicitly documents:
1. What information each matrix encodes
2. What information is lost in the transformation
3. Which assumptions are implicit (self-loops, normalization, etc.)
4. How choices affect spectral properties and operator semantics

Critical for GNN design: understanding what adjacency/Laplacian normalization does.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
import warnings


class LaplacianType(Enum):
    """Types of graph Laplacian normalization."""

    UNNORMALIZED = "unnormalized"  # L = D - A
    SYMMETRIC = "symmetric"  # L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2}AD^{-1/2}
    RANDOM_WALK = "random_walk"  # L_rw = D^{-1}L = I - D^{-1}A


@dataclass
class GraphMatrices:
    """Container for graph matrix representations.

    Attributes:
        adjacency: Adjacency matrix A
        degree: Degree matrix D (diagonal)
        laplacian: Graph Laplacian (type specified by laplacian_type)
        incidence: Incidence matrix (if computed)
        laplacian_type: Which Laplacian normalization was used

    Information preserved/lost:
        - A preserves connectivity but loses edge attributes (if not weighted)
        - D loses edge targets (only total degree)
        - L encodes diffusion operator
        - Incidence preserves edge-node relationships but requires orientation choice
    """

    adjacency: np.ndarray
    degree: np.ndarray
    laplacian: np.ndarray
    laplacian_type: LaplacianType
    incidence: Optional[np.ndarray] = None


def adjacency_matrix(
    nodes: List,
    edges: List[Tuple[int, int]],
    weights: Optional[List[float]] = None,
    directed: bool = False,
) -> np.ndarray:
    """Construct adjacency matrix A from edge list.

    Matrix definition:
        A_ij = weight of edge (i,j) if edge exists, else 0

    For undirected graphs:
        A_ij = A_ji (symmetric)

    Parameters:
        nodes: List of node IDs
        edges: List of (source, target) pairs
        weights: Optional edge weights (default: all 1.0)
        directed: If False, symmetrize the matrix

    Returns:
        Adjacency matrix A ∈ ℝ^{n×n}

    Information encoded:
        - Connectivity structure (which nodes are adjacent)
        - Edge weights (if provided)

    Information lost:
        - Edge attributes beyond weight
        - Original edge identities (multiple edges collapsed)
        - For undirected: cannot distinguish (u,v) from (v,u)

    Implicit assumptions:
        - No self-loops unless explicitly in edge list
        - For weighted: weights are on same scale
        - Nodes indexed 0 to n-1 (or mapping provided)
    """
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    A = np.zeros((n, n), dtype=float)

    if weights is None:
        weights = [1.0] * len(edges)

    for (u, v), w in zip(edges, weights):
        i, j = node_to_idx[u], node_to_idx[v]
        A[i, j] = w

        if not directed:
            A[j, i] = w  # Symmetrize

    return A


def degree_matrix(A: np.ndarray, out_degree: bool = True) -> np.ndarray:
    """Construct degree matrix D from adjacency A.

    For undirected graphs:
        D_ii = deg(i) = Σ_j A_ij (out-degree = in-degree)

    For directed graphs:
        D_ii = deg_out(i) = Σ_j A_ij (default)
        or D_ii = deg_in(i) = Σ_j A_ji (if out_degree=False)

    Parameters:
        A: Adjacency matrix
        out_degree: Use out-degree (True) or in-degree (False) for directed graphs

    Returns:
        Diagonal degree matrix D ∈ ℝ^{n×n}

    Information encoded:
        - Node degrees (total edge count per node)

    Information lost:
        - Which neighbors (only count)
        - Edge weights are summed (distribution lost)

    Spectral impact:
        - D controls diffusion rate: high degree → fast mixing
        - D^{-1/2}AD^{-1/2} normalizes spectrum to [-1, 1]
    """
    if out_degree:
        degrees = np.sum(A, axis=1)  # Row sums
    else:
        degrees = np.sum(A, axis=0)  # Column sums

    return np.diag(degrees)


def laplacian_matrix(
    A: np.ndarray, laplacian_type: LaplacianType = LaplacianType.UNNORMALIZED
) -> np.ndarray:
    """Construct graph Laplacian from adjacency matrix.

    Types:

    1. Unnormalized: L = D - A
        - Eigenvalues: 0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1} ≤ 2·max_degree
        - λ_1 = 0 iff disconnected (algebraic connectivity)
        - Not normalized by degree

    2. Symmetric normalized: L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2}AD^{-1/2}
        - Eigenvalues in [0, 2]
        - Most common in spectral GNNs (GCN uses I - L_sym)
        - Preserves symmetry → real eigenvalues

    3. Random walk: L_rw = D^{-1} L = I - D^{-1}A
        - Models random walk transition probabilities
        - Not symmetric (unless graph is regular)
        - Used in PageRank-style diffusions

    Parameters:
        A: Adjacency matrix
        laplacian_type: Which normalization to use

    Returns:
        Graph Laplacian matrix

    Critical for GNNs:
        - Repeated application of I - L diffuses signals
        - Normalization choice determines eigenvalue spectrum
        - Spectrum controls oversmoothing rate

    Information preserved:
        - Connectivity (which nodes are linked)
        - Relative diffusion rates between neighbors

    Information lost:
        - Absolute edge weights (normalized away)
        - Directionality (in symmetric versions)

    Assumptions:
        - No isolated nodes (degree 0 causes division by zero)
        - For L_sym: assumes graph is undirected or we symmetrize
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # Check for isolated nodes
    if np.any(degrees == 0):
        warnings.warn(
            "Graph contains isolated nodes (degree 0). "
            "Normalized Laplacians will have NaN/Inf values."
        )

    if laplacian_type == LaplacianType.UNNORMALIZED:
        return D - A

    elif laplacian_type == LaplacianType.SYMMETRIC:
        # D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2}AD^{-1/2}
        D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d > 0 else 0.0 for d in degrees])
        return np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt

    elif laplacian_type == LaplacianType.RANDOM_WALK:
        # D^{-1}(D - A) = I - D^{-1}A
        D_inv = np.diag([1.0 / d if d > 0 else 0.0 for d in degrees])
        return np.eye(len(A)) - D_inv @ A

    else:
        raise ValueError(f"Unknown Laplacian type: {laplacian_type}")


def normalized_adjacency(
    A: np.ndarray, symmetric: bool = True, add_self_loops: bool = False
) -> np.ndarray:
    """Construct normalized adjacency for graph convolutions.

    Symmetric normalization (GCN-style):
        Ã = D^{-1/2} A D^{-1/2}

    Row normalization (Random walk):
        Ã = D^{-1} A

    Parameters:
        A: Adjacency matrix
        symmetric: Use symmetric normalization (True) or row normalization (False)
        add_self_loops: Add identity before normalization (Ã ← A + I)

    Returns:
        Normalized adjacency matrix

    GCN usage:
        H^{(l+1)} = σ(Ã H^{(l)} W^{(l)})
    where Ã = D^{-1/2}(A+I)D^{-1/2}

    Spectral properties:
        - Symmetric: eigenvalues in [-1, 1], eigenvectors orthogonal
        - Row-normalized: eigenvalues in complex unit disk
        - Self-loops prevent degree 0 issues

    Oversmoothing mechanism:
        - Repeated application: H^{(k)} ≈ constant vector for large k
        - Rate depends on spectral gap of Ã
        - Self-loops slow down but don't prevent convergence
    """
    if add_self_loops:
        A = A + np.eye(len(A))

    degrees = np.sum(A, axis=1)

    if symmetric:
        D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d > 0 else 0.0 for d in degrees])
        return D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D_inv = np.diag([1.0 / d if d > 0 else 0.0 for d in degrees])
        return D_inv @ A


def incidence_matrix(
    num_nodes: int, edges: List[Tuple[int, int]], oriented: bool = True
) -> np.ndarray:
    """Construct incidence matrix B relating nodes and edges.

    For oriented graphs:
        B_ie = +1 if node i is target of edge e
        B_ie = -1 if node i is source of edge e
        B_ie = 0 otherwise

    For unoriented graphs (arbitrary orientation):
        B_ie ∈ {-1, 0, +1} but orientation choice is arbitrary

    Parameters:
        num_nodes: Number of nodes n
        edges: List of directed edges
        oriented: Respect edge direction (True) or use arbitrary orientation (False)

    Returns:
        Incidence matrix B ∈ ℝ^{n × m} where m = |E|

    Properties:
        - B B^T = L (unnormalized Laplacian, up to sign)
        - Rank(B) = n - c where c is number of connected components
        - Null space of B^T gives edge flows (Kirchhoff's law)

    Information encoded:
        - Edge-node relationships
        - Edge orientations (if oriented=True)

    Information lost:
        - Edge weights (unless extended to weighted incidence)
        - Cannot distinguish parallel edges (need multigraph representation)

    Note:
        Rarely used in GNNs but important for understanding Laplacian
        and for edge-based message passing.
    """
    num_edges = len(edges)
    B = np.zeros((num_nodes, num_edges))

    for e, (i, j) in enumerate(edges):
        B[i, e] = -1  # Source
        B[j, e] = +1  # Target

    if not oriented:
        # Arbitrary orientation: just ensure consistent sign
        pass

    return B


def build_all_matrices(
    nodes: List,
    edges: List[Tuple[int, int]],
    weights: Optional[List[float]] = None,
    directed: bool = False,
    laplacian_type: LaplacianType = LaplacianType.SYMMETRIC,
) -> GraphMatrices:
    """Construct complete set of graph matrices.

    Parameters:
        nodes: Node list
        edges: Edge list
        weights: Optional edge weights
        directed: Whether graph is directed
        laplacian_type: Which Laplacian normalization

    Returns:
        GraphMatrices object with all standard matrices

    Usage:
        matrices = build_all_matrices(nodes, edges)
        A = matrices.adjacency
        L = matrices.laplacian
    """
    A = adjacency_matrix(nodes, edges, weights, directed)
    D = degree_matrix(A)
    L = laplacian_matrix(A, laplacian_type)
    B = incidence_matrix(len(nodes), edges) if not directed else None

    return GraphMatrices(
        adjacency=A, degree=D, laplacian=L, laplacian_type=laplacian_type, incidence=B
    )
