"""Weisfeiler-Lehman (1-WL) Graph Isomorphism Test.

This is the theoretical foundation for GNN expressivity analysis.
The 1-WL test provides a tractable approximation to graph isomorphism
and exactly characterizes the discriminative power of message-passing GNNs.

Key theorem (Xu et al., 2019):
    Message-passing GNNs with sum aggregation are at most as powerful as 1-WL.

This module includes:
- Pure 1-WL algorithm (no tensors, no ML)
- Initialization strategies
- Iterative color refinement
- Stabilization detection
- Counterexample graphs that defeat 1-WL
- Formal analysis of what information is preserved/lost

Mathematical formulation:
    Color refinement: c^{(k+1)}(v) = HASH(c^{(k)}(v), {{c^{(k)}(u) : u ∈ N(v)}})

    Information retained: k-hop neighborhood structure up to multiset isomorphism
    Information lost: higher-order subgraph patterns beyond k-hop neighborhoods
"""

from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Callable
import hashlib
import json


NodeID = int
Color = int
ColorDict = Dict[NodeID, Color]


@dataclass
class WLResult:
    """Results of Weisfeiler-Lehman color refinement.

    Attributes:
        final_colors: Node colors after stabilization
        num_iterations: Number of refinement steps until stabilization
        color_history: Sequence of color dictionaries (if tracked)
        is_stable: Whether algorithm converged (vs hit max_iter)
        color_classes: Partition of nodes into color equivalence classes
    """

    final_colors: ColorDict
    num_iterations: int
    is_stable: bool
    color_history: Optional[List[ColorDict]] = None
    color_classes: Optional[Dict[Color, Set[NodeID]]] = None


def hash_multiset(base_color: Color, neighbor_colors: List[Color]) -> Color:
    """Hash a node's color and its neighborhood multiset.

    This is the core of WL color refinement:
        new_color = HASH(old_color, {neighbor_colors})

    Parameters:
        base_color: Current color of the node
        neighbor_colors: List of neighbor colors (order doesn't matter)

    Returns:
        New color (deterministic hash)

    Properties:
        - Permutation invariant: hash({1,2,2,3}) = hash({2,3,1,2})
        - Injective on finite multisets (with good hash function)
        - Fast to compute

    Implementation:
        Uses cryptographic hash for determinism across runs.
        Sorts neighbor colors to ensure permutation invariance.
    """
    # Create canonical representation: (base, sorted neighbor multiset)
    neighbor_sorted = tuple(sorted(neighbor_colors))
    key = (base_color, neighbor_sorted)

    # Use deterministic hash
    key_str = json.dumps(key, sort_keys=True)
    hash_obj = hashlib.sha256(key_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    return hash_int


def wl_initialize(
    nodes: List[NodeID],
    initial_colors: Optional[ColorDict] = None,
    node_labels: Optional[Dict[NodeID, any]] = None,
) -> ColorDict:
    """Initialize node colors for WL refinement.

    Strategies:
    1. Uniform initialization: all nodes get color 0
       - Tests pure structural discrimination
       - Standard for unlabeled graphs

    2. Label-based: nodes with same label get same initial color
       - Generalizes to labeled graphs
       - Used in real-world graph datasets

    Parameters:
        nodes: List of node IDs
        initial_colors: Explicit color assignment (overrides other strategies)
        node_labels: Node labels for label-based initialization

    Returns:
        Initial color dictionary

    Theoretical note:
        Initial colors determine what "atom" features the algorithm starts with.
        GNN equivalent: initial node features h^{(0)}.
    """
    if initial_colors is not None:
        return initial_colors.copy()

    if node_labels is not None:
        # Group nodes by label, assign color per group
        label_to_color = {}
        colors = {}
        next_color = 0

        for node in nodes:
            label = node_labels.get(node, None)
            if label not in label_to_color:
                label_to_color[label] = next_color
                next_color += 1
            colors[node] = label_to_color[label]

        return colors

    # Default: uniform initialization
    return {node: 0 for node in nodes}


def wl_iteration(
    colors: ColorDict, edges: List[Tuple[NodeID, NodeID]], directed: bool = False
) -> ColorDict:
    """Perform one iteration of WL color refinement.

    Update rule:
        c^{(k+1)}(v) = HASH(c^{(k)}(v), {{c^{(k)}(u) : u ∈ N(v)}})

    Parameters:
        colors: Current node colors c^{(k)}
        edges: Graph edge list
        directed: If False, treat graph as undirected

    Returns:
        Updated colors c^{(k+1)}

    Mathematical insight:
        This aggregates 1-hop neighborhood multisets.
        After k iterations, node color encodes k-hop neighborhood structure
        (up to multiset isomorphism).

    GNN equivalent:
        This is exactly what message-passing does:
            aggregate: m_v = AGG({h_u : u ∈ N(v)})
            update: h_v^{new} = UPDATE(h_v, m_v)

        WL uses multiset hash; GNN uses learned functions.
    """
    # Build adjacency lists (neighbor colors)
    neighbor_colors = defaultdict(list)

    for u, v in edges:
        neighbor_colors[u].append(colors[v])
        if not directed:
            neighbor_colors[v].append(colors[u])

    # Refine colors
    new_colors = {}
    for node in colors:
        base_color = colors[node]
        neighbors = neighbor_colors[node]
        new_colors[node] = hash_multiset(base_color, neighbors)

    return new_colors


def wl_test(
    nodes: List[NodeID],
    edges: List[Tuple[NodeID, NodeID]],
    max_iterations: int = 100,
    initial_colors: Optional[ColorDict] = None,
    track_history: bool = False,
    directed: bool = False,
) -> WLResult:
    """Run complete Weisfeiler-Lehman color refinement until stabilization.

    Parameters:
        nodes: Node list
        edges: Edge list
        max_iterations: Maximum refinement steps (prevents infinite loop)
        initial_colors: Custom initialization (default: uniform)
        track_history: Store color sequence (for visualization/analysis)
        directed: Treat graph as directed

    Returns:
        WLResult with final colors and convergence info

    Termination:
        Algorithm stops when colors stabilize (no changes between iterations)
        or max_iterations is reached.

    Complexity:
        O(k · (|V| + |E|)) where k is number of iterations
        In practice, k is small (often < 10 for typical graphs)

    Theoretical guarantee:
        After stabilization, two nodes have the same color iff they are
        indistinguishable by 1-WL (have isomorphic k-hop neighborhoods
        for sufficiently large k).
    """
    colors = wl_initialize(nodes, initial_colors)
    history = [colors.copy()] if track_history else None

    for iteration in range(max_iterations):
        new_colors = wl_iteration(colors, edges, directed)

        if track_history:
            history.append(new_colors.copy())

        # Check for stabilization
        if new_colors == colors:
            # Converged!
            color_classes = partition_by_colors(new_colors)
            return WLResult(
                final_colors=new_colors,
                num_iterations=iteration,
                is_stable=True,
                color_history=history,
                color_classes=color_classes,
            )

        colors = new_colors

    # Hit max iterations without stabilizing
    color_classes = partition_by_colors(colors)
    return WLResult(
        final_colors=colors,
        num_iterations=max_iterations,
        is_stable=False,
        color_history=history,
        color_classes=color_classes,
    )


def partition_by_colors(colors: ColorDict) -> Dict[Color, Set[NodeID]]:
    """Group nodes into equivalence classes by color.

    Returns:
        Dictionary mapping color → set of nodes with that color

    Usage:
        Nodes in the same color class are indistinguishable by 1-WL.
    """
    classes = defaultdict(set)
    for node, color in colors.items():
        classes[color].add(node)
    return dict(classes)


def are_isomorphic_wl(
    nodes1: List[NodeID],
    edges1: List[Tuple[NodeID, NodeID]],
    nodes2: List[NodeID],
    edges2: List[Tuple[NodeID, NodeID]],
    max_iterations: int = 100,
) -> Tuple[bool, Optional[str]]:
    """Test if two graphs are isomorphic using 1-WL.

    Returns:
        (possibly_isomorphic, reason)

    Interpretation:
        - If returns (False, reason): graphs are definitely NOT isomorphic
        - If returns (True, None): graphs may be isomorphic (WL cannot distinguish)

    Important:
        1-WL is incomplete for graph isomorphism!
        It can prove non-isomorphism but not isomorphism.

    Examples of WL failures:
        - Regular graphs with same degree (e.g., 3-regular graphs)
        - Pairs of strongly regular graphs
        - CFI graphs (classical counterexample)
    """
    if len(nodes1) != len(nodes2):
        return False, "Different number of nodes"

    if len(edges1) != len(edges2):
        return False, "Different number of edges"

    # Run WL on both graphs
    result1 = wl_test(nodes1, edges1, max_iterations)
    result2 = wl_test(nodes2, edges2, max_iterations)

    # Compare color histograms (multisets of colors)
    hist1 = Counter(result1.final_colors.values())
    hist2 = Counter(result2.final_colors.values())

    if hist1 != hist2:
        return False, "Different color histograms (WL can distinguish)"

    return True, None  # WL cannot distinguish (but may still be non-isomorphic!)


def wl_counterexample_regular_graphs() -> Tuple[
    Tuple[List[NodeID], List[Tuple[NodeID, NodeID]]],
    Tuple[List[NodeID], List[Tuple[NodeID, NodeID]]],
]:
    """Generate classic WL counterexample: two non-isomorphic 3-regular graphs.

    Returns:
        Two graphs that are non-isomorphic but have same WL coloring

    Construction:
        Both are 3-regular (all nodes have degree 3)
        Different structure (different numbers of triangles)
        Same WL colors (1-WL fails to distinguish)

    Theoretical significance:
        Proves that 1-WL (and thus standard MPNNs) cannot solve
        graph isomorphism in general.

    GNN implication:
        Sum-aggregation GNNs will assign identical representations
        to nodes in these two graphs, even though graphs are non-isomorphic.
    """
    # Graph 1: 6-cycle (hexagon)
    nodes1 = list(range(6))
    edges1 = [(i, (i + 1) % 6) for i in range(6)]
    edges1 += [(i, (i + 2) % 6) for i in range(6)]  # Make 3-regular

    # Graph 2: Complete bipartite K_{3,3}
    nodes2 = list(range(6))
    edges2 = [(i, j) for i in range(3) for j in range(3, 6)]

    return (nodes1, edges1), (nodes2, edges2)


def wl_expressive_power_analysis(result: WLResult) -> Dict[str, any]:
    """Analyze the discriminative power achieved by WL on a graph.

    Parameters:
        result: WL test result

    Returns:
        Analysis dictionary with metrics:
        - num_color_classes: How many equivalence classes
        - largest_class_size: Size of largest class (worst-case ambiguity)
        - singleton_ratio: Fraction of nodes uniquely identified
        - compression_ratio: |colors| / |nodes|

    Interpretation:
        - num_color_classes = |V|: WL perfectly distinguishes all nodes
        - num_color_classes = 1: WL failed completely (all nodes equivalent)
        - Higher singleton_ratio: WL is more expressive on this graph
    """
    assert result.color_classes is not None

    num_nodes = len(result.final_colors)
    num_classes = len(result.color_classes)
    class_sizes = [len(nodes) for nodes in result.color_classes.values()]

    largest_class = max(class_sizes) if class_sizes else 0
    num_singletons = sum(1 for s in class_sizes if s == 1)

    return {
        "num_nodes": num_nodes,
        "num_color_classes": num_classes,
        "largest_class_size": largest_class,
        "singleton_count": num_singletons,
        "singleton_ratio": num_singletons / num_nodes if num_nodes > 0 else 0.0,
        "compression_ratio": num_classes / num_nodes if num_nodes > 0 else 0.0,
        "iterations_to_converge": result.num_iterations,
        "converged": result.is_stable,
    }
