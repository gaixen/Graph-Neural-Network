"""Abstract Graph definition and predicates.

Defines graphs as pure mathematical objects G = (V, E) without ML or tensor assumptions.
This module provides:
- Type-safe graph representations
- Predicates for graph properties (directed, weighted, simple, etc.)
- Graph construction and validation

Design principle: Keep modeling assumptions explicit and separate from representation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional, Hashable, Generic, TypeVar
from enum import Enum
import warnings


NodeID = TypeVar('NodeID', bound=Hashable)
EdgeID = Tuple[Any, Any]


class GraphType(Enum):
    """Enumeration of fundamental graph types."""
    UNDIRECTED = "undirected"
    DIRECTED = "directed"
    MULTIGRAPH = "multigraph"  # Multiple edges between same node pair
    WEIGHTED = "weighted"


@dataclass
class Edge(Generic[NodeID]):
    """Represents a graph edge with optional weight and attributes.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        weight: Edge weight (default: 1.0)
        attributes: Additional edge metadata
    """
    source: NodeID
    target: NodeID
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.source, self.target))
    
    def reverse(self) -> 'Edge[NodeID]':
        """Return reversed edge (target → source)."""
        return Edge(
            source=self.target,
            target=self.source,
            weight=self.weight,
            attributes=self.attributes.copy()
        )


@dataclass
class Graph(Generic[NodeID]):
    """Pure mathematical graph representation G = (V, E).
    
    Attributes:
        nodes: Set of node IDs
        edges: List of Edge objects
        graph_type: Type specification (directed, weighted, etc.)
        node_attributes: Per-node metadata
        graph_attributes: Global graph metadata
    
    Invariants maintained:
        - All edge endpoints exist in nodes
        - No self-loops unless explicitly allowed
        - No duplicate edges unless multigraph
    
    Design choices made explicit:
        - Node indexing: arbitrary IDs vs sequential integers
        - Edge representation: list vs adjacency structure
        - Weight semantics: additive (distances) vs multiplicative (probabilities)
    """
    nodes: Set[NodeID] = field(default_factory=set)
    edges: List[Edge[NodeID]] = field(default_factory=list)
    graph_type: GraphType = GraphType.UNDIRECTED
    node_attributes: Dict[NodeID, Dict[str, Any]] = field(default_factory=dict)
    graph_attributes: Dict[str, Any] = field(default_factory=dict)
    allow_self_loops: bool = False
    
    def __post_init__(self):
        """Validate graph structure."""
        self.validate()
    
    def validate(self) -> None:
        """Check graph invariants.
        
        Raises:
            ValueError: If graph structure violates invariants
        """
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Edge source {edge.source} not in node set")
            if edge.target not in self.nodes:
                raise ValueError(f"Edge target {edge.target} not in node set")
            
            if not self.allow_self_loops and edge.source == edge.target:
                raise ValueError(f"Self-loop detected: {edge.source} → {edge.target}")
    
    def add_node(self, node_id: NodeID, **attributes) -> None:
        """Add node with optional attributes.
        
        Parameters:
            node_id: Unique node identifier
            **attributes: Node metadata
        """
        self.nodes.add(node_id)
        if attributes:
            self.node_attributes[node_id] = attributes
    
    def add_edge(
        self,
        source: NodeID,
        target: NodeID,
        weight: float = 1.0,
        **attributes
    ) -> None:
        """Add edge to graph.
        
        Parameters:
            source: Source node ID
            target: Target node ID  
            weight: Edge weight (default: 1.0)
            **attributes: Edge metadata
        
        Notes:
            - For undirected graphs, automatically adds (target, source) as well
            - For multigraphs, allows duplicate edges
            - Validates endpoints exist in node set
        """
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        edge = Edge(source, target, weight, attributes)
        self.edges.append(edge)
        
        # For undirected graphs, add reverse edge
        if self.graph_type == GraphType.UNDIRECTED and source != target:
            reverse_edge = edge.reverse()
            self.edges.append(reverse_edge)
    
    def num_nodes(self) -> int:
        """Return number of nodes |V|."""
        return len(self.nodes)
    
    def num_edges(self) -> int:
        """Return number of edges |E|.
        
        Note: For undirected graphs, each edge is counted once (not twice).
        """
        if self.graph_type == GraphType.UNDIRECTED:
            return len(self.edges) // 2
        return len(self.edges)
    
    def degree(self, node: NodeID) -> int:
        """Compute degree of node.
        
        For directed graphs, returns out-degree.
        Use in_degree() for in-degree.
        """
        return sum(1 for e in self.edges if e.source == node)
    
    def in_degree(self, node: NodeID) -> int:
        """Compute in-degree (for directed graphs)."""
        return sum(1 for e in self.edges if e.target == node)
    
    def neighbors(self, node: NodeID) -> Set[NodeID]:
        """Return neighbors of node.
        
        For directed graphs, returns out-neighbors.
        """
        return {e.target for e in self.edges if e.source == node}
    
    def is_directed(self) -> bool:
        """Check if graph is directed."""
        return self.graph_type == GraphType.DIRECTED
    
    def is_connected(self) -> bool:
        """Check if graph is connected (undirected) or weakly connected (directed).
        
        Uses BFS from arbitrary starting node.
        
        Returns:
            True if all nodes reachable from any starting node
        """
        if not self.nodes:
            return True
        
        start = next(iter(self.nodes))
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # For directed graphs, traverse both directions for weak connectivity
            if self.is_directed():
                neighbors = self.neighbors(current) | {
                    e.source for e in self.edges if e.target == current
                }
            else:
                neighbors = self.neighbors(current)
            
            queue.extend(neighbors - visited)
        
        return len(visited) == len(self.nodes)


def is_directed(edges: List[Tuple[Any, Any]]) -> bool:
    """Check if edge list represents a directed graph.
    
    A graph is directed if there exists an edge (u,v) without reverse edge (v,u).
    
    Parameters:
        edges: List of (source, target) tuples
    
    Returns:
        True if graph appears to be directed
    
    Note:
        This heuristic may give false negatives for directed graphs
        that happen to have symmetric edges.
    """
    edge_set = set(edges)
    for u, v in edges:
        if (v, u) not in edge_set:
            return True
    return False


def is_simple_graph(edges: List[Tuple[Any, Any]], allow_self_loops: bool = False) -> bool:
    """Check if graph is simple (no multi-edges, optionally no self-loops).
    
    Parameters:
        edges: List of edges
        allow_self_loops: If False, self-loops violate simplicity
    
    Returns:
        True if graph is simple
    """
    seen = set()
    for u, v in edges:
        if (u, v) in seen:
            return False  # Duplicate edge
        seen.add((u, v))
        
        if not allow_self_loops and u == v:
            return False  # Self-loop
    
    return True


def connected_components(graph: Graph) -> List[Set[NodeID]]:
    """Find connected components using Union-Find.
    
    For directed graphs, finds weakly connected components.
    
    Parameters:
        graph: Input graph
    
    Returns:
        List of node sets, each forming a connected component
    """
    if graph.is_directed():
        warnings.warn("Computing weakly connected components for directed graph")
    
    # Union-Find parent pointers
    parent = {node: node for node in graph.nodes}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union all edges
    for edge in graph.edges:
        union(edge.source, edge.target)
    
    # Group nodes by root
    components: Dict[NodeID, Set[NodeID]] = {}
    for node in graph.nodes:
        root = find(node)
        components.setdefault(root, set()).add(node)
    
    return list(components.values())
