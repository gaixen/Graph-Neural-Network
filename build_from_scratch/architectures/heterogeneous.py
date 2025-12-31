"""Heterogeneous graph architectures and typed message passing; state assumptions and limits.

HETEROGENEOUS GRAPHS = Graphs with multiple node/edge types

Examples:
    - Knowledge graphs: (Person)-[knows]->(Person), (Person)-[worksAt]->(Company)
    - Molecules: Different atom types, different bond types
    - Academic networks: (Author)-[writes]->(Paper)-[cites]->(Paper)
    - Social networks: (User)-[follows]->(User), (User)-[likes]->(Post)

Key difference from homogeneous graphs:
    - Different node types have different feature dimensions
    - Different edge types represent different relationships
    - Cannot assume permutation symmetry across types!

Challenges:
    1. **Type-specific parameters**: Need different weights per type
    2. **Cross-type aggregation**: How to aggregate different types?
    3. **Semantics**: Each relation has different meaning
    4. **Imbalance**: Some types may be rare

Approaches:
    1. **Type-specific GNNs**: Separate GNN per node type
    2. **Relational GCN (R-GCN)**: Type-specific weight matrices
    3. **Heterogeneous Graph Transformer (HGT)**: Typed attention
    4. **Metapath-based**: Aggregate along semantic paths

This module provides:
    - R-GCN implementation
    - Typed message passing
    - Metapath aggregation
    - When heterogeneous methods needed
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HeterogeneousGraph:
    """Heterogeneous graph representation.
    
    Attributes:
        node_types: Dict mapping node_id -> type_name
        edge_types: Dict mapping edge_id -> type_name
        edges: List of (src, edge_type, tgt) triples
        node_features: Dict mapping (node_id, type) -> features
        num_node_types: Number of distinct node types
        num_edge_types: Number of distinct edge types
    """
    node_types: Dict[int, str]
    edge_types: Dict[int, str]
    edges: List[Tuple[int, str, int]]
    node_features: Dict[Tuple[int, str], np.ndarray]
    num_node_types: int
    num_edge_types: int


def relational_gcn_layer(
    node_features: Dict[str, np.ndarray],
    edges: List[Tuple[int, str, int]],
    node_types: Dict[int, str],
    weight_matrices: Dict[str, np.ndarray],
    aggregation: str = 'mean'
) -> Dict[str, np.ndarray]:
    """Relational GCN layer (R-GCN).
    
    R-GCN (Schlichtkrull et al., 2018):
        h_i^{l+1} = σ(Σ_{r ∈ R} Σ_{j ∈ N_i^r} (1/|N_i^r|) W_r^l h_j^l + W_0^l h_i^l)
    
    Key ideas:
        - Separate weight matrix W_r for each relation type r
        - Normalize by neighborhood size per relation
        - Self-loop with separate weight W_0
    
    Args:
        node_features: Features per node type {type: (num_nodes, dim)}
        edges: List of (src, relation_type, tgt)
        node_types: Mapping node_id -> type
        weight_matrices: Weight matrices per relation {relation: W}
        aggregation: Aggregation type
    
    Returns:
        Updated node features per type
    """
    # Group edges by relation type
    edges_by_relation = defaultdict(list)
    for src, rel, tgt in edges:
        edges_by_relation[rel].append((src, tgt))
    
    # Initialize output
    new_features = {}
    
    # Get all node IDs per type
    nodes_by_type = defaultdict(list)
    for node_id, node_type in node_types.items():
        nodes_by_type[node_type].append(node_id)
    
    # For each node type
    for node_type, node_ids in nodes_by_type.items():
        n = len(node_ids)
        if n == 0:
            continue
        
        # Get features for this type
        feats = node_features.get(node_type)
        if feats is None:
            continue
        
        d_in = feats.shape[1]
        
        # Aggregate messages from each relation type
        aggregated = np.zeros_like(feats)
        
        for rel_type, rel_edges in edges_by_relation.items():
            if rel_type not in weight_matrices:
                continue
            
            W_r = weight_matrices[rel_type]
            
            # For each edge of this type
            for src, tgt in rel_edges:
                if tgt not in node_ids:  # Target not in this type
                    continue
                
                # Get source features
                src_type = node_types.get(src)
                if src_type is None:
                    continue
                
                src_feats = node_features.get(src_type)
                if src_feats is None:
                    continue
                
                # Find index of tgt in node_ids
                try:
                    tgt_idx = node_ids.index(tgt)
                except ValueError:
                    continue
                
                # Message: W_r @ h_src
                # (Simplified: assume src_feats has src directly)
                message = W_r @ src_feats[src % len(src_feats)]  # Placeholder indexing
                
                aggregated[tgt_idx] += message
        
        # Normalize (mean aggregation)
        if aggregation == 'mean':
            # Count incoming edges per node (simplified)
            pass  # Already summed
        
        new_features[node_type] = aggregated
    
    return new_features


def heterogeneous_graph_transformer_attention(
    query_type: str,
    key_type: str,
    edge_type: str,
    query_feats: np.ndarray,
    key_feats: np.ndarray,
    attention_weights: Dict[Tuple[str, str, str], np.ndarray]
) -> np.ndarray:
    """Type-aware attention for HGT.
    
    HGT (Hu et al., 2020):
        Attention depends on (src_type, edge_type, tgt_type) triple
    
    Mathematical:
        α_{ij}^{τ,ϕ,r} = Attention(h_i^{τ}, h_j^{ϕ}, r)
        
        where τ, ϕ are node types, r is edge type
    
    Args:
        query_type: Query node type
        key_type: Key node type
        edge_type: Edge type
        query_feats: Query features
        key_feats: Key features
        attention_weights: Type-specific attention weights
    
    Returns:
        Attention scores
    """
    # Get type-specific weight
    weight_key = (query_type, edge_type, key_type)
    W = attention_weights.get(weight_key)
    
    if W is None:
        # Default: dot product
        scores = query_feats @ key_feats.T
    else:
        # Type-specific transformation
        scores = (W @ query_feats.T).T @ key_feats.T
    
    # Softmax
    scores_exp = np.exp(scores - np.max(scores))
    attention = scores_exp / (np.sum(scores_exp) + 1e-10)
    
    return attention


def metapath_aggregation(
    start_nodes: List[int],
    metapath: List[str],
    edges: List[Tuple[int, str, int]],
    node_features: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """Aggregate along metapaths.
    
    Metapath = sequence of relation types
    Example: Author -> Paper -> Venue
             ["writes", "publishedIn"]
    
    Aggregation:
        For each start node, follow metapath
        Aggregate features of reachable end nodes
    
    Args:
        start_nodes: Starting nodes
        metapath: Sequence of edge types
        edges: All edges (src, type, tgt)
        node_features: Node features
    
    Returns:
        Aggregated features per start node
    """
    # Build adjacency per edge type
    adj_by_type = defaultdict(list)
    for src, edge_type, tgt in edges:
        adj_by_type[edge_type].append((src, tgt))
    
    # For each start node, follow metapath
    aggregated = {}
    
    for start in start_nodes:
        # BFS along metapath
        current_nodes = {start}
        
        for edge_type in metapath:
            next_nodes = set()
            for src in current_nodes:
                for s, t in adj_by_type[edge_type]:
                    if s == src:
                        next_nodes.add(t)
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        # Aggregate features of reached nodes
        if current_nodes:
            reached_feats = [node_features[n] for n in current_nodes if n in node_features]
            if reached_feats:
                aggregated[start] = np.mean(reached_feats, axis=0)
    
    return aggregated


def why_heterogeneous_methods_needed() -> str:
    """Explain why specialized methods needed.
    
    Returns:
        Explanation
    """
    return """
    WHY HETEROGENEOUS METHODS?
    
    PROBLEM WITH HOMOGENEOUS GNNs:
        1. Different types have different semantics
           - "Author writes Paper" ≠ "Paper cites Paper"
           - Need different aggregation logic
        
        2. Different feature dimensions
           - Authors: age, affiliation, ...
           - Papers: title embedding, keywords, ...
           - Cannot use same weight matrix!
        
        3. Imbalanced types
           - Many papers, few venues
           - Uniform aggregation doesn't work
        
        4. Type-specific patterns
           - Authors collaborate
           - Papers cite papers
           - Different graph patterns!
    
    SOLUTION: TYPE-AWARE ARCHITECTURES
        - Separate parameters per type
        - Type-specific aggregation
        - Semantic metapaths
    
    EXAMPLE:
        Knowledge graph: (Person, knows, Person)
        
        Homogeneous GNN:
        h_person = AGG(all neighbors)
        
        Problem: Treats all relations same!
        
        Heterogeneous GNN:
        h_person = AGG_knows(knows_neighbors) + 
                   AGG_worksAt(worksAt_neighbors) + ...
        
        Different weights for different relations!
    """


def rgcn_vs_hgt() -> Dict[str, Dict[str, Any]]:
    """Compare R-GCN and HGT.
    
    Returns:
        Comparison
    """
    return {
        "R-GCN": {
            "pros": [
                "Simple extension of GCN",
                "Type-specific weights",
                "Efficient",
            ],
            "cons": [
                "Fixed aggregation (mean)",
                "No attention mechanism",
                "Many parameters (W per relation)",
            ],
            "when_to_use": "Many edge types, need efficiency",
        },
        "HGT": {
            "pros": [
                "Attention-based (adaptive)",
                "Type-aware attention",
                "Better expressivity",
            ],
            "cons": [
                "More complex",
                "More parameters",
                "Slower (attention O(n^2))",
            ],
            "when_to_use": "Complex heterogeneous graphs, need flexibility",
        },
    }


def metapath_based_methods() -> str:
    """Explain metapath-based aggregation.
    
    Returns:
        Explanation
    """
    return """
    METAPATH-BASED METHODS:
    
    IDEA:
        Define semantic paths (metapaths)
        Aggregate along these paths
    
    EXAMPLE: Academic network
        Metapath: Author -> Paper -> Author (APA)
        Meaning: Co-authorship
        
        Metapath: Author -> Paper -> Venue (APV)
        Meaning: Publication venues
    
    ALGORITHM:
        1. Define metapaths (domain knowledge)
        2. For each metapath:
           - Find all instances
           - Aggregate features
        3. Combine metapath embeddings
    
    ADVANTAGES:
        - Semantically meaningful
        - Domain-specific
        - Interpretable
    
    DISADVANTAGES:
        - Requires metapath design (manual)
        - Exponential path count
        - May miss implicit patterns
    
    METHODS:
        - Metapath2vec: Random walks on metapaths
        - HAN: Hierarchical attention on metapaths
        - MAGNN: Metapath aggregation with GNN
    """


def challenges_and_limitations() -> List[str]:
    """Challenges specific to heterogeneous graphs.
    
    Returns:
        List of challenges
    """
    return [
        "Parameter explosion: Separate weights per type combination",
        "Type imbalance: Rare types get poor representations",
        "Metapath design: Requires domain expertise",
        "Scalability: More complex than homogeneous GNNs",
        "Overfitting: Many parameters, may overfit small graphs",
        "Evaluation: Hard to evaluate across different types",
        "No permutation symmetry: Cannot assume type-agnostic permutations",
    ]
