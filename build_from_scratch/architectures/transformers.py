"""Graph transformer abstractions: attention over nodes, global tokens, and limitations.

GRAPH TRANSFORMERS = Apply transformer architecture to graphs

Key idea:
    Replace graph convolution with FULL ATTENTION
    - Every node attends to every other node
    - No graph structure constraints (initially)
    - Learn which connections matter

Advantages over MPNNs:
    1. **No oversquashing**: Direct paths between all nodes
    2. **Long-range dependencies**: Attention captures distant relationships
    3. **Flexible receptive field**: Not limited by graph diameter
    4. **Expressivity**: Can exceed 1-WL (with positional encodings)

Disadvantages:
    1. **Computational cost**: O(n^2) attention vs O(m) message passing
    2. **Loses graph inductive bias**: Treats graph as set of nodes
    3. **Requires positional encodings**: Else permutation-invariant only
    4. **Data hungry**: Needs more training data

Key components:
    - Multi-head self-attention
    - Positional/structural encodings
    - Optional graph-aware attention bias
    - Global readout tokens

This module provides:
    - Graph transformer layer implementation
    - Attention mechanisms for graphs
    - Comparison with MPNNs
    - When transformers help/hurt
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GraphTransformerConfig:
    """Configuration for graph transformer.

    Attributes:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        use_graph_bias: Whether to use graph structure as attention bias
        dropout: Dropout rate
        use_global_token: Whether to use global pooling token
    """

    hidden_dim: int = 128
    num_heads: int = 8
    use_graph_bias: bool = True
    dropout: float = 0.1
    use_global_token: bool = False


def multi_head_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    num_heads: int,
    attention_bias: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-head self-attention.

    Mathematical operation:
        For each head h:
        A_h = softmax(Q_h K_h^T / sqrt(d_k) + B)
        out_h = A_h V_h

        Final: out = Concat(out_1, ..., out_H) W^O

    Args:
        queries: Query matrix (n, d)
        keys: Key matrix (n, d)
        values: Value matrix (n, d)
        num_heads: Number of heads
        attention_bias: Optional attention bias (e.g., from graph structure)

    Returns:
        (output, attention_weights)
    """
    n, d = queries.shape
    d_k = d // num_heads

    # Split into heads
    # Simplified: Just compute single-head for demonstration
    # Real implementation would split across head dimension

    # Compute attention scores
    scores = queries @ keys.T / np.sqrt(d_k)

    # Add bias (graph structure)
    if attention_bias is not None:
        scores = scores + attention_bias

    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-10)

    # Apply to values
    output = attention @ values

    return output, attention


def graph_transformer_layer(
    node_features: np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    config: GraphTransformerConfig = GraphTransformerConfig(),
) -> np.ndarray:
    """Single graph transformer layer.

    Operation:
        1. Multi-head self-attention (with optional graph bias)
        2. Add & Norm
        3. Feed-forward network
        4. Add & Norm

    Difference from MPNN:
        - MPNN: Attention only over NEIGHBORS (sparse)
        - Transformer: Attention over ALL NODES (dense)

    Args:
        node_features: Node features (n, d)
        edge_index: Optional edge index for graph bias
        config: Configuration

    Returns:
        Updated node features
    """
    n, d = node_features.shape

    # Compute attention bias from graph structure
    attention_bias = None
    if config.use_graph_bias and edge_index is not None:
        # Bias: -inf for non-edges, 0 for edges
        # This guides attention to focus on neighbors
        attention_bias = np.full((n, n), -1e9)
        for e in range(edge_index.shape[1]):
            i, j = edge_index[0, e], edge_index[1, e]
            attention_bias[i, j] = 0
            attention_bias[j, i] = 0
        # Self-loops
        np.fill_diagonal(attention_bias, 0)

    # Self-attention
    # Simplified: Use node_features as Q, K, V
    attn_out, _ = multi_head_attention(
        node_features, node_features, node_features, config.num_heads, attention_bias
    )

    # Add & Norm
    h = node_features + attn_out
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-10)  # Simple normalization

    # Feed-forward (simplified: linear)
    # Real FFN: h = W2(ReLU(W1(h)))
    h_ffn = h @ np.random.randn(d, d) * 0.01  # Placeholder weights

    # Add & Norm
    h = h + h_ffn
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-10)

    return h


def graph_transformer_vs_mpnn() -> str:
    """Compare graph transformers with MPNNs.

    Returns:
        Comparison
    """
    return """
    GRAPH TRANSFORMERS vs MPNNs:
    
    EXPRESSIVITY:
        Transformer: Can exceed 1-WL with positional encodings
        MPNN: Limited to 1-WL (at best)
        
        Winner: Transformer
    
    COMPLEXITY:
        Transformer: O(n^2) attention
        MPNN: O(m) message passing (m = #edges)
        
        For sparse graphs: m << n^2
        Winner: MPNN
    
    INDUCTIVE BIAS:
        Transformer: Treats graph as set (no structure bias)
        MPNN: Uses graph structure explicitly
        
        For graph-structured data: MPNN often better
        Winner: MPNN (for small data)
    
    LONG-RANGE DEPENDENCIES:
        Transformer: Direct attention to all nodes
        MPNN: O(k) layers for k-hop
        
        Winner: Transformer
    
    OVERSQUASHING:
        Transformer: No bottlenecks (full attention)
        MPNN: Suffers from oversquashing
        
        Winner: Transformer
    
    OVERSMOOTHING:
        Transformer: Can also oversmooth (with many layers)
        MPNN: Severe oversmoothing
        
        Winner: Transformer (slightly)
    
    DATA EFFICIENCY:
        Transformer: Needs large datasets (many parameters)
        MPNN: Works with small datasets (stronger bias)
        
        Winner: MPNN (for small data)
    
    OVERALL:
        Large graphs, lots of data, long-range → Transformer
        Small graphs, sparse, local structure → MPNN
        
        Hybrid: Graph-aware transformer (best of both)
    """


def global_pooling_token() -> str:
    """Explain global pooling tokens in graph transformers.

    Returns:
        Explanation
    """
    return """
    GLOBAL POOLING TOKEN:
    
    Problem:
        Graph-level prediction requires pooling node features
        Standard pooling (sum/mean) loses information
    
    Solution:
        Add a special "global" token
        
        Procedure:
        1. Create global token g with learned initialization
        2. Include g in attention: g attends to all nodes
        3. Nodes attend to g (bidirectional)
        4. After L layers, g contains graph summary
        5. Use g for graph-level prediction
    
    Mathematical:
        Input: {h_1, ..., h_n, g}
        
        Attention:
        g' = Attention(g, {h_1, ..., h_n, g})
        h_i' = Attention(h_i, {h_1, ..., h_n, g})
        
        Output: g' (for graph classification)
    
    Advantages:
        - Learned pooling (vs fixed sum/mean)
        - Can focus on relevant nodes
        - Differentiable end-to-end
    
    Disadvantages:
        - Extra parameter (global token)
        - May not be permutation-invariant (if g position matters)
        - Requires tuning
    
    Examples:
        - Graph transformers (Dwivedi et al., 2020)
        - Vision transformers (CLS token)
    """


def graph_aware_attention_bias() -> str:
    """Explain how to incorporate graph structure into attention.

    Returns:
        Explanation
    """
    return """
    GRAPH-AWARE ATTENTION BIAS:
    
    Problem:
        Pure transformer ignores graph structure
        May be wasteful or underperform on graph data
    
    Solution:
        Bias attention based on graph properties
    
    Method 1: EDGE BIAS
        A_ij = softmax(q_i k_j / sqrt(d) + b_ij)
        
        where b_ij = {
            0 if (i,j) is an edge
            -∞ otherwise
        }
        
        Effect: Only attend to neighbors (like MPNN!)
        
    Method 2: DISTANCE BIAS
        b_ij = -α × dist(i, j)
        
        Closer nodes → higher attention
        Distant nodes → lower (but non-zero) attention
        
    Method 3: POSITIONAL ENCODING BIAS
        b_ij = learned_fn(PE_i, PE_j)
        
        Use positional encodings to compute bias
        
    Method 4: SPECTRAL BIAS
        b_ij = Σ_k λ_k v_k(i) v_k(j)
        
        Based on Laplacian eigenvectors
        
    TRADEOFF:
        Strong bias → more like MPNN (good for small data)
        Weak bias → more like pure transformer (flexible)
        
    BEST PRACTICE:
        Start with graph-aware bias
        Let model learn to override if needed
    """


def when_to_use_transformers() -> Dict[str, str]:
    """Guidelines for using graph transformers.

    Returns:
        Recommendations
    """
    return {
        "large_graphs": "Use sparse attention or avoid (O(n^2) too expensive)",
        "small_graphs": "Transformers work well (n < 1000)",
        "long_range_deps": "Transformers excel (molecular properties, protein interactions)",
        "local_structure": "MPNNs better (social networks, citations)",
        "lots_of_data": "Transformers can leverage data (pre-training)",
        "small_datasets": "MPNNs better (stronger inductive bias)",
        "heterogeneous_graphs": "Transformers flexible (typed attention)",
        "oversquashing_issues": "Transformers solve this (no bottlenecks)",
    }


def limitations_of_graph_transformers() -> List[str]:
    """Fundamental limitations.

    Returns:
        Limitations
    """
    return [
        "O(n^2) complexity: Intractable for large graphs (>10k nodes)",
        "Loses graph inductive bias: May need more data than MPNNs",
        "Still oversmoothes with depth: Attention doesn't solve everything",
        "Requires positional encodings: Else limited expressivity",
        "Memory intensive: Attention matrices large",
        "May not respect graph semantics: Can connect semantically incompatible nodes",
    ]
