# GNN From Scratch: Theory-First Implementation

> **Non-negotiable principle:** No handwaving. Every claim about GNNs must be backed by explicit mathematics or falsifiable experiments.

## Philosophy

This repository builds Graph Neural Networks from **mathematical foundations up**, not from PyTorch down. If you don't understand _why_ something works (or fails), you shouldn't be using it.

### What makes this different

**Most GNN tutorials:**

- Start with `torch_geometric`
- Show you how to train on a benchmark
- Claim "state-of-the-art results"
- Never explain why oversmoothing exists or what WL-isomorphism means

**This repository:**

1. **Math first**: Linear algebra → Spectral theory → Group theory → Functional analysis
2. **Explicit information loss**: Every matrix operation documents what information is destroyed
3. **Falsification, not demonstration**: Experiments designed to _fail_, exposing fundamental limits
4. **No ML dependencies**: Core theory uses only `numpy` (no PyTorch/TensorFlow hidden magic)

---

## Repository Structure

```
build_from_scratch/
│
├── README.md
├── pyproject.toml / setup.py
├── requirements.txt
│
├── math/                    # [CRITICAL] Mathematical foundations
│   ├── linear_algebra.py        → Spectral decomposition, rank collapse, convergence analysis
│   ├── functional_analysis.py   → Why aggregation loses information (with proofs)
│   ├── invariance.py            → Group actions, equivariance, permutation symmetry
│   └── probability.py           → Concentration inequalities for GNN analysis
│
├── graph/                   # Pure graph objects (no ML)
│   ├── graph.py                 → Graph = (V,E) with explicit type safety
│   ├── matrices.py              → Adjacency, Laplacian, incidence (info-loss documented)
│   ├── signals.py               → Node/edge/graph-level feature spaces
│   └── generators.py            → Erdős-Rényi, SBM, rings (with invariant documentation)
│
├── algorithms/              # Classical graph algorithms (no learning)
│   ├── bfs.py
│   ├── shortest_paths.py
│   ├── isomorphism.py
│   └── wl_test.py              → [CRITICAL] 1-WL test = GNN expressivity upper bound
│
├── spectral/                # Spectral graph theory → GCN motivation
│   ├── laplacian_spectrum.py
│   ├── fourier.py               → Graph Fourier transform
│   ├── convolution.py           → Why spectral filters are global
│   └── chebyshev.py            → [CRITICAL] Polynomial approximation → locality
│
├── spatial/                 # Message-passing frameworks
│   ├── aggregation.py           → Sum/mean/max injectivity analysis
│   ├── message_passing.py       → Generic MPNN abstraction
│   ├── gcn.py                  → [CRITICAL] GCN with oversmoothing analysis
│   ├── graphsage.py
│   └── gat.py
│
├── expressivity/            # Where false beliefs die
│   ├── wl_equivalence.py        → Formal MPNN ↔ 1-WL mapping
│   ├── gin.py                   → Why GIN is optimal *within* 1-WL
│   ├── higher_order.py          → k-WL and higher-order architectures
│   └── counterexamples.py       → Graphs that defeat standard MPNNs
│
├── training_pathologies/    # Architecture-level failures
│   ├── oversmoothing.py         → Spectral analysis of diffusion convergence
│   ├── oversquashing.py         → Combinatorial bottlenecks
│   ├── depth_limits.py
│   ├── normalization.py
│   └── rewiring.py
│
├── architectures/           # Only after theory is solid
│   ├── transformers.py
│   ├── positional_encodings.py
│   ├── subgraph_gnns.py
│   └── heterogeneous.py
│
├── experiments/             # Falsification experiments
│   ├── wl_failures/             → Demonstrate WL test failures
│   ├── depth_collapse/          → Show oversmoothing empirically
│   ├── long_range_tasks/        → Stress-test receptive fields
│   └── rewiring_effects/
│
└── utils/                   # Boring but necessary
```

---

## Key Design Principles

### 1. **Explicit Information Loss**

Every matrix operation documents what information is **encoded**, **lost** (irreversibly), and which assumptions are **implicit**.

Example from [graph/matrices.py](graph/matrices.py):

- Normalized Laplacian preserves connectivity but loses absolute edge weights
- Symmetric normalization assumes undirected graphs
- Self-loops prevent degree-0 issues

### 2. **Theoretical Foundations Before Models**

You cannot understand GCN without understanding:

1. Graph Laplacian spectrum → [spectral/laplacian_spectrum.py](spectral/laplacian_spectrum.py)
2. Spectral convolution as global operator → [spectral/convolution.py](spectral/convolution.py)
3. Polynomial approximation for locality → [spectral/chebyshev.py](spectral/chebyshev.py)

### 3. **WL Test as Ground Truth**

The Weisfeiler-Lehman test ([algorithms/wl_test.py](algorithms/wl_test.py)) defines expressivity upper bounds. If you cannot predict when your model fails, stop claiming expressivity gains.

### 4. **Falsification Over Validation**

Experiments are designed to **fail** and expose structural limits, not just show improvements.

---

## Quick Start

### Installation

```bash
pip install numpy scipy matplotlib networkx
```

### Example: Understanding Oversmoothing

```python
from graph.matrices import adjacency_matrix
from spatial.gcn import analyze_oversmoothing
import numpy as np

# Create a simple graph (10-cycle)
nodes = list(range(10))
edges = [(i, (i+1) % 10) for i in range(10)]
A = adjacency_matrix(nodes, edges, directed=False)

# Analyze how features collapse with depth
analysis = analyze_oversmoothing(A, num_layers=20)

print(f"Spectral gap: {analysis['spectral_gap']:.4f}")
print(f"Effective rank at layer 10: {analysis['effective_ranks'][9]:.2f}")
```

### Example: WL Test Demonstrates Limits

```python
from algorithms.wl_test import wl_counterexample_regular_graphs, are_isomorphic_wl

# Get two non-isomorphic graphs that defeat 1-WL
(nodes1, edges1), (nodes2, edges2) = wl_counterexample_regular_graphs()

# WL cannot distinguish them (and neither can standard MPNNs)
is_iso, reason = are_isomorphic_wl(nodes1, edges1, nodes2, edges2)
print(f"WL result: {is_iso}")  # True - but graphs are NOT isomorphic!
```

---

## Core Theorems Implemented

### 1. **Oversmoothing is Inevitable** ([spatial/gcn.py](spatial/gcn.py))

Repeated application of normalized adjacency converges to rank-1 matrix. Deep GCNs collapse representations.

### 2. **MPNN ≡ 1-WL** ([expressivity/wl_equivalence.py](expressivity/wl_equivalence.py))

Message-passing GNNs with injective aggregation are at most as powerful as the WL test.

### 3. **Aggregation is Lossy** ([math/functional_analysis.py](math/functional_analysis.py))

Sum/mean/max aggregation loses information about neighbor multiplicity and ordering.

---

## Mathematical Prerequisites

- Linear Algebra: Eigendecomposition, SVD, rank
- Graph Theory: Laplacian matrices, graph isomorphism
- Functional Analysis: Operators, invariance
- Basic Group Theory: Permutation groups, equivariance

Start with [math/linear_algebra.py](math/linear_algebra.py) docstrings for foundations.

---

## Contributing

Maintain theory-first philosophy:

1. Mathematical rigor (theorems or falsifiable experiments)
2. Explicit assumptions documentation
3. Information-loss tracking
4. Negative results valued

---

## License

MIT License
