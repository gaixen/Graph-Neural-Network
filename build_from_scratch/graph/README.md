````markdown
```
graph/
├── __init__.py
├── graph.py
├── matrices.py
├── signals.py
└── generators.py
```

Purpose: treat graphs as pure mathematical objects `(V, E)`. Keep modeling and ML assumptions out of this layer.

`graph.py`

- Definition: Graph = (V, E) with explicit types for `V` and `E`.
- Distinguish directed vs undirected, simple vs multigraph, weighted vs unweighted.
- No tensors, no learning-layer helpers — only predicates and constructors.

`matrices.py`

- Constructs: adjacency (directed / undirected), incidence, degree, Laplacian (unnormalized and symmetric normalized).
- Docstrings must state explicitly:
  - What information each matrix encodes and what it destroys (e.g., adjacency loses edge identity/multiplicity when symmetrized).
  - Which assumptions are implicit (self-loops, orientation, normalization convention).
  - How those choices affect spectral properties and downstream operator semantics.

`signals.py`

- Node signals, edge signals, graph-level readouts as function spaces (R^n, functions on E, etc.).
- Lifting/pushing signals between node/edge/graph spaces.
- Docstrings: clarify what is lost when representing signals as vectors and when that projection is valid.

`generators.py`

- Pure mathematical graph generators (Erdős–Rényi, SBM, rings, lattices) documented with preserved invariants.

Design principle:

- Make all information-loss explicit here. If `matrices.py` or `signals.py` hides assumptions, downstream GNN layers will inherit accidental biases.
````
