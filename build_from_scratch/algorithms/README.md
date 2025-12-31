````markdown
```
algorithms/
├── __init__.py
├── bfs.py
├── shortest_paths.py
├── isomorphism.py
└── wl_test.py
```

Purpose: classical graph algorithms expressed cleanly as math procedures (no tensors, no learning).

`bfs.py`

- Breadth-first search and frontier semantics; formal complexity and correctness notes.

`shortest_paths.py`

- Dijkstra, Bellman–Ford, and unweighted shortest-paths with precise preconditions and output formats.

`isomorphism.py`

- Canonicalization and exact isomorphism routines (naïve and optimized), with precise definitions of node/edge labelling.

`wl_test.py` (critical)

- Raw Weisfeiler–Lehman (1-WL) implementation without tensors.
- Must include: initialization, iterative refinement, stabilization detection, and explicit counterexamples.
- Docstrings must state what information is retained after k iterations and why color-collapse is irreversible.

Design principle:

- These files are the ground truth for expressivity claims. If a WL test or isomorphism routine is incorrect, downstream claims about model power are invalid.
````
