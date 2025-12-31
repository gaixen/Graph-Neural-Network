````markdown
```
expressivity/
├── __init__.py
├── wl_equivalence.py
├── gin.py
├── higher_order.py
└── counterexamples.py
```

Purpose: expose where GNNs provably fail and why — a place for falsification, not hype.

`wl_equivalence.py`

- Formal mapping between MPNNs and 1-WL. Node-level and graph-level collapse proofs.

`gin.py`

- Demonstrate why GIN is maximally expressive within the 1-WL framework and where that still isn't enough.

`higher_order.py`

- Higher-order architectures and their theoretical properties (k-WL relations, tensorization tradeoffs).

`counterexamples.py`

- Pairs and families of non-isomorphic graphs that defeat different classes of GNNs; annotated examples and minimal constructions.

Design principle:

- If you cannot predict failures here, stop implementing models. This folder must be the oracle for expressivity claims.
````
