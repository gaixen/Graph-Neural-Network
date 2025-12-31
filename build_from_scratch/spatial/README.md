````markdown
```
spatial/
├── __init__.py
├── aggregation.py
├── message_passing.py
├── gcn.py
├── graphsage.py
└── gat.py
```

Purpose: derive message passing from first principles and make aggregation limitations explicit.

`aggregation.py`

- Formal study of sum / mean / max aggregation; injectivity analysis and permutation-invariance proofs.

`message_passing.py`

- Generic MPNN abstraction: formal definition and separation of `message`, `aggregate`, and `update`.
- Every model in this repo should be expressible with this interface.

`gcn.py`, `graphsage.py`, `gat.py`

- Each file documents the exact operator applied, its invariance/equivariance properties, and why oversmoothing occurs.
- Show which assumptions are baked into each design (normalization, attention sparsity, sampling).

Design principle:

- Keep the algebra explicit so similarities between architectures are visible; matching implementations across files is a sign of correct abstraction.
````
