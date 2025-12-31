```
math/
├── __init__.py
├── linear_algebra.py
├── functional_analysis.py
├── probability.py
└── invariance.py
```

## linear_algebra.py

- Purpose: make every contraction, projection, and collapse explicit.

### Contents (conceptual):

- Eigen-decomposition

- Spectral radius

- Matrix powers and convergence

- Orthogonal projections

- Rank collapse

#### Docstrings must explain:

- Why repeated multiplication by a normalized adjacency converges

- Why oversmoothing is inevitable

## functional_analysis.py

- Purpose: explain why aggregation loses information.

### Contents:

- Permutation-invariant functions

- Multiset functions

- Universal approximation vs injectivity

- Sum / mean / max as operators

#### This is where you mathematically justify:

`Aggregation is a lossy operator`

- No graphs here. Just functions.

## invariance.py

- Purpose: formalize equivariance/invariance before models contaminate intuition.

### Contents:

- Group actions (permutation group)

- Equivariant maps

- Invariant readouts
