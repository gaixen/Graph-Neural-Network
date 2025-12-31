````markdown
```
training_pathologies/
├── __init__.py
├── oversmoothing.py
├── oversquashing.py
├── depth_limits.py
├── normalization.py
└── rewiring.py
```

Purpose: isolate architecture-level failure modes and explain why they occur even with infinite data.

Each file must answer:

- What fails (precise mathematical statement).
- Why it fails (operator producing the failure and an intuitive proof/sketch).
- Which operator is responsible (aggregation, normalization, depth), and why optimizers or data cannot fix it.

Notes on specific topics:

- `oversmoothing.py`: spectral view of repeated graph diffusion, fixed-point analysis, and when representations collapse.
- `oversquashing.py`: combinatorial bottlenecks and curvature/expansion arguments that compress long-range information.
- `depth_limits.py`: tradeoffs between expressivity and stability as depth grows.
- `normalization.py`: what normalization does to signal energies and gradient flow.
- `rewiring.py`: rewiring strategies with theoretical guarantees and failure modes.

Design principle:

- These modules should prevent blaming optimizers for structural failures; blame the operator first, then the rest.
````
