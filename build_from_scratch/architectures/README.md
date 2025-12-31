````markdown
```
architectures/
├── __init__.py
├── transformers.py
├── positional_encodings.py
├── subgraph_gnns.py
└── heterogeneous.py
```

Purpose: implementation of architecture ideas only after theoretical motivations and tradeoffs are explicit.

Each file must include:

- What theoretical limitation the architecture targets.
- What new assumptions it introduces.
- What it still cannot fix (explicit failure modes and limitations).

Design principle:

- If an architecture file does not state its assumptions and tradeoffs, it should be deleted. Preference for concise, well-documented components over many half-justified designs.
````
