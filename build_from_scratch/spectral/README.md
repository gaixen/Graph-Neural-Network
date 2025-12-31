````markdown
```
spectral/
├── __init__.py
├── laplacian_spectrum.py
├── fourier.py
├── convolution.py
└── chebyshev.py
```

Purpose: conceptual scaffolding for spectral graph theory and its implications for filters and locality.

`laplacian_spectrum.py`

- Eigen-decomposition of Laplacians, spectrum interpretation, multiplicities, and relations to connectivity.

`fourier.py`

- Graph Fourier transform, eigenbasis interpretation, and frequency as smoothness of node signals.

`convolution.py`

- Spectral convolution as a global operator; rigorous explanation of why naive spectral filters do not localize.

`chebyshev.py`

- Polynomial approximation of spectral filters; derivation that expresses k-hop locality using polynomials.
- Docstrings must explicitly derive

$$g(L)=\sum_k \theta_k L^k$$

and show how polynomial order yields effective locality.

Design principle:

- Provide the mathematical intuition behind GCN and spectral filters so the relationship between spectral design and spatial locality is explicit.
````
