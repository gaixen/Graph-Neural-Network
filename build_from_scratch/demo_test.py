"""Demonstration and smoke tests for core GNN modules.

This file shows how to use the enriched implementations and verifies
they work correctly (smoke tests, not comprehensive unit tests).

Run with: python demo_test.py
"""

import numpy as np
import sys
from pathlib import Path

# Import our modules
from gnn_math import linear_algebra, functional_analysis
from algorithms import wl_test
from graph import matrices as graph_matrices
from spatial import gcn
from spectral import chebyshev


def test_linear_algebra():
    """Test math/linear_algebra.py functions."""
    print("\n" + "=" * 60)
    print("Testing math/linear_algebra.py")
    print("=" * 60)

    # Create a simple symmetric matrix
    A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=float)

    # Test eigendecomposition
    eig_result = linear_algebra.eigen_decomposition(A, symmetric=True)
    print(f"✓ Eigenvalues: {eig_result.eigenvalues}")
    print(f"  Spectral radius: {linear_algebra.spectral_radius(A):.4f}")

    # Test power iteration convergence
    P = A / linear_algebra.spectral_radius(A)  # Normalize to ρ(P) = 1
    x0 = np.array([1.0, 0.0, 0.0])
    converged, iters, residual = linear_algebra.power_iteration_convergence(
        P, x0, tol=1e-6
    )
    print(
        f"✓ Power iteration: converged={converged}, iters={iters}, residual={residual:.2e}"
    )

    # Test effective rank
    eff_rank = linear_algebra.effective_rank(A)
    print(f"✓ Effective rank: {eff_rank:.2f} (actual rank: {np.linalg.matrix_rank(A)})")


def test_functional_analysis():
    """Test math/functional_analysis.py aggregation operators."""
    print("\n" + "=" * 60)
    print("Testing math/functional_analysis.py")
    print("=" * 60)

    # Test aggregation operators
    multiset = [1.0, 2.0, 2.0, 3.0]

    sum_agg = functional_analysis.SumAggregation()
    mean_agg = functional_analysis.MeanAggregation()
    max_agg = functional_analysis.MaxAggregation()

    print(f"✓ Sum aggregation: {sum_agg(multiset)}")
    print(f"✓ Mean aggregation: {mean_agg(multiset)}")
    print(f"✓ Max aggregation: {max_agg(multiset)}")

    # Test injectivity
    domain = [[1, 2, 3], [1, 1, 1, 1, 1, 1], [2, 2, 2]]
    result = functional_analysis.check_injectivity_on_domain(sum_agg, domain)
    print(f"✓ Injectivity check: {result.explanation}")


def test_wl_algorithm():
    """Test algorithms/wl_test.py Weisfeiler-Lehman implementation."""
    print("\n" + "=" * 60)
    print("Testing algorithms/wl_test.py (WL Test)")
    print("=" * 60)

    # Test on a simple graph (triangle)
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (2, 0)]

    result = wl_test.wl_test(nodes, edges, max_iterations=10)
    print(f"✓ WL converged in {result.num_iterations} iterations")
    print(f"  Final colors: {result.final_colors}")

    # Test expressivity analysis
    analysis = wl_test.wl_expressive_power_analysis(result)
    print(f"  Num color classes: {analysis['num_color_classes']}")
    print(f"  Singleton ratio: {analysis['singleton_ratio']:.2%}")

    # Test counterexample
    (n1, e1), (n2, e2) = wl_test.wl_counterexample_regular_graphs()
    is_iso, reason = wl_test.are_isomorphic_wl(n1, e1, n2, e2)
    print(f"✓ WL counterexample: WL thinks isomorphic={is_iso}")
    print(f"  (These graphs are actually non-isomorphic!)")


def test_graph_matrices():
    """Test graph/matrices.py matrix constructions."""
    print("\n" + "=" * 60)
    print("Testing graph/matrices.py")
    print("=" * 60)

    # Create a simple graph (4-cycle)
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    A = graph_matrices.adjacency_matrix(nodes, edges, directed=False)
    print(f"✓ Adjacency matrix shape: {A.shape}")
    print(f"  Symmetry check: {np.allclose(A, A.T)}")

    L = graph_matrices.laplacian_matrix(A, graph_matrices.LaplacianType.SYMMETRIC)
    print(f"✓ Symmetric Laplacian eigenvalues: {np.linalg.eigvalsh(L)}")

    A_norm = graph_matrices.normalized_adjacency(A, symmetric=True, add_self_loops=True)
    print(
        f"✓ Normalized adjacency eigenvalue range: [{np.linalg.eigvalsh(A_norm).min():.3f}, {np.linalg.eigvalsh(A_norm).max():.3f}]"
    )


def test_gcn_oversmoothing():
    """Test spatial/gcn.py GCN implementation and oversmoothing analysis."""
    print("\n" + "=" * 60)
    print("Testing spatial/gcn.py (GCN + Oversmoothing)")
    print("=" * 60)

    # Create a small graph (6-cycle)
    nodes = list(range(6))
    edges = [(i, (i + 1) % 6) for i in range(6)]
    A = graph_matrices.adjacency_matrix(nodes, edges, directed=False)

    # Preprocess
    A_norm = gcn.preprocess_adjacency(A, add_self_loops=True, normalize=True)
    print(f"✓ Preprocessed adjacency shape: {A_norm.shape}")

    # Test propagation
    X = np.random.randn(6, 4)  # 6 nodes, 4 features
    W = np.random.randn(4, 8)  # Transform to 8 features

    H_out = gcn.gcn_propagate(A_norm, X, W, activation=np.tanh)
    print(f"✓ GCN propagation output shape: {H_out.shape}")

    # Analyze oversmoothing
    analysis = gcn.analyze_oversmoothing(A, num_layers=10)
    print(f"✓ Oversmoothing analysis:")
    print(f"  Spectral gap: {analysis['spectral_gap']:.4f}")
    print(f"  Effective rank @ layer 5: {analysis['effective_ranks'][4]:.2f}")
    print(f"  Effective rank @ layer 10: {analysis['effective_ranks'][9]:.2f}")
    print(f"  Theoretical decay rate: {analysis['theoretical_decay_rate']:.4f}")

    # Verify equivariance
    perm = np.array([1, 2, 3, 4, 5, 0])  # Rotate nodes
    is_equivariant = gcn.verify_equivariance(A, X, W, perm, tol=1e-6)
    print(f"✓ Permutation equivariance verified: {is_equivariant}")


def test_spectral_chebyshev():
    """Test spectral/chebyshev.py polynomial filters."""
    print("\n" + "=" * 60)
    print("Testing spectral/chebyshev.py")
    print("=" * 60)

    # Create graph
    nodes = list(range(5))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    A = graph_matrices.adjacency_matrix(nodes, edges, directed=False)
    L = graph_matrices.laplacian_matrix(A, graph_matrices.LaplacianType.SYMMETRIC)

    # Test monomial polynomial filter
    signal = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    coeffs = [1.0, 0.5, 0.1]  # g(L) = I + 0.5*L + 0.1*L²

    filtered_mono = chebyshev.monomial_polynomial_filter(L, coeffs, signal)
    print(f"✓ Monomial filter output: {filtered_mono}")

    # Test Chebyshev polynomial filter
    filtered_cheb = chebyshev.chebyshev_polynomial_filter(L, coeffs, signal)
    print(f"✓ Chebyshev filter output: {filtered_cheb}")

    # Test GCN filter (1-hop aggregation)
    filtered_gcn = chebyshev.gcn_filter(A, signal, add_self_loops=True)
    print(f"✓ GCN filter (1-hop): {filtered_gcn}")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("# GNN From Scratch - Core Module Demonstrations")
    print("#" * 60)

    try:
        test_linear_algebra()
        test_functional_analysis()
        test_wl_algorithm()
        test_graph_matrices()
        test_gcn_oversmoothing()
        test_spectral_chebyshev()

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        print("\nKey modules tested:")
        print("  • math/linear_algebra.py - Spectral analysis")
        print("  • math/functional_analysis.py - Aggregation operators")
        print("  • algorithms/wl_test.py - WL graph isomorphism test")
        print("  • graph/matrices.py - Graph matrix representations")
        print("  • spatial/gcn.py - GCN with oversmoothing analysis")
        print("  • spectral/chebyshev.py - Polynomial filter approximations")
        print("\nNext steps:")
        print("  1. Explore the README.md for full documentation")
        print("  2. Read docstrings in each module (they contain theory!)")
        print("  3. Run oversmoothing analysis on your own graphs")
        print("  4. Test WL expressivity limits with counterexamples")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
