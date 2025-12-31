"""Probability essentials for reasoning about random graphs and signals.

Provides foundational probability tools for:
- Random graph models (Erdős-Rényi, SBM)
- Signal analysis on graphs
- Concentration inequalities for GNN analysis
- Statistical properties of aggregations

Mathematical framework for analyzing GNN behavior in expectation and with high probability.
"""

from dataclasses import dataclass
from typing import Iterable, Callable
import numpy as np
import warnings


@dataclass
class DistributionStatistics:
    """Container for distribution statistics.

    Attributes:
        mean: Expected value μ = E[X]
        variance: Variance σ² = E[(X-μ)²]
        std: Standard deviation σ
        skewness: Third standardized moment (shape)
        kurtosis: Fourth standardized moment (tailedness)
    """

    mean: float
    variance: float
    std: float
    skewness: float = 0.0
    kurtosis: float = 0.0


def compute_statistics(xs: Iterable[float]) -> DistributionStatistics:
    """Compute distribution statistics from samples.

    Parameters:
        xs: Sample values

    Returns:
        DistributionStatistics object
    """
    xs = np.array(list(xs), dtype=float)

    if len(xs) == 0:
        return DistributionStatistics(
            mean=float("nan"), variance=float("nan"), std=float("nan")
        )

    mean = float(np.mean(xs))
    variance = float(np.var(xs, ddof=1) if len(xs) > 1 else 0.0)
    std = float(np.sqrt(variance))

    if std > 0 and len(xs) > 2:
        # Standardized third moment
        skewness = float(np.mean(((xs - mean) / std) ** 3))
        # Standardized fourth moment
        kurtosis = float(np.mean(((xs - mean) / std) ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    return DistributionStatistics(
        mean=mean, variance=variance, std=std, skewness=skewness, kurtosis=kurtosis
    )


def mean(xs: Iterable[float]) -> float:
    """Compute sample mean μ̂ = (1/n) Σ x_i.

    Parameters:
        xs: Sample values

    Returns:
        Sample mean
    """
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs: Iterable[float], ddof: int = 1) -> float:
    """Compute sample variance s² = (1/(n-ddof)) Σ(x_i - μ̂)².

    Parameters:
        xs: Sample values
        ddof: Delta degrees of freedom (default: 1 for unbiased estimator)

    Returns:
        Sample variance

    Notes:
        - ddof=1 gives unbiased estimator of population variance
        - ddof=0 gives MLE (biased but lower MSE)
    """
    xs = list(xs)
    n = len(xs)

    if n <= ddof:
        return 0.0

    mu = mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (n - ddof)


def covariance(xs: Iterable[float], ys: Iterable[float], ddof: int = 1) -> float:
    """Compute sample covariance Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)].

    Parameters:
        xs: First variable samples
        ys: Second variable samples
        ddof: Delta degrees of freedom

    Returns:
        Sample covariance
    """
    xs = list(xs)
    ys = list(ys)

    if len(xs) != len(ys):
        raise ValueError("Sample sizes must match")

    n = len(xs)
    if n <= ddof:
        return 0.0

    mu_x = mean(xs)
    mu_y = mean(ys)

    return sum((x - mu_x) * (y - mu_y) for x, y in zip(xs, ys)) / (n - ddof)


def correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Compute Pearson correlation coefficient ρ = Cov(X,Y) / (σ_X σ_Y).

    Parameters:
        xs: First variable samples
        ys: Second variable samples

    Returns:
        Correlation coefficient in [-1, 1]

    Notes:
        - ρ = 1: perfect positive linear relationship
        - ρ = -1: perfect negative linear relationship
        - ρ = 0: no linear relationship (but may have nonlinear dependence)
    """
    xs = list(xs)
    ys = list(ys)

    cov = covariance(xs, ys)
    std_x = np.sqrt(variance(xs))
    std_y = np.sqrt(variance(ys))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def hoeffding_bound(n: int, delta: float, value_range: float = 1.0) -> float:
    """Compute Hoeffding's inequality bound for bounded random variables.

    For i.i.d. X_1, ..., X_n with X_i ∈ [a, b], Hoeffding's inequality states:
        P(|μ̂ - μ| ≥ ε) ≤ 2 exp(-2nε² / (b-a)²)

    This function returns ε such that the probability is at most δ.

    Parameters:
        n: Number of samples
        delta: Failure probability (typically 0.05 or 0.01)
        value_range: Range (b - a) of random variables

    Returns:
        Deviation bound ε

    Notes:
        - With probability ≥ 1-δ, we have |μ̂ - μ| ≤ ε
        - Critical for analyzing GNN generalization
        - Applies to bounded aggregations (e.g., normalized features)
    """
    if n <= 0:
        return float("inf")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")

    epsilon = value_range * np.sqrt(np.log(2.0 / delta) / (2 * n))
    return float(epsilon)


def bernstein_bound(
    n: int, delta: float, variance_bound: float, value_bound: float
) -> float:
    """Compute Bernstein's inequality bound (tighter than Hoeffding when variance is small).

    For i.i.d. X_1, ..., X_n with |X_i| ≤ M and Var(X_i) ≤ σ²:
        P(|μ̂ - μ| ≥ ε) ≤ 2 exp(-nε² / (2σ² + 2Mε/3))

    Parameters:
        n: Number of samples
        delta: Failure probability
        variance_bound: Upper bound on variance σ²
        value_bound: Upper bound on absolute value M

    Returns:
        Deviation bound ε

    Notes:
        - Better than Hoeffding when variance is small
        - Useful for analyzing GNN convergence with varying neighborhood sizes
    """
    if n <= 0:
        return float("inf")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")

    # Solve quadratic inequality (simplified approximation)
    log_term = np.log(2.0 / delta)
    epsilon = np.sqrt(
        2 * variance_bound * log_term / n
    ) + 2 * value_bound * log_term / (3 * n)

    return float(epsilon)


def empirical_distribution(
    xs: Iterable[float], bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical distribution from samples.

    Parameters:
        xs: Sample values
        bins: Number of histogram bins

    Returns:
        Tuple of (bin_centers, probabilities)
    """
    counts, edges = np.histogram(xs, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, counts


def concentration_of_sum(n: int, component_bound: float, delta: float) -> float:
    """Analyze concentration of sum of bounded random variables.

    For GNNs, this bounds deviation of neighborhood aggregation:
        Σ_{j ∈ N(i)} h_j

    Parameters:
        n: Number of terms (e.g., degree)
        component_bound: Bound on each term |h_j| ≤ B
        delta: Failure probability

    Returns:
        Concentration bound

    Notes:
        - Models aggregation uncertainty in GNNs
        - High-degree nodes have better concentration
        - Low-degree nodes have high variance in aggregation
    """
    return hoeffding_bound(n, delta, value_range=2 * component_bound)
