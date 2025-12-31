"""Graph Fourier transform and frequency interpretation.

This module implements the Graph Fourier Transform (GFT), which extends
classical Fourier analysis to graph-structured data.

Key insight:
    Laplacian eigenvectors play the role of Fourier basis functions.
    Low eigenvalues = low frequencies (smooth signals)
    High eigenvalues = high frequencies (oscillatory signals)

Mathematical framework:
    Graph Fourier Transform: ŝ = Vᵀs
    Inverse GFT: s = Vŝ

    where V = [v₁ | v₂ | ... | vₙ] are Laplacian eigenvectors.

Frequency interpretation:
    Frequency of eigenvector vᵢ = corresponding eigenvalue λᵢ

    Smoothness: sᵀLs = Σᵢ λᵢ ŝᵢ²

    - Low λ → smooth (varies little across edges)
    - High λ → rough (varies rapidly)

GNN connection:
    - Spectral convolution = filtering in Fourier domain
    - GCN = low-pass filter (suppresses high frequencies)
    - Oversmoothing = over-filtering (only DC component remains)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FourierCoefficients:
    """Graph Fourier coefficients.

    Attributes:
        coefficients: Fourier coefficients ŝ = Vᵀs
        eigenvalues: Frequencies (Laplacian eigenvalues)
        signal_energy: ‖s‖² = Σᵢ ŝᵢ²
    """

    coefficients: np.ndarray
    eigenvalues: np.ndarray

    @property
    def signal_energy(self) -> float:
        """Total signal energy: ‖s‖² = Σᵢ |ŝᵢ|²."""
        return float(np.sum(np.abs(self.coefficients) ** 2))

    @property
    def frequency_energy(self) -> np.ndarray:
        """Energy per frequency: |ŝᵢ|²."""
        return np.abs(self.coefficients) ** 2

    def low_frequency_energy(self, cutoff_freq: float) -> float:
        """Energy in frequencies below cutoff.

        Interpretation:
            Low-frequency energy = smoothness of signal.
        """
        low_freq_mask = self.eigenvalues <= cutoff_freq
        return float(np.sum(self.frequency_energy[low_freq_mask]))

    def high_frequency_energy(self, cutoff_freq: float) -> float:
        """Energy in frequencies above cutoff.

        Interpretation:
            High-frequency energy = roughness of signal.
        """
        high_freq_mask = self.eigenvalues > cutoff_freq
        return float(np.sum(self.frequency_energy[high_freq_mask]))


def gft(
    signal: np.ndarray, eigvecs: np.ndarray, eigenvalues: Optional[np.ndarray] = None
) -> FourierCoefficients:
    """Graph Fourier Transform: project signal onto Laplacian eigenbasis.

    Mathematical operation:
        ŝ = Vᵀs

    where V = [v₁ | ... | vₙ] are Laplacian eigenvectors.

    Interpretation:
        ŝᵢ = ⟨s, vᵢ⟩ = component of signal along eigenvector vᵢ

    Properties:
        - Parseval's identity: ‖s‖² = ‖ŝ‖² (energy preserved)
        - Invertible: can reconstruct s from ŝ
        - Basis change: eigenbasis is orthonormal

    Frequency interpretation:
        - ŝ₁ = DC component (projection on constant vector)
        - ŝᵢ for large λᵢ = high-frequency components

    GNN connection:
        - Spectral GNN filters work in this domain
        - ŝ reveals which frequencies are present
        - Low-pass filtering = zeroing high-frequency ŝᵢ

    Args:
        signal: Node signal (n,) or (n, d)
        eigvecs: Laplacian eigenvectors (n, n)
        eigenvalues: Laplacian eigenvalues (n,) for frequency info

    Returns:
        FourierCoefficients object
    """
    coeffs = eigvecs.T @ signal

    if eigenvalues is None:
        eigenvalues = np.arange(len(coeffs))  # Fallback: use indices

    return FourierCoefficients(coefficients=coeffs, eigenvalues=eigenvalues)


def inverse_gft(fourier_coeffs: FourierCoefficients, eigvecs: np.ndarray) -> np.ndarray:
    """Inverse Graph Fourier Transform: reconstruct signal from coefficients.

    Mathematical operation:
        s = Vŝ = Σᵢ ŝᵢ vᵢ

    Interpretation:
        Signal is linear combination of eigenvectors,
        weighted by Fourier coefficients.

    Properties:
        - Exact reconstruction: inverse_gft(gft(s)) = s
        - Linear: inverse is also a matrix multiplication

    Args:
        fourier_coeffs: Fourier coefficients from gft()
        eigvecs: Laplacian eigenvectors (same as used in gft)

    Returns:
        Reconstructed signal
    """
    return eigvecs @ fourier_coeffs.coefficients


def signal_smoothness(signal: np.ndarray, L: np.ndarray) -> float:
    """Measure signal smoothness using Laplacian quadratic form.

    Mathematical definition:
        Smoothness(s) = sᵀLs = Σ_{(i,j)∈E} (sᵢ - sⱼ)²

    In Fourier domain:
        sᵀLs = ŝᵀΛŝ = Σᵢ λᵢ ŝᵢ²

    Interpretation:
        - Small value → signal is smooth (nearby nodes have similar values)
        - Large value → signal is rough (varies across edges)
        - Zero → constant signal

    GNN connection:
        - Oversmoothing minimizes this quantity
        - GCN acts as low-pass filter (reduces smoothness measure)

    Args:
        signal: Node signal (n,) or (n, d)
        L: Graph Laplacian

    Returns:
        Smoothness value (scalar or per-dimension)
    """
    if signal.ndim == 1:
        return float(signal @ L @ signal)
    else:
        # Multi-dimensional signal: compute per dimension
        return np.array(
            [signal[:, i] @ L @ signal[:, i] for i in range(signal.shape[1])]
        )


def low_pass_filter(
    signal: np.ndarray, eigvecs: np.ndarray, eigenvalues: np.ndarray, cutoff_freq: float
) -> np.ndarray:
    """Low-pass filter: remove high-frequency components.

    Mathematical operation:
        1. Compute ŝ = Vᵀs (GFT)
        2. Zero out ŝᵢ where λᵢ > cutoff
        3. Reconstruct: s_filtered = V ŝ_filtered

    Effect:
        - Smooths signal on graph
        - Removes rapid variations
        - Nearby nodes become more similar

    GNN connection:
        - GCN is approximately a low-pass filter
        - Oversmoothing = extreme low-pass filtering (only DC remains)

    Args:
        signal: Input signal
        eigvecs: Laplacian eigenvectors
        eigenvalues: Laplacian eigenvalues
        cutoff_freq: Frequency threshold

    Returns:
        Filtered (smooth) signal
    """
    # Forward GFT
    fourier = gft(signal, eigvecs, eigenvalues)

    # Apply filter: zero out high frequencies
    filtered_coeffs = fourier.coefficients.copy()
    high_freq_mask = eigenvalues > cutoff_freq
    filtered_coeffs[high_freq_mask] = 0.0

    # Inverse GFT
    filtered_fourier = FourierCoefficients(
        coefficients=filtered_coeffs, eigenvalues=eigenvalues
    )

    return inverse_gft(filtered_fourier, eigvecs)


def high_pass_filter(
    signal: np.ndarray, eigvecs: np.ndarray, eigenvalues: np.ndarray, cutoff_freq: float
) -> np.ndarray:
    """High-pass filter: remove low-frequency components.

    Effect:
        - Emphasizes rapid variations
        - Removes smooth trends
        - Detects edges/boundaries

    GNN connection:
        - Opposite of GCN
        - Useful for edge detection on graphs

    Args:
        signal: Input signal
        eigvecs: Laplacian eigenvectors
        eigenvalues: Laplacian eigenvalues
        cutoff_freq: Frequency threshold

    Returns:
        Filtered signal (emphasizing high frequencies)
    """
    fourier = gft(signal, eigvecs, eigenvalues)

    filtered_coeffs = fourier.coefficients.copy()
    low_freq_mask = eigenvalues <= cutoff_freq
    filtered_coeffs[low_freq_mask] = 0.0

    filtered_fourier = FourierCoefficients(
        coefficients=filtered_coeffs, eigenvalues=eigenvalues
    )

    return inverse_gft(filtered_fourier, eigvecs)


def band_pass_filter(
    signal: np.ndarray,
    eigvecs: np.ndarray,
    eigenvalues: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> np.ndarray:
    """Band-pass filter: keep only frequencies in specified range.

    Args:
        signal: Input signal
        eigvecs: Laplacian eigenvectors
        eigenvalues: Laplacian eigenvalues
        low_cutoff: Lower frequency bound
        high_cutoff: Upper frequency bound

    Returns:
        Filtered signal
    """
    fourier = gft(signal, eigvecs, eigenvalues)

    filtered_coeffs = fourier.coefficients.copy()
    outside_band = (eigenvalues < low_cutoff) | (eigenvalues > high_cutoff)
    filtered_coeffs[outside_band] = 0.0

    filtered_fourier = FourierCoefficients(
        coefficients=filtered_coeffs, eigenvalues=eigenvalues
    )

    return inverse_gft(filtered_fourier, eigvecs)


def spectral_filter_response(
    filter_func: Callable[[float], float],
    eigvecs: np.ndarray,
    eigenvalues: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Apply arbitrary spectral filter defined by frequency response.

    Mathematical operation:
        s_filtered = V · g(Λ) · Vᵀs

        where g(λ) is the filter function.

    Examples:
        - Low-pass: g(λ) = 1 if λ < cutoff else 0
        - Heat kernel: g(λ) = exp(-τλ)
        - GCN approximation: g(λ) ≈ 1 - λ (for small λ)

    Args:
        filter_func: Function g: ℝ → ℝ defining filter
        eigvecs: Laplacian eigenvectors
        eigenvalues: Laplacian eigenvalues
        signal: Input signal

    Returns:
        Filtered signal
    """
    # Forward GFT
    fourier = gft(signal, eigvecs, eigenvalues)

    # Apply filter in frequency domain
    filtered_coeffs = np.array(
        [
            filter_func(lam) * coeff
            for lam, coeff in zip(eigenvalues, fourier.coefficients)
        ]
    )

    filtered_fourier = FourierCoefficients(
        coefficients=filtered_coeffs, eigenvalues=eigenvalues
    )

    # Inverse GFT
    return inverse_gft(filtered_fourier, eigvecs)


def heat_diffusion(
    signal: np.ndarray, eigvecs: np.ndarray, eigenvalues: np.ndarray, time: float
) -> np.ndarray:
    """Heat diffusion on graph: s(t) = exp(-tL)s(0).

    Mathematical model:
        ∂s/∂t = -Ls

        Solution: s(t) = exp(-tL)s(0)

        In Fourier domain:
        ŝ(t) = exp(-tΛ)ŝ(0) = diag(exp(-tλ₁), ..., exp(-tλₙ)) ŝ(0)

    Interpretation:
        - Heat spreads from hot nodes to neighbors
        - Exponential decay rate depends on frequency
        - High frequencies decay faster!

    GNN connection:
        - Approximation to message passing diffusion
        - Time t analogous to number of GNN layers
        - As t → ∞, s(t) → constant (oversmoothing)

    Args:
        signal: Initial heat distribution
        eigvecs: Laplacian eigenvectors
        eigenvalues: Laplacian eigenvalues
        time: Diffusion time

    Returns:
        Heat distribution at time t
    """

    def heat_kernel(lam):
        return np.exp(-time * lam)

    return spectral_filter_response(heat_kernel, eigvecs, eigenvalues, signal)


def analyze_frequency_content(
    signal: np.ndarray, eigvecs: np.ndarray, eigenvalues: np.ndarray
) -> dict:
    """Analyze frequency content of graph signal.

    Returns:
        Dictionary with frequency analysis results
    """
    fourier = gft(signal, eigvecs, eigenvalues)

    # Find dominant frequency
    energy = fourier.frequency_energy
    dominant_idx = np.argmax(energy)

    # Compute energy distribution
    total_energy = fourier.signal_energy
    energy_fraction = energy / total_energy if total_energy > 0 else energy

    # Low/high frequency split at median eigenvalue
    median_freq = np.median(eigenvalues[eigenvalues > 1e-10])  # Exclude zeros
    low_energy = fourier.low_frequency_energy(median_freq)
    high_energy = fourier.high_frequency_energy(median_freq)

    return {
        "total_energy": total_energy,
        "dominant_frequency": float(eigenvalues[dominant_idx]),
        "dominant_frequency_energy": float(energy[dominant_idx]),
        "low_frequency_energy": low_energy,
        "high_frequency_energy": high_energy,
        "low_to_high_ratio": low_energy / high_energy
        if high_energy > 0
        else float("inf"),
        "frequency_entropy": -np.sum(energy_fraction * np.log(energy_fraction + 1e-10)),
    }
