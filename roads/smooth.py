"""Smooth road profile generator based on ISO 8608 Class A road roughness standards.

Represents a high-quality asphalt highway with low-amplitude, band-limited stochastic irregularities.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from roads.base import BaseRoadProfile


class SmoothRoadGenerator(BaseRoadProfile):
    """ISO 8608 Class A smooth highway road profile generator.

    Uses spatial harmonic superposition with uniform random phases:
        zr(s) = sum_{i=1}^N A_i * sin(2*pi * n_i * s + phi_i)
    where:
        n_i: Spatial frequency [cycles/m]
        A_i = sqrt(2 * G_d(n_i) * delta_n) [m]
        G_d(n) = G_d(n0) * (n / n0)^(-w) [m^3/cycle] (ISO road PSD)
    """

    def __init__(
        self,
        seed: Optional[int] = 42,
        roughness_factor: float = 1.0,
        n_harmonics: int = 100,
        n_min: float = 0.05,
        n_max: float = 3.0,
        name: str = "Smooth Road (ISO Class A)"
    ) -> None:
        super().__init__(seed=seed, name=name)
        self.roughness_factor = roughness_factor
        self.n_harmonics = n_harmonics
        self.n_min = n_min
        self.n_max = n_max

        # ISO 8608 Class A nominal PSD at n0 = 0.1 cycles/m is 4e-6 m^3
        self.G0 = 4.0e-6 * roughness_factor
        self.n0 = 0.1
        self.w = 2.0  # Waviness index

        self._initialize_harmonics()

    def _initialize_harmonics(self) -> None:
        """Precompute spatial frequencies, amplitudes, and reproducible phases."""
        rng = np.random.default_rng(self.seed)

        # Logarithmic or linear spacing of spatial frequencies
        self.frequencies = np.linspace(self.n_min, self.n_max, self.n_harmonics)
        delta_n = (self.n_max - self.n_min) / self.n_harmonics

        # PSD G_d(n) for each harmonic
        psd = self.G0 * (self.frequencies / self.n0) ** (-self.w)
        self.amplitudes = np.sqrt(2.0 * psd * delta_n)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.n_harmonics)

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Evaluate smooth road elevation at longitudinal distance(s) s."""
        is_scalar = np.isscalar(s)
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        # Vectorized calculation: s_arr shape (M,), frequencies shape (N,)
        # angles shape (N, M)
        angles = 2.0 * np.pi * np.outer(self.frequencies, s_arr) + self.phases[:, np.newaxis]
        elevation = np.sum(self.amplitudes[:, np.newaxis] * np.sin(angles), axis=0)

        if is_scalar:
            return float(elevation[0])
        return elevation
