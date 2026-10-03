"""Rough road profile generator based on ISO 8608 Class C / Class D stochastic road standards.

Represents degraded asphalt, cobblestone, or uneven unpaved surfaces with high stochastic roughness.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from roads.base import BaseRoadProfile


class RoughRoadGenerator(BaseRoadProfile):
    """ISO 8608 Class C / Class D rough road profile generator.

    Synthesizes stochastic terrain roughness from ISO 8608 power spectral density:
        G_d(n) = G_d(n0) * (n / n0)^(-w)
    where:
        n0 = 0.1 cycles/m
        w = 2.0 (waviness exponent)
        G_d(n0):
            Class C (poor / deteriorating asphalt): 64e-6 m^3
            Class D (very poor / gravel track): 256e-6 m^3
    """

    CLASS_PSD = {
        "B": 16.0e-6,
        "C": 64.0e-6,
        "D": 256.0e-6,
    }

    def __init__(
        self,
        iso_class: str = "C",
        seed: Optional[int] = 42,
        roughness_scale: float = 1.0,
        n_harmonics: int = 150,
        n_min: float = 0.05,
        n_max: float = 4.0,
        name: Optional[str] = None
    ) -> None:
        name = name or f"Rough Road (ISO Class {iso_class})"
        super().__init__(seed=seed, name=name)
        self.iso_class = iso_class.upper()
        if self.iso_class not in self.CLASS_PSD:
            raise ValueError(f"Unknown ISO roughness class: {iso_class}. Choose from 'B', 'C', 'D'.")

        self.roughness_scale = roughness_scale
        self.n_harmonics = n_harmonics
        self.n_min = n_min
        self.n_max = n_max

        self.G0 = self.CLASS_PSD[self.iso_class] * roughness_scale
        self.n0 = 0.1
        self.w = 2.0

        self._initialize_harmonics()

    def _initialize_harmonics(self) -> None:
        """Precompute harmonic frequencies, amplitudes, and randomized phases."""
        rng = np.random.default_rng(self.seed)
        self.frequencies = np.linspace(self.n_min, self.n_max, self.n_harmonics)
        delta_n = (self.n_max - self.n_min) / self.n_harmonics

        psd = self.G0 * (self.frequencies / self.n0) ** (-self.w)
        self.amplitudes = np.sqrt(2.0 * psd * delta_n)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.n_harmonics)

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Evaluate rough road surface elevation at distance(s) s."""
        is_scalar = np.isscalar(s)
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

        angles = 2.0 * np.pi * np.outer(self.frequencies, s_arr) + self.phases[:, np.newaxis]
        elevation = np.sum(self.amplitudes[:, np.newaxis] * np.sin(angles), axis=0)

        if is_scalar:
            return float(elevation[0])
        return elevation
