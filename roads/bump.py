"""Speed bump road profile generator with physically smooth, C1-continuous profiles.

Uses versine / half-cosine hump geometry to represent traffic speed bumps and potholes
without introducing artificial step discontinuities into the numerical integrator.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from roads.base import BaseRoadProfile


class BumpRoadGenerator(BaseRoadProfile):
    """Speed bump road profile generator.

    Each bump is characterized by:
        start_dist: Longitudinal distance s0 where bump begins [m]
        height: Peak elevation H [m] (positive for bump, negative for pothole)
        width: Longitudinal length W of the bump [m]

    Profile formulation for s in [s0, s0 + W]:
        zr(s) = (H / 2) * (1 - cos(2*pi * (s - s0) / W))
        zr'(s) = (H*pi / W) * sin(2*pi * (s - s0) / W)
        zr(s) = 0 outside the bump interval.
    Both elevation and its derivative are zero at boundaries (C1 continuity).
    """

    def __init__(
        self,
        bumps: Optional[List[Tuple[float, float, float]]] = None,
        height: float = 0.05,
        width: float = 0.8,
        start_distance: float = 15.0,
        add_noise: bool = True,
        noise_amplitude: float = 0.001,
        seed: Optional[int] = 42,
        name: str = "Speed Bump Road"
    ) -> None:
        """Initialize bump generator.

        Args:
            bumps: Optional list of tuples (start_dist, height, width). If provided,
                   overrides the single bump parameters.
            height: Default bump height [m] (e.g. 0.05 m = 50 mm).
            width: Default bump longitudinal width [m] (e.g. 0.8 m).
            start_distance: Default distance to bump start [m].
            add_noise: If True, adds minor continuous background micro-roughness.
            noise_amplitude: Amplitude of micro-roughness [m].
            seed: Random seed for noise.
        """
        super().__init__(seed=seed, name=name)
        if bumps is not None:
            self.bumps = bumps
        else:
            self.bumps = [(start_distance, height, width)]

        self.add_noise = add_noise
        self.noise_amplitude = noise_amplitude

        if self.add_noise:
            rng = np.random.default_rng(self.seed)
            self._noise_phases = rng.uniform(0.0, 2.0 * np.pi, size=15)
            self._noise_freqs = np.linspace(0.1, 2.0, 15)

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Calculate road elevation for bump profile at distance(s) s."""
        is_scalar = np.isscalar(s)
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        elevation = np.zeros_like(s_arr)

        for s0, H, W in self.bumps:
            in_bump = (s_arr >= s0) & (s_arr <= (s0 + W))
            if np.any(in_bump):
                s_local = s_arr[in_bump] - s0
                elevation[in_bump] += (H / 2.0) * (1.0 - np.cos(2.0 * np.pi * s_local / W))

        # Add minor background irregularities if requested
        if self.add_noise:
            angles = 2.0 * np.pi * np.outer(self._noise_freqs, s_arr) + self._noise_phases[:, np.newaxis]
            micro = self.noise_amplitude * np.sum(np.sin(angles), axis=0) / len(self._noise_freqs)
            elevation += micro

        if is_scalar:
            return float(elevation[0])
        return elevation
