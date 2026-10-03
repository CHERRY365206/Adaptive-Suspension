"""Base class and common interface for road profile generators.

Defines spatial road elevation r(s) and time-domain conversion zr(t) = r(v * t).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np


class BaseRoadProfile(ABC):
    """Abstract base class for all road profile generators.

    Longitudinal vehicle coordinate is s = v * t [m], where v is vehicle forward speed [m/s].
    Vertical road elevation is zr(t) = r(s(t)) = r(v * t) [m].
    """

    def __init__(self, seed: Optional[int] = None, name: str = "BaseRoad") -> None:
        self.seed = seed
        self.name = name

    @abstractmethod
    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Calculate road surface vertical elevation at longitudinal distance s.

        Args:
            s: Longitudinal distance along road [m] (scalar or numpy array).

        Returns:
            Vertical road elevation zr [m].
        """
        pass

    def elevation_time(self, t: float | np.ndarray, speed_mps: float) -> float | np.ndarray:
        """Calculate road surface vertical elevation at time t given forward speed.

        Args:
            t: Time [s] (scalar or numpy array).
            speed_mps: Vehicle longitudinal speed [m/s].

        Returns:
            Vertical road displacement zr [m].
        """
        s = np.asarray(speed_mps, dtype=np.float64) * np.asarray(t, dtype=np.float64)
        res = self.elevation_spatial(s)
        if np.isscalar(t):
            return float(res)
        return res

    def get_time_callable(self, speed_mps: float) -> Callable[[float], float]:
        """Return a callable zr(t) for use in ODE integration."""
        return lambda t: float(self.elevation_time(t, speed_mps))

    def generate_profile(self, duration: float, dt: float, speed_mps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate time, distance, and road elevation vectors.

        Args:
            duration: Simulation duration [s].
            dt: Sampling interval [s].
            speed_mps: Vehicle forward speed [m/s].

        Returns:
            Tuple of (time_array, distance_array, elevation_array).
        """
        t = np.arange(0.0, duration + dt / 2.0, dt)
        s = speed_mps * t
        zr = np.asarray(self.elevation_spatial(s), dtype=np.float64)
        return t, s, zr
