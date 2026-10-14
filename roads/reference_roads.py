"""Standard benchmark road profiles from Deshpande et al. (2014) and Chen & Huang (2005).

Provides exact analytical implementations of the three peer-reviewed road profiles:
1. ReferenceProfile1: Polynomial bump & trough with dual sinusoidal disturbance (Chen & Huang, 2005).
2. ReferenceProfile2: Harmonic excitation z(t) = 0.05 * cos(2*pi*t) * sin(0.6*pi*t).
3. ReferenceProfile3: Multi-frequency harmonic z(t) = 0.05 * sin(1.5*pi*t)*sin(0.15*pi*t) + 0.05 * cos(0.6*pi*t)*sin(0.3*pi*t).
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from roads.base import BaseRoadProfile


class ReferenceProfile1(BaseRoadProfile):
    """Road Profile 1 from Deshpande et al. (2014) and Chen & Huang (2005).

    Consists of a smooth polynomial bump around 5s and trough around 10s,
    with a continuous dual-frequency sinusoidal overlay:
    d(t) = 0.002 * sin(2*pi*t) + 0.002 * sin(7.5*pi*t)
    """

    def __init__(self, duration_s: float = 15.0, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed, name="reference_profile_1")
        self.duration_s = duration_s

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Spatial elevation mapping assuming 1 m/s nominal reference mapping."""
        return self._eval_profile(s)

    def elevation_at_time(self, t: float | np.ndarray) -> float | np.ndarray:
        """Exact analytical time-domain elevation from Eq. (39) of Deshpande et al. (2014)."""
        return self._eval_profile(t)

    def _eval_profile(self, t_in: float | np.ndarray) -> float | np.ndarray:
        is_scalar = np.isscalar(t_in)
        t_arr = np.atleast_1d(np.asarray(t_in, dtype=np.float64))
        z = np.zeros_like(t_arr)

        d = 0.002 * np.sin(2.0 * np.pi * t_arr) + 0.002 * np.sin(7.5 * np.pi * t_arr)
        z += d

        # Section 1: 3.5 <= t < 5.0
        m1 = (t_arr >= 3.5) & (t_arr < 5.0)
        t1 = t_arr[m1] - 3.5
        z[m1] += -0.0592 * (t1**3) + 0.1332 * (t1**2)

        # Section 2: 5.0 <= t < 6.5
        m2 = (t_arr >= 5.0) & (t_arr < 6.5)
        t2 = t_arr[m2] - 6.5
        z[m2] += 0.0592 * (t2**3) + 0.1332 * (t2**2)

        # Section 3: 8.5 <= t < 10.0
        m3 = (t_arr >= 8.5) & (t_arr < 10.0)
        t3 = t_arr[m3] - 8.5
        z[m3] += 0.0592 * (t3**3) - 0.1332 * (t3**2)

        # Section 4: 10.0 <= t < 11.5
        m4 = (t_arr >= 10.0) & (t_arr < 11.5)
        t4 = t_arr[m4] - 11.5
        z[m4] += -0.0592 * (t4**3) - 0.1332 * (t4**2)

        if is_scalar:
            return float(z[0])
        return z

    def get_time_callable(self, vehicle_speed_mps: float = 1.0) -> Callable[[float], float]:
        return lambda t: float(self.elevation_at_time(t))


class ReferenceProfile2(BaseRoadProfile):
    """Road Profile 2 from Deshpande et al. (2014), Eq. (40).

    Harmonic multi-frequency profile:
    z(t) = 0.05 * cos(2*pi*t) * sin(0.6*pi*t)
    """

    def __init__(self, duration_s: float = 15.0, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed, name="reference_profile_2")
        self.duration_s = duration_s

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        return self.elevation_at_time(s)

    def elevation_at_time(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        val = 0.05 * np.cos(2.0 * np.pi * t_arr) * np.sin(0.6 * np.pi * t_arr)
        if np.isscalar(t):
            return float(val)
        return val

    def get_time_callable(self, vehicle_speed_mps: float = 1.0) -> Callable[[float], float]:
        return lambda t: float(self.elevation_at_time(t))


class ReferenceProfile3(BaseRoadProfile):
    """Road Profile 3 from Deshpande et al. (2014), Eq. (41).

    Combined multi-harmonic profile:
    z(t) = 0.05 * sin(1.5*pi*t) * sin(0.15*pi*t) + 0.05 * cos(0.6*pi*t) * sin(0.3*pi*t)
    """

    def __init__(self, duration_s: float = 15.0, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed, name="reference_profile_3")
        self.duration_s = duration_s

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        return self.elevation_at_time(s)

    def elevation_at_time(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        term1 = 0.05 * np.sin(1.5 * np.pi * t_arr) * np.sin(0.15 * np.pi * t_arr)
        term2 = 0.05 * np.cos(0.6 * np.pi * t_arr) * np.sin(0.3 * np.pi * t_arr)
        val = term1 + term2
        if np.isscalar(t):
            return float(val)
        return val

    def get_time_callable(self, vehicle_speed_mps: float = 1.0) -> Callable[[float], float]:
        return lambda t: float(self.elevation_at_time(t))
