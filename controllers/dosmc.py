"""Disturbance Observer-based Sliding Mode Control (DOSMC) for Active Suspension Systems.

Implements the exact formulation from:
Vaijayanti S. Deshpande, B. Mohan, P.D. Shendge, S.B. Phadke,
"Disturbance observer based sliding mode control of active suspension systems",
Journal of Sound and Vibration, Vol. 333, Issue 11, 2014, pp. 2281-2296.
DOI: 10.1016/j.jsv.2014.01.023.

Key Attributes:
1. Online Disturbance Observer (DO) estimates the lumped uncertainty e(x, t) in real time
   (including unknown road disturbance, suspension spring/damper non-linearities, and mass variations).
2. Uses strictly sprung mass states (x1 = zs, x2 = zs_dot).
   Obviates the need for unsprung mass sensors (wheel position/velocity) or preview sensors.
3. Proven Uniform Ultimate Boundedness (UUB) stability via Lyapunov analysis.
4. Two operational variants:
   - DOSMC without u_s (k_st = 0): pure disturbance observer + equivalent control.
   - DOSMC with u_s (k_st > 0): adds boundary-layer sliding component to suppress residual estimation errors.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from controllers.saturation import saturate_force, rate_limit_force
from models.vehicle_parameters import VehicleParameters


@dataclass
class DOSMCConfig:
    """Configuration parameters for Disturbance Observer-based Sliding Mode Control."""
    S: float = 4.0           # Sliding surface slope parameter sigma = S*x1 + x2 (Eq. 12)
    k: float = 10.0          # Linear sliding gain in u_eq (Eq. 20)
    W: float = 100.0         # Disturbance observer gain partial(p)/partial(sigma) (Section 6)
    k_st: float = 0.5        # Discontinuous switching gain for u_s (Eq. 22; set to 0.0 for DOSMC without u_s)
    epsilon: float = 0.05    # Boundary layer thickness for smooth sat(sigma) (Eq. 23)
    dt: float = 0.002        # Controller sampling step [s] (500 Hz)
    max_force: float = 1800.0       # Actuator maximum positive force [N]
    min_force: float = -1800.0      # Actuator maximum negative force [N]
    slew_rate_limit: float = 18000.0 # Actuator force rate limit [N/s]
    name: str = "dosmc"
    description: str = "Disturbance Observer-based Sliding Mode Controller"

    def validate(self) -> None:
        """Validate parameter positivity and physical feasibility."""
        if self.S <= 0.0:
            raise ValueError(f"Sliding slope S must be positive, got {self.S}")
        if self.k <= 0.0:
            raise ValueError(f"Sliding gain k must be positive, got {self.k}")
        if self.W <= 0.0:
            raise ValueError(f"Observer gain W must be positive, got {self.W}")
        if self.k_st < 0.0:
            raise ValueError(f"Switching gain k_st cannot be negative, got {self.k_st}")
        if self.epsilon <= 0.0:
            raise ValueError(f"Boundary layer epsilon must be positive, got {self.epsilon}")
        if self.dt <= 0.0:
            raise ValueError(f"Sampling step dt must be positive, got {self.dt}")


class DOSMCController:
    """Disturbance Observer-based Sliding Mode Controller for active suspension.

    Implements the real-time observer state update and control force computation.
    """

    def __init__(
        self,
        config: Optional[DOSMCConfig] = None,
        vehicle_params: Optional[VehicleParameters] = None,
        actuator: Optional[ActuatorModel] = None
    ) -> None:
        self.config = config or DOSMCConfig()
        self.config.validate()
        self.vehicle_params = vehicle_params or VehicleParameters()
        if actuator is not None:
            self.actuator = actuator
        else:
            from models.actuator import ActuatorModel
            self.actuator = ActuatorModel(
                max_force=self.config.max_force,
                min_force=self.config.min_force,
                slew_rate_limit=self.config.slew_rate_limit
            )

        # Nominal reciprocal sprung mass Gm = 1 / ms [kg^-1] (Eq. 16)
        self.Gm = 1.0 / self.vehicle_params.ms

        # Internal observer state d_hat (Eq. 27)
        self.d_hat: float = 0.0
        self.prev_u_applied: float = 0.0

    def reset(self, initial_d_hat: float = 0.0) -> None:
        """Reset internal observer and actuator states for a new simulation."""
        self.d_hat = float(initial_d_hat)
        self.prev_u_applied = 0.0
        if hasattr(self.actuator, "reset"):
            self.actuator.reset()

    def sat(self, sigma: float) -> float:
        """Smooth boundary-layer saturation function sat(sigma) (Eq. 23)."""
        eps = self.config.epsilon
        if abs(sigma) > eps:
            return float(np.sign(sigma))
        return float(sigma / eps)

    def compute_force(
        self,
        x: np.ndarray,
        current_road_elev: float = 0.0,
        dt: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the active control force from measured state.

        Args:
            x: State vector [zs, zs_dot, zu, zu_dot].
               Crucially, DOSMC uses strictly x[0] (zs) and x[1] (zs_dot).
            current_road_elev: Road elevation (not required by DOSMC, accepted for interface uniformity).
            dt: Optional integration step override.

        Returns:
            Tuple of (requested_force, info_dict).
        """
        dt_step = dt or self.config.dt
        x1 = float(x[0])  # Sprung mass displacement zs
        x2 = float(x[1])  # Sprung mass velocity zs_dot

        # 1. Evaluate sliding variable sigma (Eq. 12)
        sigma = self.config.S * x1 + x2

        # 2. Evaluate saturation function (Eq. 23)
        sat_sigma = self.sat(sigma)

        # 3. Estimate lumped uncertainty e_hat = d_hat + W * sigma (Eq. 27 with p(sigma) = W*sigma)
        e_hat = self.d_hat + self.config.W * sigma

        # 4. Equivalent control u_eq (Eq. 20)
        u_eq = -(1.0 / self.Gm) * (self.config.S * x2 + self.config.k * sigma)

        # 5. Disturbance cancellation control u_n (Eq. 21)
        u_n = -(1.0 / self.Gm) * e_hat

        # 6. Discontinuous / boundary layer component u_s (Eq. 22)
        u_s = -(1.0 / self.Gm) * self.config.k_st * sat_sigma

        # 7. Total requested control force u = u_eq + u_n + u_s (Eq. 19)
        u_requested = float(u_eq + u_n + u_s)

        # 8. Actuator rate limiting and force clamping
        if hasattr(self.actuator, "apply_limits"):
            _, u_applied, _ = self.actuator.apply_limits(u_requested, dt=dt_step)
        else:
            u_lim = rate_limit_force(u_requested, self.prev_u_applied, self.config.slew_rate_limit, dt_step)
            u_applied = saturate_force(u_lim, self.config.min_force, self.config.max_force)
        self.prev_u_applied = u_applied

        # 9. Update disturbance observer state d_hat via ODE (Eq. 30):
        # dot(d_hat) = W * (k * sigma + k_st * sat(sigma))
        d_dot = self.config.W * (self.config.k * sigma + self.config.k_st * sat_sigma)
        self.d_hat += dt_step * d_dot

        info = {
            "sigma": float(sigma),
            "e_hat": float(e_hat),
            "d_hat": float(self.d_hat),
            "u_eq": float(u_eq),
            "u_n": float(u_n),
            "u_s": float(u_s),
            "u_requested": float(u_requested),
            "u_applied": float(u_applied),
            "controller_name": self.config.name
        }
        return u_requested, info

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        zr: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Callable interface matching QuarterCarModel.simulate callback specification."""
        u_req, info = self.compute_force(x, current_road_elev=zr)
        return u_req, info["u_applied"], info
