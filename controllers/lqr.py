"""Linear Quadratic Regulator (LQR) for 2-DOF quarter-car active suspension.

Includes rigorous derivation of the state-input cross-coupling term arising from
sprung mass vertical acceleration penalization in the quadratic cost functional.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.linalg import solve_continuous_are

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel


@dataclass
class LQRConfig:
    """LQR weighting parameters and metadata.

    Cost functional:
        J = integral_0^inf [ q1 * a_s^2 + q2 * (zs - zu)^2 + q3 * zu^2 + r0 * u^2 ] dt
    where:
        q1: Sprung mass vertical acceleration penalty [(m/s^2)^(-2)]
        q2: Suspension rattle-space deflection penalty [m^(-2)]
        q3: Tyre dynamic deflection penalty [m^(-2)]
        r0: Raw actuator control effort penalty [N^(-2)]
    """
    name: str
    q1: float  # Body acceleration weight
    q2: float  # Suspension deflection weight
    q3: float  # Tyre deflection weight
    r0: float  # Control effort weight
    description: str = ""

    def validate(self) -> None:
        """Validate weight positivity."""
        if self.q1 < 0.0 or self.q2 < 0.0 or self.q3 < 0.0:
            raise ValueError(f"State weights must be non-negative: q1={self.q1}, q2={self.q2}, q3={self.q3}")
        if self.r0 <= 0.0:
            raise ValueError(f"Control weight r0 must be strictly positive: r0={self.r0}")


class LQRController:
    """Continuous-Time Linear Quadratic Regulator with state-input cross-coupling.

    Mathematical Derivation:
        Sprung mass acceleration depends on both state x and control u:
            a_s = zs_ddot = C_a * x + D_a * u
        where:
            C_a = [-ks/ms, -cs/ms, ks/ms, cs/ms]
            D_a = 1/ms

        Suspension deflection:
            delta_s = zs - zu = C_sd * x,  where C_sd = [1, 0, -1, 0]

        Tyre deflection:
            delta_t = zu = C_td * x,       where C_td = [0, 0, 1, 0]

        Substituting into the cost functional J:
            a_s^2 = x^T (C_a^T C_a) x + 2 x^T (C_a^T D_a) u + u^T (D_a^2) u

        Total quadratic integrand:
            x^T Q x + 2 x^T N u + u^T R u
        where:
            Q = q1 * (C_a^T C_a) + q2 * (C_sd^T C_sd) + q3 * (C_td^T C_td)
            N = q1 * (C_a^T D_a)
            R = r0 + q1 * (D_a^2)

        Modified continuous Algebraic Riccati Equation (CARE):
            A* = A - B * R^(-1) * N^T
            Q* = Q - N * R^(-1) * N^T  (proven >= 0)
            A*^T P + P A* - P B R^(-1) B^T P + Q* = 0

        Optimal State Feedback Gain:
            K = R^(-1) * (B^T P + N^T)
            u = -K * x
    """

    def __init__(self, config: LQRConfig, vehicle_params: Optional[VehicleParameters] = None) -> None:
        self.config = config
        self.config.validate()
        self.params = vehicle_params or VehicleParameters()

        # Build quarter car model state matrices
        qc = QuarterCarModel(self.params)
        self.A = qc.A.copy()
        self.B = qc.B.copy()

        # Output observation matrices
        ms, ks, cs = self.params.ms, self.params.ks, self.params.cs
        self.C_a = np.array([[-ks / ms, -cs / ms, ks / ms, cs / ms]], dtype=np.float64)  # (1, 4)
        self.D_a = float(1.0 / ms)

        self.C_sd = np.array([[1.0, 0.0, -1.0, 0.0]], dtype=np.float64)  # (1, 4)
        self.C_td = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float64)   # (1, 4)

        # Formulate matrices and solve CARE
        self.K, self.P, self.eigenvalues, self.is_stable = self._compute_gain()

    def _compute_gain(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Compute the optimal feedback gain K by solving the modified continuous CARE."""
        q1 = self.config.q1
        q2 = self.config.q2
        q3 = self.config.q3
        r0 = self.config.r0

        # Construct generalized Q, N, R
        Q = (
            q1 * (self.C_a.T @ self.C_a)
            + q2 * (self.C_sd.T @ self.C_sd)
            + q3 * (self.C_td.T @ self.C_td)
        )
        N = q1 * (self.C_a.T * self.D_a)  # (4, 1)
        R_scalar = float(r0 + q1 * (self.D_a ** 2))
        R = np.array([[R_scalar]], dtype=np.float64)

        # Transformation to eliminate cross-term N
        # A* = A - B * (1/R) * N^T
        R_inv = 1.0 / R_scalar
        A_star = self.A - self.B @ (N.T * R_inv)
        Q_star = Q - (N * R_inv) @ N.T

        # Symmetrize Q_star to prevent numerical asymmetry
        Q_star = 0.5 * (Q_star + Q_star.T)

        try:
            P = solve_continuous_are(A_star, self.B, Q_star, R)
            K = R_inv * (self.B.T @ P + N.T)  # (1, 4)
        except Exception as e:
            raise RuntimeError(f"Riccati equation solver failed for {self.config.name}: {e}")

        # Check closed-loop stability: A_cl = A - B*K
        A_cl = self.A - self.B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        max_real_eigenvalue = float(np.max(np.real(eigenvalues)))
        is_stable = bool(max_real_eigenvalue < -1e-5)

        if not is_stable:
            raise ValueError(
                f"LQR solution for {self.config.name} is unstable! Max Re(lambda) = {max_real_eigenvalue:.4e}"
            )

        return K, P, eigenvalues, is_stable

    def compute_force(self, x: np.ndarray) -> float:
        """Compute state feedback force command: u = -K * x.

        Args:
            x: State vector [zs, zs_dot, zu, zu_dot]

        Returns:
            Commanded actuator force [N]
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(4, 1)
        u_cmd = -float((self.K @ x_vec).item())
        return u_cmd

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic controller information."""
        return {
            "name": self.config.name,
            "weights": {
                "q1_accel": self.config.q1,
                "q2_travel": self.config.q2,
                "q3_tyre": self.config.q3,
                "r0_effort": self.config.r0,
            },
            "gain_K": self.K.flatten().tolist(),
            "eigenvalues": [complex(e) for e in self.eigenvalues],
            "is_stable": self.is_stable,
            "max_real_pole": float(np.max(np.real(self.eigenvalues)))
        }
