"""Model Predictive Control (MPC) for 2-DOF quarter-car active suspension.

Implements discrete-time state-space predictive control with state-input cross-coupling,
explicit actuator force saturation, and slew-rate constraint handling.
Solves a convex Quadratic Program (QP) using precomputed condensed prediction matrices
and analytical gradient formulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
import time
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize, LinearConstraint

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel


@dataclass
class MPCConfig:
    """Configuration weights and constraint parameters for MPC."""
    name: str = "nominal_mpc"
    prediction_horizon: int = 12       # Prediction horizon steps (Np)
    dt: float = 0.002                  # Discretization and sampling interval [s]
    q1: float = 5.0e4                  # Sprung mass vertical acceleration weight
    q2: float = 2.0e4                  # Suspension deflection (rattle space) weight
    q3: float = 2.0e4                  # Tyre dynamic deflection weight
    r0: float = 2.0e-5                 # Actuator control effort penalty
    r_rate: float = 1.0e-7             # Actuator slew-rate penalty
    max_force: float = 1800.0          # Actuator peak tensile/compressive limit [N]
    min_force: float = -1800.0         # Actuator minimum force limit [N]
    slew_rate_limit: float = 18000.0   # Actuator maximum slew rate [N/s]
    use_preview: bool = False          # Whether to use road preview lookahead
    max_iter: int = 50                 # Maximum QP iterations
    ftol: float = 1.0e-5               # QP solver tolerance
    description: str = ""

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.prediction_horizon < 2:
            raise ValueError(f"prediction_horizon must be >= 2, got {self.prediction_horizon}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {self.dt}")
        if self.q1 < 0.0 or self.q2 < 0.0 or self.q3 < 0.0:
            raise ValueError(f"State weights must be non-negative: q1={self.q1}, q2={self.q2}, q3={self.q3}")
        if self.r0 <= 0.0:
            raise ValueError(f"Control weight r0 must be strictly positive, got {self.r0}")
        if self.r_rate < 0.0:
            raise ValueError(f"Rate weight r_rate must be non-negative, got {self.r_rate}")
        if self.max_force <= 0.0 or self.min_force >= self.max_force:
            raise ValueError(f"Invalid force limits: min={self.min_force}, max={self.max_force}")
        if self.slew_rate_limit <= 0.0:
            raise ValueError(f"slew_rate_limit must be strictly positive, got {self.slew_rate_limit}")


class MPCController:
    """Constrained Discrete-Time Linear Model Predictive Controller.

    Solves the receding-horizon finite-time optimal control problem:
        min_U 0.5 * U^T * H * U + f(x0, W, u_prev)^T * U
    subject to:
        u_min <= U <= u_max
        -du_max <= D * U - d_prev * u_prev <= du_max

    Uses an exact analytical QP formulation with precomputed Hessian and warm-starting.
    """

    def __init__(self, config: Optional[MPCConfig] = None, vehicle_params: Optional[VehicleParameters] = None) -> None:
        self.config = config or MPCConfig()
        self.config.validate()
        self.params = vehicle_params or VehicleParameters()

        # Build continuous-time model
        qc = QuarterCarModel(self.params)
        self.A_c = qc.A.copy()
        self.B_c = qc.B.copy()
        self.E_c = qc.E.copy()

        # Output observation matrices
        ms, ks, cs = self.params.ms, self.params.ks, self.params.cs
        self.C_a = np.array([[-ks / ms, -cs / ms, ks / ms, cs / ms]], dtype=np.float64)  # (1, 4)
        self.D_a = float(1.0 / ms)
        self.C_sd = np.array([[1.0, 0.0, -1.0, 0.0]], dtype=np.float64)                  # (1, 4)
        self.C_td = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float64)                   # (1, 4)

        # Discretize state space
        self.A_d, self.B_d, self.E_d = self._discretize_plant(self.config.dt)

        # Precompute prediction and cost matrices
        self._build_prediction_matrices()

        # Internal controller state
        self.last_applied_force = 0.0
        self.solve_times: List[float] = []
        self.unconstrained_hits = 0
        self.qp_solves = 0
        self.qp_failures = 0

    def _discretize_plant(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discretize continuous-time state-space using exact zero-order hold (ZOH) via matrix exponential."""
        n_x = self.A_c.shape[0]  # 4
        # Augmented system matrix [A, B, E; 0, 0, 0; 0, 0, 0]
        M = np.zeros((n_x + 2, n_x + 2), dtype=np.float64)
        M[:n_x, :n_x] = self.A_c
        M[:n_x, n_x:n_x + 1] = self.B_c
        M[:n_x, n_x + 1:n_x + 2] = self.E_c

        expM = expm(M * dt)
        A_d = expM[:n_x, :n_x]
        B_d = expM[:n_x, n_x:n_x + 1]
        E_d = expM[:n_x, n_x + 1:n_x + 2]
        return A_d, B_d, E_d

    def _build_prediction_matrices(self) -> None:
        """Precompute condensed prediction matrices, cost weights, and Hessian."""
        Np = self.config.prediction_horizon
        q1 = self.config.q1
        q2 = self.config.q2
        q3 = self.config.q3
        r0 = self.config.r0
        r_rate = self.config.r_rate

        # Stage cost matrices
        # Q = q1*Ca^T*Ca + q2*Csd^T*Csd + q3*Ctd^T*Ctd
        self.Q_stage = (
            q1 * (self.C_a.T @ self.C_a)
            + q2 * (self.C_sd.T @ self.C_sd)
            + q3 * (self.C_td.T @ self.C_td)
        )
        self.N_stage = q1 * (self.C_a.T * self.D_a)  # (4, 1)
        self.R_stage = float(r0 + q1 * (self.D_a ** 2))
        self.S_w = q3 * self.C_td                     # (1, 4)

        # Terminal cost Pf from discrete Algebraic Riccati Equation (DARE)
        try:
            Pf = solve_discrete_are(self.A_d, self.B_d, self.Q_stage, np.array([[self.R_stage]]))
        except Exception:
            Pf = self.Q_stage.copy()
        self.Pf = Pf

        # Condensed prediction matrices Phi_x (4*Np, 4), Gamma_u (4*Np, Np), Gamma_w (4*Np, Np)
        self.Phi_x = np.zeros((4 * Np, 4), dtype=np.float64)
        self.Gamma_u = np.zeros((4 * Np, Np), dtype=np.float64)
        self.Gamma_w = np.zeros((4 * Np, Np), dtype=np.float64)

        A_pow = np.eye(4, dtype=np.float64)
        for i in range(Np):
            A_pow = A_pow @ self.A_d
            self.Phi_x[4 * i:4 * (i + 1), :] = A_pow

        for i in range(Np):
            for j in range(i + 1):
                A_step = np.linalg.matrix_power(self.A_d, i - j)
                self.Gamma_u[4 * i:4 * (i + 1), j] = (A_step @ self.B_d).flatten()
                self.Gamma_w[4 * i:4 * (i + 1), j] = (A_step @ self.E_d).flatten()

        # Block diagonal weight matrix Q_bar (4*Np, 4*Np)
        self.Q_bar = np.zeros((4 * Np, 4 * Np), dtype=np.float64)
        for i in range(Np - 1):
            self.Q_bar[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)] = self.Q_stage
        self.Q_bar[4 * (Np - 1):4 * Np, 4 * (Np - 1):4 * Np] = self.Pf

        # Cross-coupling block matrix N_bar (4*Np, Np)
        self.N_bar = np.zeros((4 * Np, Np), dtype=np.float64)
        for i in range(1, Np):
            self.N_bar[4 * (i - 1):4 * i, i] = self.N_stage.flatten()

        # Slew rate difference matrix D (Np, Np) and initial offset vector d_prev (Np,)
        self.D_rate = np.eye(Np, dtype=np.float64) - np.eye(Np, k=-1, dtype=np.float64)
        self.d_prev = np.zeros(Np, dtype=np.float64)
        self.d_prev[0] = 1.0

        # Hessian H = 2 * (Gamma_u^T Q_bar Gamma_u + Gamma_u^T N_bar + N_bar^T Gamma_u + R I + r_rate D^T D)
        H_raw = 2.0 * (
            self.Gamma_u.T @ self.Q_bar @ self.Gamma_u
            + self.Gamma_u.T @ self.N_bar
            + self.N_bar.T @ self.Gamma_u
            + self.R_stage * np.eye(Np, dtype=np.float64)
            + r_rate * (self.D_rate.T @ self.D_rate)
        )
        self.H = 0.5 * (H_raw + H_raw.T)  # Ensure exact symmetry

        # Precompute inverse of Hessian for fast unconstrained check
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = np.linalg.pinv(self.H)

        # Constraints
        self.bounds = [(self.config.min_force, self.config.max_force) for _ in range(Np)]
        self.du_max = self.config.slew_rate_limit * self.config.dt
        self.e0 = np.zeros(Np, dtype=np.float64)
        self.e0[0] = 1.0

    def reset(self) -> None:
        """Reset controller internal state and diagnostic metrics."""
        self.last_applied_force = 0.0
        self.solve_times.clear()
        self.unconstrained_hits = 0
        self.qp_solves = 0
        self.qp_failures = 0

    def compute_force(
        self,
        x: np.ndarray,
        road_preview: Optional[np.ndarray] = None,
        current_road_elev: float = 0.0
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute optimal control input u_0 for current state x using receding-horizon QP.

        Args:
            x: State vector [zs, zs_dot, zu, zu_dot]
            road_preview: Array of future road heights [zr_0, zr_1, ..., zr_{Np-1}].
                          If None, current_road_elev is held constant across the horizon.
            current_road_elev: Current road elevation [m]

        Returns:
            Tuple of (optimal_force_u0, diagnostics_dict)
        """
        t_start = time.perf_counter()
        Np = self.config.prediction_horizon
        x0 = np.asarray(x, dtype=np.float64).flatten()
        u_prev = float(self.last_applied_force)

        # Construct road disturbance preview vector W
        if road_preview is not None and self.config.use_preview:
            W = np.asarray(road_preview, dtype=np.float64).flatten()
            if len(W) < Np:
                W = np.pad(W, (0, Np - len(W)), mode="edge")
            else:
                W = W[:Np]
        else:
            # Constant or zero preview
            W = np.full(Np, current_road_elev, dtype=np.float64)

        # Construct road tracking vector Sw_bar
        Sw_bar = np.zeros(4 * Np, dtype=np.float64)
        for i in range(1, Np):
            Sw_bar[4 * (i - 1):4 * i] = (W[i] * self.S_w).flatten()

        # Compute linear gradient vector f
        x_dist = self.Phi_x @ x0 + self.Gamma_w @ W
        f = (
            2.0 * (self.Gamma_u.T @ self.Q_bar @ x_dist)
            + 2.0 * (self.N_bar.T @ x_dist)
            + 2.0 * self.e0 * float(self.N_stage.flatten() @ x0)
            - 2.0 * (self.Gamma_u.T @ Sw_bar)
            - 2.0 * self.config.r_rate * u_prev * (self.D_rate.T @ self.d_prev)
        )

        # 1. Fast unconstrained solution check
        u_unconstrained = -self.H_inv @ f

        # Slew rate limits for current u_prev
        # -du_max <= D * U - d_prev * u_prev <= du_max
        lb_rate = -self.du_max * np.ones(Np, dtype=np.float64) + self.d_prev * u_prev
        ub_rate = self.du_max * np.ones(Np, dtype=np.float64) + self.d_prev * u_prev

        in_bounds = np.all(u_unconstrained >= self.config.min_force) and np.all(u_unconstrained <= self.config.max_force)
        du_unc = self.D_rate @ u_unconstrained
        in_rate = np.all(du_unc >= lb_rate - 1e-6) and np.all(du_unc <= ub_rate + 1e-6)

        qp_solved = False
        qp_success = True

        if in_bounds and in_rate:
            # Unconstrained solution satisfies all inequality constraints
            u_opt = u_unconstrained
            self.unconstrained_hits += 1
        else:
            # 2. Solve constrained convex QP using SLSQP
            self.qp_solves += 1
            qp_solved = True
            lin_con = LinearConstraint(self.D_rate, lb_rate, ub_rate)

            # Warm start: clip unconstrained to force bounds
            u_init = np.clip(u_unconstrained, self.config.min_force, self.config.max_force)

            try:
                res = minimize(
                    fun=lambda u: 0.5 * u @ self.H @ u + f @ u,
                    x0=u_init,
                    jac=lambda u: self.H @ u + f,
                    bounds=self.bounds,
                    constraints=[lin_con],
                    method="SLSQP",
                    options={"maxiter": self.config.max_iter, "ftol": self.config.ftol}
                )
                if res.success:
                    u_opt = res.x
                else:
                    self.qp_failures += 1
                    qp_success = False
                    # Safe fallback: clip to rate limits from u_prev
                    u_opt = np.clip(u_init, u_prev - self.du_max, u_prev + self.du_max)
            except Exception:
                self.qp_failures += 1
                qp_success = False
                u_opt = np.clip(u_init, u_prev - self.du_max, u_prev + self.du_max)

        # Receding horizon principle: apply first control action u_0
        u0_cmd = float(u_opt[0])

        # Enforce rate and saturation clamp on applied action to guarantee physical feasibility
        u0_sat = float(np.clip(u0_cmd, self.config.min_force, self.config.max_force))
        u0_applied = float(np.clip(u0_sat, u_prev - self.du_max, u_prev + self.du_max))

        self.last_applied_force = u0_applied
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0
        self.solve_times.append(solve_time_ms)

        info = {
            "solve_time_ms": solve_time_ms,
            "qp_solved": qp_solved,
            "qp_success": qp_success,
            "is_saturated": abs(u0_applied - u0_cmd) > 1e-3,
            "active_controller": self.config.name,
            "u_cmd": u0_cmd,
            "u_applied": u0_applied
        }
        return u0_cmd, info

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic controller information."""
        mean_time = float(np.mean(self.solve_times)) if self.solve_times else 0.0
        max_time = float(np.max(self.solve_times)) if self.solve_times else 0.0
        total_evals = self.unconstrained_hits + self.qp_solves
        unconstrained_pct = (self.unconstrained_hits / total_evals * 100.0) if total_evals > 0 else 0.0

        return {
            "name": self.config.name,
            "prediction_horizon": self.config.prediction_horizon,
            "dt": self.config.dt,
            "weights": {
                "q1_accel": self.config.q1,
                "q2_travel": self.config.q2,
                "q3_tyre": self.config.q3,
                "r0_effort": self.config.r0,
                "r_rate": self.config.r_rate
            },
            "constraints": {
                "min_force_n": self.config.min_force,
                "max_force_n": self.config.max_force,
                "slew_rate_limit_n_per_s": self.config.slew_rate_limit
            },
            "solver_diagnostics": {
                "mean_solve_time_ms": mean_time,
                "max_solve_time_ms": max_time,
                "total_steps": total_evals,
                "unconstrained_pct": unconstrained_pct,
                "qp_solves": self.qp_solves,
                "qp_failures": self.qp_failures
            }
        }
