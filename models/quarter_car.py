"""Two-degree-of-freedom (2-DOF) quarter-car physical dynamics model.

Implements the continuous-time state-space representation, analytical derivation,
numerical simulation using scipy.integrate.solve_ivp, and direct physical output calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp

from models.vehicle_parameters import VehicleParameters


@dataclass
class SimulationResult:
    """Complete time-history results from a quarter-car simulation run."""
    t: np.ndarray                  # Time vector [s]
    zs: np.ndarray                 # Sprung mass vertical displacement [m]
    zs_dot: np.ndarray             # Sprung mass vertical velocity [m/s]
    zs_ddot: np.ndarray            # Sprung mass vertical acceleration [m/s^2]
    zu: np.ndarray                 # Unsprung mass vertical displacement [m]
    zu_dot: np.ndarray             # Unsprung mass vertical velocity [m/s]
    zu_ddot: np.ndarray            # Unsprung mass vertical acceleration [m/s^2]
    suspension_travel: np.ndarray  # Suspension deflection (zs - zu) [m]
    tyre_deflection: np.ndarray    # Tyre radial deflection (zu - zr) [m]
    zr: np.ndarray                 # Road profile displacement [m]
    u_requested: np.ndarray        # Actuator force requested by controller [N]
    u_applied: np.ndarray          # Actuator force physically applied by actuator [N]
    dynamic_tyre_load: np.ndarray  # Dynamic tyre-load variation kt * (zu - zr) [N]
    total_tyre_load: np.ndarray    # Total estimated normal tyre force [N]
    loss_of_contact: np.ndarray    # Boolean indicator for tyre lift-off (total load <= 0)
    travel_exceeded: np.ndarray    # Boolean indicator for rattle-space travel violation
    power_mechanical: np.ndarray   # Mechanical power u_applied * (zs_dot - zu_dot) [W]
    power_electrical: np.ndarray   # Electrical power drawn from battery [W]
    predicted_classes: Optional[np.ndarray] = None  # AI predicted road class over time
    active_controllers: Optional[np.ndarray] = None # Active controller tag over time

    @property
    def duration(self) -> float:
        """Total duration of simulation in seconds."""
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def rms_body_accel(self) -> float:
        """Root-mean-square sprung mass vertical acceleration [m/s^2]."""
        return float(np.sqrt(np.mean(self.zs_ddot**2)))

    @property
    def peak_body_accel(self) -> float:
        """Peak absolute sprung mass vertical acceleration [m/s^2]."""
        return float(np.max(np.abs(self.zs_ddot)))

    @property
    def rms_suspension_travel(self) -> float:
        """Root-mean-square suspension deflection [m]."""
        return float(np.sqrt(np.mean(self.suspension_travel**2)))

    @property
    def peak_suspension_travel(self) -> float:
        """Peak absolute suspension deflection [m]."""
        return float(np.max(np.abs(self.suspension_travel)))

    @property
    def rms_dynamic_tyre_load(self) -> float:
        """Root-mean-square dynamic tyre-load variation [N]."""
        return float(np.sqrt(np.mean(self.dynamic_tyre_load**2)))

    @property
    def peak_dynamic_tyre_load(self) -> float:
        """Peak dynamic tyre-load variation [N]."""
        return float(np.max(np.abs(self.dynamic_tyre_load)))

    @property
    def min_tyre_load(self) -> float:
        """Minimum total tyre contact force [N]."""
        return float(np.min(self.total_tyre_load))

    @property
    def contact_loss_duration(self) -> float:
        """Total duration in seconds during which tyre contact was lost."""
        if len(self.t) < 2:
            return 0.0
        dt = self.t[1] - self.t[0]
        return float(np.sum(self.loss_of_contact) * dt)

    @property
    def contact_loss_percentage(self) -> float:
        """Percentage of simulation time with tyre lift-off [%]."""
        return float(np.mean(self.loss_of_contact) * 100.0)

    @property
    def travel_violation_percentage(self) -> float:
        """Percentage of simulation time exceeding allowable travel [%]."""
        return float(np.mean(self.travel_exceeded) * 100.0)

    @property
    def total_electrical_energy_j(self) -> float:
        """Total electrical energy drawn from energy store [Joules]."""
        if len(self.t) < 2:
            return 0.0
        return float(np.trapezoid(self.power_electrical, self.t))

    @property
    def total_mechanical_work_j(self) -> float:
        """Total absolute mechanical work executed by actuator [Joules]."""
        if len(self.t) < 2:
            return 0.0
        return float(np.trapezoid(np.abs(self.power_mechanical), self.t))


class QuarterCarModel:
    """Two-Degree-of-Freedom (2-DOF) Quarter-Car Vehicle Dynamics Model.

    State Vector:
        x = [zs, zs_dot, zu, zu_dot]^T
        where:
            zs: Sprung mass vertical displacement [m]
            zs_dot: Sprung mass vertical velocity [m/s]
            zu: Unsprung mass vertical displacement [m]
            zu_dot: Unsprung mass vertical velocity [m/s]

    Control and Disturbance:
        u: Active actuator force [N] (+u pushes sprung mass UP, unsprung mass DOWN)
        zr: Road vertical displacement [m]

    Governing Equations of Motion:
        ms * zs_ddot = -ks*(zs - zu) - cs*(zs_dot - zu_dot) + u
        mu * zu_ddot =  ks*(zs - zu) + cs*(zs_dot - zu_dot) - kt*(zu - zr) - u
    """

    def __init__(self, params: Optional[VehicleParameters] = None, unilateral_tyre: bool = False) -> None:
        """Initialize Quarter-Car model with vehicle parameters.

        Args:
            params: VehicleParameters instance. If None, default EV parameters are used.
            unilateral_tyre: If True, tyre cannot pull in tension (separates on lift-off).
        """
        self.params = params if params is not None else VehicleParameters()
        self.unilateral_tyre = unilateral_tyre
        self._build_state_space()

    def _build_state_space(self) -> None:
        """Derive and store continuous state-space matrices A, B, and E.

        Continuous State Equation:
            x_dot = A * x + B * u + E * zr
        """
        ms = self.params.ms
        mu = self.params.mu
        ks = self.params.ks
        cs = self.params.cs
        kt = self.params.kt

        # State matrix A (4x4)
        self.A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-ks / ms, -cs / ms, ks / ms, cs / ms],
            [0.0, 0.0, 0.0, 1.0],
            [ks / mu, cs / mu, -(ks + kt) / mu, -cs / mu]
        ], dtype=np.float64)

        # Actuator input matrix B (4x1)
        self.B = np.array([
            [0.0],
            [1.0 / ms],
            [0.0],
            [-1.0 / mu]
        ], dtype=np.float64)

        # Road disturbance input matrix E (4x1)
        self.E = np.array([
            [0.0],
            [0.0],
            [0.0],
            [kt / mu]
        ], dtype=np.float64)

    def verify_state_space(self, x: np.ndarray, u: float, zr: float) -> Tuple[bool, float]:
        """Verify that the linear matrix implementation exactly matches the direct scalar ODEs.

        Args:
            x: State vector [zs, zs_dot, zu, zu_dot]
            u: Actuator force [N]
            zr: Road displacement [m]

        Returns:
            Tuple of (is_consistent, max_absolute_difference)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        zs, zs_dot, zu, zu_dot = x[0], x[1], x[2], x[3]

        # 1. Compute via matrix state-space
        x_dot_matrix = (self.A @ x.reshape(4, 1) + self.B * u + self.E * zr).flatten()

        # 2. Compute via direct scalar equations of motion
        ms, mu = self.params.ms, self.params.mu
        ks, cs, kt = self.params.ks, self.params.cs, self.params.kt

        f_susp = ks * (zs - zu) + cs * (zs_dot - zu_dot)
        zs_ddot_direct = (-f_susp + u) / ms
        zu_ddot_direct = (f_susp - kt * (zu - zr) - u) / mu

        x_dot_direct = np.array([zs_dot, zs_ddot_direct, zu_dot, zu_ddot_direct], dtype=np.float64)

        max_err = float(np.max(np.abs(x_dot_matrix - x_dot_direct)))
        is_consistent = max_err < 1e-10
        return is_consistent, max_err

    def state_derivatives(self, t: float, x: np.ndarray, u: float, zr: float) -> np.ndarray:
        """Compute state derivatives dx/dt.

        Args:
            t: Current time [s]
            x: State vector [zs, zs_dot, zu, zu_dot]
            u: Actuator force applied [N]
            zr: Road elevation [m]

        Returns:
            Array of derivatives [zs_dot, zs_ddot, zu_dot, zu_ddot]
        """
        if not self.unilateral_tyre:
            # Linear state-space equation: x_dot = A*x + B*u + E*zr
            return (self.A @ x.reshape(4, 1) + self.B * u + self.E * zr).flatten()
        else:
            # Unilateral tyre contact: tyre only pushes in compression (zr >= zu)
            ms, mu = self.params.ms, self.params.mu
            ks, cs, kt = self.params.ks, self.params.cs, self.params.kt
            zs, zs_dot, zu, zu_dot = x[0], x[1], x[2], x[3]

            f_susp = ks * (zs - zu) + cs * (zs_dot - zu_dot)
            # Compression is positive when road pushes wheel up: zr - zu > 0
            f_tyre = max(0.0, kt * (zr - zu))

            zs_ddot = (-f_susp + u) / ms
            zu_ddot = (f_susp + f_tyre - u) / mu
            return np.array([zs_dot, zs_ddot, zu_dot, zu_ddot], dtype=np.float64)

    def calculate_outputs(
        self,
        x: np.ndarray,
        u_applied: float,
        zr: float
    ) -> Dict[str, float]:
        """Compute instantaneous physical performance outputs.

        Args:
            x: State vector [zs, zs_dot, zu, zu_dot]
            u_applied: Actuator force applied [N]
            zr: Road profile elevation [m]

        Returns:
            Dictionary containing accelerations, deflections, tyre loads, and constraint checks.
        """
        zs, zs_dot, zu, zu_dot = x[0], x[1], x[2], x[3]
        ms = self.params.ms
        mu = self.params.mu
        ks = self.params.ks
        cs = self.params.cs
        kt = self.params.kt

        susp_travel = zs - zu
        tyre_deflection = zu - zr
        f_susp = ks * susp_travel + cs * (zs_dot - zu_dot)

        zs_ddot = (-f_susp + u_applied) / ms
        zu_ddot = (f_susp - kt * tyre_deflection - u_applied) / mu

        dynamic_tyre_load = kt * tyre_deflection
        # Static normal load is (ms + mu)*g. When tyre compresses (tyre_deflection < 0),
        # contact force increases. When tyre extends (tyre_deflection > 0), contact force drops.
        total_tyre_load = self.params.static_tyre_load - dynamic_tyre_load
        loss_of_contact = bool(total_tyre_load <= 0.0)
        travel_exceeded = bool(abs(susp_travel) > self.params.max_travel)

        # Mechanical power: force * relative suspension velocity
        power_mech = u_applied * (zs_dot - zu_dot)

        return {
            "zs_ddot": float(zs_ddot),
            "zu_ddot": float(zu_ddot),
            "suspension_travel": float(susp_travel),
            "tyre_deflection": float(tyre_deflection),
            "dynamic_tyre_load": float(dynamic_tyre_load),
            "total_tyre_load": float(total_tyre_load),
            "loss_of_contact": loss_of_contact,
            "travel_exceeded": travel_exceeded,
            "power_mech": float(power_mech),
        }

    def simulate(
        self,
        road_func: Callable[[float], float],
        controller_func: Optional[Callable[[float, np.ndarray, float], Tuple[float, float, Dict[str, Any]]]] = None,
        duration: float = 6.0,
        dt: float = 0.002,
        x0: Optional[np.ndarray] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        actuator=None
    ) -> SimulationResult:
        """Run numerical simulation of the quarter-car model over a time trajectory.

        Args:
            road_func: Function returning road displacement zr(t) at time t.
            controller_func: Function returning (u_requested, u_applied, info_dict) given (t, x, zr).
                             If None, passive suspension (u = 0) is simulated.
            duration: Simulation total time [s].
            dt: Fixed sampling time interval [s] (e.g. 0.002 for 500 Hz).
            x0: Initial state vector [zs0, zs_dot0, zu0, zu_dot0]. Defaults to zeros.
            method: Integration algorithm for solve_ivp ('RK45', 'Radau', etc.).
            rtol: Relative integration tolerance.
            atol: Absolute integration tolerance.
            actuator: Optional ActuatorModel instance.

        Returns:
            SimulationResult containing all full-length time series and computed metrics.
        """
        t_eval = np.arange(0.0, duration + dt / 2.0, dt)
        if x0 is None:
            x0 = np.zeros(4, dtype=np.float64)
        else:
            x0 = np.asarray(x0, dtype=np.float64)

        n_steps = len(t_eval)

        # Pre-allocate arrays for efficiency and precision
        zs = np.zeros(n_steps, dtype=np.float64)
        zs_dot = np.zeros(n_steps, dtype=np.float64)
        zs_ddot = np.zeros(n_steps, dtype=np.float64)
        zu = np.zeros(n_steps, dtype=np.float64)
        zu_dot = np.zeros(n_steps, dtype=np.float64)
        zu_ddot = np.zeros(n_steps, dtype=np.float64)
        suspension_travel = np.zeros(n_steps, dtype=np.float64)
        tyre_deflection = np.zeros(n_steps, dtype=np.float64)
        zr = np.zeros(n_steps, dtype=np.float64)
        u_req = np.zeros(n_steps, dtype=np.float64)
        u_app = np.zeros(n_steps, dtype=np.float64)
        dyn_tyre_load = np.zeros(n_steps, dtype=np.float64)
        tot_tyre_load = np.zeros(n_steps, dtype=np.float64)
        loss_contact = np.zeros(n_steps, dtype=bool)
        travel_viol = np.zeros(n_steps, dtype=bool)
        power_mech = np.zeros(n_steps, dtype=np.float64)
        power_elec = np.zeros(n_steps, dtype=np.float64)

        predicted_classes = []
        active_controllers = []

        # We step through the simulation sequentially to allow realistic real-time control,
        # sensor feature extraction, actuator saturation, and anti-chattering logic without future leakage.
        # At each time step [t_k, t_{k+1}], we evaluate controller and integrate one step.
        curr_x = x0.copy()

        # Reset actuator internal state if present
        if actuator is not None and hasattr(actuator, "reset"):
            actuator.reset()

        for k in range(n_steps):
            t_curr = t_eval[k]
            zr_curr = float(road_func(t_curr))
            zr[k] = zr_curr

            # 1. State measurement
            zs[k] = curr_x[0]
            zs_dot[k] = curr_x[1]
            zu[k] = curr_x[2]
            zu_dot[k] = curr_x[3]

            # 2. Controller & Actuator evaluation
            if controller_func is not None:
                u_r, u_a, info = controller_func(t_curr, curr_x, zr_curr)
                if "predicted_class" in info:
                    predicted_classes.append(info["predicted_class"])
                if "active_controller" in info:
                    active_controllers.append(info["active_controller"])
            else:
                u_r, u_a = 0.0, 0.0
                info = {}

            u_req[k] = u_r
            u_app[k] = u_a

            # 3. Actuator power/energy calculation
            v_rel = curr_x[1] - curr_x[3]
            if actuator is not None and hasattr(actuator, "calculate_power"):
                p_m, p_e = actuator.calculate_power(u_a, v_rel)
            else:
                p_m = u_a * v_rel
                p_e = max(0.0, p_m / 0.85) if p_m >= 0 else p_m * 0.50

            power_mech[k] = p_m
            power_elec[k] = p_e

            # 4. Instantaneous outputs
            outputs = self.calculate_outputs(curr_x, u_a, zr_curr)
            zs_ddot[k] = outputs["zs_ddot"]
            zu_ddot[k] = outputs["zu_ddot"]
            suspension_travel[k] = outputs["suspension_travel"]
            tyre_deflection[k] = outputs["tyre_deflection"]
            dyn_tyre_load[k] = outputs["dynamic_tyre_load"]
            tot_tyre_load[k] = outputs["total_tyre_load"]
            loss_contact[k] = outputs["loss_of_contact"]
            travel_viol[k] = outputs["travel_exceeded"]

            # 5. Integrate forward to next step if not last step
            if k < n_steps - 1:
                t_next = t_eval[k + 1]

                # Use continuous integration across interval [t_curr, t_next]
                def ode_rhs(t_val, state):
                    # Linear interpolation of road and force across small interval dt
                    alpha = (t_val - t_curr) / (t_next - t_curr)
                    zr_val = zr_curr + alpha * (float(road_func(t_next)) - zr_curr)
                    return self.state_derivatives(t_val, state, u_a, zr_val)

                sol = solve_ivp(
                    ode_rhs,
                    [t_curr, t_next],
                    curr_x,
                    method=method,
                    rtol=rtol,
                    atol=atol
                )
                if not sol.success:
                    # Fallback to single RK4 step if solve_ivp encounters issue
                    dt_step = t_next - t_curr
                    k1 = self.state_derivatives(t_curr, curr_x, u_a, zr_curr)
                    k2 = self.state_derivatives(t_curr + 0.5 * dt_step, curr_x + 0.5 * dt_step * k1, u_a, 0.5 * (zr_curr + float(road_func(t_next))))
                    k3 = self.state_derivatives(t_curr + 0.5 * dt_step, curr_x + 0.5 * dt_step * k2, u_a, 0.5 * (zr_curr + float(road_func(t_next))))
                    k4 = self.state_derivatives(t_next, curr_x + dt_step * k3, u_a, float(road_func(t_next)))
                    curr_x = curr_x + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                else:
                    curr_x = sol.y[:, -1]

        pred_arr = np.array(predicted_classes) if predicted_classes else None
        ctrl_arr = np.array(active_controllers) if active_controllers else None

        return SimulationResult(
            t=t_eval,
            zs=zs,
            zs_dot=zs_dot,
            zs_ddot=zs_ddot,
            zu=zu,
            zu_dot=zu_dot,
            zu_ddot=zu_ddot,
            suspension_travel=suspension_travel,
            tyre_deflection=tyre_deflection,
            zr=zr,
            u_requested=u_req,
            u_applied=u_app,
            dynamic_tyre_load=dyn_tyre_load,
            total_tyre_load=tot_tyre_load,
            loss_of_contact=loss_contact,
            travel_exceeded=travel_viol,
            power_mechanical=power_mech,
            power_electrical=power_elec,
            predicted_classes=pred_arr,
            active_controllers=ctrl_arr
        )
