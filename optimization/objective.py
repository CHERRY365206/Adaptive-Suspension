"""Normalized multi-objective cost function for LQR controller tuning via PSO.

Combines RMS passenger acceleration, suspension deflection, dynamic tyre-load variation,
and control effort, with severe non-linear penalties for physical constraint violations.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel
from models.actuator import ActuatorModel
from controllers.lqr import LQRController
from roads.road_dataset import RoadScenario
from optimization.parameterization import decode_parameters


# Reference normalization scales (established engineering baselines)
REF_BODY_ACCEL = 1.5          # [m/s^2] (ISO 2631 threshold of comfort discomfort)
REF_SUSP_TRAVEL = 0.03        # [m] (30 mm nominal travel scale)
REF_TYRE_LOAD = 1500.0        # [N] (~30% of static tyre normal force)
REF_ACTUATOR_FORCE = 600.0    # [N] (nominal active force scale)

# Target objective weighting profiles
OBJECTIVE_WEIGHTS = {
    "comfort": {
        "w_a": 0.65,    # Heavily penalizes vertical chassis acceleration
        "w_d": 0.15,    # Moderate suspension deflection penalty
        "w_t": 0.10,    # Lower tyre load emphasis
        "w_u": 0.10,    # Control effort penalty
    },
    "balanced": {
        "w_a": 0.35,    # Balanced passenger comfort
        "w_d": 0.25,    # Balanced suspension travel
        "w_t": 0.25,    # Balanced tyre road holding
        "w_u": 0.15,    # Actuator control effort
    },
    "road_holding": {
        "w_a": 0.15,    # Lower chassis comfort emphasis
        "w_d": 0.20,    # Suspension travel penalty
        "w_t": 0.55,    # Heavily penalizes tyre normal load variation (grip)
        "w_u": 0.10,    # Actuator control effort
    }
}


def evaluate_lqr_candidate(
    particle_p: np.ndarray,
    target_mode: str,
    training_scenarios: List[RoadScenario],
    vehicle_params: Optional[VehicleParameters] = None,
    actuator: Optional[ActuatorModel] = None,
    dt: float = 0.002
) -> Dict[str, Any]:
    """Evaluate a candidate LQR parameter vector across training scenarios.

    Args:
        particle_p: 4D particle position [log10_q1, log10_q2, log10_q3, log10_r0]
        target_mode: 'comfort', 'balanced', or 'road_holding'
        training_scenarios: Dedicated training road scenarios
        vehicle_params: Optional vehicle parameters
        actuator: Optional actuator model
        dt: Simulation time step [s]

    Returns:
        Dictionary containing total cost, individual normalized metrics, and feasibility.
    """
    params = vehicle_params or VehicleParameters()
    act = actuator or ActuatorModel()
    model = QuarterCarModel(params)

    # 1. Decode parameters
    try:
        cfg = decode_parameters(particle_p, name=f"candidate_{target_mode}")
        ctrl = LQRController(cfg, params)
    except Exception as e:
        # Invalid Riccati solution or unstable candidate
        return {
            "cost": 1.0e6,
            "is_feasible": False,
            "failure_reason": f"CARE solver failed: {e}",
            "metrics": {}
        }

    # Stability check on unsaturated model
    if not ctrl.is_stable:
        return {
            "cost": 1.0e6,
            "is_feasible": False,
            "failure_reason": "Unstable closed-loop eigenvalues",
            "metrics": {}
        }

    weights = OBJECTIVE_WEIGHTS.get(target_mode, OBJECTIVE_WEIGHTS["balanced"])
    w_a = weights["w_a"]
    w_d = weights["w_d"]
    w_t = weights["w_t"]
    w_u = weights["w_u"]

    scenario_costs = []
    accel_rms_list = []
    travel_rms_list = []
    tyre_rms_list = []
    force_rms_list = []
    travel_violation_count = 0
    contact_loss_count = 0

    ctrl_fn = lambda t, x, zr: (ctrl.compute_force(x), ctrl.compute_force(x), {})

    # Evaluate across each training scenario
    for sc in training_scenarios:
        road_fn = sc.road_profile.get_time_callable(sc.speed_mps)

        # Run simulation with actuator constraints
        sim_res = model.simulate(
            road_func=road_fn,
            controller_func=ctrl_fn,
            duration=sc.duration_s,
            dt=dt,
            actuator=act
        )

        rms_a = sim_res.rms_body_accel
        rms_d = sim_res.rms_suspension_travel
        rms_t = sim_res.rms_dynamic_tyre_load
        rms_u = float(np.sqrt(np.mean(sim_res.u_applied**2)))

        accel_rms_list.append(rms_a)
        travel_rms_list.append(rms_d)
        tyre_rms_list.append(rms_t)
        force_rms_list.append(rms_u)

        # Normalized objective terms
        norm_a = rms_a / REF_BODY_ACCEL
        norm_d = rms_d / REF_SUSP_TRAVEL
        norm_t = rms_t / REF_TYRE_LOAD
        norm_u = rms_u / REF_ACTUATOR_FORCE

        base_cost = (w_a * norm_a) + (w_d * norm_d) + (w_t * norm_t) + (w_u * norm_u)

        # Constraint Penalties
        penalty = 0.0
        # 1. Rattle space travel limit: 0.08 m
        peak_travel = sim_res.peak_suspension_travel
        if peak_travel > params.max_travel:
            travel_violation_count += 1
            excess = (peak_travel - params.max_travel) / params.max_travel
            penalty += 100.0 * (excess ** 2)

        # 2. Tyre lift-off: total tyre load <= 0
        if sim_res.min_tyre_load <= 0.0:
            contact_loss_count += 1
            penalty += 50.0 * (sim_res.contact_loss_percentage / 10.0)

        total_sc_cost = base_cost + penalty
        scenario_costs.append(total_sc_cost)

    mean_cost = float(np.mean(scenario_costs)) if scenario_costs else 1.0e6
    is_feasible = (travel_violation_count == 0) and (contact_loss_count == 0)

    return {
        "cost": mean_cost,
        "is_feasible": is_feasible,
        "metrics": {
            "mean_rms_accel": float(np.mean(accel_rms_list)),
            "mean_rms_travel": float(np.mean(travel_rms_list)),
            "mean_rms_tyre_load": float(np.mean(tyre_rms_list)),
            "mean_rms_force": float(np.mean(force_rms_list)),
            "travel_violations": travel_violation_count,
            "contact_loss_events": contact_loss_count,
        }
    }
