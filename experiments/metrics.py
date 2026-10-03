"""Performance metrics and evaluation index computation for suspension simulations."""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

from models.quarter_car import SimulationResult


def compute_simulation_metrics(result: SimulationResult, ground_truth_class: str = "") -> Dict[str, Any]:
    """Extract standard academic performance indices from a SimulationResult time series.

    Metrics computed:
        - rms_body_accel: RMS vertical chassis acceleration [m/s^2] (ISO 2631 ride comfort)
        - peak_body_accel: Peak absolute vertical chassis acceleration [m/s^2]
        - rms_susp_travel: RMS suspension rattle-space travel [m]
        - peak_susp_travel: Peak absolute suspension deflection [m]
        - rms_dynamic_tyre_load: RMS tyre normal load variation [N] (road holding)
        - peak_dynamic_tyre_load: Peak tyre normal load variation [N]
        - min_total_tyre_load: Minimum estimated net tyre normal contact force [N]
        - contact_loss_pct: Percentage of time tyre lost ground contact [%]
        - travel_violation_pct: Percentage of time exceeding allowable travel [%]
        - rms_actuator_force: RMS applied active actuator force [N]
        - peak_actuator_force: Maximum applied active actuator force [N]
        - saturation_pct: Percentage of time actuator was saturated at +/- Fmax [%]
        - mechanical_work_j: Cumulative absolute mechanical work executed by actuator [J]
        - electrical_energy_j: Cumulative electrical energy drawn from energy store [J]
        - controller_switches: Number of controller transitions executed during run
        - classification_accuracy: Classifier accuracy over live timeline [%] (if available)
    """
    dt = float(result.t[1] - result.t[0]) if len(result.t) > 1 else 0.002

    # Actuator saturation percentage
    u_app = result.u_applied
    u_req = result.u_requested
    is_saturated = np.abs(u_req - u_app) > 1e-3
    saturation_pct = float(np.mean(is_saturated) * 100.0) if len(u_app) > 0 else 0.0

    # Controller switching count
    ctrls = result.active_controllers
    switch_count = int(np.sum(ctrls[1:] != ctrls[:-1])) if (ctrls is not None and len(ctrls) > 1) else 0

    # Live classification accuracy against true scenario label
    live_acc = None
    if result.predicted_classes is not None and len(result.predicted_classes) > 0 and ground_truth_class:
        live_acc = float(np.mean(result.predicted_classes == ground_truth_class) * 100.0)

    return {
        "rms_body_accel_mps2": result.rms_body_accel,
        "peak_body_accel_mps2": result.peak_body_accel,
        "rms_susp_travel_m": result.rms_suspension_travel,
        "peak_susp_travel_m": result.peak_suspension_travel,
        "rms_dynamic_tyre_load_n": result.rms_dynamic_tyre_load,
        "peak_dynamic_tyre_load_n": result.peak_dynamic_tyre_load,
        "min_total_tyre_load_n": result.min_tyre_load,
        "contact_loss_pct": result.contact_loss_percentage,
        "travel_violation_pct": result.travel_violation_percentage,
        "rms_actuator_force_n": float(np.sqrt(np.mean(u_app**2))),
        "peak_actuator_force_n": float(np.max(np.abs(u_app))),
        "saturation_pct": saturation_pct,
        "mechanical_work_j": result.total_mechanical_work_j,
        "electrical_energy_j": result.total_electrical_energy_j,
        "controller_switches": switch_count,
        "classification_accuracy_pct": live_acc
    }
