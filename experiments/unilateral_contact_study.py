"""Study and comparative analysis of Linear Tyre Model vs Unilateral Tyre Contact Model.

The unilateral contact model explicitly captures tyre lift-off (loss of road contact),
where the tyre cannot exert tensile pulling forces on the ground (F_tyre >= 0).
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from roads.bump import BumpRoadGenerator
from controllers.controller_registry import build_nominal_controllers


def run_unilateral_contact_comparison(
    bump_height: float = 0.07,  # Severe 70 mm speed bump
    bump_width: float = 0.7,
    vehicle_speed_kmh: float = 55.0,
    duration_s: float = 3.0,
    dt: float = 0.001,
    output_fig_path: str | Path = "results/figures/unilateral_contact_comparison.png",
    output_json_path: str | Path = "results/tables/unilateral_contact_metrics.json"
) -> Dict[str, Any]:
    """Compare linear tyre model against unilateral contact model under severe road disturbance.

    Args:
        bump_height: Height of speed bump [m].
        bump_width: Longitudinal width of speed bump [m].
        vehicle_speed_kmh: Vehicle forward speed [km/h].
        duration_s: Duration of simulation [s].
        dt: Integration time step [s].
        output_fig_path: Output figure file path.
        output_json_path: Output metrics JSON path.

    Returns:
        Dictionary containing comparative metrics between linear and unilateral tyre models.
    """
    params = VehicleParameters()
    speed_mps = vehicle_speed_kmh / 3.6
    road = BumpRoadGenerator(height=bump_height, width=bump_width, start_distance=10.0, add_noise=False)
    road_fn = road.get_time_callable(speed_mps)

    controllers = build_nominal_controllers(params)
    ctrl = controllers["balanced"]
    ctrl_fn = lambda t, x, zr: (ctrl.compute_force(x), ctrl.compute_force(x), {})

    # 1. Model with linear bilateral tyre
    model_linear = QuarterCarModel(params=params, unilateral_tyre=False)
    res_linear = model_linear.simulate(
        road_func=road_fn,
        controller_func=ctrl_fn,
        duration=duration_s,
        dt=dt
    )

    # 2. Model with realistic unilateral contact (tyre separates on lift-off)
    model_unilateral = QuarterCarModel(params=params, unilateral_tyre=True)
    res_unilateral = model_unilateral.simulate(
        road_func=road_fn,
        controller_func=ctrl_fn,
        duration=duration_s,
        dt=dt
    )

    metrics = {
        "scenario": {
            "bump_height_mm": bump_height * 1000.0,
            "bump_width_m": bump_width,
            "speed_kmh": vehicle_speed_kmh
        },
        "linear_model": {
            "rms_body_accel_mps2": res_linear.rms_body_accel,
            "peak_body_accel_mps2": res_linear.peak_body_accel,
            "peak_susp_travel_mm": res_linear.peak_suspension_travel * 1000.0,
            "min_total_tyre_load_n": res_linear.min_tyre_load,
            "contact_loss_duration_s": res_linear.contact_loss_duration,
            "contact_loss_pct": res_linear.contact_loss_percentage
        },
        "unilateral_model": {
            "rms_body_accel_mps2": res_unilateral.rms_body_accel,
            "peak_body_accel_mps2": res_unilateral.peak_body_accel,
            "peak_susp_travel_mm": res_unilateral.peak_suspension_travel * 1000.0,
            "min_total_tyre_load_n": max(0.0, res_unilateral.min_tyre_load),
            "contact_loss_duration_s": res_unilateral.contact_loss_duration,
            "contact_loss_pct": res_unilateral.contact_loss_percentage
        }
    }

    # Save metrics JSON
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot comparison
    _plot_unilateral_comparison(res_linear, res_unilateral, output_fig_path)

    return metrics


def _plot_unilateral_comparison(
    res_lin: SimulationResult,
    res_uni: SimulationResult,
    output_path: str | Path
) -> None:
    """Generate publication plot comparing linear vs unilateral tyre response."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, dpi=300)
    t = res_lin.t

    # 1. Unsprung wheel displacement vs Road
    axes[0].plot(t, res_lin.zr * 1000.0, "k--", label=r"Road Elevation $z_r$", alpha=0.6)
    axes[0].plot(t, res_lin.zu * 1000.0, "b-", label=r"Linear Tyre Wheel $z_u$")
    axes[0].plot(t, res_uni.zu * 1000.0, "r-", label=r"Unilateral Tyre Wheel $z_u$")
    axes[0].set_ylabel("Wheel Disp [mm]", fontsize=10)
    axes[0].set_title("Linear vs Unilateral Tyre Contact Dynamics (Severe Bump)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Total estimated normal tyre force
    axes[1].plot(t, res_lin.total_tyre_load, "b-", label="Linear Tyre Load (can go < 0)")
    axes[1].plot(t, np.maximum(0.0, res_uni.total_tyre_load), "r-", label="Unilateral Tyre Load (clamps at 0)")
    axes[1].axhline(0, color="gray", linestyle=":", label="Ground Contact Threshold")
    axes[1].set_ylabel("Normal Force [N]", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Body vertical acceleration
    axes[2].plot(t, res_lin.zs_ddot, "b-", label=r"Linear Tyre $\ddot{z}_s$")
    axes[2].plot(t, res_uni.zs_ddot, "r-", label=r"Unilateral Tyre $\ddot{z}_s$ (Impact Shock)")
    axes[2].set_ylabel("Accel [m/s²]", fontsize=10)
    axes[2].set_xlabel("Time [s]", fontsize=11)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
