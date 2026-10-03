"""Parameter sensitivity and robustness analysis under vehicle parameter mismatch.

Evaluates performance stability and vibration isolation when physical EV parameters
(battery payload mass, spring aging, damper wear, tyre pressure) deviate from design nominals.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel
from models.actuator import ActuatorModel
from roads.bump import BumpRoadGenerator
from controllers.controller_registry import build_nominal_controllers


def run_parameter_sensitivity_study(
    output_csv_path: str | Path = "results/tables/sensitivity_analysis.csv",
    output_fig_path: str | Path = "results/figures/parameter_sensitivity.png"
) -> pd.DataFrame:
    """Evaluate Passive vs Active LQR under structured parameter perturbations.

    Parameter perturbations:
    1. Sprung mass ms (battery payload): +/- 20% (360 kg to 540 kg)
    2. Damper wear cs (degraded damping): -30% (1050 Ns/m)
    3. Spring settling ks: -10% (25,200 N/m)
    4. Tyre stiffness kt: +/- 15% (161,500 N/m to 218,500 N/m)

    Returns:
        DataFrame of sensitivity results across variations.
    """
    nominal = VehicleParameters()
    speed_mps = 50.0 / 3.6  # 50 km/h
    bump = BumpRoadGenerator(height=0.05, width=0.8, start_distance=10.0, add_noise=False)
    road_fn = bump.get_time_callable(speed_mps)

    # Actuator with realistic force constraints
    actuator = ActuatorModel(max_force=1800.0, slew_rate_limit=18000.0)

    # Controller synthesized with NOMINAL parameters
    controllers = build_nominal_controllers(nominal)
    ctrl = controllers["balanced"]

    variations = [
        {"case": "Nominal Baseline", "factor": 1.0, "params": nominal},
        {"case": "Chassis Payload -20%", "factor": 0.8, "params": nominal.with_variations(ms_factor=0.8)},
        {"case": "Chassis Payload +20%", "factor": 1.2, "params": nominal.with_variations(ms_factor=1.2)},
        {"case": "Damper Degradation -30%", "factor": 0.7, "params": nominal.with_variations(cs_factor=0.7)},
        {"case": "Spring Softening -10%", "factor": 0.9, "params": nominal.with_variations(ks_factor=0.9)},
        {"case": "Tyre Pressure -15%", "factor": 0.85, "params": nominal.with_variations(kt_factor=0.85)},
        {"case": "Tyre Pressure +15%", "factor": 1.15, "params": nominal.with_variations(kt_factor=1.15)},
    ]

    records = []

    for v in variations:
        case_name = v["case"]
        pert_params = v["params"]
        model_pert = QuarterCarModel(pert_params)
        actuator.reset()

        # Check unsaturated linear stability of nominal gain K on perturbed plant
        A_cl = model_pert.A - model_pert.B @ ctrl.K
        max_real_pole = float(np.max(np.real(np.linalg.eigvals(A_cl))))
        is_linear_stable = bool(max_real_pole < 0.0)

        # 1. Passive simulation
        res_pass = model_pert.simulate(road_func=road_fn, controller_func=None, duration=4.0, dt=0.002)

        # 2. Active LQR simulation with real actuator constraints
        def ctrl_fn(t, x, zr):
            u_req = ctrl.compute_force(x)
            u_r, u_a, is_sat = actuator.apply_limits(u_req, 0.002)
            return u_r, u_a, {"is_saturated": is_sat}

        res_act = model_pert.simulate(
            road_func=road_fn,
            controller_func=ctrl_fn,
            duration=4.0,
            dt=0.002,
            actuator=actuator
        )

        sat_pct = float(np.mean(np.abs(res_act.u_requested - res_act.u_applied) > 1e-3) * 100.0)

        records.append({
            "variation_case": case_name,
            "ms_kg": pert_params.ms,
            "cs_ns_per_m": pert_params.cs,
            "ks_n_per_m": pert_params.ks,
            "kt_n_per_m": pert_params.kt,
            "unsaturated_linear_stable": is_linear_stable,
            "max_real_pole": max_real_pole,
            "actuator_saturation_pct": sat_pct,
            "passive_rms_accel_mps2": res_pass.rms_body_accel,
            "active_rms_accel_mps2": res_act.rms_body_accel,
            "accel_reduction_pct": ((res_pass.rms_body_accel - res_act.rms_body_accel) / res_pass.rms_body_accel) * 100.0,
            "passive_peak_travel_mm": res_pass.peak_suspension_travel * 1000.0,
            "active_peak_travel_mm": res_act.peak_suspension_travel * 1000.0,
            "active_rms_force_n": float(np.sqrt(np.mean(res_act.u_applied**2)))
        })

    df = pd.DataFrame(records)

    # Export CSV
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # Plot figure
    _plot_sensitivity_figure(df, output_fig_path)

    return df


def _plot_sensitivity_figure(df: pd.DataFrame, output_path: str | Path) -> None:
    """Generate sensitivity comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["passive_rms_accel_mps2"], width, label="Passive Suspension", color="#7f7f7f")
    ax.bar(x + width/2, df["active_rms_accel_mps2"], width, label="Active LQR (Nominal Gain)", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(df["variation_case"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMS Body Acceleration [m/s²]", fontsize=11, fontweight="bold")
    ax.set_title("Suspension Sensitivity & Robustness Under Physical Parameter Variations", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
