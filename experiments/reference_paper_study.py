"""Comprehensive comparative evaluation reproducing Deshpande et al. (2014) benchmarks.

Compares:
1. Passive Suspension (Baseline)
2. Fixed Nominal LQR (Continuous CARE with cross-coupling)
3. Constrained Discrete-Time MPC (Convex QP with saturation and slew limits)
4. DOSMC without u_s (Disturbance Observer Sliding Mode Control, pure DO cancellation)
5. DOSMC with u_s (Disturbance Observer Sliding Mode Control with boundary layer robust term)

Across both:
- Academic Paper Benchmarks: Profile 1 (Chen & Huang 2005 bump/trough), Profile 2 (Harmonic), Profile 3 (Multi-harmonic)
- Real Physical Datasets: LiRA-CD Danish Highway M13 and Urban CPH1
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
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from controllers.lqr import LQRConfig, LQRController
from controllers.mpc import MPCConfig, MPCController
from controllers.dosmc import DOSMCConfig, DOSMCController
from roads.reference_roads import ReferenceProfile1, ReferenceProfile2, ReferenceProfile3
from roads.measured_road import load_measured_road_profile


def run_reference_paper_study(
    output_dir: str | Path = "results",
    duration_s: float = 6.0,
    dt: float = 0.002
) -> Dict[str, Any]:
    """Execute end-to-end multi-controller benchmark on reference paper profiles and real roads."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use the reference paper vehicle parameters: ms=290kg, mu=59kg, ks=14.5kN/m, cs=1385.4, kt=190kN/m
    params = VehicleParameters(
        ms=290.0,
        mu=59.0,
        ks=14500.0,
        cs=1385.4,
        kt=190000.0,
        max_travel=0.12
    )
    actuator = ActuatorModel(max_force=1800.0, min_force=-1800.0, slew_rate_limit=18000.0)
    model = QuarterCarModel(params)

    # 1. Build Controllers
    lqr_ctrl = LQRController(
        LQRConfig(name="lqr", q1=3.0e4, q2=2.0e4, q3=2.0e4, r0=2.0e-5),
        vehicle_params=params
    )

    mpc_ctrl = MPCController(
        MPCConfig(
            name="mpc",
            prediction_horizon=12,
            dt=dt,
            q1=5.0e4,
            q2=2.0e4,
            q3=2.0e4,
            r0=2.0e-5,
            r_rate=1.0e-7,
            max_force=actuator.max_force,
            min_force=actuator.min_force,
            slew_rate_limit=actuator.slew_rate_limit
        ),
        vehicle_params=params
    )

    dosmc_no_us = DOSMCController(
        DOSMCConfig(name="dosmc_no_us", S=4.0, k=10.0, W=100.0, k_st=0.0, epsilon=0.05, max_force=1800.0, min_force=-1800.0),
        vehicle_params=params
    )

    dosmc_with_us = DOSMCController(
        DOSMCConfig(name="dosmc_with_us", S=4.0, k=10.0, W=100.0, k_st=0.5, epsilon=0.05, max_force=1800.0, min_force=-1800.0),
        vehicle_params=params
    )

    # Define test road profiles
    test_roads = [
        ("ref_profile_1", "Ref Paper Profile 1 (Bump+Sinusoid)", ReferenceProfile1(duration_s=duration_s).get_time_callable()),
        ("ref_profile_2", "Ref Paper Profile 2 (Harmonic)", ReferenceProfile2(duration_s=duration_s).get_time_callable()),
        ("ref_profile_3", "Ref Paper Profile 3 (Multi-Harmonic)", ReferenceProfile3(duration_s=duration_s).get_time_callable()),
        ("lira_highway_m13", "LiRA Highway M13 (90 km/h)", load_measured_road_profile("lira_highway_m13").get_time_callable(90.0 / 3.6)),
        ("lira_urban_cph1", "LiRA Urban CPH1 (50 km/h)", load_measured_road_profile("lira_urban_cph1").get_time_callable(50.0 / 3.6)),
    ]

    records: List[Dict[str, Any]] = []
    saved_sims: Dict[str, Dict[str, SimulationResult]] = {}

    for road_id, road_name, road_fn in test_roads:
        saved_sims[road_id] = {}

        # 1. Passive
        res_passive = model.simulate(road_func=road_fn, duration=duration_s, dt=dt)
        saved_sims[road_id]["passive"] = res_passive
        passive_accel = res_passive.rms_body_accel

        records.append({
            "road_id": road_id,
            "road_name": road_name,
            "controller_id": "passive",
            "controller_name": "Passive Suspension",
            "rms_body_accel_mps2": float(res_passive.rms_body_accel),
            "peak_body_accel_mps2": float(res_passive.peak_body_accel),
            "accel_imprv_pct": 0.0,
            "rms_susp_travel_mm": float(res_passive.rms_suspension_travel * 1000.0),
            "peak_susp_travel_mm": float(res_passive.peak_suspension_travel * 1000.0),
            "rms_dynamic_tyre_load_n": float(res_passive.rms_dynamic_tyre_load),
            "peak_dynamic_tyre_load_n": float(res_passive.peak_dynamic_tyre_load),
            "electrical_energy_j": 0.0
        })

        # 2. LQR
        actuator_lqr = ActuatorModel(max_force=1800.0, min_force=-1800.0, slew_rate_limit=18000.0)
        def lqr_cb(t, x, zr):
            u_cmd = lqr_ctrl.compute_force(x)
            u_r, u_a, _ = actuator_lqr.apply_limits(u_cmd, dt)
            return u_r, u_a, {"active_controller": "lqr"}

        res_lqr = model.simulate(road_func=road_fn, controller_func=lqr_cb, duration=duration_s, dt=dt, actuator=actuator_lqr)
        saved_sims[road_id]["lqr"] = res_lqr
        imprv_lqr = ((passive_accel - res_lqr.rms_body_accel) / (passive_accel + 1e-12)) * 100.0
        records.append({
            "road_id": road_id,
            "road_name": road_name,
            "controller_id": "lqr",
            "controller_name": "Fixed LQR (CARE)",
            "rms_body_accel_mps2": float(res_lqr.rms_body_accel),
            "peak_body_accel_mps2": float(res_lqr.peak_body_accel),
            "accel_imprv_pct": float(imprv_lqr),
            "rms_susp_travel_mm": float(res_lqr.rms_suspension_travel * 1000.0),
            "peak_susp_travel_mm": float(res_lqr.peak_suspension_travel * 1000.0),
            "rms_dynamic_tyre_load_n": float(res_lqr.rms_dynamic_tyre_load),
            "peak_dynamic_tyre_load_n": float(res_lqr.peak_dynamic_tyre_load),
            "electrical_energy_j": float(res_lqr.total_electrical_energy_j)
        })

        # 3. MPC
        mpc_ctrl.prev_u_applied = 0.0
        def mpc_cb(t, x, zr):
            u_cmd, info = mpc_ctrl.compute_force(x, current_road_elev=zr)
            return u_cmd, info["u_applied"], info

        res_mpc = model.simulate(road_func=road_fn, controller_func=mpc_cb, duration=duration_s, dt=dt)
        saved_sims[road_id]["mpc"] = res_mpc
        imprv_mpc = ((passive_accel - res_mpc.rms_body_accel) / (passive_accel + 1e-12)) * 100.0
        records.append({
            "road_id": road_id,
            "road_name": road_name,
            "controller_id": "mpc",
            "controller_name": "Constrained MPC (QP)",
            "rms_body_accel_mps2": float(res_mpc.rms_body_accel),
            "peak_body_accel_mps2": float(res_mpc.peak_body_accel),
            "accel_imprv_pct": float(imprv_mpc),
            "rms_susp_travel_mm": float(res_mpc.rms_suspension_travel * 1000.0),
            "peak_susp_travel_mm": float(res_mpc.peak_suspension_travel * 1000.0),
            "rms_dynamic_tyre_load_n": float(res_mpc.rms_dynamic_tyre_load),
            "peak_dynamic_tyre_load_n": float(res_mpc.peak_dynamic_tyre_load),
            "electrical_energy_j": float(res_mpc.total_electrical_energy_j)
        })

        # 4. DOSMC without u_s
        dosmc_no_us.reset()
        res_dosmc_no_us = model.simulate(road_func=road_fn, controller_func=dosmc_no_us, duration=duration_s, dt=dt)
        saved_sims[road_id]["dosmc_no_us"] = res_dosmc_no_us
        imprv_d1 = ((passive_accel - res_dosmc_no_us.rms_body_accel) / (passive_accel + 1e-12)) * 100.0
        records.append({
            "road_id": road_id,
            "road_name": road_name,
            "controller_id": "dosmc_no_us",
            "controller_name": "DOSMC without u_s",
            "rms_body_accel_mps2": float(res_dosmc_no_us.rms_body_accel),
            "peak_body_accel_mps2": float(res_dosmc_no_us.peak_body_accel),
            "accel_imprv_pct": float(imprv_d1),
            "rms_susp_travel_mm": float(res_dosmc_no_us.rms_suspension_travel * 1000.0),
            "peak_susp_travel_mm": float(res_dosmc_no_us.peak_suspension_travel * 1000.0),
            "rms_dynamic_tyre_load_n": float(res_dosmc_no_us.rms_dynamic_tyre_load),
            "peak_dynamic_tyre_load_n": float(res_dosmc_no_us.peak_dynamic_tyre_load),
            "electrical_energy_j": float(res_dosmc_no_us.total_electrical_energy_j)
        })

        # 5. DOSMC with u_s
        dosmc_with_us.reset()
        res_dosmc_with_us = model.simulate(road_func=road_fn, controller_func=dosmc_with_us, duration=duration_s, dt=dt)
        saved_sims[road_id]["dosmc_with_us"] = res_dosmc_with_us
        imprv_d2 = ((passive_accel - res_dosmc_with_us.rms_body_accel) / (passive_accel + 1e-12)) * 100.0
        records.append({
            "road_id": road_id,
            "road_name": road_name,
            "controller_id": "dosmc_with_us",
            "controller_name": "DOSMC with u_s (Ref Paper)",
            "rms_body_accel_mps2": float(res_dosmc_with_us.rms_body_accel),
            "peak_body_accel_mps2": float(res_dosmc_with_us.peak_body_accel),
            "accel_imprv_pct": float(imprv_d2),
            "rms_susp_travel_mm": float(res_dosmc_with_us.rms_suspension_travel * 1000.0),
            "peak_susp_travel_mm": float(res_dosmc_with_us.peak_suspension_travel * 1000.0),
            "rms_dynamic_tyre_load_n": float(res_dosmc_with_us.rms_dynamic_tyre_load),
            "peak_dynamic_tyre_load_n": float(res_dosmc_with_us.peak_dynamic_tyre_load),
            "electrical_energy_j": float(res_dosmc_with_us.total_electrical_energy_j)
        })

    df = pd.DataFrame(records)
    csv_path = tables_dir / "reference_study_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Plot multi-panel comparison figure
    fig, axes = plt.subplots(len(test_roads), 2, figsize=(14, 3.2 * len(test_roads)), squeeze=False)
    for r_idx, (road_id, road_name, _) in enumerate(test_roads):
        ax_accel = axes[r_idx, 0]
        ax_ctrl = axes[r_idx, 1]

        sim_p = saved_sims[road_id]["passive"]
        sim_lqr = saved_sims[road_id]["lqr"]
        sim_dosmc = saved_sims[road_id]["dosmc_with_us"]

        ax_accel.plot(sim_p.t, sim_p.zs_ddot, label="Passive", color="#7f7f7f", alpha=0.6, lw=1.0)
        ax_accel.plot(sim_lqr.t, sim_lqr.zs_ddot, label="LQR", color="#1f77b4", alpha=0.8, lw=1.2)
        ax_accel.plot(sim_dosmc.t, sim_dosmc.zs_ddot, label="DOSMC (Ref Paper)", color="#2ca02c", lw=1.5)
        ax_accel.set_title(f"Sprung Mass Acceleration - {road_name}")
        ax_accel.set_ylabel("Accel [m/s²]")
        ax_accel.set_xlabel("Time [s]")
        ax_accel.legend(loc="upper right", fontsize=8)
        ax_accel.grid(True, alpha=0.3)

        ax_ctrl.plot(sim_lqr.t, sim_lqr.u_applied, label="LQR", color="#1f77b4", lw=1.2)
        ax_ctrl.plot(sim_dosmc.t, sim_dosmc.u_applied, label="DOSMC (Ref Paper)", color="#2ca02c", lw=1.5)
        ax_ctrl.axhline(1800, color="r", linestyle="--", alpha=0.5, label="Limit (±1800N)")
        ax_ctrl.axhline(-1800, color="r", linestyle="--", alpha=0.5)
        ax_ctrl.set_title(f"Actuator Force - {road_name}")
        ax_ctrl.set_ylabel("Force [N]")
        ax_ctrl.set_xlabel("Time [s]")
        ax_ctrl.legend(loc="upper right", fontsize=8)
        ax_ctrl.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = figures_dir / "reference_paper_benchmark.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {"dataframe": df, "csv_path": str(csv_path), "fig_path": str(fig_path)}
