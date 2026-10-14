"""Empirical comparison study evaluating Passive, Fixed LQR, and Constrained MPC on real measured road data.

Uses physical laser profilometer data from the LiRA-CD dataset (Danish Road Directorate P79 profilometer).
Generates non-fabricated comparative simulation time series, performance metrics, and publication figures.
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
from models.actuator import ActuatorModel
from models.quarter_car import QuarterCarModel, SimulationResult
from roads.measured_road import load_measured_road_profile
from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import build_nominal_controllers
from controllers.mpc import MPCConfig, MPCController
from experiments.metrics import compute_simulation_metrics
from experiments.statistical_analysis import compute_percentage_improvement


def run_real_road_study(
    traces: Optional[List[str]] = None,
    speeds_kmh: Optional[Dict[str, List[float]]] = None,
    duration_s: float = 6.0,
    dt: float = 0.002,
    output_dir: str | Path = "results",
    vehicle_params: Optional[VehicleParameters] = None,
    actuator: Optional[ActuatorModel] = None
) -> Dict[str, Any]:
    """Execute rigorous comparative simulation on real measured road profiles.

    Evaluates:
      1. Passive Suspension (Baseline)
      2. Fixed Nominal LQR (Retained Benchmark)
      3. Constrained Discrete-Time MPC (Proposed Active Controller)
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = vehicle_params or VehicleParameters()
    actuator_model = actuator or ActuatorModel()
    model = QuarterCarModel(params)

    # Build controllers
    lqr_controllers = build_nominal_controllers(params)
    lqr_ctrl = lqr_controllers["balanced"]

    mpc_cfg = MPCConfig(
        prediction_horizon=12,
        dt=dt,
        q1=5.0e4,
        q2=2.0e4,
        q3=2.0e4,
        r0=2.0e-5,
        r_rate=1.0e-7,
        max_force=actuator_model.max_force,
        min_force=actuator_model.min_force,
        slew_rate_limit=actuator_model.slew_rate_limit
    )
    mpc_ctrl = MPCController(mpc_cfg, params)

    if traces is None:
        traces = ["lira_highway_m13", "lira_urban_cph1"]

    if speeds_kmh is None:
        speeds_kmh = {
            "lira_highway_m13": [90.0, 110.0],
            "lira_urban_cph1": [45.0, 60.0]
        }

    records: List[Dict[str, Any]] = []
    saved_time_series: Dict[str, Dict[str, SimulationResult]] = {}

    systems = ["passive", "fixed_nominal_lqr", "mpc"]

    for trace_key in traces:
        road = load_measured_road_profile(trace_key)
        saved_time_series[trace_key] = {}

        for sp_kmh in speeds_kmh.get(trace_key, [60.0]):
            sp_mps = sp_kmh / 3.6
            road_fn = road.get_time_callable(sp_mps)

            # Store results for this scenario
            scenario_sims: Dict[str, SimulationResult] = {}

            # 1. Passive Simulation
            act_passive = ActuatorModel(enabled=False)
            res_passive = model.simulate(
                road_func=road_fn,
                controller_func=None,
                duration=duration_s,
                dt=dt,
                actuator=act_passive
            )
            scenario_sims["passive"] = res_passive
            m_pass = compute_simulation_metrics(res_passive, ground_truth_class=road.route_type)
            records.append({
                "trace_key": trace_key,
                "road_name": road.name,
                "route_type": road.route_type,
                "speed_kmh": sp_kmh,
                "system_type": "passive",
                "system_name": "Passive Suspension",
                **m_pass
            })

            # 2. Fixed Nominal LQR
            actuator_model.reset()
            def lqr_callback(t, x, zr):
                u_cmd = lqr_ctrl.compute_force(x)
                u_r, u_a, is_sat = actuator_model.apply_limits(u_cmd, dt)
                return u_r, u_a, {"active_controller": "fixed_nominal_lqr"}

            res_lqr = model.simulate(
                road_func=road_fn,
                controller_func=lqr_callback,
                duration=duration_s,
                dt=dt,
                actuator=actuator_model
            )
            scenario_sims["fixed_nominal_lqr"] = res_lqr
            m_lqr = compute_simulation_metrics(res_lqr, ground_truth_class=road.route_type)
            records.append({
                "trace_key": trace_key,
                "road_name": road.name,
                "route_type": road.route_type,
                "speed_kmh": sp_kmh,
                "system_type": "fixed_nominal_lqr",
                "system_name": "Fixed Nominal LQR",
                **m_lqr
            })

            # 3. Constrained MPC
            mpc_ctrl.reset()
            actuator_model.reset()
            def mpc_callback(t, x, zr):
                u_cmd, info = mpc_ctrl.compute_force(x, current_road_elev=zr)
                u_r, u_a, is_sat = actuator_model.apply_limits(u_cmd, dt)
                info["active_controller"] = "mpc"
                return u_r, u_a, info

            res_mpc = model.simulate(
                road_func=road_fn,
                controller_func=mpc_callback,
                duration=duration_s,
                dt=dt,
                actuator=actuator_model
            )
            scenario_sims["mpc"] = res_mpc
            m_mpc = compute_simulation_metrics(res_mpc, ground_truth_class=road.route_type)
            records.append({
                "trace_key": trace_key,
                "road_name": road.name,
                "route_type": road.route_type,
                "speed_kmh": sp_kmh,
                "system_type": "mpc",
                "system_name": "Constrained MPC",
                **m_mpc
            })

            # Save representative run for plots (first speed tested)
            if f"{sp_kmh:.0f}" not in saved_time_series[trace_key]:
                saved_time_series[trace_key][f"{sp_kmh:.0f}"] = scenario_sims

    # Export CSV
    df = pd.DataFrame(records)
    csv_file = tables_dir / "real_road_benchmark_summary.csv"
    df.to_csv(csv_file, index=False)

    # Compute improvements relative to passive
    improvements = _compute_relative_improvements(df)
    json_file = tables_dir / "real_road_aggregated_metrics.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(improvements, f, indent=2)

    # Generate publication figures
    _plot_real_road_time_series(saved_time_series, figures_dir)
    _plot_real_road_bar_chart(df, figures_dir / "real_road_comparison_bar_chart.png")

    return {
        "dataframe": df,
        "improvements": improvements,
        "csv_path": str(csv_file),
        "json_path": str(json_file)
    }


def _compute_relative_improvements(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate percentage improvements over passive baseline for each trace and controller."""
    results = {}
    grouped = df.groupby(["trace_key", "speed_kmh"])

    for (trace, speed), group in grouped:
        pass_row = group[group["system_type"] == "passive"]
        if pass_row.empty:
            continue
        pass_data = pass_row.iloc[0]
        scenario_key = f"{trace}_{speed:.0f}kmh"
        results[scenario_key] = {"trace": trace, "speed_kmh": float(speed), "controllers": {}}

        for _, row in group.iterrows():
            sys = row["system_type"]
            if sys == "passive":
                continue

            accel_imp = compute_percentage_improvement(pass_data["rms_body_accel_mps2"], row["rms_body_accel_mps2"])
            travel_imp = compute_percentage_improvement(pass_data["rms_susp_travel_m"], row["rms_susp_travel_m"])
            tyre_imp = compute_percentage_improvement(pass_data["rms_dynamic_tyre_load_n"], row["rms_dynamic_tyre_load_n"])

            results[scenario_key]["controllers"][sys] = {
                "rms_body_accel_mps2": float(row["rms_body_accel_mps2"]),
                "accel_improvement_pct": float(accel_imp),
                "rms_susp_travel_m": float(row["rms_susp_travel_m"]),
                "travel_improvement_pct": float(travel_imp),
                "rms_dynamic_tyre_load_n": float(row["rms_dynamic_tyre_load_n"]),
                "tyre_load_improvement_pct": float(tyre_imp),
                "electrical_energy_j": float(row.get("electrical_energy_j", 0.0)),
                "contact_loss_pct": float(row.get("contact_loss_pct", 0.0))
            }

    return results


def _plot_real_road_time_series(saved_time_series: Dict[str, Dict[str, Dict[str, SimulationResult]]], out_dir: Path) -> None:
    """Generate high-resolution time series comparisons on real measured roads."""
    for trace_key, speed_dict in saved_time_series.items():
        if not speed_dict:
            continue
        first_speed = list(speed_dict.keys())[0]
        sims = speed_dict[first_speed]
        if "passive" not in sims or "fixed_nominal_lqr" not in sims or "mpc" not in sims:
            continue

        p_res = sims["passive"]
        l_res = sims["fixed_nominal_lqr"]
        m_res = sims["mpc"]
        t = p_res.t

        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, dpi=300)
        trace_title = "Highway M13 (P79 Profilometer)" if "highway" in trace_key else "Urban CPH1 (P79 Profilometer)"

        # 1. Road elevation & Sprung mass displacement
        axes[0].plot(t, p_res.zr * 1000.0, "k--", label=r"Measured Road $z_r$", alpha=0.55, linewidth=1.0)
        axes[0].plot(t, p_res.zs * 1000.0, "r-", label=r"Passive $z_s$", alpha=0.85, linewidth=1.2)
        axes[0].plot(t, l_res.zs * 1000.0, "g-", label=r"Fixed LQR $z_s$", alpha=0.85, linewidth=1.2)
        axes[0].plot(t, m_res.zs * 1000.0, "b-", label=r"Constrained MPC $z_s$", linewidth=1.5)
        axes[0].set_ylabel("Disp. [mm]", fontsize=10, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=8.5, ncol=4)
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].set_title(f"Quarter-Car Dynamic Response on Real Measured Trace: {trace_title} @ {first_speed} km/h", fontsize=11, fontweight="bold")

        # 2. Body Vertical Acceleration
        axes[1].plot(t, p_res.zs_ddot, "r-", label=r"Passive $\ddot{z}_s$", alpha=0.8, linewidth=1.0)
        axes[1].plot(t, l_res.zs_ddot, "g-", label=r"Fixed LQR $\ddot{z}_s$", alpha=0.8, linewidth=1.0)
        axes[1].plot(t, m_res.zs_ddot, "b-", label=r"Constrained MPC $\ddot{z}_s$", linewidth=1.3)
        axes[1].set_ylabel("Accel [m/s²]", fontsize=10, fontweight="bold")
        axes[1].legend(loc="upper right", fontsize=8.5, ncol=3)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        # 3. Dynamic Tyre Load
        axes[2].plot(t, p_res.dynamic_tyre_load, "r-", label=r"Passive $F_{td}$", alpha=0.8, linewidth=1.0)
        axes[2].plot(t, l_res.dynamic_tyre_load, "g-", label=r"Fixed LQR $F_{td}$", alpha=0.8, linewidth=1.0)
        axes[2].plot(t, m_res.dynamic_tyre_load, "b-", label=r"Constrained MPC $F_{td}$", linewidth=1.3)
        axes[2].set_ylabel("Tyre Force [N]", fontsize=10, fontweight="bold")
        axes[2].legend(loc="upper right", fontsize=8.5, ncol=3)
        axes[2].grid(True, linestyle="--", alpha=0.4)

        # 4. Actuator Force
        axes[3].plot(t, l_res.u_applied, "g-", label="LQR Applied Force", alpha=0.8, linewidth=1.0)
        axes[3].plot(t, m_res.u_applied, "b-", label="MPC Applied Force", linewidth=1.3)
        axes[3].axhline(1800.0, color="k", linestyle=":", label="Actuator Bounds (±1800 N)", alpha=0.6)
        axes[3].axhline(-1800.0, color="k", linestyle=":", alpha=0.6)
        axes[3].set_ylabel("Force [N]", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
        axes[3].legend(loc="upper right", fontsize=8.5, ncol=3)
        axes[3].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig_path = out_dir / f"real_road_time_series_{trace_key}.png"
        plt.savefig(fig_path)
        plt.close(fig)


def _plot_real_road_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Generate bar chart comparing Passive, LQR, and MPC on real measured roads."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)

    # Average metrics by system type across all measured road traces and speeds
    mean_df = df.groupby("system_type")[["rms_body_accel_mps2", "rms_susp_travel_m", "rms_dynamic_tyre_load_n"]].mean()
    sys_order = ["passive", "fixed_nominal_lqr", "mpc"]
    sys_order = [s for s in sys_order if s in mean_df.index]
    mean_df = mean_df.loc[sys_order]

    labels = ["Passive", "Fixed LQR", "Constrained MPC"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    # 1. Body Acceleration (Comfort)
    axes[0].bar(labels, mean_df["rms_body_accel_mps2"], color=colors, width=0.55)
    axes[0].set_title("RMS Body Acceleration (Comfort)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("RMS Acceleration [m/s²]", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 2. Suspension Travel
    axes[1].bar(labels, mean_df["rms_susp_travel_m"] * 1000.0, color=colors, width=0.55)
    axes[1].set_title("RMS Suspension Travel", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMS Travel [mm]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 3. Dynamic Tyre Load (Road Holding)
    axes[2].bar(labels, mean_df["rms_dynamic_tyre_load_n"], color=colors, width=0.55)
    axes[2].set_title("RMS Dynamic Tyre Load (Grip)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("RMS Tyre Force Variation [N]", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Experimental Evaluation on Real Measured Road Profiles (LiRA-CD Dataset)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
