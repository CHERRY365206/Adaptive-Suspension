"""Batch simulation runner executing comparative benchmark experiments.

Tests Passive, Fixed Nominal LQR, Fixed PSO LQR, Adaptive (No PSO), Proposed AI+PSO, and Oracle
across identical held-out test scenarios with strict reproducibility.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.vehicle_parameters import VehicleParameters
from models.actuator import ActuatorModel
from experiments.scenarios import get_held_out_benchmark_scenarios
from experiments.baselines import SuspensionBenchmarkSuite, SYSTEM_NAMES, SYSTEM_DISPLAY_NAMES
from experiments.metrics import compute_simulation_metrics
from experiments.statistical_analysis import aggregate_experiment_results


def run_comparative_benchmark(
    scenarios=None,
    systems_to_evaluate: Optional[List[str]] = None,
    output_csv_path: str | Path = "results/tables/experiments_summary.csv",
    output_json_path: str | Path = "results/tables/aggregated_metrics.json",
    output_fig_path: str | Path = "results/figures/comparative_metrics.png",
    output_ts_fig_path: str | Path = "results/figures/time_series_comparison.png",
    quick_mode: bool = False,
    dt: float = 0.002
) -> Dict[str, Any]:
    """Execute complete comparative benchmark across all systems on held-out scenarios.

    Args:
        scenarios: List of test RoadScenario objects. If None, generated automatically.
        systems_to_evaluate: List of system names (defaults to all 6 systems).
        output_csv_path: Path for tabular CSV export.
        output_json_path: Path for aggregated JSON export.
        output_fig_path: Path for comparison bar chart.
        output_ts_fig_path: Path for time-domain overlay plot.
        quick_mode: Fast execution mode for quick verification.
        dt: Simulation time step [s].

    Returns:
        Dictionary containing raw records and aggregated statistical summaries.
    """
    if scenarios is None:
        scenarios = get_held_out_benchmark_scenarios(duration_s=4.0 if quick_mode else 6.0, quick_mode=quick_mode)

    systems = systems_to_evaluate or SYSTEM_NAMES
    suite = SuspensionBenchmarkSuite()

    records: List[Dict[str, Any]] = []
    saved_time_series_res = {}

    for sc_idx, sc in enumerate(scenarios):
        road_fn = sc.road_profile.get_time_callable(sc.speed_mps)

        for sys_name in systems:
            sim_res = suite.simulate_system(
                system_type=sys_name,
                road_func=road_fn,
                ground_truth_class=sc.road_type,
                duration_s=sc.duration_s,
                dt=dt
            )

            metrics = compute_simulation_metrics(sim_res, ground_truth_class=sc.road_type)
            record = {
                "scenario_id": sc.scenario_id,
                "road_type": sc.road_type,
                "speed_kmh": sc.speed_kmh,
                "system_type": sys_name,
                "system_display_name": SYSTEM_DISPLAY_NAMES[sys_name],
                **metrics
            }
            records.append(record)

            # Save the first bump scenario for publication time-series overlay
            if sc.road_type == "bumpy" and sc_idx == 0:
                saved_time_series_res[sys_name] = sim_res

    # 1. Export CSV
    df = pd.DataFrame(records)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # 2. Compute statistical aggregations
    agg_results = aggregate_experiment_results(records)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2)

    # 3. Generate Comparative Metric Bar Chart
    _generate_comparison_bar_chart(df, output_fig_path)

    # 4. Generate Time Series Overlay Plot
    if "passive" in saved_time_series_res and "proposed_ai_pso" in saved_time_series_res:
        _generate_time_series_plot(
            saved_time_series_res["passive"],
            saved_time_series_res["proposed_ai_pso"],
            output_ts_fig_path
        )

    return {
        "records": records,
        "dataframe": df,
        "aggregated": agg_results
    }


def _generate_comparison_bar_chart(df: pd.DataFrame, output_path: str | Path) -> None:
    """Generate and save publication bar chart comparing key metrics across systems."""
    summary = df.groupby("system_type")[["rms_body_accel_mps2", "rms_susp_travel_m", "rms_dynamic_tyre_load_n"]].mean()
    systems_ordered = [s for s in SYSTEM_NAMES if s in summary.index]
    summary = summary.loc[systems_ordered]

    labels = [SYSTEM_DISPLAY_NAMES[s].replace("Suspension", "").replace("LQR", "").strip() for s in systems_ordered]

    x = np.arange(len(systems_ordered))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot RMS body acceleration (Comfort)
    rects1 = ax1.bar(x - width, summary["rms_body_accel_mps2"], width, label="RMS Body Accel [m/s²]", color="#1f77b4")
    ax1.set_ylabel("RMS Body Acceleration [m/s²]", color="#1f77b4", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # Second axis for travel and tyre load
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x, summary["rms_susp_travel_m"] * 1000.0, width, label="RMS Travel [mm]", color="#2ca02c")
    rects3 = ax2.bar(x + width, summary["rms_dynamic_tyre_load_n"] / 100.0, width, label="RMS Dynamic Tyre Load [x100 N]", color="#d62728")
    ax2.set_ylabel("Travel [mm] & Tyre Load [x100 N]", color="#333333", fontsize=11, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax1.set_title("Suspension Performance Benchmark Across Evaluation Scenarios", fontsize=13, fontweight="bold", pad=15)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def _generate_time_series_plot(res_passive, res_active, output_path: str | Path) -> None:
    """Generate time domain comparison of Passive vs Proposed AI+PSO Active Suspension."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, dpi=300)

    t = res_passive.t

    # 1. Road elevation & Sprung displacement
    axes[0].plot(t, res_passive.zr * 1000.0, "k--", label="Road Profile $z_r$", alpha=0.7)
    axes[0].plot(t, res_passive.zs * 1000.0, "r-", label="Passive $z_s$")
    axes[0].plot(t, res_active.zs * 1000.0, "b-", label="Proposed Active $z_s$")
    axes[0].set_ylabel("Disp. [mm]", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Dynamic Time-Domain Response Comparison (Speed Bump)", fontsize=12, fontweight="bold")

    # 2. Body Acceleration
    axes[1].plot(t, res_passive.zs_ddot, "r-", label=r"Passive $\ddot{z}_s$")
    axes[1].plot(t, res_active.zs_ddot, "b-", label=r"Proposed Active $\ddot{z}_s$")
    axes[1].set_ylabel("Accel [m/s²]", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Dynamic Tyre Load
    axes[2].plot(t, res_passive.dynamic_tyre_load, "r-", label="Passive $F_{td}$")
    axes[2].plot(t, res_active.dynamic_tyre_load, "b-", label="Proposed Active $F_{td}$")
    axes[2].set_ylabel("Tyre Force [N]", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # 4. Actuator Force
    axes[3].plot(t, res_active.u_requested, "m--", label="Requested $u_{req}$", alpha=0.6)
    axes[3].plot(t, res_active.u_applied, "g-", label="Applied $u_{applied}$")
    axes[3].set_ylabel("Force [N]", fontsize=10)
    axes[3].set_xlabel("Time [s]", fontsize=11)
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
