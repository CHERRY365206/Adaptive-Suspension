"""Statistical analysis and paired comparison utilities for experimental benchmark runs."""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def compute_percentage_improvement(baseline_val: float, active_val: float) -> float:
    """Calculate percentage reduction relative to baseline.

    Formula:
        pct_improvement = ((baseline - active) / baseline) * 100.0
    A positive value denotes a reduction in vibration, travel, or dynamic load (improvement).
    """
    if abs(baseline_val) < 1e-8:
        return 0.0
    return float(((baseline_val - active_val) / baseline_val) * 100.0)


def aggregate_experiment_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate raw simulation records by system and road type, computing means and standard deviations."""
    df = pd.DataFrame(records)
    if df.empty:
        return {}

    numeric_cols = [
        "rms_body_accel_mps2",
        "peak_body_accel_mps2",
        "rms_susp_travel_m",
        "peak_susp_travel_m",
        "rms_dynamic_tyre_load_n",
        "peak_dynamic_tyre_load_n",
        "min_total_tyre_load_n",
        "contact_loss_pct",
        "travel_violation_pct",
        "rms_actuator_force_n",
        "peak_actuator_force_n",
        "saturation_pct",
        "mechanical_work_j",
        "electrical_energy_j"
    ]

    summary_by_system: Dict[str, Any] = {}
    for sys_name, group in df.groupby("system_type"):
        stats_dict = {"num_runs": len(group)}
        for col in numeric_cols:
            if col in group.columns:
                stats_dict[f"{col}_mean"] = float(group[col].mean())
                stats_dict[f"{col}_std"] = float(group[col].std())
                stats_dict[f"{col}_median"] = float(group[col].median())
        summary_by_system[str(sys_name)] = stats_dict

    # Paired comparisons against passive suspension
    paired_comparison = {}
    passive_runs = df[df["system_type"] == "passive"].set_index("scenario_id")

    for sys_name in df["system_type"].unique():
        if sys_name == "passive":
            continue
        sys_runs = df[df["system_type"] == sys_name].set_index("scenario_id")
        common_scenarios = passive_runs.index.intersection(sys_runs.index)

        if len(common_scenarios) > 0:
            pass_accel = passive_runs.loc[common_scenarios, "rms_body_accel_mps2"]
            sys_accel = sys_runs.loc[common_scenarios, "rms_body_accel_mps2"]
            accel_improvements = ((pass_accel - sys_accel) / pass_accel) * 100.0

            pass_travel = passive_runs.loc[common_scenarios, "rms_susp_travel_m"]
            sys_travel = sys_runs.loc[common_scenarios, "rms_susp_travel_m"]
            travel_improvements = ((pass_travel - sys_travel) / pass_travel) * 100.0

            pass_tyre = passive_runs.loc[common_scenarios, "rms_dynamic_tyre_load_n"]
            sys_tyre = sys_runs.loc[common_scenarios, "rms_dynamic_tyre_load_n"]
            tyre_improvements = ((pass_tyre - sys_tyre) / pass_tyre) * 100.0

            paired_comparison[str(sys_name)] = {
                "num_paired_scenarios": len(common_scenarios),
                "body_accel_reduction_pct_mean": float(accel_improvements.mean()),
                "body_accel_reduction_pct_std": float(accel_improvements.std()),
                "susp_travel_reduction_pct_mean": float(travel_improvements.mean()),
                "dynamic_tyre_load_reduction_pct_mean": float(tyre_improvements.mean()),
            }

    return {
        "summary_by_system": summary_by_system,
        "paired_comparison_vs_passive": paired_comparison
    }
