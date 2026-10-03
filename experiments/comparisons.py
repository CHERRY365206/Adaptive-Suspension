"""Comparison helper routines and radar chart generation for academic reporting."""

from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_radar_comparison_plot(
    summary_dict: Dict[str, Any],
    output_path: str | Path,
    title: str = "Multi-Objective Suspension Architecture Trade-off"
) -> None:
    """Generate publication radar chart comparing normalized suspension performance indices."""
    categories = ["Ride Comfort\n(1/Accel)", "Travel\nRetention", "Road Holding\n(1/Tyre Load)", "Control\nEfficiency"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)

    # Passive baseline values as 1.0 reference
    passive_stats = summary_dict.get("passive", {})
    if not passive_stats:
        return

    ref_accel = passive_stats.get("rms_body_accel_mps2_mean", 1.0)
    ref_travel = passive_stats.get("rms_susp_travel_m_mean", 1.0)
    ref_tyre = passive_stats.get("rms_dynamic_tyre_load_n_mean", 1.0)

    colors = {
        "passive": "#7f7f7f",
        "fixed_nominal_lqr": "#1f77b4",
        "fixed_pso_lqr": "#ff7f0e",
        "proposed_ai_pso": "#2ca02c",
        "oracle_adaptive": "#9467bd"
    }

    for sys_name, stats in summary_dict.items():
        if sys_name not in colors:
            continue

        accel = stats.get("rms_body_accel_mps2_mean", ref_accel)
        travel = stats.get("rms_susp_travel_m_mean", ref_travel)
        tyre = stats.get("rms_dynamic_tyre_load_n_mean", ref_tyre)
        force = stats.get("rms_actuator_force_n_mean", 0.0)

        # Higher is better: Invert vibration metrics relative to passive
        val_comfort = min(2.0, max(0.2, ref_accel / (accel + 1e-6)))
        val_travel = min(2.0, max(0.2, ref_travel / (travel + 1e-6)))
        val_holding = min(2.0, max(0.2, ref_tyre / (tyre + 1e-6)))
        val_eff = min(2.0, max(0.2, 1000.0 / (force + 100.0)))

        values = [val_comfort, val_travel, val_holding, val_eff]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=sys_name.replace("_", " ").title(), color=colors[sys_name])
        ax.fill(angles, values, color=colors[sys_name], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
