"""Offline PSO optimization runner for comfort, balanced, and road-holding LQR controllers.

Uses strictly separated training scenarios to tune meaningful LQR diagonal weights,
logs convergence curves, and serializes optimized controller configurations.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.actuator import ActuatorModel
from controllers.lqr import LQRConfig, LQRController
from roads.road_dataset import create_scenario, RoadScenario
from optimization.parameterization import (
    DEFAULT_LOWER_BOUNDS,
    DEFAULT_UPPER_BOUNDS,
    decode_parameters,
    PARAM_NAMES
)
from optimization.objective import evaluate_lqr_candidate
from optimization.pso import ParticleSwarmOptimizer


def generate_pso_training_scenarios(duration_s: float = 4.0) -> Dict[str, List[RoadScenario]]:
    """Build dedicated, isolated training scenarios for PSO tuning.

    NOTE: These scenario seeds are dedicated to PSO and never used in final benchmark testing.
    """
    return {
        "comfort": [
            create_scenario(scenario_id=101, road_type="smooth", speed_kmh=50.0, split="pso", duration_s=duration_s, seed=101),
            create_scenario(scenario_id=102, road_type="smooth", speed_kmh=80.0, split="pso", duration_s=duration_s, seed=102),
        ],
        "balanced": [
            create_scenario(scenario_id=201, road_type="bumpy", speed_kmh=40.0, split="pso", duration_s=duration_s, seed=201),
            create_scenario(scenario_id=202, road_type="bumpy", speed_kmh=60.0, split="pso", duration_s=duration_s, seed=202),
        ],
        "road_holding": [
            create_scenario(scenario_id=301, road_type="rough", speed_kmh=50.0, split="pso", duration_s=duration_s, seed=301),
            create_scenario(scenario_id=302, road_type="rough", speed_kmh=70.0, split="pso", duration_s=duration_s, seed=302),
        ]
    }


def run_pso_tuning(
    swarm_size: int = 15,
    max_iterations: int = 20,
    seed: int = 42,
    output_configs_path: str | Path = "artifacts/controllers/pso_lqr_configs.json",
    output_results_path: str | Path = "artifacts/optimization/pso_results.json",
    output_plot_path: str | Path = "results/figures/pso_convergence.png",
    quick_mode: bool = False
) -> Dict[str, Any]:
    """Execute offline PSO tuning for all three specialized LQR controllers.

    Args:
        swarm_size: Number of particles.
        max_iterations: Maximum optimization iterations.
        seed: Random seed.
        output_configs_path: Path to save tuned controller parameters.
        output_results_path: Path to save PSO convergence data.
        output_plot_path: Path to save convergence plot.
        quick_mode: If True, uses small swarm and iterations for fast test execution.

    Returns:
        Dictionary containing optimized configurations and convergence histories.
    """
    if quick_mode:
        swarm_size = 8
        max_iterations = 8

    pso_scenarios = generate_pso_training_scenarios(duration_s=3.0 if quick_mode else 5.0)
    vehicle_params = VehicleParameters()
    actuator = ActuatorModel()

    modes = ["comfort", "balanced", "road_holding"]
    pso_results = {}
    optimized_configs = {}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for idx, mode in enumerate(modes):
        training_sc = pso_scenarios[mode]

        def cost_func(p_vec: np.ndarray) -> float:
            eval_res = evaluate_lqr_candidate(
                particle_p=p_vec,
                target_mode=mode,
                training_scenarios=training_sc,
                vehicle_params=vehicle_params,
                actuator=actuator,
                dt=0.005 if quick_mode else 0.002
            )
            return float(eval_res["cost"])

        pso = ParticleSwarmOptimizer(
            cost_function=cost_func,
            lower_bounds=DEFAULT_LOWER_BOUNDS,
            upper_bounds=DEFAULT_UPPER_BOUNDS,
            swarm_size=swarm_size,
            max_iterations=max_iterations,
            seed=seed + idx * 10
        )

        best_pos, best_cost, history = pso.optimize(verbose=False)
        cfg = decode_parameters(best_pos, name=f"pso_{mode}")
        ctrl = LQRController(cfg, vehicle_params)

        pso_results[mode] = {
            "best_cost": best_cost,
            "best_position": best_pos.tolist(),
            "runtime_seconds": pso.runtime_seconds,
            "history": history,
            "gain_K": ctrl.K.flatten().tolist(),
            "eigenvalues": [str(e) for e in ctrl.eigenvalues],
            "is_stable": ctrl.is_stable
        }

        optimized_configs[mode] = {
            "name": f"pso_{mode}",
            "q1": cfg.q1,
            "q2": cfg.q2,
            "q3": cfg.q3,
            "r0": cfg.r0,
            "gain_K": ctrl.K.flatten().tolist(),
            "description": cfg.description
        }

        # Plot convergence curve
        iters = [h["iteration"] for h in history]
        bests = [h["best_cost"] for h in history]
        ax.plot(iters, bests, marker="o", linewidth=2, label=f"PSO {mode.capitalize()}")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Normalized Objective Cost", fontsize=12)
    ax.set_title("PSO LQR Optimization Convergence", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()

    # Save outputs
    output_plot_path = Path(output_plot_path)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path)
    plt.close(fig)

    output_configs_path = Path(output_configs_path)
    output_configs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_configs_path, "w", encoding="utf-8") as f:
        json.dump(optimized_configs, f, indent=2)

    output_results_path = Path(output_results_path)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(pso_results, f, indent=2)

    return {
        "configs": optimized_configs,
        "results": pso_results
    }
