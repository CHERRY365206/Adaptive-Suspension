"""Held-out test scenarios definition for comparative evaluation.

Ensures all 6 suspension systems are tested under identical road profiles, speeds,
and environmental conditions, completely isolated from training and PSO tuning seeds.
"""

from __future__ import annotations
from typing import List
from roads.road_dataset import create_scenario, RoadScenario


def get_held_out_benchmark_scenarios(duration_s: float = 6.0, quick_mode: bool = False) -> List[RoadScenario]:
    """Generate the official held-out benchmark test matrix.

    Args:
        duration_s: Duration of each scenario simulation [s].
        quick_mode: If True, uses a compact test set for faster verification.

    Returns:
        List of strictly held-out RoadScenario instances.
    """
    scenarios: List[RoadScenario] = []

    if quick_mode:
        speeds = [40.0, 70.0]
        seeds = [801, 802]
    else:
        speeds = [40.0, 60.0, 80.0]
        seeds = [801, 802, 803]

    counter = 8000
    for r_type in ["smooth", "bumpy", "rough"]:
        for spd in speeds:
            for s_idx, seed in enumerate(seeds):
                counter += 1
                sc = create_scenario(
                    scenario_id=counter,
                    road_type=r_type,
                    speed_kmh=spd,
                    split="test",
                    duration_s=duration_s,
                    seed=seed + counter
                )
                scenarios.append(sc)

    return scenarios
