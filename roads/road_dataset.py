"""Road dataset scenario generation and management.

Provides clean isolation between training, validation, PSO tuning, and held-out test scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from roads.base import BaseRoadProfile
from roads.smooth import SmoothRoadGenerator
from roads.bump import BumpRoadGenerator
from roads.rough import RoughRoadGenerator


@dataclass
class RoadScenario:
    """A fully-defined road experiment scenario."""
    scenario_id: int
    road_type: str          # 'smooth', 'bumpy', 'rough'
    speed_kmh: float        # Vehicle speed [km/h]
    duration_s: float       # Duration of simulation run [s]
    road_profile: BaseRoadProfile
    split: str              # 'train', 'val', 'pso', 'test'
    description: str

    @property
    def speed_mps(self) -> float:
        """Vehicle longitudinal speed converted to meters per second [m/s]."""
        return self.speed_kmh / 3.6


def create_scenario(
    scenario_id: int,
    road_type: str,
    speed_kmh: float,
    split: str,
    duration_s: float = 6.0,
    seed: Optional[int] = None
) -> RoadScenario:
    """Factory function to build a single RoadScenario with appropriate parameters."""
    road_type = road_type.lower()
    seed = seed if seed is not None else (1000 + scenario_id)

    if road_type == "smooth":
        # Variations in Class A roughness
        roughness = 0.8 + 0.4 * ((scenario_id % 5) / 5.0)
        profile = SmoothRoadGenerator(
            seed=seed,
            roughness_factor=roughness,
            name=f"Smooth_s{scenario_id}"
        )
        desc = f"ISO Class A Smooth Road, roughness factor={roughness:.2f}, speed={speed_kmh:.1f} km/h"

    elif road_type in ("bump", "bumpy"):
        road_type = "bumpy"
        # Speed bump height 0.035 to 0.065 m, width 0.6 to 1.0 m, start at 15 m
        h = 0.04 + 0.02 * ((scenario_id % 4) / 4.0)
        w = 0.7 + 0.3 * ((scenario_id % 3) / 3.0)
        # Add occasional double bump for variety
        if scenario_id % 3 == 0:
            bumps = [(15.0, h, w), (22.0, h * 0.9, w)]
        else:
            bumps = [(15.0, h, w)]
        profile = BumpRoadGenerator(
            bumps=bumps,
            seed=seed,
            name=f"Bump_s{scenario_id}"
        )
        desc = f"Speed Bump profile, H={h*1000:.0f} mm, W={w:.2f} m, speed={speed_kmh:.1f} km/h"

    elif road_type == "rough":
        # Variations between Class C and Class D
        iso_cls = "C" if (scenario_id % 2 == 0) else "D"
        scale = 0.8 + 0.4 * ((scenario_id % 3) / 3.0)
        profile = RoughRoadGenerator(
            iso_class=iso_cls,
            seed=seed,
            roughness_scale=scale,
            name=f"Rough_s{scenario_id}"
        )
        desc = f"ISO Class {iso_cls} Rough Road, scale={scale:.2f}, speed={speed_kmh:.1f} km/h"
    else:
        raise ValueError(f"Unknown road_type '{road_type}'. Must be 'smooth', 'bumpy', or 'rough'.")

    return RoadScenario(
        scenario_id=scenario_id,
        road_type=road_type,
        speed_kmh=speed_kmh,
        duration_s=duration_s,
        road_profile=profile,
        split=split,
        description=desc
    )


def generate_benchmark_dataset(
    num_per_class: int = 15,
    speeds_kmh: Optional[List[float]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    duration_s: float = 6.0,
    base_seed: int = 1000
) -> Dict[str, List[RoadScenario]]:
    """Generate a balanced set of scenarios partitioned into train, val, and test splits.

    Guarantees scenario-level independence so that overlapping trajectories
    never appear in multiple splits (preventing data leakage).

    Returns:
        Dictionary mapping split name ('train', 'val', 'test', 'all') to scenario lists.
    """
    speeds = speeds_kmh or [30.0, 50.0, 70.0, 90.0]
    classes = ["smooth", "bumpy", "rough"]
    all_scenarios: List[RoadScenario] = []

    scenario_counter = 0

    for c_idx, road_cls in enumerate(classes):
        # Determine split boundaries per class to preserve class balance across splits
        n_train = int(round(num_per_class * train_ratio))
        n_val = int(round(num_per_class * val_ratio))
        n_test = num_per_class - n_train - n_val

        for i in range(num_per_class):
            scenario_counter += 1
            spd = speeds[i % len(speeds)]
            unique_seed = base_seed + 100 * c_idx + i

            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            sc = create_scenario(
                scenario_id=scenario_counter,
                road_type=road_cls,
                speed_kmh=spd,
                split=split,
                duration_s=duration_s,
                seed=unique_seed
            )
            all_scenarios.append(sc)

    return {
        "train": [s for s in all_scenarios if s.split == "train"],
        "val": [s for s in all_scenarios if s.split == "val"],
        "test": [s for s in all_scenarios if s.split == "test"],
        "all": all_scenarios
    }
