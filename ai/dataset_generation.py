"""Dataset generation module for road classification.

Simulates quarter-car dynamics across independent scenarios, extracts rolling-window
sensor features, and attaches grouping metadata to prevent train/test leakage.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel
from controllers.controller_registry import build_nominal_controllers
from roads.road_dataset import RoadScenario, generate_benchmark_dataset
from ai.feature_extraction import SensorFeatureExtractor, FEATURE_NAMES

LABEL_MAP: Dict[str, int] = {
    "smooth": 0,
    "bumpy": 1,
    "rough": 2
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def generate_scenario_dataset(
    scenarios: List[RoadScenario],
    window_s: float = 0.5,
    stride_s: float = 0.05,
    dt: float = 0.002,
    vehicle_params: Optional[VehicleParameters] = None,
    controllers_to_use: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Generate labeled feature windows from a list of scenarios.

    To ensure the classifier does not learn controller-specific feedback signatures,
    we simulate scenarios under both passive suspension and nominal active suspension.

    Args:
        scenarios: List of RoadScenario objects.
        window_s: Rolling window length [s].
        stride_s: Rolling window stride [s].
        dt: Simulation time step [s].
        vehicle_params: Optional VehicleParameters.
        controllers_to_use: List of controller modes ('passive', 'balanced').

    Returns:
        X: Feature matrix of shape (n_windows, n_features)
        y: Target label vector of shape (n_windows,) with integers {0, 1, 2}
        groups: Scenario IDs vector of shape (n_windows,) for group-based splitting
        meta: List of metadata dictionaries for each window
    """
    params = vehicle_params or VehicleParameters()
    model = QuarterCarModel(params)
    controllers = build_nominal_controllers(params)
    extractor = SensorFeatureExtractor(sampling_rate_hz=1.0 / dt)

    ctrl_modes = controllers_to_use or ["passive", "balanced"]

    window_samples = int(round(window_s / dt))
    stride_samples = int(round(stride_s / dt))

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    for sc in scenarios:
        road_fn = sc.road_profile.get_time_callable(sc.speed_mps)
        target_cls = LABEL_MAP[sc.road_type]

        for ctrl_mode in ctrl_modes:
            if ctrl_mode == "passive":
                ctrl_fn = None
            else:
                lqr_ctrl = controllers[ctrl_mode]
                ctrl_fn = lambda t, x, zr: (lqr_ctrl.compute_force(x), lqr_ctrl.compute_force(x), {})

            # Run full physical simulation
            sim_res = model.simulate(
                road_func=road_fn,
                controller_func=ctrl_fn,
                duration=sc.duration_s,
                dt=dt
            )

            a_s = sim_res.zs_ddot
            a_u = sim_res.zu_ddot
            travel = sim_res.suspension_travel
            v_rel = sim_res.zs_dot - sim_res.zu_dot
            n_pts = len(a_s)

            # Slide rolling window
            for start_idx in range(0, n_pts - window_samples + 1, stride_samples):
                end_idx = start_idx + window_samples

                win_as = a_s[start_idx:end_idx]
                win_au = a_u[start_idx:end_idx]
                win_tr = travel[start_idx:end_idx]
                win_vr = v_rel[start_idx:end_idx]

                feats = extractor.extract_features(win_as, win_au, win_tr, win_vr)

                X_list.append(feats)
                y_list.append(target_cls)
                groups_list.append(sc.scenario_id)
                meta_list.append({
                    "scenario_id": sc.scenario_id,
                    "road_type": sc.road_type,
                    "speed_kmh": sc.speed_kmh,
                    "ctrl_mode": ctrl_mode,
                    "t_start": start_idx * dt,
                    "t_end": end_idx * dt
                })

    X = np.array(X_list, dtype=np.float64) if X_list else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_list, dtype=np.int64) if y_list else np.empty((0,), dtype=np.int64)
    groups = np.array(groups_list, dtype=np.int64) if groups_list else np.empty((0,), dtype=np.int64)

    return X, y, groups, meta_list
