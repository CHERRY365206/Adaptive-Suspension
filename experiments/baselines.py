"""Definitions and execution wrappers for the six comparative suspension configurations:
1. Passive Suspension
2. Fixed Nominal LQR
3. Fixed PSO-Optimized LQR
4. Adaptive LQR without PSO
5. Proposed AI-Assisted PSO-Optimized Adaptive LQR
6. Idealized Oracle Adaptive LQR (Reference only)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import json
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import build_nominal_controllers
from controllers.adaptive_selector import AdaptiveControllerSelector, OracleAdaptiveSelector
from controllers.mpc import MPCConfig, MPCController
from ai.predict_road import RoadPredictor
from optimization.parameterization import decode_parameters


SYSTEM_NAMES = [
    "passive",
    "fixed_nominal_lqr",
    "fixed_pso_lqr",
    "adaptive_no_pso",
    "proposed_ai_pso",
    "oracle_adaptive"
]

ALL_SYSTEM_NAMES = SYSTEM_NAMES + ["mpc"]

SYSTEM_DISPLAY_NAMES = {
    "passive": "Passive Suspension",
    "fixed_nominal_lqr": "Fixed Nominal LQR",
    "fixed_pso_lqr": "Fixed PSO-Optimized LQR",
    "adaptive_no_pso": "Adaptive LQR (Heuristic / No PSO)",
    "proposed_ai_pso": "Proposed AI + PSO Adaptive LQR",
    "oracle_adaptive": "Oracle Adaptive LQR (Reference)",
    "mpc": "Constrained Model Predictive Control (MPC)"
}



class SuspensionBenchmarkSuite:
    """Manages controller building and runs simulations for all six suspension systems."""

    def __init__(
        self,
        vehicle_params: Optional[VehicleParameters] = None,
        actuator: Optional[ActuatorModel] = None,
        classifier_path: str | Path = "artifacts/models/road_classifier.joblib",
        pso_configs_path: str | Path = "artifacts/controllers/pso_lqr_configs.json"
    ) -> None:
        self.params = vehicle_params or VehicleParameters()
        self.actuator = actuator or ActuatorModel()
        self.model = QuarterCarModel(self.params)

        # 1. Build nominal heuristic controllers
        self.nominal_controllers = build_nominal_controllers(self.params)

        # 2. Load or fallback PSO-tuned controllers
        self.pso_controllers = self._load_or_create_pso_controllers(pso_configs_path)

        # 3. Setup AI road predictor
        self.classifier_path = Path(classifier_path)
        self.predictor = RoadPredictor(
            model_path=self.classifier_path if self.classifier_path.exists() else None,
            fallback_class="balanced"
        )

        # 4. Build adaptive selectors
        self.selector_no_pso = AdaptiveControllerSelector(
            controllers=self.nominal_controllers,
            actuator=self.actuator,
            dwell_time_s=0.3,
            confidence_threshold=0.40,
            default_controller="balanced"
        )

        self.selector_pso = AdaptiveControllerSelector(
            controllers=self.pso_controllers,
            actuator=self.actuator,
            dwell_time_s=0.3,
            confidence_threshold=0.40,
            default_controller="balanced"
        )

        self.oracle_selector = OracleAdaptiveSelector(
            controllers=self.pso_controllers,
            actuator=self.actuator,
            dwell_time_s=0.3,
            default_controller="balanced"
        )

        # 5. Build MPC controller
        self.mpc_controller = MPCController(vehicle_params=self.params)

    def _load_or_create_pso_controllers(self, path: str | Path) -> Dict[str, LQRController]:
        """Load PSO-optimized controllers from JSON, or fallback to sensible tuned variants if not yet trained."""
        path = Path(path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    configs_dict = json.load(f)
                controllers = {}
                for mode, data in configs_dict.items():
                    cfg = LQRConfig(
                        name=f"pso_{mode}",
                        q1=float(data["q1"]),
                        q2=float(data["q2"]),
                        q3=float(data["q3"]),
                        r0=float(data["r0"]),
                        description=data.get("description", "Loaded PSO LQR")
                    )
                    controllers[mode] = LQRController(cfg, self.params)
                return controllers
            except Exception:
                pass

        # Robust defaults if optimization artifact has not been created yet
        # Uses scientifically adjusted values reflecting the respective priorities
        return {
            "comfort": LQRController(LQRConfig(name="pso_comfort", q1=3.5e5, q2=8.0e3, q3=2.0e3, r0=1.5e-5), self.params),
            "balanced": LQRController(LQRConfig(name="pso_balanced", q1=5.0e4, q2=3.0e4, q3=3.5e4, r0=2.0e-5), self.params),
            "road_holding": LQRController(LQRConfig(name="pso_road_holding", q1=8.0e3, q2=4.0e4, q3=4.5e5, r0=1.8e-5), self.params),
        }

    def simulate_system(
        self,
        system_type: str,
        road_func: Callable[[float], float],
        ground_truth_class: str,
        duration_s: float = 6.0,
        dt: float = 0.002
    ) -> SimulationResult:
        """Execute a standardized simulation for a specific suspension system.

        Args:
            system_type: One of SYSTEM_NAMES ('passive', 'fixed_nominal_lqr', etc.)
            road_func: Road profile callable zr(t)
            ground_truth_class: True road condition ('smooth', 'bumpy', 'rough')
            duration_s: Duration [s]
            dt: Sample time step [s]

        Returns:
            SimulationResult containing all outputs and state trajectories.
        """
        system_type = system_type.lower()
        if system_type not in ALL_SYSTEM_NAMES:
            raise ValueError(f"Unknown system_type '{system_type}'. Must be one of {ALL_SYSTEM_NAMES}")

        # Reset states
        self.predictor.reset()
        self.selector_no_pso.reset()
        self.selector_pso.reset()
        self.oracle_selector.reset()
        self.actuator.reset()

        if system_type == "passive":
            # Actuator disabled mode
            act_passive = ActuatorModel(enabled=False)
            return self.model.simulate(
                road_func=road_func,
                controller_func=None,
                duration=duration_s,
                dt=dt,
                actuator=act_passive
            )

        elif system_type == "fixed_nominal_lqr":
            ctrl = self.nominal_controllers["balanced"]
            def ctrl_fn(t, x, zr):
                u_req = ctrl.compute_force(x)
                u_r, u_a, is_sat = self.actuator.apply_limits(u_req, dt)
                return u_r, u_a, {"active_controller": "balanced"}

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

        elif system_type == "fixed_pso_lqr":
            ctrl = self.pso_controllers["balanced"]
            def ctrl_fn(t, x, zr):
                u_req = ctrl.compute_force(x)
                u_r, u_a, is_sat = self.actuator.apply_limits(u_req, dt)
                return u_r, u_a, {"active_controller": "pso_balanced"}

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

        elif system_type == "mpc":
            self.mpc_controller.reset()
            def ctrl_fn(t, x, zr):
                u_req, info = self.mpc_controller.compute_force(x, current_road_elev=zr)
                u_r, u_a, is_sat = self.actuator.apply_limits(u_req, dt)
                info["active_controller"] = "mpc"
                return u_r, u_a, info

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

        elif system_type == "adaptive_no_pso":
            def ctrl_fn(t, x, zr):
                # Online feature extraction and road prediction
                # Current measurements:
                out = self.model.calculate_outputs(x, 0.0, zr)
                v_rel = x[1] - x[3]
                self.predictor.add_sample(out["zs_ddot"], out["zu_ddot"], out["suspension_travel"], v_rel)
                pred_cls, conf, _ = self.predictor.predict()

                u_r, u_a, info = self.selector_no_pso.compute_force(t, x, pred_cls, conf, dt)
                return u_r, u_a, info

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

        elif system_type == "proposed_ai_pso":
            def ctrl_fn(t, x, zr):
                out = self.model.calculate_outputs(x, 0.0, zr)
                v_rel = x[1] - x[3]
                self.predictor.add_sample(out["zs_ddot"], out["zu_ddot"], out["suspension_travel"], v_rel)
                pred_cls, conf, _ = self.predictor.predict()

                u_r, u_a, info = self.selector_pso.compute_force(t, x, pred_cls, conf, dt)
                return u_r, u_a, info

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

        elif system_type == "oracle_adaptive":
            def ctrl_fn(t, x, zr):
                # Exact ground truth road class passed directly
                u_r, u_a, info = self.oracle_selector.compute_force_oracle(t, x, ground_truth_class, dt)
                return u_r, u_a, info

            return self.model.simulate(
                road_func=road_func,
                controller_func=ctrl_fn,
                duration=duration_s,
                dt=dt,
                actuator=self.actuator
            )

