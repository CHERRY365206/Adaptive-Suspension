"""Adaptive controller selector with switching safeguards, dwell time, and bumpless transfer.

Maps road classifications to LQR controller modes, enforces anti-chattering hysteresis,
and eliminates actuator force spikes during transitions.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
from collections import deque
import numpy as np

from controllers.lqr import LQRController

DEFAULT_ROAD_TO_CONTROLLER_MAP: Dict[str, str] = {
    "smooth": "comfort",
    "bumpy": "balanced",
    "rough": "road_holding"
}


class AdaptiveControllerSelector:
    """Intelligent controller selector mapping road conditions to specialized LQR controllers.

    Safeguards:
    1. Minimum Dwell Time: Prevents rapid controller chattering by enforcing a minimum duration
       before a new controller transition is permitted.
    2. Confidence Thresholding: Ignores uncertain classifications, maintaining stability.
    3. Bumpless Transfer: Cross-fades actuator control signals over a transition duration (tau)
       to eliminate step force discontinuities on gain switching.
    4. Fallback Controller: Safe default ('balanced') during initial startup or signal degradation.
    """

    def __init__(
        self,
        controllers: Dict[str, LQRController],
        actuator: Optional[Any] = None,
        road_to_controller_map: Optional[Dict[str, str]] = None,
        dwell_time_s: float = 0.3,
        confidence_threshold: float = 0.40,
        bumpless_transfer_tau_s: float = 0.05,
        default_controller: str = "balanced"
    ) -> None:
        self.controllers = controllers
        if actuator is None:
            from models.actuator import ActuatorModel
            self.actuator = ActuatorModel()
        else:
            self.actuator = actuator
        self.mapping = road_to_controller_map or DEFAULT_ROAD_TO_CONTROLLER_MAP
        self.dwell_time = dwell_time_s
        self.confidence_threshold = confidence_threshold
        self.tau = bumpless_transfer_tau_s
        self.default_controller = default_controller

        # Validate that all mapped controllers exist
        for r_cls, c_name in self.mapping.items():
            if c_name not in self.controllers:
                raise ValueError(f"Controller '{c_name}' mapped to road '{r_cls}' not found in provided controllers.")

        # State tracking
        self.current_controller_name = self.default_controller
        self.previous_controller_name = self.default_controller
        self.last_switch_time = -1e6
        self.transition_start_time = -1.0
        self.switch_count = 0

        # Detailed event log
        self.transition_log = []

    def reset(self) -> None:
        """Reset internal selector and transition state."""
        self.current_controller_name = self.default_controller
        self.previous_controller_name = self.default_controller
        self.last_switch_time = -1e6
        self.transition_start_time = -1.0
        self.switch_count = 0
        self.transition_log.clear()
        if self.actuator is not None:
            self.actuator.reset()

    def select_controller(
        self,
        t: float,
        predicted_class: str,
        confidence: float
    ) -> str:
        """Determine active controller taking dwell time and confidence into account."""
        desired_controller = self.mapping.get(predicted_class, self.default_controller)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # Maintain current controller if prediction is uncertain
            return self.current_controller_name

        # Check dwell time hysteresis
        time_since_switch = t - self.last_switch_time
        if desired_controller != self.current_controller_name and time_since_switch >= self.dwell_time:
            # Execute controller transition
            self.previous_controller_name = self.current_controller_name
            self.current_controller_name = desired_controller
            self.last_switch_time = t
            self.transition_start_time = t
            self.switch_count += 1
            self.transition_log.append({
                "time": t,
                "from": self.previous_controller_name,
                "to": self.current_controller_name,
                "cause_class": predicted_class,
                "confidence": confidence
            })

        return self.current_controller_name

    def compute_force(
        self,
        t: float,
        x: np.ndarray,
        predicted_class: str,
        confidence: float,
        dt: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute the constrained actuator force with bumpless transfer between controllers.

        Args:
            t: Current time [s]
            x: State vector [zs, zs_dot, zu, zu_dot]
            predicted_class: Classified road condition
            confidence: Classifier confidence [0.0 - 1.0]
            dt: Time step [s]

        Returns:
            Tuple of (requested_force, applied_force, diagnostics_dict)
        """
        active_ctrl_name = self.select_controller(t, predicted_class, confidence)

        # Raw requested force from target controller
        u_target = self.controllers[active_ctrl_name].compute_force(x)

        # Apply bumpless cross-fade transfer if in active transition
        if self.transition_start_time >= 0.0 and (t - self.transition_start_time) < self.tau:
            elapsed = t - self.transition_start_time
            alpha = elapsed / self.tau  # 0 to 1
            u_prev = self.controllers[self.previous_controller_name].compute_force(x)
            u_requested = alpha * u_target + (1.0 - alpha) * u_prev
        else:
            u_requested = u_target

        # Apply physical actuator saturation and slew rate limits
        u_req, u_applied, is_sat = self.actuator.apply_limits(u_requested, dt)

        info = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "active_controller": active_ctrl_name,
            "switch_count": self.switch_count,
            "is_saturated": is_sat
        }

        return u_req, u_applied, info


class OracleAdaptiveSelector(AdaptiveControllerSelector):
    """Idealized Oracle controller selector using ground-truth road labels.

    NOTE: This is strictly an experimental reference benchmark, NOT an implementable real-time AI system.
    """

    def __init__(
        self,
        controllers: Dict[str, LQRController],
        actuator: Optional[Any] = None,
        road_to_controller_map: Optional[Dict[str, str]] = None,
        dwell_time_s: float = 0.3,
        default_controller: str = "balanced"
    ) -> None:
        super().__init__(
            controllers=controllers,
            actuator=actuator,
            road_to_controller_map=road_to_controller_map,
            dwell_time_s=dwell_time_s,
            confidence_threshold=0.0,
            default_controller=default_controller
        )

    def compute_force_oracle(
        self,
        t: float,
        x: np.ndarray,
        ground_truth_class: str,
        dt: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute force using exact ground-truth class with confidence=1.0."""
        return self.compute_force(
            t=t,
            x=x,
            predicted_class=ground_truth_class,
            confidence=1.0,
            dt=dt
        )
