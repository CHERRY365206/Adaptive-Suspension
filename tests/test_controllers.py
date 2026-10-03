"""Unit tests for adaptive controller selection, dwell-time safeguards, and bumpless transfer."""

import pytest
import numpy as np

from controllers.controller_registry import build_nominal_controllers
from controllers.adaptive_selector import AdaptiveControllerSelector, OracleAdaptiveSelector
from models.actuator import ActuatorModel


def test_adaptive_selector_mapping_and_switching():
    """Verify that road classification changes trigger correct controller transitions."""
    controllers = build_nominal_controllers()
    selector = AdaptiveControllerSelector(
        controllers=controllers,
        dwell_time_s=0.2,
        confidence_threshold=0.5
    )

    # Initial controller should be default ('balanced')
    assert selector.current_controller_name == "balanced"

    # Step at t=0.0 with 'smooth' and high confidence
    c1 = selector.select_controller(t=0.0, predicted_class="smooth", confidence=0.8)
    assert c1 == "comfort"
    assert selector.switch_count == 1

    # Immediate step at t=0.05 (< dwell time 0.2s) with 'rough' -> must remain 'comfort'
    c2 = selector.select_controller(t=0.05, predicted_class="rough", confidence=0.9)
    assert c2 == "comfort"
    assert selector.switch_count == 1  # No switch allowed due to dwell time

    # Step after dwell time: t=0.25 (0.25 - 0.0 > 0.2) with 'rough' -> switches to 'road_holding'
    c3 = selector.select_controller(t=0.25, predicted_class="rough", confidence=0.9)
    assert c3 == "road_holding"
    assert selector.switch_count == 2


def test_confidence_threshold_safeguard():
    """Verify that predictions below the confidence threshold are ignored."""
    controllers = build_nominal_controllers()
    selector = AdaptiveControllerSelector(
        controllers=controllers,
        dwell_time_s=0.1,
        confidence_threshold=0.6,
        default_controller="balanced"
    )

    # Low-confidence prediction (0.45 < 0.6)
    active = selector.select_controller(t=0.5, predicted_class="smooth", confidence=0.45)
    assert active == "balanced"
    assert selector.switch_count == 0


def test_bumpless_transfer_force_continuity():
    """Verify that force transitions between controllers are smoothed by bumpless transfer."""
    controllers = build_nominal_controllers()
    actuator = ActuatorModel(max_force=3000.0, min_force=-3000.0, slew_rate_limit=1e6)
    selector = AdaptiveControllerSelector(
        controllers=controllers,
        actuator=actuator,
        dwell_time_s=0.1,
        bumpless_transfer_tau_s=0.04
    )

    x = np.array([0.02, 0.1, -0.01, 0.05])
    u_comf = controllers["comfort"].compute_force(x)
    u_hold = controllers["road_holding"].compute_force(x)

    # Force should start with balanced
    u_req0, u_app0, _ = selector.compute_force(t=0.0, x=x, predicted_class="comfort", confidence=1.0, dt=0.002)

    # Trigger switch at t=0.2 to road_holding
    # Exactly at transition start, alpha = 0 -> output should equal previous controller's force
    u_req_trans0, _, _ = selector.compute_force(t=0.2, x=x, predicted_class="rough", confidence=1.0, dt=0.002)
    # Midway through transition (t=0.22, half of 0.04) -> intermediate force
    u_req_mid, _, _ = selector.compute_force(t=0.22, x=x, predicted_class="rough", confidence=1.0, dt=0.002)
    # After transition (t=0.26) -> full target force
    u_req_end, _, _ = selector.compute_force(t=0.26, x=x, predicted_class="rough", confidence=1.0, dt=0.002)

    assert abs(u_req_end - u_hold) < 1e-4
    # Mid-transition force must lie between start and target
    assert (u_req_trans0 <= u_req_mid <= u_req_end) or (u_req_end <= u_req_mid <= u_req_trans0)
