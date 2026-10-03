"""Unit tests for active actuator model, saturation, and power calculations."""

import pytest
from models.actuator import ActuatorModel
from controllers.saturation import saturate_force, rate_limit_force


def test_saturate_force():
    """Verify standard clipping functionality."""
    assert saturate_force(500.0, -1000.0, 1000.0) == 500.0
    assert saturate_force(1500.0, -1000.0, 1000.0) == 1000.0
    assert saturate_force(-2500.0, -1000.0, 1000.0) == -1000.0


def test_rate_limit_force():
    """Verify maximum rate of force change per time step."""
    slew_rate = 10000.0  # 10,000 N/s
    dt = 0.01  # max change per step is 100 N

    # Desired change is +500 N, should be clamped to +100 N
    u_new = rate_limit_force(current_force=0.0, target_force=500.0, slew_rate_limit=slew_rate, dt=dt)
    assert u_new == pytest.approx(100.0, rel=1e-5)

    # Desired change is -50 N, which is within the 100 N limit
    u_new2 = rate_limit_force(current_force=100.0, target_force=50.0, slew_rate_limit=slew_rate, dt=dt)
    assert u_new2 == pytest.approx(50.0, rel=1e-5)


def test_actuator_limits_and_passive_mode():
    """Verify actuator saturation, slew rate handling, and passive disabling."""
    actuator = ActuatorModel(max_force=1500.0, min_force=-1500.0, slew_rate_limit=10000.0)

    # Commanded force 3000 N exceeds 1500 N, and exceeds slew rate (dt=0.01 -> max 100 N)
    u_req, u_app, is_sat = actuator.apply_limits(u_cmd=3000.0, dt=0.01)
    assert u_req == 3000.0
    assert u_app == 100.0
    assert is_sat is True

    # Passive mode: actuator disabled -> applied force must be strictly 0
    actuator.enabled = False
    u_req_p, u_app_p, is_sat_p = actuator.apply_limits(u_cmd=1200.0, dt=0.01)
    assert u_req_p == 1200.0
    assert u_app_p == 0.0
    assert is_sat_p is False


def test_actuator_power_and_regeneration():
    """Verify motoring power and regenerative absorption."""
    actuator = ActuatorModel(
        motoring_efficiency=0.80,
        regenerative_efficiency=0.50,
        standby_power=20.0,
        allow_regeneration=True
    )

    # Motoring: u_applied = +500 N, v_rel = +0.2 m/s -> P_mech = +100 W
    p_mech, p_elec = actuator.calculate_power(u_applied=500.0, v_rel=0.2)
    assert p_mech == pytest.approx(100.0, rel=1e-5)
    # P_elec = 100 / 0.8 + 20 = 145 W
    assert p_elec == pytest.approx(145.0, rel=1e-5)

    # Regenerating: u_applied = +500 N, v_rel = -0.2 m/s -> P_mech = -100 W
    p_mech_r, p_elec_r = actuator.calculate_power(u_applied=500.0, v_rel=-0.2)
    assert p_mech_r == pytest.approx(-100.0, rel=1e-5)
    # P_elec = -100 * 0.5 + 20 = -30 W (net energy returned to battery)
    assert p_elec_r == pytest.approx(-30.0, rel=1e-5)
