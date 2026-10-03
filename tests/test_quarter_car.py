"""Unit tests for the 2-DOF quarter-car vehicle dynamics model."""

import pytest
import numpy as np
from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel, SimulationResult


def test_vehicle_parameters_validation():
    """Verify physical parameter constraints and error handling."""
    # Valid default construction
    params = VehicleParameters()
    assert params.ms == 450.0
    assert params.mu == 45.0
    assert params.ks == 28000.0
    assert params.total_mass == 495.0
    assert params.static_tyre_load == pytest.approx(495.0 * 9.81, rel=1e-5)

    # Invalid negative sprung mass
    with pytest.raises(ValueError, match="Sprung mass ms must be strictly positive"):
        VehicleParameters(ms=-100.0)

    # Invalid unsprung mass >= sprung mass
    with pytest.raises(ValueError, match="must be smaller than sprung mass"):
        VehicleParameters(ms=40.0, mu=50.0)

    # Negative stiffness
    with pytest.raises(ValueError, match="stiffness"):
        VehicleParameters(ks=-5000.0)


def test_state_space_vs_direct_equations():
    """Verify continuous state-space matrices A, B, E match analytical ODEs exactly."""
    model = QuarterCarModel()

    # Test at zero state
    ok, err = model.verify_state_space(np.zeros(4), u=0.0, zr=0.0)
    assert ok
    assert err < 1e-12

    # Test across multiple random states, forces, and road displacements
    rng = np.random.default_rng(seed=123)
    for _ in range(20):
        x = rng.uniform(-0.1, 0.1, size=4)
        u = float(rng.uniform(-1000.0, 1000.0))
        zr = float(rng.uniform(-0.05, 0.05))

        ok, err = model.verify_state_space(x, u=u, zr=zr)
        assert ok, f"Mismatch between state space and direct equations: err={err}"
        assert err < 1e-10


def test_flat_road_equilibrium():
    """On a flat road (zr=0) with zero initial condition and zero control, system remains at rest."""
    model = QuarterCarModel()
    zero_state = np.zeros(4)
    x_dot = model.state_derivatives(t=0.0, x=zero_state, u=0.0, zr=0.0)
    np.testing.assert_allclose(x_dot, np.zeros(4), atol=1e-12)

    # Run short simulation
    res = model.simulate(
        road_func=lambda t: 0.0,
        controller_func=None,
        duration=1.0,
        dt=0.01
    )
    np.testing.assert_allclose(res.zs, np.zeros_like(res.zs), atol=1e-6)
    np.testing.assert_allclose(res.zu, np.zeros_like(res.zu), atol=1e-6)
    assert res.rms_body_accel < 1e-6


def test_actuator_force_sign_convention():
    """Verify that positive actuator force u pushes sprung mass UP and unsprung mass DOWN."""
    model = QuarterCarModel()
    zero_state = np.zeros(4)
    u_pos = 1000.0  # +1000 N

    outputs = model.calculate_outputs(zero_state, u_applied=u_pos, zr=0.0)
    # ms * zs_ddot = +u => zs_ddot = +1000 / 450 > 0
    assert outputs["zs_ddot"] > 0.0
    assert outputs["zs_ddot"] == pytest.approx(1000.0 / 450.0, rel=1e-5)

    # mu * zu_ddot = -u => zu_ddot = -1000 / 45 < 0
    assert outputs["zu_ddot"] < 0.0
    assert outputs["zu_ddot"] == pytest.approx(-1000.0 / 45.0, rel=1e-5)


def test_passive_decay_stability():
    """Verify that unforced system from non-zero initial condition naturally decays to zero (stable passive damper)."""
    model = QuarterCarModel()
    x0 = np.array([0.05, 0.0, 0.02, 0.0])  # Initial displacement

    res = model.simulate(
        road_func=lambda t: 0.0,
        controller_func=None,
        duration=4.0,
        dt=0.005,
        x0=x0
    )

    # Final state after 4 seconds should be substantially dissipated by damping
    assert abs(res.zs[-1]) < 0.002
    assert abs(res.zu[-1]) < 0.002
    assert res.rms_body_accel > 0.0
    assert np.all(np.isfinite(res.zs))
    assert np.all(np.isfinite(res.zu))
