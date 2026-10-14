"""Unit tests for Model Predictive Control (MPC) implementation."""

import pytest
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel
from controllers.mpc import MPCConfig, MPCController


def test_mpc_config_validation():
    """Verify that MPCConfig enforces positive parameters and valid bounds."""
    # Invalid horizon
    with pytest.raises(ValueError, match="prediction_horizon"):
        MPCConfig(prediction_horizon=1).validate()

    # Invalid dt
    with pytest.raises(ValueError, match="dt"):
        MPCConfig(dt=-0.01).validate()

    # Invalid weights
    with pytest.raises(ValueError, match="r0"):
        MPCConfig(r0=0.0).validate()

    # Invalid force limits
    with pytest.raises(ValueError, match="Invalid force limits"):
        MPCConfig(min_force=100.0, max_force=-100.0).validate()


def test_mpc_matrix_dimensions():
    """Verify dimensions and properties of condensed prediction matrices."""
    cfg = MPCConfig(prediction_horizon=10, dt=0.005)
    mpc = MPCController(cfg)

    # Discretized system
    assert mpc.A_d.shape == (4, 4)
    assert mpc.B_d.shape == (4, 1)
    assert mpc.E_d.shape == (4, 1)

    # Prediction matrices
    assert mpc.Phi_x.shape == (40, 4)
    assert mpc.Gamma_u.shape == (40, 10)
    assert mpc.Gamma_w.shape == (40, 10)
    assert mpc.H.shape == (10, 10)

    # Hessian symmetry and positive definiteness
    np.testing.assert_allclose(mpc.H, mpc.H.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(mpc.H)
    assert np.all(eigenvalues > 0.0), "Hessian must be strictly positive definite"


def test_mpc_force_and_slew_constraints():
    """Verify that computed force strictly respects saturation and slew-rate bounds."""
    max_f = 1500.0
    min_f = -1500.0
    slew = 10000.0
    dt = 0.002
    cfg = MPCConfig(
        prediction_horizon=8,
        dt=dt,
        max_force=max_f,
        min_force=min_f,
        slew_rate_limit=slew
    )
    mpc = MPCController(cfg)

    # Test under severe displacement perturbation
    x_severe = np.array([0.20, 2.0, -0.15, -3.0])  # Extreme perturbation
    u_cmd, info = mpc.compute_force(x_severe, current_road_elev=0.0)

    # Applied force must satisfy hard limits
    u_app = info["u_applied"]
    assert min_f <= u_app <= max_f
    max_step = slew * dt
    assert abs(u_app - 0.0) <= max_step + 1e-4

    # Second step
    u_cmd2, info2 = mpc.compute_force(x_severe, current_road_elev=0.0)
    u_app2 = info2["u_applied"]
    assert min_f <= u_app2 <= max_f
    assert abs(u_app2 - u_app) <= max_step + 1e-4


def test_mpc_closed_loop_simulation_stability():
    """Verify that MPC stabilizes the 2-DOF quarter car from an initial displacement perturbation."""
    cfg = MPCConfig(prediction_horizon=10, dt=0.002, q1=5e4, q2=2e4, q3=2e4, r0=2e-5)
    mpc = MPCController(cfg)
    qc = QuarterCarModel()

    def mpc_callback(t, x, zr):
        u_cmd, info = mpc.compute_force(x, current_road_elev=zr)
        return u_cmd, info["u_applied"], info

    # Zero road disturbance, non-zero initial body displacement
    x0 = np.array([0.05, 0.0, 0.0, 0.0])  # 5 cm initial body bump
    res = qc.simulate(
        road_func=lambda t: 0.0,
        controller_func=mpc_callback,
        duration=2.0,
        dt=0.002,
        x0=x0
    )

    # State must decay significantly by t = 2.0 s
    assert abs(res.zs[-1]) < 0.01  # Reduced by at least 80%
    assert abs(res.zs_dot[-1]) < 0.05
    # Controller diagnostics recorded
    info = mpc.get_info()
    assert info["solver_diagnostics"]["total_steps"] > 500
    assert info["solver_diagnostics"]["mean_solve_time_ms"] > 0.0
