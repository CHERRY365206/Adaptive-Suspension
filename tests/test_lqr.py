"""Unit tests for continuous-time LQR controller and CARE solver."""

import pytest
import numpy as np

from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import get_nominal_configs, build_nominal_controllers
from models.quarter_car import QuarterCarModel


def test_lqr_gain_and_stability():
    """Verify CARE solver returns finite gain K and strictly stable closed-loop eigenvalues."""
    cfg = LQRConfig(name="test_ctrl", q1=1e4, q2=1e4, q3=1e4, r0=1e-5)
    ctrl = LQRController(cfg)

    # Check gain matrix shape
    assert ctrl.K.shape == (1, 4)
    assert np.all(np.isfinite(ctrl.K))

    # Check stability (all closed-loop eigenvalues must have negative real parts)
    assert ctrl.is_stable
    assert len(ctrl.eigenvalues) == 4
    for eig in ctrl.eigenvalues:
        assert np.real(eig) < -1e-4


def test_nominal_controllers_distinct_and_stable():
    """Verify all three nominal controllers (comfort, balanced, road_holding) are stable and have distinct gains."""
    controllers = build_nominal_controllers()

    assert set(controllers.keys()) == {"comfort", "balanced", "road_holding"}

    k_comfort = controllers["comfort"].K.flatten()
    k_balanced = controllers["balanced"].K.flatten()
    k_road_holding = controllers["road_holding"].K.flatten()

    for name, ctrl in controllers.items():
        assert ctrl.is_stable, f"{name} controller is not stable!"
        assert np.max(np.real(ctrl.eigenvalues)) < -0.01

    # Ensure the three designs result in distinctly different feedback gains
    assert not np.allclose(k_comfort, k_balanced, rtol=0.1)
    assert not np.allclose(k_comfort, k_road_holding, rtol=0.1)
    assert not np.allclose(k_balanced, k_road_holding, rtol=0.1)


def test_invalid_weights_rejection():
    """Verify that negative weights or zero control penalties raise ValueError."""
    # Negative q1
    with pytest.raises(ValueError, match="State weights must be non-negative"):
        cfg_neg = LQRConfig(name="bad", q1=-10.0, q2=10.0, q3=10.0, r0=1e-4)
        cfg_neg.validate()

    # Zero or negative r0
    with pytest.raises(ValueError, match="Control weight r0 must be strictly positive"):
        cfg_r0 = LQRConfig(name="bad_r0", q1=10.0, q2=10.0, q3=10.0, r0=0.0)
        cfg_r0.validate()


def test_lqr_counteracts_spring_force():
    """Verify that active LQR counteracts suspension spring force to reduce body vertical acceleration."""
    ctrl = build_nominal_controllers()["balanced"]
    model = QuarterCarModel()

    # Displaced state: spring is stretched by zs = +0.05 m
    x_disp = np.array([0.05, 0.0, 0.0, 0.0])
    u_active = ctrl.compute_force(x_disp)

    # Passive acceleration without active force: ms * zs_ddot = -ks * zs
    out_passive = model.calculate_outputs(x_disp, u_applied=0.0, zr=0.0)
    out_active = model.calculate_outputs(x_disp, u_applied=u_active, zr=0.0)

    # Active force counteracts the spring force, drastically reducing vertical acceleration
    assert abs(out_active["zs_ddot"]) < abs(out_passive["zs_ddot"])
    assert abs(out_active["zs_ddot"]) < 0.1  # Virtually eliminated acceleration
