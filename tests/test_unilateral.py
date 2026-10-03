"""Unit tests for unilateral tyre contact dynamics."""

import pytest
import numpy as np

from experiments.unilateral_contact_study import run_unilateral_contact_comparison
from models.quarter_car import QuarterCarModel
from models.vehicle_parameters import VehicleParameters


def test_unilateral_contact_non_negative_force():
    """Verify that unilateral tyre model cannot pull in tension (force >= 0 on separation)."""
    params = VehicleParameters()
    model = QuarterCarModel(params=params, unilateral_tyre=True)

    # State where wheel lifts above road: zu = +0.05, zr = 0.0 => tyre in tension if bilateral
    x_lift = np.array([0.02, 0.0, 0.05, 0.0])
    x_dot = model.state_derivatives(t=0.0, x=x_lift, u=0.0, zr=0.0)

    # In unilateral contact, f_tyre = max(0, kt*(zr - zu)) = 0.
    # Therefore, wheel is only acted upon by gravity/suspension, NOT pulled downwards by the road!
    ms, mu, ks, kt = params.ms, params.mu, params.ks, params.kt
    # suspension force = ks*(zs - zu) = 28000*(0.02 - 0.05) = -840 N
    # zu_ddot = (-840) / 45 = -18.67 m/s^2
    expected_zu_ddot = (ks * (0.02 - 0.05)) / mu
    assert x_dot[3] == pytest.approx(expected_zu_ddot, rel=1e-5)


def test_unilateral_study_execution(tmp_path):
    """Verify end-to-end execution of unilateral contact comparative study."""
    fig_path = tmp_path / "test_uni.png"
    json_path = tmp_path / "test_uni.json"

    metrics = run_unilateral_contact_comparison(
        bump_height=0.06,
        bump_width=0.8,
        vehicle_speed_kmh=50.0,
        duration_s=1.5,
        dt=0.005,
        output_fig_path=fig_path,
        output_json_path=json_path
    )

    assert fig_path.exists()
    assert json_path.exists()
    assert "linear_model" in metrics
    assert "unilateral_model" in metrics
    assert metrics["linear_model"]["rms_body_accel_mps2"] > 0.0
    assert metrics["unilateral_model"]["rms_body_accel_mps2"] > 0.0
