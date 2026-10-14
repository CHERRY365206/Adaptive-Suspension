"""Unit tests for Disturbance Observer-based Sliding Mode Control (DOSMC)."""

import pytest
import numpy as np

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel
from controllers.dosmc import DOSMCConfig, DOSMCController
from roads.reference_roads import ReferenceProfile1


def test_dosmc_config_validation():
    """Verify that DOSMCConfig validates parameters."""
    with pytest.raises(ValueError, match="Sliding slope S"):
        DOSMCConfig(S=-1.0).validate()

    with pytest.raises(ValueError, match="Sliding gain k"):
        DOSMCConfig(k=0.0).validate()

    with pytest.raises(ValueError, match="Observer gain W"):
        DOSMCConfig(W=-5.0).validate()

    with pytest.raises(ValueError, match="Boundary layer epsilon"):
        DOSMCConfig(epsilon=0.0).validate()


def test_dosmc_force_computation_and_saturation():
    """Verify force calculation, smooth sat function, and hard actuator limits."""
    cfg = DOSMCConfig(
        S=4.0,
        k=10.0,
        W=100.0,
        k_st=2.0,
        epsilon=0.01,
        max_force=1500.0,
        min_force=-1500.0,
        slew_rate_limit=18000.0
    )
    ctrl = DOSMCController(cfg)

    # Sprung mass state [zs = 0.05, zs_dot = 0.5, zu = 0.0, zu_dot = 0.0]
    x_test = np.array([0.05, 0.5, 0.0, 0.0])
    u_req, info = ctrl.compute_force(x_test, current_road_elev=0.0)

    # Check info fields
    assert "sigma" in info
    assert "e_hat" in info
    assert "d_hat" in info
    assert "u_eq" in info
    assert "u_n" in info
    assert "u_s" in info
    assert "u_applied" in info

    # Applied force must respect saturation limits
    assert -1500.0 <= info["u_applied"] <= 1500.0
    # Sliding variable sigma = 4.0 * 0.05 + 0.5 = 0.70
    assert abs(info["sigma"] - 0.70) < 1e-6


def test_dosmc_closed_loop_reduction_on_reference_road():
    """Verify that DOSMC achieves massive acceleration reduction over passive suspension."""
    # Paper benchmark parameters: ms = 290 kg, mu = 59 kg, ks = 14500, cs = 1385.4, kt = 190000
    params = VehicleParameters(
        ms=290.0,
        mu=59.0,
        ks=14500.0,
        cs=1385.4,
        kt=190000.0,
        max_travel=0.12
    )
    qc = QuarterCarModel(params)
    prof1 = ReferenceProfile1(duration_s=6.0)
    road_fn = prof1.get_time_callable()

    # 1. Passive simulation
    res_passive = qc.simulate(road_func=road_fn, duration=6.0, dt=0.002)

    # 2. DOSMC with u_s simulation
    dosmc_cfg = DOSMCConfig(
        S=4.0,
        k=10.0,
        W=100.0,
        k_st=0.5,
        epsilon=0.05,
        max_force=2000.0,
        min_force=-2000.0,
        slew_rate_limit=50000.0
    )
    dosmc_ctrl = DOSMCController(dosmc_cfg, vehicle_params=params)
    res_dosmc = qc.simulate(
        road_func=road_fn,
        controller_func=dosmc_ctrl,
        duration=6.0,
        dt=0.002
    )

    # DOSMC should drastically reduce RMS body acceleration compared to passive
    assert res_dosmc.rms_body_accel < res_passive.rms_body_accel * 0.35, (
        f"DOSMC RMS accel ({res_dosmc.rms_body_accel:.4f}) should be vastly lower than "
        f"Passive ({res_passive.rms_body_accel:.4f})"
    )
