"""Unit tests for metric calculation and statistical aggregation."""

import pytest
import numpy as np

from experiments.metrics import compute_simulation_metrics
from experiments.statistical_analysis import compute_percentage_improvement, aggregate_experiment_results
from models.quarter_car import SimulationResult


def test_percentage_improvement_formula():
    """Verify reduction percentage calculations."""
    # 50% reduction in acceleration: 2.0 -> 1.0
    assert compute_percentage_improvement(2.0, 1.0) == pytest.approx(50.0)

    # Negative improvement (degradation): 1.0 -> 1.5
    assert compute_percentage_improvement(1.0, 1.5) == pytest.approx(-50.0)

    # Zero baseline handling
    assert compute_percentage_improvement(0.0, 1.0) == 0.0


def test_simulation_metrics_on_known_signals():
    """Verify that RMS calculations match theoretical values on pure sine waves."""
    t = np.linspace(0, 1.0, 1000)
    # Sine wave with amplitude A has RMS = A / sqrt(2)
    A = 2.0
    sine_wave = A * np.sin(2.0 * np.pi * 5.0 * t)

    dummy_res = SimulationResult(
        t=t,
        zs=sine_wave,
        zs_dot=np.zeros_like(t),
        zs_ddot=sine_wave,
        zu=np.zeros_like(t),
        zu_dot=np.zeros_like(t),
        zu_ddot=np.zeros_like(t),
        suspension_travel=sine_wave,
        tyre_deflection=sine_wave,
        zr=np.zeros_like(t),
        u_requested=np.zeros_like(t),
        u_applied=np.zeros_like(t),
        dynamic_tyre_load=sine_wave * 100.0,
        total_tyre_load=np.full_like(t, 4000.0),
        loss_of_contact=np.zeros(len(t), dtype=bool),
        travel_exceeded=np.zeros(len(t), dtype=bool),
        power_mechanical=np.zeros_like(t),
        power_electrical=np.zeros_like(t)
    )

    metrics = compute_simulation_metrics(dummy_res)

    expected_rms = A / np.sqrt(2.0)
    assert metrics["rms_body_accel_mps2"] == pytest.approx(expected_rms, rel=1e-3)
    assert metrics["peak_body_accel_mps2"] == pytest.approx(A, rel=1e-3)
    assert metrics["contact_loss_pct"] == 0.0
    assert metrics["saturation_pct"] == 0.0
