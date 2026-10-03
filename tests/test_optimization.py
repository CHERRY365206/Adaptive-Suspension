"""Unit tests for Particle Swarm Optimization and multi-objective LQR tuning."""

import pytest
import numpy as np

from optimization.parameterization import (
    decode_parameters,
    encode_parameters,
    DEFAULT_LOWER_BOUNDS,
    DEFAULT_UPPER_BOUNDS
)
from optimization.objective import evaluate_lqr_candidate
from optimization.pso import ParticleSwarmOptimizer
from roads.road_dataset import create_scenario
from models.vehicle_parameters import VehicleParameters


def test_parameterization_encode_decode_roundtrip():
    """Verify that encode and decode operations are mutually inverse."""
    original_p = np.array([4.5, 3.2, 5.1, -4.3])
    cfg = decode_parameters(original_p, name="roundtrip_ctrl")

    assert cfg.q1 == pytest.approx(10.0 ** 4.5, rel=1e-5)
    assert cfg.q2 == pytest.approx(10.0 ** 3.2, rel=1e-5)
    assert cfg.q3 == pytest.approx(10.0 ** 5.1, rel=1e-5)
    assert cfg.r0 == pytest.approx(10.0 ** -4.3, rel=1e-5)

    recovered_p = encode_parameters(cfg)
    np.testing.assert_allclose(original_p, recovered_p, atol=1e-5)


def test_pso_sphere_function_minimization():
    """Verify that PSO accurately minimizes a standard benchmark convex sphere function."""
    def sphere(p: np.ndarray) -> float:
        return float(np.sum((p - 2.0)**2))

    lb = np.array([-5.0, -5.0])
    ub = np.array([5.0, 5.0])

    pso = ParticleSwarmOptimizer(
        cost_function=sphere,
        lower_bounds=lb,
        upper_bounds=ub,
        swarm_size=20,
        max_iterations=25,
        seed=123
    )

    best_pos, best_cost, history = pso.optimize()

    # Optimal minimum is at [2.0, 2.0] with cost 0.0
    assert best_cost < 0.05
    np.testing.assert_allclose(best_pos, np.array([2.0, 2.0]), atol=0.25)
    assert len(history) == 26  # iter 0 to 25


def test_pso_seed_reproducibility():
    """Verify that identical random seeds yield identical PSO trajectories and best costs."""
    def dummy_cost(p: np.ndarray) -> float:
        return float(np.sum(p**2 + 0.1 * np.sin(5.0 * p)))

    lb = np.array([-2.0, -2.0])
    ub = np.array([2.0, 2.0])

    pso1 = ParticleSwarmOptimizer(cost_function=dummy_cost, lower_bounds=lb, upper_bounds=ub, swarm_size=10, max_iterations=10, seed=999)
    pso2 = ParticleSwarmOptimizer(cost_function=dummy_cost, lower_bounds=lb, upper_bounds=ub, swarm_size=10, max_iterations=10, seed=999)

    pos1, cost1, _ = pso1.optimize()
    pos2, cost2, _ = pso2.optimize()

    assert cost1 == cost2
    np.testing.assert_allclose(pos1, pos2, atol=1e-12)


def test_objective_function_evaluation():
    """Verify that multi-objective evaluation calculates finite costs on realistic scenarios."""
    scenarios = [
        create_scenario(scenario_id=1, road_type="smooth", speed_kmh=50.0, split="pso", duration_s=1.0, seed=10)
    ]
    p_nominal = np.array([4.5, 3.5, 4.0, -4.5])

    res = evaluate_lqr_candidate(
        particle_p=p_nominal,
        target_mode="balanced",
        training_scenarios=scenarios,
        dt=0.01
    )

    assert "cost" in res
    assert np.isfinite(res["cost"])
    assert res["cost"] < 1000.0  # Feasible candidate
    assert "metrics" in res
    assert res["metrics"]["mean_rms_accel"] > 0.0
