"""Particle Swarm Optimization (PSO) and multi-objective controller tuning."""

from optimization.parameterization import (
    DEFAULT_LOWER_BOUNDS,
    DEFAULT_UPPER_BOUNDS,
    decode_parameters,
    encode_parameters,
    PARAM_NAMES,
)
from optimization.objective import (
    evaluate_lqr_candidate,
    OBJECTIVE_WEIGHTS,
    REF_BODY_ACCEL,
    REF_SUSP_TRAVEL,
    REF_TYRE_LOAD,
    REF_ACTUATOR_FORCE,
)
from optimization.pso import ParticleSwarmOptimizer
from optimization.optimize_lqr import run_pso_tuning, generate_pso_training_scenarios

__all__ = [
    "DEFAULT_LOWER_BOUNDS",
    "DEFAULT_UPPER_BOUNDS",
    "decode_parameters",
    "encode_parameters",
    "PARAM_NAMES",
    "evaluate_lqr_candidate",
    "OBJECTIVE_WEIGHTS",
    "REF_BODY_ACCEL",
    "REF_SUSP_TRAVEL",
    "REF_TYRE_LOAD",
    "REF_ACTUATOR_FORCE",
    "ParticleSwarmOptimizer",
    "run_pso_tuning",
    "generate_pso_training_scenarios",
]
