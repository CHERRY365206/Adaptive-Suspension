"""LQR parameterization and bounding for Particle Swarm Optimization.

Translates compact logarithmic particle coordinates into physically meaningful,
positive-definite state and control weighting matrices for the CARE solver.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any
import numpy as np

from controllers.lqr import LQRConfig


# Parameter order: [log10(q1), log10(q2), log10(q3), log10(r0)]
# q1: body vertical acceleration penalty
# q2: suspension travel penalty
# q3: tyre deflection penalty
# r0: actuator force penalty
DEFAULT_LOWER_BOUNDS = np.array([2.0, 1.0, 1.0, -6.0], dtype=np.float64)
DEFAULT_UPPER_BOUNDS = np.array([7.0, 6.0, 7.0, -2.0], dtype=np.float64)
PARAM_NAMES: List[str] = ["log10_q1", "log10_q2", "log10_q3", "log10_r0"]


def decode_parameters(p: np.ndarray, name: str = "pso_controller") -> LQRConfig:
    """Decode 4D log-space particle position into an LQRConfig instance.

    Args:
        p: Array of 4 floats [p1, p2, p3, pr]
        name: Name tag for the resulting LQRConfig

    Returns:
        Validated LQRConfig instance.
    """
    p = np.asarray(p, dtype=np.float64).flatten()
    q1 = float(10.0 ** p[0])
    q2 = float(10.0 ** p[1])
    q3 = float(10.0 ** p[2])
    r0 = float(10.0 ** p[3])

    return LQRConfig(
        name=name,
        q1=q1,
        q2=q2,
        q3=q3,
        r0=r0,
        description=f"PSO-tuned configuration: q1=10^{p[0]:.2f}, q2=10^{p[1]:.2f}, q3=10^{p[2]:.2f}, r0=10^{p[3]:.2f}"
    )


def encode_parameters(config: LQRConfig) -> np.ndarray:
    """Encode an existing LQRConfig into logarithmic coordinates."""
    return np.array([
        np.log10(max(1e-10, config.q1)),
        np.log10(max(1e-10, config.q2)),
        np.log10(max(1e-10, config.q3)),
        np.log10(max(1e-10, config.r0)),
    ], dtype=np.float64)
