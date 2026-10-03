"""Pure saturation and slew rate limiting utility functions."""

from __future__ import annotations
import numpy as np


def saturate_force(force: float, min_force: float, max_force: float) -> float:
    """Clamp force to lower and upper saturation bounds.

    Args:
        force: Commanded actuator force [N]
        min_force: Minimum allowed force [N]
        max_force: Maximum allowed force [N]

    Returns:
        Saturated force [N]
    """
    return float(np.clip(force, min_force, max_force))


def rate_limit_force(
    current_force: float,
    target_force: float,
    slew_rate_limit: float,
    dt: float
) -> float:
    """Enforce maximum rate-of-change (slew rate) on force command.

    Args:
        current_force: Previously applied force [N]
        target_force: Desired new force [N]
        slew_rate_limit: Maximum permissible rate of force change [N/s]
        dt: Time increment [s]

    Returns:
        Slew-rate constrained force [N]
    """
    if slew_rate_limit <= 0.0 or dt <= 0.0:
        return target_force

    max_delta = slew_rate_limit * dt
    delta = target_force - current_force
    clamped_delta = np.clip(delta, -max_delta, max_delta)
    return float(current_force + clamped_delta)
