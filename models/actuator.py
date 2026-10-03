"""Active suspension actuator dynamics, saturation, slew rate limits, and power/energy modeling.

Ensures the physical plant receives the true constrained force rather than idealized unbounded inputs,
and models electrical energy consumption with motoring and regenerative efficiencies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import yaml
import numpy as np

from controllers.saturation import saturate_force, rate_limit_force


@dataclass
class ActuatorModel:
    """Actuator model implementing force saturation, rate limiting, and power/energy calculations.

    Attributes:
        max_force: Maximum tensile/compressive actuator force [N] (default: 1800.0 N)
        min_force: Minimum actuator force [N] (default: -1800.0 N)
        slew_rate_limit: Maximum rate of force change [N/s] (default: 18000.0 N/s)
        enabled: Actuator enable flag. If False, output force is forced to 0 (passive mode)
        motoring_efficiency: Efficiency of converting battery electrical power to mechanical work [-]
        regenerative_efficiency: Efficiency of capturing energy when damping external motion [-]
        standby_power: Quiescent auxiliary electronic power consumption [W]
        allow_regeneration: Whether regenerated energy is returned to the battery
    """
    max_force: float = 1800.0
    min_force: float = -1800.0
    slew_rate_limit: float = 18000.0
    enabled: bool = True
    motoring_efficiency: float = 0.85
    regenerative_efficiency: float = 0.50
    standby_power: float = 25.0
    allow_regeneration: bool = True

    def __post_init__(self) -> None:
        if self.max_force <= 0.0:
            raise ValueError(f"max_force must be strictly positive, got {self.max_force}")
        if self.min_force >= self.max_force:
            raise ValueError(f"min_force ({self.min_force}) must be less than max_force ({self.max_force})")
        if not (0.0 < self.motoring_efficiency <= 1.0):
            raise ValueError(f"motoring_efficiency must be in (0, 1], got {self.motoring_efficiency}")
        if not (0.0 <= self.regenerative_efficiency <= 1.0):
            raise ValueError(f"regenerative_efficiency must be in [0, 1], got {self.regenerative_efficiency}")
        self._last_applied_force = 0.0

    def reset(self) -> None:
        """Reset internal rate-limiter state."""
        self._last_applied_force = 0.0

    def apply_limits(self, u_cmd: float, dt: float) -> Tuple[float, float, bool]:
        """Apply force limits, saturation, and rate limiting to commanded force.

        Args:
            u_cmd: Requested controller force [N]
            dt: Time increment [s]

        Returns:
            Tuple of (requested_force, applied_force, is_saturated)
        """
        if not self.enabled:
            # Passive suspension: actuator is switched off
            self._last_applied_force = 0.0
            return float(u_cmd), 0.0, False

        # 1. Enforce amplitude saturation limits
        u_sat = saturate_force(u_cmd, self.min_force, self.max_force)
        is_saturated = bool(abs(u_sat - u_cmd) > 1e-4)

        # 2. Enforce slew-rate rate limit
        u_rate_limited = rate_limit_force(
            self._last_applied_force,
            u_sat,
            self.slew_rate_limit,
            dt
        )
        self._last_applied_force = u_rate_limited
        return float(u_cmd), float(u_rate_limited), is_saturated

    def calculate_power(self, u_applied: float, v_rel: float) -> Tuple[float, float]:
        """Calculate mechanical and electrical power.

        Force convention: Positive u_applied pushes sprung mass UP and unsprung mass DOWN.
        Relative suspension velocity is v_rel = zs_dot - zu_dot.
        Mechanical power: P_mech = u_applied * v_rel [W].
        - If P_mech > 0: actuator is actively doing work on the suspension (motoring).
        - If P_mech < 0: suspension is doing work on the actuator (potential regeneration).

        Args:
            u_applied: Force applied by actuator [N]
            v_rel: Relative velocity (zs_dot - zu_dot) [m/s]

        Returns:
            Tuple of (power_mechanical_w, power_electrical_w)
        """
        if not self.enabled or abs(u_applied) < 1e-6:
            return 0.0, 0.0

        p_mech = u_applied * v_rel

        if p_mech >= 0.0:
            # Motoring mode: drawing energy from battery
            p_elec = (p_mech / self.motoring_efficiency) + self.standby_power
        else:
            # Damping / generator mode: absorbing kinetic energy
            if self.allow_regeneration:
                # p_mech is negative; regeneration returns power into battery
                p_regen = p_mech * self.regenerative_efficiency
                p_elec = p_regen + self.standby_power
            else:
                p_elec = self.standby_power

        return float(p_mech), float(p_elec)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> ActuatorModel:
        """Load actuator configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        mapping = {
            "max_force_n": "max_force",
            "min_force_n": "min_force",
            "slew_rate_limit_n_per_s": "slew_rate_limit",
            "actuator_efficiency": "motoring_efficiency",
            "regenerative_efficiency": "regenerative_efficiency",
            "standby_power_w": "standby_power",
            "allow_regeneration": "allow_regeneration"
        }
        kwargs = {}
        for k, v in data.items():
            if k in mapping:
                kwargs[mapping[k]] = v
            elif k in cls.__dataclass_fields__:
                kwargs[k] = v
        return cls(**kwargs)
