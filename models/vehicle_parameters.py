"""Vehicle parameters definition and physical validation for the EV quarter-car model.

Provides a structured dataclass for sprung and unsprung masses, suspension stiffness,
damping, and tyre parameters, including physical validation and sensitivity helpers.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class VehicleParameters:
    """Configurable physical parameters for a 2-DOF quarter-car active suspension model.

    NOTE: These values represent illustrative engineering simulation parameters for
    a passenger electric vehicle quarter-car assembly, NOT manufacturer specifications.

    Attributes:
        ms: Sprung mass (vehicle body quarter-chassis + battery payload) [kg].
        mu: Unsprung mass (wheel, hub, tire, brake assembly) [kg].
        ks: Suspension spring stiffness [N/m].
        cs: Suspension damper damping coefficient [N*s/m].
        kt: Radial tyre stiffness [N/m].
        max_travel: Maximum allowable suspension rattle-space travel [m].
        g: Acceleration due to gravity [m/s^2].
    """
    ms: float = 450.0
    mu: float = 45.0
    ks: float = 28000.0
    cs: float = 1500.0
    kt: float = 190000.0
    max_travel: float = 0.08
    g: float = 9.81

    def __post_init__(self) -> None:
        """Validate parameter signs and realistic ranges immediately upon construction."""
        self.validate()

    def validate(self) -> None:
        """Ensure all physical parameters are strictly positive and within valid ranges.

        Raises:
            ValueError: If any parameter violates physical constraints.
        """
        if self.ms <= 0:
            raise ValueError(f"Sprung mass ms must be strictly positive, got {self.ms} kg.")
        if self.mu <= 0:
            raise ValueError(f"Unsprung mass mu must be strictly positive, got {self.mu} kg.")
        if self.ks <= 0:
            raise ValueError(f"Suspension stiffness ks must be strictly positive, got {self.ks} N/m.")
        if self.cs <= 0:
            raise ValueError(f"Suspension damping cs must be strictly positive, got {self.cs} N*s/m.")
        if self.kt <= 0:
            raise ValueError(f"Tyre stiffness kt must be strictly positive, got {self.kt} N/m.")
        if self.max_travel <= 0:
            raise ValueError(f"Max travel must be positive, got {self.max_travel} m.")
        if self.g <= 0:
            raise ValueError(f"Gravity g must be positive, got {self.g} m/s^2.")
        if self.mu >= self.ms:
            raise ValueError(
                f"Unsprung mass ({self.mu} kg) must be smaller than sprung mass ({self.ms} kg)."
            )

    @property
    def total_mass(self) -> float:
        """Total corner mass (sprung + unsprung) [kg]."""
        return self.ms + self.mu

    @property
    def static_tyre_load(self) -> float:
        """Static equilibrium normal force exerted on tyre contact patch [N]."""
        return self.total_mass * self.g

    @property
    def sprung_natural_frequency_hz(self) -> float:
        """Approximate body-bounce natural frequency [Hz]."""
        return math.sqrt(self.ks / self.ms) / (2.0 * math.pi)

    @property
    def unsprung_natural_frequency_hz(self) -> float:
        """Approximate wheel-hop natural frequency [Hz]."""
        return math.sqrt((self.ks + self.kt) / self.mu) / (2.0 * math.pi)

    @property
    def damping_ratio_sprung(self) -> float:
        """Approximate passive damping ratio for the sprung mass mode [-]."""
        critical_damping = 2.0 * math.sqrt(self.ks * self.ms)
        return self.cs / critical_damping

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter set to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleParameters:
        """Instantiate parameters from a dictionary, filtering unknown keys."""
        valid_keys = {
            "ms", "mu", "ks", "cs", "kt", "max_travel", "g",
            # Handle possible alternate config keys:
            "sprung_mass_kg", "unsprung_mass_kg", "suspension_stiffness_n_per_m",
            "suspension_damping_ns_per_m", "tyre_stiffness_n_per_m",
            "max_suspension_travel_m", "gravity_m_per_s2"
        }
        kwargs = {}
        mapping = {
            "sprung_mass_kg": "ms",
            "unsprung_mass_kg": "mu",
            "suspension_stiffness_n_per_m": "ks",
            "suspension_damping_ns_per_m": "cs",
            "tyre_stiffness_n_per_m": "kt",
            "max_suspension_travel_m": "max_travel",
            "gravity_m_per_s2": "g"
        }
        for k, v in data.items():
            if k in mapping:
                kwargs[mapping[k]] = float(v)
            elif k in cls.__dataclass_fields__:
                kwargs[k] = float(v)
        return cls(**kwargs)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> VehicleParameters:
        """Load vehicle parameters from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def with_variations(
        self,
        ms_factor: float = 1.0,
        mu_factor: float = 1.0,
        ks_factor: float = 1.0,
        cs_factor: float = 1.0,
        kt_factor: float = 1.0,
    ) -> VehicleParameters:
        """Return a new VehicleParameters instance with percentage variations for sensitivity analysis."""
        return VehicleParameters(
            ms=self.ms * ms_factor,
            mu=self.mu * mu_factor,
            ks=self.ks * ks_factor,
            cs=self.cs * cs_factor,
            kt=self.kt * kt_factor,
            max_travel=self.max_travel,
            g=self.g,
        )
