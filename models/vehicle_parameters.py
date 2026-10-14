"""Vehicle parameters definition, physical validation, and research benchmark presets for the EV quarter-car model.

Provides a structured dataclass for sprung and unsprung masses, suspension stiffness,
damping, and tyre parameters, including physical validation, sensitivity helpers,
and research-paper-backed vehicle presets.

Source Research Paper:
    "Optimal Design of Linear Quadratic Regulator for Vehicle Suspension System Based on Bacterial Memetic Algorithm"
    Authors: Bala Abdullahi Magaji, et al.
    Journal: Mathematics (MDPI), 2025, 13(15), 2418.
    DOI: https://doi.org/10.3390/math13152418
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml


# Documented Research Benchmark Parameters from Magaji et al. (2025), DOI: 10.3390/math13152418
RESEARCH_BENCHMARK_METADATA = {
    "paper_title": "Optimal Design of Linear Quadratic Regulator for Vehicle Suspension System Based on Bacterial Memetic Algorithm",
    "authors": "Bala Abdullahi Magaji, et al.",
    "journal": "Mathematics (MDPI)",
    "year": 2025,
    "volume": 13,
    "issue": 15,
    "article_number": 2418,
    "doi": "https://doi.org/10.3390/math13152418",
    "table_description": "Quarter-car vehicle model parameters table",
    "model_type": "2-DOF quarter-car heave model",
    "tyre_damping_note": (
        "Tyre contact is modeled purely via radial tyre stiffness kt (340,000 N/m). "
        "No tyre damping ct is reported or included in the cited research paper, "
        "which aligns directly with our 2-DOF quarter-car equations of motion."
    )
}

BENCHMARK_MS = 395.3          # Sprung mass [kg]
BENCHMARK_MU = 48.3           # Unsprung mass [kg]
BENCHMARK_KS = 30010.0        # Suspension spring stiffness [N/m]
BENCHMARK_CS = 1450.0         # Suspension damping coefficient [N*s/m]
BENCHMARK_KT = 340000.0       # Radial tyre stiffness [N/m]
BENCHMARK_MAX_TRAVEL = 0.08   # Maximum allowable rattle-space travel [m]
BENCHMARK_G = 9.81            # Gravitational acceleration [m/s^2]
BENCHMARK_F_MAX = 1800.0      # Peak actuator force [N]
BENCHMARK_SLEW_RATE = 18000.0 # Actuator slew-rate limit [N/s]


@dataclass
class VehicleParameters:
    """Configurable physical parameters for a 2-DOF quarter-car active suspension model.

    Default values correspond to the peer-reviewed research benchmark published in
    Magaji et al. (2025), DOI: 10.3390/math13152418.

    Attributes:
        ms: Sprung mass (chassis + battery payload quarter-mass) [kg].
        mu: Unsprung mass (wheel, hub, tire, brake assembly) [kg].
        ks: Suspension spring stiffness [N/m].
        cs: Suspension damper damping coefficient [N*s/m].
        kt: Radial tyre stiffness [N/m].
        max_travel: Maximum allowable suspension rattle-space travel [m].
        g: Acceleration due to gravity [m/s^2].
    """
    ms: float = BENCHMARK_MS
    mu: float = BENCHMARK_MU
    ks: float = BENCHMARK_KS
    cs: float = BENCHMARK_CS
    kt: float = BENCHMARK_KT
    max_travel: float = BENCHMARK_MAX_TRAVEL
    g: float = BENCHMARK_G

    def __post_init__(self) -> None:
        """Validate parameter signs and physical consistency upon construction."""
        self.validate()

    def validate(self) -> None:
        """Ensure all physical parameters are strictly positive and satisfy mechanical consistency.

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
        mapping = {
            "sprung_mass_kg": "ms",
            "unsprung_mass_kg": "mu",
            "suspension_stiffness_n_per_m": "ks",
            "suspension_damping_ns_per_m": "cs",
            "tyre_stiffness_n_per_m": "kt",
            "max_suspension_travel_m": "max_travel",
            "gravity_m_per_s2": "g"
        }
        kwargs = {}
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
        """Return a new VehicleParameters instance with scaling factors for sensitivity analysis."""
        return VehicleParameters(
            ms=self.ms * ms_factor,
            mu=self.mu * mu_factor,
            ks=self.ks * ks_factor,
            cs=self.cs * cs_factor,
            kt=self.kt * kt_factor,
            max_travel=self.max_travel,
            g=self.g,
        )


@dataclass
class VehiclePreset:
    """Documented vehicle preset configuration and provenance metadata."""
    preset_key: str
    display_name: str
    parameters: VehicleParameters
    f_max: float
    slew_rate: float
    has_independent_source: bool
    source_citation: str
    doi: str
    is_verified_oem: bool
    status_message: str
    notes: str
    parameter_units: Dict[str, str] = field(default_factory=lambda: {
        "ms": "kg",
        "mu": "kg",
        "ks": "N/m",
        "cs": "N*s/m",
        "kt": "N/m",
        "max_travel": "m",
        "f_max": "N",
        "slew_rate": "N/s"
    })


# Single Source of Truth for Vehicle Presets
VEHICLE_PRESETS: Dict[str, VehiclePreset] = {
    "Sedan EV": VehiclePreset(
        preset_key="Sedan EV",
        display_name="Sedan EV (Research Benchmark: Magaji et al., 2025)",
        parameters=VehicleParameters(
            ms=BENCHMARK_MS,
            mu=BENCHMARK_MU,
            ks=BENCHMARK_KS,
            cs=BENCHMARK_CS,
            kt=BENCHMARK_KT,
            max_travel=BENCHMARK_MAX_TRAVEL,
            g=BENCHMARK_G
        ),
        f_max=BENCHMARK_F_MAX,
        slew_rate=BENCHMARK_SLEW_RATE,
        has_independent_source=True,
        source_citation="Bala Abdullahi Magaji, et al., 'Optimal Design of Linear Quadratic Regulator for Vehicle Suspension System Based on Bacterial Memetic Algorithm', Mathematics 2025, 13(15), 2418.",
        doi="https://doi.org/10.3390/math13152418",
        is_verified_oem=False,
        status_message=(
            "Loaded documented Research Benchmark parameter set (Magaji et al., 2025). "
            "Note: This is a published quarter-car academic benchmark, not a verified OEM manufacturer specification for a specific production Sedan EV."
        ),
        notes="Published quarter-car benchmark parameter set. Exact published values: ms=395.3 kg, mu=48.3 kg, ks=30,010 N/m, cs=1,450 N*s/m, kt=340,000 N/m."
    ),
    "Compact EV": VehiclePreset(
        preset_key="Compact EV",
        display_name="Compact EV (Fallback to Research Benchmark)",
        parameters=VehicleParameters(
            ms=BENCHMARK_MS,
            mu=BENCHMARK_MU,
            ks=BENCHMARK_KS,
            cs=BENCHMARK_CS,
            kt=BENCHMARK_KT,
            max_travel=BENCHMARK_MAX_TRAVEL,
            g=BENCHMARK_G
        ),
        f_max=BENCHMARK_F_MAX,
        slew_rate=BENCHMARK_SLEW_RATE,
        has_independent_source=False,
        source_citation="Fallback to Research Benchmark: Magaji et al. (2025) [No independent compact EV peer-reviewed source currently verified]",
        doi="https://doi.org/10.3390/math13152418",
        is_verified_oem=False,
        status_message=(
            "⚠️ No independently verified peer-reviewed quarter-car parameter set is currently configured for a Compact EV. "
            "To maintain scientific integrity and avoid inventing arbitrary scaled numbers, parameters fallback to the documented Research Benchmark (Magaji et al., 2025). "
            "Please provide a validated research paper or manufacturer source if distinct compact-car parameters are desired."
        ),
        notes="No independent compact EV peer-reviewed source available; falling back to published benchmark rather than inventing arbitrary scaled values."
    )
}


def get_vehicle_preset(name: str) -> VehiclePreset:
    """Retrieve vehicle preset configuration by name.
    
    Args:
        name: Name of preset ('Sedan EV', 'Compact EV').
        
    Returns:
        VehiclePreset instance.
        
    Raises:
        KeyError: If preset name is not recognized.
    """
    if name not in VEHICLE_PRESETS:
        raise KeyError(f"Unknown vehicle preset '{name}'. Available presets: {list(VEHICLE_PRESETS.keys())}")
    return VEHICLE_PRESETS[name]


def validate_custom_parameters(
    ms: float,
    mu: float,
    ks: float,
    cs: float,
    kt: float,
    max_travel: float = 0.08,
    f_max: float = 1800.0,
    slew_rate: float = 18000.0
) -> Tuple[bool, List[str]]:
    """Validate user-entered custom vehicle parameters for physical plausibility.
    
    Args:
        ms: Sprung mass [kg]
        mu: Unsprung mass [kg]
        ks: Suspension stiffness [N/m]
        cs: Suspension damping [N*s/m]
        kt: Tyre stiffness [N/m]
        max_travel: Rattle-space travel limit [m]
        f_max: Maximum actuator force [N]
        slew_rate: Actuator slew rate limit [N/s]
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors: List[str] = []
    
    if ms <= 0:
        errors.append(f"Sprung mass ms must be strictly positive (got {ms} kg).")
    if mu <= 0:
        errors.append(f"Unsprung mass mu must be strictly positive (got {mu} kg).")
    if ms > 0 and mu > 0 and mu >= ms:
        errors.append(f"Unsprung mass ({mu} kg) must be strictly smaller than sprung mass ({ms} kg).")
    if ks <= 0:
        errors.append(f"Suspension stiffness ks must be strictly positive (got {ks} N/m).")
    if cs <= 0:
        errors.append(f"Suspension damping cs must be strictly positive (got {cs} N*s/m).")
    if kt <= 0:
        errors.append(f"Tyre stiffness kt must be strictly positive (got {kt} N/m).")
    if max_travel <= 0:
        errors.append(f"Max travel must be strictly positive (got {max_travel} m).")
    if f_max <= 0:
        errors.append(f"Maximum actuator force Fmax must be strictly positive (got {f_max} N).")
    if slew_rate <= 0:
        errors.append(f"Actuator slew rate limit must be strictly positive (got {slew_rate} N/s).")
        
    is_valid = len(errors) == 0
    return is_valid, errors
