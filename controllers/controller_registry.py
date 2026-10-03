"""Registry and factory for nominal and optimized LQR active suspension controllers."""

from __future__ import annotations
from typing import Dict, Optional
from controllers.lqr import LQRConfig, LQRController
from models.vehicle_parameters import VehicleParameters


def get_nominal_configs() -> Dict[str, LQRConfig]:
    """Define the three nominal, heuristically-tuned LQR configurations.

    Note: These names describe intended engineering priorities, not empirical guarantees.
    1. Comfort: Prioritizes body acceleration reduction.
    2. Balanced: Balances comfort, rattle space, tyre load, and control effort.
    3. Road-Holding: Prioritizes wheel tracking and tyre normal force consistency.
    """
    return {
        "comfort": LQRConfig(
            name="comfort",
            q1=1.0e5,     # High penalty on body acceleration (comfort)
            q2=5.0e3,     # Moderate penalty on suspension travel
            q3=1.0e3,     # Lower penalty on tyre deflection
            r0=1.0e-5,    # Actuator effort penalty
            description="Comfort-oriented LQR prioritizing reduction of body vertical acceleration"
        ),
        "balanced": LQRConfig(
            name="balanced",
            q1=3.0e4,     # Balanced acceleration penalty
            q2=2.0e4,     # Balanced suspension deflection penalty
            q3=2.0e4,     # Balanced tyre load variation penalty
            r0=2.0e-5,    # Balanced actuator effort penalty
            description="Balanced LQR trading off passenger comfort, suspension deflection, and tyre grip"
        ),
        "road_holding": LQRConfig(
            name="road_holding",
            q1=5.0e3,     # Lower emphasis on body bounce
            q2=3.0e4,     # Higher suspension deflection penalty
            q3=2.5e5,     # High penalty on tyre load variation (road holding)
            r0=2.0e-5,    # Actuator effort penalty
            description="Road-holding-oriented LQR prioritizing tyre contact force and wheel grip"
        )
    }


def build_lqr_controller(
    config: LQRConfig,
    vehicle_params: Optional[VehicleParameters] = None
) -> LQRController:
    """Build and solve a single LQR controller instance."""
    return LQRController(config=config, vehicle_params=vehicle_params)


def build_nominal_controllers(
    vehicle_params: Optional[VehicleParameters] = None
) -> Dict[str, LQRController]:
    """Build all three nominal LQR controllers: comfort, balanced, and road_holding."""
    configs = get_nominal_configs()
    controllers = {}
    for name, cfg in configs.items():
        controllers[name] = build_lqr_controller(cfg, vehicle_params)
    return controllers
