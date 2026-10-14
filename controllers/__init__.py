"""Active suspension controllers: LQR, MPC, saturation, adaptive selector, and registry."""

from controllers.lqr import LQRConfig, LQRController
from controllers.mpc import MPCConfig, MPCController
from controllers.saturation import saturate_force, rate_limit_force
from controllers.controller_registry import (
    get_nominal_configs,
    build_lqr_controller,
    build_nominal_controllers
)
from controllers.adaptive_selector import (
    AdaptiveControllerSelector,
    OracleAdaptiveSelector,
    DEFAULT_ROAD_TO_CONTROLLER_MAP
)

from controllers.dosmc import DOSMCConfig, DOSMCController

__all__ = [
    "LQRConfig",
    "LQRController",
    "MPCConfig",
    "MPCController",
    "DOSMCConfig",
    "DOSMCController",
    "saturate_force",
    "rate_limit_force",
    "get_nominal_configs",
    "build_lqr_controller",
    "build_nominal_controllers",
    "AdaptiveControllerSelector",
    "OracleAdaptiveSelector",
    "DEFAULT_ROAD_TO_CONTROLLER_MAP",
]

