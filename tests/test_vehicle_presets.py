"""Unit tests for research-paper-backed vehicle parameters, presets, and validation."""

import pytest
from pathlib import Path
from models.vehicle_parameters import (
    VehicleParameters,
    VehiclePreset,
    VEHICLE_PRESETS,
    get_vehicle_preset,
    validate_custom_parameters,
    RESEARCH_BENCHMARK_METADATA,
    BENCHMARK_MS,
    BENCHMARK_MU,
    BENCHMARK_KS,
    BENCHMARK_CS,
    BENCHMARK_KT,
)
from models.quarter_car import QuarterCarModel


def test_research_benchmark_values():
    """Verify that default VehicleParameters strictly match Magaji et al. (2025)."""
    params = VehicleParameters()
    assert params.ms == 395.3
    assert params.mu == 48.3
    assert params.ks == 30010.0
    assert params.cs == 1450.0
    assert params.kt == 340000.0
    assert params.max_travel == 0.08
    assert params.total_mass == pytest.approx(443.6, rel=1e-5)
    assert params.static_tyre_load == pytest.approx(443.6 * 9.81, rel=1e-5)

    # Natural frequencies
    assert 1.3 < params.sprung_natural_frequency_hz < 1.5
    assert 13.0 < params.unsprung_natural_frequency_hz < 15.0


def test_preset_registry():
    """Verify the vehicle presets registry contains Sedan EV and Compact EV."""
    assert "Sedan EV" in VEHICLE_PRESETS
    assert "Compact EV" in VEHICLE_PRESETS

    sedan = get_vehicle_preset("Sedan EV")
    assert isinstance(sedan, VehiclePreset)
    assert sedan.preset_key == "Sedan EV"

    compact = get_vehicle_preset("Compact EV")
    assert isinstance(compact, VehiclePreset)
    assert compact.preset_key == "Compact EV"

    with pytest.raises(KeyError, match="Unknown vehicle preset"):
        get_vehicle_preset("Supercar EV")


def test_sedan_ev_provenance():
    """Verify Sedan EV loads benchmark parameters with clear academic citations."""
    sedan = get_vehicle_preset("Sedan EV")
    assert sedan.has_independent_source is True
    assert sedan.is_verified_oem is False
    assert "Magaji" in sedan.source_citation
    assert "10.3390/math13152418" in sedan.doi
    assert sedan.parameters.ms == 395.3
    assert sedan.parameters.mu == 48.3
    assert sedan.parameters.ks == 30010.0
    assert sedan.parameters.cs == 1450.0
    assert sedan.parameters.kt == 340000.0
    assert sedan.f_max == 1800.0
    assert sedan.slew_rate == 18000.0


def test_compact_ev_fallback():
    """Verify Compact EV explicitly acknowledges fallback rather than inventing numbers."""
    compact = get_vehicle_preset("Compact EV")
    assert compact.has_independent_source is False
    assert compact.is_verified_oem is False
    assert "fallback" in compact.status_message.lower()
    # Parameters must match benchmark rather than arbitrary numbers
    assert compact.parameters.ms == BENCHMARK_MS
    assert compact.parameters.mu == BENCHMARK_MU
    assert compact.parameters.ks == BENCHMARK_KS
    assert compact.parameters.cs == BENCHMARK_CS
    assert compact.parameters.kt == BENCHMARK_KT


def test_custom_parameters_validation_valid():
    """Valid custom parameters pass validation."""
    is_valid, errors = validate_custom_parameters(
        ms=420.0,
        mu=45.0,
        ks=29000.0,
        cs=1400.0,
        kt=310000.0,
        max_travel=0.08,
        f_max=1800.0,
        slew_rate=18000.0
    )
    assert is_valid is True
    assert len(errors) == 0


def test_custom_parameters_validation_invalid():
    """Invalid custom parameters return descriptive errors."""
    # mu >= ms
    is_valid, errors = validate_custom_parameters(
        ms=50.0,
        mu=50.0,
        ks=30000.0,
        cs=1400.0,
        kt=340000.0
    )
    assert is_valid is False
    assert any("strictly smaller than sprung mass" in e for e in errors)

    # Negative stiffness and damping
    is_valid, errors = validate_custom_parameters(
        ms=400.0,
        mu=40.0,
        ks=-100.0,
        cs=0.0,
        kt=-500.0
    )
    assert is_valid is False
    assert len(errors) >= 3

    # Negative actuator limits
    is_valid, errors = validate_custom_parameters(
        ms=400.0,
        mu=40.0,
        ks=30000.0,
        cs=1400.0,
        kt=340000.0,
        f_max=-100.0,
        slew_rate=0.0
    )
    assert is_valid is False
    assert any("Fmax" in e for e in errors)
    assert any("slew rate" in e for e in errors)


def test_yaml_config_loading():
    """Verify loading from config/vehicle.yaml yields research benchmark values."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "vehicle.yaml"
    params = VehicleParameters.load_from_yaml(config_path)
    assert params.ms == 395.3
    assert params.mu == 48.3
    assert params.ks == 30010.0
    assert params.cs == 1450.0
    assert params.kt == 340000.0


def test_model_tyre_damping_consistency():
    """Verify that 2-DOF quarter-car equations do not include tyre damping ct."""
    assert "No tyre damping" in RESEARCH_BENCHMARK_METADATA["tyre_damping_note"]
    model = QuarterCarModel()
    # In A matrix, row 3 is zu_dot derivative (zu_ddot)
    # Equation: mu * zu_ddot = ks*(zs - zu) + cs*(zs_dot - zu_dot) - kt*(zu - zr) - u
    # Notice that tyre velocity term zr_dot does not appear in A or E
    assert model.E.shape == (4, 1)
    # Disturbance enters purely as road displacement zr with factor kt / mu
    expected_E3 = model.params.kt / model.params.mu
    assert model.E[3, 0] == pytest.approx(expected_E3)
