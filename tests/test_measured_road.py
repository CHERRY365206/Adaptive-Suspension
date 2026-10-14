"""Unit tests for MeasuredRoadProfile and real experimental road loaders."""

import pytest
import numpy as np
from pathlib import Path

from roads.measured_road import (
    MeasuredRoadProfile,
    load_measured_road_profile,
    list_available_measured_roads,
    DEFAULT_DATA_DIR
)


def test_measured_road_from_arrays():
    """Verify MeasuredRoadProfile initialization, interpolation, and bounds handling from numpy arrays."""
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    z = np.array([0.0, 0.01, -0.01, 0.02, 0.0, -0.02])

    road = MeasuredRoadProfile(distance_m=s, elevation_m=z, name="TestArrayRoad", boundary_mode="clamp")

    assert road.total_length_m == pytest.approx(5.0)
    assert road.mean_step_m == pytest.approx(1.0)
    assert road.elevation_spatial(0.0) == pytest.approx(0.0)
    assert road.elevation_spatial(1.0) == pytest.approx(0.01)
    # Midpoint linear interpolation
    assert road.elevation_spatial(0.5) == pytest.approx(0.005)

    # Clamping behavior
    assert road.elevation_spatial(-1.0) == pytest.approx(0.0)
    assert road.elevation_spatial(10.0) == pytest.approx(-0.02)


def test_measured_road_loop_boundary():
    """Verify loop boundary condition wraps distance modulo total length."""
    s = np.array([0.0, 2.0, 4.0])
    z = np.array([0.05, -0.05, 0.05])

    road = MeasuredRoadProfile(distance_m=s, elevation_m=z, boundary_mode="loop")
    # At 5.0 m, mod 4.0 is 1.0 m (halfway between 0.0 and 2.0 -> z = 0.0)
    assert road.elevation_spatial(5.0) == pytest.approx(0.0)


def test_measured_road_validation_errors():
    """Verify that invalid inputs raise appropriate ValueErrors."""
    # Mismatched lengths
    with pytest.raises(ValueError, match="Dimension mismatch"):
        MeasuredRoadProfile(distance_m=np.array([0, 1]), elevation_m=np.array([0, 1, 2]))

    # Too few points
    with pytest.raises(ValueError, match="at least 2 points"):
        MeasuredRoadProfile(distance_m=np.array([0]), elevation_m=np.array([0]))


def test_load_real_lira_traces():
    """Verify loading real profilometer traces from disk."""
    available = list_available_measured_roads()
    assert len(available) >= 2

    keys = [item["key"] for item in available]
    assert "lira_highway_m13" in keys
    assert "lira_urban_cph1" in keys

    # Load highway trace
    highway = load_measured_road_profile("lira_highway_m13")
    assert highway.total_length_m > 300.0
    assert highway.elevation_std_m > 0.003
    assert highway.elevation_std_m < 0.020  # Typical smooth highway std < 20 mm

    # Load urban trace
    urban = load_measured_road_profile("lira_urban_cph1")
    assert urban.total_length_m > 300.0
    assert urban.elevation_std_m > highway.elevation_std_m  # Urban road is rougher than highway

    # Temporal scaling with speed
    speed = 25.0  # 25 m/s (~90 km/h)
    t = 2.0
    assert highway.elevation_time(t, speed) == pytest.approx(highway.elevation_spatial(speed * t))
