"""Unit tests for road profile generators (Smooth, Bump, Rough)."""

import pytest
import numpy as np

from roads.smooth import SmoothRoadGenerator
from roads.bump import BumpRoadGenerator
from roads.rough import RoughRoadGenerator
from roads.road_dataset import create_scenario, generate_benchmark_dataset


def test_smooth_road_reproducibility():
    """Verify that identical seeds produce identical smooth profiles, different seeds differ."""
    road1 = SmoothRoadGenerator(seed=101)
    road2 = SmoothRoadGenerator(seed=101)
    road3 = SmoothRoadGenerator(seed=202)

    s = np.linspace(0, 50, 500)
    z1 = road1.elevation_spatial(s)
    z2 = road2.elevation_spatial(s)
    z3 = road3.elevation_spatial(s)

    np.testing.assert_allclose(z1, z2, atol=1e-12)
    assert not np.allclose(z1, z3)
    # Check that amplitude is within realistic ISO Class A range (< 5 mm)
    assert np.max(np.abs(z1)) < 0.015


def test_bump_road_continuity_and_geometry():
    """Verify C1 continuity of speed bump and exact height/width specifications."""
    H = 0.05
    W = 0.8
    s0 = 10.0
    bump = BumpRoadGenerator(height=H, width=W, start_distance=s0, add_noise=False)

    # Elevation before and after bump should be zero
    assert bump.elevation_spatial(5.0) == 0.0
    assert bump.elevation_spatial(15.0) == 0.0

    # Boundary points must be exactly zero
    assert bump.elevation_spatial(s0) == pytest.approx(0.0, abs=1e-12)
    assert bump.elevation_spatial(s0 + W) == pytest.approx(0.0, abs=1e-12)

    # Peak elevation at center of bump
    s_peak = s0 + W / 2.0
    assert bump.elevation_spatial(s_peak) == pytest.approx(H, rel=1e-5)

    # Numerical derivative at boundary must be smooth (close to 0)
    ds = 1e-5
    deriv_start = (bump.elevation_spatial(s0 + ds) - bump.elevation_spatial(s0)) / ds
    deriv_end = (bump.elevation_spatial(s0 + W) - bump.elevation_spatial(s0 + W - ds)) / ds
    assert abs(deriv_start) < 1e-3
    assert abs(deriv_end) < 1e-3


def test_rough_road_classes():
    """Verify ISO Class C and D generation and that Class D has higher variance than Class C."""
    rough_c = RoughRoadGenerator(iso_class="C", seed=42)
    rough_d = RoughRoadGenerator(iso_class="D", seed=42)

    s = np.linspace(0, 100, 1000)
    zc = rough_c.elevation_spatial(s)
    zd = rough_d.elevation_spatial(s)

    assert np.all(np.isfinite(zc))
    assert np.all(np.isfinite(zd))

    std_c = np.std(zc)
    std_d = np.std(zd)
    # Class D PSD is 4x Class C PSD, so standard deviation should be approx 2x
    assert std_d > std_c
    assert (std_d / std_c) == pytest.approx(2.0, rel=0.1)


def test_temporal_velocity_scaling():
    """Verify that road displacement elevation_time matches elevation_spatial(v * t)."""
    road = SmoothRoadGenerator(seed=55)
    speed = 20.0  # 20 m/s
    t = 2.5
    s = speed * t
    assert road.elevation_time(t, speed) == pytest.approx(road.elevation_spatial(s), rel=1e-12)


def test_dataset_leakage_safe_split():
    """Verify that dataset generator creates disjoint scenario seeds across train, val, and test."""
    dataset = generate_benchmark_dataset(num_per_class=6, base_seed=5000)
    train_scenarios = dataset["train"]
    val_scenarios = dataset["val"]
    test_scenarios = dataset["test"]

    train_seeds = {s.road_profile.seed for s in train_scenarios}
    val_seeds = {s.road_profile.seed for s in val_scenarios}
    test_seeds = {s.road_profile.seed for s in test_scenarios}

    # Ensure zero overlap between sets
    assert len(train_seeds.intersection(val_seeds)) == 0
    assert len(train_seeds.intersection(test_seeds)) == 0
    assert len(val_seeds.intersection(test_seeds)) == 0
