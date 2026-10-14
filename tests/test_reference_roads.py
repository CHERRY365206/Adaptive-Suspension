"""Unit tests for the three benchmark road profiles from Deshpande et al. (2014)."""

import pytest
import numpy as np

from roads.reference_roads import (
    ReferenceProfile1,
    ReferenceProfile2,
    ReferenceProfile3
)


def test_reference_profile_1_continuity_and_ranges():
    """Verify that Profile 1 generates smooth elevation values matching Chen & Huang (2005)."""
    prof1 = ReferenceProfile1(duration_s=15.0)
    t = np.linspace(0.0, 15.0, 3000)
    z = np.array([prof1.elevation_at_time(ti) for ti in t])

    # Check that elevation remains finite and within realistic physical bounds
    assert np.all(np.isfinite(z))
    # In the paper, Profile 1 peaks around +0.1 m at 5s and -0.1 m at 10s
    assert np.max(z) > 0.08
    assert np.min(z) < -0.08
    assert np.max(np.abs(z)) < 0.20

    # Callable interface
    fn = prof1.get_time_callable()
    assert abs(fn(5.0) - prof1.elevation_at_time(5.0)) < 1e-12


def test_reference_profile_2_harmonic_bounds():
    """Verify Profile 2 harmonic excitation z = 0.05 * cos(2*pi*t) * sin(0.6*pi*t)."""
    prof2 = ReferenceProfile2(duration_s=15.0)
    t = np.linspace(0.0, 15.0, 1500)
    z = np.array([prof2.elevation_at_time(ti) for ti in t])

    assert np.all(np.isfinite(z))
    # Max peak amplitude cannot exceed 0.05 m
    assert np.max(np.abs(z)) <= 0.05 + 1e-6
    # t=0 should be 0 since sin(0) = 0
    assert abs(prof2.elevation_at_time(0.0)) < 1e-10


def test_reference_profile_3_multi_harmonic_bounds():
    """Verify Profile 3 multi-harmonic excitation bounds."""
    prof3 = ReferenceProfile3(duration_s=15.0)
    t = np.linspace(0.0, 15.0, 1500)
    z = np.array([prof3.elevation_at_time(ti) for ti in t])

    assert np.all(np.isfinite(z))
    # Max peak cannot exceed 0.05 + 0.05 = 0.10 m
    assert np.max(np.abs(z)) <= 0.10 + 1e-6
