"""Unit tests for sensor feature extraction."""

import pytest
import numpy as np

from ai.feature_extraction import SensorFeatureExtractor, FEATURE_NAMES


def test_feature_vector_dimensions_and_names():
    """Verify that extracted feature vector matches defined names and length."""
    extractor = SensorFeatureExtractor(sampling_rate_hz=500.0)
    N = 250  # 0.5 s at 500 Hz

    a_s = np.random.randn(N)
    a_u = np.random.randn(N)
    travel = np.random.randn(N) * 0.02
    v_rel = np.random.randn(N) * 0.1

    feats = extractor.extract_features(a_s, a_u, travel, v_rel)

    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feats))


def test_zero_signal_handling():
    """Verify that extracting features on purely zero signals produces finite outputs (no divide-by-zero NaNs)."""
    extractor = SensorFeatureExtractor(sampling_rate_hz=500.0)
    N = 250

    zeros = np.zeros(N)
    feats = extractor.extract_features(zeros, zeros, zeros, zeros)

    assert len(feats) == len(FEATURE_NAMES)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))
    # All RMS values should be 0.0
    assert feats[0] == 0.0


def test_insufficient_samples_fallback():
    """Verify safe handling of short windows (< 5 samples)."""
    extractor = SensorFeatureExtractor(sampling_rate_hz=500.0)
    short_sig = np.array([0.1, 0.2])

    feats = extractor.extract_features(short_sig, short_sig, short_sig, short_sig)
    assert len(feats) == len(FEATURE_NAMES)
    assert np.all(feats == 0.0)
