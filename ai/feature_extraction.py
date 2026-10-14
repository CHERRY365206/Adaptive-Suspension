"""Simulated sensor feature extraction for intelligent road classification.

Extracts time-domain statistics and spectral band energy from rolling windows of
chassis and wheel-hub accelerometer and suspension travel sensors.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats


FEATURE_NAMES: List[str] = [
    # Body vertical acceleration features (sprung mass accelerometer)
    "body_accel_rms",
    "body_accel_std",
    "body_accel_peak",
    "body_accel_crest_factor",
    "body_accel_kurtosis",
    "body_accel_skewness",
    "body_accel_zero_crossings",
    "body_accel_bounce_energy",   # 0.5 - 3.5 Hz
    # Wheel vertical acceleration features (unsprung mass accelerometer)
    "wheel_accel_rms",
    "wheel_accel_std",
    "wheel_accel_peak",
    "wheel_accel_crest_factor",
    "wheel_accel_hop_energy",     # 10.0 - 15.0 Hz
    # Suspension travel features (deflection potentiometer / LVDT)
    "susp_travel_rms",
    "susp_travel_std",
    "susp_travel_peak",
    "susp_travel_max_abs",
    # Suspension velocity features (differentiated travel)
    "susp_velocity_rms",
    "susp_velocity_std",
    # Cross-sensor spectral energy ratio (wheel-hop to body-bounce ratio)
    "hop_to_bounce_energy_ratio"
]


class SensorFeatureExtractor:
    """Extracts informative road-characterization features from sensor measurement arrays.

    All features use only signals that are physically measurable on a real vehicle:
    - a_s: Body vertical acceleration [m/s^2]
    - a_u: Wheel vertical acceleration [m/s^2]
    - travel: Suspension deflection (zs - zu) [m]
    - v_rel: Suspension velocity (zs_dot - zu_dot) [m/s]
    """

    def __init__(self, sampling_rate_hz: float = 500.0) -> None:
        self.fs = sampling_rate_hz

    def extract_features(
        self,
        a_s: np.ndarray,
        a_u: np.ndarray,
        travel: np.ndarray,
        v_rel: np.ndarray
    ) -> np.ndarray:
        """Compute feature vector from windowed sensor signals.

        Args:
            a_s: Sprung mass acceleration window [m/s^2]
            a_u: Unsprung mass acceleration window [m/s^2]
            travel: Suspension deflection window [m]
            v_rel: Suspension relative velocity window [m/s]

        Returns:
            1D numpy array of computed features corresponding to FEATURE_NAMES.
        """
        N = len(a_s)
        if N < 5:
            # Fallback for insufficient data
            return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

        # 1. Body acceleration features
        rms_as = float(np.sqrt(np.mean(a_s**2)))
        std_as = float(np.std(a_s))
        peak_as = float(np.max(np.abs(a_s)))
        crest_as = peak_as / (rms_as + 1e-8)
        kurt_as = float(stats.kurtosis(a_s)) if std_as > 1e-6 else 0.0
        skew_as = float(stats.skew(a_s)) if std_as > 1e-6 else 0.0
        zc_as = float(np.sum(np.diff(np.signbit(a_s)) != 0)) / N

        # Spectral features for body acceleration (bounce band: 0.5 - 3.5 Hz)
        bounce_energy = self._band_power(a_s, f_low=0.5, f_high=3.5)

        # 2. Wheel acceleration features
        rms_au = float(np.sqrt(np.mean(a_u**2)))
        std_au = float(np.std(a_u))
        peak_au = float(np.max(np.abs(a_u)))
        crest_au = peak_au / (rms_au + 1e-8)
        hop_energy = self._band_power(a_u, f_low=10.0, f_high=15.0)

        # 3. Suspension travel features
        rms_tr = float(np.sqrt(np.mean(travel**2)))
        std_tr = float(np.std(travel))
        peak_tr = float(np.max(travel) - np.min(travel))
        max_abs_tr = float(np.max(np.abs(travel)))

        # 4. Suspension relative velocity features
        rms_vr = float(np.sqrt(np.mean(v_rel**2)))
        std_vr = float(np.std(v_rel))

        # 5. Energy ratio
        hop_ratio = hop_energy / (bounce_energy + 1e-8)

        feat_values = [
            rms_as, std_as, peak_as, crest_as, kurt_as, skew_as, zc_as, bounce_energy,
            rms_au, std_au, peak_au, crest_au, hop_energy,
            rms_tr, std_tr, peak_tr, max_abs_tr,
            rms_vr, std_vr,
            hop_ratio
        ]

        # Replace any potential NaNs or infs with finite numbers
        clean_feats = np.nan_to_num(np.array(feat_values, dtype=np.float64), nan=0.0, posinf=1e4, neginf=-1e4)
        return clean_feats

    def _band_power(self, signal_data: np.ndarray, f_low: float, f_high: float) -> float:
        """Compute signal power within a specified frequency band using FFT."""
        N = len(signal_data)
        if N < 8:
            return 0.0

        # Detrend signal to avoid 0-Hz DC domination
        d_sig = signal_data - np.mean(signal_data)
        fft_vals = np.fft.rfft(d_sig)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)

        power = np.abs(fft_vals) ** 2 / N
        idx_band = (freqs >= f_low) & (freqs <= f_high)
        if np.any(idx_band):
            return float(np.sum(power[idx_band]))
        return 0.0


if __name__ == "__main__":
    print("================================================================================")
    print("Testing SensorFeatureExtractor on Simulated Quarter-Car Sensor Window")
    print("================================================================================")
    extractor = SensorFeatureExtractor(sampling_rate_hz=500.0)
    np.random.seed(42)
    N = 250  # 0.5 seconds at 500 Hz

    # Simulated realistic quarter-car sensor signals
    t = np.linspace(0, 0.5, N)
    a_s = 0.5 * np.sin(2.0 * np.pi * 1.5 * t) + 0.1 * np.random.randn(N)
    a_u = 2.0 * np.sin(2.0 * np.pi * 12.0 * t) + 0.3 * np.random.randn(N)
    travel = 0.015 * np.sin(2.0 * np.pi * 1.5 * t) + 0.002 * np.random.randn(N)
    v_rel = np.gradient(travel, t)

    features = extractor.extract_features(a_s, a_u, travel, v_rel)

    print(f"\nExtracted {len(features)} Features from 0.5s Sensor Window:")
    print("-" * 65)
    for name, val in zip(FEATURE_NAMES, features):
        print(f"  {name:<32} : {val:12.6f}")
    print("-" * 65)
    print("\nFeature extraction executed successfully with zero NaNs or Infs!")

