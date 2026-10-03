"""Real-time road prediction interface using rolling-window sensor buffers.

Receives sensor measurements as they arrive sequentially, maintains a fixed-length FIFO buffer,
extracts features, and generates road-class predictions and confidence scores without future leakage.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from collections import deque
from pathlib import Path
import joblib
import numpy as np

from ai.feature_extraction import SensorFeatureExtractor, FEATURE_NAMES
from ai.dataset_generation import LABEL_MAP, INV_LABEL_MAP


class RoadPredictor:
    """Real-time online road-profile classifier for active suspension."""

    def __init__(
        self,
        model_path: Optional[str | Path] = "artifacts/models/road_classifier.joblib",
        pipeline=None,
        window_s: float = 0.5,
        dt: float = 0.002,
        sampling_rate_hz: float = 500.0,
        fallback_class: str = "balanced"
    ) -> None:
        self.window_s = window_s
        self.dt = dt
        self.fs = sampling_rate_hz
        self.fallback_class = fallback_class

        self.window_size = int(round(window_s / dt))
        self.min_samples = max(20, int(self.window_size * 0.4))

        self.extractor = SensorFeatureExtractor(sampling_rate_hz=self.fs)

        # Rolling FIFO buffers
        self.buf_as: deque[float] = deque(maxlen=self.window_size)
        self.buf_au: deque[float] = deque(maxlen=self.window_size)
        self.buf_travel: deque[float] = deque(maxlen=self.window_size)
        self.buf_vrel: deque[float] = deque(maxlen=self.window_size)

        # Load pipeline if path provided
        self.pipeline = pipeline
        self.label_map = LABEL_MAP
        self.inv_label_map = INV_LABEL_MAP
        self.is_loaded = False

        if self.pipeline is None and model_path is not None:
            self.load_model(model_path)
        elif self.pipeline is not None:
            self.is_loaded = True

    def load_model(self, model_path: str | Path) -> bool:
        """Load trained classifier pipeline and metadata from disk."""
        path = Path(model_path)
        if not path.exists():
            self.is_loaded = False
            return False

        try:
            artifact = joblib.load(path)
            self.pipeline = artifact["pipeline"]
            self.label_map = artifact.get("label_map", LABEL_MAP)
            self.inv_label_map = artifact.get("inv_label_map", INV_LABEL_MAP)
            self.is_loaded = True
            return True
        except Exception:
            self.is_loaded = False
            return False

    def reset(self) -> None:
        """Clear rolling sensor buffers."""
        self.buf_as.clear()
        self.buf_au.clear()
        self.buf_travel.clear()
        self.buf_vrel.clear()

    def add_sample(
        self,
        a_s: float,
        a_u: float,
        travel: float,
        v_rel: float
    ) -> None:
        """Append the latest sensor sample to rolling FIFO buffer."""
        # Replace non-finite sensor values with zero
        self.buf_as.append(0.0 if not np.isfinite(a_s) else float(a_s))
        self.buf_au.append(0.0 if not np.isfinite(a_u) else float(a_u))
        self.buf_travel.append(0.0 if not np.isfinite(travel) else float(travel))
        self.buf_vrel.append(0.0 if not np.isfinite(v_rel) else float(v_rel))

    def predict(self) -> Tuple[str, float, np.ndarray]:
        """Classify current road condition based on buffered measurements.

        Returns:
            Tuple of:
            - predicted_class: string ('smooth', 'bumpy', 'rough')
            - confidence: probability of the winning class [0.0, 1.0]
            - probabilities: full probability vector across all 3 classes
        """
        n_samples = len(self.buf_as)

        # 1. Startup phase check
        if not self.is_loaded or n_samples < self.min_samples:
            # Safe fallback during startup
            return self.fallback_class, 0.0, np.array([0.33, 0.34, 0.33], dtype=np.float64)

        # 2. Extract features from current buffer
        arr_as = np.array(self.buf_as, dtype=np.float64)
        arr_au = np.array(self.buf_au, dtype=np.float64)
        arr_tr = np.array(self.buf_travel, dtype=np.float64)
        arr_vr = np.array(self.buf_vrel, dtype=np.float64)

        feats = self.extractor.extract_features(arr_as, arr_au, arr_tr, arr_vr)
        X = feats.reshape(1, -1)

        try:
            probas = self.pipeline.predict_proba(X)[0]
            pred_idx = int(np.argmax(probas))
            confidence = float(probas[pred_idx])
            pred_class = self.inv_label_map.get(pred_idx, self.fallback_class)
            return pred_class, confidence, probas
        except Exception:
            return self.fallback_class, 0.0, np.array([0.33, 0.34, 0.33], dtype=np.float64)
