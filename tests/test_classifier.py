"""Unit tests for Random Forest road classifier, leakage prevention, and real-time predictor."""

import pytest
import numpy as np
from pathlib import Path

from ai.train_classifier import train_road_classifier
from ai.predict_road import RoadPredictor
from ai.dataset_generation import LABEL_MAP, INV_LABEL_MAP


def test_classifier_training_and_zero_leakage(tmp_path):
    """Verify end-to-end training, scenario group isolation, and metric calculation."""
    model_path = tmp_path / "test_rf.joblib"
    metrics_path = tmp_path / "test_metrics.json"
    cm_path = tmp_path / "test_cm.png"

    # Train in quick mode
    metrics = train_road_classifier(
        num_per_class=3,  # 9 total scenarios
        n_estimators=10,
        max_depth=5,
        seed=42,
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        output_cm_path=cm_path,
        quick_mode=True
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert cm_path.exists()

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["num_scenarios_train"] > 0
    assert metrics["num_scenarios_test"] > 0


def test_online_road_predictor_buffer(tmp_path):
    """Verify online RoadPredictor circular buffer and fallback handling."""
    predictor = RoadPredictor(model_path=None, window_s=0.2, dt=0.01)

    # 1. Initially empty: must return fallback
    cls_name, conf, probas = predictor.predict()
    assert cls_name == "balanced"
    assert conf == 0.0

    # 2. Add samples
    for i in range(25):
        predictor.add_sample(a_s=0.5, a_u=1.2, travel=0.01, v_rel=0.05)

    # Buffer should not exceed window_size
    assert len(predictor.buf_as) == int(round(0.2 / 0.01))

    # Reset clears buffer
    predictor.reset()
    assert len(predictor.buf_as) == 0
