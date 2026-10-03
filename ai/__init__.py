"""AI-driven road classification and simulated sensor feature extraction."""

from ai.feature_extraction import SensorFeatureExtractor, FEATURE_NAMES
from ai.dataset_generation import generate_scenario_dataset, LABEL_MAP, INV_LABEL_MAP
from ai.train_classifier import train_road_classifier
from ai.evaluate_classifier import compute_classification_metrics, save_confusion_matrix_plot
from ai.predict_road import RoadPredictor

__all__ = [
    "SensorFeatureExtractor",
    "FEATURE_NAMES",
    "generate_scenario_dataset",
    "LABEL_MAP",
    "INV_LABEL_MAP",
    "train_road_classifier",
    "compute_classification_metrics",
    "save_confusion_matrix_plot",
    "RoadPredictor",
]
