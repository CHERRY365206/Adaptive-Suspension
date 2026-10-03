"""Training pipeline for the Random Forest road classification model.

Enforces zero-data-leakage scenario partitioning and saves fitted pipeline artifacts.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from roads.road_dataset import generate_benchmark_dataset
from ai.dataset_generation import generate_scenario_dataset, LABEL_MAP, INV_LABEL_MAP
from ai.feature_extraction import FEATURE_NAMES
from ai.evaluate_classifier import compute_classification_metrics, save_confusion_matrix_plot


def train_road_classifier(
    num_per_class: int = 15,
    n_estimators: int = 100,
    max_depth: int = 12,
    seed: int = 42,
    output_model_path: str | Path = "artifacts/models/road_classifier.joblib",
    output_metrics_path: str | Path = "results/tables/classifier_metrics.json",
    output_cm_path: str | Path = "results/figures/confusion_matrix.png",
    quick_mode: bool = False
) -> Dict[str, Any]:
    """Train Random Forest classifier on scenario-partitioned training data and evaluate on held-out test data.

    Args:
        num_per_class: Scenarios per class (15 default, reduced if quick_mode).
        n_estimators: Number of decision trees.
        max_depth: Maximum tree depth.
        seed: Random seed for model training and scenario generation.
        output_model_path: Path to save fitted joblib pipeline.
        output_metrics_path: Path to save evaluation metrics JSON.
        output_cm_path: Path to save confusion matrix image.
        quick_mode: Fast execution with fewer scenarios for testing.

    Returns:
        Dictionary containing evaluation metrics and training metadata.
    """
    if quick_mode:
        num_per_class = 6
        n_estimators = 30
        max_depth = 8

    # 1. Generate strictly partitioned scenarios
    dataset = generate_benchmark_dataset(num_per_class=num_per_class, base_seed=seed)
    train_scenarios = dataset["train"]
    val_scenarios = dataset["val"]
    test_scenarios = dataset["test"]

    # 2. Extract feature windows for train and test splits separately
    X_train, y_train, groups_train, _ = generate_scenario_dataset(train_scenarios + val_scenarios)
    X_test, y_test, groups_test, _ = generate_scenario_dataset(test_scenarios)

    # Verify zero group/scenario overlap
    train_group_set = set(groups_train)
    test_group_set = set(groups_test)
    assert len(train_group_set.intersection(test_group_set)) == 0, (
        "CRITICAL ERROR: Scenario leakage detected between train and test sets!"
    )

    # 3. Build and fit pipeline (Scaler + Random Forest)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # 4. Predict on held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # 5. Evaluate metrics
    class_names = [INV_LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    metrics = compute_classification_metrics(y_test, y_pred, target_names=class_names)

    # Feature importances
    rf_model = pipeline.named_steps["rf"]
    importances = rf_model.feature_importances_
    feat_importance_dict = {
        name: float(imp) for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    }

    metrics["feature_importances"] = feat_importance_dict
    metrics["training_samples"] = len(y_train)
    metrics["test_samples"] = len(y_test)
    metrics["num_scenarios_train"] = len(train_scenarios) + len(val_scenarios)
    metrics["num_scenarios_test"] = len(test_scenarios)
    metrics["random_seed"] = seed

    # 6. Save model artifact
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "label_map": LABEL_MAP,
        "inv_label_map": INV_LABEL_MAP,
        "metrics": metrics,
        "seed": seed
    }
    joblib.dump(artifact_payload, output_model_path)

    # 7. Save metrics JSON and confusion matrix plot
    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_arr = np.array(metrics["confusion_matrix"])
    save_confusion_matrix_plot(cm_arr, class_names, output_cm_path)

    return metrics
