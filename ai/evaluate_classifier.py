"""Evaluation metrics and confusion matrix generation for road classification."""

from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai.feature_extraction import FEATURE_NAMES


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str]
) -> Dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    per_class = {}
    for idx, name in enumerate(target_names):
        # binary metrics per class (one-vs-rest)
        y_t_bin = (y_true == idx).astype(int)
        y_p_bin = (y_pred == idx).astype(int)
        per_class[name] = {
            "precision": float(precision_score(y_t_bin, y_p_bin, zero_division=0)),
            "recall": float(recall_score(y_t_bin, y_p_bin, zero_division=0)),
            "f1_score": float(f1_score(y_t_bin, y_p_bin, zero_division=0)),
            "support": int(np.sum(y_t_bin))
        }

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_true)
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    title: str = "Road Classifier Confusion Matrix"
) -> None:
    """Generate and save publication-quality confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > (np.max(cm) / 2.0) else "black"
            ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Road Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Road Class", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
