"""CLI script to train and evaluate the Random Forest road classification model."""

from __future__ import annotations
import argparse
from pathlib import Path
from ai.train_classifier import train_road_classifier


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest road classifier.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of trees (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Starting Random Forest training pipeline (quick={args.quick}, seed={args.seed})...")
    metrics = train_road_classifier(
        n_estimators=args.n_trees if not args.quick else 25,
        seed=args.seed,
        quick_mode=args.quick
    )

    print("\n--- Model Evaluation Results (Held-Out Test Scenarios) ---")
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100.0:.2f}%")
    print(f"Macro Precision:   {metrics['macro_precision'] * 100.0:.2f}%")
    print(f"Macro Recall:      {metrics['macro_recall'] * 100.0:.2f}%")
    print(f"Macro F1-Score:    {metrics['macro_f1'] * 100.0:.2f}%")
    print(f"Test Sample Count: {metrics['test_samples']}")
    print(f"\nConfusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")

    print("\nArtifacts saved:")
    print("  - artifacts/models/road_classifier.joblib")
    print("  - results/tables/classifier_metrics.json")
    print("  - results/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()
